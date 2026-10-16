"""LabAssist 服务入口。

第 1 课：进程监听端口，并用 HTTP 返回 JSON。
第 2 课：用 POST 创建送检批次。批次先放在内存列表里，进程重启后会消失。
第 3 课：用 GET 查看全部记录，以及按编号查看一条。
第 4 课：用 CSV 一次创建多条记录。
第 5 课：记录写入 SQLite，重启服务后仍然在。
"""

import csv
from datetime import datetime
from io import StringIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import DateTime, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

app = FastAPI(
    title="LabAssist",
    version="0.1.0",
    description="实验室文档与质控助手",
)


class BatchCreate(BaseModel):
    """调用方提交的批次内容。编号和状态由服务端填写。"""
    
    sample_id: str = Field(min_length=1, max_length=20, description="样本编号")
    sample_name: str = Field(min_length=1, max_length=100, description="样本名称")
    sample_type: str = Field(min_length=1, max_length=100, description="样本类型")
    org_name: str = Field(min_length=1, max_length=100, description="送检单位")
    accept_time: datetime = Field(description="接收时间，格式 YYYY-MM-DD HH:MM:SS")


class Batch(BatchCreate):
    """已经保存的批次。"""

    id: int
    status: str


class Base(DeclarativeBase):
    pass


class BatchRecord(Base):
    """批次在 SQLite 中的一行。"""

    __tablename__ = "batches"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[str] = mapped_column(String(20))
    sample_id: Mapped[str] = mapped_column(String(20))
    sample_name: Mapped[str] = mapped_column(String(100))
    sample_type: Mapped[str] = mapped_column(String(100))
    org_name: Mapped[str] = mapped_column(String(100))
    accept_time: Mapped[datetime] = mapped_column(DateTime)


engine = create_engine("sqlite:///labassist.db")
Base.metadata.create_all(engine)


def _to_batch(record: BatchRecord) -> Batch:
    """把数据库行转成接口返回的模型。"""
    return Batch(
        id=record.id,
        status=record.status,
        sample_id=record.sample_id,
        sample_name=record.sample_name,
        sample_type=record.sample_type,
        org_name=record.org_name,
        accept_time=record.accept_time,
    )


@app.get("/health")
def health() -> dict[str, str]:
    """供本机或部署脚本确认服务已启动。"""
    return {"status": "ok"}


@app.post("/batches", status_code=201)
def create_batch(body: BatchCreate) -> Batch:
    """创建一条送检批次，并分配自增编号。"""
    record = BatchRecord(
        status="pending",
        sample_id=body.sample_id,
        sample_name=body.sample_name,
        sample_type=body.sample_type,
        org_name=body.org_name,
        accept_time=body.accept_time,
    )
    with Session(engine) as session:
        session.add(record)
        session.commit()
        session.refresh(record)
        return _to_batch(record)


@app.get("/batches")
def list_batches() -> list[Batch]:
    """返回数据库里保存的全部记录。"""
    with Session(engine) as session:
        records = session.scalars(select(BatchRecord).order_by(BatchRecord.id)).all()
        return [_to_batch(record) for record in records]


@app.get("/batches/{batch_id}")
def get_batch(batch_id: int) -> Batch:
    """按编号返回一条记录。编号不存在时返回 404。"""
    with Session(engine) as session:
        record = session.get(BatchRecord, batch_id)
        if record is None:
            raise HTTPException(status_code=404, detail="批次不存在")
        return _to_batch(record)


@app.post("/batches/import", status_code=201)
async def import_batches(
    file: UploadFile = File(description="UTF-8 CSV，表头：sample_id,sample_name,sample_type,org_name,accept_time"),
) -> list[Batch]:
    """按行创建记录。每一行使用和 POST /batches 相同的字段校验。"""
    text = (await file.read()).decode("utf-8-sig")
    reader = csv.DictReader(StringIO(text))
    created: list[Batch] = []
    for line_no, row in enumerate(reader, start=2):
        try:
            body = BatchCreate.model_validate(row)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail={"line": line_no, "errors": exc.errors()},
            ) from exc
        created.append(create_batch(body))
    return created
