"""LabAssist 服务入口。

第 1 课：进程监听端口，并用 HTTP 返回 JSON。
第 2 课：用 POST 创建送检批次。批次先放在内存列表里，进程重启后会消失。
"""

from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel, Field

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


# 进程内存中的批次。重启服务后这份列表会清空，第 2 周再换成数据库。
batches: list[Batch] = []
_next_id = 1


@app.get("/health")
def health() -> dict[str, str]:
    """供本机或部署脚本确认服务已启动。"""
    return {"status": "ok"}


@app.post("/batches", status_code=201)
def create_batch(body: BatchCreate) -> Batch:
    """创建一条送检批次，并分配自增编号。"""
    global _next_id
    batch = Batch(
        id=_next_id,
        status="pending",
        sample_id=body.sample_id,
        sample_name=body.sample_name,
        sample_type=body.sample_type,
        org_name=body.org_name,
        accept_time=body.accept_time,
    )
    _next_id += 1
    batches.append(batch)
    return batch
