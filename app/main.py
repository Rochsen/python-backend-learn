"""LabAssist 服务入口。

第 1 课：进程监听端口，并用 HTTP 返回 JSON。
第 2 课：用 POST 创建送检批次。批次先放在内存列表里，进程重启后会消失。
"""

from datetime import date

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="LabAssist",
    version="0.1.0",
    description="实验室文档与质控助手",
)


class BatchCreate(BaseModel):
    """调用方提交的批次内容。编号和状态由服务端填写。"""

    org_name: str = Field(min_length=1, max_length=100, description="送检单位")
    sample_count: int = Field(ge=1, description="样本数")
    tested_on: date = Field(description="检测日期，格式 YYYY-MM-DD")


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
        org_name=body.org_name,
        sample_count=body.sample_count,
        tested_on=body.tested_on,
    )
    _next_id += 1
    batches.append(batch)
    return batch
