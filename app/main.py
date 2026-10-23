"""LabAssist 服务入口。

第 1 课：进程监听端口，并用 HTTP 返回 JSON。
第 2 课：用 POST 创建送检批次。批次先放在内存列表里，进程重启后会消失。
第 3 课：用 GET 查看全部记录，以及按编号查看一条。
第 4 课：用 CSV 一次创建多条记录。
第 5 课：记录写入 SQLite，重启服务后仍然在。
第 6 课：注册用户。密码只存摘要，重复用户名会被拒绝。
第 7 课：登录。正确密码返回 JWT，错误密码返回 401。
第 8 课：批次接口必须登录。用户只能看到自己的记录。
第 10 课：对话发给模型后写入数据库，重启服务后仍然在。
第 11 课：/chat 用 SSE 逐段返回文字，结束后仍写入数据库。
第 12 课：质控描述解析成固定 JSON，并记下 token 用量。
第 13 课：提问时模型调用查询工具，数字来自数据库。
第 14 课：pandas 计算均值和异常数。计算失败时返回错误，不让模型编造数字。
第 15 课：文本切块、转成向量，并写入数据库。
第 16 课：按相似度找出原文。资料里没有依据时直接回答不知道。
第 17 课：用 20 条已标明出处的问题计算检索命中率。
第 18 课：把切块改短后再测一遍，留下前后两次命中率。
第 19 课：Vue 页面逐段显示回答，并列出引用的原文。
第 20 课：上传质控 CSV，用 pandas 画图并写出结论。
第 21 课：用 docker compose 启动后，完成一次登录和一次提问。
"""

import base64
import csv
import hashlib
import json
import math
import os
import secrets
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Literal
from io import BytesIO, StringIO

import jwt
import pandas as pd

# 本机访问 huggingface.co 会超时，向量模型改从国内镜像下载。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from fastembed import TextEmbedding
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
)
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
    inspect,
    select,
)
from sqlalchemy import text as sql_text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

app = FastAPI(
    title="LabAssist",
    version="0.1.0",
    description="实验室文档与质控助手",
)


class BatchCreate(BaseModel):
    """Schema：创建批次的请求体。编号和状态由服务端填写。"""

    sample_id: str = Field(min_length=1, max_length=20, description="样本编号")
    sample_name: str = Field(min_length=1, max_length=100, description="样本名称")
    sample_type: str = Field(min_length=1, max_length=100, description="样本类型")
    org_name: str = Field(min_length=1, max_length=100, description="送检单位")
    accept_time: datetime = Field(description="接收时间，格式 YYYY-MM-DD HH:MM:SS")


class BatchOut(BatchCreate):
    """Schema：接口返回的批次。"""

    id: int
    status: str
    user_id: int | None


class Base(DeclarativeBase):
    pass


class Batch(Base):
    """数据库表 batches。"""

    __tablename__ = "batches"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[str] = mapped_column(String(20))
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    sample_id: Mapped[str] = mapped_column(String(20))
    sample_name: Mapped[str] = mapped_column(String(100))
    sample_type: Mapped[str] = mapped_column(String(100))
    org_name: Mapped[str] = mapped_column(String(100))
    accept_time: Mapped[datetime] = mapped_column(DateTime)


class User(Base):
    """数据库表 users。password_hash 是密码摘要，不是原密码。"""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    password_hash: Mapped[str] = mapped_column(String(200))


class Message(Base):
    """数据库表 messages。一条记录是用户或助手说的一句话。"""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class QcReport(Base):
    """数据库表 qc_reports。保存质控 JSON 和这次调用的 token 用量。"""

    __tablename__ = "qc_reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    sample_id: Mapped[str] = mapped_column(String(50))
    conclusion: Mapped[str] = mapped_column(String(20))
    issues: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text)
    prompt_tokens: Mapped[int] = mapped_column(Integer)
    completion_tokens: Mapped[int] = mapped_column(Integer)
    total_tokens: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class QcChart(Base):
    """数据库表 qc_charts。保存 pandas 画出的质控图和据此写成的结论。"""

    __tablename__ = "qc_charts"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    threshold: Mapped[float] = mapped_column(Float)
    mean: Mapped[float] = mapped_column(Float)
    anomaly_count: Mapped[int] = mapped_column(Integer)
    conclusion: Mapped[str] = mapped_column(Text)
    image: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class Document(Base):
    """数据库表 documents。一篇自己编写的 SOP 样例。"""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    title: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime)


class Chunk(Base):
    """数据库表 chunks。一篇文档切出的文本块及其向量。"""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    position: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[str] = mapped_column(Text)


class Evaluation(Base):
    """数据库表 evaluations。一次金标评测的命中结果。"""

    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    total: Mapped[int] = mapped_column(Integer)
    hit_count: Mapped[int] = mapped_column(Integer)
    hit_rate: Mapped[float] = mapped_column(Float)
    detail: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class EvaluationComparison(Base):
    """数据库表 evaluation_comparisons。同一次评测里切块调整前后的命中率。"""

    __tablename__ = "evaluation_comparisons"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    before_chunk_size: Mapped[int] = mapped_column(Integer)
    before_chunk_overlap: Mapped[int] = mapped_column(Integer)
    before_hit_count: Mapped[int] = mapped_column(Integer)
    before_hit_rate: Mapped[float] = mapped_column(Float)
    after_chunk_size: Mapped[int] = mapped_column(Integer)
    after_chunk_overlap: Mapped[int] = mapped_column(Integer)
    after_hit_count: Mapped[int] = mapped_column(Integer)
    after_hit_rate: Mapped[float] = mapped_column(Float)
    detail: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class UserCreate(BaseModel):
    """Schema：注册请求体。"""

    username: str = Field(min_length=1, max_length=50, description="用户名")
    password: str = Field(min_length=6, max_length=100, description="密码，至少 6 位")


class UserOut(BaseModel):
    """Schema：注册成功的响应，不包含密码。"""

    id: int
    username: str


class Login(BaseModel):
    """Schema：登录请求体。"""

    username: str = Field(min_length=1, max_length=50, description="用户名")
    password: str = Field(min_length=1, max_length=100, description="密码")


class TokenOut(BaseModel):
    """Schema：登录成功的响应。"""

    access_token: str
    token_type: str


class ChatCreate(BaseModel):
    """Schema：发给模型的一句话。"""

    content: str = Field(min_length=1, max_length=2000, description="用户说的话")


class MessageOut(BaseModel):
    """Schema：已经保存的一条对话。"""

    id: int
    role: str
    content: str


class QcCreate(BaseModel):
    """Schema：一段质控描述。"""

    description: str = Field(min_length=1, max_length=2000, description="质控描述")


class QcResult(BaseModel):
    """Schema：模型必须返回的质控 JSON。"""

    sample_id: str = Field(min_length=1, max_length=50)
    conclusion: Literal["pass", "fail", "review"]
    issues: list[str]
    summary: str = Field(min_length=1, max_length=500)


class QcOut(QcResult):
    """Schema：已保存的质控结果，带上 token 用量。"""

    id: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class QcCount(BaseModel):
    """Schema：从数据库数出来的质控条数。"""

    total: int
    pass_count: int
    fail_count: int
    review_count: int


class QuestionCreate(BaseModel):
    """Schema：向模型提的一个问题。"""

    content: str = Field(min_length=1, max_length=2000, description="要问的问题")


class QuestionOut(QcCount):
    """Schema：模型的回答，以及查询工具返回的数字。"""

    answer: str


class MetricCreate(BaseModel):
    """Schema：一组测量值和异常阈值。"""

    content: str = Field(min_length=1, max_length=2000, description="要问的问题")
    values: list[float] = Field(description="测量值。空列表会让计算工具失败")
    threshold: float = Field(description="大于该值的测量值计为异常")


class MetricStats(BaseModel):
    """Schema：pandas 算出的均值和异常数。"""

    mean: float
    anomaly_count: int


class MetricOut(MetricStats):
    """Schema：模型的说明，以及 pandas 算出的数字。"""

    answer: str


class QcChartOut(BaseModel):
    """Schema：质控图、pandas 算出的数字，以及据此写成的结论。"""

    id: int
    threshold: float
    mean: float
    anomaly_count: int
    conclusion: str
    image_base64: str


class DocumentCreate(BaseModel):
    """Schema：要入库的 SOP 样例。"""

    title: str = Field(min_length=1, max_length=100, description="文档标题")
    content: str = Field(min_length=1, max_length=8000, description="文档正文")


class ChunkOut(BaseModel):
    """Schema：一个文本块。向量本身存在数据库里，这里只返回维度。"""

    position: int
    content: str
    embedding_dim: int


class DocumentOut(BaseModel):
    """Schema：已保存的文档和它的文本块。"""

    id: int
    title: str
    chunk_count: int
    embedding_dim: int
    chunks: list[ChunkOut]


class AnswerCreate(BaseModel):
    """Schema：向已入库文档提的问题。"""

    content: str = Field(min_length=1, max_length=2000, description="要问的问题")


class CitationOut(BaseModel):
    """Schema：回答所依据的一段原文。"""

    document_id: int
    title: str
    position: int
    content: str


class AnswerOut(BaseModel):
    """Schema：带原文片段的回答。没有依据时 citations 为空。"""

    answer: str
    citations: list[CitationOut]


class EvaluationCitation(BaseModel):
    """Schema：评测时检索到的一段原文。"""

    title: str
    content: str


class EvaluationItemOut(BaseModel):
    """Schema：一道金标题的检索结果。"""

    question: str
    expected_title: str
    hit: bool
    citations: list[EvaluationCitation]


class EvaluationOut(BaseModel):
    """Schema：一次评测的命中率，以及每道题的对照。"""

    id: int
    total: int
    hit_count: int
    hit_rate: float
    items: list[EvaluationItemOut]


class ChunkRunOut(BaseModel):
    """Schema：一种切块设置下的命中率。"""

    chunk_size: int
    chunk_overlap: int
    total: int
    hit_count: int
    hit_rate: float


class EvaluationComparisonOut(BaseModel):
    """Schema：切块调整前后的两次命中率。"""

    id: int
    before: ChunkRunOut
    after: ChunkRunOut
    misses: list[EvaluationItemOut]


# 第 7 课把签名密钥放在代码里，用来签发和校验 JWT。
JWT_SECRET = "labassist-lesson-7"
JWT_ALGORITHM = "HS256"
CHUNK_SIZE = 60
CHUNK_OVERLAP = 15
ADJUSTED_CHUNK_SIZE = 8
ADJUSTED_CHUNK_OVERLAP = 0
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
RETRIEVAL_MIN_SCORE = 0.5
RETRIEVAL_LIMIT = 3
_embedder: TextEmbedding | None = None
SYSTEM_PROMPT = "你是实验室文档助手，用简短中文回答。"
KNOWLEDGE_PROMPT = (
    "你是实验室文档助手。只根据给出的原文用简短中文回答，不要补充原文里没有的内容。"
)
QC_PROMPT = """你是实验室质控助手。请把用户的描述解析成 JSON。
示例：
{
    "sample_id":"S001",
    "conclusion":"fail",
    "issues":["内参 Ct 值高于阈值"],
    "summary":"本次质控不合格"
}
conclusion 只能是 pass、fail、review。没有问题时 issues 用空数组。"""
QUESTION_PROMPT = (
    "你是实验室文档助手。回答质控数量时必须调用 count_qc_reports。"
    "数字只能使用工具返回的 total、pass_count、fail_count、review_count，并用简短中文说明。"
)
QC_COUNT_TOOL: ChatCompletionFunctionToolParam = {
    "type": "function",
    "function": {
        "name": "count_qc_reports",
        "description": "统计当前用户的质控记录数量，返回 total、pass_count、fail_count、review_count。",
    },
}
METRIC_PROMPT = (
    "你是实验室文档助手。回答测量值问题时必须调用 summarize_measurements。"
    "数字只能使用工具返回的 mean 和 anomaly_count，并用简短中文说明。"
)
METRIC_TOOL: ChatCompletionFunctionToolParam = {
    "type": "function",
    "function": {
        "name": "summarize_measurements",
        "description": "用 pandas 计算测量值的均值，以及大于阈值的异常个数。",
    },
}
CHART_PROMPT = (
    "你是实验室质控助手。只用给出的均值、异常个数和阈值写一句简短中文结论。"
    "不要改动这些数字，也不要补充未给出的测量值。"
)

load_dotenv()


engine = create_engine("sqlite:///labassist.db")
Base.metadata.create_all(engine)


def _ensure_batch_user_id() -> None:
    """给已有的 batches 表补上 user_id。create_all 不会修改已经存在的表。"""
    columns = {column["name"] for column in inspect(engine).get_columns("batches")}
    if "user_id" not in columns:
        with engine.begin() as connection:
            connection.execute(
                sql_text("ALTER TABLE batches ADD COLUMN user_id INTEGER")
            )


_ensure_batch_user_id()

# 缺少令牌时自己返回 401。HTTPBearer 默认会返回 403。
_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> UserOut:
    """从 Authorization 头里的 JWT 找出当前用户。没有或无效时返回 401。"""
    if credentials is None:
        raise HTTPException(status_code=401, detail="未登录")
    try:
        payload = jwt.decode(
            credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM]
        )
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="登录已失效") from None
    username = payload.get("sub")
    if not isinstance(username, str):
        raise HTTPException(status_code=401, detail="登录已失效")
    with Session(engine) as session:
        record = session.scalar(select(User).where(User.username == username))
        if record is None:
            raise HTTPException(status_code=401, detail="登录已失效")
        return UserOut(id=record.id, username=record.username)


def _to_batch(record: Batch) -> BatchOut:
    """把数据库行转成接口响应的 schema。"""
    return BatchOut(
        id=record.id,
        status=record.status,
        user_id=record.user_id,
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
def create_batch(
    body: BatchCreate, user: UserOut = Depends(get_current_user)
) -> BatchOut:
    """创建一条送检批次，并分配自增编号。记录归当前登录用户所有。"""
    record = Batch(
        status="pending",
        user_id=user.id,
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
def list_batches(user: UserOut = Depends(get_current_user)) -> list[BatchOut]:
    """返回当前用户自己的记录。"""
    with Session(engine) as session:
        records = session.scalars(
            select(Batch).where(Batch.user_id == user.id).order_by(Batch.id)
        ).all()
        return [_to_batch(record) for record in records]


@app.get("/batches/{batch_id}")
def get_batch(batch_id: int, user: UserOut = Depends(get_current_user)) -> BatchOut:
    """按编号返回一条记录。编号不存在或不属于当前用户时返回 404。"""
    with Session(engine) as session:
        record = session.get(Batch, batch_id)
        if record is None or record.user_id != user.id:
            raise HTTPException(status_code=404, detail="批次不存在")
        return _to_batch(record)


@app.post("/batches/import", status_code=201)
async def import_batches(
    file: UploadFile = File(
        description="UTF-8 CSV，表头：sample_id,sample_name,sample_type,org_name,accept_time"
    ),
    user: UserOut = Depends(get_current_user),
) -> list[BatchOut]:
    """按行创建记录。每一行使用和 POST /batches 相同的字段校验。"""
    text = (await file.read()).decode("utf-8-sig")
    reader = csv.DictReader(StringIO(text))
    created: list[BatchOut] = []
    for line_no, row in enumerate(reader, start=2):
        try:
            body = BatchCreate.model_validate(row)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail={"line": line_no, "errors": exc.errors()},
            ) from exc
        created.append(create_batch(body, user))
    return created


def _hash_password(password: str) -> str:
    """用随机盐和 PBKDF2 生成密码摘要。"""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return f"{salt}${digest.hex()}"


@app.post("/users", status_code=201)
def register(body: UserCreate) -> UserOut:
    """注册用户。用户名已存在时返回 409。"""
    record = User(username=body.username, password_hash=_hash_password(body.password))
    with Session(engine) as session:
        session.add(record)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            raise HTTPException(status_code=409, detail="用户名已存在") from None
        session.refresh(record)
        return UserOut(id=record.id, username=record.username)


def _verify_password(password: str, password_hash: str) -> bool:
    """用保存的盐重新计算摘要，并和数据库中的摘要比较。"""
    salt, digest_hex = password_hash.split("$", 1)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return secrets.compare_digest(digest.hex(), digest_hex)


def _create_access_token(username: str) -> str:
    """签发 2 小时有效的 JWT，sub 里放用户名。"""
    payload = {
        "sub": username,
        "exp": datetime.now(timezone.utc) + timedelta(hours=2),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


@app.post("/login")
def login(body: Login) -> TokenOut:
    """登录。用户名或密码错误时返回 401。"""
    with Session(engine) as session:
        record = session.scalar(select(User).where(User.username == body.username))
        if record is None or not _verify_password(body.password, record.password_hash):
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        return TokenOut(
            access_token=_create_access_token(record.username), token_type="bearer"
        )


def _model_setting(name: str) -> str:
    """读取模型配置。没填时返回 500，避免把空密钥发给接口。"""
    value = os.environ.get(name, "")
    if not value:
        raise HTTPException(status_code=500, detail=f"缺少环境变量 {name}")
    return value


def _model_client() -> OpenAI:
    """按环境变量创建 OpenAI 兼容客户端。"""
    return OpenAI(
        api_key=_model_setting("OPENAI_API_KEY"),
        base_url=_model_setting("OPENAI_BASE_URL"),
        max_retries=0,
    )


def _stored_message(role: str, content: str) -> ChatCompletionMessageParam:
    """把数据库里的角色收成模型接受的 user 或 assistant。"""
    if role == "assistant":
        return {"role": "assistant", "content": content}
    return {"role": "user", "content": content}


def _ask_model(messages: list[ChatCompletionMessageParam]) -> str:
    """把对话发给 OpenAI 兼容接口，返回助手的文字。"""
    completion = _model_client().chat.completions.create(
        model=_model_setting("OPENAI_MODEL"),
        messages=messages,
    )
    content = completion.choices[0].message.content
    if not content:
        raise HTTPException(status_code=502, detail="模型没有返回文字")
    return content


def _stream_model(messages: list[ChatCompletionMessageParam]):
    """逐段取出模型返回的文字。没有文字的片段会被跳过。"""
    stream = _model_client().chat.completions.create(
        model=_model_setting("OPENAI_MODEL"),
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _to_message(record: Message) -> MessageOut:
    """把数据库行转成接口响应的 schema。"""
    return MessageOut(id=record.id, role=record.role, content=record.content)


@app.post("/chat")
def chat(
    body: ChatCreate, user: UserOut = Depends(get_current_user)
) -> StreamingResponse:
    """按 SSE 逐段返回模型的文字。全部片段到齐后，仍把双方的话一次写入数据库。"""
    with Session(engine) as session:
        history = session.scalars(
            select(Message).where(Message.user_id == user.id).order_by(Message.id)
        ).all()
        previous = [_stored_message(record.role, record.content) for record in history]
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *previous,
        {"role": "user", "content": body.content},
    ]

    def events():
        parts: list[str] = []
        for delta in _stream_model(messages):
            parts.append(delta)
            yield f"data: {json.dumps(delta, ensure_ascii=False)}\n\n"
        reply = "".join(parts)
        if not reply:
            return
        now = datetime.now(timezone.utc)
        with Session(engine) as session:
            session.add(
                Message(
                    user_id=user.id, role="user", content=body.content, created_at=now
                )
            )
            session.add(
                Message(
                    user_id=user.id, role="assistant", content=reply, created_at=now
                )
            )
            session.commit()

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/messages")
def list_messages(user: UserOut = Depends(get_current_user)) -> list[MessageOut]:
    """返回当前用户的全部对话。记录在数据库里，重启服务后仍然在。"""
    with Session(engine) as session:
        records = session.scalars(
            select(Message).where(Message.user_id == user.id).order_by(Message.id)
        ).all()
        return [_to_message(record) for record in records]


def _to_qc(record: QcReport) -> QcOut:
    """把数据库行转成接口响应的 schema。"""
    result = QcResult.model_validate(
        {
            "sample_id": record.sample_id,
            "conclusion": record.conclusion,
            "issues": json.loads(record.issues),
            "summary": record.summary,
        }
    )
    return QcOut(
        id=record.id,
        sample_id=result.sample_id,
        conclusion=result.conclusion,
        issues=result.issues,
        summary=result.summary,
        prompt_tokens=record.prompt_tokens,
        completion_tokens=record.completion_tokens,
        total_tokens=record.total_tokens,
    )


def _parse_qc(description: str) -> tuple[QcResult, int, int, int]:
    """让模型输出 JSON，再用 QcResult 检查字段，并取出 token 用量。"""
    completion = _model_client().chat.completions.create(
        model=_model_setting("OPENAI_MODEL"),
        messages=[
            {"role": "system", "content": QC_PROMPT},
            {"role": "user", "content": description},
        ],
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content
    usage = completion.usage
    if (
        not content
        or usage is None
        or usage.prompt_tokens is None
        or usage.completion_tokens is None
        or usage.total_tokens is None
    ):
        raise HTTPException(status_code=502, detail="模型没有返回质控 JSON")
    try:
        result = QcResult.model_validate_json(content)
    except ValidationError as exc:
        raise HTTPException(
            status_code=502, detail="模型返回的 JSON 不符合质控结构"
        ) from exc
    return result, usage.prompt_tokens, usage.completion_tokens, usage.total_tokens


@app.post("/qc-reports", status_code=201)
def create_qc_report(
    body: QcCreate, user: UserOut = Depends(get_current_user)
) -> QcOut:
    """解析质控描述并保存。token 用量和 JSON 字段一起写入数据库。"""
    result, prompt_tokens, completion_tokens, total_tokens = _parse_qc(body.description)
    record = QcReport(
        user_id=user.id,
        sample_id=result.sample_id,
        conclusion=result.conclusion,
        issues=json.dumps(result.issues, ensure_ascii=False),
        summary=result.summary,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        created_at=datetime.now(timezone.utc),
    )
    with Session(engine) as session:
        session.add(record)
        session.commit()
        session.refresh(record)
        return _to_qc(record)


@app.get("/qc-reports")
def list_qc_reports(user: UserOut = Depends(get_current_user)) -> list[QcOut]:
    """返回当前用户保存的质控结果。"""
    with Session(engine) as session:
        records = session.scalars(
            select(QcReport).where(QcReport.user_id == user.id).order_by(QcReport.id)
        ).all()
        return [_to_qc(record) for record in records]


def _count_qc_reports(user_id: int) -> QcCount:
    """按结论统计当前用户的质控记录。数字只来自数据库。"""
    with Session(engine) as session:
        conclusions = session.scalars(
            select(QcReport.conclusion).where(QcReport.user_id == user_id)
        ).all()
    return QcCount(
        total=len(conclusions),
        pass_count=sum(item == "pass" for item in conclusions),
        fail_count=sum(item == "fail" for item in conclusions),
        review_count=sum(item == "review" for item in conclusions),
    )


def _ask_about_qc(content: str, user_id: int) -> tuple[str, QcCount]:
    """先让模型调用查询工具，再根据工具返回的数字生成回答。"""
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": QUESTION_PROMPT},
        {"role": "user", "content": content},
    ]
    first = _model_client().chat.completions.create(
        model=_model_setting("OPENAI_MODEL"),
        messages=messages,
        tools=[QC_COUNT_TOOL],
        tool_choice={"type": "function", "function": {"name": "count_qc_reports"}},
        extra_body={"thinking": {"type": "disabled"}},
    )
    tool_calls = first.choices[0].message.tool_calls
    if not tool_calls:
        raise HTTPException(status_code=502, detail="模型没有查询数据库")
    call = tool_calls[0]
    if call.type != "function" or call.function.name != "count_qc_reports":
        raise HTTPException(status_code=502, detail="模型没有查询数据库")
    count = _count_qc_reports(user_id)
    assistant: ChatCompletionAssistantMessageParam = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
        ],
    }
    messages.append(assistant)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(count.model_dump(), ensure_ascii=False),
        }
    )
    second = _model_client().chat.completions.create(
        model=_model_setting("OPENAI_MODEL"),
        messages=messages,
        extra_body={"thinking": {"type": "disabled"}},
    )
    answer = second.choices[0].message.content
    if not answer:
        raise HTTPException(status_code=502, detail="模型没有返回文字")
    return answer, count


@app.post("/questions")
def ask_question(
    body: QuestionCreate, user: UserOut = Depends(get_current_user)
) -> QuestionOut:
    """回答质控数量问题。响应里的数字来自数据库，不由模型编造。"""
    answer, count = _ask_about_qc(body.content, user.id)
    return QuestionOut(
        answer=answer,
        total=count.total,
        pass_count=count.pass_count,
        fail_count=count.fail_count,
        review_count=count.review_count,
    )


def _summarize_measurements(values: list[float], threshold: float) -> MetricStats:
    """用 pandas 计算均值，并数出大于阈值的异常值。"""
    series = pd.Series(values, dtype="float64")
    if series.empty or bool(series.isna().any()):
        raise HTTPException(status_code=400, detail="没有可计算的数值")
    return MetricStats(
        mean=round(float(series.mean()), 2),
        anomaly_count=int((series > threshold).sum()),
    )


def _ask_about_metrics(
    content: str, values: list[float], threshold: float
) -> tuple[str, MetricStats]:
    """先让模型调用计算工具。pandas 失败时直接返回错误，不再生成文字。"""
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": METRIC_PROMPT},
        {"role": "user", "content": content},
    ]
    first = _model_client().chat.completions.create(
        model=_model_setting("OPENAI_MODEL"),
        messages=messages,
        tools=[METRIC_TOOL],
        tool_choice={
            "type": "function",
            "function": {"name": "summarize_measurements"},
        },
        extra_body={"thinking": {"type": "disabled"}},
    )
    tool_calls = first.choices[0].message.tool_calls
    if not tool_calls:
        raise HTTPException(status_code=502, detail="模型没有调用计算工具")
    call = tool_calls[0]
    if call.type != "function" or call.function.name != "summarize_measurements":
        raise HTTPException(status_code=502, detail="模型没有调用计算工具")
    stats = _summarize_measurements(values, threshold)
    assistant: ChatCompletionAssistantMessageParam = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
        ],
    }
    messages.append(assistant)
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(stats.model_dump(), ensure_ascii=False),
        }
    )
    second = _model_client().chat.completions.create(
        model=_model_setting("OPENAI_MODEL"),
        messages=messages,
        extra_body={"thinking": {"type": "disabled"}},
    )
    answer = second.choices[0].message.content
    if not answer:
        raise HTTPException(status_code=502, detail="模型没有返回文字")
    return answer, stats


@app.post("/metrics")
def ask_metrics(
    body: MetricCreate, user: UserOut = Depends(get_current_user)
) -> MetricOut:
    """根据 pandas 的计算结果回答。计算失败时不会让模型编造数字。"""
    answer, stats = _ask_about_metrics(body.content, body.values, body.threshold)
    return MetricOut(answer=answer, mean=stats.mean, anomaly_count=stats.anomaly_count)


def _split_text(
    text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """按固定长度切块，相邻块保留一段重叠。"""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(text):
            break
        start = end - chunk_overlap
    return chunks


def _get_embedder() -> TextEmbedding:
    """加载本机中文向量模型。DeepSeek 的对话接口不提供向量。"""
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _embedder


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """把每个文本块转成一条向量。"""
    vectors = [vector.tolist() for vector in _get_embedder().embed(texts)]
    if len(vectors) != len(texts):
        raise HTTPException(status_code=500, detail="向量数量和文本块不一致")
    return vectors


def _to_document(record: Document, chunks: list[Chunk]) -> DocumentOut:
    """把文档和文本块转成接口响应。"""
    dims = [len(json.loads(chunk.embedding)) for chunk in chunks]
    return DocumentOut(
        id=record.id,
        title=record.title,
        chunk_count=len(chunks),
        embedding_dim=dims[0] if dims else 0,
        chunks=[
            ChunkOut(position=chunk.position, content=chunk.content, embedding_dim=dim)
            for chunk, dim in zip(chunks, dims)
        ],
    )


def _load_chunks(session: Session, document_id: int, user_id: int) -> list[Chunk]:
    """读取某篇文档下、属于当前用户的文本块。"""
    return list(
        session.scalars(
            select(Chunk)
            .where(Chunk.document_id == document_id, Chunk.user_id == user_id)
            .order_by(Chunk.position)
        ).all()
    )


@app.post("/documents", status_code=201)
def create_document(
    body: DocumentCreate, user: UserOut = Depends(get_current_user)
) -> DocumentOut:
    """切分正文，生成向量，并和文本块一起写入数据库。"""
    pieces = _split_text(body.content)
    if not pieces:
        raise HTTPException(status_code=400, detail="没有可入库的文本")
    vectors = _embed_texts(pieces)
    now = datetime.now(timezone.utc)
    with Session(engine) as session:
        document = Document(user_id=user.id, title=body.title, created_at=now)
        session.add(document)
        session.commit()
        session.refresh(document)
        stored = [
            Chunk(
                document_id=document.id,
                user_id=user.id,
                position=position,
                content=piece,
                embedding=json.dumps(vector),
            )
            for position, (piece, vector) in enumerate(zip(pieces, vectors))
        ]
        session.add_all(stored)
        session.commit()
        for chunk in stored:
            session.refresh(chunk)
        return _to_document(document, stored)


@app.get("/documents")
def list_documents(user: UserOut = Depends(get_current_user)) -> list[DocumentOut]:
    """返回当前用户已入库的文档。重启服务后仍然在。"""
    with Session(engine) as session:
        documents = session.scalars(
            select(Document).where(Document.user_id == user.id).order_by(Document.id)
        ).all()
        return [
            _to_document(document, _load_chunks(session, document.id, user.id))
            for document in documents
        ]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算两条向量的余弦相似度。"""
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _search_chunks(user_id: int, question: str) -> list[CitationOut]:
    """在当前用户的文本块里找出足够相似的原文。"""
    query_vector = _embed_texts([question])[0]
    with Session(engine) as session:
        rows = session.execute(
            select(Chunk, Document)
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.user_id == user_id)
        ).all()
    ranked: list[tuple[float, CitationOut]] = []
    for chunk, document in rows:
        vector: list[float] = json.loads(chunk.embedding)
        score = _cosine_similarity(query_vector, vector)
        if score < RETRIEVAL_MIN_SCORE:
            continue
        ranked.append(
            (
                score,
                CitationOut(
                    document_id=document.id,
                    title=document.title,
                    position=chunk.position,
                    content=chunk.content,
                ),
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [citation for _, citation in ranked[:RETRIEVAL_LIMIT]]


def _knowledge_messages(
    question: str, citations: list[CitationOut]
) -> list[ChatCompletionMessageParam]:
    """把检索到的原文和问题组成发给模型的消息。"""
    excerpts = "\n".join(
        f"[{index}] {item.content}" for index, item in enumerate(citations, start=1)
    )
    return [
        {"role": "system", "content": KNOWLEDGE_PROMPT},
        {"role": "user", "content": f"原文：\n{excerpts}\n\n问题：{question}"},
    ]


def _ask_with_citations(question: str, citations: list[CitationOut]) -> str:
    """把检索到的原文交给模型，由模型写成回答。"""
    return _ask_model(_knowledge_messages(question, citations))


@app.post("/answers")
def answer_from_documents(
    body: AnswerCreate, user: UserOut = Depends(get_current_user)
) -> AnswerOut:
    """用当前用户的文档回答。没有足够相似的原文时，不调用模型，直接回答不知道。"""
    citations = _search_chunks(user.id, body.content)
    if not citations:
        return AnswerOut(answer="不知道", citations=[])
    return AnswerOut(
        answer=_ask_with_citations(body.content, citations), citations=citations
    )


@app.post("/answers/stream")
def stream_answer(
    body: AnswerCreate, user: UserOut = Depends(get_current_user)
) -> StreamingResponse:
    """先返回引用的原文，再逐段返回回答。没有依据时只返回不知道。"""
    citations = _search_chunks(user.id, body.content)
    messages = _knowledge_messages(body.content, citations) if citations else None

    def events():
        yield f"event: citations\ndata: {json.dumps([item.model_dump() for item in citations], ensure_ascii=False)}\n\n"
        if messages is None:
            yield f"event: delta\ndata: {json.dumps('不知道', ensure_ascii=False)}\n\n"
            return
        for delta in _stream_model(messages):
            yield f"event: delta\ndata: {json.dumps(delta, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


GOLD_DOCUMENTS: list[tuple[str, str]] = [
    (
        "样本接收",
        "收到样本后先核对编号和送检单是否一致。编号不一致时停止检测，并记录异常原因。核对完成后把样本交给检测岗位。交接时双方在接收记录上签名，并注明接收时间。",
    ),
    (
        "样本保存",
        "全血样本放在 2 到 8 摄氏度冷藏，不要冷冻。保存时间不超过 24 小时。超时样本需要重新采集。冷藏箱温度每天记录两次，超出范围时转移样本并上报。",
    ),
    (
        "试剂准备",
        "试剂使用前从冷藏取出，恢复到室温后再打开。不同批号的试剂不要混用。开瓶时间写在瓶身标签上。开瓶后的试剂按说明书期限使用，过期试剂不得继续检测。",
    ),
    (
        "仪器校准",
        "每天开机后先做空白校准，空白通过后才能检测样本。校准失败时停止上机，并通知设备管理员。管理员排除故障前，不得改用备用参数继续检测。",
    ),
    (
        "室内质控",
        "每批检测都要加入阴性质控和阳性质控。质控未通过时，该批样本结果不得发出。质控图上的点超出规则时，先查找原因，再决定是否复测。",
    ),
    (
        "结果复核",
        "超出参考范围的结果由另一名人员复核。复核人与检测人不能是同一人。复核意见写入原始记录。复核不同意时，结果保持未发布，直到重新检测完成。",
    ),
    (
        "废弃处理",
        "使用过的针头放入利器盒，不得丢进生活垃圾桶。利器盒装到三分之二时封口，按医疗废物交接。交接单上写明数量和日期，生活垃圾通道不能接收利器盒。",
    ),
    (
        "报告发放",
        "报告经审核人签字后才能发放。未签字的报告不得告诉受检者结果。发放时核对受检者姓名和样本编号。电话告知结果同样要先完成审核，并留下告知记录。",
    ),
    (
        "生物安全",
        "实验操作戴手套和口罩。发生样本溅出时，先覆盖消毒，再清理台面，并记录事件。消毒接触时间按消毒液说明书执行，未处理完不得继续检测下一批。",
    ),
    (
        "记录归档",
        "原始记录当天归档，保存期限按实验室规定执行。记录不得涂改，写错时划改并签名。归档后借出记录要登记归还日期，缺页记录不能入库。",
    ),
]

GOLD_QUESTIONS: list[tuple[str, str]] = [
    ("编号和送检单对不上怎么办", "样本接收"),
    ("样本核对之后交给谁", "样本接收"),
    ("全血应该放在多少度", "样本保存"),
    ("冷藏超过一天的样本还能用吗", "样本保存"),
    ("试剂拿出冷藏后要怎么处理", "试剂准备"),
    ("两个批号的试剂能倒在一起吗", "试剂准备"),
    ("每天开机第一件事是什么", "仪器校准"),
    ("空白校准失败了还能测样本吗", "仪器校准"),
    ("每一批检测要加什么质控", "室内质控"),
    ("质控没过的这批结果能发吗", "室内质控"),
    ("结果高出参考范围谁来看", "结果复核"),
    ("检测人可以自己复核自己的结果吗", "结果复核"),
    ("用过的针头扔哪里", "废弃处理"),
    ("利器盒什么时候封口", "废弃处理"),
    ("报告没有签字能告诉受检者吗", "报告发放"),
    ("发放报告时要核对哪些信息", "报告发放"),
    ("样本溅到台面上怎么处理", "生物安全"),
    ("做实验要戴什么防护", "生物安全"),
    ("原始记录当天要做什么", "记录归档"),
    ("记录写错了能不能涂掉", "记录归档"),
]


def _gold_chunks(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[tuple[str, str, list[float]]]:
    """把金标 SOP 切块并转成向量。切分方式和文档入库相同。"""
    titles: list[str] = []
    pieces: list[str] = []
    for title, content in GOLD_DOCUMENTS:
        for piece in _split_text(content, chunk_size, chunk_overlap):
            titles.append(title)
            pieces.append(piece)
    vectors = _embed_texts(pieces)
    return list(zip(titles, pieces, vectors))


def _evaluate_retrieval(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    require_top: bool = False,
) -> list[EvaluationItemOut]:
    """用同一套相似度规则检索 20 道题，并对照应引用的段落。"""
    chunks = _gold_chunks(chunk_size, chunk_overlap)
    questions = [question for question, _ in GOLD_QUESTIONS]
    query_vectors = _embed_texts(questions)
    items: list[EvaluationItemOut] = []
    for (question, expected_title), query_vector in zip(GOLD_QUESTIONS, query_vectors):
        ranked: list[tuple[float, EvaluationCitation]] = []
        for title, content, vector in chunks:
            score = _cosine_similarity(query_vector, vector)
            if score < RETRIEVAL_MIN_SCORE:
                continue
            ranked.append((score, EvaluationCitation(title=title, content=content)))
        ranked.sort(key=lambda item: item[0], reverse=True)
        citations = [citation for _, citation in ranked[:RETRIEVAL_LIMIT]]
        if require_top:
            hit = bool(citations) and citations[0].title == expected_title
        else:
            hit = any(citation.title == expected_title for citation in citations)
        items.append(
            EvaluationItemOut(
                question=question,
                expected_title=expected_title,
                hit=hit,
                citations=citations,
            )
        )
    return items


def _chunk_run(
    chunk_size: int, chunk_overlap: int, items: list[EvaluationItemOut]
) -> ChunkRunOut:
    """把一种切块设置的结果收成命中率。"""
    hit_count = sum(1 for item in items if item.hit)
    total = len(items)
    return ChunkRunOut(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        total=total,
        hit_count=hit_count,
        hit_rate=round(hit_count / total, 2),
    )


def _to_evaluation(record: Evaluation) -> EvaluationOut:
    """把一次评测记录转成接口响应。"""
    items = [
        EvaluationItemOut.model_validate(item) for item in json.loads(record.detail)
    ]
    return EvaluationOut(
        id=record.id,
        total=record.total,
        hit_count=record.hit_count,
        hit_rate=record.hit_rate,
        items=items,
    )


@app.post("/evaluations", status_code=201)
def create_evaluation(user: UserOut = Depends(get_current_user)) -> EvaluationOut:
    """对 20 条金标问题做检索，保存命中率。数字来自向量比较，不由模型生成。"""
    items = _evaluate_retrieval()
    hit_count = sum(1 for item in items if item.hit)
    total = len(items)
    now = datetime.now(timezone.utc)
    with Session(engine) as session:
        record = Evaluation(
            user_id=user.id,
            total=total,
            hit_count=hit_count,
            hit_rate=round(hit_count / total, 2),
            detail=json.dumps(
                [item.model_dump() for item in items], ensure_ascii=False
            ),
            created_at=now,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return _to_evaluation(record)


@app.get("/evaluations")
def list_evaluations(user: UserOut = Depends(get_current_user)) -> list[EvaluationOut]:
    """返回当前用户做过的评测。重启服务后仍然在。"""
    with Session(engine) as session:
        records = session.scalars(
            select(Evaluation)
            .where(Evaluation.user_id == user.id)
            .order_by(Evaluation.id)
        ).all()
        return [_to_evaluation(record) for record in records]


def _to_comparison(record: EvaluationComparison) -> EvaluationComparisonOut:
    """把一次切块对比转成接口响应。"""
    misses = [
        EvaluationItemOut.model_validate(item) for item in json.loads(record.detail)
    ]
    return EvaluationComparisonOut(
        id=record.id,
        before=ChunkRunOut(
            chunk_size=record.before_chunk_size,
            chunk_overlap=record.before_chunk_overlap,
            total=len(GOLD_QUESTIONS),
            hit_count=record.before_hit_count,
            hit_rate=record.before_hit_rate,
        ),
        after=ChunkRunOut(
            chunk_size=record.after_chunk_size,
            chunk_overlap=record.after_chunk_overlap,
            total=len(GOLD_QUESTIONS),
            hit_count=record.after_hit_count,
            hit_rate=record.after_hit_rate,
        ),
        misses=misses,
    )


@app.post("/evaluation-comparisons", status_code=201)
def create_evaluation_comparison(
    user: UserOut = Depends(get_current_user),
) -> EvaluationComparisonOut:
    """用原来的切块和改短后的切块各测一遍，两次命中率一起保存。"""
    before_items = _evaluate_retrieval(require_top=True)
    after_items = _evaluate_retrieval(
        ADJUSTED_CHUNK_SIZE, ADJUSTED_CHUNK_OVERLAP, require_top=True
    )
    before = _chunk_run(CHUNK_SIZE, CHUNK_OVERLAP, before_items)
    after = _chunk_run(ADJUSTED_CHUNK_SIZE, ADJUSTED_CHUNK_OVERLAP, after_items)
    misses = [item for item in after_items if not item.hit]
    now = datetime.now(timezone.utc)
    with Session(engine) as session:
        record = EvaluationComparison(
            user_id=user.id,
            before_chunk_size=before.chunk_size,
            before_chunk_overlap=before.chunk_overlap,
            before_hit_count=before.hit_count,
            before_hit_rate=before.hit_rate,
            after_chunk_size=after.chunk_size,
            after_chunk_overlap=after.chunk_overlap,
            after_hit_count=after.hit_count,
            after_hit_rate=after.hit_rate,
            detail=json.dumps(
                [item.model_dump() for item in misses], ensure_ascii=False
            ),
            created_at=now,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return _to_comparison(record)


@app.get("/evaluation-comparisons")
def list_evaluation_comparisons(
    user: UserOut = Depends(get_current_user),
) -> list[EvaluationComparisonOut]:
    """返回当前用户的切块对比。重启服务后仍然在。"""
    with Session(engine) as session:
        records = session.scalars(
            select(EvaluationComparison)
            .where(EvaluationComparison.user_id == user.id)
            .order_by(EvaluationComparison.id)
        ).all()
        return [_to_comparison(record) for record in records]


def _read_qc_csv(raw: bytes) -> tuple[list[str], list[float]]:
    """读取合成质控 CSV。测量值在 value 列，样本编号在 sample_id 列。"""
    try:
        text = raw.decode("utf-8-sig")
        frame = pd.read_csv(StringIO(text))
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="质控 CSV 需要 UTF-8 编码") from exc
    except pd.errors.ParserError as exc:
        raise HTTPException(status_code=400, detail="质控 CSV 无法读取") from exc
    except pd.errors.EmptyDataError as exc:
        raise HTTPException(status_code=400, detail="没有可计算的数值") from exc
    if "value" not in frame.columns:
        raise HTTPException(status_code=400, detail="质控 CSV 需要 value 列")
    if "sample_id" in frame.columns:
        labels = [str(item) for item in frame["sample_id"].tolist()]
    else:
        labels = [str(index) for index in range(1, len(frame) + 1)]
    numeric = pd.Series(pd.to_numeric(frame["value"], errors="coerce"), dtype="float64")
    values = [float(item) for item in numeric.tolist()]
    return labels, values


def _render_qc_chart(labels: list[str], values: list[float], threshold: float) -> bytes:
    """用测量值和阈值画一张质控图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(labels, values, marker="o", color="#2f6f8f")
    ax.axhline(threshold, color="#8a5a2a", linestyle="--")
    ax.set_xlabel("样本")
    ax.set_ylabel("测量值")
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()


def _conclude_chart(stats: MetricStats, threshold: float) -> str:
    """让模型根据 pandas 的数字写一句结论。"""
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": CHART_PROMPT},
        {
            "role": "user",
            "content": f"阈值 {threshold}，均值 {stats.mean}，异常个数 {stats.anomaly_count}。",
        },
    ]
    return _ask_model(messages)


def _to_qc_chart(record: QcChart) -> QcChartOut:
    """把质控图记录转成接口响应。"""
    return QcChartOut(
        id=record.id,
        threshold=record.threshold,
        mean=record.mean,
        anomaly_count=record.anomaly_count,
        conclusion=record.conclusion,
        image_base64=base64.b64encode(record.image).decode("ascii"),
    )


@app.post("/qc-charts", status_code=201)
async def create_qc_chart(
    file: UploadFile = File(description="UTF-8 CSV，表头至少包含 value，可选 sample_id"),
    threshold: float = Form(description="大于该值的测量值计为异常"),
    user: UserOut = Depends(get_current_user),
) -> QcChartOut:
    """先用 pandas 计算并画图。计算失败时不调用模型。"""
    labels, values = _read_qc_csv(await file.read())
    stats = _summarize_measurements(values, threshold)
    image = _render_qc_chart(labels, values, threshold)
    conclusion = _conclude_chart(stats, threshold)
    now = datetime.now(timezone.utc)
    with Session(engine) as session:
        record = QcChart(
            user_id=user.id,
            threshold=threshold,
            mean=stats.mean,
            anomaly_count=stats.anomaly_count,
            conclusion=conclusion,
            image=image,
            created_at=now,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return _to_qc_chart(record)


app.mount(
    "/ui",
    StaticFiles(
        directory=Path(__file__).resolve().parent.parent / "web" / "dist", html=True
    ),
    name="ui",
)
