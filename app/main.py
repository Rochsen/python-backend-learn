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
"""

import csv
import hashlib
import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Literal
from io import StringIO

import jwt
import pandas as pd

# 本机访问 huggingface.co 会超时，向量模型改从国内镜像下载。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from fastembed import TextEmbedding
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from openai import OpenAI
from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionFunctionToolParam, ChatCompletionMessageParam
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine, inspect, select
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


# 第 7 课把签名密钥放在代码里，用来签发和校验 JWT。
JWT_SECRET = "labassist-lesson-7"
JWT_ALGORITHM = "HS256"
CHUNK_SIZE = 60
CHUNK_OVERLAP = 15
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
_embedder: TextEmbedding | None = None
SYSTEM_PROMPT = "你是实验室文档助手，用简短中文回答。"
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

load_dotenv()


engine = create_engine("sqlite:///labassist.db")
Base.metadata.create_all(engine)


def _ensure_batch_user_id() -> None:
    """给已有的 batches 表补上 user_id。create_all 不会修改已经存在的表。"""
    columns = {column["name"] for column in inspect(engine).get_columns("batches")}
    if "user_id" not in columns:
        with engine.begin() as connection:
            connection.execute(sql_text("ALTER TABLE batches ADD COLUMN user_id INTEGER"))


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
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
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
def create_batch(body: BatchCreate, user: UserOut = Depends(get_current_user)) -> BatchOut:
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
    file: UploadFile = File(description="UTF-8 CSV，表头：sample_id,sample_name,sample_type,org_name,accept_time"),
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
        return TokenOut(access_token=_create_access_token(record.username), token_type="bearer")


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
def chat(body: ChatCreate, user: UserOut = Depends(get_current_user)) -> StreamingResponse:
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
            session.add(Message(user_id=user.id, role="user", content=body.content, created_at=now))
            session.add(Message(user_id=user.id, role="assistant", content=reply, created_at=now))
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
        raise HTTPException(status_code=502, detail="模型返回的 JSON 不符合质控结构") from exc
    return result, usage.prompt_tokens, usage.completion_tokens, usage.total_tokens


@app.post("/qc-reports", status_code=201)
def create_qc_report(body: QcCreate, user: UserOut = Depends(get_current_user)) -> QcOut:
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
def ask_question(body: QuestionCreate, user: UserOut = Depends(get_current_user)) -> QuestionOut:
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


def _ask_about_metrics(content: str, values: list[float], threshold: float) -> tuple[str, MetricStats]:
    """先让模型调用计算工具。pandas 失败时直接返回错误，不再生成文字。"""
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": METRIC_PROMPT},
        {"role": "user", "content": content},
    ]
    first = _model_client().chat.completions.create(
        model=_model_setting("OPENAI_MODEL"),
        messages=messages,
        tools=[METRIC_TOOL],
        tool_choice={"type": "function", "function": {"name": "summarize_measurements"}},
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
def ask_metrics(body: MetricCreate, user: UserOut = Depends(get_current_user)) -> MetricOut:
    """根据 pandas 的计算结果回答。计算失败时不会让模型编造数字。"""
    answer, stats = _ask_about_metrics(body.content, body.values, body.threshold)
    return MetricOut(answer=answer, mean=stats.mean, anomaly_count=stats.anomaly_count)


def _split_text(text: str) -> list[str]:
    """按固定长度切块，相邻块保留一段重叠。"""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP
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
def create_document(body: DocumentCreate, user: UserOut = Depends(get_current_user)) -> DocumentOut:
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
