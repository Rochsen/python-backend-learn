"""第 9 课：用 pytest 调用接口，确认登录和越权行为。

测试使用临时数据库，不会写入 labassist.db。
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

import app.main as main

BATCH = {
    "sample_id": "S001",
    "sample_name": "样本甲",
    "sample_type": "血液",
    "org_name": "检验科",
    "accept_time": "2026-10-18 16:00:00",
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    """每个测试使用一份新的 SQLite，测完即丢。"""
    database = tmp_path / "test.db"
    test_engine = create_engine(f"sqlite:///{database.as_posix()}")
    main.Base.metadata.create_all(test_engine)
    monkeypatch.setattr(main, "engine", test_engine)
    with TestClient(main.app) as test_client:
        yield test_client


def _register(client: TestClient, username: str, password: str = "secret1") -> None:
    """注册一个用户，供后面的登录测试使用。"""
    response = client.post("/users", json={"username": username, "password": password})
    assert response.status_code == 201


def test_login_success(client: TestClient) -> None:
    """正确密码返回 200，并且带有 access_token。"""
    _register(client, "alice")
    response = client.post("/login", json={"username": "alice", "password": "secret1"})
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]


def test_login_wrong_password(client: TestClient) -> None:
    """密码错误返回 401。"""
    _register(client, "alice")
    response = client.post("/login", json={"username": "alice", "password": "wrong1"})
    assert response.status_code == 401


def test_batches_require_login(client: TestClient) -> None:
    """未登录访问批次列表返回 401。"""
    response = client.get("/batches")
    assert response.status_code == 401


def test_cannot_read_other_users_batch(client: TestClient) -> None:
    """一个用户不能读取另一个用户的批次。"""
    _register(client, "alice")
    _register(client, "bob")
    alice = client.post("/login", json={"username": "alice", "password": "secret1"})
    bob = client.post("/login", json={"username": "bob", "password": "secret1"})
    created = client.post(
        "/batches",
        json=BATCH,
        headers={"Authorization": f"Bearer {alice.json()['access_token']}"},
    )
    assert created.status_code == 201
    response = client.get(
        f"/batches/{created.json()['id']}",
        headers={"Authorization": f"Bearer {bob.json()['access_token']}"},
    )
    assert response.status_code == 404
