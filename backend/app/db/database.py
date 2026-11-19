"""SQLite 持久化层：用户账号（JWT 认证）+ 对话历史（按用户 id 隔离）。

标准库 sqlite3，WAL 模式，每请求独立连接（线程安全，配合 FastAPI sync 端点线程池）。
DB 文件 backend/data/medical_assistant.db。users + conversations + messages 三表：
conversations.client_id 列现存用户 id（旧 Cookie 匿名 id 数据已弃用）；
消息核心字段进列，kind 相关的异构字段（points/note/sources/...）进 payload_json，
fresh（打字机瞬时态）不落盘。删除会话级联删除其消息。
"""

import hashlib
import hmac
import json
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

# 旧：DB_PATH = Path(__file__).resolve().parent / "data" / "hr_assistant.db"（HR主题，已改医疗）
DB_PATH = Path(__file__).resolve().parents[2] / "data" / "medical_assistant.db"

# 进列的核心字段；其余（除 fresh）序列化进 payload_json
_CORE = {"id", "role", "text", "ts", "kind", "feedback"}

_INSERT_MSG = (
    "INSERT INTO messages (id, conversation_id, role, text, ts, kind, feedback, payload_json) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    salt          TEXT NOT NULL,
    created_at    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    client_id   TEXT NOT NULL,
    title       TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    text            TEXT NOT NULL,
    ts              INTEGER NOT NULL,
    kind            TEXT,
    feedback        TEXT,
    payload_json    TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, ts);
CREATE INDEX IF NOT EXISTS idx_conv_client ON conversations(client_id, updated_at);
"""


def _now_ms() -> int:
    """当前时间戳（毫秒），用于 created_at / updated_at。"""
    return int(time.time() * 1000)


@contextmanager
def _connect():
    """创建独立数据库连接：正常提交、异常回滚，退出时关闭。"""
    conn = sqlite3.connect(DB_PATH, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """建库建表（WAL 模式），幂等，可重复调用。"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(_SCHEMA)


# ---------- 用户账号（JWT 认证） ----------

def _hash_password(password: str, salt_hex: str) -> str:
    """PBKDF2-SHA256 加盐哈希密码（10 万次迭代）。"""
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 100_000
    ).hex()


def create_user(username: str, password: str):
    """注册新用户（随机盐 + 哈希存储），用户名已存在返回 None。"""
    user_id = str(uuid.uuid4())
    salt = secrets.token_hex(16)
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO users (id, username, password_hash, salt, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, username, _hash_password(password, salt), salt, _now_ms()),
            )
    except sqlite3.IntegrityError:
        return None
    return {"id": user_id, "username": username}


def get_user_by_name(username: str):
    """按用户名查用户（含密码哈希/盐，仅供登录校验）。"""
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, salt FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: str):
    """按 id 查用户公开信息（id / username），不存在返回 None。"""
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, username FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return dict(row) if row else None


def verify_password(user: dict, password: str) -> bool:
    """校验密码是否正确（常数时间比较，防时序攻击）。"""
    expected = _hash_password(password, user["salt"])
    return hmac.compare_digest(expected, user["password_hash"])


# ---------- 对话与消息 ----------

def _msg_to_row(conv_id: str, msg: dict):
    """消息 dict → messages 表插入元组，异构字段序列化进 payload_json。"""
    payload = {k: v for k, v in msg.items() if k not in _CORE and k != "fresh"}
    return (
        msg["id"], conv_id, msg["role"], msg.get("text", ""),
        int(msg.get("ts", 0)), msg.get("kind"), msg.get("feedback"),
        json.dumps(payload, ensure_ascii=False),
    )


def _row_to_msg(row: sqlite3.Row) -> dict:
    """messages 表行 → 消息 dict（还原 payload_json 中的字段）。"""
    msg = {
        "id": row["id"], "role": row["role"], "text": row["text"],
        "ts": row["ts"], "kind": row["kind"], "feedback": row["feedback"],
    }
    msg.update(json.loads(row["payload_json"] or "{}"))
    return msg


def _conv_dict(c: sqlite3.Row, msgs) -> dict:
    """会话行 + 消息行 → 前端所需的会话 dict。"""
    return {
        "id": c["id"], "title": c["title"],
        "createdAt": c["created_at"], "updatedAt": c["updated_at"],
        "messages": [_row_to_msg(m) for m in msgs],
    }


def list_conversations(client_id: str):
    """列出用户全部会话（含消息，按创建时间倒序）。"""
    with _connect() as conn:
        convs = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "WHERE client_id = ? ORDER BY created_at DESC", (client_id,)
        ).fetchall()
        result = []
        for c in convs:
            msgs = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY ts, rowid",
                (c["id"],)
            ).fetchall()
            result.append(_conv_dict(c, msgs))
        return result


def get_conversation(client_id: str, conv_id: str):
    """取用户单个会话（含消息），不存在或不属于该用户返回 None。"""
    with _connect() as conn:
        c = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "WHERE id = ? AND client_id = ?", (conv_id, client_id)
        ).fetchone()
        if not c:
            return None
        msgs = conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY ts, rowid", (conv_id,)
        ).fetchall()
        return _conv_dict(c, msgs)


def create_conversation(client_id: str, conv_id: str, title: str, created_at=None, messages=None):
    """新建会话（可同时写入撤销删除恢复的消息），返回完整会话。"""
    now = int(created_at) if created_at else _now_ms()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO conversations (id, client_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conv_id, client_id, title, now, now),
        )
        for m in (messages or []):
            conn.execute(_INSERT_MSG, _msg_to_row(conv_id, m))
    return get_conversation(client_id, conv_id)


def update_conversation(client_id: str, conv_id: str, title: str) -> bool:
    """修改会话标题并刷新 updated_at，返回是否命中。"""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND client_id = ?",
            (title, _now_ms(), conv_id, client_id),
        )
        return cur.rowcount > 0


def delete_conversation(client_id: str, conv_id: str) -> bool:
    """删除会话（级联删除其消息），返回是否命中。"""
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM conversations WHERE id = ? AND client_id = ?", (conv_id, client_id)
        )
        return cur.rowcount > 0


def append_message(client_id: str, conv_id: str, msg: dict) -> bool:
    """向会话追加消息并刷新 updated_at，会话不属于该用户返回 False。"""
    with _connect() as conn:
        owns = conn.execute(
            "SELECT 1 FROM conversations WHERE id = ? AND client_id = ?", (conv_id, client_id)
        ).fetchone()
        if not owns:
            return False
        conn.execute(_INSERT_MSG, _msg_to_row(conv_id, msg))
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (_now_ms(), conv_id)
        )
        return True


def update_message_feedback(client_id: str, msg_id: str, feedback) -> bool:
    """更新消息反馈（限本人会话内的消息），返回是否命中。"""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE messages SET feedback = ? WHERE id = ? AND conversation_id IN "
            "(SELECT id FROM conversations WHERE client_id = ?)",
            (feedback, msg_id, client_id),
        )
        return cur.rowcount > 0