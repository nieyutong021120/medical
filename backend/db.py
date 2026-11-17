"""SQLite 持久化层：对话历史（按 client_id 隔离）。

标准库 sqlite3，WAL 模式，每请求独立连接（线程安全，配合 FastAPI sync 端点线程池）。
DB 文件 backend/data/medical_assistant.db。conversations + messages 两表：
消息核心字段进列，kind 相关的异构字段（points/note/sources/...）进 payload_json，
fresh（打字机瞬时态）不落盘。删除会话级联删除其消息。
"""

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

# 旧：DB_PATH = Path(__file__).resolve().parent / "data" / "hr_assistant.db"（HR主题，已改医疗）
DB_PATH = Path(__file__).resolve().parent / "data" / "medical_assistant.db"

# 进列的核心字段；其余（除 fresh）序列化进 payload_json
_CORE = {"id", "role", "text", "ts", "kind", "feedback"}

_INSERT_MSG = (
    "INSERT INTO messages (id, conversation_id, role, text, ts, kind, feedback, payload_json) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
)

_SCHEMA = """
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
    return int(time.time() * 1000)


@contextmanager
def _connect():
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
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(_SCHEMA)


def _msg_to_row(conv_id: str, msg: dict):
    payload = {k: v for k, v in msg.items() if k not in _CORE and k != "fresh"}
    return (
        msg["id"], conv_id, msg["role"], msg.get("text", ""),
        int(msg.get("ts", 0)), msg.get("kind"), msg.get("feedback"),
        json.dumps(payload, ensure_ascii=False),
    )


def _row_to_msg(row: sqlite3.Row) -> dict:
    msg = {
        "id": row["id"], "role": row["role"], "text": row["text"],
        "ts": row["ts"], "kind": row["kind"], "feedback": row["feedback"],
    }
    msg.update(json.loads(row["payload_json"] or "{}"))
    return msg


def _conv_dict(c: sqlite3.Row, msgs) -> dict:
    return {
        "id": c["id"], "title": c["title"],
        "createdAt": c["created_at"], "updatedAt": c["updated_at"],
        "messages": [_row_to_msg(m) for m in msgs],
    }


def list_conversations(client_id: str):
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
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND client_id = ?",
            (title, _now_ms(), conv_id, client_id),
        )
        return cur.rowcount > 0


def delete_conversation(client_id: str, conv_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM conversations WHERE id = ? AND client_id = ?", (conv_id, client_id)
        )
        return cur.rowcount > 0


def append_message(client_id: str, conv_id: str, msg: dict) -> bool:
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
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE messages SET feedback = ? WHERE id = ? AND conversation_id IN "
            "(SELECT id FROM conversations WHERE client_id = ?)",
            (feedback, msg_id, client_id),
        )
        return cur.rowcount > 0