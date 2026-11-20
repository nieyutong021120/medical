"""会话业务规则：持久化 id 统一生成、撤销删除时重建消息 id。SQL 在 db/database.py。"""

import uuid

from ..db import database as db
from ..schemas.conversation import ConversationIn


def uid() -> str:
    """持久化 id（会话/消息）统一由后端生成，前端不再自带 id。"""
    return str(uuid.uuid4())


def list_conversations(user_id: str):
    return db.list_conversations(user_id)


def create_conversation(user_id: str, body: ConversationIn):
    conv_id = uid()
    # 撤销删除会带回旧消息，其 id 同样由后端重新生成
    messages = [{**m, "id": uid()} for m in body.messages] if body.messages else None
    return db.create_conversation(user_id, conv_id, body.title, body.createdAt, messages)


def update_title(user_id: str, conv_id: str, title: str) -> bool:
    return db.update_conversation(user_id, conv_id, title)


def delete_conversation(user_id: str, conv_id: str) -> bool:
    return db.delete_conversation(user_id, conv_id)


def append_message(user_id: str, conv_id: str, message: dict):
    """落库成功返回带真实 id 的消息；会话不属于该用户返回 None。"""
    message["id"] = uid()
    if not db.append_message(user_id, conv_id, message):
        return None
    return message


def update_feedback(user_id: str, msg_id: str, feedback) -> bool:
    return db.update_message_feedback(user_id, msg_id, feedback)


def build_history(user_id: str, conv_id: str, question: str,
                  max_turns: int = 4, max_chars: int = 1500) -> list:
    """取会话近期历史供多轮问答：最近 max_turns 轮、总长 ≤ max_chars，从新到旧整轮装桶。"""
    conv = db.get_conversation(user_id, conv_id)
    if not conv:
        return []
    msgs = [
        m for m in conv["messages"]
        if m["role"] in ("user", "assistant")
        and m.get("kind") != "error" and (m.get("text") or "").strip()
    ]
    # 当前问题可能已由前端先行落库，剔除末尾重复，避免混入"历史"
    if msgs and msgs[-1]["role"] == "user" and msgs[-1]["text"].strip() == question.strip():
        msgs.pop()
    turns = []
    for m in msgs:
        text = m["text"].strip()
        if m["role"] == "user":
            turns.append({"user": text, "assistant": ""})
        elif turns:
            turns[-1]["assistant"] = text
    kept, total = [], 0
    for t in reversed(turns[-max_turns:]):
        cost = len(t["user"]) + len(t["assistant"])
        if total + cost > max_chars:
            # 最新一轮本身超预算：user 完整保留，assistant 只留开头
            if not kept:
                a = t["assistant"]
                if len(a) > 600:
                    a = a[:600] + "…(回答已截断)"
                kept.append({"user": t["user"], "assistant": a})
            break
        kept.append(t)
        total += cost
    kept.reverse()
    history = []
    for t in kept:
        history.append({"role": "user", "text": t["user"]})
        if t["assistant"]:
            history.append({"role": "assistant", "text": t["assistant"]})
    return history