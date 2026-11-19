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