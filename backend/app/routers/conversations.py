"""会话与消息 CRUD（均需登录，按用户 id 隔离）。"""

from fastapi import APIRouter, Body, Depends, HTTPException

from ..schemas.conversation import ConversationIn, ConversationPatch, MessagePatch
from ..services import conversation_service
from .deps import current_user

router = APIRouter(tags=["conversations"])


@router.get("/api/conversations")
def get_conversations(user: dict = Depends(current_user)):
    # 旧：cid: str = Depends(client_id)（Cookie 匿名身份）
    return conversation_service.list_conversations(user["id"])


@router.post("/api/conversations")
def create_conversation(body: ConversationIn, user: dict = Depends(current_user)):
    return conversation_service.create_conversation(user["id"], body)


@router.patch("/api/conversations/{conv_id}")
def patch_conversation(conv_id: str, body: ConversationPatch, user: dict = Depends(current_user)):
    if not conversation_service.update_title(user["id"], conv_id, body.title):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True}


@router.delete("/api/conversations/{conv_id}")
def remove_conversation(conv_id: str, user: dict = Depends(current_user)):
    if not conversation_service.delete_conversation(user["id"], conv_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True}


@router.post("/api/conversations/{conv_id}/messages")
def add_message(conv_id: str, message: dict = Body(...), user: dict = Depends(current_user)):
    if not message.get("role"):
        raise HTTPException(status_code=400, detail="message 需含 role")
    saved = conversation_service.append_message(user["id"], conv_id, message)
    if saved is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return saved


@router.patch("/api/messages/{msg_id}")
def patch_message(msg_id: str, body: MessagePatch, user: dict = Depends(current_user)):
    if not conversation_service.update_feedback(user["id"], msg_id, body.feedback):
        raise HTTPException(status_code=404, detail="消息不存在")
    return {"ok": True}