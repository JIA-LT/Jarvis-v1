"""聊天路由"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

class ChatMessage(BaseModel):
    """聊天消息模型"""
    role: str  # user, assistant, system
    content: str

class ChatRequest(BaseModel):
    """聊天请求模型"""
    messages: List[ChatMessage]
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    """聊天响应模型"""
    message: str
    sources: Optional[List[str]] = None

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """处理聊天请求"""
    # TODO: 实现RAG聊天逻辑
    return ChatResponse(
        message="这是一个示例响应",
        sources=[]
    )

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天响应"""
    # TODO: 实现流式响应逻辑
    yield {"message": "streaming", "content": "示例内容"}

