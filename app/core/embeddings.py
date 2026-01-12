"""嵌入向量生成模块"""
from typing import List
import openai
from app.config import settings

def get_embedding(text: str) -> List[float]:
    """
    获取文本的嵌入向量
    
    Args:
        text: 输入文本
    
    Returns:
        嵌入向量
    """
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not configured")
    
    response = openai.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    批量获取嵌入向量
    
    Args:
        texts: 文本列表
    
    Returns:
        嵌入向量列表
    """
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not configured")
    
    response = openai.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]

