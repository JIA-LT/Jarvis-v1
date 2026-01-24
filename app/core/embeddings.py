"""嵌入向量生成模块"""
from typing import List
import openai
from app.config import settings

class EmbeddingError(Exception):
    """嵌入向量生成错误"""
    pass

def get_embedding(text: str) -> List[float]:
    """
    获取文本的嵌入向量
    
    Args:
        text: 输入文本
    
    Returns:
        嵌入向量
    
    Raises:
        EmbeddingError: 当 API 调用失败时
    """
    if not settings.OPENAI_API_KEY:
        raise EmbeddingError("OPENAI_API_KEY not configured")
    # 调试：只显示部分字符，避免泄露完整 key
    key_preview = f"{settings.OPENAI_API_KEY[:8]}...{settings.OPENAI_API_KEY[-4:]}" if settings.OPENAI_API_KEY else "None"
    print(f"[DEBUG] OPENAI_API_KEY 已加载: {key_preview} (长度: {len(settings.OPENAI_API_KEY)})")
    try:
        response = openai.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except openai.RateLimitError as e:
        raise EmbeddingError(
            "API 配额超限。请检查：\n"
            "1. OpenAI 账号是否有可用额度\n"
            "2. 是否已添加支付方式（部分模型需要）\n"
            "3. 账号是否在审核中\n"
            f"错误详情: {e}"
        ) from e
    except openai.APIError as e:
        raise EmbeddingError(f"OpenAI API 错误: {e}") from e

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    批量获取嵌入向量
    
    Args:
        texts: 文本列表
    
    Returns:
        嵌入向量列表
    
    Raises:
        EmbeddingError: 当 API 调用失败时
    """
    if not settings.OPENAI_API_KEY:
        raise EmbeddingError("OPENAI_API_KEY not configured")
    # 调试：只显示部分字符，避免泄露完整 key
    key_preview = f"{settings.OPENAI_API_KEY[:8]}...{settings.OPENAI_API_KEY[-4:]}" if settings.OPENAI_API_KEY else "None"
    print(f"[DEBUG] OPENAI_API_KEY 已加载: {key_preview} (长度: {len(settings.OPENAI_API_KEY)})")
    try:
        response = openai.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=texts
        )
        return [item.embedding for item in response.data]
    except openai.RateLimitError as e:
        raise EmbeddingError(
            "API 配额超限。请检查：\n"
            "1. OpenAI 账号是否有可用额度\n"
            "2. 是否已添加支付方式（部分模型需要）\n"
            "3. 账号是否在审核中\n"
            f"错误详情: {e}"
        ) from e
    except openai.APIError as e:
        raise EmbeddingError(f"OpenAI API 错误: {e}") from e

