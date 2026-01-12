"""LLM (大语言模型) 模块"""
import openai
from app.config import settings

def generate_response(prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
    """
    使用LLM生成响应
    
    Args:
        prompt: 提示词
        temperature: 温度参数
        max_tokens: 最大token数
    
    Returns:
        LLM生成的文本
    """
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not configured")
    
    openai.api_key = settings.OPENAI_API_KEY
    
    response = openai.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return response.choices[0].message.content

def generate_stream(prompt: str, temperature: float = 0.7, max_tokens: int = 1000):
    """
    流式生成响应
    
    Args:
        prompt: 提示词
        temperature: 温度参数
        max_tokens: 最大token数
    
    Yields:
        LLM生成的文本片段
    """
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not configured")
    
    openai.api_key = settings.OPENAI_API_KEY
    
    response = openai.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

