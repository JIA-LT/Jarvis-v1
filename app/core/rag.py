"""RAG (检索增强生成) 模块"""
from typing import List, Dict
from app.core.llm import generate_response
from app.vector.qdrant import search_similar

def retrieve_context(query: str, top_k: int = 5) -> List[Dict]:
    """
    检索相关上下文
    
    Args:
        query: 查询文本
        top_k: 返回前k个结果
    
    Returns:
        相关文档列表
    """
    results = search_similar(query, top_k=top_k)
    return results

def rag_query(query: str, top_k: int = 5) -> str:
    """
    RAG查询：检索相关上下文并生成响应
    
    Args:
        query: 用户查询
        top_k: 检索的文档数量
    
    Returns:
        LLM生成的响应
    """
    # 检索相关上下文
    context_docs = retrieve_context(query, top_k)
    
    # 构建上下文
    context = "\n\n".join([doc.get("content", "") for doc in context_docs])
    
    # 构建提示词
    prompt = f"""基于以下上下文回答用户的问题。如果上下文中没有相关信息，请说明你不知道。

上下文：
{context}

问题：{query}

回答："""
    
    # 生成响应
    response = generate_response(prompt)
    return response

