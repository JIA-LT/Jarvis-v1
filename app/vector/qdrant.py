"""Qdrant向量数据库操作模块"""
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.config import settings
from app.core.embeddings import get_embedding

# 初始化Qdrant客户端
client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

def init_collection():
    """初始化集合（如果不存在）"""
    try:
        # 检查集合是否存在
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if settings.QDRANT_COLLECTION_NAME not in collection_names:
            # 创建新集合 (假设使用1536维的嵌入向量，text-embedding-ada-002)
            client.create_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
    except Exception as e:
        print(f"Error initializing collection: {e}")

def add_vector(vector_id: str, vector: List[float], payload: Dict):
    """
    添加向量到集合
    
    Args:
        vector_id: 向量ID
        vector: 向量数据
        payload: 元数据
    """
    client.upsert(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        points=[
            PointStruct(
                id=vector_id,
                vector=vector,
                payload=payload
            )
        ]
    )

def search_similar(query: str, top_k: int = 5) -> List[Dict]:
    """
    搜索相似向量
    
    Args:
        query: 查询文本
        top_k: 返回前k个结果
    
    Returns:
        相似文档列表
    """
    # 获取查询向量
    query_vector = get_embedding(query)
    
    # 搜索
    results = client.search(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    
    # 格式化结果
    formatted_results = []
    for result in results:
        formatted_results.append({
            "id": result.id,
            "score": result.score,
            "content": result.payload.get("content", ""),
            "metadata": result.payload
        })
    
    return formatted_results

def delete_vector(vector_id: str):
    """删除向量"""
    client.delete(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        points_selector=[vector_id]
    )

