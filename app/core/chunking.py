"""文本分块处理模块"""
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    将文本分割成块
    
    Args:
        text: 要分割的文本
        chunk_size: 块大小
        chunk_overlap: 块重叠大小
    
    Returns:
        文本块列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def chunk_documents(documents: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    批量处理文档
    
    Args:
        documents: 文档列表
        chunk_size: 块大小
        chunk_overlap: 块重叠大小
    
    Returns:
        文本块列表
    """
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)
    return all_chunks

