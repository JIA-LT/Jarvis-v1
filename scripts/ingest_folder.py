"""批量导入文件夹中的文档"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.chunking import chunk_text
# from app.core.embeddings import get_embeddings
# from app.vector.qdrant import init_collection, add_vector
import uuid

def ingest_folder(folder_path: str):
    """
    导入文件夹中的所有Markdown文件
    
    Args:
        folder_path: 文件夹路径
    """
    # 初始化集合
    # init_collection()
    
    folder = Path(folder_path)
    if not folder.exists():
        print(f"文件夹不存在: {folder_path}")
        return
    
    # 查找所有Markdown文件
    md_files = list(folder.rglob("*.md"))
    
    print(f"找到 {len(md_files)} 个Markdown文件")
    
    for md_file in md_files:
        print(f"处理文件: {md_file}")
        
        # 读取文件内容
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 分块
        chunks = chunk_text(content)
        
        # 统计信息
        total_chars = sum(len(chunk["content"]) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks) if chunks else 0
        
        print(f"\n{'='*80}")
        print(f"文件: {md_file.name}")
        print(f"{'='*80}")
        print(f"总块数: {len(chunks)}")
        print(f"总字符数: {total_chars}")
        print(f"平均块大小: {avg_chunk_size:.1f} 字符")
        print(f"{'='*80}\n")
        
        # 详细展示每个 chunk
        for i, chunk in enumerate(chunks, 1):
            print(f"\n【Chunk {i}/{len(chunks)}】")
            print(f"  Chunk ID: {chunk['chunk_id']}")
            print(f"  Header Path: {chunk['header_path'] if chunk['header_path'] else '(无标题)'}")
            print(f"  行号范围: {chunk['start_line']}-{chunk['end_line']}")
            print(f"  字符数: {len(chunk['content'])}")
            print(f"  内容预览:")
            # 显示内容的前200个字符
            preview = chunk['content'][:200]
            if len(chunk['content']) > 200:
                preview += "..."
            # 按行显示预览，每行缩进
            for line in preview.split('\n'):
                print(f"    {line}")
            print()
        
        print(f"{'='*80}")
        print(f"完成处理: {md_file.name}")
        print(f"{'='*80}\n")
        
        # 生成嵌入向量
        # embeddings = get_embeddings(chunks)
        
        # # 添加到向量数据库
        # for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        #     vector_id = str(uuid.uuid4())
        #     add_vector(
        #         vector_id=vector_id,
        #         vector=embedding,
        #         payload={
        #             "filename": md_file.name,
        #             "file_path": str(md_file),
        #             "chunk_index": i,
        #             "content": chunk
        #         }
        #     )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python ingest_folder.py <folder_path>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    ingest_folder(folder_path)

