"""演示聊天功能"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.rag import rag_query

def main():
    """主函数"""
    print("Jarvis 聊天演示")
    print("输入 'quit' 或 'exit' 退出\n")
    
    while True:
        query = input("你: ").strip()
        
        if query.lower() in ["quit", "exit", "退出"]:
            print("再见！")
            break
        
        if not query:
            continue
        
        try:
            response = rag_query(query)
            print(f"\nJarvis: {response}\n")
        except Exception as e:
            print(f"\n错误: {e}\n")

if __name__ == "__main__":
    main()

