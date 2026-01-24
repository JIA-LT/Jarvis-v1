"""检查 API Key 是否被正确读取"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from app.config import settings

print("=" * 80)
print("API Key 检查")
print("=" * 80)

# 1. 检查环境变量
env_key = os.getenv("OPENAI_API_KEY")
print(f"\n1. 环境变量 OPENAI_API_KEY:")
if env_key:
    # 只显示前8个字符和后4个字符，中间用*代替
    masked = f"{env_key[:8]}...{env_key[-4:]}" if len(env_key) > 12 else "***"
    print(f"   ✅ 已设置: {masked}")
    print(f"   长度: {len(env_key)} 字符")
else:
    print(f"   ❌ 未设置")

# 2. 检查 .env 文件
env_file = Path(__file__).parent.parent / ".env"
print(f"\n2. .env 文件:")
if env_file.exists():
    print(f"   ✅ 存在: {env_file}")
    # 读取 .env 文件内容（不显示完整 key）
    with open(env_file, "r") as f:
        content = f.read()
        if "OPENAI_API_KEY" in content:
            print(f"   ✅ 包含 OPENAI_API_KEY")
            # 检查是否有值
            for line in content.split("\n"):
                if line.startswith("OPENAI_API_KEY"):
                    if "=" in line and len(line.split("=", 1)[1].strip()) > 0:
                        key_part = line.split("=", 1)[1].strip()
                        masked = f"{key_part[:8]}...{key_part[-4:]}" if len(key_part) > 12 else "***"
                        print(f"   值: {masked}")
                    else:
                        print(f"   ⚠️  值为空")
        else:
            print(f"   ❌ 不包含 OPENAI_API_KEY")
else:
    print(f"   ❌ 不存在: {env_file}")

# 3. 检查 settings 中的值
print(f"\n3. Settings 配置:")
if settings.OPENAI_API_KEY:
    masked = f"{settings.OPENAI_API_KEY[:8]}...{settings.OPENAI_API_KEY[-4:]}" if len(settings.OPENAI_API_KEY) > 12 else "***"
    print(f"   ✅ 已读取: {masked}")
    print(f"   长度: {len(settings.OPENAI_API_KEY)} 字符")
    print(f"   模型: {settings.EMBEDDING_MODEL}")
else:
    print(f"   ❌ 未读取到 API Key")
    print(f"\n   可能的原因:")
    print(f"   - 环境变量未设置或未 export")
    print(f"   - .env 文件不存在或格式错误")
    print(f"   - 需要重启终端或重新 source ~/.zshrc")

# 4. 测试 API Key 格式
print(f"\n4. API Key 格式检查:")
if settings.OPENAI_API_KEY:
    key = settings.OPENAI_API_KEY
    if key.startswith("sk-"):
        print(f"   ✅ 格式正确（以 sk- 开头）")
    else:
        print(f"   ⚠️  格式异常（通常以 sk- 开头）")
    if len(key) >= 20:
        print(f"   ✅ 长度合理（{len(key)} 字符）")
    else:
        print(f"   ⚠️  长度可能过短（{len(key)} 字符）")

print("\n" + "=" * 80)
