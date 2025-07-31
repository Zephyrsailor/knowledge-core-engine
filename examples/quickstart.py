"""
快速开始示例 - 5行代码完成RAG
"""

import asyncio
from knowledge_core_engine import KnowledgeEngine


async def main():
    # 创建知识引擎
    engine = KnowledgeEngine()
    
    # 添加文档（可以是文件、目录或列表）
    # await engine.add("data/source_docs/")  # 使用您自己的文档目录
    
    # 提问
    answer = await engine.ask("是官方网站么？",retrieval_only=True)
    
    # 打印答案
    # for text in answer:
    #     print(text)
    print(answer)


if __name__ == "__main__":
    # 确保设置了环境变量
    # export DEEPSEEK_API_KEY=your_key
    # export DASHSCOPE_API_KEY=your_key
    
    asyncio.run(main())