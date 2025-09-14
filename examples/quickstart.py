"""
快速开始示例 - 5行代码完成RAG
"""

import asyncio
import json

from knowledge_core_engine import KnowledgeEngine


async def main():
    # 创建知识引擎
    engine = KnowledgeEngine()
    
    # 添加文档（可以是文件、目录或列表）
    # await engine.add_v2("data/source_docs/lazada入驻解疑.pdf")  # 使用您自己的文档目录

    # 列出文档
    answer = await engine.document_detail(kb_id='',file_id='lazada入驻解疑.pdf')
    
    # 提问
    # answer = await engine.ask("玉湖翠瓶是什么样的？",retrieval_only=True,top_k=2)
    
    # 打印答案
    for text in answer:
        print(text)
        print("----------------------------------------------------------------->")
        print("----------------------------------------------------------------->")
    print(answer)
    # json_str = json.dumps(answer)
    # print(json_str)


if __name__ == "__main__":
    # 确保设置了环境变量
    # export DEEPSEEK_API_KEY=your_key
    # export DASHSCOPE_API_KEY=your_key
    
    asyncio.run(main())