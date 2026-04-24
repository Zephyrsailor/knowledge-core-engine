"""
快速开始示例 - 5行代码完成RAG
"""

import asyncio
import json

from knowledge_core_engine import KnowledgeEngine


async def main():
    # 创建知识引擎（已开启中文场景优化配置）
    engine = KnowledgeEngine(
        enable_metadata_enhancement=True,   # LLM 生成 summary / questions / keywords
        language="zh",                      # jieba 中文分词（BM25 必需）
        vector_weight=0.5,                  # BM25 和向量平权，对专名更友好
        bm25_weight=0.5,
        enable_reranking=True,              # cross-encoder 精排
        reranker_provider="api",
        reranker_api_provider="dashscope",
        reranker_model="gte-rerank",
    )
    
    # 添加文档（可以是文件、目录或列表）
    await engine.add_v2("data/source_docs/lazada.pdf")  # 使用您自己的文档目录

    # 列出文档
    # answer = await engine.document_detail(kb_id='',file_id='qcnew_1.pdf')
    
    # 提问
    answer = await engine.search("lazada该如何入驻？",retrieval_only=True,top_k=5)
    
    # 打印答案
    for text in answer:
        if text['metadata']['chunk_type'] == 'image':
            print(json.dumps(text, indent=4))
            print(text)
            print("----------------------------------------------------------------->")
    #     print("----------------------------------------------------------------->")
    print(answer)
    json_str = json.dumps(answer)
    print(json_str)

    # 修改image类chunk的content
    # result = await engine.update_image_chunk_content("ceacf157b5c857b1431b754b360e05dc","仅仅是测试下")
    # print(result)


if __name__ == "__main__":
    # 确保设置了环境变量
    # export DEEPSEEK_API_KEY=your_key
    # export DASHSCOPE_API_KEY=your_key
    
    asyncio.run(main())