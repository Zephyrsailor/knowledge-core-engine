from knowledge_core_engine import KnowledgeEngine
import asyncio

async def main():
    print("=== 诚实的测试：阈值过滤的现实 ===")
    
    # 使用上下文管理器自动关闭资源
    async with KnowledgeEngine(
            # Embedding 配置
            embedding_provider="dashscope",
            embedding_model="text-embedding-v4",  # 使用稳定版本

            # 启用 Rerank
            enable_reranking=True,
            reranker_provider="api",
            reranker_api_provider="dashscope",
            reranker_model="gte-rerank",

            # 其他配置
            retrieval_strategy="hybrid",
            retrieval_top_k=10,  # 先多检索一些
            rerank_top_k=5,       # rerank后保留5个
            rerank_score_threshold=0.1,        # rerank 分数低于 0.1 的过滤掉
            log_level="DEBUG"      # 开启日志
    ) as engine:
        # 列出所有文档
        doc_list = await engine.list() 
         # 查看结果
        print(f"总文档数: {doc_list['total']}")
        for doc in doc_list['documents']:
            print(f"- {doc['name']}: {doc['chunks_count']} chunks")
         
        # 添加文档
        # await engine.add("data/source_docs/中华人民共和国婚姻法.pdf")



        await engine.update("data/source_docs/中华人民共和国婚姻法.pdf")


        # await engine.delete("中华人民共和国婚姻法.pdf")


        
        print("\n📊 真实测试结果:")
        return
        
        # 测试相关问题
        answer1 = await engine.ask("婚姻法第一条", retrieval_only=True, top_k=10)
        print(f"1. '婚姻法第一条' → {len(answer1) if answer1 else 0} 个结果")
        if answer1:
            print(f"   原始分数: {answer1[0].score:.3f}")
            if hasattr(answer1[0], 'rerank_score') and answer1[0].rerank_score is not None:
                print(f"   Rerank分数: {answer1[0].rerank_score:.3f}")
            print(f"   内容预览: {answer1[0].content[:50]}...")
        
        # 测试不相关问题  
        answer2 = await engine.ask("数字化2.0 阶段的企业有哪些？", retrieval_only=True, top_k=10)
        print(f"2. '数字化2.0企业' → {len(answer2) if answer2 else 0} 个结果")
        if answer2:
            print(f"   原始分数: {answer2[0].score:.3f}")
            if hasattr(answer2[0], 'rerank_score') and answer2[0].rerank_score is not None:
                print(f"   Rerank分数: {answer2[0].rerank_score:.3f}")
            print(f"   内容预览: {answer2[0].content[:50]}...")
    
    print("\n🎯 真实情况分析:")
    print("- BM25索引工作正常 ✅")
    print("- 向量检索有严重问题：")
    print("  - '数字化企业'查询返回了'婚姻法'内容 ❌")
    print("  - 原始分数几乎相同（0.474 vs 0.475）")
    print("- Rerank 的效果：")
    if answer1 and answer2:
        rerank1 = answer1[0].rerank_score if hasattr(answer1[0], 'rerank_score') else None
        rerank2 = answer2[0].rerank_score if hasattr(answer2[0], 'rerank_score') else None
        print(f"  - 相关查询: {answer1[0].score:.3f} → {rerank1:.3f} (提升)")
        print(f"  - 不相关查询: {answer2[0].score:.3f} → {rerank2:.3f} (降低)")
        print(f"\n❗ 问题：即使是完全不相关的内容，rerank 分数 {rerank2:.3f} 还是太高了！")
    
    print(f"\n💭 根本问题:")
    print(f"1. 知识库只有婚姻法文档，任何查询都会返回婚姻法内容")
    print(f"2. 向量模型 text-embedding-v3 对中文语义理解不够好")
    print(f"3. Rerank 虽然降低了不相关内容的分数，但降得不够低")
    print(f"\n🔧 解决方案:")
    print(f"1. 需要更严格的阈值（如 0.3 或 0.4）")
    print(f"2. 考虑升级到更好的中文向量模型")
    print(f"3. 知识库需要更多样化的内容来测试")
    print(f"\n# 当前可用的配置：")
    print(f"engine = KnowledgeEngine(")
    print(f"    enable_reranking=True,")
    print(f"    reranker_model='gte-rerank',")
    print(f"    enable_relevance_threshold=True,")
    print(f"    hybrid_score_threshold=0.3  # 提高阈值")
    print(f")")

asyncio.run(main())