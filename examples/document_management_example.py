"""文档管理功能示例

本示例展示如何使用 KnowledgeEngine 的文档管理功能：
- 添加文档
- 列出文档
- 删除文档
- 更新文档
"""

import asyncio
from pathlib import Path
from knowledge_core_engine import KnowledgeEngine, KnowledgeConfig


async def main():
    # 创建引擎配置
    config = KnowledgeConfig(
        llm_provider="deepseek",
        embedding_provider="dashscope",
        enable_metadata_enhancement=True,
        enable_hierarchical_chunking=True,
        retrieval_strategy="hybrid"
    )
    
    # 初始化引擎
    engine = KnowledgeEngine(config)
    
    print("=" * 50)
    print("Knowledge Engine 文档管理示例")
    print("=" * 50)
    
    # 1. 添加一些测试文档
    print("\n1. 添加文档到知识库...")
    test_docs_dir = Path("data/source_docs")
    if test_docs_dir.exists():
        pdf_files = list(test_docs_dir.glob("*.pdf"))
        if pdf_files:
            # 添加第一个PDF文件
            result = await engine.add([pdf_files[0]])
            print(f"添加了 {result['processed']} 个文档")
    
    # 2. 列出所有文档
    print("\n2. 列出知识库中的所有文档...")
    doc_list = await engine.list()
    
    print(f"总文档数: {doc_list['total']}")
    print(f"当前页: {doc_list['page']}/{doc_list['pages']}")
    print("\n文档列表:")
    for doc in doc_list['documents']:
        print(f"  - 名称: {doc['name']}")
        print(f"    路径: {doc['path']}")
        print(f"    分块数: {doc.get('chunks_count', 'N/A')}")
        print(f"    总大小: {doc.get('total_size', 'N/A')} 字符")
        print()
    
    # 3. 按文件类型过滤
    print("\n3. 过滤PDF文档...")
    pdf_list = await engine.list(filter={"file_type": "pdf"})
    print(f"PDF文档数: {pdf_list['total']}")
    
    # 4. 分页示例
    print("\n4. 分页获取文档...")
    page_size = 5
    first_page = await engine.list(page=1, page_size=page_size)
    print(f"第一页文档数: {len(first_page['documents'])}")
    
    # 5. 不返回统计信息
    print("\n5. 获取文档列表（不含统计信息）...")
    simple_list = await engine.list(return_stats=False)
    if simple_list['documents']:
        doc = simple_list['documents'][0]
        print(f"文档名: {doc['name']}")
        print(f"包含统计信息: {'chunks_count' in doc}")
    
    # 6. 删除文档示例
    if doc_list['documents']:
        print("\n6. 删除文档示例...")
        doc_to_delete = doc_list['documents'][0]['name']
        print(f"准备删除文档: {doc_to_delete}")
        
        # 删除前的文档数
        before = await engine.list()
        print(f"删除前文档数: {before['total']}")
        
        # 执行删除
        delete_result = await engine.delete(source=doc_to_delete)
        print(f"删除了 {delete_result['deleted_count']} 个块")
        
        # 删除后的文档数
        after = await engine.list()
        print(f"删除后文档数: {after['total']}")
    
    # 7. 关于原始文件管理的说明
    print("\n7. 原始文件管理策略说明:")
    print("  - 知识引擎只管理向量索引，不管理原始文件")
    print("  - 原始文件保留在原位置，由用户自行管理")
    print("  - 删除文档只删除向量索引，不删除原始文件")
    print("  - 如需删除原始文件，请手动删除文件系统中的文件")
    
    # 关闭引擎
    await engine.close()


if __name__ == "__main__":
    asyncio.run(main())