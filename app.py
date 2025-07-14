from knowledge_core_engine import KnowledgeEngine
import asyncio

async def main():
    # 创建引擎（现在会自动加载.env文件中的配置）
    engine = KnowledgeEngine()
    
    try:
        # 添加文档
        result = await engine.add("data/source_docs/")
        print(f"文档处理结果: {result}")
        
        if result['processed_files'] > 0:
            # 简单使用 - 直接获取答案（包含引用）
            print("\n简单查询：")
            answer = await engine.ask("什么是知行合一？")
            print(f"答案: {answer}")
            
            # 详细使用 - 获取完整信息
            print("\n\n详细查询：")
            detailed = await engine.ask("RAG技术有什么特点？", return_details=True)
            print(f"问题: {detailed['question']}")
            print(f"答案: {detailed['answer']}")
            print(f"\n引用来源:")
            for cite in detailed['citations']:
                print(f"  [{cite['index']}] {cite['source']}")
            print(f"\n找到 {len(detailed['contexts'])} 个相关上下文")
        else:
            print("\n❌ 没有成功处理任何文档")
            if result['failed_files']:
                print("失败的文件:")
                for fail in result['failed_files']:
                    print(f"  - {fail['file']}: {fail['error']}")
    
    except Exception as e:
        print(f"❌ 发生错误: {e}")
    
    finally:
        await engine.close()

asyncio.run(main())