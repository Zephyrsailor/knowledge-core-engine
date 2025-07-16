"""
KnowledgeCore Engine - 简洁的高级封装

设计理念：
1. 一行代码初始化
2. 三行代码完成RAG流程
3. 隐藏所有复杂性
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import os

from .core.config import RAGConfig
from .core.parsing.document_processor import DocumentProcessor
from .core.chunking.pipeline import ChunkingPipeline
from .core.chunking.enhanced_chunker import EnhancedChunker
from .core.chunking.smart_chunker import SmartChunker
from .core.enhancement.metadata_enhancer import MetadataEnhancer, EnhancementConfig
from .core.embedding.embedder import TextEmbedder
from .core.embedding.vector_store import VectorStore, VectorDocument
from .core.retrieval.retriever import Retriever
from .core.retrieval.reranker_wrapper import Reranker
from .core.generation.generator import Generator
from .utils.metadata_cleaner import clean_metadata
from .utils.logger import get_logger, log_process, log_step, log_detailed

logger = get_logger(__name__)


class KnowledgeEngine:
    """知识引擎的统一入口。
    
    使用示例：
        # 最简单的使用方式
        engine = KnowledgeEngine()
        
        # 添加文档
        await engine.add("docs/file.pdf")
        
        # 提问
        answer = await engine.ask("什么是RAG?")
        print(answer)
    """
    
    def __init__(
        self,
        llm_provider: Optional[str] = None,  # Will use default from RAGConfig
        embedding_provider: str = "dashscope", 
        persist_directory: str = "./data/knowledge_base",
        log_level: Optional[str] = None,
        **kwargs
    ):
        """初始化知识引擎。
        
        Args:
            llm_provider: LLM提供商 (deepseek/qwen/openai)
            embedding_provider: 嵌入模型提供商 (dashscope/openai)
            persist_directory: 知识库存储路径
            log_level: 日志级别 (DEBUG/INFO/WARNING/ERROR)，默认使用环境变量或INFO
            **kwargs: 其他配置参数
        """
        # 设置日志级别
        if log_level:
            from .utils.logger import setup_logger
            setup_logger("knowledge_core_engine", log_level=log_level)
            logger.info(f"Log level set to {log_level}")
        # 自动从环境变量读取API密钥
        # 创建配置，如果llm_provider为None，RAGConfig将使用默认值
        config_args = {}
        if llm_provider is not None:
            config_args['llm_provider'] = llm_provider
        if kwargs.get('llm_api_key'):
            config_args['llm_api_key'] = kwargs.get('llm_api_key')
        
        self.config = RAGConfig(
            **config_args,
            embedding_provider=embedding_provider,
            embedding_api_key=kwargs.get('embedding_api_key') or os.getenv(
                "DASHSCOPE_API_KEY" if embedding_provider == "dashscope" 
                else f"{embedding_provider.upper()}_API_KEY"
            ),
            vectordb_provider="chromadb",
            persist_directory=persist_directory,
            include_citations=kwargs.get('include_citations', True),
            # 传递所有其他参数到RAGConfig
            enable_query_expansion=kwargs.get('enable_query_expansion', False),
            query_expansion_method=kwargs.get('query_expansion_method', 'llm'),
            query_expansion_count=kwargs.get('query_expansion_count', 3),
            retrieval_strategy=kwargs.get('retrieval_strategy', 'hybrid'),
            retrieval_top_k=kwargs.get('retrieval_top_k', 10),
            vector_weight=kwargs.get('vector_weight', 0.7),
            bm25_weight=kwargs.get('bm25_weight', 0.3),
            enable_reranking=kwargs.get('enable_reranking', False),
            reranker_provider=kwargs.get('reranker_provider', 'huggingface'),
            reranker_model=kwargs.get('reranker_model', None),
            reranker_api_provider=kwargs.get('reranker_api_provider', None),
            reranker_api_key=kwargs.get('reranker_api_key', None),
            rerank_top_k=kwargs.get('rerank_top_k', 5),
            use_fp16=kwargs.get('use_fp16', True),
            reranker_device=kwargs.get('reranker_device', None),
            enable_hierarchical_chunking=kwargs.get('enable_hierarchical_chunking', False),
            enable_semantic_chunking=kwargs.get('enable_semantic_chunking', True),
            enable_metadata_enhancement=kwargs.get('enable_metadata_enhancement', False),
            chunk_size=kwargs.get('chunk_size', 512),
            chunk_overlap=kwargs.get('chunk_overlap', 50),
            extra_params=kwargs.get('extra_params', {})
        )
        
        # 内部组件（延迟初始化）
        self._initialized = False
        self._parser = None
        self._chunker = None
        self._metadata_enhancer = None
        self._embedder = None
        self._vector_store = None
        self._retriever = None
        self._reranker = None
        self._generator = None
    
    async def _ensure_initialized(self):
        """确保所有组件已初始化。"""
        if self._initialized:
            return
            
        # 创建所有组件
        self._parser = DocumentProcessor()
        
        # 根据配置选择合适的分块器
        if self.config.enable_hierarchical_chunking:
            # 使用增强分块器，支持层级关系
            chunker = EnhancedChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        elif self.config.enable_semantic_chunking:
            # 使用智能分块器
            chunker = SmartChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        else:
            # 使用默认分块器
            chunker = None
        
        self._chunker = ChunkingPipeline(
            chunker=chunker,
            enable_smart_chunking=self.config.enable_semantic_chunking
        )
        
        # 如果启用元数据增强，创建增强器
        if self.config.enable_metadata_enhancement:
            enhancement_config = EnhancementConfig(
                llm_provider=self.config.llm_provider,
                model_name=self.config.llm_model,
                api_key=self.config.llm_api_key,
                temperature=0.1,
                max_tokens=500
            )
            self._metadata_enhancer = MetadataEnhancer(enhancement_config)
        
        self._embedder = TextEmbedder(self.config)
        self._vector_store = VectorStore(self.config)
        self._retriever = Retriever(self.config)
        
        # 如果启用重排序，创建重排器
        if self.config.enable_reranking:
            self._reranker = Reranker(self.config)
        
        self._generator = Generator(self.config)
        
        # 初始化异步组件
        await self._embedder.initialize()
        await self._vector_store.initialize()
        await self._retriever.initialize()
        if self._reranker:
            await self._reranker.initialize()
        await self._generator.initialize()
        
        self._initialized = True
    
    @log_step("Add Documents to Knowledge Base")
    async def add(
        self, 
        source: Union[str, Path, List[Union[str, Path]]]
    ) -> Dict[str, Any]:
        """添加文档到知识库。
        
        Args:
            source: 文档路径，可以是单个文件、目录或文件列表
            
        Returns:
            处理结果统计
            
        Example:
            # 添加单个文件
            await engine.add("doc.pdf")
            
            # 添加整个目录
            await engine.add("docs/")
            
            # 添加多个文件
            await engine.add(["doc1.pdf", "doc2.md"])
        """
        await self._ensure_initialized()
        
        # 统一处理输入
        if isinstance(source, (str, Path)):
            source = Path(source)
            if source.is_dir():
                files = list(source.glob("**/*"))
                files = [f for f in files if f.suffix in ['.pdf', '.docx', '.md', '.txt']]
            else:
                files = [source]
        else:
            files = [Path(f) for f in source]
        
        # 处理统计
        total_files = len(files)
        total_chunks = 0
        failed_files = []
        
        log_detailed(f"Processing {total_files} files", 
                    data={"files": [str(f) for f in files]})
        
        for file_path in files:
            try:
                # 首先检查文档是否已存在于知识库中
                doc_check_id = f"{file_path.stem}_0_0"  # 使用第一个chunk的ID作为检查标识
                existing_doc = await self._vector_store.get_document(doc_check_id)
                
                if existing_doc:
                    logger.info(f"Document {file_path.name} already exists in knowledge base, skipping")
                    # 统计现有chunks数量
                    chunk_count = 0
                    while True:
                        check_id = f"{file_path.stem}_{chunk_count}_0"
                        if not await self._vector_store.get_document(check_id):
                            break
                        chunk_count += 1
                    total_chunks += chunk_count
                    continue
                
                with log_process(f"Processing {file_path.name}", 
                               file_type=file_path.suffix,
                               file_size=file_path.stat().st_size):
                    
                    # 解析文档
                    with log_process("Document Parsing"):
                        parse_result = await self._parser.process(file_path)
                        # 展示解析结果的实际内容
                        preview = parse_result.content[:300].replace('\n', ' ')
                        if len(parse_result.content) > 300:
                            preview += "..."
                        log_detailed(f"Parse result", 
                                   data={
                                       "method": parse_result.metadata.get('parse_method', 'unknown'),
                                       "length": len(parse_result.content),
                                       "preview": preview
                                   })
                    
                    # 分块
                    with log_process("Document Chunking"):
                        chunk_result = await self._chunker.process_parse_result(parse_result)
                        # 展示分块策略和结果
                        chunk_info = []
                        for i, chunk in enumerate(chunk_result.chunks[:3]):  # 只显示前3个
                            chunk_info.append({
                                "chunk": i,
                                "size": len(chunk.content),
                                "start": chunk.content[:50].replace('\n', ' ') + "..."
                            })
                        log_detailed(f"Chunking result", 
                                   data={
                                       "strategy": "hierarchical" if self.config.enable_hierarchical_chunking else "fixed_size",
                                       "chunk_size": self.config.chunk_size,
                                       "overlap": self.config.chunk_overlap,
                                       "total_chunks": chunk_result.total_chunks,
                                       "samples": chunk_info
                                   })
                    
                    # 如果启用元数据增强，对每个块进行增强
                    if self._metadata_enhancer:
                        with log_process("Metadata Enhancement"):
                            enhanced_chunks = []
                            enhanced_count = 0
                            for chunk in chunk_result.chunks:
                                try:
                                    # 增强元数据（方法会就地修改chunk并返回）
                                    enhanced_chunk = await self._metadata_enhancer.enhance_chunk(chunk)
                                    enhanced_chunks.append(enhanced_chunk)
                                    enhanced_count += 1
                                except Exception as e:
                                    logger.warning(f"Failed to enhance chunk metadata: {e}")
                                    enhanced_chunks.append(chunk)
                            chunk_result.chunks = enhanced_chunks
                            log_detailed(f"Enhanced {enhanced_count}/{len(chunk_result.chunks)} chunks")
                    
                    # 嵌入和存储
                    with log_process("Embedding and Indexing"):
                        for i, chunk in enumerate(chunk_result.chunks):
                            embedding_result = await self._embedder.embed_text(chunk.content)
                            # 生成唯一ID
                            chunk_id = f"{file_path.stem}_{i}_{chunk.start_char}"
                            # 合并并清理元数据
                            metadata = clean_metadata({
                                **chunk.metadata,
                                "source": str(file_path.name),
                                "file_path": str(file_path),
                                "chunk_index": i
                            })
                            
                            # 创建VectorDocument对象
                            doc = VectorDocument(
                                id=chunk_id,
                                text=chunk.content,
                                embedding=embedding_result.embedding,
                                metadata=metadata
                            )
                            await self._vector_store.add_documents([doc])
                            
                            # 如果配置了BM25，同时添加到BM25索引
                            if self._retriever and self._retriever._bm25_index:
                                await self._retriever._bm25_index.add_documents(
                                    documents=[chunk.content],
                                    doc_ids=[chunk_id],
                                    metadata=[metadata]
                                )
                            
                            if i == 0:  # 只在DEBUG模式下记录第一个chunk的详情
                                log_detailed(f"Sample chunk metadata", 
                                           data={k: v for k, v in metadata.items() 
                                                if k in ['summary', 'questions', 'chunk_type']})
                        
                        log_detailed(f"Indexed {len(chunk_result.chunks)} chunks to vector store")
                    
                    total_chunks += chunk_result.total_chunks
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                failed_files.append({
                    "file": str(file_path),
                    "error": str(e)
                })
        
        result = {
            "total_files": total_files,
            "processed_files": total_files - len(failed_files),
            "failed_files": failed_files,
            "total_chunks": total_chunks
        }
        
        logger.info(f"Document ingestion completed: {result['processed_files']}/{total_files} files, "
                   f"{total_chunks} chunks created")
        
        return result
    
    @log_step("Question Answering")
    async def ask(
        self, 
        question: str,
        top_k: int = 5,
        return_details: bool = False,
        retrieval_only: bool = False,
        **kwargs
    ) -> Union[str, Dict[str, Any], List[Any]]:
        """向知识库提问。
        
        Args:
            question: 问题
            top_k: 检索的文档数量
            return_details: 是否返回详细信息（默认False只返回答案）
            **kwargs: 其他参数
            
        Returns:
            如果return_details=False: 返回答案文本（包含引用）
            如果return_details=True: 返回包含答案、引用、上下文等的字典
            
        Example:
            # 简单使用
            answer = await engine.ask("什么是RAG技术?")
            
            # 获取详细信息
            details = await engine.ask("什么是RAG技术?", return_details=True)
            print(details["answer"])
            print(details["citations"])
        """
        await self._ensure_initialized()
        
        log_detailed(f"Processing question: {question}", 
                    data={"top_k": top_k, "return_details": return_details})
        
        # 检索
        with log_process("Retrieval", query=question[:50] + "..." if len(question) > 50 else question):
            contexts = await self._retriever.retrieve(question, top_k=top_k)
            
            # 展示检索结果
            retrieval_results = []
            expansion_info = {}
            
            for i, ctx in enumerate(contexts[:5]):  # 展示前5个
                result_info = {
                    "rank": i + 1,
                    "score": round(ctx.score, 3),
                    "source": ctx.metadata.get('source', 'unknown'),
                    "preview": ctx.content[:100].replace('\n', ' ') + "..."
                }
                
                # 如果有查询扩展信息，添加到结果中
                if 'expansion_appearances' in ctx.metadata:
                    result_info["found_by_queries"] = ctx.metadata.get('expansion_appearances', 1)
                    # 收集扩展统计
                    if not expansion_info:
                        expansion_info["expansion_used"] = True
                        expansion_info["queries"] = set()
                    for q in ctx.metadata.get('expansion_queries', []):
                        expansion_info["queries"].add(q)
                
                retrieval_results.append(result_info)
            
            # 构建日志数据
            log_data = {
                "total_retrieved": len(contexts),
                "top_results": retrieval_results
            }
            
            # 如果使用了查询扩展，添加扩展信息
            if expansion_info:
                log_data["query_expansion"] = {
                    "enabled": True,
                    "num_queries": len(expansion_info["queries"]),
                    "sample_queries": list(expansion_info["queries"])[:3]
                }
            
            log_detailed(f"Retrieval results", data=log_data)
            
            # 如果启用重排序，对结果进行重排
            if self._reranker and contexts:
                with log_process("Reranking"):
                    # 保存原始排序用于对比
                    original_order = [(ctx.metadata.get('source', ''), ctx.score) for ctx in contexts[:5]]
                    
                    initial_count = len(contexts)
                    contexts = await self._reranker.rerank(question, contexts, top_k=self.config.rerank_top_k)
                    
                    # 展示重排序效果
                    rerank_results = []
                    for i, ctx in enumerate(contexts[:5]):
                        rerank_results.append({
                            "rank": i + 1,
                            "score": round(ctx.score, 3),
                            "source": ctx.metadata.get('source', 'unknown'),
                            "preview": ctx.content[:100].replace('\n', ' ') + "..."
                        })
                    
                    log_detailed(f"Reranking effect", 
                               data={
                                   "method": self.config.reranker_model if hasattr(self.config, 'reranker_model') else 'default',
                                   "before": original_order[:3],
                                   "after": [(ctx.metadata.get('source', ''), round(ctx.score, 3)) for ctx in contexts[:3]],
                                   "top_results": rerank_results
                               })
        
        if not contexts:
            logger.warning("No relevant contexts found for the question")
            if retrieval_only:
                return []
            no_context_answer = "抱歉，我在知识库中没有找到相关信息。"
            if return_details:
                return {
                    "question": question,
                    "answer": no_context_answer,
                    "contexts": [],
                    "citations": []
                }
            return no_context_answer
        
        # 如果只需要检索结果，直接返回
        if retrieval_only:
            log_detailed("Returning retrieval results only")
            return contexts
        
        # 生成答案
        with log_process("Generation", 
                        num_contexts=len(contexts),
                        llm_provider=self.config.llm_provider):
            result = await self._generator.generate(question, contexts)
            log_detailed(f"Generated answer with {len(result.citations or [])} citations")
        
        if return_details:
            # 返回详细信息
            details = {
                "question": question,
                "answer": result.answer,
                "contexts": [
                    {
                        "content": ctx.content[:200] + "..." if len(ctx.content) > 200 else ctx.content,
                        "metadata": ctx.metadata,
                        "score": ctx.score
                    } 
                    for ctx in contexts
                ],
                "citations": [
                    {
                        "index": cite.index,
                        "source": cite.document_title,
                        "text": cite.text
                    }
                    for cite in (result.citations or [])
                ]
            }
            log_detailed("Returning detailed response", 
                        data={"answer_length": len(result.answer), 
                              "num_citations": len(details["citations"])})
            return details
        else:
            # 返回简单答案（包含引用）
            if result.citations:
                citations_text = "\n\n**引用来源：**\n"
                for cite in result.citations:
                    source = cite.document_title or "未知来源"
                    citations_text += f"[{cite.index}] {source}\n"
                answer = result.answer + citations_text
            else:
                answer = result.answer
                
            log_detailed("Returning simple answer", 
                        data={"answer_length": len(answer)})
            return answer
    
    # 保留 ask_with_details 作为向后兼容的别名
    async def ask_with_details(
        self,
        question: str,
        top_k: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """向知识库提问并返回详细信息。
        
        注意：此方法已弃用，请使用 ask(question, return_details=True)
        
        Args:
            question: 问题
            top_k: 检索的文档数量
            
        Returns:
            包含答案、引用等详细信息的字典
        """
        return await self.ask(question, top_k=top_k, return_details=True, **kwargs)
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """搜索相关文档片段。
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量
            
        Returns:
            相关文档片段列表
        """
        await self._ensure_initialized()
        
        contexts = await self._retriever.retrieve(query, top_k=top_k)
        
        return [
            {
                "content": ctx.content,
                "score": ctx.score,
                "rerank_score": ctx.rerank_score,
                "final_score": ctx.final_score,
                "metadata": ctx.metadata
            }
            for ctx in contexts
        ]
    
    async def clear(self):
        """清空知识库。"""
        await self._ensure_initialized()
        await self._vector_store.clear()
    
    async def close(self):
        """关闭引擎，释放资源。"""
        if self._initialized:
            # 这里可以添加资源清理逻辑
            pass
    
    # 支持上下文管理器
    async def __aenter__(self):
        await self._ensure_initialized()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# 便捷函数
async def create_engine(**kwargs) -> KnowledgeEngine:
    """创建并初始化知识引擎。
    
    Example:
        engine = await create_engine()
        answer = await engine.ask("什么是RAG?")
    """
    engine = KnowledgeEngine(**kwargs)
    await engine._ensure_initialized()
    return engine