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
# 在导入部分添加
from .core.embedding.multimodal_embedder import MultimodalEmbedder
import base64

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
        
        # 如果设置了 rerank_score_threshold，自动启用 enable_relevance_threshold
        if kwargs.get('rerank_score_threshold') is not None:
            kwargs['enable_relevance_threshold'] = True
        
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
            # 阈值过滤参数
            enable_relevance_threshold=kwargs.get('enable_relevance_threshold', False),
            vector_score_threshold=kwargs.get('vector_score_threshold', 0.5),
            bm25_score_threshold=kwargs.get('bm25_score_threshold', 0.05),
            hybrid_score_threshold=kwargs.get('hybrid_score_threshold', 0.45),
            rerank_score_threshold=kwargs.get('rerank_score_threshold', None),
            reranker_device=kwargs.get('reranker_device', None),
            enable_hierarchical_chunking=kwargs.get('enable_hierarchical_chunking', False),
            enable_semantic_chunking=kwargs.get('enable_semantic_chunking', True),
            enable_metadata_enhancement=kwargs.get('enable_metadata_enhancement', False),
            chunk_size=kwargs.get('chunk_size', 512),
            chunk_overlap=kwargs.get('chunk_overlap', 50),
            language=kwargs.get('language', 'en'),  # 添加语言配置
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
        # 添加多模态嵌入器（延迟初始化）
        self._multimodal_embedder = None
    
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

        # 初始化多模态嵌入器（如果有DashScope API密钥）
        try:
            dashscope_key = self.config.embedding_api_key if self.config.embedding_provider == "dashscope" else None
            if dashscope_key:
                self._multimodal_embedder = MultimodalEmbedder(dashscope_key)
                await self._multimodal_embedder.initialize()
                logger.info("Multimodal embedder initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize multimodal embedder: {e}")
            self._multimodal_embedder = None
        
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
                        
                        # 检查是否有多模态数据
                        has_multimodal_data = (
                            parse_result.image is not None and 
                            isinstance(parse_result.image, dict) and
                            'images' in parse_result.image and
                            len(parse_result.image['images']) > 0
                        )
                        
                        if has_multimodal_data and self._multimodal_embedder:
                            # 使用多模态处理流程
                            chunk_count = await self._process_multimodal_content(parse_result, file_path)
                            total_chunks += chunk_count
                        else:
                            # 使用原有的处理流程（只处理文本）
                            chunk_count = await self._process_standard_content(parse_result, file_path)
                            total_chunks += chunk_count
        
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
        # 检索
        with log_process("Retrieval", query=question[:50] + "..." if len(question) > 50 else question):
            contexts = await self._retriever.retrieve(question, top_k=top_k)
            
            # 处理多模态数据
            for ctx in contexts:
                # 如果是图像类型且包含图像数据，确保有base64格式
                if (ctx.metadata.get('content_type') == 'image' and 
                    'image_data' in ctx.metadata):
                    # 确保图像数据以base64格式可用
                    if not ctx.metadata.get('image_base64'):
                        ctx.metadata['image_base64'] = ctx.metadata['image_data']
                # 如果content包含图像占位符，但metadata中有图像数据，更新content
                elif (ctx.metadata.get('content_type') == 'image' and 
                      '[图像' in ctx.content and 
                      'image_data' in ctx.metadata):
                    # 为图像内容添加实际的base64数据到content中
                    ctx.content = f"{ctx.content}\n\nimage_base64:{ctx.metadata['image_data']}"
                    ctx.metadata['image_base64'] = ctx.metadata['image_data']
            
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
    
    async def delete(
        self,
        source: Union[str, Path, List[str]]
    ) -> Dict[str, Any]:
        """从知识库删除文档。
        
        Args:
            source: 要删除的文档路径或文档ID列表
            
        Returns:
            删除结果统计
            
        Example:
            # 按文件名删除
            await engine.delete("doc.pdf")
            
            # 按文档ID删除
            await engine.delete(["file1_0_0", "file1_1_512"])
        """
        await self._ensure_initialized()
        
        # 统计
        deleted_vector_count = 0
        deleted_bm25_count = 0
        
        # 判断输入类型
        if isinstance(source, list):
            # 直接是文档ID列表
            doc_ids = source
        else:
            # 是文件路径或文件名，需要找到对应的文档ID
            source_path = Path(source)
            
            # 智能处理：如果输入看起来像完整文件名（包含扩展名），使用文件名
            # 否则使用stem（不含扩展名的部分）
            if '.' in source_path.name:
                # 使用完整文件名（包含扩展名）
                file_identifier = source_path.name
                # 移除扩展名用于匹配doc_id
                file_stem = source_path.stem
            else:
                # 输入可能已经是stem，直接使用
                file_stem = str(source_path)
                file_identifier = file_stem
            
            # 获取所有匹配的文档ID（格式：filename_chunkindex_startchar）
            doc_ids = []
            
            # 从向量存储获取所有文档
            try:
                # 通过provider获取collection
                if hasattr(self._vector_store, '_provider') and hasattr(self._vector_store._provider, '_collection'):
                    all_docs = self._vector_store._provider._collection.get()
                    for doc_id in all_docs["ids"]:
                        # 匹配以文件stem开头的文档ID
                        if doc_id.startswith(f"{file_stem}_"):
                            doc_ids.append(doc_id)
                else:
                    logger.warning("Vector store does not support direct document retrieval")
            except Exception as e:
                logger.error(f"Failed to retrieve document IDs: {e}")
        
        if not doc_ids:
            if isinstance(source, list):
                logger.warning(f"No documents found for deletion with IDs: {source}")
            else:
                logger.warning(f"No documents found for deletion: {file_identifier}")
            return {
                "deleted_ids": [],
                "deleted_count": 0,
                "vector_deleted": 0,
                "bm25_deleted": 0
            }
        
        # 从向量存储删除
        try:
            await self._vector_store.delete_documents(doc_ids)
            deleted_vector_count = len(doc_ids)
            logger.info(f"Deleted {deleted_vector_count} documents from vector store")
        except Exception as e:
            logger.error(f"Failed to delete from vector store: {e}")
        
        # 从BM25索引删除
        if self._retriever and self._retriever._bm25_index:
            try:
                deleted_bm25_count = await self._retriever._bm25_index.delete_documents(doc_ids)
                logger.info(f"Deleted {deleted_bm25_count} documents from BM25 index")
            except Exception as e:
                logger.error(f"Failed to delete from BM25 index: {e}")
        
        return {
            "deleted_ids": doc_ids,
            "deleted_count": len(doc_ids),  # 总删除数
            "vector_deleted": deleted_vector_count,
            "bm25_deleted": deleted_bm25_count
        }
    
    async def update(
        self,
        source: Union[str, Path]
    ) -> Dict[str, Any]:
        """更新知识库中的文档（删除旧的，添加新的）。
        
        Args:
            source: 要更新的文档路径
            
        Returns:
            更新结果统计
            
        Example:
            # 更新文档
            await engine.update("doc.pdf")
            await engine.update("path/to/doc.pdf")
        """
        # 转换为Path对象
        file_path = Path(source)
        
        # 先删除旧文档 - 只使用文件名进行删除
        # 这样无论传入的是相对路径还是绝对路径都能正确匹配
        delete_result = await self.delete(file_path.name)
        
        # 再添加新文档 - 使用完整路径
        add_result = await self.add([file_path])
        
        return {
            "deleted": delete_result,
            "added": add_result
        }
    
    async def clear(self):
        """清空知识库。"""
        await self._ensure_initialized()
        await self._vector_store.clear()
        if self._retriever and self._retriever._bm25_index:
            await self._retriever._bm25_index.clear()
    
    async def list(
        self,
        filter: Optional[Dict[str, Any]] = None,
        page: int = 1,
        page_size: int = 20,
        return_stats: bool = True
    ) -> Dict[str, Any]:
        """列出知识库中的文档。
        
        Args:
            filter: 过滤条件，支持：
                - file_type: 文件类型，如 "pdf", "md"
                - name_pattern: 文件名模式匹配
                - created_after: 创建时间之后
                - created_before: 创建时间之前
            page: 页码，从1开始
            page_size: 每页数量
            return_stats: 是否返回统计信息（chunks数量、总大小等）
            
        Returns:
            包含文档列表和元信息的字典：
            {
                "documents": [
                    {
                        "name": "文档名.pdf",
                        "path": "/path/to/文档名.pdf",
                        "chunks_count": 10,  # 仅当return_stats=True时
                        "total_size": 1024,  # 仅当return_stats=True时
                        "created_at": "2024-01-01T00:00:00",
                        "metadata": {...}
                    }
                ],
                "total": 100,  # 总文档数
                "page": 1,
                "page_size": 20,
                "pages": 5  # 总页数
            }
        """
        await self._ensure_initialized()
        
        # 调用向量存储的list方法
        return await self._vector_store.list_documents(
            filter=filter,
            page=page,
            page_size=page_size,
            return_stats=return_stats
        )

    async def _process_standard_content(self, parse_result, file_path) -> int:
        """处理标准文本内容（非多模态）"""
        # 分块处理
        with log_process("Text Chunking"):
            chunking_result = await self._chunker.process_parse_result(parse_result)

        # 元数据增强
        if self._metadata_enhancer:
            with log_process("Metadata Enhancement"):
                chunking_result = await self._metadata_enhancer.enhance_chunks(chunking_result)

        # 生成嵌入向量
        with log_process("Text Embedding"):
            # 准备文本内容
            texts = [chunk.content for chunk in chunking_result.chunks]
            embeddings = await self._embedder.embed_batch(texts)

        # 创建向量文档
        vector_docs = []
        for i, (chunk, embedding) in enumerate(zip(chunking_result.chunks, embeddings)):
            # 生成文档ID
            doc_id = f"{file_path.stem}_{i}_{chunk.start_char}"

            # 清理元数据
            metadata = clean_metadata({
                **chunking_result.document_metadata,
                **chunk.metadata,
                'chunk_index': i,
                'total_chunks': len(chunking_result.chunks)
            })

            vector_doc = VectorDocument(
                id=doc_id,
                text=chunk.content,
                embedding=embedding.embedding,
                metadata=metadata
            )
            vector_docs.append(vector_doc)

        # 存储到向量数据库
        with log_process("Vector Storage"):
            await self._vector_store.add_documents(vector_docs)

        # 添加到BM25索引
        if self._retriever and hasattr(self._retriever, '_bm25_index') and self._retriever._bm25_index:
            with log_process("BM25 Indexing"):
                # 分离文档数据为三个列表
                documents = [doc.text for doc in vector_docs]
                doc_ids = [doc.id for doc in vector_docs]
                metadata_list = [doc.metadata for doc in vector_docs]
                
                await self._retriever._bm25_index.add_documents(
                    documents=documents,
                    doc_ids=doc_ids,
                    metadata=metadata_list
                )

        return len(vector_docs)

    async def _process_multimodal_content(self, parse_result, file_path) -> int:
        """处理多模态内容（文本+图像）"""
        # 准备多模态内容
        multimodal_contents = []

        # 添加文本块
        if parse_result.image and 'text_chunks' in parse_result.image:
            for text_chunk in parse_result.image['text_chunks']:
                multimodal_contents.append({
                    'type': 'text',
                    'content': text_chunk['content'],
                    'metadata': {
                        'page': text_chunk.get('page', 0),
                        'chunk_type': 'text'
                    }
                })

        # 添加图像
        if parse_result.image and 'images' in parse_result.image:
            for img in parse_result.image['images']:
                multimodal_contents.append({
                    'type': 'image',
                    'content': img,
                    'metadata': {
                        'page': img.get('page', 0),
                        'index': img.get('index', 0),
                        'chunk_type': 'image'
                    }
                })

        # 生成多模态嵌入
        with log_process("Multimodal Embedding"):
            embeddings = await self._multimodal_embedder.generate_embeddings(multimodal_contents)

        # 创建向量文档
        vector_docs = []
        for i, (content, embedding) in enumerate(zip(multimodal_contents, embeddings)):
            # 生成文档ID
            doc_id = f"{file_path.stem}_{i}_{content['metadata'].get('page', 0)}"

            # 准备文本内容（用于存储和检索）
            if content['type'] == 'text':
                text_content = content['content']
            else:
                text_content = f"[图像 - 页面 {content['metadata'].get('page', 0)}]"

            # 清理元数据
            metadata = clean_metadata({
                **parse_result.metadata,
                **content['metadata'],
                'chunk_index': i,
                'total_chunks': len(multimodal_contents),
                'content_type': content['type'],
                'is_multimodal': True
            })
            
            # 如果是图像类型，保存原始图像数据到metadata
            if content['type'] == 'image' and 'data' in content['content']:
                # 将图像字节数据转换为base64字符串存储
                image_base64 = base64.b64encode(content['content']['data']).decode('utf-8')
                metadata['image_data'] = image_base64
                # 同时更新文本内容，包含base64数据
                text_content = f"[图像 - 页面 {content['metadata'].get('page', 0)}]\n\nimage_base64:{image_base64}"

            vector_doc = VectorDocument(
                id=doc_id,
                text=text_content,
                embedding=embedding.embedding,
                metadata=metadata
            )
            vector_docs.append(vector_doc)

        # 存储到向量数据库
        with log_process("Vector Storage"):
            await self._vector_store.add_documents(vector_docs)

        # 添加到BM25索引（只添加文本内容）
        if self._retriever and hasattr(self._retriever, '_bm25_index') and self._retriever._bm25_index:
            with log_process("BM25 Indexing"):
                # 过滤出文本类型的文档
                text_vector_docs = [
                    doc for doc in vector_docs
                    if doc.metadata.get('content_type') == 'text'
                ]
                
                if text_vector_docs:
                    # 分离为三个列表
                    documents = [doc.text for doc in text_vector_docs]
                    doc_ids = [doc.id for doc in text_vector_docs]
                    metadata_list = [doc.metadata for doc in text_vector_docs]
                    
                    await self._retriever._bm25_index.add_documents(
                        documents=documents,
                        doc_ids=doc_ids,
                        metadata=metadata_list
                    )
                # text_docs = [
                #     {
                #         'id': doc.id,
                #         'text': doc.text,
                #         'metadata': doc.metadata
                #     }
                #     for doc in vector_docs
                #     if doc.metadata.get('content_type') == 'text'
                # ]
                # if text_docs:
                #     await self._retriever._bm25_index.add_documents(text_docs)

        return len(vector_docs)

    async def close(self):
        """关闭引擎，释放资源。"""
        if self._initialized:
            # 关闭所有组件
            if self._retriever:
                await self._retriever.close()
            if self._generator:
                # Generator might need close in future
                pass
            self._initialized = False
    
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