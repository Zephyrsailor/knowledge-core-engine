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
import asyncio
import base64
import json
from datetime import datetime



from .core.config import RAGConfig
from .core.parsing.document_processor import DocumentProcessor
from .core.chunking.pipeline import ChunkingPipeline
from .core.chunking.enhanced_chunker import EnhancedChunker
from .core.chunking.smart_chunker import SmartChunker
from .core.chunking import ChunkAgent, ChunkConfig
from .core.chunking.base import ChunkResult
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
# 添加ChromaAgent导入
from .core.embedding.chroma_agent import ChromaAgent

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
        auto_caption: bool = True,  # 添加自动图片描述配置
        **kwargs
    ):
        """初始化知识引擎。
        
        Args:
            llm_provider: LLM提供商 (deepseek/qwen/openai)
            embedding_provider: 嵌入模型提供商 (dashscope/openai)
            persist_directory: 知识库存储路径
            log_level: 日志级别 (DEBUG/INFO/WARNING/ERROR)，默认使用环境变量或INFO
            auto_caption: 是否自动为图片生成描述
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
            extra_params=kwargs.get('extra_params', {}),
            # 切片配置
            max_chunk_size=kwargs.get('max_chunk_size',1000),  # 最大切片大小（字符数）
            min_chunk_size=kwargs.get('min_chunk_size',100),  # 最小切片大小（字符数）
            overlap_size=kwargs.get('overlap_size',100),  # 重叠大小（字符数）
            preserve_sentences=kwargs.get('preserve_sentences',True),  # 保持句子完整性
            preserve_paragraphs=kwargs.get('preserve_paragraphs',True)  # 保持段落完整性
        )
        
        # 添加图片描述配置
        self.auto_caption = auto_caption
        self.caption_llm = None  # 延迟初始化
        
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
        # 添加ChromaAgent（延迟初始化）
        self._chroma_agent = None
        
        # 初始化ChunkAgent
        chunk_config = ChunkConfig(
            max_chunk_size=self.config.chunk_size,
            min_chunk_size=self.config.chunk_size // 4,
            overlap_size=self.config.chunk_overlap,
            preserve_sentences=True,
            preserve_paragraphs=True
        )
        self.chunk_agent = ChunkAgent(chunk_config)
    
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
        
        # 初始化图片描述LLM（如果启用自动描述）
        if self.auto_caption:
            from langchain_openai import ChatOpenAI
            from .core.parsing.utils.mineru_utils import Qwen3VL8BInstruct
            try:
                model_config = Qwen3VL8BInstruct()
                self.caption_llm = ChatOpenAI(
                    openai_api_base=model_config.api_base,
                    openai_api_key=model_config.api_key,
                    model_name=model_config.model,
                    streaming=False,
                    temperature=0.1,
                    max_tokens=512,
                    extra_body={
                        "vl_high_resolution_images": "True",
                        "top_k": 1,
                    }
                )
                logger.info("图片描述模型初始化成功")
            except Exception as e:
                logger.warning(f"初始化图片描述LLM失败: {e}")
                self.caption_llm = None
        
        # 初始化异步组件
        await self._embedder.initialize()
        await self._vector_store.initialize()
        await self._retriever.initialize()
        if self._reranker:
            await self._reranker.initialize()
        await self._generator.initialize()
        
        # 初始化ChromaAgent
        try:
            config = {
                'output_dir': getattr(self.config, 'output_dir', './output'),
                'cls_dir': getattr(self.config, 'cls_dir', 'cls')
            }
            self._chroma_agent = ChromaAgent(vector_store=self._vector_store, config=config)
            logger.info("ChromaAgent initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize ChromaAgent: {e}")
            self._chroma_agent = None
        
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
                files = [f for f in files if f.suffix in ['.pdf', '.docx', '.md', '.txt','.jpg','.png']]
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

    @log_step("Add Documents to Knowledge Base V2")
    async def add_v2(
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
                    await engine.add_v2("doc.pdf")

                    # 添加整个目录
                    await engine.add_v2("docs/")

                    # 添加多个文件
                    await engine.add_v2(["doc1.pdf", "doc2.md"])
                """
        await self._ensure_initialized()

        # 统一处理输入
        if isinstance(source, (str, Path)):
            source = Path(source)
            if source.is_dir():
                files = list(source.glob("**/*"))
                files = [f for f in files if f.suffix in ['.pdf', '.docx', '.md', '.txt', '.jpg', '.png']]
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

                #todo:重复上传检测待完成

                # doc_check_id = f"{file_path.stem}_0_0"  # 使用第一个chunk的ID作为检查标识
                # existing_doc = await self._vector_store.get_document(doc_check_id)
                #
                # if existing_doc:
                #     logger.info(f"Document {file_path.name} already exists in knowledge base, skipping")
                #     # 统计现有chunks数量
                #     chunk_count = 0
                #     while True:
                #         check_id = f"{file_path.stem}_{chunk_count}_0"
                #         if not await self._vector_store.get_document(check_id):
                #             break
                #         chunk_count += 1
                #     total_chunks += chunk_count
                #     continue

                with log_process(f"Processing {file_path.name}",
                                 file_type=file_path.suffix,
                                 file_size=file_path.stat().st_size):

                    # 解析文档
                    with log_process("Document Parsing"):
                        parse_result = await self._parser.process(file_path)

                    # 添加图片描述逻辑
                    if parse_result.success and self.auto_caption:
                        output_dir = parse_result.output_dir
                        file_stem = file_path.stem
                        
                        # 查找并处理content_list.json
                        content_list_path = self._find_content_list_path(output_dir, file_stem)
                        if content_list_path:
                            content_list = self._load_content_list(content_list_path)
                            if content_list:
                                with log_process("Image Captioning"):
                                    await self._process_content_list_captions(content_list, str(content_list_path.parent))
                        
                        # 处理单独的图片文件
                        elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
                            try:
                                with log_process("Image Captioning"):
                                    caption = await self._generate_image_caption(file_path)
                                    if caption and hasattr(parse_result, 'data') and parse_result.data:
                                        parse_result.data['image_caption'] = caption
                                        logger.info(f"为图片文件 {file_path.name} 生成描述")
                            except Exception as e:
                                logger.warning(f"为图片文件生成描述时出错: {e}")

                    # 添加chunk逻辑
                    if parse_result.success:
                        # 步骤3: 文档切片
                        with log_process("Document Chunking"):
                            # 查找content_list文件
                            content_list_path = self._find_content_list_path(parse_result.output_dir, file_path.stem)
                            if content_list_path:
                                # 使用content_list文件的父目录作为output_dir，这样可以正确找到images目录
                                actual_output_dir = str(content_list_path.parent)
                                chunks = self.chunk_agent.chunk_document(
                                    str(content_list_path),
                                    actual_output_dir,
                                    file_path.stem
                                )

                                # 转换为原有格式
                                chunk_result = {
                                    'success': True,
                                    'data': {
                                        'chunks': [chunk.to_dict() for chunk in chunks],
                                        'total_count': len(chunks)
                                    }
                                }
                            else:
                                logger.error("未找到content_list文件")
                                chunk_result = {'success': False, 'error': '未找到content_list文件'}

                    if chunk_result.get('success'):
                        chunks = chunk_result['data']['chunks']
                        logger.info(f"发现 {len(chunks)} 个切片")

                        # 步骤4: 使用ChromaAgent直接处理chunks
                        with log_process("Processing Chunks with ChromaAgent"):
                            if self._chroma_agent:
                                # 直接调用ChromaAgent的process_chunks_for_service方法
                                duplicate_check = self._chroma_agent.process_chunks_for_service(
                                    chunks=chunks,  # 直接传递chunks
                                    source_file_name=file_path.name,
                                    is_reparse=False  # 根据实际需求设置
                                )
                            else:
                                # 如果ChromaAgent未初始化，使用原有逻辑作为fallback
                                chunks_data = self._prepare_chunks_for_storage([chunk.to_dict() for chunk in chunks], file_path.name)
                                duplicate_check = self._check_duplicates(chunks_data)

                        # 根据处理结果选择处理方式
                        if duplicate_check['new_items']:
                            # 进行向量化和存储
                            chunk_count = await self._process_chunks_with_vectorization(
                                duplicate_check, file_path
                            )
                            total_chunks += chunk_count
                    # else:
                    #     # 如果解析失败，使用原有逻辑作为fallback
                    #     # 检查是否有多模态数据 - 参考Vision_RAG的逻辑
                    #     has_multimodal_data = False
                    #
                    #     # 检查解析结果中的图像数据
                    #     if parse_result.image is not None:
                    #         # 支持多种图像数据格式
                    #         if isinstance(parse_result.image, list) and len(parse_result.image) > 0:
                    #             # 图像列表格式
                    #             has_multimodal_data = True
                    #         elif isinstance(parse_result.image, dict):
                    #             # 字典格式，检查是否包含图像
                    #             if ('images' in parse_result.image and
                    #                 len(parse_result.image['images']) > 0):
                    #                 has_multimodal_data = True
                    #             # 检查是否有其他多模态内容
                    #             elif any(key in parse_result.image for key in
                    #                    ['text_chunks', 'tables', 'equations']):
                    #                 has_multimodal_data = True
                    #
                    #     # 根据Vision_RAG的处理逻辑选择处理方式
                    #     if has_multimodal_data and self._multimodal_embedder:
                    #         # 使用多模态处理流程 - 采用Vision_RAG的向量化策略
                    #         chunk_count = await self._process_multimodal_content_v2(parse_result, file_path)
                    #         total_chunks += chunk_count
                    #     else:
                    #         # 使用原有的处理流程（只处理文本）
                    #         chunk_count = await self._process_standard_content(parse_result, file_path)
                    #         total_chunks += chunk_count

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



    async def document_detail(
        self,
        kb_id: str,
        file_id: str,
        include_embeddings: bool = False,
        chunk_limit: Optional[int] = None,
        deduplicate: bool = True
    ) -> Dict[str, Any]:
        """获取文档的详细切片信息。
        
        Args:
            kb_id: 知识库ID（当前实现中暂未使用，为未来多知识库支持预留）
            file_id: 文件ID或文件名
            include_embeddings: 是否包含嵌入向量数据
            chunk_limit: 限制返回的切片数量，None表示返回所有
            deduplicate: 是否对切片进行去重处理（合并双入库的图像和表格切片）
            
        Returns:
            包含文档详细信息和切片数据的字典
        """
        await self._ensure_initialized()
        
        # 使用ChromaAgent处理文档详情查询
        if not hasattr(self, '_chroma_agent') or self._chroma_agent is None:
            # 初始化ChromaAgent
            from .core.embedding.chroma_agent import ChromaAgent
            self._chroma_agent = ChromaAgent(
                vector_store=self._vector_store,
                config=self.config.dict() if hasattr(self.config, 'dict') else {}
            )
        
        return self._chroma_agent.get_document_detail(
            file_id=file_id,
            include_embeddings=include_embeddings,
            chunk_limit=chunk_limit,
            deduplicate=deduplicate
        )

    async def _describe_html_tables(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """把 HTML 表格 chunk 转成自然语言描述用于 embedding（add_v2 专用）。

        过滤条件：chunk_type == 'table' AND embedding_type == 'text' AND content 含 <table
        处理：content 替换为 LLM 描述；HTML 原文保留在 metadata['table_content_html']
        目的：避免 <table><tr><td> 这类标签稀释语义向量，保留 HTML 供生成阶段使用

        复用 self._metadata_enhancer.llm_provider 作为 LLM，未启用 enhancer 时直接跳过。
        """
        if not self._metadata_enhancer or not chunks:
            return chunks

        targets = [
            c for c in chunks
            if c.get('metadata', {}).get('chunk_type') == 'table'
            and c.get('metadata', {}).get('embedding_type') == 'text'
            and '<table' in (c.get('content') or '')
        ]
        if not targets:
            return chunks

        await self._metadata_enhancer._ensure_initialized()
        provider = self._metadata_enhancer.llm_provider
        if not provider:
            logger.warning("Table description skipped: LLM provider unavailable")
            return chunks

        semaphore = asyncio.Semaphore(self._metadata_enhancer.config.max_concurrent_requests)

        async def _describe_one(c: Dict[str, Any]) -> None:
            async with semaphore:
                html = c['content']
                prompt = self._build_table_description_prompt(html)
                try:
                    response = await provider.generate(
                        messages=[
                            {"role": "system", "content": "你是一个专业的文档分析助手，擅长把 HTML 表格翻译成精炼的中文自然语言描述。"},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.1,
                        max_tokens=400,
                    )
                    desc = (response.get('content') or '').strip()
                    if desc:
                        c['metadata']['table_content_html'] = html
                        c['metadata']['table_description_applied'] = True
                        c['content'] = desc
                    else:
                        c['metadata']['table_description_failed'] = True
                        logger.warning(f"Empty table description for chunk {c.get('doc_id')}")
                except Exception as e:
                    logger.error(f"Table description failed for {c.get('doc_id')}: {e}")
                    c['metadata']['table_description_failed'] = True
                    c['metadata']['table_description_error'] = str(e)

        with log_process(f"Table HTML -> Description (n={len(targets)})"):
            await asyncio.gather(*(_describe_one(c) for c in targets))

        success = sum(1 for c in targets if c['metadata'].get('table_description_applied'))
        logger.info(f"Table descriptions: {success}/{len(targets)} succeeded")
        return chunks

    def _build_table_description_prompt(self, html: str) -> str:
        """构造 HTML → 自然语言描述的 prompt。"""
        return (
            "以下是从文档中提取的 HTML 表格。请用 2-4 句中文自然语言描述这个表格的主要内容，"
            "突出用户可能会查询的关键信息（例如列含义、关键行数据、适用范围等），"
            "避免直接复读 HTML 标签。\n\n"
            f"HTML 表格：\n{html}\n\n"
            "请直接输出描述文本，不要添加任何前缀、解释或 Markdown 标记。"
        )

    async def _enhance_text_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """统一的元数据增强入口（add / add_v2 共用）。

        若未启用 MetadataEnhancer 则原样返回；启用则调用 LLM
        给每个 chunk 注入 summary / questions / keywords / llm_chunk_type。

        Args:
            chunks: [{'content': str, 'metadata': dict, ...}, ...]

        Returns:
            原 list（in-place 增强 metadata）
        """
        if not self._metadata_enhancer or not chunks:
            return chunks

        with log_process("Metadata Enhancement"):
            chunk_results = [
                ChunkResult(
                    content=c['content'],
                    metadata=c['metadata'].copy(),
                    start_char=0,
                    end_char=len(c['content'])
                )
                for c in chunks
            ]
            enhanced = await self._metadata_enhancer.enhance_batch(chunk_results)

            for orig, enh in zip(chunks, enhanced):
                for key in ('summary', 'questions', 'keywords'):
                    if key in enh.metadata:
                        orig['metadata'][key] = enh.metadata[key]
                # LLM 的语义分类用独立 key 存，避免覆盖 ChunkAgent 的结构 chunk_type
                if 'chunk_type' in enh.metadata:
                    orig['metadata']['llm_chunk_type'] = enh.metadata['chunk_type']

        return chunks

    def _build_embed_text(self, content: str, metadata: Dict[str, Any]) -> str:
        """根据 use_multi_vector 决定用于嵌入的文本（add / add_v2 共用）。

        开启时拼接 Content + Summary + Questions + Keywords；
        关闭或缺字段时退化到裸 content。
        """
        if not getattr(self.config, 'use_multi_vector', False):
            return content

        parts = [f"Content: {content}"]
        if metadata.get('summary'):
            parts.append(f"Summary: {metadata['summary']}")
        if metadata.get('questions'):
            qs = metadata['questions']
            qs_text = ' '.join(qs) if isinstance(qs, list) else str(qs)
            parts.append(f"Questions: {qs_text}")
        if metadata.get('keywords'):
            kws = metadata['keywords']
            kws_text = ', '.join(kws) if isinstance(kws, list) else str(kws)
            parts.append(f"Keywords: {kws_text}")

        return "\n\n".join(parts)

    async def _index_text_chunks_to_bm25(self, vector_docs: List[VectorDocument]) -> None:
        """把"走文本 embedding"的 chunk 索引到 BM25（add / add_v2 共用）。

        入选规则（满足其一即入 BM25）：
          1. 没有 chunk_type 字段（兼容 add 路径的裸文本 chunk）
          2. chunk_type == 'text'
          3. chunk_type 是 'table' / 'equation' 且 embedding_type == 'text'
             （表格描述 / 公式文本这类非 'text' 结构类型但走文本 embedding 的）

        排除：visual 分支的 chunk（图像/表格截图本体，通过多模态嵌入器入库）。
        """
        if not (self._retriever and getattr(self._retriever, '_bm25_index', None)):
            return

        def _should_index(meta: Dict[str, Any]) -> bool:
            ctype = meta.get('chunk_type', 'text')
            if ctype == 'text':
                return True
            # 非 text 结构类型，但走文本 embedding 路径的也要进 BM25
            return meta.get('embedding_type') == 'text'

        text_docs = [d for d in vector_docs if _should_index(d.metadata)]
        if not text_docs:
            return

        with log_process("BM25 Indexing"):
            await self._retriever._bm25_index.add_documents(
                documents=[d.text for d in text_docs],
                doc_ids=[d.id for d in text_docs],
                metadata=[d.metadata for d in text_docs]
            )

    async def _process_standard_content(self, parse_result, file_path) -> int:
        """处理标准文本内容（非多模态）"""
        # 分块处理
        with log_process("Text Chunking"):
            chunking_result = await self._chunker.process_parse_result(parse_result)

        # 转成统一 dict 格式喂给共用方法
        chunks_data = [
            {
                'content': chunk.content,
                'metadata': {
                    **chunking_result.document_metadata,
                    **chunk.metadata,
                    'chunk_index': i,
                    'total_chunks': len(chunking_result.chunks),
                },
                '_doc_id': f"{file_path.stem}_{i}_{chunk.start_char}",
            }
            for i, chunk in enumerate(chunking_result.chunks)
        ]

        # 元数据增强（共用）
        chunks_data = await self._enhance_text_chunks(chunks_data)

        # 多向量嵌入文本（共用拼接逻辑）
        with log_process("Text Embedding"):
            embed_texts = [self._build_embed_text(c['content'], c['metadata'])
                           for c in chunks_data]
            embeddings = await self._embedder.embed_batch(embed_texts)

        # 构造 VectorDocument
        vector_docs = []
        for c, emb in zip(chunks_data, embeddings):
            if not emb or not getattr(emb, 'embedding', None):
                continue
            vector_docs.append(VectorDocument(
                id=c['_doc_id'],
                text=c['content'],
                embedding=emb.embedding,
                metadata=clean_metadata(c['metadata'])
            ))

        # 入库
        if vector_docs:
            with log_process("Vector Storage"):
                await self._vector_store.add_documents(vector_docs)
            await self._index_text_chunks_to_bm25(vector_docs)

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
    
    def _prepare_chunks_for_storage(self, chunks: list, source_file: str) -> list:
        """准备切片数据用于存储"""
        prepared_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_data = {
                'content': chunk.get('content', ''),
                'metadata': {
                    'source_file': source_file,
                    'chunk_type': chunk.get('type', 'text'),
                    'chunk_id': chunk.get('id', f"chunk_{i}"),
                    'page_idx': chunk.get('page', 0),
                    'chunk_idx': i,
                    'parent_document': source_file,
                    'embedding_type': chunk.get('embedding_type', 'text'),
                    **chunk.get('metadata', {})
                }
            }
            
            # 如果有原始内容（如图像base64），保存
            if 'original_content' in chunk:
                chunk_data['original_content'] = chunk['original_content']
                chunk_data['metadata']['has_original_content'] = True
            
            prepared_chunks.append(chunk_data)
        
        return prepared_chunks
    
    def _check_duplicates(self, chunks_data: list) -> dict:
        """检查重复项"""
        # 简化版本的重复检测
        # 实际实现中可以查询向量数据库进行更精确的检测
        new_items = []
        existing_items = []
        
        for chunk_data in chunks_data:
            # 生成基于内容的唯一ID
            # content_id = self._generate_content_id(
            #     chunk_data['content'], 
            #     chunk_data['metadata']
            # )
            # chunk_data['doc_id'] = content_id
            if 'chunk_id' in chunk_data['metadata']:
                chunk_data['doc_id'] = chunk_data['metadata']['chunk_id']
            else:
                # fallback：生成UUID
                import uuid
                chunk_data['doc_id'] = str(uuid.uuid4())
            
            # 简单的重复检测逻辑
            # TODO: 实现更精确的重复检测
            new_items.append(chunk_data)
        
        return {
            'new_items': new_items,
            'existing_items': existing_items,
            'total_new': len(new_items),
            'total_existing': len(existing_items)
        }
    
    async def _process_chunks_with_vectorization(self, duplicate_check: dict, file_path) -> int:
        """处理切片并进行向量化（多模态版本：text 走共用文本流程，visual 单独走多模态嵌入器）"""
        new_chunks = duplicate_check['new_items']

        # 按嵌入类型分流（修复：原来用 hasattr 判 dict 属性恒为 False，视觉分支永不进入）
        visual_chunks = [c for c in new_chunks
                         if c['metadata'].get('embedding_type') == 'visual']
        text_chunks = [c for c in new_chunks
                       if c['metadata'].get('embedding_type') != 'visual']

        vector_docs: List[VectorDocument] = []

        # ---------- 文本分支：完全复用 add 路径的共用逻辑 ----------
        if text_chunks:
            # 表格 HTML → 自然语言描述（放在增强前，让 summary/questions 基于描述而非 HTML）
            text_chunks = await self._describe_html_tables(text_chunks)
            text_chunks = await self._enhance_text_chunks(text_chunks)

            with log_process("Text Embedding"):
                embed_texts = [self._build_embed_text(c['content'], c['metadata'])
                               for c in text_chunks]
                embeddings = await self._embedder.embed_batch(embed_texts)

            for c, emb in zip(text_chunks, embeddings):
                if not emb or not getattr(emb, 'embedding', None):
                    continue
                vector_docs.append(VectorDocument(
                    id=c['doc_id'],
                    text=c['content'],
                    embedding=emb.embedding,
                    metadata=clean_metadata(c['metadata'])
                ))

        # ---------- 视觉分支：add_v2 独有，逐个调多模态嵌入器 ----------
        if visual_chunks and self._multimodal_embedder:
            with log_process("Multimodal Embedding"):
                for c in visual_chunks:
                    image_path = (c['metadata'].get('table_image_path')
                                  or c['metadata'].get('image_path'))
                    if not image_path:
                        logger.warning(f"Visual chunk {c.get('doc_id')} missing image_path, skipped")
                        continue
                    img_type = Path(image_path).suffix[1:].lower()
                    embedding_result = await self._multimodal_embedder.generate_embeddings([{
                        'type': 'image',
                        'content': f"data:image/{img_type};base64,{c.get('original_content', '')}",
                        'metadata': c['metadata']
                    }])
                    if not embedding_result or not embedding_result[0].embedding:
                        continue
                    vector_docs.append(VectorDocument(
                        id=c['doc_id'],
                        text=c['content'],
                        embedding=embedding_result[0].embedding,
                        metadata=clean_metadata(c['metadata'])
                    ))

        # ---------- 入库 + BM25（共用） ----------
        if vector_docs:
            with log_process("Vector Storage"):
                await self._vector_store.add_documents(vector_docs)
            await self._index_text_chunks_to_bm25(vector_docs)

        return len(vector_docs)

    def _image_to_data_url(self, image_path: Union[str, Path]) -> str:
        """将图像路径转换为data URL格式"""
        try:
            from PIL import Image
            import io
            
            with Image.open(image_path) as img:
                # 转换为RGB格式
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 调整图像大小以避免过大
                max_size = (1024, 1024)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # 转换为base64
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                return f"data:image/jpeg;base64,{img_data}"
        except Exception as e:
            logger.error(f"转换图像 {image_path} 时出错: {e}")
            return ""
    
    async def _generate_image_caption(self, image_path: Union[str, Path]) -> str:
        """为图片生成描述"""
        if not self.caption_llm:
            return ""
            
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            
            data_url = self._image_to_data_url(image_path)
            if not data_url:
                return ""
                
            system_message = SystemMessage(content="你是一个专业的图像分析助手。请仔细观察图像并提供准确、详细的中文描述。描述应该包括图像的主要内容、对象、场景、颜色、布局等关键信息。请保持描述简洁明了，不超过200字。")
            
            human_message = HumanMessage(content=[
                {
                    "type": "image_url",
                    "image_url": {"url": data_url}
                },
                {
                    "type": "text",
                    "text": "请详细描述这张图片的内容。"
                }
            ])
            
            response = await self.caption_llm.ainvoke([system_message, human_message])
            caption = response.content.strip()
            
            logger.debug(f"为图片 {Path(image_path).name} 生成描述: {caption[:50]}...")
            return caption
            
        except Exception as e:
            logger.error(f"生成图片描述失败 {image_path}: {e}")
            return ""
    
    def _find_content_list_path(self, output_dir: str, file_stem: str) -> Optional[Path]:
        """查找content_list.json文件路径"""
        output_path = Path(output_dir)
        
        # 首先尝试基本路径
        content_list_path = output_path / f'{file_stem}_content_list.json'
        if content_list_path.exists():
            return content_list_path
        
        # 尝试子目录路径（MinerU 2.0）
        subdir = output_path / file_stem
        if subdir.exists():
            method = "auto"  # 或从配置中获取
            content_list_path = subdir / method / f'{file_stem}_content_list.json'
            if content_list_path.exists():
                return content_list_path
        
        return None

    def _load_content_list(self, content_list_path: Path) -> Optional[List[Dict]]:
        """加载content_list.json文件"""
        try:
            import json
            with open(content_list_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"加载content_list.json失败: {e}")
            return None

    async def _process_content_list_captions(self, content_list: List[Dict], output_dir: str):
        """异步处理内容列表中的图片描述"""
        if not self.auto_caption or not self.caption_llm:
            return
            
        import json
        from pathlib import Path
        
        updated = False
        output_dir_path = Path(output_dir)
        
        for content in content_list:
            if content.get('type') == 'image' and not content.get('image_caption'):
                img_path = content.get('img_path', '')
                if img_path:
                    # 处理相对路径和绝对路径
                    image_path = Path(img_path) if Path(img_path).is_absolute() else output_dir_path / img_path
                        
                    if image_path.exists():
                        caption = await self._generate_image_caption(image_path)
                        if caption:
                            if 'image_caption' not in content:
                                content['image_caption'] = []
                            content['image_caption'].append(caption)
                            updated = True
                            logger.info(f"为图片 {image_path.name} 添加描述")
        
        # 保存更新
        if updated:
            self._save_content_list(content_list, output_dir_path)

    def _save_content_list(self, content_list: List[Dict], output_dir_path: Path):
        """保存content_list到文件"""
        import json
        
        # 查找现有的content_list文件
        existing_files = list(output_dir_path.glob('*_content_list.json'))
        if existing_files:
            content_list_path = existing_files[0]
        else:
            file_stem = output_dir_path.name
            content_list_path = output_dir_path / f'{file_stem}_content_list.json'
        
        try:
            with open(content_list_path, 'w', encoding='utf-8') as f:
                json.dump(content_list, f, ensure_ascii=False, indent=2)
            logger.info(f"已更新 {content_list_path}")
        except Exception as e:
            logger.error(f"保存content_list.json失败: {e}")

    async def _process_multimodal_content_v2(self, parse_result, file_path) -> int:
        """处理多模态内容（参考Vision_RAG的向量化逻辑）"""
        # 准备多模态内容列表
        prepared_chunks = []
        
        # 处理文本块
        if parse_result.image and 'text_chunks' in parse_result.image:
            for i, text_chunk in enumerate(parse_result.image['text_chunks']):
                chunk_data = {
                    'content': text_chunk['content'],
                    'metadata': {
                        'source_file': file_path.name,
                        'chunk_type': 'text',
                        'chunk_id': f"{file_path.stem}_text_{i}",
                        'page_idx': text_chunk.get('page', 0),
                        'chunk_idx': i,
                        'parent_document': file_path.name,
                        'embedding_type': 'text',
                        **parse_result.metadata
                    }
                }
                prepared_chunks.append(chunk_data)
        
        # 处理图像
        if parse_result.image and 'images' in parse_result.image:
            for i, img in enumerate(parse_result.image['images']):
                # 准备图像内容描述
                image_content = f"[图像 - 页面 {img.get('page', 0)}]"
                if 'caption' in img:
                    image_content += f"\n描述: {img['caption']}"
                
                chunk_data = {
                    'content': image_content,
                    'metadata': {
                        'source_file': file_path.name,
                        'chunk_type': 'image',
                        'chunk_id': f"{file_path.stem}_image_{i}",
                        'page_idx': img.get('page', 0),
                        'chunk_idx': i,
                        'parent_document': file_path.name,
                        'embedding_type': 'visual',
                        'image_path': img.get('path'),
                        **parse_result.metadata
                    }
                }
                
                # 添加Base64图像数据（参考Vision_RAG的做法）
                if 'data' in img:
                    image_base64 = base64.b64encode(img['data']).decode('utf-8')
                    chunk_data['original_content'] = image_base64
                    chunk_data['metadata']['has_original_content'] = True
                
                prepared_chunks.append(chunk_data)
        
        # 检查重复项（参考Vision_RAG的重复检测逻辑）
        new_chunks = []
        for chunk_data in prepared_chunks:
            # 生成基于内容的唯一ID
            content_id = self._generate_content_id(
                chunk_data['content'], 
                chunk_data['metadata']
            )
            
            # 检查是否已存在（简化版本，实际可以查询向量数据库）
            chunk_data['doc_id'] = content_id
            new_chunks.append(chunk_data)
        
        # 批量向量化和存储
        vector_docs = []
        for chunk_data in new_chunks:
            # 根据类型选择嵌入方式
            if chunk_data['metadata']['embedding_type'] == 'visual':
                # 使用多模态嵌入器处理图像
                embedding_result = await self._multimodal_embedder.generate_embeddings([
                    {
                        'type': 'image',
                        'content': chunk_data.get('original_content', ''),
                        'metadata': chunk_data['metadata']
                    }
                ])
                embedding = embedding_result[0].embedding if embedding_result else None
            else:
                # 使用文本嵌入器处理文本
                embedding_result = await self._embedder.embed_text(chunk_data['content'])
                embedding = embedding_result.embedding if embedding_result else None
            
            if embedding:
                # 清理元数据
                metadata = clean_metadata(chunk_data['metadata'])
                
                vector_doc = VectorDocument(
                    id=chunk_data['doc_id'],
                    text=chunk_data['content'],
                    embedding=embedding,
                    metadata=metadata
                )
                vector_docs.append(vector_doc)
        
        # 存储到向量数据库
        if vector_docs:
            with log_process("Vector Storage"):
                await self._vector_store.add_documents(vector_docs)
            
            # 添加文本内容到BM25索引
            if self._retriever and hasattr(self._retriever, '_bm25_index') and self._retriever._bm25_index:
                text_docs = [doc for doc in vector_docs if doc.metadata.get('chunk_type') == 'text']
                if text_docs:
                    with log_process("BM25 Indexing"):
                        documents = [doc.text for doc in text_docs]
                        doc_ids = [doc.id for doc in text_docs]
                        metadata_list = [doc.metadata for doc in text_docs]
                        
                        await self._retriever._bm25_index.add_documents(
                            documents, doc_ids, metadata_list
                        )
        
        return len(vector_docs)
    
    def _generate_content_id(self, content: str, metadata: dict) -> str:
        """生成基于内容的唯一ID（参考Vision_RAG的实现）"""
        import hashlib
        
        # 创建用于生成ID的字符串
        id_components = [content]
        
        if metadata:
            # 添加关键元数据
            if 'source_file' in metadata:
                id_components.append(metadata['source_file'])
            if 'page_idx' in metadata:
                id_components.append(str(metadata['page_idx']))
            if 'chunk_type' in metadata:
                id_components.append(metadata['chunk_type'])
        
        # 生成MD5哈希作为ID
        content_str = '|'.join(id_components)
        return hashlib.md5(content_str.encode('utf-8')).hexdigest()


    @log_step("Update Image Chunk Content")
    async def update_image_chunk_content(
        self,
        chunk_id: str,
        new_content: str,
        regenerate_embedding: bool = True
    ) -> Dict[str, Any]:
        """更新图像类型chunk的content内容。
        
        Args:
            chunk_id: 要更新的chunk ID
            new_content: 新的content内容
            regenerate_embedding: 是否重新生成嵌入向量（默认True）
            
        Returns:
            更新结果统计
            
        Example:
            # 更新图像chunk的描述
            result = await engine.update_image_chunk_content(
                chunk_id="doc_image_1",
                new_content="这是一张展示深度学习架构的图表，包含输入层、隐藏层和输出层。"
            )
        """
        await self._ensure_initialized()
        
        try:
            # 1. 验证chunk存在性和类型
            with log_process("Validating Chunk"):
                # 从向量数据库获取现有chunk
                existing_doc = await self._vector_store.get_document(chunk_id)
                if not existing_doc:
                    raise ValueError(f"Chunk with ID '{chunk_id}' not found")
                
                # 验证chunk类型
                chunk_type = existing_doc.metadata.get('chunk_type', '')
                if chunk_type != 'image':
                    raise ValueError(f"Chunk '{chunk_id}' is not an image type (current type: {chunk_type})")
                
                logger.info(f"Found image chunk '{chunk_id}' for content update")
            
            # 2. 准备更新数据
            updated_metadata = existing_doc.metadata.copy()
            updated_metadata['content_updated_at'] = str(datetime.now().isoformat())
            updated_metadata['original_content'] = existing_doc.text  # 保存原始内容
            
            # 3. 重新生成嵌入向量（如果需要）
            new_embedding = None
            if regenerate_embedding:
                with log_process("Regenerating Embedding"):
                    # 根据embedding_type选择合适的嵌入方式
                    embedding_type = updated_metadata.get('embedding_type', 'text')
                    
                    if embedding_type == 'visual' and self._multimodal_embedder:
                        # 使用多模态嵌入器
                        # 如果有原始图像数据，使用图像嵌入
                        if updated_metadata.get('has_original_content') and 'original_content' in existing_doc.metadata:
                            original_content = existing_doc.metadata.get('original_content', '')
                            embedding_result = await self._multimodal_embedder.generate_embeddings([
                                {
                                    'type': 'image',
                                    'content': f"data:image/jpeg;base64,{original_content}",
                                    'metadata': updated_metadata
                                }
                            ])
                            new_embedding = embedding_result[0].embedding if embedding_result else None
                        else:
                            # 如果没有原始图像数据，使用文本嵌入
                            embedding_result = await self._embedder.embed_text(new_content)
                            new_embedding = embedding_result.embedding if embedding_result else None
                    else:
                        # 使用文本嵌入器
                        embedding_result = await self._embedder.embed_text(new_content)
                        new_embedding = embedding_result.embedding if embedding_result else None
                    
                    if not new_embedding:
                        logger.warning(f"Failed to generate new embedding for chunk '{chunk_id}'")
                        new_embedding = existing_doc.embedding  # 使用原有嵌入
            else:
                new_embedding = existing_doc.embedding
            
            # 4. 创建更新后的向量文档
            updated_vector_doc = VectorDocument(
                id=chunk_id,
                text=new_content,
                embedding=new_embedding,
                metadata=clean_metadata(updated_metadata)
            )
            
            # 5. 更新向量数据库
            with log_process("Updating Vector Store"):
                # 删除旧文档
                await self._vector_store.delete_documents([chunk_id])
                # 添加新文档
                await self._vector_store.add_documents([updated_vector_doc])
                logger.info(f"Updated vector store for chunk '{chunk_id}'")
            
            # 6. 更新BM25索引（如果存在）
            bm25_updated = False
            if self._retriever and hasattr(self._retriever, '_bm25_index') and self._retriever._bm25_index:
                with log_process("Updating BM25 Index"):
                    try:
                        # 删除旧索引
                        await self._retriever._bm25_index.delete_documents([chunk_id])
                        # 添加新索引
                        await self._retriever._bm25_index.add_documents(
                            documents=[new_content],
                            doc_ids=[chunk_id],
                            metadata=[updated_metadata]
                        )
                        bm25_updated = True
                        logger.info(f"Updated BM25 index for chunk '{chunk_id}'")
                    except Exception as e:
                        logger.warning(f"Failed to update BM25 index: {e}")
            
            # 7. 返回更新结果
            result = {
                "chunk_id": chunk_id,
                "success": True,
                "original_content": existing_doc.text,
                "new_content": new_content,
                "content_length_change": len(new_content) - len(existing_doc.text),
                "embedding_regenerated": regenerate_embedding,
                "vector_store_updated": True,
                "bm25_index_updated": bm25_updated,
                "updated_at": updated_metadata['content_updated_at']
            }
            
            logger.info(f"Successfully updated image chunk '{chunk_id}' content")
            return result
            
        except Exception as e:
            logger.error(f"Failed to update image chunk content: {e}")
            return {
                "chunk_id": chunk_id,
                "success": False,
                "error": str(e)
            }

    @log_step("Batch Update Image Chunks Content")
    async def batch_update_image_chunks_content(
        self,
        updates: List[Dict[str, str]],
        regenerate_embedding: bool = True
    ) -> Dict[str, Any]:
        """批量更新多个图像chunk的content内容。
        
        Args:
            updates: 更新列表，每个元素包含 {'chunk_id': str, 'new_content': str}
            regenerate_embedding: 是否重新生成嵌入向量
            
        Returns:
            批量更新结果统计
            
        Example:
            # 批量更新多个图像chunk
            updates = [
                {'chunk_id': 'doc_image_1', 'new_content': '新的图像描述1'},
                {'chunk_id': 'doc_image_2', 'new_content': '新的图像描述2'}
            ]
            result = await engine.batch_update_image_chunks_content(updates)
        """
        await self._ensure_initialized()
        
        total_updates = len(updates)
        successful_updates = []
        failed_updates = []
        
        logger.info(f"Starting batch update of {total_updates} image chunks")
        
        for i, update in enumerate(updates):
            chunk_id = update.get('chunk_id')
            new_content = update.get('new_content')
            
            if not chunk_id or not new_content:
                failed_updates.append({
                    'chunk_id': chunk_id or 'unknown',
                    'error': 'Missing chunk_id or new_content'
                })
                continue
            
            try:
                with log_process(f"Updating chunk {i+1}/{total_updates}"):
                    result = await self.update_image_chunk_content(
                        chunk_id=chunk_id,
                        new_content=new_content,
                        regenerate_embedding=regenerate_embedding
                    )
                    
                    if result.get('success'):
                        successful_updates.append(result)
                    else:
                        failed_updates.append({
                            'chunk_id': chunk_id,
                            'error': result.get('error', 'Unknown error')
                        })
                        
            except Exception as e:
                failed_updates.append({
                    'chunk_id': chunk_id,
                    'error': str(e)
                })
        
        batch_result = {
            'total_requested': total_updates,
            'successful_count': len(successful_updates),
            'failed_count': len(failed_updates),
            'successful_updates': successful_updates,
            'failed_updates': failed_updates,
            'success_rate': len(successful_updates) / total_updates if total_updates > 0 else 0
        }
        
        logger.info(f"Batch update completed: {len(successful_updates)}/{total_updates} successful")
        return batch_result

    @log_step("Get Image Chunk Content")
    async def get_image_chunk_content(
        self,
        chunk_id: str,
        include_metadata: bool = False
    ) -> Dict[str, Any]:
        """获取图像chunk的当前content内容。
        
        Args:
            chunk_id: chunk ID
            include_metadata: 是否包含完整的元数据
            
        Returns:
            chunk内容信息
            
        Example:
            # 获取图像chunk内容
            content_info = await engine.get_image_chunk_content("doc_image_1")
            print(content_info['content'])
        """
        await self._ensure_initialized()
        
        try:
            # 从向量数据库获取chunk
            existing_doc = await self._vector_store.get_document(chunk_id)
            if not existing_doc:
                return {
                    'chunk_id': chunk_id,
                    'found': False,
                    'error': 'Chunk not found'
                }
            
            # 验证chunk类型
            chunk_type = existing_doc.metadata.get('chunk_type', '')
            if chunk_type != 'image':
                return {
                    'chunk_id': chunk_id,
                    'found': True,
                    'is_image_chunk': False,
                    'actual_chunk_type': chunk_type,
                    'error': f'Chunk is not an image type (current type: {chunk_type})'
                }
            
            result = {
                'chunk_id': chunk_id,
                'found': True,
                'is_image_chunk': True,
                'content': existing_doc.text,
                'content_length': len(existing_doc.text),
                'chunk_type': chunk_type,
                'embedding_type': existing_doc.metadata.get('embedding_type', 'unknown'),
                'has_original_content': existing_doc.metadata.get('has_original_content', False),
                'last_updated': existing_doc.metadata.get('content_updated_at'),
                'original_content_preserved': 'original_content' in existing_doc.metadata
            }
            
            if include_metadata:
                result['full_metadata'] = existing_doc.metadata
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get image chunk content: {e}")
            return {
                'chunk_id': chunk_id,
                'found': False,
                'error': str(e)
            }

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