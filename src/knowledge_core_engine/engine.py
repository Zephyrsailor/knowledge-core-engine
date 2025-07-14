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
import logging

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

logger = logging.getLogger(__name__)


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
        llm_provider: str = "deepseek",
        embedding_provider: str = "dashscope", 
        persist_directory: str = "./data/knowledge_base",
        **kwargs
    ):
        """初始化知识引擎。
        
        Args:
            llm_provider: LLM提供商 (deepseek/qwen/openai)
            embedding_provider: 嵌入模型提供商 (dashscope/openai)
            persist_directory: 知识库存储路径
            **kwargs: 其他配置参数
        """
        # 自动从环境变量读取API密钥
        self.config = RAGConfig(
            llm_provider=llm_provider,
            llm_api_key=kwargs.get('llm_api_key') or os.getenv(f"{llm_provider.upper()}_API_KEY"),
            embedding_provider=embedding_provider,
            embedding_api_key=kwargs.get('embedding_api_key') or os.getenv(
                "DASHSCOPE_API_KEY" if embedding_provider == "dashscope" 
                else f"{embedding_provider.upper()}_API_KEY"
            ),
            vectordb_provider="chromadb",
            persist_directory=persist_directory,
            include_citations=kwargs.get('include_citations', True),
            **{k: v for k, v in kwargs.items() 
               if k not in ['llm_api_key', 'embedding_api_key', 'include_citations']}
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
        
        for file_path in files:
            try:
                # 解析文档
                parse_result = await self._parser.process(file_path)
                
                # 分块
                chunk_result = await self._chunker.process_parse_result(parse_result)
                
                # 如果启用元数据增强，对每个块进行增强
                if self._metadata_enhancer:
                    enhanced_chunks = []
                    for chunk in chunk_result.chunks:
                        try:
                            # 增强元数据（方法会就地修改chunk并返回）
                            enhanced_chunk = await self._metadata_enhancer.enhance_chunk(chunk)
                            enhanced_chunks.append(enhanced_chunk)
                        except Exception as e:
                            logger.warning(f"Failed to enhance chunk metadata: {e}")
                            enhanced_chunks.append(chunk)
                    chunk_result.chunks = enhanced_chunks
                
                # 嵌入和存储
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
                
                total_chunks += chunk_result.total_chunks
                
            except Exception as e:
                failed_files.append({
                    "file": str(file_path),
                    "error": str(e)
                })
        
        return {
            "total_files": total_files,
            "processed_files": total_files - len(failed_files),
            "failed_files": failed_files,
            "total_chunks": total_chunks
        }
    
    async def ask(
        self, 
        question: str,
        top_k: int = 5,
        return_details: bool = False,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
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
        
        # 检索
        contexts = await self._retriever.retrieve(question, top_k=top_k)
        
        # 如果启用重排序，对结果进行重排
        if self._reranker and contexts:
            contexts = await self._reranker.rerank(question, contexts, top_k=self.config.rerank_top_k)
        
        if not contexts:
            no_context_answer = "抱歉，我在知识库中没有找到相关信息。"
            if return_details:
                return {
                    "question": question,
                    "answer": no_context_answer,
                    "contexts": [],
                    "citations": []
                }
            return no_context_answer
        
        # 生成答案
        result = await self._generator.generate(question, contexts)
        
        if return_details:
            # 返回详细信息
            return {
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
        else:
            # 返回简单答案（包含引用）
            if result.citations:
                citations_text = "\n\n**引用来源：**\n"
                for cite in result.citations:
                    source = cite.document_title or "未知来源"
                    citations_text += f"[{cite.index}] {source}\n"
                return result.answer + citations_text
            
            return result.answer
    
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