"""Main retriever implementation with multiple strategies."""

import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime

from ..config import RAGConfig
from ..embedding.embedder import TextEmbedder
from ..embedding.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Supported retrieval strategies."""
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = None
    rerank_score: Optional[float] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def final_score(self) -> float:
        """Get final score (prefer rerank score if available)."""
        return self.rerank_score if self.rerank_score is not None else self.score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "rerank_score": self.rerank_score,
            "final_score": self.final_score,
            "metadata": self.metadata,
            "timestamp": datetime.now().isoformat()
        }


class Retriever:
    """Main retriever with support for multiple strategies."""
    
    def __init__(self, config: RAGConfig):
        """Initialize retriever with configuration.
        
        Args:
            config: RAG configuration object
        """
        self.config = config
        self._embedder = None
        self._vector_store = None
        self._bm25_index = None
        self._reranker = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize retrieval components."""
        if self._initialized:
            return
        
        # Initialize embedder and vector store
        self._embedder = TextEmbedder(self.config)
        self._vector_store = VectorStore(self.config)
        
        await self._embedder.initialize()
        await self._vector_store.initialize()
        
        # Initialize BM25 if needed
        if self.config.retrieval_strategy in ["bm25", "hybrid"]:
            await self._initialize_bm25()
        
        # Initialize hierarchical retriever if configured
        self._hierarchical_retriever = None
        if self.config.enable_hierarchical_chunking:
            logger.info(f"Hierarchical chunking enabled, initializing hierarchical retriever")
            await self._initialize_hierarchical_retriever()
        
        # Initialize reranker if enabled
        if self.config.enable_reranking:
            await self._initialize_reranker()
        
        self._initialized = True
        logger.info(f"Initialized retriever with strategy: {self.config.retrieval_strategy}")
    
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant chunks for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of retrieval results
        """
        await self.initialize()
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if top_k is None:
            top_k = self.config.retrieval_top_k
        
        # Apply query expansion if enabled
        if self.config.enable_query_expansion:
            expanded_queries = await self._expand_query(query)
            if len(expanded_queries) > 1:
                # 根据集成规则8.2.4：扩展的查询必须被独立使用
                logger.info(f"Using {len(expanded_queries)} expanded queries independently")
                results = await self._retrieve_with_expansion(
                    expanded_queries, top_k, filters
                )
                # Apply reranking on expanded results
                if self.config.enable_reranking:
                    results = await self._apply_reranking(query, results, top_k)
                return results
        
        # Route to appropriate strategy for single query
        strategy = RetrievalStrategy(self.config.retrieval_strategy)
        
        # Get more results if reranking is enabled (to rerank from a larger pool)
        retrieval_k = top_k * 3 if self.config.enable_reranking else top_k
        
        if strategy == RetrievalStrategy.VECTOR:
            results = await self._vector_retrieve(query, retrieval_k, filters)
        elif strategy == RetrievalStrategy.BM25:
            results = await self._bm25_retrieve(query, retrieval_k, filters)
        elif strategy == RetrievalStrategy.HYBRID:
            results = await self._hybrid_retrieve(query, retrieval_k, filters)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")
        
        # Apply hierarchical enhancement if configured
        if self._hierarchical_retriever and self.config.enable_hierarchical_chunking:
            logger.info(f"Applying hierarchical enhancement to {len(results)} results")
            results = await self._apply_hierarchical_enhancement(results, query)
        
        # Apply reranking if enabled
        if self.config.enable_reranking:
            results = await self._apply_reranking(query, results, top_k)
        else:
            # If no reranking, just return top_k
            results = results[:top_k]
        
        return results
    
    async def _vector_retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve using vector similarity."""
        # Embed query
        embedding_result = await self._embedder.embed_text(query)
        
        # Search vector store
        query_results = await self._vector_store.query(
            query_embedding=embedding_result.embedding,
            top_k=top_k,
            filter=filters
        )
        
        # Convert to RetrievalResult
        results = []
        for qr in query_results:
            results.append(RetrievalResult(
                chunk_id=qr.id,
                content=qr.text,
                score=qr.score,
                metadata=qr.metadata
            ))
        
        return results
    
    async def _bm25_retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve using BM25."""
        if not self._bm25_index:
            # 根据新规则，不返回空列表，而是抛出明确异常
            raise RuntimeError(
                "BM25 index not initialized. This should not happen if "
                "retrieval_strategy is 'bm25' or 'hybrid'. Check initialization."
            )
        
        # 执行BM25搜索
        bm25_results = await self._bm25_index.search(
            query=query,
            top_k=top_k,
            filter_metadata=filters
        )
        
        # 转换为RetrievalResult格式
        results = []
        for br in bm25_results:
            results.append(RetrievalResult(
                chunk_id=br.document_id,
                content=br.document,
                score=br.score,
                metadata=br.metadata or {}
            ))
        
        logger.info(f"BM25 retrieved {len(results)} documents for query: {query[:50]}...")
        
        # 根据新规则，添加监控
        if not results and self.config.extra_params.get("debug_mode", False):
            logger.warning(
                f"BM25 returned no results for query: {query}. "
                "This might indicate an empty index or configuration issue."
            )
        
        return results
    
    
    async def _hybrid_retrieve(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve using hybrid strategy (vector + BM25)."""
        # Get weights
        vector_weight = self.config.vector_weight
        bm25_weight = self.config.bm25_weight
        
        # Perform both searches in parallel
        vector_task = self._vector_retrieve(query, top_k * 2, filters)
        bm25_task = self._bm25_retrieve(query, top_k * 2, filters)
        
        vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)
        
        # 根据新规则添加监控
        if not bm25_results:
            logger.error(
                "INTEGRATION ERROR: Hybrid retrieval configured but "
                "BM25 returned no results. Check BM25 integration! "
                f"Vector results: {len(vector_results)}"
            )
            # 在调试模式下抛出异常
            if self.config.extra_params.get("debug_mode", False):
                raise RuntimeError(
                    "BM25 integration failure in hybrid mode. "
                    "BM25 must return results when documents are indexed."
                )
        
        logger.info(
            f"Hybrid retrieval - Vector: {len(vector_results)} results, "
            f"BM25: {len(bm25_results)} results"
        )
        
        # Combine results
        combined = self._combine_results(
            vector_results,
            bm25_results,
            vector_weight,
            bm25_weight
        )
        
        # Return top k
        return combined[:top_k]
    
    def _combine_results(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        vector_weight: float,
        bm25_weight: float
    ) -> List[RetrievalResult]:
        """Combine results from multiple sources."""
        # Track all unique documents
        combined_dict = {}
        
        # Add vector results
        for result in vector_results:
            combined_dict[result.chunk_id] = result
            result.metadata["vector_score"] = result.score
            result.metadata["fusion_method"] = "weighted"
            # Normalize vector score
            result.metadata["normalized_vector_score"] = self._normalize_score(result.score, "vector")
        
        # Add/merge BM25 results
        for result in bm25_results:
            # Normalize BM25 score
            normalized_bm25_score = self._normalize_score(result.score, "bm25")
            
            if result.chunk_id in combined_dict:
                # Merge scores
                existing = combined_dict[result.chunk_id]
                existing.metadata["bm25_score"] = result.score
                existing.metadata["normalized_bm25_score"] = normalized_bm25_score
                
                # Weighted combination using normalized scores
                # 确保权重和为1
                total_weight = vector_weight + bm25_weight
                norm_vector_weight = vector_weight / total_weight
                norm_bm25_weight = bm25_weight / total_weight
                
                existing.score = (
                    existing.metadata["normalized_vector_score"] * norm_vector_weight +
                    normalized_bm25_score * norm_bm25_weight
                )
            else:
                # New result from BM25
                result.metadata["bm25_score"] = result.score
                result.metadata["normalized_bm25_score"] = normalized_bm25_score
                result.metadata["fusion_method"] = "weighted"
                # 确保权重和为1
                total_weight = vector_weight + bm25_weight
                norm_bm25_weight = bm25_weight / total_weight
                result.score = normalized_bm25_score * norm_bm25_weight
                combined_dict[result.chunk_id] = result
        
        # Sort by combined score
        combined_results = list(combined_dict.values())
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        return combined_results
    
    async def _expand_query(self, query: str) -> List[str]:
        """Expand query for better retrieval.
        
        Args:
            query: Original query
            
        Returns:
            List of expanded queries (including original)
        """
        expanded = [query]  # Always include original
        
        if self.config.query_expansion_method == "llm":
            # Use LLM to generate related queries
            try:
                from ..generation.providers import create_llm_provider
                
                # Create a temporary LLM client for query expansion
                llm_provider = await create_llm_provider(self.config)
                
                # Build prompt for query expansion
                prompt = f"""为以下搜索查询生成{self.config.query_expansion_count - 1}个相关的搜索词或短语，
这些词应该：
1. 与原始查询相关但使用不同的表达方式
2. 包含同义词或相关概念
3. 帮助检索更多相关文档

原始查询：{query}

请直接返回相关查询，每行一个，不要编号或其他格式："""
                
                # Generate expanded queries
                messages = [{"role": "user", "content": prompt}]
                response = await llm_provider.generate(
                    messages=messages,
                    temperature=0.3,  # Lower temperature for more focused expansion
                    max_tokens=200
                )
                
                # Parse response
                if response and "content" in response:
                    lines = response["content"].strip().split("\n")
                    for line in lines[:self.config.query_expansion_count - 1]:
                        line = line.strip()
                        if line and not line[0].isdigit():  # Skip numbered items
                            expanded.append(line)
                
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}, using original query only")
        
        elif self.config.query_expansion_method == "rule_based":
            # Simple rule-based expansion
            # Add common synonyms and variations
            import jieba
            
            # Tokenize query
            tokens = list(jieba.cut(query))
            
            # Simple synonym mapping (in production, use a proper synonym dictionary)
            synonyms = {
                "什么": ["哪些", "何种"],
                "如何": ["怎么", "怎样"],
                "为什么": ["为何", "原因"],
                "RAG": ["检索增强生成", "Retrieval Augmented Generation"],
                "技术": ["方法", "技巧"],
                "优势": ["优点", "好处", "长处"],
                "问题": ["挑战", "困难", "难点"],
                "AI": ["人工智能", "Artificial Intelligence", "智能"],
                "人工智能": ["AI", "Artificial Intelligence", "智能"],
                "机器学习": ["ML", "Machine Learning"],
                "深度学习": ["DL", "Deep Learning", "深度神经网络"]
            }
            
            # Generate variations
            for token in tokens:
                if token in synonyms:
                    for synonym in synonyms[token][:self.config.query_expansion_count - 1]:
                        variation = query.replace(token, synonym)
                        if variation != query and variation not in expanded:
                            expanded.append(variation)
                            if len(expanded) >= self.config.query_expansion_count:
                                break
                
                if len(expanded) >= self.config.query_expansion_count:
                    break
        
        logger.info(f"Query expanded from '{query}' to {len(expanded)} variations")
        return expanded
    
    def _normalize_score(self, score: float, source: str) -> float:
        """Normalize scores from different sources."""
        if source == "vector":
            # Vector scores are typically 0-1
            return min(max(score, 0.0), 1.0)
        elif source == "bm25":
            # BM25 scores can be negative or positive
            # Use min-max normalization with fixed range
            # Typical BM25 scores range from -200 to 50
            min_score = -200.0
            max_score = 50.0
            
            # Clamp to range
            score = max(min_score, min(max_score, score))
            
            # Normalize to 0-1
            normalized = (score - min_score) / (max_score - min_score)
            return normalized
        else:
            return score
    
    async def _initialize_bm25(self):
        """Initialize BM25 index."""
        from .bm25.factory import create_bm25_retriever
        
        # 使用当前配置创建BM25检索器
        # factory会根据config中的bm25_provider等参数自动选择合适的实现
        self._bm25_index = create_bm25_retriever(self.config)
        
        if self._bm25_index:
            # 初始化
            await self._bm25_index.initialize()
            logger.info("BM25 index initialized successfully")
            
            # 检查向量存储中是否有文档但BM25索引为空
            # 这可能发生在文档已经被索引但BM25是后来启用的情况
            await self._sync_bm25_with_vector_store()
        else:
            # 根据新规则，不应该发生这种情况
            raise RuntimeError(
                "Failed to create BM25 index despite being configured for "
                f"retrieval_strategy='{self.config.retrieval_strategy}'"
            )
    
    async def _retrieve_with_expansion(
        self,
        expanded_queries: List[str],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """Retrieve using multiple expanded queries and merge results.
        
        根据集成规则，每个扩展查询都被独立使用，然后合并结果。
        
        Args:
            expanded_queries: List of expanded queries
            top_k: Number of final results to return
            filters: Optional metadata filters
            
        Returns:
            Merged retrieval results
        """
        strategy = RetrievalStrategy(self.config.retrieval_strategy)
        
        # 为每个扩展查询获取更多结果，因为后续要去重和合并
        per_query_k = min(top_k * 2, self.config.extra_params.get("expansion_per_query_k", top_k * 2))
        
        # 并行执行所有扩展查询
        tasks = []
        for exp_query in expanded_queries:
            logger.debug(f"Creating retrieval task for expanded query: '{exp_query}'")
            if strategy == RetrievalStrategy.VECTOR:
                task = self._vector_retrieve(exp_query, per_query_k, filters)
            elif strategy == RetrievalStrategy.BM25:
                task = self._bm25_retrieve(exp_query, per_query_k, filters)
            elif strategy == RetrievalStrategy.HYBRID:
                task = self._hybrid_retrieve(exp_query, per_query_k, filters)
            else:
                raise ValueError(f"Unknown retrieval strategy: {strategy}")
            tasks.append(task)
        
        # 执行所有查询
        all_results = await asyncio.gather(*tasks)
        
        # 日志记录每个查询的结果数
        for i, (query, results) in enumerate(zip(expanded_queries, all_results)):
            logger.debug(f"Expanded query {i+1} '{query}': {len(results)} results")
        
        # 合并结果
        merged = self._merge_expansion_results(all_results, expanded_queries)
        
        # 监控：确保查询扩展确实找到了不同的结果
        unique_ids = set(r.chunk_id for r in merged)
        logger.info(
            f"Query expansion: {len(expanded_queries)} queries -> "
            f"{sum(len(r) for r in all_results)} total results -> "
            f"{len(unique_ids)} unique results"
        )
        
        # 返回top_k结果
        return merged[:top_k]
    
    def _merge_expansion_results(
        self,
        all_results: List[List[RetrievalResult]],
        queries: List[str]
    ) -> List[RetrievalResult]:
        """Merge results from multiple expanded queries.
        
        使用投票机制：如果一个文档被多个查询检索到，提高其分数。
        
        Args:
            all_results: Results from each expanded query
            queries: The expanded queries (for logging)
            
        Returns:
            Merged and re-ranked results
        """
        # 使用字典跟踪每个文档
        doc_scores = {}  # chunk_id -> (result, appearance_count, sum_score)
        
        for query_idx, results in enumerate(all_results):
            for rank, result in enumerate(results):
                if result.chunk_id not in doc_scores:
                    # 首次出现
                    doc_scores[result.chunk_id] = {
                        "result": result,
                        "appearances": 1,
                        "sum_score": result.score,
                        "best_rank": rank,
                        "found_by_queries": [queries[query_idx]]
                    }
                else:
                    # 已经出现过，更新信息
                    info = doc_scores[result.chunk_id]
                    info["appearances"] += 1
                    info["sum_score"] += result.score
                    info["best_rank"] = min(info["best_rank"], rank)
                    info["found_by_queries"].append(queries[query_idx])
        
        # 计算最终分数
        merged_results = []
        for chunk_id, info in doc_scores.items():
            result = info["result"]
            
            # 组合分数：考虑出现次数和平均分数
            # 出现次数越多，说明与查询越相关
            # 限制最大boost为1.2，避免分数超过1
            appearance_boost = min(1.0 + (info["appearances"] - 1) * 0.1, 1.2)
            avg_score = info["sum_score"] / info["appearances"]
            final_score = avg_score * appearance_boost
            
            # 确保分数不超过1
            result.score = min(final_score, 1.0)
            result.metadata["expansion_appearances"] = info["appearances"]
            result.metadata["expansion_queries"] = info["found_by_queries"]
            
            merged_results.append(result)
        
        # 按分数排序
        merged_results.sort(key=lambda x: x.score, reverse=True)
        
        return merged_results
    
    async def batch_retrieve(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[List[RetrievalResult]]:
        """Retrieve for multiple queries.
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            filters: Optional metadata filters
            
        Returns:
            List of result lists, one per query
        """
        results = []
        
        for query in queries:
            try:
                query_results = await self.retrieve(query, top_k, filters)
                results.append(query_results)
            except Exception as e:
                logger.error(f"Error retrieving for query '{query}': {e}")
                if self.config.extra_params.get("batch_ignore_errors", False):
                    results.append([])
                else:
                    raise
        
        return results
    
    async def _initialize_hierarchical_retriever(self):
        """Initialize hierarchical retriever."""
        from .hierarchical_retriever import HierarchicalRetriever, HierarchicalConfig
        
        # Create config from RAGConfig extra_params
        hier_config = HierarchicalConfig(
            include_parent_context=self.config.extra_params.get('include_parent_context', True),
            include_siblings=self.config.extra_params.get('retrieve_siblings', False),
            include_children=self.config.extra_params.get('include_children', False)
        )
        
        self._hierarchical_retriever = HierarchicalRetriever(hier_config)
        logger.info("Hierarchical retriever initialized")
    
    async def _apply_hierarchical_enhancement(
        self,
        results: List[RetrievalResult],
        query: str
    ) -> List[RetrievalResult]:
        """Apply hierarchical enhancement to retrieval results.
        
        Args:
            results: Initial retrieval results
            query: Original query
            
        Returns:
            Enhanced results
        """
        if not self._hierarchical_retriever:
            return results
        
        # Enhance with hierarchically related chunks
        enhanced = await self._hierarchical_retriever.enhance_with_hierarchy(
            results,
            self._vector_store,
            max_additional=self.config.extra_params.get('max_hierarchical_additions', 5)
        )
        
        # Adjust scores based on query type
        enhanced = self._hierarchical_retriever.adjust_scores_by_query_type(
            enhanced,
            query
        )
        
        logger.info(
            f"Hierarchical enhancement: {len(results)} -> {len(enhanced)} results"
        )
        
        return enhanced
    
    async def _initialize_reranker(self):
        """Initialize reranker component."""
        from .reranker_wrapper import Reranker
        
        self._reranker = Reranker(self.config)
        await self._reranker.initialize()
        logger.info("Reranker initialized successfully")
    
    async def _apply_reranking(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """Apply reranking to retrieval results.
        
        Args:
            query: Original query
            results: Initial retrieval results
            top_k: Number of results to return after reranking
            
        Returns:
            Reranked results
        """
        if not self._reranker or not results:
            return results[:top_k]
        
        logger.info(f"Applying reranking to {len(results)} results")
        
        # Perform reranking
        reranked = await self._reranker.rerank(
            query=query,
            results=results,
            top_k=top_k
        )
        
        if reranked:
            logger.info(
                f"Reranking complete: {len(results)} -> {len(reranked)} results, "
                f"top score: {reranked[0].final_score:.3f}"
            )
        else:
            logger.info(
                f"Reranking complete: {len(results)} -> 0 results"
            )
        
        return reranked
    
    async def _sync_bm25_with_vector_store(self):
        """Sync BM25 index with vector store if needed."""
        if not self._bm25_index or not self._vector_store:
            return
        
        # Check if BM25 index is empty
        bm25_doc_count = len(self._bm25_index._documents) if hasattr(self._bm25_index, '_documents') else 0
        
        if bm25_doc_count == 0:
            # Check vector store for existing documents
            try:
                # Get all documents from vector store (using a large query limit)
                # This is a workaround since ChromaDB doesn't have a simple "get all" method
                dummy_embedding = [0.0] * self.config.embedding_dimensions
                all_results = await self._vector_store.query(
                    query_embedding=dummy_embedding,
                    top_k=10000  # Large number to get all documents
                )
                
                if all_results:
                    logger.info(f"Found {len(all_results)} documents in vector store, syncing to BM25 index...")
                    
                    # Add documents to BM25 index
                    documents = []
                    doc_ids = []
                    metadata_list = []
                    
                    for result in all_results:
                        documents.append(result.text)
                        doc_ids.append(result.id)
                        metadata_list.append(result.metadata)
                    
                    await self._bm25_index.add_documents(
                        documents=documents,
                        doc_ids=doc_ids,
                        metadata=metadata_list
                    )
                    
                    logger.info(f"Successfully synced {len(documents)} documents to BM25 index")
            except Exception as e:
                logger.warning(f"Failed to sync BM25 index with vector store: {e}")
                # Continue anyway - BM25 will be empty but system will still work