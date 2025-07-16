"""Hierarchical retrieval strategy that uses parent-child relationships."""

import json
import logging
from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass

from .retriever import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical retrieval."""
    include_parent_context: bool = True
    include_siblings: bool = False
    include_children: bool = False
    parent_weight: float = 0.8  # Weight for parent chunks
    sibling_weight: float = 0.6  # Weight for sibling chunks
    child_weight: float = 0.7  # Weight for child chunks
    max_hierarchy_depth: int = 3  # Maximum depth to traverse


class HierarchicalRetriever:
    """Retriever that uses hierarchical relationships between chunks."""
    
    def __init__(self, config: HierarchicalConfig = None):
        """Initialize hierarchical retriever.
        
        Args:
            config: Hierarchical retrieval configuration
        """
        self.config = config or HierarchicalConfig()
        
    async def enhance_with_hierarchy(
        self,
        initial_results: List[RetrievalResult],
        vector_store,
        max_additional: int = 5
    ) -> List[RetrievalResult]:
        """Enhance retrieval results with hierarchically related chunks.
        
        Args:
            initial_results: Initial retrieval results
            vector_store: Vector store instance to fetch additional chunks
            max_additional: Maximum additional chunks to add
            
        Returns:
            Enhanced list of retrieval results
        """
        if not initial_results:
            return initial_results
            
        logger.info(f"Enhancing {len(initial_results)} results with hierarchy")
            
        # Track all chunks to avoid duplicates
        seen_chunk_ids = {r.chunk_id for r in initial_results}
        additional_results = []
        
        # Process each initial result
        for result in initial_results:
            hierarchy_info = self._parse_hierarchy_metadata(result.metadata)
            if not hierarchy_info:
                continue
                
            # Get related chunks based on configuration
            related_chunks = await self._get_related_chunks(
                hierarchy_info,
                vector_store,
                seen_chunk_ids
            )
            
            additional_results.extend(related_chunks)
            
            # Update seen set
            seen_chunk_ids.update(r.chunk_id for r in related_chunks)
            
            # Check if we've reached the limit
            if len(additional_results) >= max_additional:
                break
        
        # Combine and re-rank results
        all_results = initial_results + additional_results[:max_additional]
        return self._rerank_by_hierarchy(all_results)
    
    def _parse_hierarchy_metadata(self, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse hierarchy information from metadata.
        
        Args:
            metadata: Chunk metadata
            
        Returns:
            Parsed hierarchy information or None
        """
        hierarchy_data = metadata.get('hierarchy')
        if not hierarchy_data:
            return None
            
        # Handle both dict and JSON string formats
        if isinstance(hierarchy_data, str):
            try:
                return json.loads(hierarchy_data)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse hierarchy JSON: {hierarchy_data}")
                return None
        elif isinstance(hierarchy_data, dict):
            return hierarchy_data
        else:
            return None
    
    async def _get_related_chunks(
        self,
        hierarchy_info: Dict[str, Any],
        vector_store,
        seen_chunk_ids: Set[str]
    ) -> List[RetrievalResult]:
        """Get hierarchically related chunks.
        
        Args:
            hierarchy_info: Hierarchy information for a chunk
            vector_store: Vector store to fetch chunks
            seen_chunk_ids: Set of already seen chunk IDs
            
        Returns:
            List of related chunks
        """
        related_results = []
        
        # Get parent chunk if configured
        if self.config.include_parent_context and hierarchy_info.get('parent_chunk_id'):
            parent_id = hierarchy_info['parent_chunk_id']
            if parent_id not in seen_chunk_ids:
                parent_doc = await vector_store.get_document(parent_id)
                if parent_doc:
                    related_results.append(self._doc_to_result(
                        parent_doc,
                        score_multiplier=self.config.parent_weight,
                        relation_type='parent'
                    ))
        
        # Get sibling chunks if configured
        if self.config.include_siblings:
            for sibling_id in hierarchy_info.get('sibling_chunks', []):
                if sibling_id not in seen_chunk_ids:
                    sibling_doc = await vector_store.get_document(sibling_id)
                    if sibling_doc:
                        related_results.append(self._doc_to_result(
                            sibling_doc,
                            score_multiplier=self.config.sibling_weight,
                            relation_type='sibling'
                        ))
        
        # Get child chunks if configured
        if self.config.include_children:
            for child_id in hierarchy_info.get('child_chunk_ids', []):
                if child_id not in seen_chunk_ids:
                    child_doc = await vector_store.get_document(child_id)
                    if child_doc:
                        related_results.append(self._doc_to_result(
                            child_doc,
                            score_multiplier=self.config.child_weight,
                            relation_type='child'
                        ))
        
        return related_results
    
    def _doc_to_result(
        self,
        doc: Any,
        score_multiplier: float = 1.0,
        relation_type: str = 'related'
    ) -> RetrievalResult:
        """Convert a document to a retrieval result.
        
        Args:
            doc: Document from vector store
            score_multiplier: Score multiplier based on relation type
            relation_type: Type of hierarchical relation
            
        Returns:
            RetrievalResult instance
        """
        # Adjust score based on relation type
        base_score = getattr(doc, 'score', 0.5)
        adjusted_score = base_score * score_multiplier
        
        # Add relation information to metadata
        metadata = getattr(doc, 'metadata', {}).copy()
        metadata['hierarchical_relation'] = relation_type
        metadata['score_adjustment'] = score_multiplier
        
        return RetrievalResult(
            chunk_id=doc.id,
            content=doc.text,
            score=adjusted_score,
            metadata=metadata
        )
    
    def _rerank_by_hierarchy(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Re-rank results considering hierarchical relationships.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Re-ranked results
        """
        # Group by hierarchy path to boost related content
        hierarchy_groups = {}
        
        for result in results:
            hierarchy_info = self._parse_hierarchy_metadata(result.metadata)
            if hierarchy_info and 'hierarchy_path' in hierarchy_info:
                path_key = '/'.join(hierarchy_info['hierarchy_path'][:2])  # Use top 2 levels
                if path_key not in hierarchy_groups:
                    hierarchy_groups[path_key] = []
                hierarchy_groups[path_key].append(result)
        
        # Boost scores for chunks in the same hierarchy
        for group in hierarchy_groups.values():
            if len(group) > 1:
                # Small boost for being in the same section
                for result in group:
                    result.score = min(result.score * 1.1, 1.0)
        
        # Sort by final score
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def adjust_scores_by_query_type(
        self,
        results: List[RetrievalResult],
        query: str
    ) -> List[RetrievalResult]:
        """Adjust scores based on query type and hierarchical level.
        
        Args:
            results: List of retrieval results
            query: Original query
            
        Returns:
            Results with adjusted scores
        """
        # Detect if query is asking for overview/general information
        overview_keywords = ['什么是', '介绍', '概述', 'what is', 'overview', 'introduction']
        is_overview_query = any(keyword in query.lower() for keyword in overview_keywords)
        
        # Detect if query is asking for specific details
        detail_keywords = ['如何', '怎么', '具体', '详细', 'how to', 'specific', 'detail']
        is_detail_query = any(keyword in query.lower() for keyword in detail_keywords)
        
        for result in results:
            hierarchy_info = self._parse_hierarchy_metadata(result.metadata)
            if not hierarchy_info:
                continue
                
            depth = hierarchy_info.get('depth', 0)
            
            # Adjust scores based on query type and chunk depth
            if is_overview_query and depth <= 2:
                # Boost parent/high-level chunks for overview queries
                result.score = min(result.score * 1.2, 1.0)
            elif is_detail_query and depth >= 2:
                # Boost child/detailed chunks for detail queries
                result.score = min(result.score * 1.2, 1.0)
            
            # Add query type to metadata for transparency
            result.metadata['query_type'] = (
                'overview' if is_overview_query else
                'detail' if is_detail_query else
                'general'
            )
        
        return results