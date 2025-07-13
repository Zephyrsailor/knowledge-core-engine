"""Demo of advanced retrieval scenarios using enhanced metadata."""

from typing import List, Dict, Any, Set
from knowledge_core_engine.core.chunking.enhanced_chunker import EnhancedChunker


class AdvancedRetriever:
    """Demonstrates advanced retrieval patterns using enhanced metadata."""
    
    def __init__(self, chunks: List[Dict[str, Any]]):
        """Initialize with chunked documents.
        
        Args:
            chunks: List of chunks with enhanced metadata
        """
        self.chunks = chunks
        self.chunk_index = {c['chunk_id']: c for c in chunks}
        
    def parent_child_retrieval(self, query: str, expand_children: bool = True,
                              include_parent: bool = True) -> List[Dict[str, Any]]:
        """Retrieve chunks with their hierarchical context.
        
        Example: Query "RAG implementation steps"
        1. Find chunks matching "RAG implementation"
        2. Expand to include all child chunks (detailed steps)
        3. Optionally include parent chunk (overview)
        """
        # Step 1: Find matching chunks (simplified - in real system use vector search)
        matching_chunks = self._simple_search(query)
        
        result_chunks = set()
        
        for chunk in matching_chunks:
            chunk_id = chunk['chunk_id']
            result_chunks.add(chunk_id)
            
            hierarchy = chunk.get('hierarchy', {})
            
            # Expand to children
            if expand_children and hierarchy.get('child_chunk_ids'):
                for child_id in hierarchy['child_chunk_ids']:
                    if child_id in self.chunk_index:
                        result_chunks.add(child_id)
                        # Recursively get grandchildren
                        self._get_all_descendants(child_id, result_chunks)
            
            # Include parent for context
            if include_parent and hierarchy.get('parent_chunk_id'):
                parent_id = hierarchy['parent_chunk_id']
                if parent_id in self.chunk_index:
                    result_chunks.add(parent_id)
        
        # Return chunks in hierarchical order
        return self._sort_by_hierarchy(list(result_chunks))
    
    def multi_hop_retrieval(self, query: str, max_hops: int = 3) -> List[Dict[str, Any]]:
        """Retrieve chunks following semantic relationships.
        
        Example: Query "How to optimize RAG performance"
        1. Find "RAG performance" chunks
        2. Follow prerequisites to understand basics
        3. Follow references to find optimization techniques
        4. Build complete knowledge path
        """
        # Initial retrieval
        start_chunks = self._simple_search(query)
        
        visited = set()
        to_visit = [(c['chunk_id'], 0) for c in start_chunks]  # (chunk_id, hop_count)
        result_chunks = []
        
        while to_visit:
            chunk_id, hop_count = to_visit.pop(0)
            
            if chunk_id in visited or hop_count > max_hops:
                continue
                
            visited.add(chunk_id)
            chunk = self.chunk_index.get(chunk_id)
            
            if not chunk:
                continue
                
            result_chunks.append({
                **chunk,
                'retrieval_hop': hop_count,
                'retrieval_path': self._get_path_to_chunk(chunk_id, start_chunks)
            })
            
            # Follow semantic relationships
            semantics = chunk.get('semantics', {})
            context = chunk.get('context', {})
            
            # Prerequisites (go backwards)
            for prereq_id in semantics.get('prerequisites', []):
                if prereq_id not in visited:
                    to_visit.append((prereq_id, hop_count + 1))
            
            # Forward references
            for ref_id in context.get('references_to', []):
                if ref_id not in visited:
                    to_visit.append((ref_id, hop_count + 1))
            
            # Related chunks (same concepts)
            for related_id in semantics.get('related_chunks', []):
                if related_id not in visited:
                    to_visit.append((related_id, hop_count + 1))
        
        return sorted(result_chunks, key=lambda x: (x['retrieval_hop'], x['importance_score']))
    
    def context_aware_retrieval(self, query: str, context_window: int = 2) -> List[Dict[str, Any]]:
        """Retrieve chunks with surrounding context.
        
        Example: Query about specific implementation detail
        1. Find exact match
        2. Include previous chunks for setup/background
        3. Include next chunks for consequences/examples
        """
        # Find core chunks
        core_chunks = self._simple_search(query)
        
        result_chunks = {}
        
        for chunk in core_chunks:
            chunk_id = chunk['chunk_id']
            result_chunks[chunk_id] = {**chunk, 'is_core': True}
            
            context = chunk.get('context', {})
            
            # Add previous context
            for prev_id in context.get('prev_chunks', [])[:context_window]:
                if prev_id in self.chunk_index and prev_id not in result_chunks:
                    result_chunks[prev_id] = {
                        **self.chunk_index[prev_id],
                        'is_context': True,
                        'context_type': 'previous'
                    }
            
            # Add next context
            for next_id in context.get('next_chunks', [])[:context_window]:
                if next_id in self.chunk_index and next_id not in result_chunks:
                    result_chunks[next_id] = {
                        **self.chunk_index[next_id],
                        'is_context': True,
                        'context_type': 'next'
                    }
        
        # Sort by document order
        return sorted(result_chunks.values(), 
                     key=lambda x: x.get('chunk_index', float('inf')))
    
    def concept_graph_traversal(self, start_concept: str, end_concept: str,
                               max_depth: int = 5) -> List[Dict[str, Any]]:
        """Find path between concepts through the knowledge graph.
        
        Example: From "向量数据库" to "RAG优化"
        1. Find chunks defining start concept
        2. Follow concept relationships
        3. Find path to end concept
        """
        # Find starting chunks
        start_chunks = [c for c in self.chunks 
                       if start_concept in c.get('semantics', {}).get('key_concepts', [])]
        
        if not start_chunks:
            return []
        
        # BFS to find path
        paths = []
        visited = set()
        queue = [(c['chunk_id'], [c['chunk_id']]) for c in start_chunks]
        
        while queue and len(paths) < 3:  # Find up to 3 different paths
            chunk_id, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
                
            chunk = self.chunk_index.get(chunk_id)
            if not chunk:
                continue
                
            # Check if we reached the target
            concepts = chunk.get('semantics', {}).get('key_concepts', [])
            if end_concept in concepts:
                paths.append(path)
                continue
            
            # Explore connections
            visited.add(chunk_id)
            
            # Follow related chunks
            for related_id in chunk.get('semantics', {}).get('related_chunks', []):
                if related_id not in visited:
                    queue.append((related_id, path + [related_id]))
        
        # Return chunks from shortest path
        if paths:
            shortest_path = min(paths, key=len)
            return [self.chunk_index[chunk_id] for chunk_id in shortest_path]
        
        return []
    
    def _simple_search(self, query: str) -> List[Dict[str, Any]]:
        """Simple keyword search (in real system, use vector search)."""
        query_lower = query.lower()
        results = []
        
        for chunk in self.chunks:
            content_lower = chunk.get('content', '').lower()
            if query_lower in content_lower:
                # Simple relevance score based on frequency
                score = content_lower.count(query_lower) / len(content_lower.split())
                results.append({
                    **chunk,
                    'relevance_score': score
                })
        
        return sorted(results, key=lambda x: x['relevance_score'], reverse=True)[:5]
    
    def _get_all_descendants(self, chunk_id: str, result_set: Set[str]):
        """Recursively get all descendant chunks."""
        chunk = self.chunk_index.get(chunk_id)
        if not chunk:
            return
            
        for child_id in chunk.get('hierarchy', {}).get('child_chunk_ids', []):
            if child_id not in result_set:
                result_set.add(child_id)
                self._get_all_descendants(child_id, result_set)
    
    def _sort_by_hierarchy(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Sort chunks by their hierarchical position."""
        chunks = [self.chunk_index[cid] for cid in chunk_ids if cid in self.chunk_index]
        
        # Sort by depth and position
        return sorted(chunks, key=lambda x: (
            len(x.get('hierarchy', {}).get('hierarchy_levels', [])),
            x.get('chunk_index', 0)
        ))
    
    def _get_path_to_chunk(self, chunk_id: str, start_chunks: List[Dict[str, Any]]) -> List[str]:
        """Get the path from start chunks to target chunk (simplified)."""
        # In a real implementation, this would trace the actual retrieval path
        return [c['chunk_id'] for c in start_chunks] + [chunk_id]


# Demo usage
if __name__ == "__main__":
    # Example document
    sample_doc = """# RAG系统设计指南

## 1. 简介
RAG（Retrieval Augmented Generation）是一种结合检索和生成的技术。

## 2. 核心组件

### 2.1 向量数据库
向量数据库是RAG系统的核心组件，用于存储和检索文档向量。

### 2.2 嵌入模型
嵌入模型将文本转换为向量表示。

## 3. 实现步骤

### 3.1 数据准备
首先需要准备高质量的文档数据。

### 3.2 向量化
使用嵌入模型对文档进行向量化。

### 3.3 索引构建
将向量存储到向量数据库中。

## 4. 优化策略

### 4.1 检索优化
- 使用混合检索
- 实现重排序

### 4.2 生成优化
- 提示工程
- 上下文管理
"""
    
    # Create enhanced chunks
    chunker = EnhancedChunker(chunk_size=200, context_window=2)
    result = chunker.chunk(sample_doc)
    
    # Convert to format for retriever
    chunks_data = []
    for chunk in result.chunks:
        chunks_data.append({
            'chunk_id': chunk.metadata['chunk_id'],
            'content': chunk.content,
            'chunk_index': chunk.metadata['chunk_index'],
            'importance_score': chunk.metadata['importance_score'],
            **chunk.metadata
        })
    
    # Create retriever
    retriever = AdvancedRetriever(chunks_data)
    
    print("=== Parent-Child Retrieval Demo ===")
    results = retriever.parent_child_retrieval("实现步骤")
    for r in results:
        print(f"- {r.get('hierarchy', {}).get('hierarchy_path', [])} "
              f"(Children: {len(r.get('hierarchy', {}).get('child_chunk_ids', []))})")
    
    print("\n=== Multi-Hop Retrieval Demo ===")
    results = retriever.multi_hop_retrieval("优化策略", max_hops=2)
    for r in results:
        print(f"- Hop {r['retrieval_hop']}: {r['content'][:50]}...")
    
    print("\n=== Context-Aware Retrieval Demo ===")
    results = retriever.context_aware_retrieval("向量数据库", context_window=1)
    for r in results:
        core_marker = "[CORE]" if r.get('is_core') else "[CONTEXT]"
        print(f"- {core_marker} {r['content'][:50]}...")