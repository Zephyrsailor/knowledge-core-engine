"""HuggingFace-based reranker implementations."""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
import torch

from .base import BaseReranker, RerankResult

logger = logging.getLogger(__name__)


class HuggingFaceReranker(BaseReranker):
    """Reranker using HuggingFace models."""
    
    # Supported models and their configurations
    SUPPORTED_MODELS = {
        # BGE models
        "bge-reranker-v2-m3": {
            "model_id": "BAAI/bge-reranker-v2-m3",
            "type": "cross-encoder",
            "max_length": 512,
            "backend": "flagembedding"  # Preferred backend
        },
        "bge-reranker-large": {
            "model_id": "BAAI/bge-reranker-large",
            "type": "cross-encoder",
            "max_length": 512,
            "backend": "flagembedding"
        },
        "bge-reranker-base": {
            "model_id": "BAAI/bge-reranker-base",
            "type": "cross-encoder",
            "max_length": 512,
            "backend": "flagembedding"
        },
        
        # Qwen models
        "qwen3-reranker-8b": {
            "model_id": "Qwen/Qwen3-Reranker-8B",
            "type": "causal-lm",
            "max_length": 32768,
            "backend": "transformers"
        },
        "qwen3-reranker-4b": {
            "model_id": "Qwen/Qwen3-Reranker-4B",
            "type": "causal-lm",
            "max_length": 32768,
            "backend": "transformers"
        },
        "qwen3-reranker-0.6b": {
            "model_id": "Qwen/Qwen3-Reranker-0.6B",
            "type": "causal-lm",
            "max_length": 32768,
            "backend": "transformers"
        }
    }
    
    def __init__(self, model_name: str = "bge-reranker-v2-m3", use_fp16: bool = True, device: str = None):
        """Initialize HuggingFace reranker.
        
        Args:
            model_name: Name of the model to use
            use_fp16: Whether to use half precision (saves memory)
            device: Device to use (None = auto detect)
        """
        super().__init__()
        
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported models: {list(self.SUPPORTED_MODELS.keys())}"
            )
        
        self.model_name = model_name
        self.model_config = self.SUPPORTED_MODELS[model_name]
        self.use_fp16 = use_fp16
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self._model = None
        self._tokenizer = None
    
    async def _initialize(self) -> None:
        """Initialize the model based on its type."""
        model_type = self.model_config["type"]
        backend = self.model_config["backend"]
        
        try:
            if backend == "flagembedding" and model_type == "cross-encoder":
                await self._init_flagembedding()
            elif backend == "transformers" and model_type == "causal-lm":
                await self._init_transformers_causal()
            else:
                # Fallback to sentence-transformers for cross-encoders
                await self._init_sentence_transformers()
                
        except ImportError as e:
            logger.error(f"Failed to import required library: {e}")
            raise RuntimeError(
                f"Please install required dependencies for {self.model_name}. "
                f"Run: pip install knowledge-core-engine[reranker-hf]"
            )
        except Exception as e:
            logger.error(f"Failed to initialize model {self.model_name}: {e}")
            raise
    
    async def _init_flagembedding(self) -> None:
        """Initialize using FlagEmbedding (for BGE models)."""
        try:
            from FlagEmbedding import FlagReranker
            
            logger.info(f"Loading {self.model_name} with FlagEmbedding...")
            self._model = FlagReranker(
                self.model_config["model_id"],
                use_fp16=self.use_fp16,
                device=self.device
            )
            self._backend = "flagembedding"
            logger.info(f"Successfully loaded {self.model_name}")
            
        except ImportError:
            logger.warning("FlagEmbedding not available, falling back to sentence-transformers")
            await self._init_sentence_transformers()
    
    async def _init_sentence_transformers(self) -> None:
        """Initialize using sentence-transformers (fallback for BGE models)."""
        from sentence_transformers import CrossEncoder
        
        logger.info(f"Loading {self.model_name} with sentence-transformers...")
        self._model = CrossEncoder(
            self.model_config["model_id"],
            max_length=self.model_config["max_length"],
            device=self.device
        )
        self._backend = "sentence_transformers"
        logger.info(f"Successfully loaded {self.model_name}")
    
    async def _init_transformers_causal(self) -> None:
        """Initialize Qwen causal LM models."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading {self.model_name} with transformers...")
        
        model_id = self.model_config["model_id"]
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            padding_side='left'
        )
        
        # Load model
        dtype = torch.float16 if self.use_fp16 else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        if self.device == "cpu":
            self._model = self._model.to(self.device)
            
        self._model.eval()
        self._backend = "transformers_causal"
        logger.info(f"Successfully loaded {self.model_name}")
    
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_documents: bool = True
    ) -> List[RerankResult]:
        """Rerank documents based on relevance to query."""
        if not self._initialized:
            await self.initialize()
        
        if not documents:
            return []
        
        # Get scores based on backend
        if self._backend == "flagembedding":
            scores = await self._rerank_flagembedding(query, documents)
        elif self._backend == "sentence_transformers":
            scores = await self._rerank_sentence_transformers(query, documents)
        elif self._backend == "transformers_causal":
            scores = await self._rerank_transformers_causal(query, documents)
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")
        
        # Create results
        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            results.append(RerankResult(
                document=doc if return_documents else "",
                score=score,
                index=i,
                metadata={"model": self.model_name}
            ))
        
        # Sort by score (descending)
        results.sort(reverse=True)
        
        # Return top_k if specified
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    async def _rerank_flagembedding(self, query: str, documents: List[str]) -> List[float]:
        """Rerank using FlagEmbedding."""
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Compute scores
        scores = self._model.compute_score(pairs, normalize=True)
        
        # Ensure scores is a list
        if isinstance(scores, (int, float)):
            scores = [scores]
        elif isinstance(scores, np.ndarray):
            scores = scores.tolist()
            
        return scores
    
    async def _rerank_sentence_transformers(self, query: str, documents: List[str]) -> List[float]:
        """Rerank using sentence-transformers."""
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Predict scores
        scores = self._model.predict(pairs)
        
        # Convert to list and normalize
        scores = scores.tolist() if isinstance(scores, np.ndarray) else scores
        scores = self._normalize_scores(scores)
        
        return scores
    
    async def _rerank_transformers_causal(self, query: str, documents: List[str]) -> List[float]:
        """Rerank using Qwen causal LM models."""
        # Format inputs for Qwen reranker
        def format_instruction(query: str, doc: str, instruction: str = None):
            if instruction is None:
                instruction = 'Given a web search query, retrieve relevant passages that answer the query'
            return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
        
        # Prepare inputs
        pairs = [format_instruction(query, doc) for doc in documents]
        
        # Get special tokens
        prefix_tokens = self._tokenizer("<" + " " * 1600, return_tensors=None)['input_ids']
        suffix_tokens = self._tokenizer("\nAnswer: ", return_tensors=None)['input_ids']
        
        # Process in batches
        batch_size = 4
        all_scores = []
        
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                
                # Tokenize with truncation
                inputs = self._tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.model_config["max_length"] - len(prefix_tokens) - len(suffix_tokens),
                    return_tensors="pt"
                )
                
                # Add prefix and suffix tokens
                for j, input_ids in enumerate(inputs['input_ids']):
                    full_ids = prefix_tokens + input_ids.tolist() + suffix_tokens
                    inputs['input_ids'][j] = torch.tensor(full_ids)
                
                # Move to device
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
                
                # Get outputs
                outputs = self._model(**inputs)
                batch_scores = outputs.logits[:, -1, :]
                
                # Get yes/no token IDs
                yes_token_id = self._tokenizer.encode("yes", add_special_tokens=False)[0]
                no_token_id = self._tokenizer.encode("no", add_special_tokens=False)[0]
                
                # Calculate scores
                yes_logits = batch_scores[:, yes_token_id]
                no_logits = batch_scores[:, no_token_id]
                logits_matrix = torch.stack([no_logits, yes_logits], dim=1)
                scores = torch.softmax(logits_matrix, dim=-1)[:, 1]
                
                all_scores.extend(scores.cpu().numpy().tolist())
        
        return all_scores
    
    async def _close(self) -> None:
        """Clean up resources."""
        self._model = None
        self._tokenizer = None
        
        # Clear GPU cache if using CUDA
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()