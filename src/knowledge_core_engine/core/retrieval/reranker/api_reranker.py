"""API-based reranker implementations."""

import logging
from typing import List, Dict, Any, Optional
import os

from .base import BaseReranker, RerankResult

logger = logging.getLogger(__name__)


class APIReranker(BaseReranker):
    """Reranker using external API services."""
    
    SUPPORTED_PROVIDERS = {
        "dashscope": {
            "endpoint": "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
            "model": "gte-rerank-v2",
            "max_documents": 500,
            "max_tokens_per_doc": 4000,
            "max_total_tokens": 30000
        },
        "cohere": {
            "endpoint": "https://api.cohere.ai/v1/rerank",
            "model": "rerank-english-v2.0",
            "max_documents": 1000
        },
        "jina": {
            "endpoint": "https://api.jina.ai/v1/rerank",
            "model": "jina-reranker-v1-base-en",
            "max_documents": 100
        }
    }
    
    def __init__(
        self,
        provider: str = "dashscope",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 30
    ):
        """Initialize API reranker.
        
        Args:
            provider: API provider name
            api_key: API key (if not provided, will look for env var)
            model: Model name (if not provided, uses default for provider)
            timeout: Request timeout in seconds
        """
        super().__init__()
        
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider: {provider}. "
                f"Supported providers: {list(self.SUPPORTED_PROVIDERS.keys())}"
            )
        
        self.provider = provider
        self.provider_config = self.SUPPORTED_PROVIDERS[provider]
        self.model = model or self.provider_config["model"]
        self.timeout = timeout
        
        # Get API key
        if api_key:
            self.api_key = api_key
        else:
            # Try to get from environment with KCE_ prefix
            env_var_map = {
                "dashscope": "KCE_DASHSCOPE_API_KEY",
                "cohere": "KCE_COHERE_API_KEY",
                "jina": "KCE_JINA_API_KEY"
            }
            env_var = env_var_map.get(provider)
            # Try KCE_ prefix first, then fallback to original
            if env_var:
                self.api_key = os.getenv(env_var) or os.getenv(env_var.replace("KCE_", ""))
            
        self._session = None
    
    async def _initialize(self) -> None:
        """Initialize API client."""
        import aiohttp
        
        if not self.api_key:
            raise ValueError(
                f"API key not provided for {self.provider}. "
                f"Please set it in config or environment variable."
            )
        
        # Create aiohttp session
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        
        logger.info(f"Initialized {self.provider} API reranker with model {self.model}")
    
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_documents: bool = True
    ) -> List[RerankResult]:
        """Rerank documents using API."""
        if not self._initialized:
            await self.initialize()
        
        if not documents:
            return []
        
        # Call provider-specific method
        if self.provider == "dashscope":
            results = await self._rerank_dashscope(query, documents, top_k)
        elif self.provider == "cohere":
            results = await self._rerank_cohere(query, documents, top_k)
        elif self.provider == "jina":
            results = await self._rerank_jina(query, documents, top_k)
        else:
            raise RuntimeError(f"Unknown provider: {self.provider}")
        
        # Process results
        if not return_documents:
            for result in results:
                result.document = ""
        
        return results
    
    async def _rerank_dashscope(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """Rerank using DashScope API."""
        import dashscope
        from http import HTTPStatus
        
        try:
            # Use DashScope SDK
            response = dashscope.TextReRank.call(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_k or len(documents),
                return_documents=True,
                api_key=self.api_key
            )
            
            if response.status_code != HTTPStatus.OK:
                raise RuntimeError(f"DashScope API error: {response}")
            
            # Parse results
            results = []
            for item in response.output.results:
                results.append(RerankResult(
                    document=item.document.text,
                    score=item.relevance_score,
                    index=item.index,
                    metadata={
                        "model": self.model,
                        "provider": self.provider
                    }
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"DashScope rerank error: {e}")
            # Fallback to HTTP request if SDK fails
            return await self._rerank_dashscope_http(query, documents, top_k)
    
    async def _rerank_dashscope_http(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """Rerank using DashScope HTTP API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": documents
            },
            "parameters": {
                "return_documents": True,
                "top_n": top_k or len(documents)
            }
        }
        
        async with self._session.post(
            self.provider_config["endpoint"],
            headers=headers,
            json=data
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"DashScope API error: {response.status} - {error_text}")
            
            result = await response.json()
            
            # Parse results
            results = []
            for item in result["output"]["results"]:
                results.append(RerankResult(
                    document=item["document"]["text"],
                    score=item["relevance_score"],
                    index=item["index"],
                    metadata={
                        "model": self.model,
                        "provider": self.provider
                    }
                ))
            
            return results
    
    async def _rerank_cohere(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """Rerank using Cohere API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_k or len(documents),
            "return_documents": True
        }
        
        async with self._session.post(
            self.provider_config["endpoint"],
            headers=headers,
            json=data
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Cohere API error: {response.status} - {error_text}")
            
            result = await response.json()
            
            # Parse results
            results = []
            for item in result["results"]:
                results.append(RerankResult(
                    document=documents[item["index"]],
                    score=item["relevance_score"],
                    index=item["index"],
                    metadata={
                        "model": self.model,
                        "provider": self.provider
                    }
                ))
            
            return results
    
    async def _rerank_jina(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """Rerank using Jina API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_k or len(documents)
        }
        
        async with self._session.post(
            self.provider_config["endpoint"],
            headers=headers,
            json=data
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Jina API error: {response.status} - {error_text}")
            
            result = await response.json()
            
            # Parse results
            results = []
            for item in result["results"]:
                results.append(RerankResult(
                    document=documents[item["index"]],
                    score=item["score"],
                    index=item["index"],
                    metadata={
                        "model": self.model,
                        "provider": self.provider
                    }
                ))
            
            return results
    
    async def _close(self) -> None:
        """Clean up resources."""
        if self._session:
            await self._session.close()
            self._session = None