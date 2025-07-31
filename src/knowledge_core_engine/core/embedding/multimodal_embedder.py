import base64
import dashscope
from PIL import Image
import io
from typing import List, Dict, Any, Optional

from .embedder import EmbeddingResult
from ...utils.logger import get_logger
from ...utils.config import get_settings

logger = get_logger(__name__)


class MultimodalEmbedder:
    """多模态嵌入器，支持文本和图像嵌入"""
    
    def __init__(self, api_key: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.dashscope_api_key
        if self.api_key:
            dashscope.api_key = self.api_key
        else:
            raise ValueError("DashScope API key is required for multimodal embedding")
    
    async def initialize(self):
        """初始化嵌入器"""
        pass
    
    async def generate_embeddings(self, contents: List[Dict[str, Any]]) -> List[EmbeddingResult]:
        """生成多模态嵌入向量"""
        embeddings = []
        
        for i, item in enumerate(contents):
            try:
                logger.debug(f"Processing item {i+1}/{len(contents)}, type: {item.get('type', 'unknown')}")
                
                if not isinstance(item, dict) or 'type' not in item:
                    logger.warning(f"Invalid item structure at index {i}")
                    embeddings.append(self._create_zero_embedding())
                    continue
                
                if item["type"] == "text":
                    embedding = await self._embed_text(item["content"])
                elif item["type"] == "image":
                    embedding = await self._embed_image(item["content"])
                else:
                    logger.warning(f"Unknown item type: {item['type']}")
                    embedding = self._create_zero_embedding()
                
                embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Error processing item {i+1}: {e}")
                embeddings.append(self._create_zero_embedding())
        
        return embeddings
    
    async def _embed_text(self, text: str) -> EmbeddingResult:
        """生成文本嵌入"""
        try:
            resp = dashscope.TextEmbedding.call(
                model="text-embedding-v3",
                input=text
            )
            
            if resp.status_code == 200 and hasattr(resp, 'output') and resp.output:
                if 'embeddings' in resp.output:
                    embedding = resp.output['embeddings'][0]['embedding']
                elif 'embedding' in resp.output:
                    embedding = resp.output['embedding']
                else:
                    logger.warning("Unexpected text embedding response format")
                    return self._create_zero_embedding()
                
                return EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    model="text-embedding-v3",
                    usage=getattr(resp, 'usage', {}),
                    metadata={"type": "text"}
                )
            else:
                logger.error(f"Text embedding failed: {resp.status_code}")
                return self._create_zero_embedding()
                
        except Exception as e:
            logger.error(f"Text embedding error: {e}")
            return self._create_zero_embedding()
    
    async def _embed_image(self, image_data: Dict[str, Any]) -> EmbeddingResult:
        """生成图像嵌入"""
        try:
            # 处理图像数据
            img_bytes = image_data["data"]
            img = Image.open(io.BytesIO(img_bytes))
            img = img.convert("RGB")
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # 尝试多种API调用格式
            resp = None
            formats = [
                [f"data:image/jpeg;base64,{img_str}"],
                f"data:image/jpeg;base64,{img_str}",
                {"image": f"data:image/jpeg;base64,{img_str}"}
            ]
            
            for fmt in formats:
                try:
                    resp = dashscope.MultiModalEmbedding.call(
                        model="multimodal-embedding-v1",
                        input=fmt
                    )
                    if resp.status_code == 200:
                        break
                except Exception as e:
                    logger.debug(f"Image embedding format failed: {e}")
                    continue
            
            if resp and resp.status_code == 200 and hasattr(resp, 'output') and resp.output:
                if 'embedding' in resp.output:
                    embedding = resp.output['embedding']
                elif 'embeddings' in resp.output:
                    emb_list = resp.output['embeddings']
                    if isinstance(emb_list, list) and len(emb_list) > 0:
                        embedding = emb_list[0]
                    else:
                        return self._create_zero_embedding()
                elif 'data' in resp.output:
                    embedding = resp.output['data']
                else:
                    logger.warning("Unexpected image embedding response format")
                    return self._create_zero_embedding()
                
                return EmbeddingResult(
                    text=f"Image from page {image_data.get('page', 'unknown')}",
                    embedding=embedding,
                    model="multimodal-embedding-v1",
                    usage=getattr(resp, 'usage', {}),
                    metadata={
                        "type": "image",
                        "page": image_data.get('page'),
                        "index": image_data.get('index')
                    }
                )
            else:
                logger.error(f"Image embedding failed: {resp.status_code if resp else 'No response'}")
                return self._create_zero_embedding()
                
        except Exception as e:
            logger.error(f"Image embedding error: {e}")
            return self._create_zero_embedding()
    
    def _create_zero_embedding(self) -> EmbeddingResult:
        """创建零向量嵌入作为占位符"""
        return EmbeddingResult(
            text="",
            embedding=[0.0] * 1536,  # 标准嵌入维度
            model="placeholder",
            usage={},
            metadata={"type": "placeholder"}
        )