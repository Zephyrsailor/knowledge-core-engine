"""
KnowledgeCore Engine - 下一代知识引擎

简单、强大、可扩展的RAG系统。

Quick Start:
    from knowledge_core_engine import KnowledgeEngine
    
    async def main():
        # 创建引擎
        engine = KnowledgeEngine()
        
        # 添加文档
        await engine.add("docs/")
        
        # 提问
        answer = await engine.ask("什么是RAG?")
        print(answer)
"""

# 自动加载环境变量
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    # 查找项目根目录的 .env 文件
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        # 可选：打印加载成功的提示（仅在DEBUG模式）
        if os.getenv("DEBUG", "").lower() == "true":
            print(f"✅ Loaded environment variables from {env_path}")
except ImportError:
    # 如果没有安装 python-dotenv，忽略
    pass

__version__ = "0.1.0"
__author__ = "Knowledge Core Team"

# 在导入其他模块之前设置日志系统
from .utils.logger import setup_logger

# 从环境变量读取日志级别（默认为INFO）
log_level = os.getenv("LOG_LEVEL", "INFO")

# 初始化全局日志系统
logger = setup_logger("knowledge_core_engine", log_level=log_level)

# 如果是DEBUG模式，显示当前配置
if log_level == "DEBUG":
    logger.debug(f"KnowledgeCore Engine initialized with LOG_LEVEL={log_level}")

from .engine import KnowledgeEngine, create_engine
from .core.config import RAGConfig

__all__ = [
    "__version__", 
    "__author__", 
    "KnowledgeEngine", 
    "create_engine", 
    "RAGConfig",
    "logger"
]