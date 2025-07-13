"""Logging configuration for KnowledgeCore Engine."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from knowledge_core_engine.utils.config import get_settings


def setup_logger(
    module_name: str,
    log_file: Optional[Path] = None,
    log_level: Optional[str] = None,
) -> "logger":
    """
    Set up logger for a module.
    
    Args:
        module_name: Name of the module
        log_file: Path to log file (uses settings if not provided)
        log_level: Log level (uses settings if not provided)
        
    Returns:
        Configured logger instance
    """
    settings = get_settings()
    
    # Remove default handler
    logger.remove()
    
    # Get log level and file from settings if not provided
    level = log_level or settings.log_level
    file_path = log_file or settings.log_file
    
    # Console handler with color
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )
    
    # File handler without color
    if file_path:
        logger.add(
            file_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )
    
    return logger.bind(name=module_name)