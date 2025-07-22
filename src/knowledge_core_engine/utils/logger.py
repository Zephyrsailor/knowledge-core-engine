"""Logging configuration for KnowledgeCore Engine."""

import sys
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from loguru import logger

from knowledge_core_engine.utils.config import get_settings


# 全局logger实例，供所有模块使用
_logger = logger


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
    
    # 设置第三方库的日志级别，减少噪音
    import logging
    logging.getLogger("xet-core").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    
    # Get log level and file from settings if not provided
    level = log_level or settings.log_level
    file_path = log_file or settings.log_file
    
    # Ensure log level is uppercase
    if isinstance(level, str):
        level = level.upper()
    
    # Define format based on log level
    if level == "DEBUG":
        # Detailed format for DEBUG mode
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level> | "
            "<dim>{extra}</dim>"
        )
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message} | "
            "{extra}"
        )
    else:
        # Simpler format for other levels
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan> - <level>{message}</level>"
        )
        file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}"
    
    # Console handler with color
    logger.add(
        sys.stdout,
        format=console_format,
        level=level,
        colorize=True,
        filter=lambda record: record["level"].name != "SUCCESS" or level == "DEBUG",
    )
    
    # File handler without color
    if file_path:
        logger.add(
            file_path,
            format=file_format,
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )
    
    # Add special SUCCESS level handler for process tracking
    if level == "DEBUG":
        separator = "=" * 80
        logger.add(
            sys.stdout,
            format=f"<yellow>{separator}</yellow>\n<green>{{time:HH:mm:ss.SSS}}</green> | <yellow>PROCESS</yellow> | <yellow>{{message}}</yellow>\n<yellow>{separator}</yellow>",
            level="SUCCESS",
            colorize=True,
            filter=lambda record: record["level"].name == "SUCCESS",
        )
    
    return logger.bind(name=module_name)


def get_logger(module_name: str) -> "logger":
    """
    Get a logger instance for a module.
    
    Args:
        module_name: Name of the module
        
    Returns:
        Logger instance bound to the module
    """
    return _logger.bind(name=module_name)


@contextmanager
def log_process(process_name: str, **extra_info):
    """
    Context manager for logging process execution with timing.
    
    Args:
        process_name: Name of the process
        **extra_info: Additional information to log
    """
    start_time = time.time()
    _logger.success(f"[START] {process_name}", **extra_info)
    
    try:
        yield
    except Exception as e:
        elapsed = time.time() - start_time
        _logger.error(
            f"[FAILED] {process_name} after {elapsed:.2f}s - {str(e)}",
            **extra_info
        )
        raise
    else:
        elapsed = time.time() - start_time
        _logger.success(f"[COMPLETED] {process_name} in {elapsed:.2f}s", **extra_info)


def log_step(step_name: str = None):
    """
    Decorator for logging function execution as a processing step.
    
    Args:
        step_name: Optional name for the step (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = step_name or func.__name__.replace('_', ' ').title()
            
            # Extract useful info from args/kwargs for logging
            extra_info = {}
            if args and hasattr(args[0], '__class__'):
                extra_info['class'] = args[0].__class__.__name__
            
            with log_process(name, **extra_info):
                result = func(*args, **kwargs)
                
                # Log result info in DEBUG mode
                if _logger._core.min_level <= 10:  # DEBUG level
                    if isinstance(result, (list, tuple)):
                        _logger.debug(f"Result: {len(result)} items")
                    elif isinstance(result, dict):
                        _logger.debug(f"Result keys: {list(result.keys())}")
                    elif hasattr(result, '__len__'):
                        _logger.debug(f"Result length: {len(result)}")
                
                return result
        
        return wrapper
    return decorator


def log_detailed(message: str, data: Any = None, **kwargs):
    """
    Log detailed information in DEBUG mode, simple message otherwise.
    
    Args:
        message: Log message
        data: Optional data to include in DEBUG mode
        **kwargs: Additional context
    """
    if _logger._core.min_level <= 10:  # DEBUG level
        if data is not None:
            _logger.debug(f"{message} | Data: {data}", **kwargs)
        else:
            _logger.debug(message, **kwargs)
    else:
        _logger.info(message)