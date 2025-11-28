"""
Professional Logging Configuration for LookBench
Fashion Image Retrieval Benchmark
"""

import sys
import logging
import logging.handlers
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import json
import time
import functools

# Color codes for console output
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'

# Custom formatter with colors
class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Colors.GRAY,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.MAGENTA,
    }

    def format(self, record):
        record.timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        level_color = self.COLORS.get(record.levelno, Colors.WHITE)
        record.levelname_colored = f"{level_color}{record.levelname}{Colors.RESET}"
        record.module_info = f"{Colors.CYAN}{record.name}{Colors.RESET}"

        if hasattr(record, 'structured_data'):
            data_str = json.dumps(record.structured_data, indent=2, ensure_ascii=False)
            record.msg = f"{record.msg}\n{Colors.BLUE}Data:{Colors.RESET} {data_str}"

        formatted = super().format(record)

        if hasattr(record, 'performance_info'):
            perf_str = f"{Colors.MAGENTA}Performance:{Colors.RESET} {record.performance_info}"
            formatted = f"{formatted}\n{perf_str}"

        return formatted

# Performance tracking decorator
def log_performance(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            duration = time.time() - start_time
            logger = get_logger(func.__module__)
            performance_info = {
                'function': func.__name__,
                'duration_ms': round(duration * 1000, 2),
                'success': success,
                'error': error
            }

            if duration > 1.0:
                logger.warning(f"Slow operation detected",
                             extra={'performance_info': performance_info})
            else:
                logger.debug(f"Operation completed",
                           extra={'performance_info': performance_info})

        return result
    return wrapper

# Context manager for timing operations
class TimerContext:
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger or get_logger(__name__)
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting operation: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            if exc_type:
                self.logger.error(f"Operation failed: {self.operation_name} (took {duration:.2f}s)")
            else:
                self.logger.info(f"Operation completed: {self.operation_name} (took {duration:.2f}s)")

# Global logger instance
_root_logger = None

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
    structured_logging: bool = False,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    suppress_external: bool = True
) -> None:
    """Setup logging configuration"""
    global _root_logger

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    console_formatter = ColoredFormatter(
        fmt='%(timestamp)s - %(module_info)s - %(levelname_colored)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    if suppress_external:
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('torch').setLevel(logging.WARNING)
        logging.getLogger('transformers').setLevel(logging.WARNING)
        logging.getLogger('huggingface_hub').setLevel(logging.WARNING)

    _root_logger = root_logger

def configure_logging(config: Dict[str, Any]) -> None:
    """Configure logging from config dictionary"""
    logging_config = config.get('logging', {})

    setup_logging(
        level=getattr(logging, logging_config.get('level', 'INFO')),
        log_file=logging_config.get('log_file'),
        console_output=logging_config.get('console_output', True),
        structured_logging=logging_config.get('structured_logging', False),
        max_bytes=logging_config.get('max_bytes', 10 * 1024 * 1024),
        backup_count=logging_config.get('backup_count', 5),
        suppress_external=logging_config.get('suppress_external', True)
    )

def get_logger(name: str) -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name)

def log_structured(logger: logging.Logger, level: int, message: str, **data):
    """Log structured data"""
    logger.log(level, message, extra={'structured_data': data})

def log_performance_info(logger: logging.Logger, level: int, message: str, **performance_data):
    """Log performance information"""
    logger.log(level, message, extra={'performance_info': performance_data})

def log_error_with_context(logger: logging.Logger, message: str, error: Exception, **context):
    """Log error with context"""
    logger.error(f"{message}: {str(error)}",
                extra={'structured_data': context}, exc_info=True)

# Initialize default logging if not already configured
if _root_logger is None:
    setup_logging()

