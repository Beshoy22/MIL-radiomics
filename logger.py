"""
Centralized logging configuration for MIL-radiomics.

This module provides a standardized logging framework to replace print statements
throughout the codebase with proper logging levels and formatting.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import os


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for different log levels."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        """Format log record with colors if outputting to terminal."""
        log_message = super().format(record)

        # Only add colors if outputting to a terminal
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            return f"{color}{log_message}{reset}"

        return log_message


def setup_logger(
    name: str = 'mil_radiomics',
    level: str = 'INFO',
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.

    Args:
        name: Logger name (usually __name__ or module name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file. If None, only console logging
        console_output: Whether to output to console

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger('my_module', level='DEBUG')
        >>> logger.info('Processing started')
        >>> logger.warning('Missing optional parameter')
        >>> logger.error('Failed to load file', exc_info=True)
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    # Set logging level
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    logger.setLevel(level_map.get(level.upper(), logging.INFO))

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = ColoredFormatter(
        fmt='%(levelname)s - %(message)s'
    )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get an existing logger or create a new one with default configuration.

    Args:
        name: Logger name. If None, returns root MIL-radiomics logger

    Returns:
        Logger instance

    Example:
        >>> from logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info('Message from my module')
    """
    if name is None:
        name = 'mil_radiomics'

    logger = logging.getLogger(name)

    # If logger doesn't have handlers, set it up with defaults
    if not logger.handlers:
        # Get log level from environment variable or use INFO
        level = os.getenv('LOG_LEVEL', 'INFO')
        return setup_logger(name, level=level)

    return logger


def configure_logging_from_args(args):
    """
    Configure logging based on command-line arguments.

    Args:
        args: Argument namespace with 'verbose' and optional 'log_file' attributes

    Returns:
        Configured logger instance

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument('--verbose', action='store_true')
        >>> parser.add_argument('--log_file', type=str, default=None)
        >>> args = parser.parse_args()
        >>> logger = configure_logging_from_args(args)
    """
    level = 'DEBUG' if getattr(args, 'verbose', False) else 'INFO'
    log_file = getattr(args, 'log_file', None)

    return setup_logger('mil_radiomics', level=level, log_file=log_file)


# Module-level convenience instance
logger = get_logger('mil_radiomics')


if __name__ == '__main__':
    # Test the logging setup
    test_logger = setup_logger('test', level='DEBUG')

    test_logger.debug('This is a debug message')
    test_logger.info('This is an info message')
    test_logger.warning('This is a warning message')
    test_logger.error('This is an error message')
    test_logger.critical('This is a critical message')

    print("\n--- With file logging ---")
    file_logger = setup_logger('file_test', level='DEBUG', log_file='test.log')
    file_logger.info('This message goes to both console and file')
    print("Check test.log for the log file output")
