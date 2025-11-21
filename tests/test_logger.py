"""
Tests for the logging module.
"""

import pytest
import logging
import tempfile
from pathlib import Path
from logger import setup_logger, get_logger, configure_logging_from_args


class TestSetupLogger:
    """Tests for setup_logger function."""

    def test_basic_logger_creation(self):
        """Test basic logger creation with default settings."""
        logger = setup_logger('test_logger', level='INFO')

        assert logger.name == 'test_logger'
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

    def test_logger_with_file_output(self, temp_dir):
        """Test logger creation with file output."""
        log_file = temp_dir / "test.log"

        logger = setup_logger(
            'test_file_logger',
            level='DEBUG',
            log_file=str(log_file)
        )

        # Log a message
        logger.info("Test message")

        # Check file was created
        assert log_file.exists()

        # Check log content
        content = log_file.read_text()
        assert "Test message" in content
        assert "INFO" in content

    def test_logger_levels(self):
        """Test different logger levels."""
        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }

        for level_str in levels:
            logger = setup_logger(f'test_{level_str}', level=level_str)
            assert logger.level == level_map[level_str]

    def test_logger_no_console_output(self, temp_dir):
        """Test logger without console output."""
        log_file = temp_dir / "no_console.log"

        logger = setup_logger(
            'test_no_console',
            level='INFO',
            log_file=str(log_file),
            console_output=False
        )

        # Should have only file handler
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_default_logger(self):
        """Test getting the default logger."""
        logger = get_logger()
        assert logger.name == 'mil_radiomics'

    def test_get_named_logger(self):
        """Test getting a named logger."""
        logger = get_logger('my_module')
        assert 'my_module' in logger.name


class TestConfigureLoggingFromArgs:
    """Tests for configure_logging_from_args function."""

    def test_configure_with_verbose(self):
        """Test configuration with verbose flag."""
        class Args:
            verbose = True
            log_file = None

        logger = configure_logging_from_args(Args())
        assert logger.level == logging.DEBUG

    def test_configure_without_verbose(self):
        """Test configuration without verbose flag."""
        class Args:
            verbose = False
            log_file = None

        logger = configure_logging_from_args(Args())
        assert logger.level == logging.INFO

    def test_configure_with_log_file(self, temp_dir):
        """Test configuration with log file."""
        log_file = temp_dir / "config_test.log"

        class Args:
            verbose = False
            log_file = str(log_file)

        logger = configure_logging_from_args(Args())

        # Log something
        logger.info("Configuration test")

        # Check file exists
        assert log_file.exists()


class TestLoggerIntegration:
    """Integration tests for logger functionality."""

    def test_multiple_loggers_dont_interfere(self):
        """Test that multiple loggers don't interfere with each other."""
        logger1 = setup_logger('logger1', level='DEBUG')
        logger2 = setup_logger('logger2', level='ERROR')

        assert logger1.level == logging.DEBUG
        assert logger2.level == logging.ERROR

    def test_logger_exc_info(self, temp_dir):
        """Test logging with exception info."""
        log_file = temp_dir / "exc_test.log"

        logger = setup_logger('exc_logger', log_file=str(log_file))

        try:
            raise ValueError("Test exception")
        except ValueError:
            logger.error("An error occurred", exc_info=True)

        content = log_file.read_text()
        assert "ValueError" in content
        assert "Test exception" in content
