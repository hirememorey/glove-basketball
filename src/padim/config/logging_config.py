"""
Centralized logging configuration for PADIM.
Ensures consistent logging across all modules with file output and monitoring.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

from .settings import LOG_LEVEL, LOG_FILE, ENABLE_MONITORING


class StructuredFormatter(logging.Formatter):
    """Custom formatter that adds structured data to log records."""

    def format(self, record):
        # Add timestamp in ISO format
        record.iso_timestamp = datetime.fromtimestamp(record.created).isoformat()

        # Add structured fields for monitoring
        if not hasattr(record, 'extra_data'):
            record.extra_data = {}

        # Format the message
        message = super().format(record)

        # Add structured data as JSON if monitoring is enabled
        if ENABLE_MONITORING and record.extra_data:
            try:
                structured_data = {
                    'timestamp': record.iso_timestamp,
                    'level': record.levelname,
                    'module': record.name,
                    'message': record.getMessage(),
                    'extra_data': record.extra_data
                }
                return json.dumps(structured_data)
            except (TypeError, ValueError):
                # Fallback to regular formatting if JSON serialization fails
                pass

        return message


class MonitoringFilter(logging.Filter):
    """Filter that adds monitoring metadata to log records."""

    def filter(self, record):
        # Add monitoring metadata
        if not hasattr(record, 'extra_data'):
            record.extra_data = {}

        # Add common monitoring fields
        record.extra_data.update({
            'process_type': 'padim_data_processing',
            'version': '5.0'
        })

        return True


def setup_logging(
    log_level: str = LOG_LEVEL,
    log_file: str = LOG_FILE,
    enable_monitoring: bool = ENABLE_MONITORING,
    console_output: bool = True
) -> logging.Logger:
    """
    Setup centralized logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        enable_monitoring: Whether to enable structured monitoring logs
        console_output: Whether to also output to console

    Returns:
        Root logger configured with handlers
    """

    # Convert string log level to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(exist_ok=True)

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(numeric_level)

    # Create formatters
    if enable_monitoring:
        formatter = StructuredFormatter(
            '%(iso_timestamp)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    # File handler with rotation (10MB, keep 5 backups)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)

    # Add monitoring filter if enabled
    if enable_monitoring:
        file_handler.addFilter(MonitoringFilter())

    root_logger.addHandler(file_handler)

    # Console handler (optional)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        if enable_monitoring:
            console_handler.addFilter(MonitoringFilter())
        root_logger.addHandler(console_handler)

    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={log_level}, file={log_file}, monitoring={enable_monitoring}")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_performance_metric(operation: str, duration: float, success: bool = True,
                          extra_data: Optional[dict] = None):
    """
    Log a performance metric for monitoring.

    Args:
        operation: Name of the operation
        duration: Duration in seconds
        success: Whether the operation was successful
        extra_data: Additional data to include
    """
    logger = get_logger(__name__)

    metric_data = {
        'operation': operation,
        'duration_seconds': duration,
        'success': success,
        'metric_type': 'performance'
    }

    if extra_data:
        metric_data.update(extra_data)

    if success:
        logger.info(f"Performance metric: {operation} completed in {duration:.2f}s", extra={'extra_data': metric_data})
    else:
        logger.warning(f"Performance metric: {operation} failed after {duration:.2f}s", extra={'extra_data': metric_data})


def log_api_call(endpoint: str, success: bool, response_time: float,
                status_code: Optional[int] = None, error_message: Optional[str] = None,
                extra_data: Optional[dict] = None):
    """
    Log API call metrics for monitoring.

    Args:
        endpoint: API endpoint called
        success: Whether the call was successful
        response_time: Response time in seconds
        status_code: HTTP status code if applicable
        error_message: Error message if call failed
        extra_data: Additional data to include
    """
    logger = get_logger(__name__)

    api_data = {
        'endpoint': endpoint,
        'response_time_seconds': response_time,
        'metric_type': 'api_call'
    }

    if status_code is not None:
        api_data['status_code'] = status_code

    if error_message:
        api_data['error_message'] = error_message

    if extra_data:
        api_data.update(extra_data)

    if success:
        logger.debug(f"API call successful: {endpoint}", extra={'extra_data': api_data})
    else:
        logger.warning(f"API call failed: {endpoint}", extra={'extra_data': api_data})


def log_data_quality_metric(game_id: str, metric_name: str, value: float,
                           expected_range: Optional[tuple] = None):
    """
    Log data quality metrics.

    Args:
        game_id: Game ID being processed
        metric_name: Name of the quality metric
        value: Metric value
        expected_range: Tuple of (min, max) expected values
    """
    logger = get_logger(__name__)

    quality_data = {
        'game_id': game_id,
        'metric_name': metric_name,
        'value': value,
        'metric_type': 'data_quality'
    }

    if expected_range:
        quality_data['expected_min'] = expected_range[0]
        quality_data['expected_max'] = expected_range[1]
        quality_data['within_range'] = expected_range[0] <= value <= expected_range[1]

    logger.info(f"Data quality: {metric_name} = {value}", extra={'extra_data': quality_data})


# Initialize logging when module is imported
_root_logger = setup_logging()
