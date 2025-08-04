"""
Logger module for centralized logging functionality.
"""
import os
import sys
import logging
import shutil
import warnings
from datetime import datetime
from typing import Optional, Dict, Any, Union
from dotenv import load_dotenv
import inspect


class Logger:
    """
    Centralized logging system that collects all logs, exceptions, and messages.
    Outputs to both console and a log file specified in the environment configuration.
    """
    _instance = None
    
    def __new__(cls) -> 'Logger':
        """Singleton pattern to ensure only one logger instance exists."""
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """
        Initialize the logger with configurations from environment variables.
        """
        # Load environment variables
        env_path = os.path.join(os.path.dirname(__file__), 'config', '.env')
        load_dotenv(env_path)
        
        # Get log file path from environment or use default
        log_dir = os.getenv('LOG_DIR', 'logs')
        
        # Ensure log directory exists
        if os.path.exists(log_dir):
            # Clean log directory (optional based on environment setting)
            if os.getenv('CLEAN_LOGS', 'True').lower() == 'true':
                try:
                    # Delete and recreate log directory
                    shutil.rmtree(log_dir)
                    print(f"Log directory cleaned: {os.path.abspath(log_dir)}")
                except Exception as e:
                    print(f"Warning: Could not clean log directory: {str(e)}")
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_log_path = os.path.join(log_dir, f"borai_{timestamp}.log")
        self.log_file_path = os.path.abspath(os.getenv('LOG_FILE_PATH', default_log_path))
        
        # Configure the Python logging module
        self.logger = logging.getLogger()  # Root logger to capture all logs
        self.logger.setLevel(logging.DEBUG)
        
        # Remove any existing handlers to prevent duplication
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler - all logs go to file
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter with class/function info instead of static name
        formatter = logging.Formatter('[%(asctime)s] - %(module)s:%(funcName)s - %(levelname)s - %(message)s', 
                                     datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        
        # Add only file handler to logger (no console handler)
        self.logger.addHandler(file_handler)
        
        # Redirect warnings to logging system
        logging.captureWarnings(True)
        
        # Print initial message to console
        print(f"Logger initialized. All logs will be saved to: {self.log_file_path}")
        
        # Redirect standard outputs to our logger
        sys.stdout = LoggerWriter(self.logger.info)
        sys.stderr = LoggerWriter(self.logger.error)
        
        # Log startup message to file
        self._log(logging.INFO, f"Logger initialized. Logs will be saved to {self.log_file_path}")
    
    def _get_caller_info(self) -> str:
        """Get the caller's class or module name and function."""
        stack = inspect.stack()
        # Go up 2 frames to get the caller of the logging method
        if len(stack) > 2:
            frame = stack[2]
            module = frame.frame.f_globals.get('__name__', 'unknown')
            function = frame.function
            return f"{module}:{function}"
        return "unknown:unknown"
    
    def _log(self, level: int, message: str) -> None:
        """Internal logging method that adds caller information."""
        self.logger.log(level, message)
    
    def info(self, message: str) -> None:
        """
        Log an info level message.
        
        Args:
            message: The message to log
        """
        self._log(logging.INFO, message)
    
    def debug(self, message: str) -> None:
        """
        Log a debug level message.
        
        Args:
            message: The message to log
        """
        self._log(logging.DEBUG, message)
    
    def warning(self, message: str) -> None:
        """
        Log a warning level message.
        
        Args:
            message: The message to log
        """
        self._log(logging.WARNING, message)
    
    def error(self, message: str) -> None:
        """
        Log an error level message.
        
        Args:
            message: The message to log
        """
        self._log(logging.ERROR, message)
    
    def critical(self, message: str) -> None:
        """
        Log a critical level message.
        
        Args:
            message: The message to log
        """
        self._log(logging.CRITICAL, message)
    
    def exception(self, e: Exception, context: Optional[str] = None) -> None:
        """
        Log an exception with optional context.
        
        Args:
            e: The exception to log
            context: Optional context information about where the exception occurred
        """
        message = f"Exception: {str(e)}"
        if context:
            message = f"{context} - {message}"
        self._log(logging.ERROR, message)
    
    def log_operation(self, operation: str, status: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an operation with status and details.
        
        Args:
            operation: The name of the operation being performed
            status: The status of the operation (started, completed, failed)
            details: Optional details about the operation
        """
        message = f"Operation '{operation}' {status}"
        
        if details:
            message += f" - Details: {details}"
            
        if status == "failed":
            self.error(message)
        else:
            self.info(message)
            
    def log_db_connection(self, status: str, db_type: str, connection_info: Optional[Dict[str, str]] = None) -> None:
        """
        Log database connection events.
        
        Args:
            status: Connection status (connecting, connected, failed)
            db_type: Type of database
            connection_info: Optional connection information (with sensitive data masked)
        """
        connection_details = connection_info or {}
        # Mask sensitive data
        if 'password' in connection_details:
            connection_details['password'] = '*****'
            
        message = f"Database {db_type} connection {status}"
        
        if connection_details:
            message += f" - Details: {connection_details}"
            
        if status == "failed":
            self.error(message)
        else:
            self.info(message)
            
    def log_model_loading(self, model_name: str, device_type: str, status: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log AI model loading events.
        
        Args:
            model_name: Name of the model being loaded
            device_type: Device type (GPU, CPU)
            status: Loading status (started, completed, failed)
            details: Optional details about the model loading
        """
        model_details = details or {}
        message = f"Model '{model_name}' loading on {device_type} {status}"
        
        if model_details:
            message += f" - Details: {model_details}"
            
        if status == "failed":
            self.error(message)
        else:
            self.info(message)


class LoggerWriter:
    """
    A class that redirects stdout/stderr to the logging system.
    """
    def __init__(self, log_method):
        self.log_method = log_method
        self.buffer = ''

    def write(self, message):
        if message and not message.isspace():
            # Only log non-empty, non-whitespace messages
            self.buffer += message
            if self.buffer.endswith('\n'):
                self.log_method(self.buffer.rstrip())
                self.buffer = ''

    def flush(self):
        if self.buffer:
            self.log_method(self.buffer.rstrip())
            self.buffer = ''


# Create a global instance that can be imported directly
logger = Logger()