import logging

from rich.logging import RichHandler

# Create a logger
logger = logging.getLogger(__name__)

# Create handlers for shell and file logging
shell_handler = RichHandler()
file_handler = logging.FileHandler("debug.log")

# Set logging levels
logger.setLevel(logging.DEBUG)
shell_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.DEBUG)

# Define format for shell and file outputs
fmt_shell = "%(message)s"
fmt_file = (
    "%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
)

# Create formatters
shell_formatter = logging.Formatter(fmt_shell)
file_formatter = logging.Formatter(fmt_file)

# Assign formatters to handlers
shell_handler.setFormatter(shell_formatter)
file_handler.setFormatter(file_formatter)

# Add handlers to the logger
logger.addHandler(shell_handler)
logger.addHandler(file_handler)
