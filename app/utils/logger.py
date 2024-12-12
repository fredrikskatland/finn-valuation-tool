import logging
from rich.logging import RichHandler

def setup_logger():
    """
    Set up a logger with Rich for colorful and visually appealing logs.
    """
    # Configure the logging format
    logging.basicConfig(
        level=logging.INFO,  # Default log level
        format="%(message)s",  # Rich handles formatting
        datefmt="[%X]",  # Time format
        handlers=[RichHandler(rich_tracebacks=True, show_time=True)],
    )

    # Create and return the logger
    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)
    return logger
