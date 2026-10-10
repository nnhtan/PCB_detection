import logging


def get_logger(name: str) -> logging.Logger:
    """Return a module logger using the standard Python logging configuration."""
    return logging.getLogger(name)
