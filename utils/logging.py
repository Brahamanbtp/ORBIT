import logging

def get_logger(name: str) -> logging.Logger:
    """Return a named logger configured with an INFO-level StreamHandler."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    has_stream_handler = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    if not has_stream_handler:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = get_logger("ORBIT")
