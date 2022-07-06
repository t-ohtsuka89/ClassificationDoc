from logging import DEBUG, FileHandler, Formatter, StreamHandler, getLogger


def set_logger(log_path: str, log_level: int = DEBUG) -> None:
    """
    Set logger.
    """
    logger = getLogger(__name__)
    logger.setLevel(DEBUG)
    formatter = Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler = StreamHandler()
    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    handler = FileHandler(log_path, mode="a")
    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
