from logging import DEBUG, FileHandler, Formatter, StreamHandler, getLogger


def set_logger(logfile):
    logger = getLogger(__name__)
    logger.setLevel(DEBUG)

    stream_handler = StreamHandler()
    stream_handler.setFormatter(Formatter("%(asctime)s |%(levelname)s| %(message)s"))

    file_handler = FileHandler(filename=logfile)
    file_handler.setFormatter(Formatter("%(asctime)s |%(levelname)s| %(message)s"))

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger
