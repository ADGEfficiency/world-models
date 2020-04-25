import logging
import logging.config
import sys


def setup_logging(log_file):
    formatter = logging.Formatter('%(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger
