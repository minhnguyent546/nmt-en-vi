import logging


logger = logging.getLogger('nmt-en-vi')

def init_logger(*, name: str = 'nmt-en-vi', log_level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    logging_format = logging.Formatter('[%(levelname)s:%(funcName)s] %(message)s')
    console_handler.setFormatter(logging_format)

    logger.addHandler(console_handler)

    return logger
