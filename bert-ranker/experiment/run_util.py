import logging
import sys
from os import path


def setup_logger(config_logger, out_folder, name):
    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler_stdout = logging.StreamHandler(sys.stdout)
    handler_stdout.setLevel(config_logger['level'])
    handler_stdout.setFormatter(formatter)
    logger.addHandler(handler_stdout)

    log_file_path = path.join(out_folder, 'log.txt')
    handler_file = logging.FileHandler(log_file_path)
    handler_file.setLevel(config_logger['level'])
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)

    logger.setLevel(config_logger['level'])
    return logger
