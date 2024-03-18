import datetime
import os
import sys
import logging


class Logger:
    def __init__(self, logs_dir, logger_level=logging.DEBUG):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.filename = os.path.join(logs_dir, f"{timestamp}.log")
        self.terminal = sys.stdout
        self.log = open(self.filename, "w")
        sys.stdout = self
        self._init_logger(logger_level)

    def _init_logger(self, logger_level=logging.DEBUG):
        self.logger = logging.getLogger()
        self.logger.setLevel(logger_level)
        file_handler = logging.FileHandler(self.filename)
        file_handler.setLevel(logger_level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logger_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def read_logs(self):
        self.flush()
        with open(self.filename, "r") as f:
            return f.read()

    def isatty(self):
        return False

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message)
