import datetime
import os
import sys


class Logger:
    def __init__(self, logs_dir):
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.filename = os.path.join(logs_dir, f"{timestamp}.log")
        self.terminal = sys.stdout
        self.log = open(self.filename, "w")
        sys.stdout = self

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
