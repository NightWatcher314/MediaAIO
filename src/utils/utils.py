import datetime
import os
import sys
import shutil
from config import base_logs_dir  # noqa: E402


class Logger:
    def __init__(self, logs_dir) -> None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.filename = os.path.join(logs_dir, f"{timestamp}.log")
        self.terminal = sys.stdout
        self.log = open(self.filename, "w")
        sys.stdout = self

    def __enter__(self):
        self.terminal = sys.stdout
        self.log = open(self.filename, "w")
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.terminal
        self.log.close()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def read_logs(self):
        sys.stdout.flush()
        with open(self.filename, "r") as f:
            return f.read()

    def isatty(self):
        return False


def _generate_logs_dir(task_type: str, task_name: str, token=None):
    if token is None:
        logs_dir = os.path.join(base_logs_dir, task_type, task_name)
    else:
        logs_dir = os.path.join(base_logs_dir, task_type, task_name, token)
    # input_dir = os.path.join(logs_dir, "input")
    # output_dir = os.path.join(logs_dir, "output")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        # os.makedirs(input_dir)
        # os.makedirs(output_dir)
    return logs_dir


def get_logs_dir(task_type: str, task_name: str):
    logs_dir = os.path.join(base_logs_dir, task_type, task_name)
    if not os.path.exists(logs_dir):
        _generate_logs_dir(task_type, task_name)
    return logs_dir


def get_exec_logs_dir(task_type: str, task_name: str, token):
    logs_dir = os.path.join(base_logs_dir, task_type, task_name, token)
    if not os.path.exists(logs_dir):
        _generate_logs_dir(task_type, task_name, token)
    return logs_dir


def get_task_token(task_type, task_name):
    """
    依据当前时间生成任务 token
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{timestamp}_{task_type}_{task_name}__"


def save_gradio_file(file, save_dir, filename):
    """
    保存 gradio 上传的文件
    """
    # with open(os.path.join(save_dir, filename), "wb") as f:
    #     f.write(file)
    shutil.copy(file.name, os.path.join(save_dir, filename))


def read_text_file(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    return text
