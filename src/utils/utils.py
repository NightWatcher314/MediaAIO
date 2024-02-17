import datetime
import os
import sys
import shutil
from config import base_logs_dir  # noqa: E402
import zipfile

# def get_tmp_logs_dir(task_name: str):
#     logs_dir = os.path.join(base_logs_dir, task_name, "tmp")
#     if not os.path.exists(logs_dir):
#         os.makedirs(logs_dir)
#     return logs_dir


def get_exec_logs_dir(task_name: str, token):
    logs_dir = os.path.join(base_logs_dir, task_name, token)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    return logs_dir


def get_task_token(task_name):
    """
    依据当前时间生成任务 token
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{timestamp}_{task_name}_"


def zip_dir(dir_path, zip_file_path):
    """
    压缩文件夹
    """
    with zipfile.ZipFile(zip_file_path, "w") as z:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.endswith(".zip"):
                    continue
                z.write(
                    os.path.join(root, file),
                    os.path.relpath(
                        os.path.join(root, file), os.path.join(dir_path, "..")
                    ),
                )


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
