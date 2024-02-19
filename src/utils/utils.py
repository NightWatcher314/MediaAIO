import datetime
import os
import sys
import shutil
from config import base_logs_dir  # noqa: E402
import zipfile


class RamDisk:
    def __init__(self, disk_size_mb=1024, dist_dir="/mnt/ramdisk"):
        self.disk_size_mb = disk_size_mb
        self.dist_dir = dist_dir
        print("Create ramdisk")
        print("Please check the console to input your password to create ramdisk")
        os.system(f"sudo mkdir -p {dist_dir}")
        os.system(f"sudo mount -t tmpfs -o size={disk_size_mb}m tmpfs {dist_dir}")
        print(f"Ramdisk created at {dist_dir}")

    def __del__(self):
        print("Delete ramdisk")
        print("Please check the console to input your password to delete ramdisk")
        os.system("sudo umount /mnt/ramdisk")
        os.system("sudo rm -rf /mnt/ramdisk")
        print("Ramdisk deleted")


def get_exec_logs_dir(task_name: str, token, base_dir=base_logs_dir):
    logs_dir = os.path.join(base_dir, task_name, token)
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
    shutil.copy(file, os.path.join(save_dir, filename))


def copy_file(src_file, dst_dir, dst_file_name):
    """
    复制文件
    """
    shutil.copy(src_file, os.path.join(dst_dir, dst_file_name))


def copy_dir(src_dir, dst_dir):
    """
    复制文件夹
    """
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


def read_text_file(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    return text
