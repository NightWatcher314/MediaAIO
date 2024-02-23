import datetime
import os
import subprocess
import sys
from tqdm import tqdm
from config import logger
import shutil
from config import base_logs_dir  # noqa: E402
import zipfile


class RamDisk:
    def __init__(self, disk_size_mb=1024, dist_dir="/mnt/ramdisk"):
        self.disk_size_mb = disk_size_mb
        self.dist_dir = dist_dir
        logger.info("创建 RAM 磁盘")
        os.system(f"sudo mkdir -p {dist_dir}")
        os.system(f"sudo mount -t tmpfs -o size={disk_size_mb}m tmpfs {dist_dir}")
        logger.info(f"RAM 磁盘创建成功，路径：{dist_dir}，大小：{disk_size_mb}MB")

    def __del__(self):
        logger.info("删除 RAM 磁盘")
        os.system("sudo umount /mnt/ramdisk")
        os.system("sudo rm -rf /mnt/ramdisk")
        logger.info("RAM 磁盘已删除")


class StrWarp:
    def __init__(self):
        self.str = ""

    def get_str(self):
        return self.str

    def set_str(self, str):
        self.str = str

    def append_str(self, str):
        self.str += f"{str}\n"


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


def zip_dir(dir_path="", zip_file_path=""):
    """
    压缩文件夹
    """
    if zip_file_path == "":
        dir_path = dir_path.rstrip(os.sep)
        zip_file_path = f"{dir_path}.zip"
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
    return zip_file_path


def copy_file(src_file, dst_dir, dst_file_name=""):
    """
    复制文件
    """
    if dst_file_name == "":
        return shutil.copy(src_file, dst_dir)
    else:
        return shutil.copy(src_file, os.path.join(dst_dir, dst_file_name))


def copy_files(src_files, dst_dir):
    """
    复制一系列文件
    """
    output_paths = []
    for src_file in src_files:
        output_paths.append(shutil.copy(src_file, dst_dir))
    return output_paths


def copy_dir(src_dir, dst_dir):
    """
    复制文件夹
    """
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)


def generate_io_dir(src_dir):
    """
    生成输入输出文件夹
    """
    input_dir = os.path.join(src_dir, "input")
    output_dir = os.path.join(src_dir, "output")
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return input_dir, output_dir


def read_text_file(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    return text


def clear_dir(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"无法删除 {file_path}: {e}")
        logger.info(f"文件夹 {folder_path} 已清空")
    else:
        logger.error(f"指定的路径 {folder_path} 不是一个有效的目录")


def detect_file_type(file_path):
    import magic

    mime = magic.Magic(mime=True)
    file_type = mime.from_file(file_path)
    if "audio" in file_type:
        return "audio"
    elif "video" in file_type:
        return "video"
    else:
        return "unknown"


def run_util_complete(command):
    logger.info(f"执行命令：{command}")
    os.system(command)
    logger.info("命令执行完成")
    # p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    # for line in p.stdout.readlines():
    #     print(line.decode("utf-8"))


def convert_video_to_audio(input_path: str, output_dir="", output_format="mp3"):
    from moviepy.editor import VideoFileClip

    input_name_without_ext = os.path.splitext(os.path.basename(input_path))[0]
    print(input_name_without_ext)
    # 音频文件路径
    if output_dir == "":
        output_dir = os.path.dirname(input_path)
    audio_path = f"{output_dir}/{input_name_without_ext}.{output_format}"
    # 加载视频文件
    video_clip = VideoFileClip(input_path)
    # 提取视频中的音频部分
    audio_clip = video_clip.audio
    # 保存音频文件
    audio_clip.write_audiofile(audio_path)
    # 关闭视频和音频文件
    video_clip.close()
    audio_clip.close()
    return audio_path


def convert_video_to_frames(input_path: str, output_dir="", output_format="png"):
    import imageio

    logger.info(f"开始将视频文件 {input_path} 转换为图片序列")
    file_name = os.path.splitext(os.path.basename(input_path))[0]
    if output_dir == "":
        output_dir = os.path.dirname(input_path)
    output_dir = os.path.join(output_dir, file_name)
    reader = imageio.get_reader(input_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"输出文件夹创建成功：{output_dir}")
    if (not os.path.isdir(output_dir)) or len(os.listdir(output_dir)) > 0:
        logger.error("输出路径有误或者输出文件夹不为空，请检查！")
        return
    for frame_count, im in tqdm(enumerate(reader)):
        frame_path = os.path.join(output_dir, f"frame_{frame_count}.{output_format}")
        imageio.imwrite(frame_path, im)
    logger.info(f"视频文件 {input_path} 转换为图片序列完成")
    return output_dir
