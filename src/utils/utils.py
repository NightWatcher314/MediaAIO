import datetime
import os
import subprocess
import sys
import threading
import cv2
import imageio
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
    return file_type


def run_util_complete(command):
    logger.info(f"执行命令：{command}")
    try:
        os.system(command)
    except Exception as e:
        logger.error(f"命令执行失败：{e}")
    logger.info("命令执行完成")


def get_file_name_without_ext(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]


def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        logger.info(f"文件 {file_path} 已删除")
    else:
        logger.error(f"文件 {file_path} 不存在，无法删除")


def convert_audio_to_audio(input_path: str, output_path="", output_format="mp3"):
    from pydub import AudioSegment

    input_name_without_ext = os.path.splitext(os.path.basename(input_path))[0]
    if output_path == "":
        output_path = os.path.join(
            os.path.dirname(input_path), f"{input_name_without_ext}.{output_format}"
        )
    # 读取 AAC 格式音频文件
    audio = AudioSegment.from_file(input_path)
    # 将音频文件转换为 MP3 格式
    audio.export(output_path, format=output_format)
    return output_path


def convert_video_to_audio(input_path: str, output_path=""):
    from moviepy.editor import VideoFileClip

    input_name_without_ext = os.path.splitext(os.path.basename(input_path))[0]
    print(input_name_without_ext)
    # 音频文件路径
    if output_path == "":
        output_path = os.path.join(
            os.path.dirname(input_path), f"{input_name_without_ext}.mp3"
        )
    # 加载视频文件
    video_clip = VideoFileClip(input_path)
    # 提取视频中的音频部分
    audio_clip = video_clip.audio
    # 保存音频文件
    audio_clip.write_audiofile(output_path)
    # 关闭视频和音频文件
    video_clip.close()
    audio_clip.close()
    return output_path


def convert_video_to_frames(
    input_path: str, output_dir: str = "", output_format="png", num_threads=8
):
    from moviepy.editor import VideoFileClip

    def extract_frames(video_path, start_frame, end_frame, output_folder):
        video = VideoFileClip(video_path)
        total_frames = int(video.duration * video.fps)
        frames_len = len(str(total_frames))
        for i in tqdm(range(start_frame, end_frame)):
            frame = video.get_frame(i / video.fps)
            frame_path = os.path.join(
                output_folder, f"{i:0{frames_len}}.{output_format}"
            )
            cv2.imwrite(frame_path, frame[:, :, ::-1])
        video.close()

    logger.info(f"开始将视频文件 {input_path} 转换为图片序列")
    if output_dir == "":
        output_dir = os.path.dirname(input_path)
        input_name_without_ext = get_file_name_without_ext(input_path)
        output_dir = os.path.join(output_dir, input_name_without_ext)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video = VideoFileClip(input_path)
    fps = video.fps
    logger.info(f"视频帧率：{fps}")
    total_frames = int(video.duration * video.fps)
    video.close()
    frames_per_thread = total_frames // num_threads

    threads = []
    for i in range(num_threads):
        start_frame = i * frames_per_thread
        end_frame = min((i + 1) * frames_per_thread, total_frames)

        thread = threading.Thread(
            target=extract_frames,
            args=(input_path, start_frame, end_frame, output_dir),
        )
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

    logger.info(f"视频文件 {input_path} 转换为图片序列完成")
    return output_dir, fps


def convert_frames_to_video(
    frame_dir: str = "", output_path: str = "", fps=30, num_threads=4
):
    import imageio
    import concurrent.futures

    def read_frames(images):
        frames = []
        for image in tqdm(images):
            image_path = os.path.join(frame_dir, image)
            frame = imageio.imread(image_path)
            frames.append(frame)
        return frames

    logger.info(f"开始将图片序列 {frame_dir} 转换为视频文件 {output_path}")
    images = [img for img in os.listdir(frame_dir)]
    images.sort()
    writer = imageio.get_writer(output_path, fps=fps)
    images_per_thread = len(images) // num_threads
    threads = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(num_threads):
            start_image = i * images_per_thread
            end_image = min((i + 1) * images_per_thread, len(images))
            thread = executor.submit(read_frames, images[start_image:end_image])
            threads.append(thread)
        for i in range(num_threads):
            frames = threads[i].result()
            for frame in frames:
                writer.append_data(frame)
    writer.close()
    logger.info(f"图片序列 {frame_dir} 转换为视频文件 {output_path} 完成")
    return output_path


def merge_audio_to_video(video_path: str, audio_path: str, output_path: str):
    import moviepy.editor as mp

    logger.info(f"开始合并音频 {audio_path} 到视频 {video_path}")
    video = mp.VideoFileClip(video_path)
    audio = mp.AudioFileClip(audio_path)
    video = video.set_audio(audio)
    video.write_videofile(output_path, codec="libx264", audio_codec="aac")
    logger.info(f"音频 {audio_path} 已合并到视频 {video_path}，输出文件：{output_path}")
    return output_path


def convert_timedelta_to_srt(td):
    total_seconds = td.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
