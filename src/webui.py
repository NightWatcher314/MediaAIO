from time import sleep
import gradio as gr
import os
import sys
import gradio_pages.whisper_page as whisper_page
import utils.utils as utils
from warp import whisper_warp
from warp import realesrgan_warp
from warp import RobustVideoMatting_warp as rvm_warp
from warp import qwen_warp

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_path)

qwen_warp._check_and_download_model("tiny")

# file_path = "/home/night/Code/MediaAIO/test_assets/1862_1707890526_2X_56fps.mp4"
# print(
#     # realesrgan_warp.exec_realesrgan_command(
#     #     input_path=file_path,
#     #     model="RealESRGAN_x4plus",
#     #     output_dir="/home/night/Code/MediaAIO",
#     #     scale=4,
#     #     face_enhance=True,
#     #     type="video",
#     # )
#     # whisper_warp._check_and_download_model("tiny")
# )
# rvm_warp.exec_rvm_command(
#     input_path=file_path,
#     output_dir="/home/night/Code/MediaAIO/test_assets",
#     model="mobilenetv3",
#     output_mbps=4,
#     output_type="video",
#     output_alpha=True,
#     output_foreground=True,
# )

# def whisper(text):
#     sleep(10)
#     return text.lower() + "..."


# demo = gr.TabbedInterface(
#     [final2x_page.ui(), whisper_page.ui()], ["Whisper", "Super_Resolution"]
# )
# demo.queue(max_size=4)

# if __name__ == "__main__":
#     demo.launch(max_threads=4, share=False, debug=True, inline=True, auth=None)
