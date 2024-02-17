from time import sleep
import gradio as gr
import os
import sys
import gradio_pages.whisper_page as whisper_page
import gradio_pages.uvr_block as uvr_block
import utils.utils as utils
from warp import whisper_warp
from warp import realesrgan_warp
from warp import RobustVideoMatting_warp as rvm_warp
from warp import qwen_warp
from warp import uvr_warp
import asyncio

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_path)


# evnet_loop = asyncio.get_event_loop()
# asyncio.set_event_loop(evnet_loop)
# evnet_loop.run_until_complete(
#     uvr_warp.exec_uvr_command(
#         "UVR-DeEcho-DeReverb.pth",
#         first_out_dir="/home/night/Code/MediaAIO/test_assets/",
#         second_out_dir="/home/night/Code/MediaAIO/test_assets/",
#         model_type="vr",
#         audio_files=[
#             "/home/night/Code/MediaAIO/test_assets/audio.aac",
#             "/home/night/Code/MediaAIO/test_assets/audio_copy.aac",
#         ],
#     )
# )
# evnet_loop.stop()


# uvr_warp.exec_uvr_command(
#     "UVR-DeNoise.pth",
#     output_dir="/home/night/Code/MediaAIO/test_assets",
#     model_type="vr",
#     audio_file="/home/night/Code/MediaAIO/test_assets/audio.aac",
# )

# qwen_warp._check_and_download_model("tiny")

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


demo = gr.TabbedInterface([uvr_block.ui()], ["UVR5"])
demo.queue(max_size=512)
if __name__ == "__main__":
    demo.launch(
        max_threads=4, share=False, debug=True, inline=True, auth=None, ssl_verify=False
    )
