import asyncio
import os
import sys

import gradio as gr
import gradio_pages.uvr_block as uvr_block
import gradio_pages.speech_recognition_block as speech_recognition_block
import gradio_pages.video_super_inter_block as video_super_inter_block
import utils.utils as utils
from warp import RobustVideoMatting_warp as rvm_warp
from warp import (
    qwen_warp,
    realesrgan_warp,
    uvr_warp,
    whisper_warp,
    rife_warp,
    funasr_warp,
)

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_path)

# rife_warp.test_video_execution()
# realesrgan_warp.test_exec_realesrgan_video_command()
# funasr_warp.test_exec_funasr_command()

demo = gr.TabbedInterface([speech_recognition_block.ui()], ["Speech Recognition"])
demo.queue(max_size=512)
if __name__ == "__main__":
    demo.launch(
        max_threads=40,
        share=False,
        debug=True,
        inline=True,
        auth=None,
        ssl_verify=False,
    )


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
