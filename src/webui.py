import os
import sys

import gradio as gr
import gradio_pages.uvr_block as uvr_block
import gradio_pages.speech_recognition_block as speech_recognition_block
import gradio_pages.video_super_inter_block as video_super_inter_block
import gradio_pages.image_super_block as image_super_block
from warp import (
    whisper_warp,
    uvr_warp,
)

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_path)
# whisper_warp._test_whisper_exec()
# uvr_warp._test_uvr_command()
# realsr_warp._test_exec_realsr_command()
# VRT_warp.test_VRT_warp()
# rife_warp.test_video_execution()
# realesrgan_warp.test_exec_realesrgan_video_command()
# funasr_warp.test_exec_funasr_command()
# llm_warp._test_llm()
# nafnet_warp._test_exec_nafnet()
demo = gr.TabbedInterface(
    [
        image_super_block.ui(),
        uvr_block.ui(),
        speech_recognition_block.ui(),
        video_super_inter_block.ui(),
    ],
    [
        image_super_block.block_name,
        uvr_block.block_name,
        speech_recognition_block.block_name,
        video_super_inter_block.block_name,
    ],
)
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
