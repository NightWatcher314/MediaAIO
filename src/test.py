import os
import sys
import utils.utils as utils

from warp import (
    whisper_warp,
    uvr_warp,
    funasr_warp,
    realsr_warp,
    rife_warp,
    SwinIR_warp,
    realesrgan_warp,
    nafnet_warp,
)

script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_path)

# uvr_warp._test_uvr_command()
# whisper_warp._test_whisper_exec()
funasr_warp._test_exec_funasr_command()
# realsr_warp._test_exec_realsr_command()
# rife_warp.test_video_execution()
# SwinIR_warp._test_swinir_warp()
# realesrgan_warp.test_exec_realesrgan_command()
# nafnet_warp._test_exec_nafnet()


# utils.convert_video_to_frames(
#     r"C:\Users\night\Document\Code\MediaAIO\test_assets\video\test.mp4",
#     r"C:\Users\night\Document\Code\MediaAIO\test_assets\video\test\test",
# )
# utils.convert_video_to_audio(
#     r"C:\Users\night\Document\Code\MediaAIO\test_assets\video\test.mp4",
#     r"C:\Users\night\Document\Code\MediaAIO\test_assets\video\test.mp3",
# )


# utils.convert_frames_to_video(
#     r"C:\Users\night\Document\Code\MediaAIO\test_assets\video\test\test",
#     r"C:\Users\night\Document\Code\MediaAIO\test_assets\output\merge.mp4",
# )

# utils.merge_audio_to_video(
#     r"C:\Users\night\Document\Code\MediaAIO\test_assets\output\merge.mp4",
#     r"C:\Users\night\Document\Code\MediaAIO\test_assets\video\test.mp3",
#     r"C:\Users\night\Document\Code\MediaAIO\test_assets\video\merge.mp4",
# )
