import asyncio
import os
import gradio as gr
from config import logger
import utils.utils as utils
from warp.realesrgan_warp import (
    reg_model_list,
    exec_realesrgan_command,
)
from warp.rife_warp import rife_model_list, exec_rife_video_command

task_name = "video_super_inter"
block_name = "video_super_inter"
task_token = ""
exec_logs_dir = ""
event_loop = asyncio.new_event_loop()
process = ""
inter_model_list = rife_model_list
ramdisk = None


def on_single_convert_click(
    input_video,
    super_model,
    super_scale,
    super_denoise,
    super_face_enhance,
    super_tile,
    super_fp32,
    enable_ramdisk,
    ramdisk_size_mb,
    inter_model,
    inter_scale,
    inter_exp,
    inter_fp16,
    inter_montage,
    enable_super,
    enable_inter,
):
    global task_token, exec_logs_dir, process, ramdisk
    if not input_video:
        gr.Error("Please upload a video file.")
        return gr.Video()
    task_token = utils.get_task_token(task_name)
    if not (enable_super or enable_inter):
        gr.Error("Please select at least one model.")
        return gr.Video()

    # 创建 ramdisk 并修改工作目录
    if enable_ramdisk:
        ramdisk = utils.RamDisk(ramdisk_size_mb)
        exec_logs_dir = utils.get_exec_logs_dir(task_name, task_token, ramdisk.dist_dir)
        utils.copy_file(input_video, exec_logs_dir, input_video.split("/")[-1])
        input_video = os.path.join(exec_logs_dir, input_video.split("/")[-1])
    else:
        exec_logs_dir = utils.get_exec_logs_dir(task_name, task_token)

    # 开始执行 超分辨率 视频命令
    if enable_super:
        print("Start to exec RealESRGAN video command\n\n")
        output_file_path = exec_realesrgan_command(
            input_path=input_video,
            output_dir=exec_logs_dir,
            model=super_model,
            scale=super_scale,
            denoise=super_denoise,
            face_enhance=super_face_enhance,
            tile=super_tile,
            fp32=super_fp32,
        )[0]

    if enable_inter:
        # 开始执行 插帧 视频命令
        print("Start to exec RIFE video command\n\n")
        output_file_path = exec_rife_video_command(
            input_path=output_file_path if enable_super else input_video,
            exp=inter_exp,
            scale=inter_scale,
            fp16=inter_fp16,
            montage=inter_montage,
        )[0]

    # 删除 ramdisk，并拷贝至 logs 目录，并修改输出路径
    if enable_ramdisk:
        exec_logs_dir_new = utils.get_exec_logs_dir(task_name, task_token)
        utils.copy_dir(exec_logs_dir, exec_logs_dir_new)
        exec_logs_dir = exec_logs_dir_new
        output_file_path = os.path.join(exec_logs_dir, output_file_path.split("/")[-1])
        del ramdisk

    return gr.Video(value=output_file_path, interactive=False)


def on_stop_click():
    global ramdisk, process, exec_logs_dir
    if ramdisk is not None:
        exec_logs_dir_new = utils.get_exec_logs_dir(task_name, task_token)
        utils.copy_dir(exec_logs_dir, exec_logs_dir_new)
        exec_logs_dir = exec_logs_dir_new
        del ramdisk
    process.kill()
    print("Stop the process and delete the ramdisk.")


def ui():
    with gr.Blocks() as demo:
        with gr.Accordion(label="Single Audio File"):
            with gr.Row():
                with gr.Column():
                    input_video = gr.Video(interactive=True)
                with gr.Column():
                    output_video = gr.Video(interactive=False)
                with gr.Column():
                    with gr.Row():
                        single_convert_btn = gr.Button(
                            interactive=True, value="Convert"
                        )
                        single_stop_btn = gr.Button(interactive=True, value="Stop")
                    with gr.Row():
                        ramdisk_size_mb = gr.Number(
                            value=8192,
                            label="Ramdisk Size in MB",
                            minimum=512,
                            maximum=8192 * 2,
                            step=512,
                            interactive=True,
                        )
                        enable_ramdisk = gr.Checkbox(
                            value=False, label="Enable Ramdisk", interactive=True
                        )
                    with gr.Row():
                        single_logs_area = gr.Textbox(interactive=False, max_lines=8)
        with gr.Accordion(label="Batch Audio Files"):
            pass
        with gr.Accordion(label="Super Resolution Settings"):
            with gr.Row():
                with gr.Column():
                    enable_super = gr.Checkbox(
                        label="Enable Super Resolution", value=True, interactive=True
                    )
                    super_model = gr.Dropdown(
                        choices=reg_model_list,
                        label="Model",
                        value=reg_model_list[0],
                        interactive=True,
                    )

                with gr.Column():
                    super_denoise = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=0.1,
                        label="Denoise",
                        value=0.5,
                        interactive=True,
                    )
                    super_face_enhance = gr.Checkbox(
                        label="Face Enhance", value=False, interactive=True
                    )
                    super_fp32 = gr.Checkbox(
                        label="FP32", value=False, interactive=True
                    )
                with gr.Column():
                    super_scale = gr.Number(
                        value=4,
                        label="Scale",
                        minimum=1,
                        maximum=16,
                        step=1,
                        interactive=True,
                    )
                    super_tile = gr.Number(
                        value=0,
                        label="Tile",
                        minimum=0,
                        maximum=16,
                        step=1,
                        interactive=True,
                    )
        with gr.Accordion(label="Frame Interpolation Settings"):
            with gr.Row():
                with gr.Column():
                    enable_inter = gr.Checkbox(
                        label="Enable Frame Interpolation", value=True, interactive=True
                    )
                    inter_model = gr.Dropdown(
                        choices=inter_model_list,
                        label="Model",
                        value=inter_model_list[0],
                        interactive=True,
                    )
                with gr.Column():
                    inter_scale = gr.Dropdown(
                        choices=[0.25, 0.5, 1.0, 2.0, 4.0],
                        value=1.0,
                        label="Scale",
                        interactive=True,
                    )
                    inter_exp = gr.Number(
                        value=2,
                        maximum=4,
                        minimum=1,
                        step=1,
                        interactive=True,
                        label="Exp",
                    )
                with gr.Column():
                    inter_fp16 = gr.Checkbox(
                        label="FP16", value=False, interactive=True
                    )
                    inter_montage = gr.Checkbox(
                        label="Montage", value=False, interactive=True
                    )

        # demo.load(logger.read_logs, None, single_logs_area, every=1)
        single_convert_btn.click(
            on_single_convert_click,
            [
                input_video,
                super_model,
                super_scale,
                super_denoise,
                super_face_enhance,
                super_tile,
                super_fp32,
                enable_ramdisk,
                ramdisk_size_mb,
                inter_model,
                inter_scale,
                inter_exp,
                inter_fp16,
                inter_montage,
                enable_super,
                enable_inter,
            ],
            output_video,
        )
        single_stop_btn.click(on_stop_click)
    return demo
