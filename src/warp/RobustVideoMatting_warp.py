import os
from typing import Optional, Tuple
import torch
import pims
from torch.utils.data import DataLoader
import av
import numpy as np
from PIL import Image
import subprocess
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
from tqdm import tqdm
from config import base_models_dir, base_work_dir
from torch.utils.data import Dataset


def convert_video(
    model,
    input_source: str,
    input_resize: Optional[Tuple[int, int]] = None,
    downsample_ratio: Optional[float] = None,
    output_type: str = "video",
    output_composition: Optional[str] = None,
    output_alpha: Optional[str] = None,
    output_foreground: Optional[str] = None,
    output_video_mbps: Optional[float] = None,
    seq_chunk: int = 1,
    num_workers: int = 0,
    progress: bool = True,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
):
    """
    Args:
        input_source:A video file, or an image sequence directory. Images must be sorted in accending order, support png and jpg.
        input_resize: If provided, the input are first resized to (w, h).
        downsample_ratio: The model's downsample_ratio hyperparameter. If not provided, model automatically set one.
        output_type: Options: ["video", "png_sequence"].
        output_composition:
            The composition output path. File path if output_type == 'video'. Directory path if output_type == 'png_sequence'.
            If output_type == 'video', the composition has green screen background.
            If output_type == 'png_sequence'. the composition is RGBA png images.
        output_alpha: The alpha output from the model.
        output_foreground: The foreground output from the model.
        seq_chunk: Number of frames to process at once. Increase it for better parallelism.
        num_workers: PyTorch's DataLoader workers. Only use >0 for image input.
        progress: Show progress bar.
        device: Only need to manually provide if model is a TorchScript freezed model.
        dtype: Only need to manually provide if model is a TorchScript freezed model.
    """

    def auto_downsample_ratio(h, w):
        """
        Automatically find a downsample ratio so that the largest side of the resolution be 512px.
        """
        return min(512 / max(h, w), 1)

    class ImageSequenceReader(Dataset):
        def __init__(self, path, transform=None):
            self.path = path
            self.files = sorted(os.listdir(path))
            self.transform = transform

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            with Image.open(os.path.join(self.path, self.files[idx])) as img:
                img.load()
            if self.transform is not None:
                return self.transform(img)
            return img

    class ImageSequenceWriter:
        def __init__(self, path, extension="jpg"):
            self.path = path
            self.extension = extension
            self.counter = 0
            os.makedirs(path, exist_ok=True)

        def write(self, frames):
            # frames: [T, C, H, W]
            for t in range(frames.shape[0]):
                to_pil_image(frames[t]).save(
                    os.path.join(
                        self.path, str(self.counter).zfill(4) + "." + self.extension
                    )
                )
                self.counter += 1

        def close(self):
            pass

    class VideoReader(Dataset):
        def __init__(self, path, transform=None):
            self.video = pims.PyAVVideoReader(path)
            self.rate = self.video.frame_rate
            self.transform = transform

        @property
        def frame_rate(self):
            return self.rate

        def __len__(self):
            return len(self.video)

        def __getitem__(self, idx):
            frame = self.video[idx]
            frame = Image.fromarray(np.asarray(frame))
            if self.transform is not None:
                frame = self.transform(frame)
            return frame

    class VideoWriter:
        def __init__(self, path, frame_rate, bit_rate=1000000):
            self.container = av.open(path, mode="w")
            self.stream = self.container.add_stream("h264", rate=f"{frame_rate:.4f}")
            self.stream.pix_fmt = "yuv420p"
            self.stream.bit_rate = bit_rate

        def write(self, frames):
            # frames: [T, C, H, W]
            self.stream.width = frames.size(3)
            self.stream.height = frames.size(2)
            if frames.size(1) == 1:
                frames = frames.repeat(1, 3, 1, 1)  # convert grayscale to RGB
            frames = frames.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()
            for t in range(frames.shape[0]):
                frame = frames[t]
                frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                self.container.mux(self.stream.encode(frame))

        def close(self):
            self.container.mux(self.stream.encode())
            self.container.close()

    assert downsample_ratio is None or (
        downsample_ratio > 0 and downsample_ratio <= 1
    ), "Downsample ratio must be between 0 (exclusive) and 1 (inclusive)."
    assert any(
        [output_composition, output_alpha, output_foreground]
    ), "Must provide at least one output."
    assert output_type in [
        "video",
        "png_sequence",
    ], 'Only support "video" and "png_sequence" output modes.'
    assert seq_chunk >= 1, "Sequence chunk must be >= 1"
    assert num_workers >= 0, "Number of workers must be >= 0"

    # Initialize transform
    if input_resize is not None:
        transform = transforms.Compose(
            [transforms.Resize(input_resize[::-1]), transforms.ToTensor()]
        )
    else:
        transform = transforms.ToTensor()

    # Initialize reader
    if os.path.isfile(input_source):
        source = VideoReader(input_source, transform)
    else:
        source = ImageSequenceReader(input_source, transform)
    reader = DataLoader(
        source, batch_size=seq_chunk, pin_memory=True, num_workers=num_workers
    )

    # Initialize writers
    if output_type == "video":
        frame_rate = source.frame_rate if isinstance(source, VideoReader) else 30
        output_video_mbps = 1 if output_video_mbps is None else output_video_mbps
        if output_composition is not None:
            writer_com = VideoWriter(
                path=output_composition,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000),
            )
        if output_alpha is not None:
            writer_pha = VideoWriter(
                path=output_alpha,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000),
            )
        if output_foreground is not None:
            writer_fgr = VideoWriter(
                path=output_foreground,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000),
            )
    else:
        if output_composition is not None:
            writer_com = ImageSequenceWriter(output_composition, "png")
        if output_alpha is not None:
            writer_pha = ImageSequenceWriter(output_alpha, "png")
        if output_foreground is not None:
            writer_fgr = ImageSequenceWriter(output_foreground, "png")

    # Inference
    model = model.eval()
    if device is None or dtype is None:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device

    if (output_composition is not None) and (output_type == "video"):
        bgr = (
            torch.tensor([120, 255, 155], device=device, dtype=dtype)
            .div(255)
            .view(1, 1, 3, 1, 1)
        )

    try:
        with torch.no_grad():
            bar = tqdm(total=len(source), disable=not progress, dynamic_ncols=True)
            rec = [None] * 4
            for src in reader:
                if downsample_ratio is None:
                    downsample_ratio = auto_downsample_ratio(*src.shape[2:])

                src = src.to(device, dtype, non_blocking=True).unsqueeze(
                    0
                )  # [B, T, C, H, W]
                fgr, pha, *rec = model(src, *rec, downsample_ratio)

                if output_foreground is not None:
                    writer_fgr.write(fgr[0])
                if output_alpha is not None:
                    writer_pha.write(pha[0])
                if output_composition is not None:
                    if output_type == "video":
                        com = fgr * pha + bgr * (1 - pha)
                    else:
                        fgr = fgr * pha.gt(0)
                        com = torch.cat([fgr, pha], dim=-3)
                    writer_com.write(com[0])

                bar.update(src.size(1))

    finally:
        # Clean up
        if output_composition is not None:
            writer_com.close()
        if output_alpha is not None:
            writer_pha.close()
        if output_foreground is not None:
            writer_fgr.close()


model_list = ["mobilenetv3", "resnet50"]

inference_path = os.path.join(base_models_dir, "RobustVideoMatting/inference.py")
srcipt_dir = os.path.join(base_models_dir, "RobustVideoMatting")
checkpoint_path_dict = {
    "mobilenetv3": os.path.join(
        base_models_dir, "RobustVideoMatting/weights/rvm_mobilenetv3.pth"
    ),
    "resnet50": os.path.join(
        base_models_dir, "RobustVideoMatting/weights/rvm_resnet50.pth"
    ),
}


async def exec_rvm_command(
    input_path="",
    output_dir="",
    model="",
    output_mbps=4,
    output_type="video",
    output_alpha=False,
    output_foreground=False,
):
    pass
