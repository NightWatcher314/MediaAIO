import os
import subprocess
import whisper


def whisper_command_warp(
    file_path,
    output_dir=None,
    language="Chinese",
    model="base",
    format="txt",
):
    command = f"whisper {file_path} --model {model} --language {language} --verbose True\
        --output_dir {output_dir} --output_format {format}"
    return command


def exec_whisper_command(file_path, output_dir, language, model, format):
    model = whisper.model_list.index(model)
    pass
