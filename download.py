from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="lj1995/VoiceConversionWebUI",
    allow_patterns="uvr5_weights/*",
    local_dir="uvr5_weights",
    # cache_dir="uvr5_weights",
    local_dir_use_symlinks=False,
)
