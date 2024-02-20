# MediaAIO

集成音视频处理相关的模型

修改了 site-packages/skvideo/io/ffmpeg.py 中的 np.float np.int来使其适应高于 1.20 版本的 numpy

修改了 separator 中的 logging 来使其能够输出到 stdout 被本 repo的 log 记录


前端：视频超分补帧 音频分离降噪 语音识别
后端：视频超分补帧 音频分离降噪 语音识别 本地 llm 视频背景分离 图片超分 图片降噪 图片风格化 

环境部署的指南
汉化
文档
设置页面

### 适配
N 卡 A 卡 在 win 和 linux 下的适配


poetry 虚拟

conda 负责不同版本的 python 安装