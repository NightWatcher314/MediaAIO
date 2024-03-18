import os
import sys
from utils.logger import Logger


base_logs_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
)

base_models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)

base_work_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)

platform = sys.platform

_tmp_logs_dir = os.path.join(base_logs_dir, "all")
if not os.path.exists(_tmp_logs_dir):
    os.makedirs(_tmp_logs_dir)
logger = Logger(_tmp_logs_dir, logger_level="INFO")
logger.info("Logger 在 {} 初始化".format(_tmp_logs_dir))
logger.info("平台：{}".format(platform))
