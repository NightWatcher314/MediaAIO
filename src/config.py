import os
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

_tmp_logs_dir = os.path.join(base_logs_dir, "tmp")
if not os.path.exists(_tmp_logs_dir):
    os.makedirs(_tmp_logs_dir)
logger = Logger(_tmp_logs_dir)
print(f"Logger initialized with logs dir: {_tmp_logs_dir}")
