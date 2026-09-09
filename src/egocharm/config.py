"""读取项目配置，并提供训练需要的基础工具。"""

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_SECTIONS = ("data", "model", "training")


def load_config(path: Path) -> dict[str, Any]:
    """读取 YAML 配置，并检查三个主要配置区域是否存在。"""
    config_path = Path(path)
    content = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(content)

    if not isinstance(config, dict):
        raise ValueError(f"配置文件顶层必须是字典: {config_path}")

    for section_name in REQUIRED_SECTIONS:
        if section_name not in config:
            raise ValueError(f"配置文件缺少 {section_name} 区域: {config_path}")
        if not isinstance(config[section_name], dict):
            raise ValueError(f"配置区域 {section_name} 必须是字典")

    return config


def project_path(value: str | Path) -> Path:
    """把相对路径转换成相对于项目根目录的绝对路径。"""
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def set_random_seed(seed: int):
    """统一设置 Python、NumPy 和 PyTorch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
