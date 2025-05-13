"""
视频摘要系统配置文件
"""

import json
from pathlib import Path
import time
from typing import Dict, Any, Optional, TypeVar, Type, Union

from .config_objects import (
    BaseConfig,
    PathConfig,
    PipelineConfig,
    DatasetConfig,
    SumMeConfig,
    TVSumConfig,
    ResultsConfig,
    ModelConfig,
    LLMConfig,
    BlipConfig,
    ExtractionConfig,
    FrameExtractionConfig,
    FeatureExtractionConfig,
    SummarizationConfig
)

def load_config_from_json(config_path="default_config.json") -> Dict[str, Any]:
    """从JSON文件简单加载原始配置数据"""
    absolute_config_path = (Path(__file__).parent / config_path).resolve()
    with open(absolute_config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    return config_data
