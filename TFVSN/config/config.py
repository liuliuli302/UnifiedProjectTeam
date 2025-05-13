"""
视频摘要系统配置文件
"""

import json
from pathlib import Path
import time
import logging
from typing import Dict, Any, Optional, TypeVar, Type, Union

from .config_objects import (
    BaseConfig,
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

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseConfig)

def process_paths(paths_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理配置中的路径，将字符串路径转换为Path对象
    
    Args:
        paths_dict: 包含路径配置的字典
        
    Returns:
        处理后的路径字典，字符串路径已转换为Path对象
    """
    result = {}
    for key, value in paths_dict.items():
        if value is not None and isinstance(value, str):
            result[key] = Path(value)
        else:
            result[key] = value
    return result

def load_config_from_json(config_path="default_config.json") -> Dict[str, Any]:
    """从JSON文件简单加载原始配置数据"""
    absolute_config_path = (Path(__file__).parent / config_path).resolve()
    with open(absolute_config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    return config_data


def load_config_object(
    config_type: Type[T], 
    config_dict: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = "default_config.json"
) -> T:
    """加载指定类型的配置对象"""
    if config_dict is None:
        config_dict = load_config_from_json(config_path)
    
    # 通过各自类的from_dict方法创建配置对象
    return config_type.from_dict(config_dict)


def create_config_from_json(config_path="default_config.json"):
    """从JSON文件创建完整的配置对象"""
    config_dict = load_config_from_json(config_path)
    
    # 处理并转换路径字典中的路径为Path对象
    paths_config = process_paths(config_dict.get("paths", {}))
    
    # 创建流水线配置
    pipeline_config = PipelineConfig.from_dict(config_dict.get("pipeline", {}))
    
    # 创建数据集配置
    datasets_dict = config_dict.get("datasets", {})
    summe_config = SumMeConfig.from_dict(datasets_dict.get("summe", {}))
    tvsum_config = TVSumConfig.from_dict(datasets_dict.get("tvsum", {}))
    
    # 创建结果输出配置
    results_config = ResultsConfig.from_dict(config_dict.get("results_output", {}))
    
    # 创建模型配置
    models_dict = config_dict.get("models", {})
    llm_config = LLMConfig.from_dict(models_dict.get("llm", {}))
    blip_config = BlipConfig.from_dict(models_dict.get("blip", {}))
    
    # 创建特征提取配置
    extraction_dict = config_dict.get("extraction", {})
    frame_extraction_config = FrameExtractionConfig.from_dict(extraction_dict.get("frame", {}))
    feature_extraction_config = FeatureExtractionConfig.from_dict(extraction_dict.get("feature", {}))
    
    # 创建摘要配置
    summary_config = SummarizationConfig.from_dict(config_dict.get("summarization", {}))
    
    # 返回配置字典
    return {
        "model_type": config_dict.get("model_type", "video_summarization"),
        "paths": paths_config,
        "pipeline": pipeline_config,
        "datasets": {
            "summe": summe_config,
            "tvsum": tvsum_config
        },
        "results_output": results_config,
        "models": {
            "llm": llm_config,
            "blip": blip_config
        },
        "extraction": {
            "frame": frame_extraction_config,
            "feature": feature_extraction_config
        },
        "summarization": summary_config
    }
