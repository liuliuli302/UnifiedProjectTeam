"""
配置包初始化模块
提供配置对象的访问接口
"""

from .config import load_config_from_json, load_config_object, create_config_from_json

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

# 添加便捷的配置加载函数
def get_config(config_path="default_config.json"):
    """
    加载配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        包含所有配置对象的字典
    """
    return create_config_from_json(config_path)