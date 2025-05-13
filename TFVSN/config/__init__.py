"""
配置包初始化模块
提供配置对象的访问接口
"""

from .config import (
    load_config_from_json
)

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