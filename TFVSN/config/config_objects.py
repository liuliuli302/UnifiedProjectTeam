"""
配置对象定义模块
为项目中的每个组件定义对应的配置类
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path


class BaseConfig:
    """基础配置类，所有配置类的父类"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        初始化配置对象
        
        Args:
            config_dict: 包含配置参数的字典
        """
        for key, value in config_dict.items():
            setattr(self, key, value)
    
    def __getitem__(self, key):
        """支持字典访问方式"""
        return getattr(self, key)
    
    def __contains__(self, key):
        """支持 in 操作符"""
        return hasattr(self, key)
    
    def get(self, key, default=None):
        """获取配置项，支持默认值"""
        return getattr(self, key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置对象转换为字典"""
        return {k: v for k, v in self.__dict__.items()}


class PathConfig(BaseConfig):
    """路径配置"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        初始化路径配置
        
        Args:
            config_dict: 包含路径配置的字典
        """
        super().__init__(config_dict)
        
        # 确保路径对象是Path类型
        if hasattr(self, 'root_dir') and self.root_dir:
            self.root_dir = Path(self.root_dir)
        if hasattr(self, 'dataset_dir') and self.dataset_dir:
            self.dataset_dir = Path(self.dataset_dir)
        if hasattr(self, 'output_dir') and self.output_dir:
            self.output_dir = Path(self.output_dir)


class PipelineConfig(BaseConfig):
    """流水线配置"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        初始化流水线配置
        
        Args:
            config_dict: 包含流水线配置的字典
        """
        super().__init__(config_dict)


class DatasetConfig(BaseConfig):
    """数据集配置基类"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        初始化数据集配置
        
        Args:
            config_dict: 包含数据集配置的字典
        """
        super().__init__(config_dict)


class SumMeConfig(DatasetConfig):
    """SumMe数据集配置"""
    pass


class TVSumConfig(DatasetConfig):
    """TVSum数据集配置"""
    pass


class ResultsConfig(BaseConfig):
    """结果输出配置"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        初始化结果输出配置
        
        Args:
            config_dict: 包含结果输出配置的字典
        """
        super().__init__(config_dict)


class ModelConfig(BaseConfig):
    """模型配置基类"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        初始化模型配置
        
        Args:
            config_dict: 包含模型配置的字典
        """
        super().__init__(config_dict)


class LLMConfig(ModelConfig):
    """LLM模型配置"""
    pass


class BlipConfig(ModelConfig):
    """BLIP模型配置"""
    pass


class ExtractionConfig(BaseConfig):
    """特征提取配置基类"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        初始化特征提取配置
        
        Args:
            config_dict: 包含特征提取配置的字典
        """
        super().__init__(config_dict)


class FrameExtractionConfig(ExtractionConfig):
    """帧提取配置"""
    pass


class FeatureExtractionConfig(ExtractionConfig):
    """特征提取配置"""
    pass


class SummarizationConfig(BaseConfig):
    """摘要生成配置"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        初始化摘要生成配置
        
        Args:
            config_dict: 包含摘要生成配置的字典
        """
        super().__init__(config_dict)
