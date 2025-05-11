"""
模型接口：定义与各种模型交互的标准接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union, Optional, Tuple
from pathlib import Path
import torch
import numpy as np


class ModelInterface(ABC):
    """模型接口基类"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        """
        初始化模型接口
        
        Args:
            model_name: 模型名称或路径
            device: 运行设备
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        
    @abstractmethod
    def load_model(self) -> None:
        """
        加载模型
        """
        pass
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """
        使用模型处理输入
        """
        pass
    
    def is_model_loaded(self) -> bool:
        """
        检查模型是否已加载
        
        Returns:
            模型是否已加载
        """
        return self.model is not None
    
    def to(self, device: str) -> 'ModelInterface':
        """
        将模型移动到指定设备
        
        Args:
            device: 设备名称
            
        Returns:
            模型接口自身
        """
        if self.model is not None:
            self.device = device
            self.model.to(device)
        return self


class LLMInterface(ModelInterface):
    """大型语言模型接口"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        """
        初始化LLM接口
        
        Args:
            model_name: 模型名称或路径
            device: 运行设备
        """
        super().__init__(model_name, device)
        self.tokenizer = None
        
    @abstractmethod
    def generate(self, prompt: str, *args, **kwargs) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            
        Returns:
            生成的文本
        """
        pass
    
    @abstractmethod
    def encode_images(self, images: Union[List[Path], List[np.ndarray]]) -> Any:
        """
        编码图像
        
        Args:
            images: 图像路径或数组列表
            
        Returns:
            编码后的图像表示
        """
        pass


class VisionEncoderInterface(ModelInterface):
    """视觉编码器接口"""
    
    @abstractmethod
    def encode(self, images: Union[List[Path], List[np.ndarray]]) -> np.ndarray:
        """
        编码图像
        
        Args:
            images: 图像路径或数组列表
            
        Returns:
            编码后的特征
        """
        pass


class TextEncoderInterface(ModelInterface):
    """文本编码器接口"""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        编码文本
        
        Args:
            texts: 文本列表
            
        Returns:
            编码后的特征
        """
        pass
