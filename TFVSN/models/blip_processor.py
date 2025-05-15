"""
BLIP处理器：处理BLIP模型相关的操作
"""

import torch
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import numpy as np
from PIL import Image
from transformers import Blip2Processor, Blip2Model

from ..config import BlipConfig


class BlipProcessor:
    """BLIP模型处理器"""
    
    def __init__(self, config: BlipConfig = None, model_name: str = None, device: str = None):
        """
        初始化BLIP处理器
        
        Args:
            config: BLIP模型配置对象
            model_name: 模型名称，如果提供则覆盖配置中的名称
            device: 运行设备，如果提供则覆盖配置中的设备
        """
        # 获取配置
        if config is None:
            from ..config.config import load_config_from_json
            config_data = load_config_from_json()
            self.config = BlipConfig(config_data.get('models', {}).get('blip', {}))
        else:
            self.config = config
        
        # 模型名称和设备可以被参数覆盖
        self.model_name = model_name or self.config.model_name
        self.device = device or self.config.device
        self.model = None
        self.processor = None
        
    def load_model(self) -> None:
        """加载BLIP模型"""
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        self.model = Blip2Model.from_pretrained(
            self.model_name,
            torch_dtype=getattr(torch, self.config.dtype)
        )
        self.model.to(self.device)
        self.model.eval()
        
    def process_images(self, images: Union[List[Path], List[np.ndarray]]) -> torch.Tensor:
        """
        处理图像
        
        Args:
            images: 图像路径或数组列表
            
        Returns:
            处理后的图像张量
        """
        # 确保图像是PIL Image格式
        pil_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_images.append(Image.open(img).convert('RGB'))
            elif isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img).convert('RGB'))
            else:
                pil_images.append(img)
                
        # 处理图像
        inputs = self.processor(
            images=pil_images,
            return_tensors="pt"
        ).to(self.device)
        
        return inputs.pixel_values
        
    def encode(self, images: Union[List[Path], List[np.ndarray]]) -> np.ndarray:
        """
        编码图像
        
        Args:
            images: 图像路径或数组列表
            
        Returns:
            编码后的特征
        """
        if not self.is_model_loaded():
            self.load_model()
            
        with torch.no_grad():
            # 处理图像
            pixel_values = self.process_images(images)
            
            # 获取视觉特征
            vision_outputs = self.model.vision_model(pixel_values)
            image_embeds = vision_outputs[0]
            
            # 准备注意力掩码
            image_attention_mask = torch.ones(
                image_embeds.size()[:-1],
                dtype=torch.long,
                device=image_embeds.device
            )
            
            # 获取query tokens
            query_tokens = self.model.query_tokens.expand(
                image_embeds.shape[0], -1, -1
            )
            
            # 使用Qformer处理
            qformer_outputs = self.model.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True
            )
            
            query_output = qformer_outputs.last_hidden_state
            
            # 投影到语言空间
            projected_features = self.model.language_projection(query_output)
            
            if projected_features.dtype != image_embeds.dtype:
                projected_features = projected_features.to(image_embeds.dtype)
                
            return projected_features.cpu().numpy()
            
    def process(self, images: Union[List[Path], List[np.ndarray]], *args, **kwargs) -> np.ndarray:
        """
        使用模型处理输入
        
        Args:
            images: 图像路径或数组列表
            
        Returns:
            处理后的特征
        """
        # 这个方法是对encode方法的包装，以实现抽象接口
        return self.encode(images)
