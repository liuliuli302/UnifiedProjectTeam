"""
BLIP处理器：处理BLIP模型相关的操作
"""

import torch
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import numpy as np
from PIL import Image
from transformers import Blip2Processor, Blip2Model

from ..core.model_interface import VisionEncoderInterface
from ..config.settings import BLIP_CONFIG


class BlipProcessor(VisionEncoderInterface):
    """BLIP模型处理器"""
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        初始化BLIP处理器
        
        Args:
            model_name: 模型名称，默认使用配置中的名称
            device: 运行设备，默认使用配置中的设备
        """
        super().__init__(
            model_name or BLIP_CONFIG["model_name"],
            device or BLIP_CONFIG["device"]
        )
        
    def load_model(self) -> None:
        """加载BLIP模型"""
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        self.model = Blip2Model.from_pretrained(
            self.model_name,
            torch_dtype=getattr(torch, BLIP_CONFIG["dtype"])
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
