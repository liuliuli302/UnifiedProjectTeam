"""
LLM处理器：处理LLM相关的操作
"""

import torch
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import numpy as np
from PIL import Image

from ..core.model_interface import LLMInterface
from ..config.settings import LLM_CONFIG


class LLMProcessor(LLMInterface):
    """LLaVA模型处理器"""
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        初始化LLaVA处理器
        
        Args:
            model_name: 模型名称，默认使用配置中的名称
            device: 运行设备，默认使用配置中的设备
        """
        super().__init__(
            model_name or LLM_CONFIG["model_name"],
            device or LLM_CONFIG["device"]
        )
        self.conv_template = LLM_CONFIG["conv_template"]
        
    def load_model(self) -> None:
        """加载LLaVA模型"""
        from llava.model.builder import load_pretrained_model
        from llava.conversation import conv_templates
        
        # 加载模型和分词器
        self.tokenizer, self.model, self.processor, _ = load_pretrained_model(
            self.model_name,
            None,
            LLM_CONFIG["model_type"],
            torch_dtype="bfloat16",
            device_map="auto"
        )
        
        self.model = self.model.eval()
        self.conversation = conv_templates[self.conv_template]
        
    def encode_images(self, images: Union[List[Path], List[np.ndarray]]) -> torch.Tensor:
        """
        编码图像
        
        Args:
            images: 图像路径或数组列表
            
        Returns:
            编码后的图像张量
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
        ).to(self.device, torch.bfloat16)
        
        return inputs.pixel_values
        
    def generate(self, 
                prompt: str,
                images: Optional[List[Union[Path, np.ndarray]]] = None,
                **kwargs) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            images: 可选的图像输入
            
        Returns:
            生成的文本
        """
        from llava.mm_utils import tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX
        
        if not self.is_model_loaded():
            self.load_model()
            
        # 处理图像
        if images is not None:
            processed_images = [self.encode_images(images)]
        else:
            processed_images = None
            
        # 准备输入
        conv = self.conversation.copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        ).unsqueeze(0).to(self.device)
        
        # 生成文本
        outputs = self.model.generate(
            input_ids,
            images=processed_images,
            do_sample=False,
            temperature=0,
            max_new_tokens=500,
            **kwargs
        )
        
        response = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )[0].strip()
        
        return response
        
    def process(self, 
               prompt: str,
               images: Optional[List[Union[Path, np.ndarray]]] = None,
               **kwargs) -> str:
        """
        使用模型处理输入
        
        Args:
            prompt: 输入提示
            images: 可选的图像输入
            
        Returns:
            处理后的文本
        """
        # 这个方法是对generate方法的包装，以实现抽象接口
        return self.generate(prompt, images, **kwargs)
