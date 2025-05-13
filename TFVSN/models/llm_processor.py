"""
LLM处理器：处理LLM相关的操作
"""

import torch
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import numpy as np
from PIL import Image

from ..core.model_interface import LLMInterface
from ..config import LLMConfig


class LLMProcessor(LLMInterface):
    """LLaVA模型处理器"""
    
    def __init__(self, config: LLMConfig = None, model_name: str = None, device: str = None):
        """
        初始化LLaVA处理器
        
        Args:
            config: LLM模型配置对象
            model_name: 模型名称，如果提供则覆盖配置中的名称
            device: 运行设备，如果提供则覆盖配置中的设备
        """
        # 获取配置
        if config is None:
            from ..config.config import load_config_from_json
            config_data = load_config_from_json()
            self.config = LLMConfig(config_data.get('models', {}).get('llm', {}))
        else:
            self.config = config
        
        # 模型名称和设备可以被参数覆盖
        model_name_to_use = model_name or self.config.model_name
        device_to_use = device or self.config.device
        
        super().__init__(model_name_to_use, device_to_use)
        self.conv_template = self.config.conv_template
        
    def load_model(self) -> None:
        """加载LLaVA模型"""
        from llava.model.builder import load_pretrained_model
        from llava.conversation import conv_templates
        
        # 加载模型和分词器
        self.tokenizer, self.model, self.processor, _ = load_pretrained_model(
            self.model_name,
            None,
            self.config.model_type,
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
