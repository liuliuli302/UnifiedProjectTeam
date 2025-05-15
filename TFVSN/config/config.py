"""
视频摘要系统配置文件
"""

import json
from pathlib import Path
import time
import logging
from typing import Dict, Any, Optional, TypeVar, Type, Union

"""
配置对象定义模块
为项目中的每个组件定义对应的配置类
"""

from typing import Dict, List, Any, Optional, Union, Type, TypeVar, ClassVar
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='BaseConfig')


# 通用方法，可以在每个配置类中使用
def config_from_json(cls, json_path: str):
    """从JSON文件创建配置对象"""
    with open(json_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    return cls.from_dict(config_dict)


# PathConfig 已移除，路径将直接作为字典使用


class PipelineConfig:
    """流水线配置"""
    
    pipeline_id: Optional[str] = None
    dataset: str = "all"
    
    def __init__(
        self,
        pipeline_id: Optional[str] = None,
        dataset: str = "all",
        **kwargs
    ):
        """
        初始化流水线配置
        
        Args:
            pipeline_id: 流水线标识符
            dataset: 处理的数据集，可以是 "all", "summe", "tvsum"
            **kwargs: 其他流水线配置参数
        """
        # 设置属性
        if pipeline_id is not None:
            self.pipeline_id = pipeline_id
        if dataset is not None:
            self.dataset = dataset
            
        # 仅设置在类中已定义的属性，避免添加未知属性
        for key, value in kwargs.items():
            if key in self.__class__.__annotations__ or hasattr(self.__class__, key):
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
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """从字典创建流水线配置"""
        return cls(
            pipeline_id=config_dict.get('pipeline_id'),
            dataset=config_dict.get('dataset', "all"),
            **{k: v for k, v in config_dict.items() if k not in ['pipeline_id', 'dataset']}
        )
        
    @classmethod
    def from_json(cls, json_path: str):
        """从JSON文件创建配置对象"""
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class DatasetConfig:
    """数据集配置基类"""
    
    hdf_path: str = ""
    dataset_jump: str = ""
    dataset_turn: str = ""
    eval_method: str = "avg"
    
    def __init__(
        self,
        hdf_path: str = "",
        dataset_jump: str = "",
        dataset_turn: str = "",
        eval_method: str = "avg",
        **kwargs
    ):
        """
        初始化数据集配置
        
        Args:
            hdf_path: HDF文件路径
            dataset_jump: jump数据集路径
            dataset_turn: turn数据集路径
            eval_method: 评估方法
            **kwargs: 其他配置参数
        """
        if hdf_path:
            self.hdf_path = hdf_path
        if dataset_jump:
            self.dataset_jump = dataset_jump
        if dataset_turn:
            self.dataset_turn = dataset_turn
        if eval_method:
            self.eval_method = eval_method
            
        # 仅设置在类中已定义的属性
        for key, value in kwargs.items():
            if key in self.__class__.__annotations__ or hasattr(self.__class__, key):
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
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """从字典创建数据集配置"""
        return cls(
            hdf_path=config_dict.get('hdf_path', ""),
            dataset_jump=config_dict.get('dataset_jump', ""),
            dataset_turn=config_dict.get('dataset_turn', ""),
            eval_method=config_dict.get('eval_method', "avg"),
            **{k: v for k, v in config_dict.items() 
               if k not in ['hdf_path', 'dataset_jump', 'dataset_turn', 'eval_method']}
        )
        
    @classmethod
    def from_json(cls, json_path: str):
        """从JSON文件创建配置对象"""
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class SumMeConfig:
    """SumMe数据集配置"""
    
    hdf_path: str = "SumMe/summe.h5"
    dataset_jump: str = "SumMe/summe_dataset_jump.json"
    dataset_turn: str = "SumMe/summe_dataset_turn.json"
    eval_method: str = "max"
    
    def __init__(
        self,
        hdf_path: str = "SumMe/summe.h5",
        dataset_jump: str = "SumMe/summe_dataset_jump.json",
        dataset_turn: str = "SumMe/summe_dataset_turn.json",
        eval_method: str = "max",
        **kwargs
    ):
        """
        初始化SumMe数据集配置
        
        Args:
            hdf_path: HDF文件路径
            dataset_jump: jump数据集路径
            dataset_turn: turn数据集路径
            eval_method: 评估方法
            **kwargs: 其他配置参数
        """
        self.hdf_path = hdf_path
        self.dataset_jump = dataset_jump
        self.dataset_turn = dataset_turn
        self.eval_method = eval_method
        
        # 仅设置在类中已定义的属性
        for key, value in kwargs.items():
            if key in self.__class__.__annotations__ or hasattr(self.__class__, key):
                setattr(self, key, value)


class TVSumConfig:
    """TVSum数据集配置"""
    
    hdf_path: str = "TVSum/tvsum.h5"
    dataset_jump: str = "TVSum/tvsum_dataset_jump.json"
    dataset_turn: str = "TVSum/tvsum_dataset_turn.json"
    eval_method: str = "avg"
    
    def __init__(
        self,
        hdf_path: str = "TVSum/tvsum.h5",
        dataset_jump: str = "TVSum/tvsum_dataset_jump.json",
        dataset_turn: str = "TVSum/tvsum_dataset_turn.json",
        eval_method: str = "avg",
        **kwargs
    ):
        """
        初始化TVSum数据集配置
        
        Args:
            hdf_path: HDF文件路径
            dataset_jump: jump数据集路径
            dataset_turn: turn数据集路径
            eval_method: 评估方法
            **kwargs: 其他配置参数
        """
        self.hdf_path = hdf_path
        self.dataset_jump = dataset_jump
        self.dataset_turn = dataset_turn
        self.eval_method = eval_method
        
        # 仅设置在类中已定义的属性
        for key, value in kwargs.items():
            if key in self.__class__.__annotations__ or hasattr(self.__class__, key):
                setattr(self, key, value)


class ResultsConfig:
    """结果输出配置"""
    
    llm_output_dir_relative: str = "raw"
    scores_dir_relative: str = "scores"
    similarity_scores_dir_relative: str = "similarity_scores"
    f1score_dir_relative: str = "f1score"
    
    def __init__(
        self,
        llm_output_dir_relative: str = "raw",
        scores_dir_relative: str = "scores",
        similarity_scores_dir_relative: str = "similarity_scores",
        f1score_dir_relative: str = "f1score",
        **kwargs
    ):
        """
        初始化结果输出配置
        
        Args:
            llm_output_dir_relative: LLM输出相对路径
            scores_dir_relative: 分数输出相对路径
            similarity_scores_dir_relative: 相似度分数输出相对路径
            f1score_dir_relative: F1分数输出相对路径
            **kwargs: 其他配置参数
        """
        if llm_output_dir_relative:
            self.llm_output_dir_relative = llm_output_dir_relative
        if scores_dir_relative:
            self.scores_dir_relative = scores_dir_relative
        if similarity_scores_dir_relative:
            self.similarity_scores_dir_relative = similarity_scores_dir_relative
        if f1score_dir_relative:
            self.f1score_dir_relative = f1score_dir_relative
            
        # 仅设置在类中已定义的属性
        for key, value in kwargs.items():
            if key in self.__class__.__annotations__ or hasattr(self.__class__, key):
                setattr(self, key, value)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """从字典创建结果输出配置"""
        return cls(
            llm_output_dir_relative=config_dict.get('llm_output_dir_relative', "raw"),
            scores_dir_relative=config_dict.get('scores_dir_relative', "scores"),
            similarity_scores_dir_relative=config_dict.get(
                'similarity_scores_dir_relative', "similarity_scores"
            ),
            f1score_dir_relative=config_dict.get('f1score_dir_relative', "f1score"),
            **{k: v for k, v in config_dict.items() 
               if k not in ['llm_output_dir_relative', 'scores_dir_relative', 
                           'similarity_scores_dir_relative', 'f1score_dir_relative']}
        )


class ModelConfig:
    """模型配置基类"""
    
    model_name: str = ""
    device: str = "cuda"
    
    def __init__(
        self,
        model_name: str = "",
        device: str = "cuda",
        **kwargs
    ):
        """
        初始化模型配置
        
        Args:
            model_name: 模型名称
            device: 设备
            **kwargs: 其他模型配置参数
        """
        if model_name:
            self.model_name = model_name
        if device:
            self.device = device
        
        # 仅设置在类中已定义的属性
        for key, value in kwargs.items():
            if key in self.__class__.__annotations__ or hasattr(self.__class__, key):
                setattr(self, key, value)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """从字典创建模型配置"""
        return cls(
            model_name=config_dict.get('model_name', ""),
            device=config_dict.get('device', "cuda"),
            **{k: v for k, v in config_dict.items() if k not in ['model_name', 'device']}
        )


class LLMConfig:
    """LLM模型配置"""
    
    model_name: str = "gpt-3.5-turbo"
    device: str = "cuda"
    api_key: str = ""
    api_base: str = ""
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9
    history_len: int = 5
    system_prompt_template: str = ""
    user_prompt_template: str = ""
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        device: str = "cuda",
        api_key: str = "",
        api_base: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 0.9,
        history_len: int = 5,
        system_prompt_template: str = "",
        user_prompt_template: str = "",
        **kwargs
    ):
        """
        初始化LLM配置
        
        Args:
            model_name: 模型名称
            device: 设备
            api_key: API密钥
            api_base: API基础URL
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: Top p参数
            history_len: 历史记录长度
            system_prompt_template: 系统提示模板
            user_prompt_template: 用户提示模板
            **kwargs: 其他LLM配置参数
        """
        if model_name:
            self.model_name = model_name
        if device:
            self.device = device
        
        if api_key:
            self.api_key = api_key
        if api_base:
            self.api_base = api_base
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if top_p is not None:
            self.top_p = top_p
        if history_len is not None:
            self.history_len = history_len
        if system_prompt_template:
            self.system_prompt_template = system_prompt_template
        if user_prompt_template:
            self.user_prompt_template = user_prompt_template
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """从字典创建LLM配置"""
        return cls(
            model_name=config_dict.get('model_name', "gpt-3.5-turbo"),
            api_key=config_dict.get('api_key', ""),
            api_base=config_dict.get('api_base', ""),
            temperature=config_dict.get('temperature', 0.7),
            max_tokens=config_dict.get('max_tokens', 1000),
            top_p=config_dict.get('top_p', 0.9),
            history_len=config_dict.get('history_len', 5),
            system_prompt_template=config_dict.get('system_prompt_template', ""),
            user_prompt_template=config_dict.get('user_prompt_template', ""),
            device=config_dict.get('device', "cuda"),
            **{k: v for k, v in config_dict.items() if k not in [
                'model_name', 'api_key', 'api_base', 'temperature', 
                'max_tokens', 'top_p', 'history_len', 
                'system_prompt_template', 'user_prompt_template', 'device'
            ]}
        )


class BlipConfig:
    """BLIP模型配置"""
    
    model_name: str = "blip2"
    processor_name: str = "Salesforce/blip2-opt-2.7b"
    model_path: str = "Salesforce/blip2-opt-2.7b"
    device: str = "cuda"
    
    def __init__(
        self,
        model_name: str = "blip2",
        processor_name: str = "Salesforce/blip2-opt-2.7b",
        model_path: str = "Salesforce/blip2-opt-2.7b",
        device: str = "cuda",
        **kwargs
    ):
        """
        初始化BLIP配置
        
        Args:
            model_name: 模型名称
            processor_name: 处理器名称
            model_path: 模型路径
            device: 设备
            **kwargs: 其他BLIP配置参数
        """
        if model_name:
            self.model_name = model_name
        if device:
            self.device = device
        
        if processor_name:
            self.processor_name = processor_name
        if model_path:
            self.model_path = model_path
            
        # 仅设置在类中已定义的属性
        for key, value in kwargs.items():
            if key in self.__class__.__annotations__ or hasattr(self.__class__, key):
                setattr(self, key, value)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """从字典创建BLIP配置"""
        return cls(
            model_name=config_dict.get('model_name', "blip2"),
            processor_name=config_dict.get('processor_name', "Salesforce/blip2-opt-2.7b"),
            model_path=config_dict.get('model_path', "Salesforce/blip2-opt-2.7b"),
            device=config_dict.get('device', "cuda"),
            **{k: v for k, v in config_dict.items() 
               if k not in ['model_name', 'processor_name', 'model_path', 'device']}
        )


class ExtractionConfig:
    """特征提取配置基类"""
    
    fps: int = 1
    
    def __init__(
        self,
        fps: int = 1,
        **kwargs
    ):
        """
        初始化特征提取配置
        
        Args:
            fps: 每秒帧数
            **kwargs: 其他特征提取配置参数
        """
        if fps is not None:
            self.fps = fps
        
        # 仅设置在类中已定义的属性
        for key, value in kwargs.items():
            if key in self.__class__.__annotations__ or hasattr(self.__class__, key):
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
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """从字典创建特征提取配置"""
        return cls(
            fps=config_dict.get('fps', 1),
            **{k: v for k, v in config_dict.items() if k not in ['fps']}
        )
        
    @classmethod
    def from_json(cls, json_path: str):
        """从JSON文件创建配置对象"""
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class FrameExtractionConfig:
    """帧提取配置"""
    
    fps: int = 1
    frame_height: int = 224
    frame_width: int = 224
    
    def __init__(
        self,
        fps: int = 1,
        frame_height: int = 224,
        frame_width: int = 224,
        **kwargs
    ):
        """
        初始化帧提取配置
        
        Args:
            fps: 每秒帧数
            frame_height: 帧高度
            frame_width: 帧宽度
            **kwargs: 其他帧提取配置参数
        """
        if fps is not None:
            self.fps = fps
        if frame_height is not None:
            self.frame_height = frame_height
        if frame_width is not None:
            self.frame_width = frame_width
            
        # 仅设置在类中已定义的属性
        for key, value in kwargs.items():
            if key in self.__class__.__annotations__ or hasattr(self.__class__, key):
                setattr(self, key, value)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """从字典创建帧提取配置"""
        return cls(
            fps=config_dict.get('fps', 1),
            frame_height=config_dict.get('frame_height', 224),
            frame_width=config_dict.get('frame_width', 224),
            **{k: v for k, v in config_dict.items() 
               if k not in ['fps', 'frame_height', 'frame_width']}
        )


class FeatureExtractionConfig(ExtractionConfig):
    """特征提取配置"""
    
    fps: int = 1
    feature_type: str = "blip"
    
    def __init__(
        self,
        fps: int = 1,
        feature_type: str = "blip",
        **kwargs
    ):
        """
        初始化特征提取配置
        
        Args:
            fps: 每秒帧数
            feature_type: 特征类型
            **kwargs: 其他特征提取配置参数
        """
        super().__init__(fps=fps, **kwargs)
        
        if feature_type:
            self.feature_type = feature_type
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """从字典创建特征提取配置"""
        return cls(
            fps=config_dict.get('fps', 1),
            feature_type=config_dict.get('feature_type', "blip"),
            **{k: v for k, v in config_dict.items() if k not in ['fps', 'feature_type']}
        )


class SummarizationConfig(BaseConfig):
    """摘要生成配置"""
    
    summary_method: str = "llm"
    summary_length: int = 200
    key_frame_ratio: float = 0.15
    window_size: int = 60
    stride: int = 30
    
    def __init__(
        self,
        summary_method: str = "llm",
        summary_length: int = 200,
        key_frame_ratio: float = 0.15,
        window_size: int = 60,
        stride: int = 30,
        **kwargs
    ):
        """
        初始化摘要生成配置
        
        Args:
            summary_method: 摘要方法
            summary_length: 摘要长度
            key_frame_ratio: 关键帧比例
            window_size: 窗口大小
            stride: 步长
            **kwargs: 其他摘要生成配置参数
        """
        if summary_method:
            self.summary_method = summary_method
        if summary_length is not None:
            self.summary_length = summary_length
        if key_frame_ratio is not None:
            self.key_frame_ratio = key_frame_ratio
        if window_size is not None:
            self.window_size = window_size
        if stride is not None:
            self.stride = stride
            
        super().__init__(**kwargs)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """从字典创建摘要生成配置"""
        return cls(
            summary_method=config_dict.get('summary_method', "llm"),
            summary_length=config_dict.get('summary_length', 200),
            key_frame_ratio=config_dict.get('key_frame_ratio', 0.15),
            window_size=config_dict.get('window_size', 60),
            stride=config_dict.get('stride', 30),
            **{k: v for k, v in config_dict.items() 
               if k not in ['summary_method', 'summary_length', 
                           'key_frame_ratio', 'window_size', 'stride']}
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
