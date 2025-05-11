"""
VideoProcessor基类：处理视频的基础功能
"""

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union


class VideoProcessor(ABC):
    """视频处理基类，定义视频处理的通用接口"""
    
    def __init__(self, video_path: Union[str, Path]):
        """
        初始化VideoProcessor
        
        Args:
            video_path: 视频文件路径
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {self.video_path}")
        
    @abstractmethod
    def process(self, *args, **kwargs):
        """
        处理视频的抽象方法，需要被子类实现
        """
        pass
    
    def get_video_info(self) -> Dict[str, Any]:
        """
        获取视频基本信息
        
        Returns:
            包含视频信息的字典
        """
        # 子类可以覆盖此方法以提供更多信息
        return {
            "path": str(self.video_path),
            "name": self.video_path.stem
        }


class FrameExtractor(VideoProcessor):
    """从视频中提取帧"""
    
    def __init__(self, video_path: Union[str, Path], output_dir: Union[str, Path], fps: float = 1):
        """
        初始化帧提取器
        
        Args:
            video_path: 视频文件路径
            output_dir: 提取的帧保存目录
            fps: 每秒提取的帧数
        """
        super().__init__(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        
    def process(self, max_frames: Optional[int] = None, force_sample: bool = False) -> Tuple[np.ndarray, str, float]:
        """
        从视频中提取帧
        
        Args:
            max_frames: 最大帧数，如果指定，则均匀提取不超过此数量的帧
            force_sample: 是否强制采样到max_frames数量
            
        Returns:
            Tuple包含: 
            - 提取的帧 (numpy数组)
            - 帧时间点字符串
            - 视频总时长
        """
        from decord import VideoReader, cpu
        
        if max_frames == 0:
            return np.zeros((1, 336, 336, 3)), "", 0
            
        vr = VideoReader(str(self.video_path), ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        
        # 计算采样间隔
        sample_fps = round(vr.get_avg_fps() / self.fps)
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
        
        # 如果需要限制帧数
        if (len(frame_idx) > max_frames and max_frames is not None) or force_sample:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]
            
        frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
        extracted_frames = vr.get_batch(frame_idx).asnumpy()
        
        return extracted_frames, frame_time_str, video_time
        
    def save_frames(self, frames: np.ndarray, name_prefix: str = "") -> List[Path]:
        """
        保存提取的帧为图片文件
        
        Args:
            frames: 要保存的帧
            name_prefix: 文件名前缀
            
        Returns:
            保存的帧文件路径列表
        """
        from PIL import Image
        
        output_paths = []
        for i, frame in enumerate(frames):
            frame_path = self.output_dir / f"{name_prefix}{i:06d}.jpg"
            Image.fromarray(frame).save(frame_path)
            output_paths.append(frame_path)
            
        return output_paths


class FeatureExtractor(VideoProcessor):
    """从视频中提取特征"""
    
    def __init__(self, video_path: Union[str, Path], output_dir: Union[str, Path]):
        """
        初始化特征提取器
        
        Args:
            video_path: 视频文件路径
            output_dir: 特征保存目录
        """
        super().__init__(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def process(self, *args, **kwargs):
        """
        提取视频特征，具体实现由子类决定
        """
        pass


class TextFeatureExtractor(ABC):
    """文本特征提取器"""
    
    def __init__(self, model_name: str = None):
        """
        初始化文本特征提取器
        
        Args:
            model_name: 使用的模型名称
        """
        self.model_name = model_name
        self.model = None
        
    @abstractmethod
    def load_model(self):
        """加载模型"""
        pass
    
    @abstractmethod
    def process(self, texts: List[str], *args, **kwargs) -> np.ndarray:
        """
        处理文本并提取特征
        
        Args:
            texts: 文本列表
            
        Returns:
            提取的特征
        """
        pass
