"""
DatasetBuilder: 数据集构建和管理
"""

from abc import ABC, abstractmethod
from pathlib import Path
import json
import h5py
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple
import os


class DatasetBuilder(ABC):
    """数据集构建器基类"""
    
    def __init__(self, dataset_dir: Union[str, Path]):
        """
        初始化数据集构建器
        
        Args:
            dataset_dir: 数据集目录
        """
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {self.dataset_dir}")
            
    @abstractmethod
    def load_data(self) -> Dict[str, Any]:
        """
        加载数据集
        
        Returns:
            数据集内容
        """
        pass
    
    @abstractmethod
    def save_data(self, data: Dict[str, Any], output_path: Union[str, Path] = None) -> Dict[str, Any]:
        """
        处理数据集（不再保存到文件）
        
        Args:
            data: 数据集
            output_path: 仅为兼容性保留，不再使用
            
        Returns:
            处理后的数据
        """
        pass
    
    @staticmethod
    def hdf5_to_dict(hdf5_file: Union[str, Path]) -> Dict[str, Any]:
        """
        将HDF5文件转换为字典
        
        Args:
            hdf5_file: HDF5文件路径
            
        Returns:
            转换后的字典
        """
        def recursively_convert_to_dict(h5_obj):
            if isinstance(h5_obj, h5py.Dataset):
                return h5_obj[()]
            elif isinstance(h5_obj, h5py.Group):
                return {key: recursively_convert_to_dict(h5_obj[key]) for key in h5_obj.keys()}
            else:
                raise TypeError(f"不支持的类型: {type(h5_obj)}")

        with h5py.File(hdf5_file, "r") as f:
            return recursively_convert_to_dict(f)
            
    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        加载JSON文件
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            JSON内容
        """
        with open(file_path, 'r') as f:
            return json.load(f)
            
    @staticmethod
    def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 4) -> None:
        """
        保存为JSON文件
        
        Args:
            data: 数据
            file_path: 保存路径
            indent: 缩进
        """
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)


class VideoSummarizationDataset(DatasetBuilder):
    """视频摘要数据集基类"""
    
    def __init__(self, dataset_dir: Union[str, Path], dataset_name: str):
        """
        初始化视频摘要数据集
        
        Args:
            dataset_dir: 数据集目录
            dataset_name: 数据集名称
        """
        super().__init__(dataset_dir)
        self.dataset_name = dataset_name
        self.frames_dir = Path(self.dataset_dir, dataset_name, "frames")
        self.data_list = {}
        self.video_name_dict = {}
        
    def load_data(self) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        加载数据集
        
        Returns:
            Tuple包含:
            - 数据列表
            - 视频名称字典
        """
        raise NotImplementedError("子类必须实现此方法")
        
    def get_score_index_from_image_list(self, image_list: List[str], frame_interval: int) -> List[int]:
        """
        从图像路径列表中获取分数索引
        
        Args:
            image_list: 图像路径列表
            frame_interval: 帧间隔
            
        Returns:
            分数索引列表
        """
        score_index = []
        for path in image_list:
            index = int(int(os.path.basename(path).split(".")[0]) / frame_interval)
            score_index.append(index)
        return score_index
        
    def get_video_paths(self) -> List[Path]:
        """
        获取数据集中所有视频的路径
        
        Returns:
            视频路径列表
        """
        raise NotImplementedError("子类必须实现此方法")


class SumMeDataset(VideoSummarizationDataset):
    """SumMe数据集"""
    
    def __init__(self, dataset_dir: Union[str, Path], config=None):
        """
        初始化SumMe数据集
        
        Args:
            dataset_dir: 数据集目录
            config: 数据集配置对象
        """
        super().__init__(dataset_dir, "SumMe")
        
        # 直接从JSON加载配置
        if config is None:
            from ..config.config import load_config_from_json
            from ..config import SumMeConfig
            config_data = load_config_from_json()
            self.config = SumMeConfig(config_data.get('datasets', {}).get('summe', {}))
        else:
            self.config = config
            
        self.hdf_path = self.config.hdf_path
        self.eval_method = self.config.eval_method
        
        # 内存缓存
        self.jump_dataset = None
        self.turn_dataset = None
        
    def load_data(self) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        加载SumMe数据集
        
        Returns:
            Tuple包含:
            - 数据列表
            - 视频名称字典
        """
        # 加载HDF5数据
        data_dict = self.hdf5_to_dict(self.hdf_path)
        
        # 构建视频名称字典
        video_name_dict = {}
        for i, video_name in enumerate(data_dict.keys()):
            video_name_dict[f"video_{i+1}"] = video_name
            
        self.data_list = data_dict
        self.video_name_dict = video_name_dict
        
        return data_dict, video_name_dict
        
    def get_video_paths(self) -> List[Path]:
        """
        获取SumMe数据集中所有视频的路径
        
        Returns:
            视频路径列表
        """
        video_dir = Path(self.dataset_dir, "SumMe", "videos")
        if not video_dir.exists():
            raise FileNotFoundError(f"视频目录不存在: {video_dir}")
            
        return list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mov"))
        
    def get_jump_dataset(self) -> Dict[str, Any]:
        """
        获取jump数据集
        
        Returns:
            jump数据集
        """
        # 如果内存中已有缓存，直接返回
        if self.jump_dataset is not None:
            return self.jump_dataset
        
        # 第一次使用时，从数据字典生成jump数据集
        self.jump_dataset = self._generate_jump_dataset()
        return self.jump_dataset
        
    def get_turn_dataset(self) -> Dict[str, Any]:
        """
        获取turn数据集
        
        Returns:
            turn数据集
        """
        # 如果内存中已有缓存，直接返回
        if self.turn_dataset is not None:
            return self.turn_dataset
            
        # 第一次使用时，从数据字典生成turn数据集
        self.turn_dataset = self._generate_turn_dataset()
        return self.turn_dataset
        
    def _generate_jump_dataset(self) -> Dict[str, Any]:
        """
        从数据字典生成jump模式的数据集
        
        Returns:
            生成的jump数据集
        """
        # 这里应该实现从HDF数据生成jump数据集的逻辑
        # 实际实现会根据数据结构而定
        
        # 示例实现，实际应根据数据结构调整
        if not hasattr(self, 'data_list') or not self.data_list:
            self.load_data()
            
        jump_dataset = {}
        for video_id, video_name in self.video_name_dict.items():
            video_data = self.data_list[video_name]
            # 根据需要生成jump模式数据
            jump_dataset[video_id] = {
                "video_name": video_name,
                "change_points": video_data.get("change_points", []),
                "n_frames": video_data.get("n_frames", 0),
                "picks": video_data.get("picks", []),
                "user_summary": video_data.get("user_summary", [])
            }
            
        return jump_dataset
        
    def _generate_turn_dataset(self) -> Dict[str, Any]:
        """
        从数据字典生成turn模式的数据集
        
        Returns:
            生成的turn数据集
        """
        # 这里应该实现从HDF数据生成turn数据集的逻辑
        # 实际实现会根据数据结构而定
        
        # 示例实现，实际应根据数据结构调整
        if not hasattr(self, 'data_list') or not self.data_list:
            self.load_data()
            
        turn_dataset = {}
        for video_id, video_name in self.video_name_dict.items():
            video_data = self.data_list[video_name]
            # 根据需要生成turn模式数据
            turn_dataset[video_id] = {
                "video_name": video_name,
                "change_points": video_data.get("change_points", []),
                "n_frames": video_data.get("n_frames", 0),
                "picks": video_data.get("picks", []),
                "user_summary": video_data.get("user_summary", [])
            }
            
        return turn_dataset
        
    def save_data(self, data: Dict[str, Any], output_path: Union[str, Path] = None) -> Dict[str, Any]:
        """
        "保存"数据集 - 现在只在内存中处理
        
        Args:
            data: 数据集
            output_path: 仅为兼容性保留，不再使用
            
        Returns:
            处理后的数据
        """
        # 只在内存中处理数据，不再写入文件
        print(f"数据处理完成，维持在内存中（跳过磁盘写入）")
        return data


class TVSumDataset(VideoSummarizationDataset):
    """TVSum数据集"""
    
    def __init__(self, dataset_dir: Union[str, Path], config=None):
        """
        初始化TVSum数据集
        
        Args:
            dataset_dir: 数据集目录
            config: 数据集配置对象
        """
        super().__init__(dataset_dir, "TVSum")
        
        # 直接从JSON加载配置
        if config is None:
            from ..config.config import load_config_from_json
            from ..config import TVSumConfig
            config_data = load_config_from_json()
            self.config = TVSumConfig(config_data.get('datasets', {}).get('tvsum', {}))
        else:
            self.config = config
            
        self.hdf_path = self.config.hdf_path
        self.eval_method = self.config.eval_method
        
        # 内存缓存
        self.jump_dataset = None
        self.turn_dataset = None
        
    def load_data(self) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        加载TVSum数据集
        
        Returns:
            Tuple包含:
            - 数据列表
            - 视频名称字典
        """
        # 加载HDF5数据
        data_dict = self.hdf5_to_dict(self.hdf_path)
        
        # 构建视频名称字典
        video_name_dict = {}
        for i, video_name in enumerate(data_dict.keys()):
            video_name_dict[f"video_{i+1}"] = video_name
            
        self.data_list = data_dict
        self.video_name_dict = video_name_dict
        
        return data_dict, video_name_dict
        
    def get_video_paths(self) -> List[Path]:
        """
        获取TVSum数据集中所有视频的路径
        
        Returns:
            视频路径列表
        """
        video_dir = Path(self.dataset_dir, "TVSum", "videos")
        if not video_dir.exists():
            raise FileNotFoundError(f"视频目录不存在: {video_dir}")
            
        return list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mov"))
        
    def get_jump_dataset(self) -> Dict[str, Any]:
        """
        获取jump数据集
        
        Returns:
            jump数据集
        """
        # 如果内存中已有缓存，直接返回
        if self.jump_dataset is not None:
            return self.jump_dataset
        
        # 第一次使用时，从数据字典生成jump数据集
        self.jump_dataset = self._generate_jump_dataset()
        return self.jump_dataset
        
    def get_turn_dataset(self) -> Dict[str, Any]:
        """
        获取turn数据集
        
        Returns:
            turn数据集
        """
        # 如果内存中已有缓存，直接返回
        if self.turn_dataset is not None:
            return self.turn_dataset
            
        # 第一次使用时，从数据字典生成turn数据集
        self.turn_dataset = self._generate_turn_dataset()
        return self.turn_dataset
        
    def _generate_jump_dataset(self) -> Dict[str, Any]:
        """
        从数据字典生成jump模式的数据集
        
        Returns:
            生成的jump数据集
        """
        # 这里实现从HDF数据生成jump数据集的逻辑
        
        if not hasattr(self, 'data_list') or not self.data_list:
            self.load_data()
            
        jump_dataset = {}
        for video_id, video_name in self.video_name_dict.items():
            video_data = self.data_list[video_name]
            # 根据需要生成jump模式数据
            jump_dataset[video_id] = {
                "video_name": video_name,
                "change_points": video_data.get("change_points", []),
                "n_frames": video_data.get("n_frames", 0),
                "picks": video_data.get("picks", []),
                "user_summary": video_data.get("user_summary", [])
            }
            
        return jump_dataset
        
    def _generate_turn_dataset(self) -> Dict[str, Any]:
        """
        从数据字典生成turn模式的数据集
        
        Returns:
            生成的turn数据集
        """
        # 这里实现从HDF数据生成turn数据集的逻辑
        
        if not hasattr(self, 'data_list') or not self.data_list:
            self.load_data()
            
        turn_dataset = {}
        for video_id, video_name in self.video_name_dict.items():
            video_data = self.data_list[video_name]
            # 根据需要生成turn模式数据
            turn_dataset[video_id] = {
                "video_name": video_name,
                "change_points": video_data.get("change_points", []),
                "n_frames": video_data.get("n_frames", 0),
                "picks": video_data.get("picks", []),
                "user_summary": video_data.get("user_summary", [])
            }
            
        return turn_dataset
