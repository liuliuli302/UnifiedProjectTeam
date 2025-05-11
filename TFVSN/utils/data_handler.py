"""
工具类：提供通用的数据处理功能
"""

from typing import Dict, List, Any, Union
from pathlib import Path
import json
import h5py
import numpy as np


class JsonHandler:
    """JSON文件处理工具"""
    
    @staticmethod
    def load(file_path: Union[str, Path]) -> Dict[str, Any]:
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
    def save(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 4) -> None:
        """
        保存为JSON文件
        
        Args:
            data: 要保存的数据
            file_path: 保存路径
            indent: 缩进
        """
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)


class HDF5Handler:
    """HDF5文件处理工具"""
    
    @staticmethod
    def load(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        加载HDF5文件
        
        Args:
            file_path: HDF5文件路径
            
        Returns:
            HDF5内容
        """
        def recursively_convert_to_dict(h5_obj):
            if isinstance(h5_obj, h5py.Dataset):
                return h5_obj[()]
            elif isinstance(h5_obj, h5py.Group):
                return {key: recursively_convert_to_dict(h5_obj[key]) 
                       for key in h5_obj.keys()}
            else:
                raise TypeError(f"不支持的类型: {type(h5_obj)}")

        with h5py.File(file_path, "r") as f:
            return recursively_convert_to_dict(f)
            
    @staticmethod
    def save(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """
        保存为HDF5文件
        
        Args:
            data: 要保存的数据
            file_path: 保存路径
        """
        def recursively_save_dict_to_hdf5(h5file, path, dic):
            for key, item in dic.items():
                if isinstance(item, (np.ndarray, np.int64, np.float64, str, int, float)):
                    h5file[path + key] = item
                elif isinstance(item, dict):
                    recursively_save_dict_to_hdf5(
                        h5file, path + key + '/', item)
                else:
                    raise ValueError(f'无法保存类型 {type(item)}')

        with h5py.File(file_path, 'w') as f:
            recursively_save_dict_to_hdf5(f, '/', data)
