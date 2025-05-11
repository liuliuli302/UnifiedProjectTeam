"""
算法工具：提供通用的算法实现
"""

from typing import List, Tuple
import numpy as np


class KnapsackSolver:
    """0-1背包问题求解器"""
    
    def solve(self, W: int, wt: List[int], val: List[float]) -> List[int]:
        """
        解决0-1背包问题
        
        Args:
            W: 背包容量
            wt: 物品重量列表
            val: 物品价值列表
            
        Returns:
            选中的物品索引列表
        """
        n = len(val)
        K = [[0 for _ in range(W + 1)] for _ in range(n + 1)]
        
        # 构建动态规划表
        for i in range(n + 1):
            for w in range(W + 1):
                if i == 0 or w == 0:
                    K[i][w] = 0
                elif wt[i - 1] <= w:
                    K[i][w] = max(
                        val[i - 1] + K[i - 1][w - wt[i - 1]],
                        K[i - 1][w]
                    )
                else:
                    K[i][w] = K[i - 1][w]
                    
        # 回溯找出选中的物品
        selected = []
        w = W
        for i in range(n, 0, -1):
            if K[i][w] != K[i - 1][w]:
                selected.insert(0, i - 1)
                w -= wt[i - 1]
                
        return selected


class ShotBoundaryDetector:
    """镜头边界检测器"""
    
    def __init__(self, threshold: float = 0.5):
        """
        初始化检测器
        
        Args:
            threshold: 差异阈值
        """
        self.threshold = threshold
        
    def detect(self, features: np.ndarray) -> List[Tuple[int, int]]:
        """
        检测镜头边界
        
        Args:
            features: 视频特征序列
            
        Returns:
            镜头边界列表，每个元素为(起始帧，结束帧)
        """
        # 计算相邻帧之间的差异
        diffs = np.linalg.norm(
            features[1:] - features[:-1],
            axis=1
        )
        
        # 找到差异大于阈值的位置
        boundaries = np.where(diffs > self.threshold)[0]
        
        # 生成镜头边界
        shots = []
        start = 0
        for boundary in boundaries:
            shots.append((start, boundary))
            start = boundary + 1
        shots.append((start, len(features) - 1))
        
        return shots
