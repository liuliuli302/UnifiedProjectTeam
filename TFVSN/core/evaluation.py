"""
评估系统：处理视频摘要的评估相关功能
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple
from pathlib import Path


class Evaluator(ABC):
    """评估器基类"""
    
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> float:
        """
        执行评估
        
        Returns:
            评估分数
        """
        pass


class SimilarityCalculator:
    """相似度计算器"""
    
    @staticmethod
    def split_segments(data: np.ndarray, num_segments: int, axis: int = 0) -> List[np.ndarray]:
        """
        将数据分割成段
        
        Args:
            data: 输入数据
            num_segments: 段数
            axis: 分割轴
            
        Returns:
            分割后的数据列表
        """
        length = data.shape[axis]
        seg_sizes = [(length + i) // num_segments for i in range(num_segments)]
        indices = np.cumsum(seg_sizes)[:-1]
        return np.array_split(data, indices, axis=axis)

    def compute_similarity_scores(self, 
                                visual_feat: np.ndarray, 
                                text_feat: np.ndarray, 
                                segment_num: int) -> Dict[str, List[float]]:
        """
        计算视觉特征和文本特征之间的相似度分数
        
        Args:
            visual_feat: 视觉特征
            text_feat: 文本特征
            segment_num: 段数
            
        Returns:
            包含不同相似度计算方法结果的字典
        """
        visual_segs = self.split_segments(visual_feat, segment_num, axis=0)
        text_segs = self.split_segments(text_feat, segment_num, axis=0)
        
        max_p_scores = []
        mean_p_scores = []
        max_m_scores = []
        mean_m_scores = []
        
        for vis_seg, txt_seg in zip(visual_segs, text_segs):
            # 计算相似度矩阵
            scores = np.einsum('md,npd->nmp', txt_seg, vis_seg)
            
            # 计算不同的池化策略
            max_p = np.max(scores, axis=2)
            mean_p = np.mean(scores, axis=2)
            
            max_p_max_m = np.expand_dims(np.max(max_p, axis=1), 1)
            max_p_mean_m = np.expand_dims(np.mean(max_p, axis=1), 1)
            mean_p_max_m = np.expand_dims(np.max(mean_p, axis=1), 1)
            mean_p_mean_m = np.expand_dims(np.mean(mean_p, axis=1), 1)
            
            max_p_scores.append(max_p_max_m)
            mean_p_scores.append(mean_p_max_m)
            max_m_scores.append(mean_p_max_m)
            mean_m_scores.append(mean_p_mean_m)
            
        # 合并所有段的分数
        score_dict = {
            'max_p_max_m': np.concatenate(max_p_scores, axis=0),
            'mean_p_max_m': np.concatenate(mean_p_scores, axis=0),
            'max_p_mean_m': np.concatenate(max_m_scores, axis=0),
            'mean_p_mean_m': np.concatenate(mean_m_scores, axis=0)
        }
        
        # 归一化分数
        for k in score_dict:
            seq = score_dict[k]
            min_v = np.min(seq)
            max_v = np.max(seq)
            score_dict[k] = ((seq - min_v) / (max_v - min_v + 1e-8)).squeeze(1).tolist()
            
        return score_dict


class F1ScoreEvaluator(Evaluator):
    """F1分数评估器"""
    
    def __init__(self, eval_method: str = "avg"):
        """
        初始化F1分数评估器
        
        Args:
            eval_method: 评估方法，"avg"或"max"
        """
        self.eval_method = eval_method

    def evaluate_single_summary(self, 
                              predicted_summary: np.ndarray, 
                              user_summary: np.ndarray) -> float:
        """
        评估单个摘要
        
        Args:
            predicted_summary: 预测的摘要
            user_summary: 用户标注的摘要
            
        Returns:
            F1分数
        """
        max_len = max(len(predicted_summary), user_summary.shape[1])
        S = np.zeros(max_len, dtype=int)
        G = np.zeros(max_len, dtype=int)
        S[:len(predicted_summary)] = predicted_summary
        
        f_scores = []
        for user in range(user_summary.shape[0]):
            G[:user_summary.shape[1]] = user_summary[user]
            overlapped = S & G
            
            # 计算precision, recall, f-score
            precision = sum(overlapped) / (sum(S) + 1e-8)
            recall = sum(overlapped) / (sum(G) + 1e-8)
            
            if precision + recall == 0:
                f_scores.append(0)
            else:
                f_scores.append(2 * precision * recall * 100 / (precision + recall))
                
        if self.eval_method == "max":
            return max(f_scores)
        else:
            return sum(f_scores) / len(f_scores)

    def evaluate(self, 
                predicted_summary: np.ndarray, 
                user_summary: np.ndarray, 
                *args, **kwargs) -> float:
        """
        评估摘要
        
        Args:
            predicted_summary: 预测的摘要
            user_summary: 用户标注的摘要
            
        Returns:
            F1分数
        """
        return self.evaluate_single_summary(predicted_summary, user_summary)


class ScoreCalculator:
    """分数计算器"""
    
    def __init__(self, summary_ratio: float = None):
        """
        初始化分数计算器
        
        Args:
            summary_ratio: 摘要长度占原视频长度的比例，如果为None则使用配置中的值
        """
        from ..config.config import SUMMARIZATION_CONFIG
        self.summary_ratio = summary_ratio or SUMMARIZATION_CONFIG["summary_ratio"]
        
    def generate_summary(self, shot_bound: np.ndarray, 
                        scores: np.ndarray, 
                        n_frames: int, 
                        positions: np.ndarray,
                        summary_ratio: float = None) -> np.ndarray:
        """
        生成视频摘要
        
        Args:
            shot_bound: 镜头边界
            scores: 重要性分数
            n_frames: 总帧数
            positions: 关键帧位置
            summary_ratio: 摘要长度比例
            
        Returns:
            摘要标记数组
        """
        from ..utils.algorithms import KnapsackSolver
        
        # 确保positions是整数类型
        positions = positions.astype(np.int32)
        if positions[-1] != n_frames:
            positions = np.append(positions, n_frames)
            
        # 计算每帧的分数
        frame_scores = np.zeros(n_frames, dtype=np.float32)
        for i in range(len(positions) - 1):
            pos_left, pos_right = positions[i], positions[i + 1]
            if i >= len(scores):
                frame_scores[pos_left:pos_right] = 0
            else:
                frame_scores[pos_left:pos_right] = scores[i]
                
        # 计算每个镜头的重要性分数
        shot_imp_scores = []
        shot_lengths = []
        for shot in shot_bound:
            shot_lengths.append(shot[1] - shot[0] + 1)
            shot_imp_scores.append(
                float(frame_scores[shot[0]: shot[1] + 1].mean())
            )
            
        # 使用背包算法选择最佳镜头
        final_shot = shot_bound[-1]
        # 使用传入的summary_ratio参数或类属性
        actual_summary_ratio = summary_ratio if summary_ratio is not None else self.summary_ratio
        final_max_length = int((final_shot[1] + 1) * actual_summary_ratio)
        
        knapsack = KnapsackSolver()
        selected = knapsack.solve(final_max_length, shot_lengths, shot_imp_scores)
        
        # 生成摘要
        summary = np.zeros(final_shot[1] + 1, dtype=np.int8)
        for shot_idx in selected:
            summary[shot_bound[shot_idx][0]: shot_bound[shot_idx][1] + 1] = 1
            
        return summary
