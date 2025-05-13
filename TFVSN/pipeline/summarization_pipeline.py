"""
视频摘要流水线：整合所有组件完成视频摘要任务
"""

from pathlib import Path
from typing import Dict, List, Any, Union, Optional
import numpy as np
import time
from tqdm import tqdm

# Use absolute imports
from TFVSN.core.video_processor import FrameExtractor, FeatureExtractor
from TFVSN.core.dataset_builder import SumMeDataset, TVSumDataset
from TFVSN.core.evaluation import SimilarityCalculator, F1ScoreEvaluator, ScoreCalculator
from TFVSN.models.llm_processor import LLMProcessor
from TFVSN.models.blip_processor import BlipProcessor
from ..utils.data_handler import JsonHandler
from ..config.settings import (
    FRAME_EXTRACTION_CONFIG,
    FEATURE_EXTRACTION_CONFIG,
    SUMMARIZATION_CONFIG,
    RESULTS_CONFIG,
    DATASET_DIR,
    SUMME_DATASET,
    TVSUM_DATASET
)


class VideoSummarizationPipeline:
    """视频摘要流水线"""
    
    def __init__(self, dataset_dir: Union[str, Path], pipeline_id: str = None):
        """
        初始化视频摘要流水线
        
        Args:
            dataset_dir: 数据集目录
            pipeline_id: 流水线标识符，用于区分不同的运行实例
        """
        self.dataset_dir = Path(dataset_dir)
        self.pipeline_id = pipeline_id or f"run_{int(time.time())}"
        
        # 初始化组件
        self.llm = LLMProcessor()
        self.blip = BlipProcessor()
        self.similarity_calculator = SimilarityCalculator()
        # 使用配置中的summary_ratio参数初始化ScoreCalculator
        self.score_calculator = ScoreCalculator(SUMMARIZATION_CONFIG["summary_ratio"])
        
        # 创建结果目录
        self.out_dir = Path("/root/TFVSN/oop/out")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # 为当前pipeline创建特定的输出目录
        self.pipeline_out_dir = self.out_dir / self.pipeline_id
        self.pipeline_out_dir.mkdir(parents=True, exist_ok=True)
        
        # 在pipeline特定目录下创建分类目录
        self.summaries_dir = self.pipeline_out_dir / "summaries"
        self.metrics_dir = self.pipeline_out_dir / "metrics"
        self.logs_dir = self.pipeline_out_dir / "logs"
        
        for dir_path in [self.summaries_dir, self.metrics_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # 原配置的结果目录也保留
        for dir_path in [
            RESULTS_CONFIG["llm_output_dir"],
            RESULTS_CONFIG["scores_dir"],
            RESULTS_CONFIG["similarity_scores_dir"],
            RESULTS_CONFIG["f1score_dir"]
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
    def process_video(self, 
                     video_path: Union[str, Path],
                     output_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        处理单个视频
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            
        Returns:
            处理结果
        """
        # 1. 提取帧
        frame_extractor = FrameExtractor(
            video_path,
            output_dir,
            fps=FRAME_EXTRACTION_CONFIG["fps"],
            frame_interval=FRAME_EXTRACTION_CONFIG["frame_interval"]
        )
        frames, frame_time, video_time = frame_extractor.process(
            force_sample=True
        )
        frame_paths = frame_extractor.save_frames(frames)
        
        # 2. 使用LLM分析帧重要性
        time_instruction = (
            f"The video lasts for {video_time:.2f} seconds, "
            f"and {len(frames)} frames are uniformly sampled from it. "
            f"These frames are located at {frame_time}."
        )
        
        prompt = (
            f"{time_instruction}\n"
            "For every frame I gave, first analyzed the content of each frame "
            "and then gave the importance score of the frame in the video "
            "summarization task. Make sure you have given scores for every frame "
            "on a scale from 0 to 1."
        )
        
        llm_output = self.llm.generate(prompt, frames)
        
        # 解析LLM输出提取重要性分数
        scores = self._parse_llm_output_for_scores(llm_output, len(frames))
        
        # 3. 提取视觉特征
        # 使用BLIP处理器提取图像特征
        visual_features = self.blip.encode(frames)
        
        # 4. 使用FeatureExtractor提取额外特征（如果需要）
        feature_extractor = FeatureExtractor(
            video_path,
            output_dir,
            stride=FEATURE_EXTRACTION_CONFIG["stride"],
            batch_size=FEATURE_EXTRACTION_CONFIG["batch_size"]
        )
        # 如果需要额外的特征处理，可以取消下面的注释
        # additional_features = feature_extractor.process(frames)
        
        return {
            "frames": frame_paths,
            "frame_time": frame_time,
            "video_time": video_time,
            "llm_output": llm_output,
            "scores": scores,
            "visual_features": visual_features
        }
        
    def process_dataset(self, dataset_name: str) -> None:
        """
        处理整个数据集
        
        Args:
            dataset_name: 数据集名称 ("SumMe" 或 "TVSum")
        """
        # 加载数据集
        if dataset_name == "SumMe":
            dataset = SumMeDataset(self.dataset_dir)
            eval_method = dataset.eval_method
        else:
            dataset = TVSumDataset(self.dataset_dir)
            eval_method = dataset.eval_method
            
        data_dict, video_name_dict = dataset.load_data()
        
        # 获取数据集的视频
        videos = dataset.get_video_paths()
        
        results = []
        for video in tqdm(videos, desc=f"Processing {dataset_name}"):
            video_name = video.stem
            output_dir = self.dataset_dir / dataset_name / "frames" / video_name
            
            # 处理视频
            result = self.process_video(video, output_dir)
            result["id"] = video_name
            results.append(result)
            
        # 保存结果到原始目录
        JsonHandler.save(
            results,
            RESULTS_CONFIG["llm_output_dir"] / f"{dataset_name.lower()}_result.json"
        )
        
        # 同时保存到pipeline特定目录
        JsonHandler.save(
            results,
            self.summaries_dir / f"{dataset_name.lower()}_result.json"
        )
        
    def evaluate(self, dataset_name: str) -> Dict[str, float]:
        """
        评估处理结果
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            评估结果
        """
        # 加载数据集
        if dataset_name == "SumMe":
            dataset = SumMeDataset(self.dataset_dir)
            eval_method = dataset.eval_method
        else:
            dataset = TVSumDataset(self.dataset_dir)
            eval_method = dataset.eval_method
            
        data_dict, _ = dataset.load_data()
        
        # 加载处理结果
        results = JsonHandler.load(
            RESULTS_CONFIG["llm_output_dir"] / f"{dataset_name.lower()}_result.json"
        )
        
        # 初始化评估器
        evaluator = F1ScoreEvaluator(eval_method)
        f_scores = []
        
        for result in results:
            video_name = result["id"]
            video_data = data_dict[video_name]
            
            # 生成摘要 - 使用score_calculator中已配置的summary_ratio
            summary = self.score_calculator.generate_summary(
                video_data["change_points"],
                np.array(result["scores"]),
                video_data["n_frames"],
                video_data["picks"]
                # 不再传递SUMMARIZATION_CONFIG["summary_ratio"]，使用ScoreCalculator中的默认值
            )
            
            # 计算F1分数
            f_score = evaluator.evaluate(
                summary,
                video_data["user_summary"]
            )
            f_scores.append(f_score)
            
        # 评估结果
        eval_results = {
            "individual_scores": f_scores,
            "average_score": np.mean(f_scores)
        }
        
        # 保存评估结果到pipeline特定目录
        JsonHandler.save(
            eval_results,
            self.metrics_dir / f"{dataset_name.lower()}_evaluation.json"
        )
        
        return eval_results
        
    def run(self) -> Dict[str, Any]:
        """
        运行完整的处理流程
        
        Returns:
            处理结果
        """
        results = {}
        start_time = time.time()
        
        # 处理两个数据集
        for dataset_name in ["SumMe", "TVSum"]:
            print(f"\nProcessing {dataset_name} dataset...")
            
            # 处理数据集
            self.process_dataset(dataset_name)
            
            # 评估结果
            eval_results = self.evaluate(dataset_name)
            results[dataset_name] = eval_results
            
            print(f"{dataset_name} Average F1-score: {eval_results['average_score']:.4f}")
        
        # 计算总运行时间
        run_time = time.time() - start_time
        
        # 添加运行元数据
        metadata = {
            "pipeline_id": self.pipeline_id,
            "run_time_seconds": run_time,
            "start_timestamp": start_time,
            "end_timestamp": time.time(),
            "results_summary": {
                "SumMe": results.get("SumMe", {}).get("average_score"),
                "TVSum": results.get("TVSum", {}).get("average_score")
            }
        }
        
        # 保存总体结果到日志目录
        JsonHandler.save(
            {**results, **metadata},
            self.logs_dir / "pipeline_run_summary.json"
        )
        
        print(f"\nPipeline run completed in {run_time:.2f} seconds")
        print(f"Results saved to: {self.pipeline_out_dir}")
            
        return results
    
    def _parse_llm_output_for_scores(self, llm_output: str, num_frames: int) -> List[float]:
        """
        从LLM输出中解析出每一帧的重要性分数
        
        Args:
            llm_output: LLM的输出文本
            num_frames: 帧的数量
            
        Returns:
            每一帧的重要性分数列表
        """
        import re
        
        # 初始化默认分数（如果无法解析，则使用中等重要性）
        default_score = 0.5
        scores = [default_score] * num_frames
        
        # 尝试提取分数（寻找类似 "Frame X: score 0.7" 或 "Importance: 0.7" 的模式）
        frame_patterns = [
            r'[Ff]rame\s*(\d+)\s*:.*?(\d+\.\d+)',  # Frame 1: ... 0.7
            r'[Ff]rame\s*(\d+).*?[Ii]mportance\s*:?\s*(\d+\.\d+)',  # Frame 1 ... Importance: 0.7
            r'[Ff]rame\s*(\d+).*?[Ss]core\s*:?\s*(\d+\.\d+)',  # Frame 1 ... Score: 0.7
            r'(\d+)\s*\)\s*.*?[Ii]mportance\s*:?\s*(\d+\.\d+)',  # 1) ... Importance: 0.7
            r'(\d+)\.\s*.*?[Ii]mportance\s*:?\s*(\d+\.\d+)',  # 1. ... Importance: 0.7
        ]
        
        for pattern in frame_patterns:
            matches = re.finditer(pattern, llm_output)
            for match in matches:
                try:
                    frame_idx = int(match.group(1)) - 1  # 转为0-based索引
                    if 0 <= frame_idx < num_frames:
                        score = float(match.group(2))
                        # 确保分数在0-1范围内
                        score = max(0.0, min(1.0, score))
                        scores[frame_idx] = score
                except (ValueError, IndexError):
                    continue
                    
        return scores
