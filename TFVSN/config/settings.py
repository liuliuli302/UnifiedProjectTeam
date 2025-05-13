"""
视频摘要系统配置文件
"""

import os
from pathlib import Path

# 基础路径配置
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATASET_DIR = Path("/root/autodl-tmp/data")
OUTPUT_DIR = ROOT_DIR / "dataset" / "result"

# 数据集配置
SUMME_DATASET = {
    "hdf_path": DATASET_DIR / "SumMe" / "summe.h5",
    "dataset_jump": DATASET_DIR / "SumMe" / "summe_dataset_jump.json",
    "dataset_turn": DATASET_DIR / "SumMe" / "summe_dataset_turn.json",
    "eval_method": "max"
}

TVSUM_DATASET = {
    "hdf_path": DATASET_DIR / "TVSum" / "tvsum.h5",
    "dataset_jump": DATASET_DIR / "TVSum" / "tvsum_dataset_jump.json",
    "dataset_turn": DATASET_DIR / "TVSum" / "tvsum_dataset_turn.json",
    "eval_method": "avg"
}

# 结果输出配置
RESULTS_CONFIG = {
    "llm_output_dir": OUTPUT_DIR / "raw",
    "scores_dir": OUTPUT_DIR / "scores",
    "similarity_scores_dir": OUTPUT_DIR / "similarity_scores",
    "f1score_dir": OUTPUT_DIR / "f1score"
}

# 模型配置
LLM_CONFIG = {
    "model_name": "lmms-lab/LLaVA-Video-7B-Qwen2",
    "conv_template": "qwen_1_5",
    "device": "cuda",
    "model_type": "llava_qwen"
}

BLIP_CONFIG = {
    "model_name": "Salesforce/blip2-opt-2.7b",
    "device": "cuda",
    "dtype": "float32"
}

# 提取配置
FRAME_EXTRACTION_CONFIG = {
    "fps": 1,  # 每秒提取的帧数
    "frame_interval": 15  # 帧间隔（每隔多少帧提取一帧，优先级高于fps）
}

FEATURE_EXTRACTION_CONFIG = {
    "stride": 1,  # 特征提取的步长（每隔多少帧提取一次特征）
    "batch_size": 16  # 特征提取时的批处理大小，影响内存使用和速度
}

# 摘要生成配置
SUMMARIZATION_CONFIG = {
    "summary_ratio": 0.15  # 摘要长度占原视频长度的比例
}

# 创建必要的目录
for directory in [OUTPUT_DIR, RESULTS_CONFIG["llm_output_dir"], RESULTS_CONFIG["scores_dir"], 
                  RESULTS_CONFIG["similarity_scores_dir"], RESULTS_CONFIG["f1score_dir"]]:
    directory.mkdir(parents=True, exist_ok=True)
