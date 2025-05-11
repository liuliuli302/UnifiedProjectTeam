"""
视频摘要系统配置文件
"""

import os
from pathlib import Path

# 基础路径配置
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATASET_DIR = ROOT_DIR / "dataset"
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
    "fps": 1,
    "frame_interval": 15
}

FEATURE_EXTRACTION_CONFIG = {
    "stride": 1,
    "batch_size": 16
}

# 摘要生成配置
SUMMARIZATION_CONFIG = {
    "summary_ratio": 0.15  # 摘要长度占原视频长度的比例
}

# 创建必要的目录
for directory in [OUTPUT_DIR, RESULTS_CONFIG["llm_output_dir"], RESULTS_CONFIG["scores_dir"], 
                  RESULTS_CONFIG["similarity_scores_dir"], RESULTS_CONFIG["f1score_dir"]]:
    directory.mkdir(parents=True, exist_ok=True)
