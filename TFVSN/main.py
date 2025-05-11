#!/usr/bin/env python3
"""
视频摘要系统主入口
运行完整的视频摘要流水线处理
"""

import argparse
import time
from pathlib import Path

# Use absolute import now that the package is properly installed
from TFVSN.pipeline.summarization_pipeline import VideoSummarizationPipeline


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="视频摘要系统")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/root/TFVSN/dataset",
        help="数据集目录路径"
    )
    parser.add_argument(
        "--pipeline-id",
        type=str,
        default=None,
        help="流水线ID，用于标识不同的运行实例。如不指定将使用时间戳"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["SumMe", "TVSum", "all"],
        default="all",
        help="要处理的数据集名称"
    )
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 如果未指定pipeline_id，使用时间戳
    pipeline_id = args.pipeline_id or f"run_{int(time.time())}"
    
    print(f"启动视频摘要流水线 (ID: {pipeline_id})")
    print(f"数据集目录: {args.dataset_dir}")
    
    # 初始化流水线
    pipeline = VideoSummarizationPipeline(
        dataset_dir=args.dataset_dir,
        pipeline_id=pipeline_id
    )
    
    # 运行流水线
    if args.dataset == "all":
        # 运行完整流水线（包括两个数据集）
        results = pipeline.run()
    else:
        # 仅处理指定数据集
        print(f"\n处理 {args.dataset} 数据集...")
        pipeline.process_dataset(args.dataset)
        results = {args.dataset: pipeline.evaluate(args.dataset)}
        print(f"{args.dataset} Average F1-score: {results[args.dataset]['average_score']:.4f}")
    
    print(f"\n处理完成。结果已保存到: {pipeline.pipeline_out_dir}")
    
    return results


if __name__ == "__main__":
    main()
