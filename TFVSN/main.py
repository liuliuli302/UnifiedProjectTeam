#!/usr/bin/env python3
"""
视频摘要系统主入口
运行完整的视频摘要流水线处理
"""

# Use absolute import now that the package is properly installed
from TFVSN.pipeline.summarization_pipeline import VideoSummarizationPipeline
# Import necessary configurations from settings.py
from TFVSN.config.config import PIPELINE_ID, DATASET_DIR, DATASET_TO_PROCESS, OUTPUT_DIR


def main():
    """主函数"""
    # 使用从settings.py导入的配置
    print(f"启动视频摘要流水线 (ID: {PIPELINE_ID})")
    print(f"数据集目录: {DATASET_DIR}")
    print(f"输出目录: {OUTPUT_DIR}") # Added for clarity
    
    # 初始化流水线
    # Pass all necessary config objects to the pipeline
    pipeline = VideoSummarizationPipeline(
        # dataset_dir=DATASET_DIR, # No longer needed if pipeline directly uses settings
        # pipeline_id=PIPELINE_ID  # No longer needed if pipeline directly uses settings
    )
    
    # 运行流水线
    if DATASET_TO_PROCESS == "all":
        # 运行完整流水线（包括两个数据集）
        results = pipeline.run()
    else:
        # 仅处理指定数据集
        print(f"\n处理 {DATASET_TO_PROCESS} 数据集...")
        pipeline.process_dataset(DATASET_TO_PROCESS)
        results = {DATASET_TO_PROCESS: pipeline.evaluate(DATASET_TO_PROCESS)}
        print(f"{DATASET_TO_PROCESS} Average F1-score: {results[DATASET_TO_PROCESS]['average_score']:.4f}")
    
    print(f"\n处理完成。结果已保存到: {pipeline.pipeline_out_dir}")
    
    return results


if __name__ == "__main__":
    main()
