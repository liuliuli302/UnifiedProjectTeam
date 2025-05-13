#!/usr/bin/env python3
"""
测试配置文件加载和使用
"""

from TFVSN.config.settings import (
    FRAME_EXTRACTION_CONFIG,
    FEATURE_EXTRACTION_CONFIG,
    SUMMARIZATION_CONFIG,
    RESULTS_CONFIG,
    SUMME_DATASET,
    TVSUM_DATASET,
    LLM_CONFIG,
    BLIP_CONFIG
)

def test_config_loading():
    """测试配置加载"""
    print("=== 测试配置加载 ===")
    
    # 打印帧提取配置
    print(f"帧提取配置:")
    print(f"  FPS: {FRAME_EXTRACTION_CONFIG['fps']}")
    print(f"  帧间隔: {FRAME_EXTRACTION_CONFIG['frame_interval']}")
    
    # 打印特征提取配置
    print(f"特征提取配置:")
    print(f"  步长: {FEATURE_EXTRACTION_CONFIG['stride']}")
    print(f"  批处理大小: {FEATURE_EXTRACTION_CONFIG['batch_size']}")
    
    # 打印摘要生成配置
    print(f"摘要生成配置:")
    print(f"  摘要比例: {SUMMARIZATION_CONFIG['summary_ratio']}")
    
    # 打印数据集配置
    print(f"SumMe数据集配置:")
    print(f"  HDF路径: {SUMME_DATASET['hdf_path']}")
    print(f"  评估方法: {SUMME_DATASET['eval_method']}")
    
    print(f"TVSum数据集配置:")
    print(f"  HDF路径: {TVSUM_DATASET['hdf_path']}")
    print(f"  评估方法: {TVSUM_DATASET['eval_method']}")
    
    # 打印模型配置
    print(f"LLM配置:")
    print(f"  模型名称: {LLM_CONFIG['model_name']}")
    print(f"  设备: {LLM_CONFIG['device']}")
    
    print(f"BLIP配置:")
    print(f"  模型名称: {BLIP_CONFIG['model_name']}")
    print(f"  设备: {BLIP_CONFIG['device']}")
    
    # 打印结果输出配置
    print(f"结果输出配置:")
    for key, value in RESULTS_CONFIG.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_config_loading()