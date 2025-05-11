# 视频摘要算法 - OOP重构计划

## 项目背景

这是一个视频摘要生成系统，当前的实现是一系列线性脚本处理流程。项目流程包括：视频帧提取、视频数据集创建、使用LLM分析视频帧、提取视觉特征、获取视频描述、计算语义相似度和最终评估等步骤。

## 重构目标

将当前的线性处理流程改造为模块化、可扩展的面向对象设计，提高代码的可读性、可维护性和可扩展性。

## 整体架构设计

### 核心组件

1. **VideoProcessor**: 视频处理的基类
   - FrameExtractor: 视频帧提取器
   - FeatureExtractor: 视频特征提取器
   - TextFeatureExtractor: 文本特征提取器

2. **DatasetBuilder**: 数据集构建器
   - SumMeDataset: SumMe数据集
   - TVSumDataset: TVSum数据集

3. **ModelInterface**: 模型接口
   - LLMProcessor: LLM处理器类
   - BlipModelProcessor: Blip模型处理器类

4. **EvaluationSystem**: 评估系统
   - ScoreCalculator: 分数计算器
   - SimilarityCalculator: 相似度计算器
   - F1ScoreEvaluator: F1评分器

5. **Pipeline**: 整合所有组件的流水线系统
   - VideoSummarizationPipeline: 视频摘要流水线

6. **Utils**: 工具类
   - JsonHandler: JSON处理工具
   - HDF5Handler: HDF5处理工具
   - KnapsackSolver: 背包算法解算器

## 数据流设计

1. 视频输入 → FrameExtractor → 视频帧
2. 视频帧 → DatasetBuilder → 数据集
3. 数据集 → LLMProcessor → 帧重要性分数
4. 数据集 → FeatureExtractor → 视觉特征
5. 视觉特征 + 文本特征 → SimilarityCalculator → 语义相似度
6. 帧重要性分数 + 语义相似度 → ScoreCalculator → 最终分数
7. 最终分数 → EvaluationSystem → 评估结果

## 实现计划

### 第一阶段: 基础设施和接口定义

1. 创建基础的类结构和接口
2. 实现工具类和数据结构
3. 定义核心组件的抽象接口

### 第二阶段: 核心组件实现

1. 实现VideoProcessor及其子类
2. 实现ModelInterface及其子类
3. 实现DatasetBuilder及其子类

### 第三阶段: 评估系统和流水线

1. 实现EvaluationSystem
2. 构建Pipeline系统
3. 连接所有组件

### 第四阶段: 测试和优化

1. 单元测试各组件
2. 集成测试整个流水线
3. 性能优化和代码清理

## 文件结构设计

```
oop/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── video_processor.py
│   ├── dataset_builder.py
│   ├── model_interface.py
│   └── evaluation.py
├── models/
│   ├── __init__.py
│   ├── llm_processor.py
│   └── blip_processor.py
├── utils/
│   ├── __init__.py
│   ├── data_handler.py
│   └── algorithms.py
├── pipeline/
│   ├── __init__.py
│   └── summarization_pipeline.py
└── config/
    ├── __init__.py
    └── settings.py
```

## 下一步工作

1. 创建基础的文件夹结构
2. 定义核心接口和基类
3. 开始实现VideoProcessor及其子类
4. 逐步迁移现有功能到新的OOP架构中
