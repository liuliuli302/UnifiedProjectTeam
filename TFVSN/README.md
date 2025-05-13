# TFVSN：一种免训练的视频摘要框架

TFVSN是一个基于视觉-语言模型的视频摘要生成系统，能够无需训练即可生成高质量的视频摘要。

## 项目结构

```
TFVSN/
├── __init__.py          # 包初始化文件
├── main.py              # 主入口脚本
├── test.py              # 测试脚本
├── config/              # 配置文件目录
│   ├── __init__.py
│   └── settings.py      # 全局配置参数
├── core/                # 核心组件
│   ├── __init__.py
│   ├── dataset_builder.py   # 数据集处理
│   ├── evaluation.py        # 评估系统
│   ├── model_interface.py   # 模型接口
│   └── video_processor.py   # 视频处理
├── models/              # 模型实现
│   ├── __init__.py
│   ├── blip_processor.py    # BLIP模型处理器
│   └── llm_processor.py     # LLM模型处理器
├── pipeline/            # 处理流水线
│   ├── __init__.py
│   └── summarization_pipeline.py  # 摘要生成流水线
├── result/              # 结果输出目录
└── utils/               # 工具类
    ├── __init__.py
    ├── algorithms.py    # 算法实现
    └── data_handler.py  # 数据处理工具
```

## 配置参数说明

所有配置参数都在`config/settings.py`文件中集中管理：

### 1. 基础路径配置
- `ROOT_DIR`: 项目根目录
- `DATASET_DIR`: 数据集目录
- `OUTPUT_DIR`: 输出结果目录

### 2. 数据集配置
- `SUMME_DATASET`: SumMe数据集相关配置
  - `hdf_path`: HDF5文件路径
  - `dataset_jump`/`dataset_turn`: 数据集JSON文件路径
  - `eval_method`: 评估方法("max")
  
- `TVSUM_DATASET`: TVSum数据集相关配置
  - `hdf_path`: HDF5文件路径
  - `dataset_jump`/`dataset_turn`: 数据集JSON文件路径
  - `eval_method`: 评估方法("avg")

### 3. 结果输出配置
- `RESULTS_CONFIG`: 包含输出目录路径配置
  - `llm_output_dir`: LLM输出结果保存路径
  - `scores_dir`: 分数结果保存路径
  - `similarity_scores_dir`: 相似度分数保存路径
  - `f1score_dir`: F1分数结果保存路径

### 4. 模型配置
- `LLM_CONFIG`: LLM模型配置
  - `model_name`: 模型名称
  - `conv_template`: 对话模板
  - `device`: 运行设备
  - `model_type`: 模型类型

- `BLIP_CONFIG`: BLIP模型配置
  - `model_name`: 模型名称
  - `device`: 运行设备
  - `dtype`: 数据类型

### 5. 提取配置
- `FRAME_EXTRACTION_CONFIG`: 帧提取配置
  - `fps`: 每秒提取的帧数
  - `frame_interval`: 帧间隔

- `FEATURE_EXTRACTION_CONFIG`: 特征提取配置
  - `stride`: 特征提取步长
  - `batch_size`: 批处理大小

### 6. 摘要生成配置
- `SUMMARIZATION_CONFIG`:
  - `summary_ratio`: 摘要长度占原视频长度的比例

## 运行方法

1. 安装依赖：
```bash
pip install -e .
```

2. 运行系统：
```bash
python -m TFVSN.main --dataset-dir /path/to/dataset --dataset all
```

3. 测试配置：
```bash
python TFVSN/test.py
```






