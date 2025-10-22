# YOLOv10 烟火检测模型在寒武纪设备上的推理实现

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Platform](https://img.shields.io/badge/platform-Cambricon-red)

基于寒武纪（Cambricon）AI加速设备的 YOLOv10 烟火检测模型推理实现。该项目实现了在寒武纪设备上运行 YOLOv10 目标检测模型进行烟火检测，并对推理过程的各个阶段进行性能分析。

## 目录

- [项目介绍](#项目介绍)
- [功能特性](#功能特性)
- [环境依赖](#环境依赖)
- [模型配置](#模型配置)
- [使用方法](#使用方法)
- [性能分析](#性能分析)
- [目录结构](#目录结构)

## 项目介绍

本项目是一个基于 YOLOv10 的烟火检测系统，专为寒武纪 AI 加速设备优化。它能够实时检测视频流中的火焰和烟雾，适用于安防监控、火灾预警等场景。项目通过将模型推理过程分解为预处理、推理和后处理三个阶段，分别测量各阶段耗时，为性能调优提供数据支持。

## 功能特性

- 🔥 烟火目标检测：支持火焰和烟雾两种目标的检测
- 🚀 寒武纪设备优化：专为寒武纪 AI 加速器优化的推理实现
- 📊 性能分析：提供预处理、推理、后处理各阶段耗时分析
- 🎯 高精度检测：基于 YOLOv10 的先进检测算法
- 🖼️ 多格式支持：支持图片和视频输入
- 🎨 可视化结果：检测结果可视化并保存

## 环境依赖

- Python 3.10
- PyTorch 1.13.0 with MLU support
- OpenCV
- NumPy
- Ultralytics YOLO

安装依赖：
```bash
pip install https://sdk.cambricon.com/static/PyTorch/MLU370_1.13_v1.17.0_X86_centos8.3_python3.10_pip/torch-1.13.1-cp310-cp310-linux_x86_64.whl
pip install https://sdk.cambricon.com/static/PyTorch/MLU370_1.13_v1.17.0_X86_centos8.3_python3.10_pip/torch_mlu-1.17.0+torch1.13-cp310-cp310-linux_x86_64.whl
pip install https://sdk.cambricon.com/static/PyTorch/MLU370_1.13_v1.17.0_X86_centos8.3_python3.10_pip/torchvision-0.14.1a0+5e8e2f1-cp310-cp310-linux_x86_64.whl
pip install ultralytics
pip install opencv-python
```

检查torch_mlu是否安装成功：
```bash
# python
>>> import torch
>>> import torch_mlu
>>> import torchvision
>>> torch.__version__
'1.9.0'
>>> torch_mlu.__version__
'1.17.0-torch1.9'
>>> torchvision.__version__
'0.10.0a0+300a8a4'
```

## 模型配置

项目提供了多种 YOLOv10 模型配置：

- `yolov10n.yaml` - Nano 版本，轻量级模型
- `yolov10s.yaml` - Small 版本，平衡性能和速度
- `yolov10m.yaml` - Medium 版本，默认使用模型
- `yolov10l.yaml` - Large 版本，注重检测性能
- `yolov10x.yaml` - XL 版本，最高检测精度

所有模型都针对 2 类目标（火焰和烟雾）进行了配置。

## 使用方法

1. 准备模型权重文件（需自行获取或训练）
2. 修改 [test.py](test.py) 中的 `weight_path` 变量指向模型权重文件路径
3. 准备测试视频文件，放置在 `data/test.mp4`
4. 运行推理脚本：

```bash
python test.py
```

推理结果将保存在 `detect_res/` 目录中。

## 性能分析

项目对推理过程进行了详细的性能分析，将整个推理过程分为三个阶段：

1. **预处理阶段**：图像缩放、格式转换、归一化等操作
2. **推理阶段**：模型前向传播计算
3. **后处理阶段**：检测框处理、非极大值抑制等操作

各阶段耗时会在控制台输出，便于分析性能瓶颈和优化方向。

## 目录结构

```
cambricon_infer/
├── cfg/                  # 模型配置文件
│   ├── yolov10n.yaml
│   ├── yolov10s.yaml
│   ├── yolov10m.yaml
│   ├── yolov10l.yaml
│   └── yolov10x.yaml
├── data/                 # 测试数据
│   └── test.mp4          # 测试视频文件
├── detect_res/           # 检测结果保存目录
├── test.py               # 主程序文件
└── README.md             # 项目说明文件
```