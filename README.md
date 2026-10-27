# NAVIS

NAVIS 是一个面向低带宽 NB-IoT 链路的语义视频压缩、传输与重建研究项目，通过场景分解、分层编码和 ACK 反馈驱动的调度，在受限带宽下传输背景及感兴趣区域（ROI）。

## 主要功能

- 场景分解：目标检测、跟踪、背景建模与 ROI 提取。
- 视频压缩：DCVC 分层编码，以及 ELIC 背景 + JPEG ROI 基线。
- NB-IoT 传输：SIM7080 串口通信、分片、ACK 反馈、自适应调度与多节点发送。
- 视频重建与评估：接收载荷解码、ROI 合成、质量分析及检测、跟踪实验。

## 环境与安装

使用 Python 3.12 和 uv，在项目根目录执行：

```bash
uv sync --group dev
```

按需安装功能依赖，例如场景分解与任务评估：

```bash
uv sync --group dev --extra scene-factorization --extra evaluation
```

其他可选依赖包括 `vision`、`dcvc`、`elic`、`acceleration` 和 `reports`。DCVC 包含本地 C++ 扩展，需要可用的 C++ 编译工具链；GPU 运行还需匹配的 PyTorch/CUDA 环境。

数据集、预训练模型、安装程序和运行产物不随仓库上传，请另行准备：

- MOT17 数据集放置于 `MOT17/`。
- YOLO 模型放置于 `src/common/models/`；默认使用 `yolov8s-seg.pt`，未找到本地文件时由 Ultralytics 按模型名加载。
- DCVC 权重放置于 `src/dcvc_compression/models/dcvc_rt/`，路径见对应 `config.json`。
- ELIC 权重放置于 `src/elic_jpg_compression/models/elic/`，路径见对应 `config.json`。

## 使用

完成依赖和数据准备后，可先查看场景分解参数：

```bash
uv run scene-factorization --help
uv run scene-factorization --mot17-root MOT17 --split train --sequences MOT17-04-FRCNN
```

各阶段的入口如下，请在运行前检查相应目录中的配置文件：

| 阶段 | 入口 |
| --- | --- |
| DCVC 编码 | `uv run python -m dcvc_compression.main` |
| ELIC/JPEG 编码 | `uv run python -m elic_jpg_compression.main` |
| NB-IoT 发送 | `uv run python -m nbiot_transport.main` |
| 离线重建 | `uv run python -m rebuild_video.main` |

发送前需按实际环境配置串口、APN、接收端地址和候选载荷路径。接收端代码及配置位于 `deploy/ecs_receiver/`。编码、发送和重建依赖前序阶段生成的产物，不能只凭源码直接完成端到端运行。

## 目录

```text
src/          场景分解、编码、传输、重建及公共模块
deploy/       接收端程序与服务配置
experiments/  基线、消融、任务评估及结果说明
tests/        自动化测试
scripts/      项目维护与实验结果整理脚本
docs/         研究资料
```

实验目录说明见 [experiments/results/README.md](experiments/results/README.md)。

## 开发检查

```bash
uv run pytest
uv run ruff check src tests experiments scripts
uv run mypy
```

`src/dcvc_compression/vendor/DCVC/` 包含带本地修改的第三方 DCVC 源码，使用时请遵循该目录内的许可证和说明。
