# ODPlatform 模块使用说明

本文档说明当前项目中各模块的功能、主要产物和常用运行命令。默认在项目根目录执行命令。

## 环境准备

推荐使用项目环境:

```powershell
D:\app\miniconda3\envs\odplat\python.exe -m pip install -e apps/platform
```

若已激活 `odplat` 环境, 可以直接使用 `python` 和 `odp-*` 命令。

## 核心工具层

| 模块 | 功能 | 主要产物 | 运行命令 |
| --- | --- | --- | --- |
| `od_platform.common.paths` | 管理项目根目录、数据目录、模型目录、日志目录、元工具目录等路径; 提供初始化目录清单、reset 清理目录清单、保护目录判断。 | 路径常量; `.odp-workspace`; `.odp-meta/logging/`; `.odp-meta/backups/` | `python apps/platform/src/od_platform/common/paths.py` |
| `od_platform.common.logging_utils` | 初始化项目日志, 同时输出到控制台和文件; 支持按业务类型分目录; 临时日志名包含 `log_type` 和可选 `run_id`。 | `apps/platform/logging/<log_type>/*.log`; `.odp-meta/logging/<log_type>/*.log` | 由各 CLI 自动调用, 通常不单独运行 |
| `od_platform.common.performance_utils` | 提供 `@time_it` 装饰器, 记录函数耗时、平均耗时和自动单位。 | 日志中的性能报告 | `python apps/platform/src/od_platform/common/performance_utils.py` |
| `od_platform.common.string_utils` | 提供中英文混排宽度计算和表格对齐工具。 | 对齐后的日志表格文本 | `python apps/platform/src/od_platform/common/string_utils.py` |
| `od_platform.common.system_utils` | 采集并记录 Python、OS、CPU、内存、GPU 等运行环境信息。 | 初始化、训练、验证日志中的环境快照 | 由 `odp-init`、训练、验证流程自动调用 |

## 初始化与重置

| 模块 | 功能 | 主要产物 | 运行命令 |
| --- | --- | --- | --- |
| `od_platform.cli.init_project` | 创建项目核心目录, 检查 `data/raw/` 原始数据状态, 记录环境信息。 | `data/`; `models/`; `runs/`; `apps/platform/logging/`; `.odp-meta/` | `odp-init` 或 `python scripts/init_project.py` |
| `od_platform.cli.reset_project` | 安全清理可再生成运行产物。默认 dry-run; 保护原始数据、预训练权重、代码、文档和元工具日志。 | 清理 `data/processed/`; `runs/`; `apps/platform/logging/`; `models/trained/` | 预览: `odp-reset`; 删除: `odp-reset --yes`; 跳过确认: `odp-reset --yes --force` |
| `reset 核心备份扩展` | 在真正 reset 前可选打包核心代码、配置、脚本和文档。不会备份 `data/`、`models/`、`runs/` 等大目录。 | `.odp-meta/backups/reset-core-*.zip`; `.odp-meta/backups/reset-core-*.manifest.json` | `odp-reset --backup-core --yes`; 自动确认场景: `odp-reset --backup-core --yes --force` |

## 数据处理与校验

| 模块 | 功能 | 主要产物 | 运行命令 |
| --- | --- | --- | --- |
| `od_platform.data_pipeline` | 将原始数据转换为平台使用的数据格式, 支持数据划分和物化输出。 | `data/processed/<dataset>/`; `apps/platform/configs/datasets/*.yaml` | `odp-transform ...` 或 `python scripts/transform_data.py ...` |
| `od_platform.cli.transform_data` | 数据转换 CLI 入口。 | 转换后的图片、标签、数据集 yaml | 查看参数: `odp-transform --help` |
| `od_platform.data_validation` | 对 YOLO 数据集执行结构、标签、bbox、类别覆盖、重复标注等检查。 | `runs/data_validation/<run_id>/report.json`; `report.md`; `report.html`; `summary.xlsx` | `odp-validate --dataset <name>` 或 `odp-validate --yaml <path>` |
| `od_platform.cli.validate_data` | 数据校验 CLI 入口, 支持指定数据集名、yaml 路径、输出目录和 run id。 | 校验报告和审计文件 | `odp-validate --dataset nwpu --run-id smoke` |

## 运行配置

| 模块 | 功能 | 主要产物 | 运行命令 |
| --- | --- | --- | --- |
| `od_platform.runtime_config` | 定义 train / val / infer 的运行配置模型, 合并默认值、yaml 和 CLI 参数。 | 运行时配置对象 | 由训练、验证、推理流程自动调用 |
| `od_platform.runtime_config.generator` | 根据配置模型生成带注释的 yaml 模板; 覆盖时默认备份旧文件。 | `apps/platform/configs/runtime/train.yaml`; `val.yaml`; `infer.yaml`; `*.bak.*` | `odp-gen-config train`; 覆盖: `odp-gen-config train --overwrite` |

## 训练、验证与模型产物

| 模块 | 功能 | 主要产物 | 运行命令 |
| --- | --- | --- | --- |
| `od_platform.training` | 编排训练流程: 读取配置、训练前校验、调用 Ultralytics、归档权重、生成训练图表和审计信息。 | `runs/detect/...`; `models/trained/*.pt`; 训练日志; 审计 json | `odp-train --help`; 常用: `odp-train --data <dataset.yaml> --epochs 1` |
| `od_platform.training.archive` | 归档训练生成的 `best.pt` 和 `last.pt`。 | `models/trained/<run>_best.pt`; `<run>_last.pt` | 由 `odp-train` 自动调用; 可用 `--no-archive` 关闭 |
| `od_platform.cli.evaluate_model` | 模型验证 CLI 入口, 调用验证服务并输出指标。 | 验证日志; `runs/...` 下的验证结果 | `odp-val --help`; 常用: `odp-val --weights <model.pt> --data <dataset.yaml>` |
| `od_platform.evaluation` | 封装模型验证服务, 读取配置、调用 Ultralytics val、整理指标。 | 验证指标、审计信息、日志 | 由 `odp-val` 自动调用 |

## 可视化与视频/摄像头输入

| 模块 | 功能 | 主要产物 | 运行命令 |
| --- | --- | --- | --- |
| `od_platform.visualization` | 绘制检测框、标签、置信度和美化后的可视化结果。 | 渲染后图片或视频帧 | 由演示脚本或业务代码调用 |
| `od_platform.frame_source` | 抽象图片、图片文件夹、视频、摄像头输入, 支持同步、线程和异步封装。 | 标准化帧对象 | 由视频/摄像头脚本调用 |
| `scripts/run_beautified_video.py` | 对视频文件执行检测并输出美化后视频。 | 输出视频文件 | `python scripts/run_beautified_video.py ...` |
| `scripts/run_beautified_camera.py` | 打开摄像头执行实时检测和可视化。 | 实时窗口或输出流 | `python scripts/run_beautified_camera.py ...` |

## 推荐安全命令

```powershell
# 1. 初始化目录
odp-init

# 2. 查看 reset 会清理什么, 不会删除
odp-reset

# 3. 真正 reset 前备份核心文件
odp-reset --backup-core --yes

# 4. 跳过交互确认, 适合脚本环境
odp-reset --backup-core --yes --force

# 5. 运行定向测试
D:\app\miniconda3\envs\odplat\python.exe -m pytest apps/platform/tests/test_paths.py apps/platform/tests/test_reset_project.py apps/platform/tests/test_training_common.py -q
```
