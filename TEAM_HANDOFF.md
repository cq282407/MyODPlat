# ODPlatform 团队交接与服务器运行说明

本文档面向团队交接和租赁服务器部署。目标是让后续同学拿到项目后, 能在服务器上直接输入 `odp-*` 命令运行, 而不是依赖某一台电脑上的绝对路径。

## 1. 项目框架

```text
ODPlatform/
├── apps/platform/
│   ├── src/od_platform/
│   │   ├── common/             公共工具: 路径、日志、常量、系统信息
│   │   ├── cli/                命令入口: odp-init / odp-reset / odp-transform / odp-validate / odp-train
│   │   ├── data_pipeline/      数据转换、划分、生成 YOLO 数据集
│   │   ├── data_validation/    数据质量检查
│   │   ├── runtime_config/     train / val / infer 运行配置
│   │   └── training/           D6 训练编排: D5 配置 + D4 校验 + ultralytics 训练
│   ├── configs/
│   │   ├── datasets/           数据集 yaml
│   │   └── runtime/            train.yaml / val.yaml / infer.yaml
│   └── tests/
├── data/
│   ├── raw/                    原始数据, 不由 reset 删除
│   └── processed/              转换后的数据, 可再生成
├── models/
│   ├── pretrained/             预训练权重, 不由 reset 删除
│   └── trained/                训练结果, 可再生成
├── runs/                       运行产物
├── scripts/                    开发辅助脚本
└── .odp-workspace              工作区 marker, 不要删除
```

## 2. 服务器首次安装

推荐在服务器上创建独立环境:

```bash
conda create -n odplat python=3.12 -y
conda activate odplat
```

进入项目根目录后安装平台包:

```bash
cd /path/to/ODPlatform
python -m pip install -e apps/platform
```

如果是在本机 Windows PowerShell, 你的环境目录是:

```text
D:\app\miniconda3\envs\odplat
```

对应 Python 解释器是:

```text
D:\app\miniconda3\envs\odplat\python.exe
```

对应 `odp-*` 命令目录是:

```text
D:\app\miniconda3\envs\odplat\Scripts
```

如果 PowerShell 没有激活 conda 环境, 可以临时把命令目录加入当前窗口的 PATH:

```powershell
$env:Path = "D:\app\miniconda3\envs\odplat;D:\app\miniconda3\envs\odplat\Scripts;D:\app\miniconda3\envs\odplat\Library\bin;$env:Path"
```

这一步会安装依赖并注册命令:

```text
odp-init
odp-reset
odp-transform
odp-validate
odp-gen-config
odp-train
```

确认命令已经进入 PATH:

```bash
which odp-init
which odp-gen-config
which odp-train
```

Windows PowerShell 对应:

```powershell
Get-Command odp-init
Get-Command odp-gen-config
Get-Command odp-train
```

## 3. 初始化与重置

初始化项目目录:

```bash
odp-init
```

查看 reset 计划, 默认不会删除:

```bash
odp-reset
```

确认要清理可再生成产物时:

```bash
odp-reset --yes
```

`odp-reset` 会保护:

```text
data/raw/
models/pretrained/
代码、文档、元工具日志
```

## 4. 换新数据集的标准流程

把数据放到:

```text
data/raw/<dataset>/
├── images/
└── annotations/
```

例如:

```text
data/raw/helmet/
├── images/
└── annotations/
```

转换数据:

```bash
odp-transform --dataset helmet --format pascal_voc
```

验证数据:

```bash
odp-validate --dataset helmet
```

生成运行配置:

```bash
odp-gen-config train --overwrite
odp-gen-config val --overwrite
odp-gen-config infer --overwrite
```

当前已落地的转换器以 `pascal_voc` 为主。若新数据集是 COCO、YOLO、DOTA、LabelMe 等其他格式, 需要在 `data_pipeline/convert/converters/` 下新增对应 converter 并注册。

## 5. NWPU VHR-10 数据集服务器完整运行流程

这一节是当前遥感数据集的推荐服务器流程。核心原则:

- 服务器上不要复用本机 Windows 生成的 `dataset.yaml`。
- 本机生成的 yaml 里可能有 `C:/Users/Chen Qiang/...` 绝对路径, Linux 服务器不能用。
- 到服务器后重新运行 D3 `odp-transform`, 让 yaml 自动写成服务器路径。

### 5.1 本地上传项目和原始数据

在本机 PowerShell 中上传项目:

```powershell
scp -r "C:\Users\Chen Qiang\Desktop\MyODPlatform" 用户名@服务器IP:~/
```

如果项目太大, 至少要保证服务器上有代码和原始数据:

```text
~/MyODPlatform/
~/MyODPlatform/data/raw/NWPU VHR-10 dataset/
```

NWPU VHR-10 原始数据目录应保持:

```text
data/raw/NWPU VHR-10 dataset/
├── positive image set/
├── negative image set/
└── ground truth/
```

### 5.2 登录服务器并进入项目

```bash
ssh 用户名@服务器IP
cd ~/MyODPlatform
```

### 5.3 创建并激活 Python 环境

推荐 Python 3.11 或 3.12:

```bash
conda create -n odplat python=3.11 -y
conda activate odplat
```

检查 GPU:

```bash
nvidia-smi
```

如果没有 GPU 或 `nvidia-smi` 不可用, 训练会很慢; 正式训练建议换有 NVIDIA GPU 的机器。

### 5.4 安装项目命令

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e apps/platform
```

检查命令:

```bash
odp-init --help
odp-transform --help
odp-validate --help
odp-gen-config --help
odp-train --help
```

如果提示 `odp-train: command not found`, 先执行:

```bash
export PATH="$CONDA_PREFIX/bin:$PATH"
```

然后再检查:

```bash
odp-train --help
```

### 5.5 初始化目录

```bash
odp-init
```

### 5.6 清理旧的 processed 数据

如果服务器上之前跑过同名输出, 建议先删除:

```bash
rm -rf data/processed/nwpu_demo_random
```

说明: `odp-transform --output-name nwpu_demo_random` 会写回同一个目录:

```text
data/processed/nwpu_demo_random
```

它不会自动创建 `nwpu_demo_random2`。为避免旧文件残留, 换服务器或换数据集时建议先 `rm -rf`。

### 5.7 转换并划分数据集

```bash
odp-transform \
  --dataset "NWPU VHR-10 dataset" \
  --format nwpu_vhr10 \
  --task detect \
  --split-strategy random \
  --train-rate 0.8 \
  --val-rate 0.1 \
  --test-rate 0.1 \
  --seed 42 \
  --output-name nwpu_demo_random \
  --include-unlabeled
```

转换后检查:

```bash
ls data/processed/nwpu_demo_random
cat data/processed/nwpu_demo_random/dataset.yaml
```

服务器上的 `dataset.yaml` 应类似:

```yaml
path: /home/你的用户名/MyODPlatform/data/processed/nwpu_demo_random
train: images/train
val: images/val
test: images/test
nc: 10
```

如果看到 `C:/Users/...`, 说明误用了本机生成的 yaml, 需要重新跑 transform。

### 5.8 数据质量检测

```bash
odp-validate \
  --yaml data/processed/nwpu_demo_random/dataset.yaml \
  --operator ChenQiang \
  --operator-role Architect \
  --operation server_pretrain_quality_gate \
  --notes "Server-side validation before D6 training"
```

检查报告:

```bash
ls runs/data_validation
```

如果有 ERROR, 先修数据。正式训练不建议使用 `--no-pre-validate` 跳过校验。

### 5.9 准备预训练模型

如果服务器能联网, 可以直接使用:

```text
yolo11n.pt
```

ultralytics 会自动下载。

如果服务器不能联网, 在本机先下载或准备 `yolo11n.pt`, 上传到:

```text
models/pretrained/yolo11n.pt
```

本机 PowerShell 示例:

```powershell
scp "C:\path\to\yolo11n.pt" 用户名@服务器IP:~/MyODPlatform/models/pretrained/
```

服务器上确认:

```bash
ls models/pretrained
```

### 5.10 正式训练

推荐先使用:

```bash
odp-train \
  --yaml train.yaml \
  --data data/processed/nwpu_demo_random/dataset.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --device 0
```

如果显存不足, 降低 batch:

```bash
odp-train \
  --yaml train.yaml \
  --data data/processed/nwpu_demo_random/dataset.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --batch 8 \
  --imgsz 640 \
  --device 0
```

仍然不足时:

```bash
odp-train \
  --yaml train.yaml \
  --data data/processed/nwpu_demo_random/dataset.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --batch 4 \
  --imgsz 640 \
  --device 0
```

### 5.11 后台运行训练

服务器训练建议用 `tmux`, 防止 SSH 断开导致训练中断:

```bash
tmux new -s odtrain
```

在 tmux 里运行训练命令:

```bash
odp-train \
  --yaml train.yaml \
  --data data/processed/nwpu_demo_random/dataset.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --device 0
```

临时离开 tmux:

```text
Ctrl+B
D
```

重新进入:

```bash
tmux attach -t odtrain
```

### 5.12 查看训练结果

训练输出目录:

```bash
ls runs/detect_train
```

第一次通常是:

```text
runs/detect_train/train
```

第二次、第三次可能是:

```text
runs/detect_train/train2
runs/detect_train/train3
```

查看关键产物:

```bash
ls runs/detect_train/train
ls runs/detect_train/train/weights
ls runs/detect_train/train/odp_audit.json
ls runs/detect_train/train/training_results.png
```

权重归档目录:

```bash
ls models/trained
```

D6 训练完成后应重点保留:

```text
runs/detect_train/<run_name>/weights/best.pt
runs/detect_train/<run_name>/weights/last.pt
runs/detect_train/<run_name>/odp_audit.json
runs/detect_train/<run_name>/training_results.png
models/trained/
```

### 5.13 最短完整命令清单

服务器已经有代码和原始数据时, 核心命令如下:

```bash
cd ~/MyODPlatform
conda activate odplat
python -m pip install -e apps/platform

odp-init

rm -rf data/processed/nwpu_demo_random

odp-transform \
  --dataset "NWPU VHR-10 dataset" \
  --format nwpu_vhr10 \
  --task detect \
  --split-strategy random \
  --train-rate 0.8 \
  --val-rate 0.1 \
  --test-rate 0.1 \
  --seed 42 \
  --output-name nwpu_demo_random \
  --include-unlabeled

odp-validate \
  --yaml data/processed/nwpu_demo_random/dataset.yaml \
  --operator ChenQiang \
  --operator-role Architect \
  --operation server_pretrain_quality_gate

odp-train \
  --yaml train.yaml \
  --data data/processed/nwpu_demo_random/dataset.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --device 0
```

## 6. 验收命令

在项目根目录运行:

```bash
python -m pytest apps/platform/tests
```

预期结果:

```text
46 passed
```

## 7. 架构师交接重点

- `common/paths.py` 是路径唯一真相来源, 不要在业务代码里手写项目绝对路径。
- `data/raw/` 是原始数据保护区。
- `data/processed/`、`runs/`、`models/trained/` 是可再生成产物。
- `apps/platform/pyproject.toml` 的 `[project.scripts]` 决定 `odp-*` 命令是否能直接运行。
- 服务器上不要写死本机路径, 正确方式是激活环境后使用 `odp-*` 命令。
- D6 训练入口是 `odp-train`, 训练前默认会调用 D4 数据校验。
- 本机生成的 `dataset.yaml` 可能含 Windows 绝对路径, 服务器训练前必须重新 transform 或手动确认 yaml 的 `path` 是服务器路径。
