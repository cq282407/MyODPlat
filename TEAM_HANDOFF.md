# ODPlatform 团队交接与服务器运行说明

本文档面向团队交接和租赁服务器部署。目标是让后续同学拿到项目后, 能在服务器上直接输入 `odp-*` 命令运行, 而不是依赖某一台电脑上的绝对路径。

## 1. 项目框架

```text
ODPlatform/
├── apps/platform/
│   ├── src/od_platform/
│   │   ├── common/             公共工具: 路径、日志、常量、系统信息
│   │   ├── cli/                命令入口: odp-init / odp-reset / odp-transform / odp-validate
│   │   ├── data_pipeline/      数据转换、划分、生成 YOLO 数据集
│   │   ├── data_validation/    数据质量检查
│   │   └── runtime_config/     train / val / infer 运行配置
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
```

确认命令已经进入 PATH:

```bash
which odp-init
which odp-gen-config
```

Windows PowerShell 对应:

```powershell
Get-Command odp-init
Get-Command odp-gen-config
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

## 5. 验收命令

在项目根目录运行:

```bash
python -m pytest apps/platform/tests
```

预期结果:

```text
21 passed
```

## 6. 架构师交接重点

- `common/paths.py` 是路径唯一真相来源, 不要在业务代码里手写项目绝对路径。
- `data/raw/` 是原始数据保护区。
- `data/processed/`、`runs/`、`models/trained/` 是可再生成产物。
- `apps/platform/pyproject.toml` 的 `[project.scripts]` 决定 `odp-*` 命令是否能直接运行。
- 服务器上不要写死本机路径, 正确方式是激活环境后使用 `odp-*` 命令。
