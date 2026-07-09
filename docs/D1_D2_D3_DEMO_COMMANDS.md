# D1 / D2 / D3 / D4 展示命令说明

本文档用于课堂展示。当前整理 D1 初始化、D2 重置、D3 数据流水线和 D4 质检系统相关命令，后续模块再继续补充。

## 准备步骤

先进入项目根目录:

```powershell
cd "C:\Users\Chen Qiang\Desktop\MyODPlat"
```

## D1: 项目初始化 init

展示命令:

```powershell
& "D:\app\miniconda3\envs\odplat\Scripts\odp-init.exe"
```

展示说明:

- D1 初始化会检查并创建项目基础目录。
- 已经存在的目录不会覆盖。
- 主要新增产物是初始化日志。
- 不会修改检测代码、数据集、模型权重。
- 不会重新训练模型。

常见产物:

```text
apps/platform/logging/init_project/*.log
data/
models/
runs/
.odp-meta/
```

## D2: 项目重置 reset 安全预览

展示命令:

```powershell
& "D:\app\miniconda3\envs\odplat\Scripts\odp-reset.exe" --dry-run
```

展示说明:

- D2 reset 默认支持安全预览。
- `--dry-run` 只展示将会清理哪些运行产物、文件数和大小。
- 不会真正删除任何文件。
- 只有加 `--yes` 才会进入真正删除流程。

重点保护目录:

```text
data/raw/
models/pretrained/
apps/platform/src/
apps/platform/configs/
docs/
.odp-meta/
```

## D2 扩展项: reset 前核心文件备份预览

展示命令:

```powershell
& "D:\app\miniconda3\envs\odplat\Scripts\odp-reset.exe" --backup-core --dry-run
```

展示说明:

- 这是 D2 扩展项。
- 表示 reset 支持在真正删除前备份核心文件。
- 当前命令仍然是 dry-run, 只展示备份参数已启用。
- 不会删除文件。
- 不会实际写入备份包。

真正执行 reset 前备份的命令如下, 课堂展示时不建议运行:

```powershell
& "D:\app\miniconda3\envs\odplat\Scripts\odp-reset.exe" --backup-core --yes
```

如果真正执行, 备份产物会生成在:

```text
.odp-meta/backups/reset-core-*.zip
.odp-meta/backups/reset-core-*.manifest.json
```

## D3: 数据流水线 data_pipeline

本示例使用的数据集目录:

```text
C:\Users\Chen Qiang\Desktop\MyODPlat\data\raw\NWPU VHR-10 dataset
```

在命令中, `--dataset` 只填写 `data/raw/` 下面的文件夹名:

```text
NWPU VHR-10 dataset
```

### D3.1 查看已支持的数据格式

展示命令:

```powershell
& "D:\app\miniconda3\envs\odplat\Scripts\odp-transform.exe" --list-capabilities
```

展示说明:

- 用于展示当前数据转换器能力。
- 当前支持 `pascal_voc`、`nwpu_vhr10`、`coco`。
- 这个命令只打印能力列表, 不会生成数据产物。

### D3.2 NWPU 数据转换 + 随机划分 + 默认真实落盘

展示命令:

```powershell
& "D:\app\miniconda3\envs\odplat\Scripts\odp-transform.exe" --dataset "NWPU VHR-10 dataset" --format nwpu_vhr10 --split-strategy random --output-name d3_demo_nwpu
```

展示说明:

- 将 `data/raw/NWPU VHR-10 dataset` 转换为 YOLO 标签格式。
- 使用 `random` 策略划分 train / val / test。
- 默认 `--materialize-mode hardlink`, 会真实落盘到 `images/train`、`labels/train` 等目录。
- 如果 `data/processed/d3_demo_nwpu` 已存在, 程序会自动改成 `d3_demo_nwpu_2`、`d3_demo_nwpu_3` 等, 不覆盖旧成果。

常见产物:

```text
data/processed/d3_demo_nwpu*/
├── images/train/
├── images/val/
├── images/test/
├── labels/train/
├── labels/val/
├── labels/test/
├── dataset.yaml
├── classes.txt
├── conversion_report.txt
├── split_report.json
├── dataset_fingerprint.csv
└── dataset_fingerprint.json
```

### D3.3 扩展项: txt 清单模式 + 数据指纹

展示命令:

```powershell
& "D:\app\miniconda3\envs\odplat\Scripts\odp-transform.exe" --dataset "NWPU VHR-10 dataset" --format nwpu_vhr10 --split-strategy random --output-name d3_demo_nwpu_txt --materialize-mode txt
```

展示说明:

- 这是 D3 扩展项展示命令。
- 会额外生成 `train.txt`、`val.txt`、`test.txt`。
- `dataset.yaml` 会指向 txt 清单:

```yaml
train: train.txt
val: val.txt
test: test.txt
```

- 同时生成数据指纹:

```text
dataset_fingerprint.csv
dataset_fingerprint.json
```

- 指纹记录每个样本的 split、image、label、size_bytes、sha256、image_sha256、label_sha256。
- 如果 `data/processed/d3_demo_nwpu_txt` 已存在, 程序会自动改成 `d3_demo_nwpu_txt_2`、`d3_demo_nwpu_txt_3` 等, 不覆盖旧成果。

### D3.4 COCO 格式扩展示例

如果后续有 COCO 数据集, 命令形式如下:

```powershell
& "D:\app\miniconda3\envs\odplat\Scripts\odp-transform.exe" --dataset "<COCO数据集目录名>" --format coco --split-strategy random --output-name d3_demo_coco
```

展示说明:

- `coco` 是新增支持的数据格式。
- 默认不会影响 `pascal_voc` 和 `nwpu_vhr10`。
- 展示时建议使用新的 `--output-name`, 避免和已有产物混在一起。

## D4: 质检系统 data_validation

D4 质检命令需要传入 D3 生成的 `dataset.yaml`。如果刚刚运行了 D3.2, 常见路径如下:

```text
data\processed\d3_demo_nwpu\dataset.yaml
```

如果因为同名目录已存在, D3 自动生成了 `d3_demo_nwpu_2` 或 `d3_demo_nwpu_3`, 请把下面命令里的路径替换成实际目录。

### D4.1 普通质检

展示命令:

```powershell
& "D:\app\miniconda3\envs\odplat\Scripts\odp-validate.exe" --yaml "C:\Users\Chen Qiang\Desktop\MyODPlat\data\processed\d3_demo_nwpu\dataset.yaml" --run-id d4_demo_nwpu --operator "课堂展示"
```

展示说明:

- 对 D3 生成的数据集执行结构、标签、bbox、类别分布、重复标注等检查。
- 只读数据集, 不修改图片、标签或 yaml。
- 生成报告到 `runs/data_validation/d4_demo_nwpu/`。

常见产物:

```text
runs/data_validation/d4_demo_nwpu/
├── report.json
├── report.md
├── report.html
├── report.docx
├── data_dictionary.json
├── audit.json
├── recommendations.json
├── repair_items.csv
├── repair_items.xlsx
└── charts/
```

### D4.2 扩展项: pHash 重型近重复检测

展示命令:

```powershell
& "D:\app\miniconda3\envs\odplat\Scripts\odp-validate.exe" --yaml "C:\Users\Chen Qiang\Desktop\MyODPlat\data\processed\d3_demo_nwpu\dataset.yaml" --run-id d4_demo_nwpu_phash --operator "课堂展示" --check-phash
```

展示说明:

- `--check-phash` 会启用 pHash 感知哈希近重复图片检测。
- 这是重型检查, 会读取图片内容并计算视觉哈希, 所以默认关闭。
- 检出的近重复图片会写入 `repair_items.csv` / `repair_items.xlsx`, 包含 `paired_file` 和 `distance` 字段。
- 不会修改数据集, 只生成质检报告。

可选阈值:

```powershell
& "D:\app\miniconda3\envs\odplat\Scripts\odp-validate.exe" --yaml "C:\Users\Chen Qiang\Desktop\MyODPlat\data\processed\d3_demo_nwpu\dataset.yaml" --run-id d4_demo_nwpu_phash_tight --operator "课堂展示" --check-phash --phash-threshold 4
```

阈值说明:

- `--phash-threshold` 表示 pHash 汉明距离小于等于该值时, 视为近重复。
- 数值越小越严格, 数值越大越容易报出近重复。

## 推荐展示顺序

```powershell
cd "C:\Users\Chen Qiang\Desktop\MyODPlat"

& "D:\app\miniconda3\envs\odplat\Scripts\odp-init.exe"

& "D:\app\miniconda3\envs\odplat\Scripts\odp-reset.exe" --dry-run

& "D:\app\miniconda3\envs\odplat\Scripts\odp-reset.exe" --backup-core --dry-run

& "D:\app\miniconda3\envs\odplat\Scripts\odp-transform.exe" --list-capabilities

& "D:\app\miniconda3\envs\odplat\Scripts\odp-transform.exe" --dataset "NWPU VHR-10 dataset" --format nwpu_vhr10 --split-strategy random --output-name d3_demo_nwpu

& "D:\app\miniconda3\envs\odplat\Scripts\odp-transform.exe" --dataset "NWPU VHR-10 dataset" --format nwpu_vhr10 --split-strategy random --output-name d3_demo_nwpu_txt --materialize-mode txt

& "D:\app\miniconda3\envs\odplat\Scripts\odp-validate.exe" --yaml "C:\Users\Chen Qiang\Desktop\MyODPlat\data\processed\d3_demo_nwpu\dataset.yaml" --run-id d4_demo_nwpu --operator "课堂展示"

& "D:\app\miniconda3\envs\odplat\Scripts\odp-validate.exe" --yaml "C:\Users\Chen Qiang\Desktop\MyODPlat\data\processed\d3_demo_nwpu\dataset.yaml" --run-id d4_demo_nwpu_phash --operator "课堂展示" --check-phash
```

## 现场提醒

课堂展示时只运行:

```text
odp-init
odp-reset --dry-run
odp-reset --backup-core --dry-run
odp-transform --list-capabilities
odp-transform --dataset "NWPU VHR-10 dataset" --format nwpu_vhr10 --split-strategy random --output-name d3_demo_nwpu
odp-transform --dataset "NWPU VHR-10 dataset" --format nwpu_vhr10 --split-strategy random --output-name d3_demo_nwpu_txt --materialize-mode txt
odp-validate --yaml "data\processed\d3_demo_nwpu\dataset.yaml" --run-id d4_demo_nwpu --operator "课堂展示"
odp-validate --yaml "data\processed\d3_demo_nwpu\dataset.yaml" --run-id d4_demo_nwpu_phash --operator "课堂展示" --check-phash
```

不要现场运行带 `--yes` 的 reset 命令, 避免清理当前已经可以检测的项目产物。

D3 展示时建议始终使用新的 `--output-name`。如果目录已经存在, 程序会自动创建带 `_2`、`_3` 后缀的新目录, 不覆盖已有成果。

D4 展示时请确认 `--yaml` 指向实际存在的 `dataset.yaml`。质检系统只读数据集, 不会修改数据; pHash 是重型检查, 建议作为扩展项单独展示。
