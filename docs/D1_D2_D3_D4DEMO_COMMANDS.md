# D1 / D2 / D3 / D4 / D5 / D6 / D7 / D8 展示命令说明

本文档用于课堂展示。当前整理 D1 初始化、D2 重置、D3 数据流水线、D4 质检系统、D5 运行配置、D6 训练系统、D7 模型验证和 D8 推理系统相关命令，后续模块再继续补充。

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

## D5: 运行配置系统 runtime_config

D5 用于生成 train / val / infer 的运行配置模板。这里只展示已有命令，不做扩展。

### D5.1 查看配置生成命令帮助

展示命令:

```powershell
& "D:\app\miniconda3\envs\odplat\Scripts\odp-gen-config.exe" --help
```

展示说明:

- 只打印配置生成命令帮助，不会写入项目文件。
- 可以看到当前支持生成 `train`、`val`、`infer` 三类配置。

### D5.2 生成 train 配置模板到临时目录

展示命令:

```powershell
& "D:\app\miniconda3\envs\odplat\Scripts\odp-gen-config.exe" train -o "C:\tmp\odplat_d5_train_demo.yaml"
```

展示说明:

- 生成训练配置模板到 `C:\tmp`，不修改项目已有配置。
- 如果目标文件已经存在，默认跳过，不覆盖。

### D5.3 生成 val / infer 配置模板到临时目录

展示命令:

```powershell
& "D:\app\miniconda3\envs\odplat\Scripts\odp-gen-config.exe" val -o "C:\tmp\odplat_d5_val_demo.yaml"
& "D:\app\miniconda3\envs\odplat\Scripts\odp-gen-config.exe" infer -o "C:\tmp\odplat_d5_infer_demo.yaml"
```

展示说明:

- `val` 用于模型验证配置。
- `infer` 用于模型推理配置。
- 两个命令都写到 `C:\tmp`，不会影响项目当前运行结果。

常见产物:

```text
C:\tmp\odplat_d5_train_demo.yaml
C:\tmp\odplat_d5_val_demo.yaml
C:\tmp\odplat_d5_infer_demo.yaml
```

## D6: 训练系统 training

D6 训练命令会调用 Ultralytics，真实执行时会读取数据集、加载模型并开始训练。课堂安全展示建议先展示帮助和单元测试，不建议现场跑真实训练。

### D6.1 查看训练命令帮助

展示命令:

```powershell
& "D:\app\miniconda3\envs\odplat\Scripts\odp-train.exe" --help
```

展示说明:

- 只打印训练命令参数，不会读取数据集，不会加载模型，不会开始训练。
- 可以重点展示新增参数:
  - `--run-id`: 预设训练运行 ID，用于统一训练目录、日志、报告和归档权重命名。
  - `--operator`: 记录是谁执行训练。
  - `--operator-role`: 记录执行者角色或小组。
  - `--notes`: 记录本次训练说明。

### D6.2 D6 相关测试

展示命令:

```powershell
$env:PYTHONPATH="C:\Users\Chen Qiang\Desktop\MyODPlat\apps\platform\src"
& "D:\app\miniconda3\envs\odplat\python.exe" -m pytest apps\platform\tests\test_training.py apps\platform\tests\test_training_common.py -q
```

展示说明:

- 这是安全展示命令，不会触发真实训练。
- 测试覆盖训练服务编排、训练前质检拦截、训练报告、权重归档和日志重命名。
- 当前验证结果:

```text
19 passed
```

### D6.3 真实训练命令示例

课堂现场不建议运行。该命令会真正开始训练:

```powershell
& "D:\app\miniconda3\envs\odplat\Scripts\odp-train.exe" --data "C:\Users\Chen Qiang\Desktop\MyODPlat\data\processed\d3_demo_nwpu\dataset.yaml" --model "yolo11n.pt" --epochs 1 --imgsz 640 --batch 4 --run-id d6_demo_nwpu --operator "课堂展示" --operator-role "学生小组" --notes "D6 training report demo"
```

展示说明:

- `--run-id d6_demo_nwpu` 会让训练输出、归档权重、训练报告使用统一名称。
- 训练开始前会自动执行 D4 数据质检，质检失败达到 ERROR 时会阻止训练。
- 真实训练不会修改原始数据集，但会生成训练产物和模型权重。

常见产物:

```text
runs/detect_train/d6_demo_nwpu*/
├── weights/best.pt
├── weights/last.pt
├── training_report.json
├── training_report.md
├── training_results.png
└── odp_audit.json

models/trained/
├── d6_demo_nwpu-best.pt
└── d6_demo_nwpu-last.pt

runs/data_validation/d6_demo_nwpu_precheck/
├── report.json
├── report.md
├── report.html
└── report.docx

apps/platform/logging/train/*d6_demo_nwpu*.log
```

### D6.4 训练报告展示点

训练成功后，重点打开:

```text
runs\detect_train\d6_demo_nwpu\training_report.md
runs\detect_train\d6_demo_nwpu\training_report.json
```

可以回答老师的问题:

- 谁训练的: `operator` / `operator_role`
- 什么时间训练的: `started_at` / `created_at`
- 用哪份数据训练的: `dataset.yaml`
- 数据质量怎么样: 自动关联训练前质检摘要
- 训练结果怎么样: metrics、best/last 权重、训练曲线和日志路径

## D7: 模型验证系统 evaluation

D7 使用已经训练好的模型在数据集上做验证评估，不会重新训练模型。这里只展示已有命令，不做扩展。

当前可用模型:

```text
models\trained\train-8-20260707-200344-yolo11s-best.pt
```

### D7.1 查看模型验证命令帮助

展示命令:

```powershell
& "D:\app\miniconda3\envs\odplat\Scripts\odp-val.exe" --help
```

展示说明:

- 只打印验证命令参数，不会读取模型，不会生成验证结果。
- 可以看到 `--model`、`--data`、`--split`、`--conf`、`--iou`、`--save-json` 等参数。

### D7.2 使用 NWPU 模型做 val 集验证

展示命令:

```powershell
& "D:\app\miniconda3\envs\odplat\Scripts\odp-val.exe" --model "train-8-20260707-200344-yolo11s-best.pt" --data "C:\Users\Chen Qiang\Desktop\MyODPlat\data\processed\d3_demo_nwpu\dataset.yaml" --split val --imgsz 640 --batch 4 --name d7_demo_nwpu --no-plots
```

展示说明:

- 使用已经训练好的 NWPU 模型做验证评估。
- 不会重新训练模型。
- 会读取 `dataset.yaml` 和模型权重，生成验证指标和审计信息。
- 如果 `runs/detect_val/d7_demo_nwpu` 已存在，Ultralytics 会自动生成带后缀的新目录。

常见产物:

```text
runs/detect_val/d7_demo_nwpu*/
├── odp_audit.json
└── 其他 Ultralytics 验证产物

apps/platform/logging/val/*.log
```

### D7.3 D7 相关测试

展示命令:

```powershell
$env:PYTHONPATH="C:\Users\Chen Qiang\Desktop\MyODPlat\apps\platform\src"
& "D:\app\miniconda3\envs\odplat\python.exe" -m pytest apps\platform\tests\test_evaluation.py -q
```

展示说明:

- 这是安全展示命令，不会触发真实模型验证。
- 测试覆盖 D7 服务编排、模型路径解析、CLI 参数解析、验证审计产物。

## D1-D7 推荐展示顺序

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

& "D:\app\miniconda3\envs\odplat\Scripts\odp-gen-config.exe" --help

& "D:\app\miniconda3\envs\odplat\Scripts\odp-gen-config.exe" train -o "C:\tmp\odplat_d5_train_demo.yaml"

& "D:\app\miniconda3\envs\odplat\Scripts\odp-train.exe" --help

$env:PYTHONPATH="C:\Users\Chen Qiang\Desktop\MyODPlat\apps\platform\src"
& "D:\app\miniconda3\envs\odplat\python.exe" -m pytest apps\platform\tests\test_training.py apps\platform\tests\test_training_common.py -q

& "D:\app\miniconda3\envs\odplat\Scripts\odp-val.exe" --help

$env:PYTHONPATH="C:\Users\Chen Qiang\Desktop\MyODPlat\apps\platform\src"
& "D:\app\miniconda3\envs\odplat\python.exe" -m pytest apps\platform\tests\test_evaluation.py -q
```

## D8: 推理系统 inference

D8 使用已经训练完成的模型进行推理演示，不需要重新训练。当前可用模型为:

```text
models\trained\train-8-20260707-200344-yolo11s-best.pt
```

课堂演示建议使用已经准备好的快版视频:

```text
runs\d8_video_demo\nwpu_val_fast_demo.mp4
```

该视频约 240 帧，20 FPS，时长约 12 秒，适合展示弹窗推理过程。

### D8.1 查看推理命令帮助

展示命令:

```powershell
& "D:\app\miniconda3\envs\odplat\Scripts\odp-infer.exe" --help
```

展示说明:

- 用于证明 D8 推理系统已经接入正式 `odp-*` 命令体系。
- 可以看到 `--source`、`--model`、`--show`、`--save`、`--vid-stride` 等参数。
- 只打印帮助信息，不会读取模型，也不会生成推理结果。

### D8.2 视频推理，无弹窗保存结果

展示命令:

```powershell
& "D:\app\miniconda3\envs\odplat\Scripts\odp-infer.exe" --source "runs\d8_video_demo\nwpu_val_fast_demo.mp4" --model "train-8-20260707-200344-yolo11s-best.pt" --conf 0.25 --save --no-show --name d8_video_no_show --batch 4
```

展示说明:

- 使用训练好的 NWPU 模型对视频逐帧推理。
- `--no-show` 表示不弹窗，适合快速跑完整个视频并保存结果。
- `--save` 会保存标注后的视频到 `runs\detect_infer\d8_video_no_show*`。
- 如果同名输出目录已经存在，程序会自动生成 `d8_video_no_show2`、`d8_video_no_show3` 等目录，不会覆盖旧结果。

常见产物:

```text
runs/detect_infer/d8_video_no_show*/
├── output.mp4
└── odp_audit.json
```

### D8.3 视频推理，弹窗显示过程

展示命令:

```powershell
& "D:\app\miniconda3\envs\odplat\Scripts\odp-infer.exe" --source "runs\d8_video_demo\nwpu_val_fast_demo.mp4" --model "train-8-20260707-200344-yolo11s-best.pt" --conf 0.25 --show --save --name d8_video_show_fast --batch 4
```

展示说明:

- `--show` 会打开 OpenCV 窗口，实时显示检测框、中文标签和 HUD 信息。
- `--save` 同时保存标注后的视频结果。
- 弹窗退出按 `q` 或 `Esc`。
- 如果现场觉得弹窗播放偏慢，可以改用 D8.2 的 `--no-show` 命令证明推理速度。

### D8.4 视频抽帧推理 vid_stride

展示命令:

```powershell
& "D:\app\miniconda3\envs\odplat\Scripts\odp-infer.exe" --source "runs\d8_video_demo\nwpu_val_fast_demo.mp4" --model "train-8-20260707-200344-yolo11s-best.pt" --conf 0.25 --vid-stride 2 --save --no-show --name d8_stride2_demo --batch 4
```

展示说明:

- `--vid-stride 2` 表示每隔 2 帧取 1 帧进行推理。
- 快版视频约 240 帧，使用 `--vid-stride 2` 后，日志中处理帧数应约为 120。
- 该命令用于证明 D8-1 帧源捕获模块的视频间隔帧功能已经接入正式 CLI。
- 如需弹窗展示抽帧过程，可以把 `--no-show` 改成 `--show`。

### D8.5 摄像头实时推理

展示命令:

```powershell
& "D:\app\miniconda3\envs\odplat\Scripts\odp-infer.exe" --source 0 --model "train-8-20260707-200344-yolo11s-best.pt" --conf 0.25 --show --save --name d8_camera_demo
```

展示说明:

- `--source 0` 表示打开本机默认摄像头。
- 摄像头画面会实时显示检测结果，退出按 `q` 或 `Esc`。
- 当前训练模型是 NWPU 航拍数据集模型，主要识别 airplane、ship、storage_tank、harbor、vehicle 等类别。
- 如果摄像头对人、帽子等日常物体识别效果不好，这是模型类别决定的，不是推理系统故障。

### D8.6 D8 相关测试

展示命令:

```powershell
$env:PYTHONPATH="C:\Users\Chen Qiang\Desktop\MyODPlat\apps\platform\src"
& "D:\app\miniconda3\envs\odplat\python.exe" -m pytest apps\platform\tests\test_frame_source.py apps\platform\tests\test_visualization_beautify.py apps\platform\tests\test_evaluation.py -q
```

展示说明:

- `test_frame_source.py` 覆盖图片、图片文件夹、视频、摄像头工厂、注册表、stride 抽帧。
- `test_visualization_beautify.py` 覆盖中文标签、美化绘制、Pillow 字体渲染。
- `test_evaluation.py` 中包含 `odp-infer --vid-stride` CLI 参数解析测试。

如需展示全项目测试:

```powershell
$env:PYTHONPATH="C:\Users\Chen Qiang\Desktop\MyODPlat\apps\platform\src"
& "D:\app\miniconda3\envs\odplat\python.exe" -m pytest apps\platform\tests -q
```

当前验证结果:

```text
74 passed
```

## D8 推荐展示顺序

```powershell
cd "C:\Users\Chen Qiang\Desktop\MyODPlat"

& "D:\app\miniconda3\envs\odplat\Scripts\odp-infer.exe" --help

& "D:\app\miniconda3\envs\odplat\Scripts\odp-infer.exe" --source "runs\d8_video_demo\nwpu_val_fast_demo.mp4" --model "train-8-20260707-200344-yolo11s-best.pt" --conf 0.25 --save --no-show --name d8_video_no_show --batch 4

& "D:\app\miniconda3\envs\odplat\Scripts\odp-infer.exe" --source "runs\d8_video_demo\nwpu_val_fast_demo.mp4" --model "train-8-20260707-200344-yolo11s-best.pt" --conf 0.25 --show --save --name d8_video_show_fast --batch 4

& "D:\app\miniconda3\envs\odplat\Scripts\odp-infer.exe" --source "runs\d8_video_demo\nwpu_val_fast_demo.mp4" --model "train-8-20260707-200344-yolo11s-best.pt" --conf 0.25 --vid-stride 2 --save --no-show --name d8_stride2_demo --batch 4

& "D:\app\miniconda3\envs\odplat\Scripts\odp-infer.exe" --source 0 --model "train-8-20260707-200344-yolo11s-best.pt" --conf 0.25 --show --save --name d8_camera_demo
```

## D8 现场提醒

课堂展示 D8 时，不要重新训练模型，只使用已经训练好的 `train-8-20260707-200344-yolo11s-best.pt`。

`--show` 用于展示弹窗效果，`--no-show` 用于快速跑完并保存结果。

重复运行 `odp-infer` 不会覆盖旧结果，会自动创建带数字后缀的新输出目录。

如果老师问 `--vid-stride`，可以直接展示 D8.4: 240 帧视频使用 `--vid-stride 2` 后处理帧数约为 120。

如果老师问摄像头为什么不能识别人或帽子，需要说明当前模型训练类别来自 NWPU 航拍数据集，不是 COCO 通用物体模型。

## Web 端实时推理展示台

Web 端展示台用于把 D8 视频推理、摄像头推理、上传视频推理、模型状态和 D1-D8 模块状态集中展示给老师。页面不是播放已经推理好的视频，而是每次点击按钮都会启动一次新的 D8 推理任务，并实时显示模型画框后的帧。

启动 Web 后端:

```powershell
cd "C:\Users\Chen Qiang\Desktop\MyODPlat"
& "D:\app\miniconda3\envs\odplat\python.exe" "C:\Users\Chen Qiang\Desktop\MyODPlat\apps\web-backend\server.py"
```

浏览器打开:

```text
http://127.0.0.1:8765
```

展示说明:

- 页面打开后，点击“启动默认长视频实时推理”，会调用 D8 推理服务实时处理 `runs\d8_video_demo\nwpu_val_long_demo.mp4`。
- 点击“选择本地视频”后，再点击“上传视频并实时推理”，会先上传到 `runs\web_uploads\`，再启动实时推理。
- 点击“打开摄像头实时推理”，会打开本机默认摄像头并把标注帧实时显示在网页中。
- 页面右侧可以手动选择推理模型，模型来源限制为 `models\trained\*.pt` 和 `models\pretrained\*.pt`。
- 页面显示的是本次推理过程中的实时标注帧，不是旧的 `output.mp4` 回放。
- 页面不包含额外答辩提示，只保留演示所需的状态、画面和控制按钮。

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
odp-gen-config --help
odp-gen-config train -o "C:\tmp\odplat_d5_train_demo.yaml"
odp-train --help
python -m pytest apps\platform\tests\test_training.py apps\platform\tests\test_training_common.py -q
odp-val --help
python -m pytest apps\platform\tests\test_evaluation.py -q
```

不要现场运行带 `--yes` 的 reset 命令, 避免清理当前已经可以检测的项目产物。

D3 展示时建议始终使用新的 `--output-name`。如果目录已经存在, 程序会自动创建带 `_2`、`_3` 后缀的新目录, 不覆盖已有成果。

D4 展示时请确认 `--yaml` 指向实际存在的 `dataset.yaml`。质检系统只读数据集, 不会修改数据; pHash 是重型检查, 建议作为扩展项单独展示。

D5 展示时建议输出到 `C:\tmp`，不要覆盖 `apps/platform/configs/runtime/` 下的正式配置。

D6 展示时优先运行 `odp-train --help` 和 D6 训练测试。真实训练命令会启动训练，只在需要重新训练模型时运行。

D7 展示时优先运行 `odp-val --help` 和 D7 测试。真实 `odp-val --model ... --data ...` 会读取权重和数据集并生成验证产物，但不会重新训练。
