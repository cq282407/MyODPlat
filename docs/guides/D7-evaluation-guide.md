# D7 Evaluation Guide

This guide describes how to run the new D7 model evaluation flow with
`odp-val`.

## What D7 Does

`odp-val` evaluates a trained YOLO weight on a dataset split and writes the
evaluation outputs into `runs/<task>_val/`.

It reuses:

- D5 runtime config: `val.yaml`
- D3 path resolution: trained weights under `models/trained/`
- D6 metric formatting and log alignment

It intentionally does **not**:

- download missing trained weights
- archive new weights
- depend on the training subsystem

## Prerequisites

1. Install the platform package:

```bash
python -m pip install -e apps/platform
```

2. Make sure you already have:

- a dataset yaml produced by D3, for example `data/processed/<dataset>/dataset.yaml`
- a trained weight archived under `models/trained/`, for example `train3-best.pt`

## Step 1: Generate or Inspect `val.yaml`

Generate the default template if needed:

```bash
odp-gen-config val --overwrite
```

Default runtime config path:

```text
apps/platform/configs/runtime/val.yaml
```

You can edit fields such as:

- `data`
- `model`
- `split`
- `batch`
- `imgsz`
- `device`
- `conf`
- `iou`

## Step 2: Run Evaluation

### Option A: use `val.yaml`

```bash
odp-val --yaml val.yaml
```

### Option B: override by CLI

```bash
odp-val \
  --yaml val.yaml \
  --model train3-best.pt \
  --data data/processed/nwpu_demo_random/dataset.yaml \
  --split test \
  --batch 8 \
  --imgsz 640 \
  --device 0
```

Windows PowerShell example:

```powershell
odp-val `
  --yaml val.yaml `
  --model train3-best.pt `
  --data data/processed/nwpu_demo_random/dataset.yaml `
  --split test `
  --batch 8 `
  --imgsz 640 `
  --device 0
```

## Weight Resolution Rules

If you pass a bare model name such as:

```text
train3-best.pt
```

ODPlatform resolves it as:

```text
models/trained/train3-best.pt
```

If that file does not exist, D7 fails fast and exits with an error instead of
trying to download anything.

## Outputs

Evaluation outputs are written under:

```text
runs/<task>_val/<run_name>/
```

Typical artifacts:

- ultralytics validation outputs
- aligned log file
- `odp_audit.json`

`odp_audit.json` includes:

- `kind: "val"`
- merged config snapshot
- config source chain
- summary metrics
- model path and dataset path

## Recommended Workflow

```bash
odp-transform ...
odp-validate --yaml data/processed/<dataset>/dataset.yaml
odp-train --yaml train.yaml --data data/processed/<dataset>/dataset.yaml --model yolo11n.pt
odp-val --yaml val.yaml --model train3-best.pt --data data/processed/<dataset>/dataset.yaml
```

## Troubleshooting

### `找不到已训练权重`

Check:

- the weight is really under `models/trained/`
- the filename is correct
- you are evaluating an archived trained model, not a pretrained model

### `数据集配置文件未找到`

Check:

- the dataset yaml path is correct
- the D3 output directory still exists
- the `data` field in `val.yaml` points to the right yaml

### How to see available CLI options

```bash
odp-val --help
```
