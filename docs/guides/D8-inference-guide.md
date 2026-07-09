# D8 Inference Guide

This guide describes the full D8 inference subsystem now implemented in ODPlatform.

## What Is Included

D8 now has:

- `odp-infer` CLI
- four-stage threaded pipeline
- FPS HUD for CLI display mode
- audit snapshot output
- desktop GUI demo based on PySide6

The pipeline stages are:

1. reader
2. infer
3. render
4. display

## Runtime Files

Main runtime files:

- `apps/platform/configs/runtime/infer.yaml`
- `apps/platform/configs/runtime/infer_pipeline.yaml`

`infer.yaml` controls YOLO inference arguments.

`infer_pipeline.yaml` controls:

- camera capture settings
- label mapping
- color mapping
- visualization style

## CLI Usage

From repo root:

```bat
cd /d D:\myodplat\MyODPlat
python -m pip install -e apps/platform
```

Single image:

```bat
odp-infer --model train-3-20260707-142443-yolo11n-best.pt --source data\processed\nwpu_demo_random\images\test\001.jpg
```

Image directory:

```bat
odp-infer --model train-3-20260707-142443-yolo11n-best.pt --source data\processed\nwpu_demo_random\images\test
```

Video:

```bat
odp-infer --model train-3-20260707-142443-yolo11n-best.pt --source demo.mp4 --imgsz 640 --conf 0.25
```

Camera with live FPS HUD:

```bat
odp-infer --model yolo11n.pt --source 0 --show --no-save
```

Use the pipeline yaml explicitly:

```bat
odp-infer --yaml infer.yaml --pipeline-yaml infer_pipeline.yaml --model yolo11n.pt --source 0 --show
```

Important CLI controls:

- `q` or `Esc`: quit live window
- `Space`: pause / resume live window

## GUI Usage

PySide6 is not part of the current environment by default, so install it first:

```bat
python -m pip install PySide6
```

Then run:

```bat
python apps\desktop\main.py
```

The GUI includes:

- model input box
- source input box
- Start / Stop
- live frame display
- status bar FPS text

GUI FPS is shown in the status bar as `loop XX.X FPS`.

## Output Location

D8 outputs are written under:

```text
runs/<task>_infer/<run_name>/
```

Typical artifacts:

- annotated images or `output.mp4`
- optional txt outputs
- `odp_audit.json`

## Acceleration: Should You Do It Now?

Short answer: not by default.

Do acceleration now only if:

- camera or RTSP real-time inference is already too slow
- deployment machine is fixed
- `.pt` inference has become the main bottleneck

Do not rush acceleration if:

- you are still validating D8 behavior
- image/video offline inference is already acceptable
- you have not measured baseline FPS yet

Recommended order:

1. run D8 with `.pt`
2. measure current loop FPS
3. only then decide whether to export `.engine`

The current D8 implementation already accepts a local `.engine` model path through `--model`, so later TensorRT access can be added without rewriting the whole CLI flow.
