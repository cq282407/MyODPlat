# apps/desktop

## Current Status

This app now provides a minimal D8 desktop demo for local inference preview.

## What It Uses

- `od_platform.inference.InferService`
- `QtSignalSink`
- `InferHooks`
- `CancelToken`

The inference engine itself stays inside `apps/platform`; the desktop app only consumes the exposed D8 seams.

## Install

`PySide6` is required and is not bundled in the current environment by default:

```bat
cd /d D:\myodplat\MyODPlat
python -m pip install PySide6
```

## Run

```bat
python apps\desktop\main.py
```

## Features

- choose model path
- choose source path
- Start / Stop
- live frame display
- status bar loop FPS text

## Notes

- GUI mode does not use `cv2.imshow`
- Stop uses `CancelToken`, so the pipeline can exit gracefully
- The GUI should stay responsive because inference runs in `QThread`
