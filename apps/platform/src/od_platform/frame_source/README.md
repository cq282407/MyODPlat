# frame_source

`frame_source` gives inference code one iterator protocol for common inputs:
single images, image folders, videos, cameras and network video streams.

```python
from od_platform.frame_source import create_frame_source

with create_frame_source("images/") as source:
    for frame in source:
        results = model(frame.image)
```

The module intentionally does not import `od_platform.common` or other host
subsystems. It only depends on OpenCV, NumPy and Pydantic so it can be moved as
a small infrastructure package.

