# D1 / D2 展示命令说明

本文档用于课堂展示。当前只整理 D1 初始化和 D2 重置相关命令，后续 D3、D4 等模块再继续补充。

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

## 推荐展示顺序

```powershell
cd "C:\Users\Chen Qiang\Desktop\MyODPlat"

& "D:\app\miniconda3\envs\odplat\Scripts\odp-init.exe"

& "D:\app\miniconda3\envs\odplat\Scripts\odp-reset.exe" --dry-run

& "D:\app\miniconda3\envs\odplat\Scripts\odp-reset.exe" --backup-core --dry-run
```

## 现场提醒

课堂展示时只运行:

```text
odp-init
odp-reset --dry-run
odp-reset --backup-core --dry-run
```

不要现场运行带 `--yes` 的 reset 命令, 避免清理当前已经可以检测的项目产物。
