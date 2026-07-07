# ADR-005: runtime_config 子系统设计

## 状态

Accepted

## 背景

训练、验证、推理都需要大量运行参数。如果直接把 YAML、CLI 参数和代码默认值用普通 `dict.update()` 混在一起，会出现三个问题：

- 字段合法性分散，错误要到训练启动后才暴露。
- 覆盖关系不透明，不知道最终值来自默认值、YAML 还是 CLI。
- 模板手写容易和配置模型漂移。

## 决策

新增 `od_platform.runtime_config` 子系统，作为 D6 训练/验证/推理服务之前的配置底座。

核心组件：

- `BaseConfig`：所有运行模式共享字段、验证规则和 `to_ultralytics_kwargs()`。
- `YOLOTrainConfig` / `YOLOValConfig` / `YOLOInferConfig`：分别承载训练、验证、推理参数。
- `YAMLLoader` / `CLILoader`：只负责把外部输入读成 `dict`，不做业务验证。
- `ConfigMerger`：按 `DEFAULT < YAML < CLI` 合并，并记录字段来源和覆盖链。
- `CONFIG_REGISTRY`：统一维护 `train` / `val` / `infer` 到配置类的映射。
- `ConfigGenerator` / `odp-gen-config`：从字段元数据反射生成自解释 YAML 模板。

## 约束

- Loader 不能偷偷生成配置文件；缺文件必须 fail-fast，并提示用户运行 `odp-gen-config <name>`。
- CLI 显式传入的 `False` / `0` / 空字符串要保留，只有 `None` 表示“用户没传”。
- `task` 取值复用 `od_platform.common.constants.Task`，保持 SSoT。
- 框架内部字段可以进入配置对象，但不能原样传给 Ultralytics。
- 生成器默认不覆盖已有文件；使用 `--overwrite` 时默认先备份。

## 后果

正向影响：

- D6 调用方可以通过 `build_train_config()` / `build_val_config()` / `build_infer_config()` 一行拿到配置对象。
- 每个字段都能解释“当前值来自哪里、覆盖了谁”。
- YAML 模板由配置模型反射生成，减少手写模板漂移。

代价：

- 配置模型字段较多，需要通过测试保护字段契约。
- 终端和模板输出必须统一 UTF-8，否则中文注释和覆盖链会影响验收观感。

## 验收

- `odp-gen-config train/val/infer` 能生成模板。
- 重复生成默认跳过，`--overwrite` 会覆盖并备份。
- `build_train_config("train.yaml")` 能加载模板并返回配置对象。
- `ConfigMerger` 能输出类似 `batch: 4(CLI) ← 8(YAML) ← 16(DEFAULT)` 的覆盖链。
- `to_ultralytics_kwargs()` 不包含 `verbose`、`experiment_name` 等平台内部字段。
