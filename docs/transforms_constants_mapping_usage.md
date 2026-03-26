# `src/lerobot/transforms/constants.py` 中各类 `MAPPING` 的作用与使用位置

本文档说明 `src/lerobot/transforms/constants.py` 里三类核心映射：

- `MASK_MAPPING`
- `FEATURE_MAPPING`
- `IMAGE_MAPPING`

并回答两个问题：
1. 它们分别解决什么问题？
2. 在代码中具体在哪里被消费（used）？

---

## 1. 先看背景：为什么需要这些 MAPPING

InternVLA-A1 需要兼容多种机器人/数据集（如 `a2d`、`genie1`、`franka`、`ARX Lift-2` 等），而不同来源的数据字段命名并不统一，例如：

- 状态可能叫 `observation.state`，也可能拆成 `states.left_joint.position` + `states.right_joint.position`。
- 动作也可能是单字段 `action`，或多个子字段 `actions.*`。
- 图像键名在不同数据集差异更大（`observation.images.*`、`images.rgb.*` 等）。

因此，该文件本质上是“**机器人类型 -> 标准化规则**”的字典集合，用于把异构字段统一到训练/推理流水线期望的规范格式。  

---

## 2. `MASK_MAPPING`：控制哪些动作维度做 delta（相对动作）计算

## 2.1 定义与含义

`MASK_MAPPING` 为每种 `robot_type` 提供一个布尔 mask（通过 `make_bool_mask(...)` 构造）。该 mask 用在 delta action 计算时，决定某个动作维度是否减去对应状态维度（即 `action - state`）。

简化理解：

- `True` 位置：做相对化（减状态）
- `False` 位置：保持原值（常见于 gripper 等维度）

## 2.2 使用位置

### A) 训练/数据变换链中的 `DeltaActionTransformFn`

在 `hydrate_delta_action_transform` 中，框架根据 `dataset.meta.robot_type` 注入：

- `mapping=FEATURE_MAPPING[robot_type]`
- `mask=MASK_MAPPING[robot_type]`

之后 `DeltaActionTransformFn.__call__` 内部执行：

```python
action -= torch.where(mask, state, 0)[None]
```

即真正用到 mask 来控制逐维 delta 计算。  

### B) 统计脚本 `lerobot_data_stats.py`

该脚本计算 `delta` 模式归一化统计时，同样读取 `mask = MASK_MAPPING[robot_type]`，并在 `delta_action` 计算时应用：

```python
delta_action = action_chunk - torch.where(mask, truncated_state, 0)[:, None]
```

因此，`MASK_MAPPING` 同时影响：

- 训练时输入动作的构造
- 预先计算的 `stats.json`（如果用 `delta`）

二者一致性很关键。  

---

## 3. `FEATURE_MAPPING`：将“多字段状态/动作”组合为统一键

## 3.1 定义与含义

`FEATURE_MAPPING` 是 `defaultdict`，核心结构如下：

- `OBS_STATE -> [若干源状态字段]`
- `ACTION -> [若干源动作字段]`

默认情况是：

- `OBS_STATE: ["observation.state"]`
- `ACTION: ["action"]`

对特殊机器人则覆写为多个子键（如 `states.left_joint.position` 等）。

## 3.2 使用位置

### A) `ComposeFieldsTransform` 的参数注入

`hydrate_compose_field_transform` 会把 `FEATURE_MAPPING[robot_type]` 注入 `ComposeFieldsTransform`。该 transform 会：

1. 从 `mapping` 指定的多个源键取值
2. 在最后一维拼接
3. 写回标准键（`observation.state` / `action`）
4. 删除旧键

这一步完成了**字段统一**。

### B) `DeltaActionTransformFn` 的状态/动作键来源

`DeltaActionTransformFn.__call__` 里直接按 `mapping[OBS_STATE]` 和 `mapping[ACTION]` 取字段拼接，再执行 delta 逻辑，因此 `FEATURE_MAPPING` 决定了“哪些原始键参与动作-状态对齐”。

### C) `NormalizeTransformFn` 的选键

`hydrate_normalize_transform` 使用：

```python
selected_keys = EMBODIMENT_SPEC[OBS_STATE] + EMBODIMENT_SPEC[ACTION]
```

这意味着归一化不是盲目全字段，而是优先按 `FEATURE_MAPPING` 声明的状态/动作键执行。

### D) 数据统计脚本中的键选择

`src/lerobot/scripts/lerobot_data_stats.py` 用 `FEATURE_MAPPING[robot_type]` 决定：

- 哪些 action 键在 `delta` 下需要特殊处理
- 哪些 state/action 键参与拼接计算

### E) 数据集时间窗口（delta_timestamps）解析

`src/lerobot/datasets/factory.py` 的 `resolve_delta_timestamps` 中：

```python
elif key in FEATURE_MAPPING[ds_meta.robot_type][ACTION] and cfg.action_delta_indices is not None:
```

说明除标准 `action` 外，映射中声明的子动作键也会得到动作时间偏移配置，保证多字段动作数据在采样时序上行为一致。

---

## 4. `IMAGE_MAPPING`：将异构图像键统一到 `observation.images.image{0,1,2}`

## 4.1 定义与含义

`IMAGE_MAPPING` 也是 `defaultdict`，含义是：

- 输入数据中的图像键（例如 `images.rgb.head`）
- 映射到统一键（例如 `observation.images.image0`）

这样模型输入侧不需要关心每个数据源的原始命名差异。

## 4.2 使用位置

### A) `RemapImageKeyTransformFn` 的参数注入

`hydrate_remap_image_key_transform` 将 `IMAGE_MAPPING[robot_type]` 注入 `RemapImageKeyTransformFn`。该 transform 会：

1. 将旧图像键重命名为统一键
2. 为每路图像补齐 `<key>_mask` 与 `<key>_is_pad`
3. 如果缺某路相机（如只有 1~2 路），自动创建占位图并标记缺失

这一步保障视觉输入接口一致。

### B) `filter_image_features`

`filter_image_features` 根据 `IMAGE_MAPPING[robot_type]` 过滤 `dataset.meta.video_keys`：

- 不在映射表中的视频键会被从 `dataset.meta.features` 删除

用于防止无关视频特征进入后续流程。

### C) 数据集时间窗口（图像 delta）解析

在 `resolve_delta_timestamps` 中还有：

```python
if key in IMAGE_MAPPING[ds_meta.robot_type].keys() and ...:
```

即映射表里声明的图像键会应用 `image_delta_indices`，用于时序图像采样。

---

## 5. 三类 MAPPING 的协同关系（一个最小流程）

给定 `robot_type` 后，流水线大致按以下逻辑运作：

1. `FEATURE_MAPPING`：定义状态/动作由哪些原始键组成。
2. `IMAGE_MAPPING`：定义图像键如何改名到统一空间。
3. `MASK_MAPPING`：定义动作哪些维度按 delta 规则减状态。

在 transform hydrate 阶段，三类映射被自动注入具体 transform 实例，然后在样本级执行时生效。这样就把多机器人、多数据格式转成同一训练接口。

---

## 6. 维护建议

1. **新增机器人类型时**：三张表尽量同时补齐（`FEATURE/MASK/IMAGE`），避免训练与统计不一致。
2. **字段改名时**：同时检查 `transforms/core.py` 与 `scripts/lerobot_data_stats.py` 的映射消费点。
3. **delta 训练异常时**：优先核对 `MASK_MAPPING` 长度、`FEATURE_MAPPING` 拼接后维度是否与 action/state 对齐。

---

## 7. 快速索引（代码位置）

- 映射定义：`src/lerobot/transforms/constants.py`
- transform 注入与执行：`src/lerobot/transforms/core.py`
- 统计脚本消费：`src/lerobot/scripts/lerobot_data_stats.py`
- 数据集时间窗口消费：`src/lerobot/datasets/factory.py`
