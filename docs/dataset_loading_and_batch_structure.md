# InternVLA-A1 数据读取与 Batch 结构说明

本文档说明 InternVLA-A1 在训练阶段如何读取数据，以及经过 DataLoader 后 `batch` 的结构（重点覆盖默认 `qwena1` 配置）。

## 1. 总体链路

训练入口在：
- `src/lerobot/scripts/lerobot_train.py`

核心数据链路：
1. `train()` 调用 `make_dataset(cfg)` 构建 dataset。
2. `make_dataset` 根据配置选择：
   - 非流式：`LeRobotDataset` + `TransformedLeRobotDataset`
   - 流式：`StreamingLeRobotDataset` + `TransformedStreamingLeRobotDataset`
3. 训练脚本使用 `torch.utils.data.DataLoader(dataset, ...)`。
4. DataLoader 未传入自定义 `collate_fn`，使用 PyTorch 默认 collate，将“样本字典列表”拼成“批字典”。

关键代码：
- `src/lerobot/scripts/lerobot_train.py:267`
- `src/lerobot/datasets/factory.py:404`

---

## 2. Dataset 构建阶段（`make_dataset`）

位置：`src/lerobot/datasets/factory.py`

### 2.1 repo 分配与多数据集

`make_dataset` 会先处理 `cfg.dataset.repo_id`（可用空格分隔多个 repo）。
- 单 repo：直接返回一个 transformed dataset。
- 多 repo：构建多个 transformed dataset，再包装成：
  - 非流式：`MultiLeRobotDataset`
  - 流式：`MultiStreamingLeRobotDataset`

如果启用 `dist_loading`，会根据 rank/world size 做 repo 分配（优先按 frame 数平衡）。

### 2.2 delta_timestamps 的来源

`resolve_delta_timestamps(cfg.policy, ds_meta)` 会把 policy 里的 delta 索引转成时间戳（按 fps 换算）。
- `observation_delta_indices`
- `action_delta_indices`
- `reward_delta_indices`
- 对 `InternVLA_A1` 还支持 `image_delta_indices`

以 `qwena1` 为例（`src/lerobot/policies/InternVLA_A1_3B/configuration_internvla_a1.py`）：
- `action_delta_indices = [0,1,...,49]`
- `image_delta_indices = [-15, 0, 15]`
- `observation_delta_indices = None`

这决定了后续单样本中 action / image 是单帧还是多帧时序堆叠。

### 2.3 transform 挂载

`TransformedLeRobotDataset.from_base(...)` 对底层 dataset 做 transform 组合：
- 自动 hydrate normalize / compose_fields / delta_action / remap_image_key
- 然后在 `__getitem__` 中执行 `self._transform(sample)`

位置：
- `src/lerobot/datasets/transformed_dataset.py:30`
- `src/lerobot/datasets/transformed_dataset.py:84`

---

## 3. 非流式数据读取细节（`LeRobotDataset`）

位置：`src/lerobot/datasets/lerobot_dataset.py`

### 3.1 parquet 到 tensor

`load_hf_dataset()` 从 `data/*/*.parquet` 读取为 HF Dataset，并设置 transform：
- `hf_transform_to_torch` 会把基础字段转为 `torch.Tensor`
- 图片（PIL）会转成 `(C,H,W)` 且值域 `[0,1]`

位置：
- `src/lerobot/datasets/lerobot_dataset.py:849`
- `src/lerobot/datasets/utils.py:411`

### 3.2 `__getitem__` 做了什么

`LeRobotDataset.__getitem__(idx)` 主要步骤：
1. 取当前 frame：`item = self.hf_dataset[idx]`
2. 若启用 delta：
   - 计算 query indices（超出 episode 边界会夹到边界）
   - 查询非视频字段并 `torch.stack`
   - 生成 `*_is_pad` 布尔 mask
3. 若有视频键：
   - 根据 timestamp 解码 mp4 对应帧
4. 应用 `image_transforms`（如果配置启用）
5. 额外附加：
   - `task`（字符串）
   - `robot_type`（字符串）

位置：`src/lerobot/datasets/lerobot_dataset.py:1025`

### 3.3 单样本返回类型

在进入 policy 专用 transform 前，单样本是一个 Python 字典，通常包含：
- 多个 `torch.Tensor`（状态、动作、图像、mask、索引等）
- 少量字符串（如 `task`, `robot_type`）

注意：后续 `UnifyQwenA1InputsTransformFn` 会丢弃很多非模型输入键，仅保留模型所需字段。

---

## 4. 流式读取细节（`StreamingLeRobotDataset`）

位置：`src/lerobot/datasets/streaming_dataset.py`

`StreamingLeRobotDataset` 是 `IterableDataset`，不是随机索引数据集：
- `__iter__` 中对 shard 做随机混合与 buffer 打散
- `make_frame()` 内执行 delta 帧拼接、视频解码、padding mask 生成
- 返回单条 frame 字典后再走 transformed wrapper

关键点：
- 非视频字段 delta 通过 backtrack/peek 机制取历史/未来帧
- 视频字段按 query timestamp 解码
- 仍会生成 `task` 字段

位置：
- `src/lerobot/datasets/streaming_dataset.py:190`
- `src/lerobot/datasets/streaming_dataset.py:322`

---

## 5. DataLoader 拼 batch 规则

训练中 DataLoader 创建如下（简化）：
```python
DataLoader(
    dataset,
    batch_size=cfg.batch_size,
    num_workers=...,
    shuffle=...,
    sampler=...,
)
```

未提供 `collate_fn`，因此使用默认 collate：
- 输入：`List[Dict[str, ...]]`
- 输出：`Dict[str, Tensor/其他]`
- 同名键会在 batch 维拼接（stack）

位置：`src/lerobot/scripts/lerobot_train.py:267`

---

## 6. 默认 `qwena1` 下的 batch 结构（重点）

默认 finetune 脚本：
- `launch/internvla_a1_3b_finetune.sh`
- `--policy.type=qwena1`
- `--dataset.type=qwena1`

对应数据 transform（`QwenA1DatasetConfig.data_transforms.inputs`）包括：
1. `DeltaActionTransformFn`
2. `ResizeImagesWithPadFn`
3. `RemapImageKeyTransformFn`
4. `Qwen3_VLProcessorTransformFn`
5. `NormalizeTransformFn`
6. `ComposeFieldsTransform`
7. `PadStateAndActionTransformFn`
8. `UnifyQwenA1InputsTransformFn`

位置：
- `src/lerobot/policies/InternVLA_A1_3B/configuration_internvla_a1.py:38`

### 6.1 最终保留键（由 `UnifyQwenA1InputsTransformFn` 决定）

最终单样本只保留这些键：
- `observation.state`
- `action`
- `observation.images.image0`
- `observation.images.image1`
- `observation.images.image2`
- `observation.images.image0_mask`
- `observation.images.image1_mask`
- `observation.images.image2_mask`
- `observation.pixel_values`
- `observation.image_grid_thw`
- `observation.input_ids`
- `observation.attention_mask`

位置：`src/lerobot/policies/InternVLA_A1_3B/transform_internvla_a1.py:83`

### 6.2 batch 后常见 shape（`batch_size=B`）

以下是默认配置下的常见形状（具体数值可能受数据集特征维度和 processor 输出影响）：

- `observation.state`: `torch.Tensor`，约 `[B, 32]`
  - 由 `PadStateAndActionTransformFn` 补齐到 `max_state_dim=32`

- `action`: `torch.Tensor`，约 `[B, 50, 32]`
  - `action_delta_indices=range(50)` 带来时间维 50
  - 再补齐到 `max_action_dim=32`

- `observation.images.image{0,1,2}`: `torch.Tensor`，约 `[B, 3, 3, 224, 224]`
  - 维度解释：`B, T, C, H, W`
  - `T=3` 来自 `image_delta_indices=[-15,0,15]`

- `observation.images.image{0,1,2}_mask`: `torch.BoolTensor`，通常 `[B]`

- `observation.pixel_values`: `torch.Tensor`
  - 来自 Qwen3-VL image processor，维度与视觉 token 化结果相关

- `observation.image_grid_thw`: `torch.Tensor`
  - 视觉网格信息（T,H,W）

- `observation.input_ids`: `torch.Tensor`
  - 图像 token + 文本 token 拼接后的序列

- `observation.attention_mask`: `torch.Tensor`
  - 与 `input_ids` 对齐

### 6.3 policy 如何消费这个 batch

`QwenA1Policy.forward(batch)` 直接按上述 key 取值：
- `observation.pixel_values`
- `observation.image_grid_thw`
- `observation.input_ids`
- `observation.attention_mask`
- `observation.state`
- `action`
- `observation.images.image{0,1,2}` 与 mask

位置：`src/lerobot/policies/InternVLA_A1_3B/modeling_internvla_a1.py:1087`

---

## 7. 离线 vs 流式 的结构差异

相同点：
- 最终进入 policy 的 batch key 取决于 transform（尤其 `UnifyQwenA1InputsTransformFn`）。

不同点：
- 非流式：`Dataset` + 随机索引 `__getitem__`
- 流式：`IterableDataset` + `__iter__` 顺流生成 + shard/buffer 混洗
- 流式模式下，训练脚本会把 `num_workers` 设为 1（当前实现）

---

## 8. 常见排查建议

1. 如果想看“原始样本键”与“transform 后键”差异：
   - 在 `TransformedLeRobotDataset.__getitem__` 前后打印 `sample.keys()`。

2. 如果 batch shape 异常：
   - 检查 `policy` 的 `*_delta_indices`
   - 检查 `ComposeFieldsTransform` / `RemapImageKeyTransformFn`
   - 检查数据集 `meta/info.json` 中 feature 定义

3. 如果多数据集混训报 collate 相关错误：
   - 优先核对各数据集 transform 后 key 是否完全一致（同名键 shape/dtype 也要一致）

---

## 9. 一句话结论

在当前默认 `qwena1` 训练配置里，DataLoader 输出是：
- **类型**：`dict[str, torch.Tensor]`（以 tensor 为主）
- **语义**：一个融合了状态、动作时序、三路图像时序、视觉 token、文本 token 的统一 batch 字典。
