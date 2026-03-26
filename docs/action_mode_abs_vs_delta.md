# `abs` 与 `delta` 动作模式详解（训练 + 推理）

本文基于 InternVLA-A1 当前代码实现，系统说明 `dataset.action_mode` 选择 `abs` 或 `delta` 时，
数据预处理、归一化统计、模型学习目标与推理执行链路会发生什么变化。

---

## 1. 一句话结论

- `abs`：模型直接学习**绝对动作**（例如关节目标位姿/夹爪绝对开合值）。
- `delta`：模型学习**相对动作**，即“动作 - 当前状态”的残差（按机器人类型的 mask 逐维决定是否做差分）。

在本仓库中，是否启用 `delta` 的核心开关是 `dataset.action_mode`，默认值为 `abs`。  
当设为 `delta` 时，会自动向数据变换链路注入 `DeltaActionTransformFn`；设为 `abs` 时则移除该变换。  

---

## 2. 配置入口与开关传播

### 2.1 统一配置入口

`DatasetConfig` 定义了 `action_mode: str = "abs"`，并限制取值只能是 `abs|delta`。  
这意味着所有训练脚本最终都通过这个字段控制动作表征。  

### 2.2 launch 脚本如何传参

例如 `launch/internvla_a1_3b_finetune.sh`：

- 第二个参数 `ACTION_TYPE=${2:-abs}`（默认 abs）。
- 最终透传为 `--dataset.action_mode="${ACTION_TYPE}"`。
- 外部统计路径也随模式切换为 `${HF_HOME}/lerobot/stats/${ACTION_TYPE}/${DATASET_REPO_ID}/stats.json`。

因此在同一数据集上切换 abs/delta，本质上会切换：
1) 数据预处理逻辑（是否减状态）；2) 统计文件来源；3) 模型要拟合的目标分布。

---

## 3. 训练时的数据处理差异

> 关键点：模型本身不“知道”你用 abs 还是 delta；真正改变的是**喂给模型的 action 标签值**以及对应统计。

### 3.1 变换链路中如何插入/移除 delta 变换

在 `QwenA1DatasetConfig.__post_init__` 中：

- 若 `action_mode == "delta"` 且链路里没有 `DeltaActionTransformFn`，则插入。
- 若 `action_mode == "abs"` 且链路里已有 `DeltaActionTransformFn`，则删除。

这保证了同一 policy 配置在不同 action mode 下会自动重写输入 transform pipeline。

### 3.2 DeltaActionTransformFn 的具体计算

`DeltaActionTransformFn` 的核心步骤：

1. 按机器人类型映射（`FEATURE_MAPPING`）取出状态键和动作键并拼接。
2. 取 `MASK_MAPPING[robot_type]` 决定哪些维度做差分。
3. 执行：
   \[
   action_{delta} = action_{abs} - mask \odot state
   \]
   对 mask=False 的维度，相当于减 0（保持原值）。
4. 再按原 action 键拆分回写。

> 这使得不同机器人（单臂/双臂、夹爪维度等）可以共享“delta 机制”，但逐维行为由 mask 精细控制。

### 3.3 transform 的“水合（hydrate）”机制

`TransformedLeRobotDataset.from_base(...)` 会在真正迭代样本前自动：

- 用数据集 `robot_type` 给 `DeltaActionTransformFn` 注入 `mapping + mask`。
- 给 `NormalizeTransformFn` 注入 `dataset.meta.stats` 和选定键。

所以 `delta` 并非只靠命令行开关，而是会在 runtime 被“绑定”到对应机器人语义。

### 3.4 对训练目标分布的影响

- `abs`：action 目标分布更依赖数据集本身绝对坐标系/姿态范围。
- `delta`：目标更接近“控制增量/纠偏量”，通常均值更接近 0、动态范围更集中。

这会直接影响 loss 数值尺度、收敛速度和跨 embodiment 泛化难度（尤其是多机器人混合训练时）。

---

## 4. 归一化统计（stats）为何必须与模式匹配

### 4.1 训练加载外部 stats 的路径依赖 action_mode

`make_dataset` 在 `use_external_stats=true` 时，会按 `stats/{robot_type}/{action_mode}/stats.json` 选择统计文件。
如果 launch 显式给了 `external_stats_path`，通常也会把 `${ACTION_TYPE}` 拼到路径里。

### 4.2 统计脚本如何区分 abs/delta

`src/lerobot/scripts/lerobot_data_stats.py` 的逻辑：

- `--action_mode abs`：直接统计原始 action 键。
- `--action_mode delta`：
  - 构造 `action_chunk`（长度 = `chunk_size`），
  - 与截断后的 state 做按维差分（同样使用 `mask`），
  - 对差分后的动作序列统计 mean/std/min/max。

因此：

- **abs 训练 + delta stats**（或反之）会导致归一化失配。
- 失配常表现为训练不稳定、loss 异常大、推理动作尺度失真。

---

## 5. 推理阶段的影响（最容易误解的部分）

## 5.1 模型输出语义取决于训练时标签语义

`select_action -> predict_action_chunk` 只是输出模型预测动作块并执行 action queue；
它本身不会自动把 delta 再“加回状态”。

换句话说：

- 用 `abs` 训练的模型，输出就是绝对动作语义。
- 用 `delta` 训练的模型，输出就是相对动作语义（除非你在后处理/控制器中自行还原到绝对命令）。

## 5.2 对部署控制器的直接要求

如果机器人控制接口需要绝对目标，而模型输出是 delta，则部署侧必须做：
\[
action_{abs}=action_{delta}+mask\odot state
\]
并保持与训练同一 `FEATURE_MAPPING/MASK_MAPPING` 对齐。

若控制器天然接收增量命令（例如速度/位移增量），则可直接使用 delta 输出，
但仍需确认与环境 action space 的单位、夹爪通道语义一致。

## 5.3 action chunking 与模式关系

本仓库策略使用 chunk 预测（`chunk_size`）并缓存 `n_action_steps`：

- 该机制与 abs/delta **正交**（都适用）。
- 但在 delta 模式下，chunk 内每一步都代表“相对当前状态定义的残差目标”。

在长时执行中，如果不做反馈闭环（或执行器有延迟），delta 的累计误差特性会和 abs 不同，
这是部署时需重点评估的系统层问题。

---

## 6. 何时选 abs，何时选 delta（工程建议）

### 更适合 `abs` 的场景

- 数据动作标签本来就是稳定绝对目标（且坐标定义统一）。
- 控制器直接消费绝对位姿/关节目标，后处理链路希望最简。
- 任务对“到达某个全局位姿”更敏感。

### 更适合 `delta` 的场景

- 不同数据源的绝对参考系差异较大，想弱化坐标系偏置。
- 希望模型学习“纠偏量/微调量”，提升局部闭环修正能力。
- 多 embodiment 训练中，希望把动作学习尽量对齐到“状态相对变化”。

---

## 7. 常见错误清单（排障优先级）

1. **训练模式与 stats 模式不一致**（最高频）。
2. **部署端把 delta 当 abs 直接下发**（导致动作幅度/方向异常）。
3. **mask 与 mapping 不匹配**（维度错位，尤其是双臂 + 夹爪）。
4. 切换 `action_mode` 后未重新核对旧 checkpoint 的语义兼容性。

---

## 8. 最小可执行核对步骤

1. 启动命令里确认 `--dataset.action_mode`。
2. 核对 stats 路径是否包含同样的 mode（abs/delta）。
3. 打印一批训练样本 action（变换后）确认数值语义。
4. 推理端对照控制器接口，确认是否需要 delta->abs 还原。

---

## 9. 关键代码索引

- action mode 定义与校验：`src/lerobot/configs/default.py`
- mode 控制 transform 注入：
  - `src/lerobot/policies/InternVLA_A1_3B/configuration_internvla_a1.py`
  - 同类逻辑也存在于 2B/pi0/pi05/f1 配置
- delta 逐维计算：`src/lerobot/transforms/core.py`（`DeltaActionTransformFn`）
- mask/mapping 机器人定义：`src/lerobot/transforms/constants.py`
- transform 水合入口：`src/lerobot/datasets/transformed_dataset.py`
- 外部 stats 加载路径：`src/lerobot/datasets/factory.py`
- stats 计算（abs/delta 分支）：`src/lerobot/scripts/lerobot_data_stats.py`
- 推理取动作接口：`src/lerobot/policies/InternVLA_A1_3B/modeling_internvla_a1.py`

---

## 10. 总结

`abs` 与 `delta` 的差别，不只是“标签换个形式”，而是影响了：

- 数据标签构造方式；
- 统计归一化分布；
- 模型输出语义；
- 部署控制器后处理责任边界。

最稳妥的实践是把它当成“端到端协议”：**训练、统计、推理后处理必须三者一致**。
