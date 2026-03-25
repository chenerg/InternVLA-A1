# AGENTS.md

本文件定义了代码代理（包括 Codex）在本仓库中的工作方式。

## 适用范围
- 适用于当前目录为根的整个仓库。
- 如果后续某个子目录新增自己的 `AGENTS.md`，则该子树以内以更深层文件为准。

## 项目概览
- 项目：InternVLA-A1（基于 LeRobot 的 VLA 训练与评测代码库）。
- 主要语言：Python（>=3.10）。
- 核心包路径：`src/lerobot`。
- 主要流程：预训练、微调、数据集转换/聚合、RoboTwin 评测。

## 仓库结构
- `src/lerobot/`：核心库代码。
- `src/lerobot/scripts/`：主要入口脚本，如 `lerobot_train.py`。
- `src/lerobot/policies/`：策略/模型定义（InternVLA_A1_3B、InternVLA_A1_2B、pi0、pi05）。
- `src/lerobot/datasets/`：数据集加载、变换、统计与转换辅助。
- `launch/`：常用训练启动脚本。
- `tutorials/`：安装与任务教程。
- `evaluation/RoboTwin/`：RoboTwin 基准集成与评测脚本。
- `util_scripts/`：数据与通用工具脚本。
- `tests/`：测试与基于 notebook 的评测产物。
- `third_party/`：外部依赖（含 RoboTwin 子模块）。

## 环境基线
- 仓库文档推荐基线：
- Python 3.10
- CUDA 12.8
- PyTorch 2.7.1
- 常见环境名为 `conda` 环境 `internvla_a1`。
- 依赖安装流程见 `tutorials/installation.md`。

## 必要初始化流程（等价 /init）
开始新的 Codex 任务时，在修改代码前先执行以下流程：
1. 阅读 `README.md`，并查看与任务相关的 `tutorials/` 文档。
2. 确认当前分支与工作区状态。
3. 明确目标模块（`src/lerobot/`）及关联的 launch/eval 脚本。
4. 列出假设与风险（分布式训练参数、离线模式、数据路径、checkpoint 来源）。
5. 之后再进行最小化、范围可控的修改。

## 常用命令
- 可编辑安装：
- `pip install -e .`
- 快速微调启动：
- `bash launch/internvla_a1_3b_finetune.sh lerobot/pusht abs false`
- 预训练启动：
- `bash launch/internvla_a1_3b_pretrain.sh`
- RoboTwin 评测：
- `bash evaluation/RoboTwin/eval.sh`
- 运行测试（如适用）：
- `pytest -q`

## 代理工作规则
- 修改应最小化并聚焦任务；非必要不要做大规模重构。
- 不修改 `data/`、`outputs/` 中的生成产物、模型权重或大文件。
- 优先修改 Python 源码与启动脚本，不要直接改已安装的 site-packages。
- 涉及分布式训练参数的改动需保持与现有 launch 脚本兼容。
- 遵循现有 CLI/配置风格（draccus/参数覆盖）与命名模式。
- 除非任务明确要求联网行为，否则保留 launch 脚本中的离线开关。

## Transformers 替换注意事项
- 本仓库依赖若干策略对应的自定义 `transformers_replace/models` 覆盖。
- 若功能依赖这些补丁过的 HF 模块，需在 PR/提交说明中明确写出。
- 不要默认上游 `transformers` 行为与本仓库当前行为一致。

## 验证要求
- Python 逻辑改动：运行针对性测试，至少做 import/CLI 冒烟检查。
- 训练/评测脚本改动：执行 `bash -n <script>` 并验证参数拼装。
- 数据流程改动：用小样本验证，并说明输入/输出 shape 的预期变化。
- 若受 GPU/数据/时间限制无法完整验证，必须明确说明未执行项。

## Git 规范
- 在功能分支上工作（本仓库默认使用 `codex` 分支进行代理改动）。
- 不回滚与当前任务无关的本地修改。
- 未经明确要求，不使用破坏性 git 命令。
- 提交信息应描述“行为变化”，而不只是“改了哪些文件”。

## 代理报告输出格式
每次任务完成后，报告应包含：
1. 改了什么。
2. 为什么改。
3. 做了哪些验证。
4. 剩余风险或后续建议。

## 需要人工确认的场景
出现以下情况时，先询问用户再继续：
- 修改模型架构或 checkpoint 兼容性。
- 调整数据集格式/版本边界（v2.1 与 v3.0 转换逻辑）。
- 修改评测协议或基准任务定义。
- 引入新的重依赖。
