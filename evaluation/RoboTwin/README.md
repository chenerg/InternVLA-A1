# Evaluation on RoboTwin 2.0 Benchmark

## Table of Contents

- [Installation](#installation)
- [Running Evaluation](#running-evaluation)
- [Viewing Results](#viewing-results)

---

## Installation

### Step 1: Install Basic Environment

Install Vulkan drivers for graphics rendering:

```bash
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools
```

### Step 2: Clone RoboTwin Repository

**RoboTwin 2.0 Repository:** [https://github.com/RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin)

```bash
cd InternVLA-A1
git submodule update --init third_party/RoboTwin
```

### Step 3: Install Dependencies

Replace the requirements.txt and install dependencies:

```bash
cp -r evaluation/RoboTwin/requirements.txt third_party/RoboTwin/script/requirements.txt
cd third_party/RoboTwin
bash script/_install.sh
bash script/_download_assets.sh
cd ../../
```


> **Note:** For more details, please refer to the [official RoboTwin installation tutorial](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).



## Running Evaluation

```bash
bash evaluation/RoboTwin/eval.sh
```

By default, using the finetuned InternVLA-A1-3B from [Huggingface](https://huggingface.co/InternRobotics/InternVLA-A1-3B-RoboTwin):


> You can modify the following variables in `evaluation/RoboTwin/eval.sh`:

- **`PRETRAINED_CKPT`**: The model checkpoint to load.
- **`TASK_CONFIG`**: The RoboTwin evaluation settting "demo_clean/demo_randomized"
  - Example: `demo_clean`
- **`TASK_IDX`**: The task index to evaluate (index into the `TASK_NAMES` list inside `inference.py`).
  - Example: `0  # adjust_bottle` (the comment is only a human hint for the corresponding task name).
- **`BASE_OUTPUT_PATH`**: The root directory for evaluation outputs.
  - The final output directory is `${BASE_OUTPUT_PATH}/${TASK_CONFIG}/${TASK_IDX}` (the script variable `OUTPUT_PATH`).

---

## Client/Server Split Inference

You can split environment execution (client) and model inference (server) into two processes.

### 1) Start the inference server

```bash
python evaluation/RoboTwin/inference_server.py \
  --ckpt_path InternRobotics/InternVLA-A1-3B-RoboTwin \
  --stats_key aloha \
  --resize_size 224 \
  --dtype float32 \
  --host 0.0.0.0 \
  --port 8000
```

### 2) Start the inference client (RoboTwin env runner)

```bash
python evaluation/RoboTwin/inference_client.py \
  --server_url http://127.0.0.1:8000/infer \
  --task_config demo_clean \
  --task_idx 0 \
  --video_dir evaluation/RoboTwin/output/demo_clean/0
```

> Notes:
> - The server loads the model and handles input transforms and action post-processing.
> - The client handles RoboTwin environment stepping, observation collection, and video writing.




## Viewing Results

> Videos are saved to `OUTPUT_PATH` and named `<success|failure>_<test_num>.mp4`.
