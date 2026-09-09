# EgoCHARM PyTorch Style Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将整个 EgoCHARM Python 项目重构为接近 `pretrained-imu-encoders` 的直接 PyTorch 风格，并为完整数据流补充统一的张量维度注释，同时保持现有行为不变。

**Architecture:** 保留现有 CLI、YAML 和数据格式，将纯训练循环、checkpoint 和指标写入拆成独立模块；`train.py` 与 `evaluate.py` 只负责编排并重新导出旧接口。模型使用直接的 `nn.ModuleList`/`nn.Sequential` 表达，数据、模型、训练循环中的形状统一用 `B/N/S/C/T/F/E/H/K` 标注。

**Tech Stack:** Python 3.10、PyTorch 2.x、NumPy、SciPy、scikit-learn、Matplotlib、PyYAML、pytest。

## Global Constraints

- 保持 `python scripts/preprocess_data.py`、`python -m egocharm.train` 和 `python -m egocharm.evaluate` 的用法不变。
- 保持 `configs/egocharm.yaml` 的字段和含义不变。
- 保持官方 train/val/test 划分、窗口策略、模型结构、层执行顺序和参数量不变。
- Dataset 继续返回未缩放的原始 IMU 数值；不得恢复均值/标准差输入归一化。
- 模型内部的 `BatchNorm1d` 保留。
- 不打开真实 VRS，不运行真实预处理，不启动长时间训练。
- 不修改 `ego-exoD4_dataset` 中的任何数据。
- 当前工作目录不是 Git 仓库；不得为了本计划自行执行 `git init`，计划中的阶段性验证代替 commit 检查点。
- Python 命令使用 `/home/xreal/miniconda3/envs/egocharm/bin/python`。

---

## File Map

**Create**

- `src/egocharm/engine.py`：设备选择、类别权重、epoch 指标、训练和验证循环。
- `src/egocharm/checkpoint.py`：训练状态、随机数状态、checkpoint 原子保存与恢复。
- `src/egocharm/metrics.py`：分类指标、JSON/CSV 与混淆矩阵输出。

**Modify**

- `src/egocharm/model.py`：参考仓库风格的模型定义和完整维度注释。
- `src/egocharm/dataset.py`：直接的数据管线表达和维度注释。
- `src/egocharm/config.py`：减少无价值类型噪声，保持配置校验。
- `src/egocharm/train.py`：薄训练入口，并重新导出旧接口。
- `src/egocharm/evaluate.py`：薄评估入口，并重新导出旧接口。
- `scripts/preprocess_data.py`：整理研究代码风格并标注 `(N,)`、`(N, 6)` 等数组形状。
- `tests/test_*.py`：去掉测试函数返回标注，补充模块边界和形状回归测试。
- `README.md`：更新模块结构和张量数据流。
- `pyproject.toml`：只在需要时补充现有 pytest 配置；不新增运行时依赖。

---

### Task 1: 锁定重构前行为与维度契约

**Files:**

- Modify: `tests/test_model.py`
- Modify: `tests/test_dataset.py`
- Test: `tests/test_model.py`
- Test: `tests/test_dataset.py`

**Interfaces:**

- Consumes: `LowLevelEncoder`, `EgoCHARM`, `EgoCharmDataset` 当前接口。
- Produces: 明确的 LLE `(B, 6, 50) -> (B, 32)` 和分层编码 `(B, S, 6, 50) -> (B, S, 32)` 回归契约。

- [ ] **Step 1: 运行当前完整测试作为基线**

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/egocharm-matplotlib \
  /home/xreal/miniconda3/envs/egocharm/bin/python -m pytest -p no:cacheprovider -q
```

Expected: `26 passed`，无失败和错误。

- [ ] **Step 2: 添加低层编码器输出形状测试**

Add to `tests/test_model.py`:

```python
def test_low_level_encoder_returns_one_embedding_per_window():
    encoder = LowLevelEncoder()
    x = torch.randn(4, 6, 50)  # (B, C, T)

    embeddings = encoder(x)

    assert embeddings.shape == (4, 32)  # (B, E)
```

- [ ] **Step 3: 添加30秒分层编码形状测试**

Add to `tests/test_model.py`:

```python
def test_encode_each_second_preserves_batch_and_seconds():
    model = EgoCHARM(
        low_level_encoder=LowLevelEncoder(),
        high_level_classifier=HighLevelClassifier(num_classes=7),
    )
    x = torch.randn(2, 30, 6, 50)  # (B, S, C, T)

    embeddings = model.encode_each_second(x)

    assert embeddings.shape == (2, 30, 32)  # (B, S, E)
```

- [ ] **Step 4: 运行形状测试**

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 /home/xreal/miniconda3/envs/egocharm/bin/python \
  -m pytest -p no:cacheprovider -q tests/test_model.py tests/test_dataset.py
```

Expected: 新旧测试全部通过。这是行为刻画，不预期 RED。

---

### Task 2: 重构模型为直接 PyTorch 风格

**Files:**

- Modify: `src/egocharm/model.py`
- Modify: `tests/test_model.py`
- Test: `tests/test_model.py`

**Interfaces:**

- Consumes: YAML `model` 配置中的现有字段。
- Produces: `SamePaddingConv1d`, `ParallelDilatedBlock`, `LowLevelEncoder`, `HighLevelClassifier`, `EgoCHARM`, `build_model`, `count_trainable_parameters`，名称和调用方式保持兼容。

- [ ] **Step 1: 统一模型导入和方法签名风格**

Use:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SamePaddingConv1d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dilation):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size,
            dilation=dilation,
        )

    def forward(self, x):
        total_padding = self.dilation * (self.kernel_size - 1)
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding
        x = F.pad(x, (left_padding, right_padding))
        return self.conv(x)
```

Remove `from __future__ import annotations` and model-only `Mapping`/`Sequence` return annotations. Keep constructor argument names so callers do not break.

- [ ] **Step 2: 让并行分支直接注册到 ModuleList**

Replace temporary module lists with:

```python
self.branches = nn.ModuleList(
    [
        SamePaddingConv1d(
            input_channels,
            branch_channels,
            kernel_size,
            dilation,
        )
        for dilation in dilations
    ]
)
```

Keep post-branch order exactly `BatchNorm1d -> MaxPool1d -> LeakyReLU -> Dropout`.

- [ ] **Step 3: 直接构造 LowLevelEncoder blocks**

Use an explicitly readable loop:

```python
self.blocks = nn.ModuleList()
for block_index in range(number_of_blocks):
    block_input_channels = input_channels if block_index == 0 else merged_channels
    self.blocks.append(
        ParallelDilatedBlock(
            block_input_channels,
            branch_channels,
            kernel_size,
            dilations,
            dropout,
        )
    )
```

Do not replace this with a nested one-line comprehension.

- [ ] **Step 4: 为每个模型阶段添加维度注释**

`LowLevelEncoder.forward()` must show:

```python
def forward(self, x):
    # x: (B, C, T)，论文默认 (B, 6, 50)。
    for block in self.blocks:
        x = block(x)

    # Conv1d: (B, F, T') -> GRU: (B, T', F)。
    x = x.transpose(1, 2)
    x, _ = self.gru(x)  # (B, T', E)

    # 取最后一个时间步: (B, T', E) -> (B, E)。
    return self.norm(x[:, -1])
```

`EgoCHARM.encode_each_second()` must show:

```python
# (B, S, C, T) -> (B * S, C, T)
x = x.reshape(batch_size * num_seconds, num_channels, samples_per_second)
x = self.low_level_encoder(x)  # (B * S, E)

# (B * S, E) -> (B, S, E)
return x.reshape(batch_size, num_seconds, -1)
```

`HighLevelClassifier.forward()` must show `(B, S, E) -> (B, S, H) -> (B, K)`.

- [ ] **Step 5: 验证模型行为和参数量**

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 /home/xreal/miniconda3/envs/egocharm/bin/python \
  -m pytest -p no:cacheprovider -q tests/test_model.py tests/test_integration.py
```

Expected: all pass; parameter counts remain `21_863`, `63_111`, `63_369`.

---

### Task 3: 整理 Dataset、配置与预处理数据流

**Files:**

- Modify: `src/egocharm/dataset.py`
- Modify: `src/egocharm/config.py`
- Modify: `scripts/preprocess_data.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_dataset.py`
- Modify: `tests/test_preprocess_data.py`
- Test: `tests/test_config.py`
- Test: `tests/test_dataset.py`
- Test: `tests/test_preprocess_data.py`

**Interfaces:**

- Consumes: official selection JSON、VRS reader callback、当前 NPZ schema。
- Produces: unchanged `TakeRecord`, `WindowReference`, `EgoCharmDataset`, `create_data_loader`, `SelectedTake`, `RawImu`, `PreprocessSettings`, preprocessing functions and CLI.

- [ ] **Step 1: 简化 Dataset 方法签名但保留边界类型**

Keep dataclass field types and filesystem boundary types. Model-facing methods become:

```python
class EgoCharmDataset(Dataset):
    def __init__(self, window_references, sampling_rate_hz, window_seconds):
        super().__init__()
        self.window_references = tuple(window_references)
        self.sampling_rate_hz = sampling_rate_hz
        self.window_seconds = window_seconds

        if sampling_rate_hz <= 0 or window_seconds <= 0:
            raise ValueError("采样率和窗口长度必须大于零")

    def __len__(self):
        return len(self.window_references)
```

Keep the existing `__getitem__()` body and change only its signature to
`def __getitem__(self, index):`. Do not remove `super().__init__()`.

- [ ] **Step 2: 在 Dataset 中标注数组布局变化**

Use comments adjacent to operations:

```python
# NPZ values: (N, C)，取出30秒窗口后为 (S * T, C)。
x = cached_values[start:stop]

# (S * T, C) -> (S, T, C) -> (S, C, T)
x = x.reshape(self.window_seconds, self.sampling_rate_hz, len(CHANNEL_NAMES))
x = x.transpose(0, 2, 1)
```

Keep the finite check, incomplete-window error, `float32`, contiguous conversion and scalar `long` target.

- [ ] **Step 3: 标注 DataLoader 输出契约**

Add a concise docstring/comment to `create_data_loader()`:

```text
single sample: features (S, C, T), target scalar
batch: features (B, S, C, T), targets (B,)
```

Do not change shuffle, workers, pin-memory or persistent-worker behavior.

- [ ] **Step 4: 整理预处理函数并添加数组形状注释**

At the relevant operations document:

```python
# timestamps: (N,), values: (N, 6)
# segment_values: (M, 6)
# resampled_values: (M', 6)
# segment_bounds: (num_segments, 2)
```

Keep all existing public function names, atomic NPZ replacement, source VRS read-only behavior and dry-run behavior. Remove return annotations only where they add no clarity; retain callback and dataclass types that define file/data boundaries.

- [ ] **Step 5: 去掉测试函数的无价值返回标注**

For example:

```python
def test_dataset_returns_one_hierarchical_window(tmp_path):
```

The line above replaces only the existing function declaration; its current test body remains
unchanged. Keep fixture types only when they clarify a non-obvious pytest object.

- [ ] **Step 6: 运行数据相关测试**

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 /home/xreal/miniconda3/envs/egocharm/bin/python \
  -m pytest -p no:cacheprovider -q \
  tests/test_config.py tests/test_dataset.py tests/test_preprocess_data.py
```

Expected: all pass, including unscaled-cache and dry-run-without-VRS tests.

---

### Task 4: 提取训练引擎和 Checkpoint 模块

**Files:**

- Create: `src/egocharm/engine.py`
- Create: `src/egocharm/checkpoint.py`
- Modify: `src/egocharm/train.py`
- Modify: `src/egocharm/evaluate.py`
- Modify: `tests/test_train.py`
- Test: `tests/test_train.py`

**Interfaces:**

- Produces from `engine.py`: `EpochMetrics`, `choose_device`, `calculate_class_weights`, `train_one_epoch`, `validate_one_epoch`.
- Produces from `checkpoint.py`: `TrainingState`, `save_checkpoint`, `load_checkpoint`.
- Compatibility: all eight names remain importable from `egocharm.train`.

- [ ] **Step 1: 添加模块边界兼容测试并验证 RED**

Add to `tests/test_train.py`:

```python
def test_train_module_reexports_engine_and_checkpoint_interfaces():
    from egocharm import checkpoint, engine, train

    assert train.train_one_epoch is engine.train_one_epoch
    assert train.validate_one_epoch is engine.validate_one_epoch
    assert train.save_checkpoint is checkpoint.save_checkpoint
    assert train.load_checkpoint is checkpoint.load_checkpoint
```

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 /home/xreal/miniconda3/envs/egocharm/bin/python \
  -m pytest -p no:cacheprovider -q \
  tests/test_train.py::test_train_module_reexports_engine_and_checkpoint_interfaces
```

Expected: FAIL because `egocharm.engine` and `egocharm.checkpoint` do not yet exist.

- [ ] **Step 2: 创建 engine.py**

Move `EpochMetrics` and the functions currently located at `src/egocharm/train.py:36-223`
without behavioral changes. The resulting module exports these exact interfaces:

```python
@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    macro_f1: float
    number_of_samples: int
```

- `choose_device(device_name)`
- `calculate_class_weights(labels, number_of_classes)`
- `_summarize_epoch(total_loss, labels, predictions, number_of_classes)`
- `train_one_epoch(model, data_loader, optimizer, loss_function, device, gradient_scaler, maximum_gradient_norm, number_of_classes)`
- `validate_one_epoch(model, data_loader, loss_function, device, number_of_classes)`

Training loop dimension comments must include:

```python
# features: (B, S, C, T), labels: (B,)
# logits: (B, K), loss: scalar
```

Keep AMP and gradient clipping order unchanged.

- [ ] **Step 3: 创建 checkpoint.py**

Move `TrainingState`, `save_checkpoint`, and `load_checkpoint` unchanged. Preserve Python, NumPy, CPU Torch and CUDA RNG state handling plus atomic `os.replace()`.

- [ ] **Step 4: 从 train.py 重新导出旧接口**

At module scope use:

```python
from egocharm.checkpoint import TrainingState, load_checkpoint, save_checkpoint
from egocharm.engine import (
    EpochMetrics,
    calculate_class_weights,
    choose_device,
    train_one_epoch,
    validate_one_epoch,
)
```

Update `evaluate.py` to use these modules directly where appropriate, while keeping compatibility imports available from `train.py`.

- [ ] **Step 5: 运行训练和兼容测试**

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 /home/xreal/miniconda3/envs/egocharm/bin/python \
  -m pytest -p no:cacheprovider -q tests/test_train.py tests/test_integration.py
```

Expected: all pass; checkpoint round trip restores parameters and state.

---

### Task 5: 提取指标模块并简化评估入口

**Files:**

- Create: `src/egocharm/metrics.py`
- Modify: `src/egocharm/evaluate.py`
- Modify: `tests/test_evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**

- Produces from `metrics.py`: `calculate_metrics(labels, predictions, class_names)` and `save_metrics(metrics, output_directory)`.
- Compatibility: both names remain importable from `egocharm.evaluate`.

- [ ] **Step 1: 添加评估模块兼容测试并验证 RED**

Add:

```python
def test_evaluate_module_reexports_metric_interfaces():
    from egocharm import evaluate, metrics

    assert evaluate.calculate_metrics is metrics.calculate_metrics
    assert evaluate.save_metrics is metrics.save_metrics
```

Run the single test and expect import failure because `egocharm.metrics` does not exist.

- [ ] **Step 2: 创建 metrics.py**

Move `calculate_metrics`, `_atomic_json`, `_atomic_class_csv`, `_save_confusion_matrix`, and `save_metrics`. Keep class order, zero-division behavior, atomic files, filenames, plot labels and normalization unchanged.

- [ ] **Step 3: 从 evaluate.py 重新导出并简化 main**

Use:

```python
from egocharm.metrics import calculate_metrics, save_metrics
```

Keep inference dimension comments:

```python
# features: (B, S, C, T)
logits = model(features)       # (B, K)
predictions = logits.argmax(1) # (B,)
```

- [ ] **Step 4: 运行评估测试**

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/egocharm-matplotlib \
  /home/xreal/miniconda3/envs/egocharm/bin/python \
  -m pytest -p no:cacheprovider -q tests/test_evaluate.py tests/test_integration.py
```

Expected: all pass and no `.tmp` files remain.

---

### Task 6: 简化训练入口并统一全部测试风格

**Files:**

- Modify: `src/egocharm/train.py`
- Modify: `src/egocharm/evaluate.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_dataset.py`
- Modify: `tests/test_evaluate.py`
- Modify: `tests/test_integration.py`
- Modify: `tests/test_model.py`
- Modify: `tests/test_preprocess_data.py`
- Modify: `tests/test_train.py`
- Test: `tests/`

**Interfaces:**

- Consumes: extracted engine/checkpoint/metrics modules.
- Produces: unchanged CLI parser and `main(arguments=None)` behavior.

- [ ] **Step 1: 按阶段整理 train.main()**

Keep the flow visibly ordered:

```text
parse arguments
load config
handle dry-run
set seed and device
build datasets/loaders
build model/loss/optimizer/scheduler/scaler
resume checkpoint
run epochs
save last/best checkpoint
```

Use short section comments only. Do not create a generic factory abstraction merely to reduce line count.

- [ ] **Step 2: 简化局部名称和类型噪声**

Use PyTorch-standard local names (`x`, `targets`, `logits`, `num_classes`, `num_epochs`) where scope is short. Preserve config keys such as `number_of_classes` and constructor parameter names used externally.

- [ ] **Step 3: 统一测试文件风格**

Remove all test-function `-> None`, simplify helper annotations that repeat obvious values, keep `tmp_path`/`monkeypatch` types only where useful, and retain descriptive test names. Do not merge tests into parameterized forms if that obscures the scenario.

- [ ] **Step 4: 运行完整测试**

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/egocharm-matplotlib \
  /home/xreal/miniconda3/envs/egocharm/bin/python -m pytest -p no:cacheprovider -q
```

Expected: all tests pass; expected total is at least 30 after new contract tests.

---

### Task 7: 更新说明并执行最终无数据验证

**Files:**

- Modify: `README.md`
- Modify: `docs/simplified-beginner-workflow-design.md` only where module paths are stale
- Test: whole project

**Interfaces:**

- Produces: accurate module map, commands and dimension flow documentation.

- [ ] **Step 1: 更新 README 模块结构**

Document `engine.py`, `checkpoint.py`, and `metrics.py`, plus the unified dimension symbols and end-to-end flow:

```text
(N, 6) -> (S, C, T) -> (B, S, C, T)
-> (B*S, C, T) -> (B*S, E) -> (B, S, E) -> (B, K)
```

- [ ] **Step 2: 扫描风格和旧接口残留**

Run:

```bash
rg -n "def (forward|__init__|test_[^(]+).*->|blocks: list\[ParallelDilatedBlock\]|from torch import nn|from torch\.nn import functional" \
  src scripts tests
```

Expected: no unwanted model/test signature or old temporary-list pattern. Necessary boundary annotations outside model/tests may remain.

- [ ] **Step 3: 编译全部 Python 文件**

Run:

```bash
/home/xreal/miniconda3/envs/egocharm/bin/python -m compileall -q scripts src tests
```

Expected: exit code 0.

- [ ] **Step 4: 运行完整测试套件**

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/egocharm-matplotlib \
  /home/xreal/miniconda3/envs/egocharm/bin/python -m pytest -p no:cacheprovider -q
```

Expected: all tests pass with zero failures.

- [ ] **Step 5: 运行预处理 dry-run**

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 /home/xreal/miniconda3/envs/egocharm/bin/python \
  scripts/preprocess_data.py --config configs/egocharm.yaml \
  --split train --max-takes 1 --dry-run
```

Expected JSON contains `"vrs_opened": false`.

- [ ] **Step 6: 运行训练 dry-run**

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 /home/xreal/miniconda3/envs/egocharm/bin/python \
  -m egocharm.train --config configs/egocharm.yaml --run-name style-smoke --dry-run
```

Expected JSON contains `"training_started": false`.

- [ ] **Step 7: 清理验证生成的缓存**

Find only project-local `__pycache__` and `.pytest_cache` directories created by compile/test commands and move those exact paths to Trash. Do not delete data, artifacts, configs, source files, or unrelated caches.
