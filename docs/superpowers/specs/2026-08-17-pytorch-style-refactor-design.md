# EgoCHARM PyTorch 风格重构设计

日期：2026-08-17

## 目标

在不改变算法、数据格式、配置格式和命令行用法的前提下，统一重构
EgoCHARM 项目的 Python 代码。代码风格参考 Nokia Bell Labs 的
`pretrained-imu-encoders`，采用直接、紧凑且容易调试的 PyTorch 研究代码表达，
同时避免参考仓库中的硬编码路径、格式不一致和废弃代码堆积。

本次重构必须满足：

- 保持现有预处理、训练和评估命令不变。
- 保持 YAML 配置字段不变。
- 保持 Dataset 输出、模型结构、参数量和训练行为不变。
- 不重新加入原始 IMU 数据归一化。
- 在所有数据流和模型维度发生变化的位置添加简洁注释。
- 不读取真实 VRS 文件，不启动正式训练。

## 参考风格

主要参考：

- <https://github.com/Nokia-Bell-Labs/pretrained-imu-encoders/blob/main/lib/imu_models.py>
- <https://github.com/Nokia-Bell-Labs/pretrained-imu-encoders/blob/main/pretraining.py>

采用的特点：

- 使用 `import torch.nn as nn` 和直接的 PyTorch 层定义。
- 使用 `nn.Sequential`、`nn.ModuleList` 等容器表达模型结构。
- `forward()` 保持短小，集中描述张量如何流过模型。
- 模型和测试代码不使用密集的返回类型标注。
- 训练入口按照配置、数据、模型、优化器、训练循环的顺序组织。

明确不采用的特点：

- 硬编码本机绝对路径。
- 大段注释掉的旧实现。
- 不一致的空行、空格和命名。
- 使用全局 CLI 参数驱动普通函数。

## 模块边界

目标结构如下：

```text
src/egocharm/
├── config.py       # 配置、项目路径和随机种子
├── dataset.py      # 数据记录、窗口索引、Dataset 和 DataLoader
├── model.py        # LLE、HLA 和完整 EgoCHARM
├── engine.py       # 类别权重、单轮训练和验证
├── checkpoint.py   # checkpoint 保存与恢复
├── metrics.py      # Accuracy、F1、CSV、JSON 和图表
├── train.py        # 训练命令入口与流程编排
└── evaluate.py     # 评估命令入口与流程编排

scripts/
└── preprocess_data.py  # 独立的 IMU 数据预处理
```

`train.py` 和 `evaluate.py` 继续提供现有导入接口。已经被测试或其他模块使用的
函数即使移动到 `engine.py`、`checkpoint.py` 或 `metrics.py`，也会从原模块重新
导入，从而避免破坏现有调用。

`scripts/preprocess_data.py` 保持独立文件，不把预处理重新混入训练包。

## 代码风格规则

### 导入

导入顺序统一为：标准库、第三方库、本项目模块。模型代码统一使用：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

只导入实际使用的名称，不保留无效导入。

### 类型标注

- 模型的 `__init__()`、`forward()` 和测试函数不写 `-> None` 等密集标注。
- 文件路径、dataclass、配置入口和复杂数据结构保留有助于防错的标注。
- 不为了追求全覆盖而引入难读的 `Mapping`、`Sequence` 嵌套。
- 类型标注不得比函数本身更难理解。

### 命名

- `forward()` 中使用 PyTorch 常见名称，如 `x`、`features`、`embeddings`、
  `logits` 和 `targets`。
- 跨越多个步骤或容易混淆的值使用完整名称。
- 配置字段名保持现状，避免破坏 YAML。
- 类名和对外函数名保持兼容。

### PyTorch 容器

- 需要注册多个子模块时直接使用 `nn.ModuleList`。
- 严格线性的无分支层组合可使用 `nn.Sequential`。
- 不使用普通 Python `list` 长期保存网络层。
- 模型构造函数必须调用 `super().__init__()`。

### 注释

- 注释使用中文，保持简短。
- 注释说明张量维度、数据单位或设计原因，不复述代码语法。
- 维度变化的注释放在对应语句上方。
- 使用统一的大写符号，不在不同文件中创造同义符号。

## 统一维度符号

```text
B：batch size
N：原始或连续采样点数量
S：高层窗口中的秒数，默认 30
C：IMU 通道数，默认 6
T：每秒采样点数，默认 50
F：并行卷积分支拼接后的特征通道数
E：低层 embedding 维度，默认 32
H：高层 GRU 隐藏维度
K：类别数量
```

动态维度使用符号注释，默认论文配置可在需要时补充具体数值。例如：

```python
# (B, S, C, T) -> (B * S, C, T)，默认 S=30、C=6、T=50。
x = x.reshape(batch_size * num_seconds, num_channels, samples_per_second)
```

## 数据流

完整数据流保持为：

```text
原始左侧 IMU             timestamps: (N,), values: (N, 6)
    ↓ 清洗、连续段切分、50 Hz 线性重采样
NPZ 缓存                 timestamps: (N',), values: (N', 6)
    ↓ 30 秒窗口切分
Dataset 单样本            features: (30, 6, 50), target: scalar
    ↓ DataLoader
训练 batch               features: (B, 30, 6, 50), targets: (B,)
    ↓ 合并 B 与 S
LLE 输入                  (B * 30, 6, 50)
    ↓ LowLevelEncoder
低层 embedding            (B * 30, 32)
    ↓ 恢复 B 与 S
高层 GRU 输入             (B, 30, 32)
    ↓ HighLevelClassifier
分类 logits               (B, K)
```

原始加速度保持 `m/s²`，原始角速度保持 `rad/s`。Dataset 只转换布局和 dtype，
不执行均值/标准差归一化。模型内部的 BatchNorm 属于网络结构，继续保留。

## 模型重构

### ParallelDilatedBlock

使用 `nn.ModuleList` 保存不同 dilation 的卷积分支。各分支输出形状均为
`(B, branch_channels, T)`，沿通道维拼接为 `(B, F, T)`，其中
`F = branch_channels * num_branches`。后续 BatchNorm、池化、激活和 Dropout
可以组织成清晰的顺序模块，但不得改变当前执行顺序。

### LowLevelEncoder

直接创建 `nn.ModuleList` 并逐个追加卷积块，不先构造临时 Python 模块列表。
默认形状变化为：

```text
(B, 6, 50)
→ (B, 55, 25)
→ (B, 55, 12)
→ (B, 55, 6)
→ transpose: (B, 6, 55)
→ GRU: (B, 6, 32)
→ last step: (B, 32)
```

### EgoCHARM

只创建一个 LowLevelEncoder 实例。将 `(B, S, C, T)` 变换为
`(B*S, C, T)`，并行编码所有一秒窗口，再恢复为 `(B, S, E)`。
30 秒对应 30 个 embedding，而不是 30 份独立参数的 LLE。

### HighLevelClassifier

输入 `(B, S, E)`，高层 GRU 输出 `(B, S, H)`，取最后时间步 `(B, H)`，
线性层输出 `(B, K)`。不在模型中调用 softmax，因为 `CrossEntropyLoss`
直接接收 logits。

## 训练、评估与预处理

### 训练

`engine.py` 负责纯训练逻辑，`train.py` 只负责编排。训练和验证循环在取得 batch、
模型输出 logits 以及计算 loss 的位置标注形状：

```text
inputs: (B, S, C, T)
targets: (B,)
logits: (B, K)
loss: scalar
```

AMP、梯度裁剪、类别权重、学习率调度、恢复训练和 best checkpoint 行为保持不变。

### 评估

指标计算与结果写入移动到 `metrics.py`。`evaluate.py` 负责加载配置、模型、
checkpoint 和 test DataLoader，然后调用指标模块。Accuracy、macro-F1、逐类指标、
CSV、JSON 和两张混淆矩阵保持不变。

### 预处理

预处理继续独立于训练。关键数组标注为：

```text
raw timestamps: (N,)
raw values: (N, 6)
segment values: (M, 6)
resampled values: (M', 6)
segment bounds: (num_segments, 2)
```

不得修改或覆盖源 VRS。写入 NPZ 继续采用临时文件加原子替换。

## 错误处理

保留当前对以下情况的明确报错：

- 配置缺失必需区域。
- split 非 `train`、`val`、`test`。
- 缓存缺失或元数据不匹配。
- 窗口形状不完整。
- 数据包含 NaN 或 Inf。
- 模型输入不是四维。
- checkpoint 与模型或优化器状态不兼容。

重构不得使用宽泛的 `except Exception` 隐藏真实错误。

## 验证标准

实施前后均以自动化测试验证行为。完成条件是：

- 完整 pytest 套件通过。
- Dataset 单样本形状仍为 `(30, 6, 50)`。
- DataLoader batch 形状仍为 `(B, 30, 6, 50)`。
- 模型输出仍为 `(B, K)`。
- LLE 可训练参数仍为 `21,863`。
- 7 类高层分类器可训练参数仍为 `63,111`。
- 9 类高层分类器可训练参数仍为 `63,369`。
- 输入缓存数值不被 Dataset 缩放。
- 预处理 dry-run 不打开 VRS。
- 训练 dry-run 不启动训练。
- Python 编译检查通过。

验证过程中不运行真实数据预处理或长时间训练，不影响 Ego-Exo4D 下载和现有数据。
