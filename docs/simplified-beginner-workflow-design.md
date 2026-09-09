# EgoCHARM 初学者版代码重构设计

日期：2026-08-14

## 目标

将项目整理成清晰的两阶段流程：

1. `scripts/preprocess_data.py` 单独负责把 Ego-Exo4D VRS 转换为训练缓存；
2. `src/egocharm` 只负责加载缓存、构造 DataLoader、定义模型、训练和评估。

代码面向 Python 和 PyTorch 初学者。变量名使用完整单词，关键步骤分行书写，
函数使用类型提示和中文注释。注释重点解释数据形状、处理原因和训练状态变化，
不使用为了缩短代码而增加理解难度的复杂表达式。

## 最终目录

```text
egocharm/
├── README.md
├── environment.yml
├── pyproject.toml
├── configs/
│   └── egocharm.yaml
├── scripts/
│   └── preprocess_data.py
├── src/egocharm/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── engine.py
│   ├── checkpoint.py
│   ├── metrics.py
│   ├── train.py
│   └── evaluate.py
├── tests/
│   ├── test_preprocess_data.py
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_train.py
│   ├── test_evaluate.py
│   ├── test_config.py
│   └── test_integration.py
└── ego-exoD4_dataset/
```

不保留 `legacy/`。现有多层 CLI、`src/egocharm/data`、`models`、`training` 和
`evaluation` 目录在新实现通过测试后直接移除。

## 数据预处理脚本

`scripts/preprocess_data.py` 是唯一接触 VRS 和 Project Aria Tools 的代码。

它按顺序完成：

1. 读取 available selection，并按官方 split 或 take UID 筛选；
2. 用文件存在性和字节数确认 VRS 已完整下载；
3. 从 `_noimagestreams.vrs` 中只读取 `imu-left`；
4. 固定六通道顺序为加速度 XYZ、角速度 XYZ；
5. 检查有限值、排序时间戳、重复时间戳保留最后一条；
6. 在相邻时间戳间隔超过 0.1 秒时切分连续片段；
7. 丢弃不足 30 秒的片段；
8. 对每个片段线性重采样到 50 Hz，不跨数据断点插值；
9. 原子保存每个 take 的 NPZ 缓存及连续片段边界。

脚本支持 `--split`、`--take-uid`、`--max-takes` 和 `--dry-run`。
重复执行时，源文件信息和配置未改变的有效缓存直接跳过。

## Dataset 与 DataLoader

`src/egocharm/dataset.py` 不导入 Project Aria Tools，也不理解 VRS。它只读取：

- `artifacts/cache/imu-left/<take_uid>.npz`；
- selection 中的官方 split 和活动标签。

它先在每个连续片段内部建立 30 秒、10 秒步长的窗口索引。`__getitem__` 读取
一个 `(1500, 6)` 窗口，不做均值/标准差归一化，直接转换成 PyTorch 所需的
`(30, 6, 50)`。加速度保持 `m/s²`，角速度保持 `rad/s`。训练 DataLoader
打乱窗口，验证和测试 DataLoader 不打乱。

## 模型

`src/egocharm/model.py` 在一个文件中定义三个容易顺序阅读的类：

1. LLE（`LowLevelEncoder`）：把一秒 `(6, 50)` 编码成 32 维向量；
2. HLA（`HighLevelClassifier`）：用 GRU 处理 30 个一秒向量并输出七类 logits；
3. `EgoCHARM`：把 `(B, 30, 6, 50)` 向量化连接前两个模块。

保留论文重实现的实际参数配置与计数，不在模型内部添加 softmax。

完整维度流为：

```text
(N, 6)
-> (S, C, T)
-> (B, S, C, T)
-> (B * S, C, T)
-> (B * S, E)
-> (B, S, E)
-> (B, K)
```

默认 `S=30`、`C=6`、`T=50`、`E=32`。`N` 是缓存中一个 take 的总采样点数，
`B` 是 batch 大小，`K=7` 是类别数。模型内部 `F` 是并行卷积拼接后的特征通道数
（默认 55），`H` 是 HLA 的 GRU 隐藏维度（默认 128）。Dataset 保留原始 IMU
数值和单位、不做均值/标准差缩放；LLE 内的 `BatchNorm1d` 保留。

## 训练

训练职责拆分为便于复用的模块：`engine.py` 选择设备、计算类别权重并执行单轮
训练和验证；`checkpoint.py` 原子保存和恢复完整 checkpoint；`train.py` 是
`python -m egocharm.train` 的 CLI 编排入口。训练流程按阅读顺序是：

1. `config.py` 加载配置、解析项目路径并设置随机种子；
2. `dataset.py` 创建 train/val Dataset 和 DataLoader；
3. `engine.py` 从 train 窗口统计类别权重，并执行 `train_one_epoch`；
4. `train.py` 创建模型、加权交叉熵、Adam 和 StepLR；
5. `engine.py` 使用 `model.eval()` 和 `torch.no_grad()` 验证；
6. `train.py` 根据 val macro-F1 选择 `best.pt` 并安排每轮保存 `last.pt`，
   `checkpoint.py` 只负责持久化和恢复这些 checkpoint；
7. `--resume` 恢复模型、优化器、scheduler、AMP 和随机状态。

训练过程不构造 test DataLoader。

## 评估

`src/egocharm/evaluate.py` 通过 `python -m egocharm.evaluate` 单独运行。
它用 `checkpoint.py` 恢复模型、用 `engine.py` 选择设备、用 `metrics.py` 计算并
保存 accuracy、macro-F1、逐类别指标、CSV、JSON 和混淆矩阵 PNG；显式接收
checkpoint 和 `val` 或 `test` split。

## 清理范围

确认新测试通过后删除：

- `.bootstrap/`：不再使用的 virtualenv 引导依赖；
- `.venv/`：不再使用的旧虚拟环境；
- `.pytest_cache/`：pytest 可再生成缓存；
- 所有 `__pycache__/` 和 `.pyc`：Python 可再生成字节码；
- 空的 `.git/`：当前不是有效 Git 仓库；
- 已被新结构替代的旧源码、旧测试和三份拆分配置。

保留：

- `ego-exoD4_dataset/` 及其中全部数据和 metadata；
- `/home/xreal/miniconda3/envs/egocharm` Conda 环境；
- `.vscode/` IDE 设置；
- `environment.yml`、`pyproject.toml`、README 和有效设计文档。

任何清理操作都不进入或修改 `ego-exoD4_dataset/takes/`。

## 验证标准

1. 预处理单元测试使用临时伪 VRS reader，不读取真实数据；
2. Dataset 测试确认窗口不跨断点且形状为 `(30, 6, 50)`；
3. 模型测试确认 logits 形状和论文参数量；
4. 一步训练测试确认参数发生更新；
5. checkpoint 往返测试确认恢复完整状态；
6. 评估测试确认 macro-F1、CSV 和 PNG；
7. `python -m compileall` 与完整 pytest 通过；
8. 只运行预处理 dry-run，不自动读取真实 VRS 或启动长时间训练。
