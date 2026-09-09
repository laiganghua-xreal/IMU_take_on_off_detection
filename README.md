# EgoCHARM PyTorch 复现

本项目使用 Ego-Exo4D 左侧 IMU，实现论文 EgoCHARM 的七分类方法级复现。
代码刻意分成两个阶段：先把 VRS 预处理成普通 NumPy 缓存，再由 PyTorch
Dataset、DataLoader 和模型完成训练。训练代码不读取 VRS。

当前七个类别依次为 Basketball、Soccer、Dance、Rock Climbing、Cooking、
Bike Repair 和 Music。论文还使用 Nymeria 的两个类别，因此本项目的七分类结果
不能直接当成论文九分类结果。

## 1. 目录结构

```text
scripts/preprocess_data.py   VRS → 清洗、分段、50 Hz 六轴 IMU NPZ

src/egocharm/
├── config.py               配置、路径和随机种子
├── dataset.py              索引、Dataset 和 DataLoader
├── model.py                LLE、HLA 和 EgoCHARM
├── engine.py               类别权重、单轮训练和验证
├── checkpoint.py           checkpoint 保存和恢复
├── metrics.py              分类指标、CSV/JSON 和图表
├── train.py                EgoCHARM 七分类训练入口
├── evaluate.py             EgoCHARM 七分类评估入口
├── wear_dataset.py         Xreal 标签、selection 和 Dataset
├── wear_model.py           随机、冻结 LLE 和微调 LLE 模型
├── wear_evaluation.py      摘戴事件指标与阈值选择
├── train_wear.py           Xreal 摘戴分类训练入口
└── evaluate_wear.py        Xreal 摘戴分类评估入口
```

原始复现参数在 [configs/egocharm.yaml](configs/egocharm.yaml)，Xreal 摘戴任务
使用 `configs/xreal_*.yaml` 中对应的配置。

## 2. 环境

```bash
conda activate egocharm
cd /home/xreal/egocharm
python -m pip install -e . --no-deps --no-build-isolation
```

当前环境使用 Python 3.10 和 PyTorch 2.5.1+cu121。PyTorch 自带 CUDA 运行时，
训练本项目不要求系统命令 `nvcc` 可用。

## 3. 阶段一：预处理数据

先查看将要处理哪些 take。dry-run 只读 selection，不打开 VRS：

```bash
python scripts/preprocess_data.py \
  --config configs/egocharm.yaml \
  --split train \
  --max-takes 1 \
  --dry-run
```

确认后，真实处理一个 take：

```bash
python scripts/preprocess_data.py \
  --config configs/egocharm.yaml \
  --split train \
  --max-takes 1
```

预处理脚本依次完成：

1. 根据 selection 检查文件存在性和准确字节数；
2. 只读取 VRS 的 `imu-left`；
3. 按 `[accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]` 排列通道；
4. 检查有限值，排序时间戳，重复时间戳保留最后一条；
5. 相邻时间戳相差超过 0.1 秒时切开，绝不跨断点插值；
6. 丢弃不足 30 秒的连续片段；
7. 各片段分别线性重采样到 50 Hz；
8. 原子保存 `artifacts/cache/imu-left/<take_uid>.npz`。

NPZ 保存：

```text
timestamps_seconds  (N,)
values              (N, 6)
segment_bounds      (连续片段数, 2)
metadata_json       take、split、类别、源文件与处理设置
```

NPZ 始终保存传感器原始物理单位，不在预处理阶段归一化。这样修改模型输入处理方式
时不需要重新生成缓存。

可以按官方 split 完整预处理，命令可重复执行，匹配的有效缓存会跳过：

```bash
python scripts/preprocess_data.py --split train
python scripts/preprocess_data.py --split val
python scripts/preprocess_data.py --split test
```

## 4. 阶段二：Dataset 和 DataLoader

[dataset.py](src/egocharm/dataset.py) 只读取 NPZ，不导入 Project Aria。
每个连续片段内部按 30 秒窗口、10 秒步长建立索引。取一个样本时的数据形状为：

```text
缓存切片       (1500, 6)       30 秒 × 50 Hz
按秒分组       (30, 50, 6)
交换维度       (30, 6, 50)
DataLoader     (B, 30, 6, 50)
```

Dataset 在取出完整 30 秒窗口后，根据 `data.normalization` 处理模型输入。当前默认
的 `gravity` 策略将 `accel_x`、`accel_y`、`accel_z` 分别除以 `9.8`，将加速度
表示为重力加速度的倍数；`gyro_x`、`gyro_y`、`gyro_z` 仍保持 `rad/s` 原值。
设置为 `none` 时，六个通道都保持 NPZ 中的原始物理单位。输出始终为
`torch.float32`，模型内部的 `BatchNorm1d` 仍按网络结构保留。

训练 DataLoader 使用 `shuffle=True`，验证和测试使用 `shuffle=False`。

## 5. 完整维度流与符号

原始缓存中的一个 take 有 `N` 个采样点，每个采样点为六轴 IMU；`Dataset`
从中取 30 秒窗口并交给模型。完整的默认数据流为：

```text
(N, 6)
-> (S, C, T)
-> (B, S, C, T)
-> (B * S, C, T)
-> (B * S, E)
-> (B, S, E)
-> (B, K)
```

默认 `S=30`（每窗口秒数）、`C=6`（IMU 通道）、`T=50`（每秒采样数）和
`E=32`（LLE 输出的每秒嵌入维度）。`B` 是 DataLoader 的 batch 大小，`N` 是
缓存中一个 take 的总采样点数，`K=7` 是类别数。模型内部，`F` 表示并行膨胀卷积
拼接后的特征通道数（默认 `11 × 5 = 55`），`H` 表示 HLA 的 GRU 隐藏维度
（默认 128）。LLE（`LowLevelEncoder`）执行每秒编码，HLA
（`HighLevelClassifier`）把连续秒级嵌入映射为类别 logits；最后一层不做 softmax。

## 6. 训练

先用 dry-run 检查输出目录，不加载缓存、不启动训练：

```bash
python -m egocharm.train \
  --config configs/egocharm.yaml \
  --run-name gravity-baseline \
  --dry-run
```

开始训练：

```bash
python -m egocharm.train \
  --config configs/egocharm.yaml \
  --run-name gravity-baseline
```

现有 `artifacts/runs/baseline` 是使用未归一化输入训练的旧实验。启用 `gravity`
后不要从该目录恢复训练，也不要覆盖它；应使用新的 run name。

训练使用 official train 更新参数，official val 选择最佳模型，不构造 test
DataLoader。默认配置使用类别加权交叉熵、Adam、StepLR 和 CUDA 混合精度。

产物：

```text
artifacts/runs/gravity-baseline/
├── resolved-config.json
├── last.pt       每个 epoch 的完整训练状态
└── best.pt       val macro-F1 最佳状态
```

恢复训练使用 `last.pt`：

```bash
python -m egocharm.train --run-name gravity-baseline --resume
```

checkpoint 包含模型、优化器、scheduler、AMP scaler、epoch、最佳 macro-F1，
以及 Python、NumPy、PyTorch CPU/CUDA 随机状态。

## 7. 独立评估

先评估 val：

```bash
python -m egocharm.evaluate \
  --config configs/egocharm.yaml \
  --checkpoint artifacts/runs/gravity-baseline/best.pt \
  --split val
```

全部训练和调参完成后，再显式评估 test：

```bash
python -m egocharm.evaluate \
  --config configs/egocharm.yaml \
  --checkpoint artifacts/runs/gravity-baseline/best.pt \
  --split test
```

输出包括 `metrics.json`、`per-class.csv`、原始混淆矩阵 PNG 和行归一化混淆
矩阵 PNG。

## 8. 对比归一化与未归一化训练

下面的脚本从头顺序训练两组模型，每组默认 30 轮。两组使用相同的官方 split、
随机种子、模型、优化器和训练参数，唯一差别是 `data.normalization` 分别为
`none` 和 `gravity`：

```bash
python scripts/run_normalization_comparison.py
```

运行前可以只生成并查看两份配置和六条子命令，不启动训练：

```bash
python scripts/run_normalization_comparison.py --dry-run
```

中断后，两组对应目录都存在 `last.pt` 时可以恢复：

```bash
python scripts/run_normalization_comparison.py --resume
```

默认输出相互隔离：

```text
artifacts/comparisons/normalization/
├── none/
│   ├── config.yaml
│   ├── resolved-config.json
│   ├── best.pt
│   ├── last.pt
│   ├── train.log
│   ├── val/
│   └── test/
├── gravity/
│   └── 同样结构
├── comparison.csv
└── comparison.json
```

每个 `val/` 和 `test/` 都包含独立指标、逐类别 CSV、混淆矩阵和评估日志。
`comparison.csv` 与 `comparison.json` 汇总两组的 accuracy、macro-F1、最佳
epoch、最佳 val macro-F1 和 checkpoint 路径。没有 `--resume` 时，脚本拒绝
覆盖任何已有 `best.pt` 或 `last.pt`。

## 9. 模型参数量

当前论文方法级重实现保持：

- 低层编码器：21,863；论文报告 21,868，因论文未公开全部逐层参数而相差 5；
- 七分类高层分类器：63,111；
- 论文九分类高层结构：63,369。

## 10. 测试

```bash
python -m compileall -q scripts src tests
python -m pytest -q
```

测试中的六轴正弦信号只写入 pytest 临时目录，用来检查预处理、Dataset、一步
训练、checkpoint 和评估接口，不会用于报告实验结果，也不会读取真实 VRS。

## 11. Xreal 摘戴检测实验

Xreal 四分类迁移实验使用独立的 CPF cache、配置和运行目录，不会覆盖本 README
前面介绍的论文复现结果。完整的数据转换、CPF 预训练、冻结 LLE 训练和评估命令
见 [Xreal_datasets/README.md](Xreal_datasets/README.md)。
