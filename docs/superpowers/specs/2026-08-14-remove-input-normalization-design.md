# 删除输入数据归一化设计

日期：2026-08-14

## 目标

完全删除重采样 IMU 进入 Dataset 后的全局均值/标准差归一化。Dataset 直接把
NPZ 中的 float32 六轴 IMU 切成 30 秒窗口、变换维度并返回 PyTorch Tensor。

本变更只删除数据输入归一化。模型内部 `BatchNorm1d` 属于 EgoCHARM 网络结构，
继续保留，因此模型结构和参数量不变。

## 新数据流

```text
VRS
  → imu-left 六轴数据
  → 时间戳清洗与断点分段
  → 各片段线性重采样到 50 Hz
  → NPZ values: (N, 6), float32
  → Dataset 截取 (1500, 6)
  → reshape/transpose 为 (30, 6, 50)
  → DataLoader 组成 (B, 30, 6, 50)
  → EgoCHARM
```

Dataset 返回的数值保持传感器单位：加速度为 `m/s²`，角速度为 `rad/s`。

## 删除内容

- 从 `configs/egocharm.yaml` 删除 `data.normalization_path`；
- 从 `scripts/preprocess_data.py` 删除流式均值/方差算法；
- 删除 `fit_training_normalization()`；
- 删除命令参数 `--fit-normalization`；
- 从 `src/egocharm/dataset.py` 删除 `NormalizationStatistics`；
- 删除 `load_normalization()`；
- 从 `EgoCharmDataset` 构造函数删除归一化参数；
- 删除 Dataset 中 `(values - mean) / standard_deviation`；
- 从训练和评估入口删除归一化文件读取；
- 删除对应测试、README 命令和归一化说明。

## 保留内容

- 输入文件完整性检查；
- `imu-left` 六轴通道顺序；
- 非有限值检查、时间戳去重、断点分段和 50 Hz 重采样；
- NPZ 原子缓存与连续片段边界；
- 30 秒窗口、10 秒步长；
- official train/val/test 划分；
- 类别加权交叉熵；
- 模型中所有 `BatchNorm1d`；
- 低层 21,863、高层七分类 63,111、九分类 63,369 的参数量。

## 接口变化

原 Dataset 构造：

```python
EgoCharmDataset(window_references, normalization, sampling_rate_hz, window_seconds)
```

修改为：

```python
EgoCharmDataset(window_references, sampling_rate_hz, window_seconds)
```

预处理命令不再接受 `--fit-normalization`。train 和 evaluate 不再要求
`artifacts/normalization/train.json` 存在。

## 测试标准

1. Dataset 测试使用非零均值的固定输入，确认输出值未经缩放；
2. 预处理 parser 测试确认 `--fit-normalization` 已不存在；
3. 端到端夹具不创建归一化 JSON；
4. 模型精确参数量测试继续通过；
5. 完整 pytest、预处理 dry-run 和训练 dry-run 通过；
6. 测试与 dry-run 不读取真实 VRS，也不启动真实训练。
