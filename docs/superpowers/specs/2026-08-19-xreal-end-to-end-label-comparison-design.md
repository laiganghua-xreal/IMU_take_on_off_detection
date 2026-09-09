# Xreal 端到端三种标签方案对比设计

## 目标

合并 2026-08-18 与 2026-08-19 两天的 Xreal IMU 数据，覆盖现有 Xreal
cache 和 selection。在不使用 Ego-Exo4D 预训练权重的前提下，分别从随机
初始化开始端到端训练 LLE 与 HLE，对比四分类、三分类和二分类任务。

现有冻结 LLE、二分类及其他历史权重和评估结果全部保留。新 cache 将来也供
冻结 Ego LLE 的对照实验复用。

## 原始数据与标签

数据根目录：

- `/home/xreal/record_data_tools/data/20260818/free_record_only_imu`
- `/home/xreal/record_data_tools/data/20260819/free_record_only_imu`

目录名称按以下优先级映射，必须先匹配 `WALK_ON/WALK_OFF`，避免被普通
`ON/OFF` 规则误判：

| 目录标记 | 原始四分类标签 |
|---|---|
| `WORKING_WEAR`、`WALK_ON` | `WORN` |
| `NOWEAR`、`WALK_OFF` | `NOT_WORN` |
| 普通 `ON` | `PUT_ON` |
| 普通 `OFF` | `TAKE_OFF` |

2026-08-19 包含 42 条 `PUT_ON`、31 条 `TAKE_OFF`、一条约 146 秒的
`WORN` 走路序列和一条约 122 秒的 `NOT_WORN` 走路序列。两天合并后预计
共有 62 条 `PUT_ON` 与 55 条 `TAKE_OFF` 事件录制。

## 预处理

所有数据执行与现有流程一致的处理：

1. 读取时间对齐的 acc XYZ 和 gyro XYZ；
2. 从 Xreal 右、上、后方向旋转到 CPF 左、上、前方向；
3. 重采样到 50 Hz；
4. 保存 5 秒、`(250, 6)` 的 NPZ；
5. 训练读取时对三个加速度轴除以 9.8，陀螺仪不变。

稳定录制使用 5 秒窗口、1 秒起点步长。事件录制以完整源文件为单位，每个
文件截取一个居中的 5 秒窗口。

新样本先写到同文件系统的临时目录。全部成功后再替换
`artifacts/xreal_wear_detection/cache/xreal_cpf` 中旧 selection 引用的
cache，并原子更新 `Xreal_datasets/metadata/xreal_wear_selection.json`。
不删除该目录以外的文件。

## 数据划分

train/val/test 比例固定为 `7:1:2`，随机种子为 42。

事件数据按“日期 + 原始标签”分层，然后以完整录制文件为单位划分。这样两天
的 `PUT_ON` 和 `TAKE_OFF` 都能进入各 split，同一文件绝不跨 split。
整数数量使用确定性的最大余数法分配。

每条稳定长录制独立按时间顺序划分为 70%/10%/20%。相邻 split 之间各跳过
一个完整 5 秒区间，重叠窗口只在自己的时间区域内部生成，不能跨越边界。

三种标签任务严格复用同一个四分类 selection；标签映射只在 Dataset 内发生。

## 三种任务

| 原始标签 | 四分类 | 三分类 | 二分类 |
|---|---|---|---|
| `WORN` | `WORN` | `OTHERS` | `NO_EVENT` |
| `NOT_WORN` | `NOT_WORN` | `OTHERS` | `NO_EVENT` |
| `PUT_ON` | `PUT_ON` | `PUT_ON` | `EVENT` |
| `TAKE_OFF` | `TAKE_OFF` | `TAKE_OFF` | `EVENT` |

任务模式名称为 `four_class`、`three_class` 和 `binary_event`。每个配置声明与
模式严格一致的类别顺序和分类头大小。

## 模型与训练

LLE 暂时保持 EgoCHARM 结构，确保以后可以与 Ego 预训练 LLE 做受控对比。
HLE 不直接沿用 EgoCHARM 的 30 秒高层活动参数，而是针对当前 5 秒 Xreal
任务单独选择容量。LLE 与 HLE 均随机初始化，全部参数参与反向传播，不读取
Ego-Exo4D checkpoint。

正式标签对比前，先运行一次三分类 HLE 容量搜索：

- 固定 seed 42 和其他全部超参数；
- 比较 `high_level_hidden_size = 16, 32, 64`；
- 只使用 val 计算
  `hle_selection_score = (Event F1 + direction Macro-F1) / 2`；
- 选择分数最高的容量；若分数差不超过 0.005，选择参数更少的容量；
- test 在容量确定前保持封闭，不生成也不读取结果。

选出的 HLE hidden size 固定用于四分类、三分类和二分类。三个任务只允许输出层
大小因类别数量不同而变化，不能分别调节 HLE 容量。容量搜索结果保存候选值、
可训练参数量、最佳轮次和验证指标。

现有冻结 LLE 配置保持向后兼容。训练入口根据配置中的初始化策略选择：

- `random_end_to_end`：随机创建完整模型并训练全部参数；
- `pretrained_frozen_lle`：保持现有加载并冻结 Ego LLE 的行为。

容量搜索完成后，正式比较运行 seed 42、43、44，三个任务共九次训练，分别
保存 checkpoint、val/test 报告，不覆盖历史结果：

```text
artifacts/xreal_wear_detection/runs/xreal_end_to_end/
├── hle_search/three_class/hidden_16 ... hidden_64
├── four_class/seed_42 ... seed_44
├── three_class/seed_42 ... seed_44
├── binary_event/seed_42 ... seed_44
└── comparison/
```

所有实验使用类别加权交叉熵，并以各自任务的验证集 Macro-F1 选择最佳
checkpoint。比较报告记录最佳轮次以及训练参数量。

## 公平评估

### 任务内指标

每个任务保存 Accuracy、Macro-F1、逐类 precision/recall/F1、原始和归一化
混淆矩阵。Accuracy 只作辅助指标，不能用于不平衡任务的主要结论。

### 公共事件指标

三种任务统一折叠成 `NO_EVENT/EVENT`，报告 Event Precision、Recall、F1、
漏检率与稳定窗口 FPR。四分类和三分类还报告真实事件样本上的方向 Accuracy
与方向 Macro-F1，以及端到端正确识别 `PUT_ON/TAKE_OFF` 的比例。

公共事件分数定义为：

- 四分类、三分类：`P(PUT_ON) + P(TAKE_OFF)`；
- 二分类：`P(EVENT)`。

每次训练只在 val 上选择最大化 Event F1 的阈值，然后固定该阈值评估 test。
test 不参与 checkpoint 或阈值选择。

### 连续录制误报

稳定窗口每秒输出一次预测。时间上相交或相接的阳性 5 秒窗口合并为一个报警
区间。按原始稳定录制时间区间的并集计算事件级 `false alarms/hour`，不再用
“窗口数量 × 5 秒”错误估计重叠数据的时长。同时保留窗口级 FPR。

### 汇总

比较目录保存每个 seed 的详细结果，以及三个 seed 的均值和标准差。主表包括：

- Task Macro-F1；
- Event Precision、Recall、F1；
- 方向 Macro-F1（仅四分类、三分类）；
- 稳定窗口 FPR；
- 合并后的 false alarms/hour；
- 最佳轮次和可训练参数量。

由于测试事件仅约二十多个，报告必须注明样本数，小幅差异不作强结论。

## 测试与验收

自动化测试覆盖：

- `WALK_ON/WALK_OFF` 优先标签映射；
- 两个 source root 的发现与 UID 唯一性；
- 7:1:2 分层文件划分及可复现性；
- 稳定重叠窗口不跨 split 和 5 秒隔离区；
- 三种标签映射及分类头大小；
- 随机端到端模型不加载 checkpoint，LLE/HLE 都可训练；
- HLE 容量搜索只读取 val，并将选定容量固定到三个正式任务；
- val 阈值选择不会读取 test；
- 重叠阳性窗口合并与真实时长误报率；
- 三组输出目录互不覆盖。

完成标准：预处理和全部测试成功；九次训练均生成 best checkpoint；三种任务的
val/test 评估及三 seed 汇总完成；旧权重目录未被改动。
