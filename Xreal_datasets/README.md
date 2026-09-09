# Xreal Wear-State Detection

这个目录保存 Xreal 佩戴状态数据转换入口和 selection。训练输出完全位于
`artifacts/xreal_wear_detection`，不覆盖此前的 EgoCHARM 复现结果。

## 坐标与标签

Ego-Exo4D 和 Xreal 最终都使用 CPF 通道方向：`x=左，y=上，z=前`。

- Ego-Exo4D：每个 VRS 读取自己的 `T_Cpf_ImuLeft`，只使用旋转部分；
- Xreal：原始方向为右、上、后，转换为 `(-x, y, -z)`；
- 加速度和陀螺仪使用相同旋转；
- 两边都不应用平移或杆臂补偿。

Xreal 类别固定为：

```text
WORN, NOT_WORN, PUT_ON, TAKE_OFF
```

当前 train/val/test 来自同一人、同一天和同一批录制，只适合开发验证。正式
泛化测试需要另一天或另一位参与者的独立 session。

## 1. Ego-Exo4D CPF cache

先预览，不读取 VRS：

```bash
python scripts/preprocess_data.py \
  --config configs/xreal_wear_ego_pretrain_cpf.yaml \
  --max-takes 1 \
  --dry-run
```

确认后生成全部 CPF cache：

```bash
python scripts/preprocess_data.py \
  --config configs/xreal_wear_ego_pretrain_cpf.yaml \
  --split all
```

输出：

```text
artifacts/xreal_wear_detection/cache/ego_exo_cpf/
```

## 2. Ego-Exo4D CPF 预训练

```bash
python -m egocharm.train \
  --config configs/xreal_wear_ego_pretrain_cpf.yaml \
  --run-name ego_pretrain_cpf
```

冻结迁移阶段需要：

```text
artifacts/xreal_wear_detection/runs/ego_pretrain_cpf/best.pt
```

## 3. 转换 Xreal 数据

只检查发现的文件数量：

```bash
python Xreal_datasets/preprocess_xreal_imu.py --dry-run
```

生成五秒 CPF cache 和 selection。默认情况下，稳定状态窗口不重叠：

```bash
python Xreal_datasets/preprocess_xreal_imu.py
```

只重建 `WORN/NOT_WORN`，让五秒窗口每隔一秒开始一次，并原样保留已有的
`PUT_ON/TAKE_OFF` cache 和 split：

```bash
python Xreal_datasets/preprocess_xreal_imu.py \
  --stable-stride-seconds 1 \
  --reuse-existing-events
```

稳定状态仍沿用原来的按时间顺序 train/val/test 区域，并在相邻 split 之间保留
五秒隔离区，因此重叠窗口不会跨越 split 边界。

同时绘制每个源录制的六轴信号：

```bash
python Xreal_datasets/preprocess_xreal_imu.py --plot
```

输出：

```text
Xreal_datasets/metadata/xreal_wear_selection.json
artifacts/xreal_wear_detection/cache/xreal_cpf/
artifacts/xreal_wear_detection/plots/          # 仅 --plot 时生成
```

## 4. 冻结 LLE，训练四分类 HLE

先检查输出路径：

```bash
python -m egocharm.train_wear \
  --config configs/xreal_wear_frozen_lle.yaml \
  --run-name frozen_lle \
  --dry-run
```

开始训练：

```bash
python -m egocharm.train_wear \
  --config configs/xreal_wear_frozen_lle.yaml \
  --run-name frozen_lle
```

输出：

```text
artifacts/xreal_wear_detection/runs/frozen_lle/
├── best.pt
├── last.pt
└── resolved-config.json
```

## 5. 评估

```bash
python -m egocharm.evaluate_wear \
  --config configs/xreal_wear_frozen_lle.yaml \
  --checkpoint artifacts/xreal_wear_detection/runs/frozen_lle/best.pt \
  --split test
```

测试结果写入 `artifacts/xreal_wear_detection/runs/frozen_lle/test`，包括
`metrics.json`、逐类别 CSV 和两张混淆矩阵。

## 6. 可选的二分类摘戴事件实验

二分类模式直接复用已有的 50 Hz、五秒、六轴 cache，不需要重新预处理。
它只判断窗口内是否发生摘戴事件，不判断空间运动方向：

```text
WORN, NOT_WORN  -> NO_EVENT
PUT_ON, TAKE_OFF -> EVENT
```

训练：

```bash
python -m egocharm.train_wear \
  --config configs/xreal_wear_binary_event.yaml \
  --run-name binary_event
```

验证集与测试集评估：

```bash
python -m egocharm.evaluate_wear \
  --config configs/xreal_wear_binary_event.yaml \
  --checkpoint artifacts/xreal_wear_detection/runs/binary_event/best.pt \
  --split val

python -m egocharm.evaluate_wear \
  --config configs/xreal_wear_binary_event.yaml \
  --checkpoint artifacts/xreal_wear_detection/runs/binary_event/best.pt \
  --split test
```

独立输出位于：

```text
artifacts/xreal_wear_detection/runs/binary_event/
```

`metrics.json` 除二分类指标外，还按原始 `WORN`、`NOT_WORN`、`PUT_ON`、
`TAKE_OFF` 标签分别报告稳定窗口误报率、每小时误报次数和事件召回率。
当前版本不包含在线状态机、高采样率输入、模长或变化率等额外特征。

## 7. 纯 Xreal 端到端标签对比

预处理脚本默认合并 2026-08-18 和 2026-08-19 两个目录，并按 `7:1:2`
生成 train/val/test。稳定状态使用五秒窗口、一秒步长：

```bash
python Xreal_datasets/preprocess_xreal_imu.py \
  --stable-stride-seconds 1
```

三组实验均随机初始化 LLE/HLE，不加载 Ego checkpoint：

```text
four_class:  WORN / NOT_WORN / PUT_ON / TAKE_OFF
three_class: OTHERS / PUT_ON / TAKE_OFF
binary_event: NO_EVENT / EVENT
```

先搜索 HLE hidden size，再执行三种任务、三个 seed 的正式比较：

```bash
python scripts/run_xreal_end_to_end_comparison.py --stage hle-search
python scripts/run_xreal_end_to_end_comparison.py --stage formal
```

也可以连续完成两步：

```bash
python scripts/run_xreal_end_to_end_comparison.py --stage all
```

全部新权重和比较结果位于：

```text
artifacts/xreal_wear_detection/runs/xreal_end_to_end/
```

容量只根据三分类验证集选择；正式测试阈值也只根据各自验证集选择。连续稳定
录制中的重叠阳性窗口会合并后再计算 `false_alarms_per_hour`。
