# EgoCHARM Ego-Exo4D 数据子集

本目录用于准备 EgoCHARM 的 Ego-Exo4D V2 方法级复现数据。由于完整的论文相关无图像流 VRS 数据约为 857.11 GB，本项目在 280 GB 目标预算内，按“活动类别 × Ego-Exo4D 官方 split”分层，并以参与者为最小抽样单位生成子集。

生成后的子集使用官方 `train`、`val`、`test` 划分。同一参与者不会跨 split，避免参与者数据泄漏。抽样种子固定为 42。

## 子集概览

> 当前下载在 242 GiB 左右停止。训练代码默认使用
> `metadata/egocharm_available_selection.json`，其中仅保留 1159 个已完整下载、
> 文件大小匹配的 takes（251.79 GB）；原始 1201-take 清单仍完整保存在
> `metadata/egocharm_official_selection.json`，不会被覆盖。

当前可训练清单的官方划分如下：

| 官方 split | Takes | 参与者 | 时长 | 大小 |
|---|---:|---:|---:|---:|
| Train | 702 | 129 | 32.33 小时 | 147.60 GB |
| Val | 203 | 36 | 9.00 小时 | 41.00 GB |
| Test | 254 | 50 | 13.88 小时 | 63.19 GB |
| **合计** | **1159** | **215** | **54.20 小时** | **251.79 GB** |

下面的 1201-take 表描述最初计划下载的完整抽样清单：

| 官方 split | Takes | 参与者 | 时长 | 大小 |
|---|---:|---:|---:|---:|
| Train | 727 | 133 | 35.33 小时 | 161.31 GB |
| Val | 213 | 37 | 10.30 小时 | 46.97 GB |
| Test | 261 | 50 | 14.66 小时 | 66.82 GB |
| **合计** | **1201** | **220** | **60.28 小时** | **275.10 GB** |

类别分布如下：

| 类别 | Train | Val | Test | 合计 | 大小 |
|---|---:|---:|---:|---:|---:|
| Basketball | 161 | 41 | 41 | 243 | 19.07 GB |
| Soccer | 52 | 9 | 20 | 81 | 12.53 GB |
| Dance | 115 | 45 | 56 | 216 | 23.19 GB |
| Rock Climbing | 122 | 52 | 59 | 233 | 19.19 GB |
| Cooking | 155 | 39 | 49 | 243 | 142.68 GB |
| Bike Repair | 63 | 15 | 13 | 91 | 20.79 GB |
| Music | 59 | 12 | 23 | 94 | 37.66 GB |

Cooking 占比较高是原始数据分布的结果。训练时应使用类别加权交叉熵，与 EgoCHARM 论文的处理方式一致。

## 下载内容

每个选中的 take 下载一个文件。Aria 设备编号由原始 capture 决定，通常为 `aria01`，部分 take 为 `aria02`：

```text
takes/<take_name>/<aria_device>_noimagestreams.vrs
```

无图像流 VRS 包含论文需要的时间戳和 Aria IMU 数据：

- `imu-left`：左侧 IMU，论文使用该传感器；
- 三轴加速度和三轴角速度；
- IMU 配置与校准信息；
- `imu-right`、音频以及其他可能存在的非图像传感器流，复现时不使用。

这些文件不包含 RGB、SLAM 或眼动相机的图像载荷，也不会下载 GoPro 外部视频、MP4、点云、轨迹或眼动数据。由于音频与 IMU 封装在同一个无图像流 VRS 中，官方下载器不能只下载 IMU；预处理阶段只读取左侧 IMU 并忽略其他流。

## 目录结构

准备及下载完成后，主要目录如下：

```text
ego-exoD4_dataset/
├── README.md
├── generate_egocharm_official_selection.py
├── metadata/
│   ├── takes.json
│   ├── participants.json
│   ├── take_vrs_noimagestream_manifest.json
│   └── egocharm_official_selection.json
└── takes/
    └── <take_name>/
        └── <aria_device>_noimagestreams.vrs
```

`egocharm_official_selection.json` 是本项目的数据索引，每条记录包含：

- `take_uid` 和 `take_name`；
- 活动类别；
- `participant_uid`；
- 官方 `train`、`val` 或 `test` split；
- take 时长和 VRS 文件大小；
- 下载后的相对路径。

## 1. 获取无图像流 VRS manifest

如果 manifest 尚不存在，先运行：

```bash
aws s3 cp \
  s3://ego4d-consortium-sharing/egoexo-public/v2/take_vrs_noimagestream/manifest.json \
  ./ego-exoD4_dataset/metadata/take_vrs_noimagestream_manifest.json
```

这一步只下载约 2.4 MB 的文件清单，不下载传感器数据。

## 2. 生成官方划分子集清单

```bash
python3 ./ego-exoD4_dataset/generate_egocharm_official_selection.py
```

默认目标预算为 280 GB，输出文件为：

```text
./ego-exoD4_dataset/metadata/egocharm_official_selection.json
```

如需改变预算，可使用 `--target-gb`：

```bash
python3 ./ego-exoD4_dataset/generate_egocharm_official_selection.py \
  --target-gb 250
```

## 3. 下载选中的无图像流 VRS

```bash
mapfile -t EGOCHARM_UIDS < <(
  python3 ./ego-exoD4_dataset/generate_egocharm_official_selection.py \
    --print-uids
)

egoexo \
  -o ./ego-exoD4_dataset \
  --release v2 \
  --parts take_vrs_noimagestream \
  --splits train val test \
  --uids "${EGOCHARM_UIDS[@]}"
```

下载器会在实际传输前显示文件数量和总大小。确认约为 275.10 GB 后再继续。

## 筛选规则

生成脚本只保留满足以下条件的 take：

1. 类别属于论文使用的七个 Ego-Exo4D 场景；
2. 有有效的 `participant_uid`；
3. 存在 trimmed、无图像流 VRS；
4. metadata 标记为 validated 且没有被 dropped；
5. take 时长至少为 30 秒；
6. 恰好属于一个官方 `train`、`val` 或 `test` split。

脚本根据各“类别 × split”在完整候选集中的字节占比分配预算，然后对参与者进行固定种子排序。一个参与者在同一类别和 split 下的所有合格 takes 会整体保留或整体排除，不会为满足预算而拆散。

实际大小略低于 280 GB，是因为脚本不会拆分参与者数据组。

## 验证生成脚本

从项目根目录运行：

```bash
python3 -m unittest \
  ego-exoD4_dataset/tests/test_generate_egocharm_official_selection.py
```
