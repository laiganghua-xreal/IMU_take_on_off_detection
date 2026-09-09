"""EgoCHARM 的低层编码器、高层分类器和完整层次模型。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SamePaddingConv1d(nn.Module):
    """显式补零，使不同 dilation 分支保持相同的时间长度。"""

    def __init__(self, input_channels, output_channels, kernel_size, dilation):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.convolution = nn.Conv1d(
            in_channels=input_channels,#这是输入的维度，1D卷积不是说输入的维度是1d的，是指在1个方向上（这里是时间）做卷积
            out_channels=output_channels,#这个指的是卷积核的个数，每个核都相当于一个特征提取器。
            kernel_size=kernel_size, #这给的是卷积方向上的size,实际的大小是（in_channels x kernal_size）
            dilation=dilation, #这是卷机方向上每个跳多少个数据
        )

    def forward(self, x):
        total_padding = self.dilation * (self.kernel_size - 1)
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding
        x = F.pad(x, (left_padding, right_padding))
        return self.convolution(x)


class ParallelDilatedBlock(nn.Module):
    """并行 dilation 卷积后拼接、归一化并降低时间分辨率。"""

    def __init__(
        self,
        input_channels,
        branch_channels,
        kernel_size,
        dilations,
        dropout,
    ):
        super().__init__()
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

        merged_channels = branch_channels * len(dilations)
        self.batch_normalization = nn.BatchNorm1d(merged_channels)
        self.pooling = nn.MaxPool1d(kernel_size=2, stride=2)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 每个分支: (B, C, T) -> (B, branch_channels, T)
        # 沿通道维拼接: num_branches * branch_channels -> F
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.batch_normalization(x)
        x = self.pooling(x)
        x = self.activation(x)
        return self.dropout(x)


class LowLevelEncoder(nn.Module):
    """把一秒六轴 IMU ``(B, 6, 50)`` 编码成 32 维运动向量。"""

    def __init__(
        self,
        input_channels=6,
        branch_channels=11,
        kernel_size=2,
        dilations=(1, 2, 4, 8, 16),
        number_of_blocks=3,
        hidden_size=32,
        dropout=0.2,
    ):
        super().__init__()
        if number_of_blocks <= 0:
            raise ValueError("number_of_blocks 必须大于零")

        merged_channels = branch_channels * len(dilations)
        self.blocks = nn.ModuleList()
        for block_index in range(number_of_blocks):
            block_input_channels = (
                input_channels if block_index == 0 else merged_channels
            )
            self.blocks.append(
                ParallelDilatedBlock(
                    block_input_channels,
                    branch_channels,
                    kernel_size,
                    dilations,
                    dropout,
                )
            )

        self.recurrent_layer = nn.GRU(
            input_size=merged_channels,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output_normalization = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        # x: (B, C, T)，论文默认 (B, 6, 50)。
        for block in self.blocks:
            x = block(x)

        # 默认三个卷积块后: (B, 55, 6)
        # Conv1d 布局转 GRU 布局: (B, F, T') -> (B, T', F)
        x = x.transpose(1, 2)
        # GRU 输出: (B, T', E)
        x, _ = self.recurrent_layer(x)
        # 最后时间步: (B, T', E) -> (B, E)
        return self.output_normalization(x[:, -1, :])


class HighLevelClassifier(nn.Module):
    """根据连续 30 个低层向量识别高层活动。"""

    def __init__(self, embedding_size=32, hidden_size=128, num_classes=7):
        super().__init__()
        self.recurrent_layer = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, embeddings):
        # embeddings: (B, S, E)
        recurrent_output, _ = self.recurrent_layer(embeddings)
        # recurrent_output: (B, S, H)
        last_time_step = recurrent_output[:, -1, :]
        # last_time_step: (B, H)
        # CrossEntropyLoss 直接接收 logits，因此这里不使用 softmax。
        logits = self.classifier(last_time_step)
        # logits: (B, K)
        return logits


class EgoCHARM(nn.Module):
    """组合低层与高层网络，输入形状为 ``(B, 30, 6, 50)``。"""

    def __init__(self, low_level_encoder, high_level_classifier):
        super().__init__()
        self.low_level_encoder = low_level_encoder
        self.high_level_classifier = high_level_classifier

    def encode_each_second(self, x):
        if x.ndim != 4:
            raise ValueError("EgoCHARM 输入必须是四维 (B, 秒, 通道, 采样点)")

        batch_size, number_of_seconds, number_of_channels, samples_per_second = x.shape
        # (B, S, C, T) -> (B * S, C, T)
        x = x.reshape(
            batch_size * number_of_seconds,
            number_of_channels,
            samples_per_second,
        )
        # LLE: (B * S, C, T) -> (B * S, E)
        x = self.low_level_encoder(x)
        # (B * S, E) -> (B, S, E)
        return x.reshape(batch_size, number_of_seconds, -1)

    def forward(self, x):
        embeddings = self.encode_each_second(x)
        # 高层输出: (B, K)
        return self.high_level_classifier(embeddings)


def build_model(model_config):
    """根据 YAML 的 model 区域创建完整 EgoCHARM。"""
    low_level_encoder = LowLevelEncoder(
        input_channels=int(model_config["input_channels"]),
        branch_channels=int(model_config["branch_channels"]),
        kernel_size=int(model_config["kernel_size"]),
        dilations=tuple(int(value) for value in model_config["dilations"]),
        number_of_blocks=int(model_config["number_of_blocks"]),
        hidden_size=int(model_config["low_level_hidden_size"]),
        dropout=float(model_config["dropout"]),
    )
    high_level_classifier = HighLevelClassifier(
        embedding_size=int(model_config["low_level_hidden_size"]),
        hidden_size=int(model_config["high_level_hidden_size"]),
        num_classes=int(model_config["number_of_classes"]),
    )
    return EgoCHARM(low_level_encoder, high_level_classifier)


def count_trainable_parameters(model):
    """统计需要梯度的模型参数数量。"""
    number_of_parameters = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            number_of_parameters += parameter.numel()
    return number_of_parameters
