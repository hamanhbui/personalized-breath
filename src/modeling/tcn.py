import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp = Chomp1d(padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_dilateds, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** num_dilateds[i]
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Audio_TCN(nn.Module):
    def __init__(self, no_outer, one_vs_all = False):
        super(Audio_TCN, self).__init__()
        self.audio_TCN = nn.Sequential(
            TemporalConvNet(num_inputs=20, num_dilateds = list(range(0, 4)), num_channels=[32]*4, kernel_size=16, dropout=0.2),
            TemporalConvNet(num_inputs=32, num_dilateds = list(range(0, 4)), num_channels=[128]*4, kernel_size=16, dropout=0.2)
        )
        if one_vs_all:
            self.fc = nn.Linear(in_features=128, out_features=2)
        else:
            self.fc = nn.Linear(in_features=128, out_features=20-no_outer)

    def forward(self, audio_features):
        audio_features = self.audio_TCN(audio_features)
        out = self.fc(audio_features[:, :, -1])
        return out

class Acce_Gyro_TCN(nn.Module):
    def __init__(self, no_outer, one_vs_all = False):
        super(Acce_Gyro_TCN, self).__init__()
        self.acce_gyro_TCN = nn.Sequential(
            TemporalConvNet(num_inputs=6, num_dilateds = list(range(0, 4)), num_channels=[32]*4, kernel_size=16, dropout=0.2),
            TemporalConvNet(num_inputs=32, num_dilateds = list(range(0, 4)), num_channels=[128]*4, kernel_size=16, dropout=0.2)
        )
        if one_vs_all:
            self.fc = nn.Linear(in_features=128, out_features=2)
        else:
            self.fc = nn.Linear(in_features=128, out_features=20-no_outer)

    def forward(self, acce_gyro_features):
        acce_gyro_features = self.acce_gyro_TCN(acce_gyro_features)
        out = self.fc(acce_gyro_features[:, :, -1])
        return out

class Multimodality_TCN(nn.Module):
    def __init__(self, no_outer, one_vs_all = False):
        super(Multimodality_TCN, self).__init__()
        self.audio_TCN = TemporalConvNet(num_inputs=20, num_dilateds = list(range(0, 4)), num_channels=[32]*4, kernel_size=16, dropout=0.2)
        self.acce_gyro_TCN = TemporalConvNet(num_inputs=6, num_dilateds = list(range(0, 4)), num_channels=[32]*4, kernel_size=16, dropout=0.2)
        self.sharing_TCN = TemporalConvNet(num_inputs=64, num_dilateds = list(range(0, 4)), num_channels=[128]*4, kernel_size=16, dropout=0.2)
        if one_vs_all:
            self.fc = nn.Linear(in_features=128, out_features=2)
        else:
            self.fc = nn.Linear(in_features=128, out_features=20-no_outer)

    def forward(self, acce_gyro_features, audio_features):
        acce_gyro_features=self.acce_gyro_TCN(acce_gyro_features)
        audio_features=self.audio_TCN(audio_features)
        merged_features=torch.cat((acce_gyro_features,audio_features),dim=1)
        merged_features = self.sharing_TCN(merged_features)
        out = self.fc(merged_features[:, :, -1])
        return out