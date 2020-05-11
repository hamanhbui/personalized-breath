import numpy as np
import torch
import torch.nn as nn

class Audio_CNN_LSTM(nn.Module):
    def __init__(self,no_outer):
        super(Audio_CNN_LSTM, self).__init__()
        self.audio_layers = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=128, num_layers=1)
        self.fc = nn.Linear(in_features=128, out_features=22-no_outer)

    def forward(self, audio_features):
        audio_features = self.audio_layers(audio_features)
        audio_features = audio_features.permute(2,0,1).contiguous()

        audio_features,_ = self.lstm(audio_features)
        audio_features = audio_features[-1]

        out = self.fc(audio_features)
        return out

class Acce_Gyro_CNN_LSTM(nn.Module):
    def __init__(self,no_outer):
        super(Acce_Gyro_CNN_LSTM, self).__init__()
        self.acce_gyro_layers = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=32, kernel_size=8),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=128, num_layers=1)
        self.fc = nn.Linear(in_features=128, out_features=22-no_outer)

    def forward(self, acce_gyro_features):
        acce_gyro_features = self.acce_gyro_layers(acce_gyro_features)
        acce_gyro_features = acce_gyro_features.permute(2,0,1).contiguous()

        acce_gyro_features,_ = self.lstm(acce_gyro_features)
        acce_gyro_features = acce_gyro_features[-1]

        out=self.fc(acce_gyro_features)
        return out

class Multimodality_CNN_LSTM(nn.Module):
    def __init__(self,no_outer):
        super(Multimodality_CNN_LSTM, self).__init__()
        self.audio_layers = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8),
            nn.ReLU()
        )
        self.acce_gyro_layers = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=32, kernel_size=8),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=1)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features=128, out_features=22-no_outer)

    def forward(self,acce_gyro_features,audio_features):
        acce_gyro_features=self.acce_gyro_layers(acce_gyro_features)
        audio_features=self.audio_layers(audio_features)
        acce_gyro_features=acce_gyro_features.permute(2,0,1).contiguous()
        audio_features=audio_features.permute(2,0,1).contiguous()

        merged_features=torch.cat((acce_gyro_features,audio_features),dim=2)
        merged_features,_ = self.lstm(merged_features)
        merged_features = merged_features[-1]
        
        merged_features = self.dropout(merged_features)
        out = self.fc(merged_features)
        return out
