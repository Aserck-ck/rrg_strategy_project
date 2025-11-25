import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_size = 256,dropout=0.2):
        super().__init__()

        # LSTM layers with Dropout
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size//2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size//2, hidden_size//4)
        self.fc2 = nn.Linear(hidden_size//4, output_dim)

        # Activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        # LSTM layers
        x=x.float()
        x, _ = self.lstm1(x)  # (batch, seq_len, 100)
        x = self.dropout1(x)

        # Only take last timestep (since return_sequences=False in 2nd LSTM)
        x, _ = self.lstm2(x)  # (batch, seq_len, 50)
        x = x[:, -1:, :]  # Keep only the last timestamp
        x = self.dropout2(x)

        # Dense layers
        x = self.relu(self.fc1(x))  # (batch, 1, 20)
        x = self.fc2(x)  # (batch, 1, 1)

        return x



class GRUModel(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_size=256, dropout=0.2, predict_head=1):
        super().__init__()
        self.output_dim = output_dim
        self.d_model = hidden_size
        self.predict_head = predict_head
        # GRU layers with Dropout
        self.gru1 = nn.GRU(input_size=input_dim, hidden_size=hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.gru2 = nn.GRU(input_size=hidden_size, hidden_size=hidden_size//2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)

        self.norm = nn.BatchNorm1d(hidden_size//2)
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size//2, hidden_size//4)
        self.fc2 = nn.Linear(hidden_size//4, output_dim)

        # Activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.float()
        # GRU layers
        x, _ = self.gru1(x)  # (batch, seq_len, 100)
        x = self.dropout1(x)

        # Only take last timestep (since return_sequences=False in 2nd GRU)
        x, _ = self.gru2(x)  # (batch, seq_len, 50)
        x = x[:, -self.predict_head:, :]  # Keep only the last out_len timestamp
        x = self.dropout2(x)

        # original_shape = x.shape
        # x = x.reshape(-1, original_shape[-1])
        # x = self.norm(x)
        # x = x.reshape(original_shape)


        # Dense layers
        x = self.relu(self.fc1(x))  # (batch, out_len, 20)
        x = self.fc2(x)  # (batch, out_len, output_dim)

        return x

class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_dim=1,output_dim=1, hidden_size=128, num_layers=3, dropout=0.2):
        super().__init__()

        # 增强的LSTM层
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        # 第一层
        self.lstm_layers.append(
            nn.LSTM(input_size=input_dim, hidden_size=hidden_size,
                    batch_first=True, bidirectional=False)
        )
        self.dropout_layers.append(nn.Dropout(dropout))

        # 后续层 - 逐步减小隐藏层大小
        prev_size = hidden_size
        for i in range(1, num_layers):
            current_size = max(prev_size // (2 ** i), 32)  # 逐步减半，最小32

            self.lstm_layers.append(
                nn.LSTM(input_size=prev_size, hidden_size=current_size,
                        batch_first=True, bidirectional=False)
            )
            prev_size = current_size
            self.dropout_layers.append(nn.Dropout(dropout))

        self.final_hidden_size = current_size

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(self.final_hidden_size, self.final_hidden_size // 2),
            nn.Tanh(),
            nn.Linear(self.final_hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(self.final_hidden_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(16, output_dim)
        )

        # 初始化权重
        # self._initialize_weights()

    # def _initialize_weights(self):
    #     for name, param in self.named_parameters():
    #         if 'weight' in name and 'lstm' in name:
    #             nn.init.orthogonal_(param)
    #         elif 'weight' in name and 'fc' in name:
    #             nn.init.kaiming_normal_(param)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 多层LSTM
        for lstm, dropout in zip(self.lstm_layers, self.dropout_layers):
            x, (h_n, c_n) = lstm(x)
            x = dropout(x)

        # 注意力机制
        attention_weights = self.attention(x)  # (batch, seq_len, 1)
        weighted_output = torch.sum(x * attention_weights, dim=1)  # (batch, hidden_size)

        # 全连接层
        output = self.fc_layers(weighted_output)

        return output.unsqueeze(1)