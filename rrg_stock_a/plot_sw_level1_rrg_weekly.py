from matplotlib import pyplot as plt
import torch
from torch import nn
from market_data.sw_level1.sw_level1_jdk_functions import get_sw_level1_ratio, get_sw_level1_momentum
from market_data.sw_level1.sw_level1_functions import get_sw_level1_info
from market_data.sw_level1.sw_level1_crowdedness_functions import get_sw_level1_crowdedness
import numpy as np
from datetime import timedelta
import pandas as pd
from joblib import load
import subprocess
import os

# matplotlib中文环境配置
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    # Get filepath
    file_path = os.path.abspath(os.path.dirname(__file__))
    sw_level1_updater_path = \
        os.path.join(file_path, 'market_data', 'update_jdk_rs_crowdedness_sw_level1.py')
    subprocess.run(['python', sw_level1_updater_path])

    # Define the predictive model
    class LSTMModel(nn.Module):
        def __init__(self, input_dim=1):
            super().__init__()
            
            # LSTM layers with Dropout
            self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=100, batch_first=True)
            self.dropout1 = nn.Dropout(0.2)
            
            self.lstm2 = nn.LSTM(input_size=100, hidden_size=50, batch_first=True)
            self.dropout2 = nn.Dropout(0.2)
            
            # Fully connected layers
            self.fc1 = nn.Linear(50, 20)
            self.fc2 = nn.Linear(20, 1)
            
            # Activation functions
            self.relu = nn.ReLU()
        
        def forward(self, x):
            # LSTM layers
            x, _ = self.lstm1(x)  # (batch, seq_len, 100)
            x = self.dropout1(x)
            
            # Only take last timestep (since return_sequences=False in 2nd LSTM)
            x, _ = self.lstm2(x)  # (batch, seq_len, 50)
            x = x[:, -1, :] # Keep only the last timestamp
            x = torch.unsqueeze(x, dim=1)
            x = self.dropout2(x)
            
            # Dense layers
            x = self.relu(self.fc1(x))  # (batch, 1, 20)
            x = self.fc2(x)  # (batch, 1, 1)
            
            return x
        
    # Load saved models300
    ratio_model = torch.load(os.path.join(file_path, 'lstm_models', 'sw_level1_ratio_weekly.pth'), weights_only=False)
    momentum_model = torch.load(os.path.join(file_path, 'lstm_models', 'sw_level1_momentum_weekly.pth'), weights_only=False)

    ratio_model.eval()
    momentum_model.eval()

    # Get ratio and momentum data
    ratio_df = get_sw_level1_ratio(ts_code='all', frequency='weekly').set_index('trade_date')
    momentum_df = get_sw_level1_momentum(ts_code='all', frequency='weekly').set_index('trade_date')

    # Get crowdedness
    crowdedness_df = get_sw_level1_crowdedness(ts_code='all', frequency='weekly').set_index('trade_date')

    # Remove the trivial SPX column
    ratio_df = ratio_df[ratio_df.columns[1:]]
    momentum_df = momentum_df[momentum_df.columns[1:]]

    # Load scalers
    scaler_ratio = load(os.path.join(file_path, 'lstm_models', 'sw_level1_ratio_scaler_weekly.joblib'))
    scaler_momentum = load(os.path.join(file_path, 'lstm_models', 'sw_level1_momentum_scaler_weekly.joblib'))

    # Define the length of tails in the RRG plot (including the predicted data point)
    tail_length = 6

    # Define time window
    time_window = 60
    # Input ratio and momentum to predict the next data point
    ratio_input = np.array(ratio_df.iloc[-time_window:])
    momentum_input = np.array(momentum_df.iloc[-time_window:])
    # Scale the input
    ratio_input_scaled = scaler_ratio.transform(ratio_input.reshape(-1, 1))
    momentum_input_scaled = scaler_momentum.transform(momentum_input.reshape(-1, 1))
    ratio_input_scaled = ratio_input_scaled.reshape(ratio_input.shape[0], ratio_input.shape[1])
    momentum_input_scaled = momentum_input_scaled.reshape(momentum_input.shape[0], momentum_input.shape[1])

    ratio_input_scaled = torch.Tensor(ratio_input_scaled)
    momentum_input_scaled = torch.Tensor(momentum_input_scaled)

    # Make predictions on ratios (one etf by one)
    ratio_pred_scaled = np.zeros((1, ratio_input_scaled.shape[-1]))
    for i in range(ratio_input_scaled.shape[-1]):
        input = ratio_input_scaled[:, i]
        # Add dimensions so that the shape matches the input to the trained model
        input = torch.unsqueeze(input, dim=0)
        input = torch.unsqueeze(input, dim=-1)

        # Make inference
        y_pred = ratio_model(input)
        ratio_pred_scaled[0, i] = y_pred[0, 0, 0].detach().numpy()

    # Make predictions on momenta (one etf by one)
    momentum_pred_scaled = np.zeros((1, momentum_input_scaled.shape[-1]))
    for i in range(momentum_input_scaled.shape[-1]):
        input = momentum_input_scaled[:, i]
        # Add dimensions so that the shape matches the input to the trained model
        input = torch.unsqueeze(input, dim=0)
        input = torch.unsqueeze(input, dim=-1)

        # Make inference
        y_pred = momentum_model(input)
        momentum_pred_scaled[0, i] = y_pred[0, 0, 0].detach().numpy()

    # Inverse transform the predicted values
    ratio_pred = scaler_ratio.inverse_transform(ratio_pred_scaled)
    momentum_pred = scaler_momentum.inverse_transform(momentum_pred_scaled)

    # Calculate the date for prediction
    last_date = ratio_df.index[-1]
    pred_date = last_date + timedelta(weeks=1)

    # Concat predicted results to the original dfs
    ratio_pred_df = pd.DataFrame(ratio_pred, index=[pred_date], columns=ratio_df.columns)
    momentum_pred_df = pd.DataFrame(momentum_pred, index=[pred_date], columns=momentum_df.columns)
    ratio_all_df = pd.concat((ratio_df, ratio_pred_df))
    momentum_all_df = pd.concat((momentum_df, momentum_pred_df))

    etf_info_df = get_sw_level1_info()

    # Make the plot and save the figure
    x_cut = 100
    y_cut = 100
    plt.figure(figsize=(20, 20))
    plt.subplots_adjust(top=0.9, bottom=0, right=1, left=0, hspace=0, wspace=0)
    for etf_code in momentum_df.columns[:]:
        # Get etf name corresponding to the etf code
        etf_name = etf_info_df[etf_info_df['ts_code'] == etf_code]['industry_name'].values[0]
        dot_position = etf_code.index('.')
        plt.text(ratio_all_df[etf_code].iloc[-1], momentum_all_df[etf_code].iloc[-1], \
                etf_name, fontsize='xx-large')
        line1, = plt.plot(ratio_all_df[etf_code].iloc[-tail_length:-1], \
                momentum_all_df[etf_code].iloc[-tail_length:-1], '.-', linewidth=2, markersize=15)
        
        # Plot crowdedness
        for date, crowdedness in crowdedness_df.iloc[-tail_length+1:].iterrows():
            if crowdedness[etf_code] == 1:
                plt.plot(ratio_all_df[etf_code].loc[date], momentum_all_df[etf_code].loc[date], \
                        'k.', markersize=15)

        # Plot predicted point
        plt.plot(ratio_all_df[etf_code].iloc[-2:], \
                momentum_all_df[etf_code].iloc[-2:], '.--', \
                color = line1.get_color(), markerfacecolor='none', linewidth=2, markersize=15)

        for i, (x, y) in enumerate(zip(ratio_all_df[etf_code].iloc[-tail_length:-1], momentum_all_df[etf_code].iloc[-tail_length:-1])):
            plt.arrow(x, y, \
                    (ratio_all_df[etf_code].iloc[-tail_length+1+i] - ratio_all_df[etf_code].iloc[-tail_length+i]) / 2, \
                    (momentum_all_df[etf_code].iloc[-tail_length+1+i] - momentum_all_df[etf_code].iloc[-tail_length+i]) / 2, \
                    head_width = 0.15, linewidth=0, fill=True, color='black')

    axes = plt.gca()
    y_b, y_t = axes.get_ylim()
    x_b, x_t = axes.get_xlim()

    plt.fill_between([x_b, x_cut], y_b, y_cut, alpha=0.2, facecolor='r')
    plt.fill_between([x_cut, x_t], y_b, y_cut, alpha=0.2, facecolor='y')
    plt.fill_between([x_b, x_cut], y_cut, y_t, alpha=0.2, facecolor='b')
    plt.fill_between([x_cut, x_t], y_cut, y_t, alpha=0.2, facecolor='g')

    plt.xlabel('JDK RS Ratio')
    plt.xlabel('JDK RS Momentum')
    plt.title('Date: '+ratio_all_df.index[-2].strftime('%Y-%m-%d')+'; Frequency: weekly', fontsize=30)
    plt.axis('off')

    plt.savefig(os.path.join(file_path, 'graphs', 'rrg_weekly', \
                            'sw_level1_'+ratio_all_df.index[-2].strftime('%Y-%m-%d')+'.png'), dpi=300)
