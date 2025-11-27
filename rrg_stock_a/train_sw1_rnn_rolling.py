import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
import os
import gc

# 从您的项目中导入所需模块
from models.RNN import LSTMModel, GRUModel
from market_data.sw_level1.sw_level1_functions import get_sw_level1_codes
from market_data.sw_level1.sw_level1_jdk_functions import get_sw_level1_ratio, get_sw_level1_momentum
from models.module import set_seed

# --- 1. 配置参数 ---
class Config():
    def __init__(self):
        self.model_type = 'GRU'  # 在这里切换 'LSTM' 或 'GRU'
        self.input_size = 2       # 输入特征数 (ratio, momentum)
        self.output_size = 2      # 输出特征数 (pred_ratio, pred_momentum)
        self.hidden_size = 128    # 增大了模型容量以适应更复杂的数据
        self.num_layers = 2
        self.dropout = 0.2
        self.time_window = 52
        self.predict_window = 1
        self.epochs = 20          # 减少epochs，因为数据集更大了
        self.batch_size = 256     # 增大了批次大小
        self.lr = 1e-4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 模型现在是单个文件，而不是目录
        self.model_save_path = f'd:/files/qt/guangfa/rrg_stock_a/lstm_models/single_general_{self.model_type.lower()}.pth'
        self.results_save_dir = 'd:/files/qt/guangfa/rrg_stock_a/analyse/'

def run_rolling_rnn_autoregressive_cross_section(config, train_window=500, step=20):
    all_ratio_df = get_sw_level1_ratio(frequency='weekly').set_index('trade_date').drop(columns=['000300.SH'])
    all_momentum_df = get_sw_level1_momentum(frequency='weekly').set_index('trade_date').drop(columns=['000300.SH'])
    common_index = all_ratio_df.index.intersection(all_momentum_df.index)
    all_ratio_df = all_ratio_df.loc[common_index]
    all_momentum_df = all_momentum_df.loc[common_index]
    sector_codes = get_sw_level1_codes()
    dates = all_ratio_df.index

    pred_ratios_df = pd.DataFrame(index=all_ratio_df.index, columns=all_ratio_df.columns)
    pred_momentums_df = pd.DataFrame(index=all_momentum_df.index, columns=all_momentum_df.columns)
    if config.model_type == 'LSTM':
        model = LSTMModel(config.input_size, config.output_size, config.hidden_size, config.dropout, 1).to(config.device)
    else:
        model = GRUModel(config.input_size, config.output_size, config.hidden_size, config.dropout, 1).to(config.device)
    t = (len(dates)-train_window) % step
    round_num = 1
    while t + train_window + step <= len(dates):
        train_start = dates[t]
        train_end = dates[t + train_window - 1]
        pred_start = dates[t + train_window]
        pred_end = dates[t + train_window + step - 1]
        print(f"\n=== 滚动训练轮次 {round_num} ===")
        print(f"训练区间: {train_start} ~ {train_end}")
        print(f"预测区间: {pred_start} ~ {pred_end}")

        # 构造训练集（所有板块拼接）
        X_train, Y_train = [], []
        for code in sector_codes:
            if code not in all_ratio_df.columns: continue
            ratio_data = all_ratio_df[code].values
            momentum_data = all_momentum_df[code].values
            for i in range(t, t + train_window - config.time_window):
                ratio_mean = ratio_data[i:(i + config.time_window)].mean()
                ratio_std = ratio_data[i:(i + config.time_window)].std()
                momentum_mean = momentum_data[i:(i + config.time_window)].mean()
                momentum_std = momentum_data[i:(i + config.time_window)].std()
                ratio_norm = (ratio_data[i:(i + config.time_window)] - ratio_mean) / (ratio_std + 1e-8)
                momentum_norm = (momentum_data[i:(i + config.time_window)] - momentum_mean) / (momentum_std + 1e-8)
                feature = np.column_stack((ratio_norm, momentum_norm))
                X_train.append(feature)
                target_ratio = ratio_data[i + config.time_window]
                target_momentum = momentum_data[i + config.time_window]
                target_ratio = (target_ratio - ratio_mean) / (ratio_std + 1e-8)
                target_momentum = (target_momentum - momentum_mean) / (momentum_std + 1e-8)
                Y_train.append([target_ratio, target_momentum])
        X_train = torch.FloatTensor(np.array(X_train))
        Y_train = torch.FloatTensor(np.array(Y_train))

        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

        # 初始化模型
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.MSELoss()

        # 训练模型
        print(f"训练样本数: {len(X_train)}")
        best_loss = float('inf')
        current_epochs = int(config.epochs*(1 - t/(2*len(dates))))  # 每轮增加10%的训练周期
        for epoch in range(current_epochs):
            model.train()
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x.to(config.device)).squeeze(1)
                loss = criterion(outputs, batch_y.to(config.device))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
            avg_loss = epoch_loss / len(train_loader.dataset)
            if avg_loss < best_loss:
                best_loss = avg_loss
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == current_epochs - 1:
                print(f"Epoch {epoch+1}/{current_epochs} | 当前训练损失: {avg_loss:.6f}")

        print(f"本轮训练最优损失: {best_loss:.6f}")

        # 递推预测 step 个点（所有板块一起）
        model.eval()
        with torch.no_grad():
            for code in sector_codes:
                if code not in all_ratio_df.columns: continue
                ratio_data = all_ratio_df[code].values
                momentum_data = all_momentum_df[code].values
                # 用最后一个训练窗口做递推预测
                start_idx = t + train_window - config.time_window
                input_ratio = ratio_data[start_idx : start_idx + config.time_window].copy()
                input_momentum = momentum_data[start_idx : start_idx + config.time_window].copy()
                for s in range(step):
                    ratio_mean = input_ratio.mean()
                    ratio_std = input_ratio.std()
                    momentum_mean = input_momentum.mean()
                    momentum_std = input_momentum.std()
                    ratio_norm = (input_ratio - ratio_mean) / (ratio_std + 1e-8)
                    momentum_norm = (input_momentum - momentum_mean) / (momentum_std + 1e-8)
                    feature = np.column_stack((ratio_norm, momentum_norm))
                    input_tensor = torch.FloatTensor(feature).unsqueeze(0).to(config.device)
                    pred = model(input_tensor).squeeze(0).cpu().numpy()  # shape: (output_size,)
                    # 反标准化
                    pred_ratio = pred[0][0] * ratio_std + ratio_mean
                    pred_momentum = pred[0][1] * momentum_std + momentum_mean
                    pred_date = dates[start_idx + config.time_window + s]
                    pred_ratios_df.loc[pred_date, code] = pred_ratio
                    pred_momentums_df.loc[pred_date, code] = pred_momentum
                    # 递推：将预测结果加入输入序列
                    input_ratio = np.append(input_ratio[1:], pred_ratio)
                    input_momentum = np.append(input_momentum[1:], pred_momentum)

        t += step  # 窗口推进
        round_num += 1

    # 保存结果
    x_pred_filename = os.path.join(config.results_save_dir, f'x_pred_autoregressive_{config.hidden_size}_{config.model_type.lower()}_rolling.csv')
    y_pred_filename = os.path.join(config.results_save_dir, f'y_pred_autoregressive_{config.hidden_size}_{config.model_type.lower()}_rolling.csv')
    pred_ratios_df.dropna(how='all', inplace=True)
    pred_momentums_df.dropna(how='all', inplace=True)
    pred_ratios_df.to_csv(x_pred_filename)
    pred_momentums_df.to_csv(y_pred_filename)
    print(f"\n递推预测文件已保存:\n{x_pred_filename}\n{y_pred_filename}")

# ...existing code...
if __name__ == "__main__":
    set_seed(1000)
    cfg = Config()
    run_rolling_rnn_autoregressive_cross_section(cfg, train_window=500, step=20)