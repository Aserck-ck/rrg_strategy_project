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
        self.hidden_size = 512    # 增大了模型容量以适应更复杂的数据
        self.num_layers = 2
        self.dropout = 0.2
        self.time_window = 52
        self.predict_window = 2
        self.epochs = 20          # 减少epochs，因为数据集更大了
        self.batch_size = 256     # 增大了批次大小
        self.lr = 1e-4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 模型现在是单个文件，而不是目录
        self.model_save_path = f'd:/files/qt/guangfa/rrg_stock_a/lstm_models/single_general_{self.model_type.lower()}.pth'
        self.results_save_dir = 'd:/files/qt/guangfa/rrg_stock_a/analyse/'

def create_pooled_dataset(config):
    """
    将所有板块的数据序列汇集到一个大数据集中。
    """
    print("--- 正在创建数据池 (Data Pooling) ---")
    all_ratio_df = get_sw_level1_ratio(frequency='weekly').set_index('trade_date').drop(columns=['000300.SH'])
    all_momentum_df = get_sw_level1_momentum(frequency='weekly').set_index('trade_date').drop(columns=['000300.SH'])
    common_index = all_ratio_df.index.intersection(all_momentum_df.index)
    # 使用共同的索引重新对齐DataFrame，确保它们的行数和日期完全一致
    all_ratio_df = all_ratio_df.loc[common_index]
    all_momentum_df = all_momentum_df.loc[common_index]
    print(f"数据对齐完成，共同的日期数量: {len(common_index)}")
    sector_codes = get_sw_level1_codes()

    all_X, all_Y = [], []

    for code in tqdm(sector_codes, desc="处理板块数据"):
        if code not in all_ratio_df.columns: continue
        
        ratio_data = all_ratio_df[code].values
        momentum_data = all_momentum_df[code].values

        # 标准化整个序列
        


        for i in range(min(len(ratio_data), len(momentum_data)) - config.time_window - config.predict_window + 1):
            ratio_mean, ratio_std = ratio_data[i:(i + config.time_window)].mean(), ratio_data[i:(i + config.time_window)].std()
            momentum_mean, momentum_std = momentum_data[i:(i + config.time_window)].mean(), momentum_data[i:(i + config.time_window)].std()
            ratio_norm = (ratio_data[i:(i + config.time_window)] - ratio_mean) / (ratio_std + 1e-8)
            momentum_norm = (momentum_data[i:(i + config.time_window)] - momentum_mean) / (momentum_std + 1e-8)
            feature = np.column_stack((
                ratio_norm,
                momentum_norm
            ))
            all_X.append(feature)
            
            target_ratio = ratio_data[(i + config.time_window) :(i + config.time_window + config.predict_window)]
            target_momentum = momentum_data[(i + config.time_window):(i + config.time_window + config.predict_window)]
            target_ratio = (target_ratio - ratio_mean) / (ratio_std + 1e-8) 
            target_momentum = (target_momentum - momentum_mean) / (momentum_std + 1e-8)
            res = np.column_stack((
                target_ratio,
                target_momentum
            ))
            all_Y.append(res)

    return torch.FloatTensor(np.array(all_X)), torch.FloatTensor(np.array(all_Y))

def train_single_general_model(config):
    """
    训练一个能泛化预测所有板块的通用RNN模型。
    """
    print(f"====== 阶段一: 训练单一通用 {config.model_type} 模型 ======")
    print(f"使用设备: {config.device}")

    # 1. 创建合并后的数据集
    X, y = create_pooled_dataset(config)
    dataset = TensorDataset(X, y)
    train_size = int(0.6 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    print(f"数据集创建完成。总样本数: {len(dataset)}, 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

    # 2. 初始化单一模型
    if config.model_type == 'LSTM':
        model = LSTMModel(config.input_size, config.output_size, config.hidden_size, config.dropout, config.predict_window).to(config.device)
    else:
        model = GRUModel(config.input_size, config.output_size, config.hidden_size, config.dropout, config.predict_window).to(config.device)
        
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    # 3. 训练循环
    best_val_loss = float('inf')
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        for features, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [训练]"):
            features, targets = features.to(config.device), targets.to(config.device)
            optimizer.zero_grad()
            outputs = model(features)
            # outputs = outputs.squeeze(1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(config.device), targets.to(config.device)
                outputs = model(features)
                # outputs = outputs.squeeze(1)
                val_loss += criterion(outputs, targets).item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} | 平均训练损失: {train_loss / len(train_loader):.6f} | 平均验证损失: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.model_save_path)
            print(f"验证损失改善，模型已保存至: {config.model_save_path}")

    print("\n训练完成！")

def run_prediction_with_general_model(config):
    """使用单一通用模型为所有板块生成多步预测文件"""
    print(f"\n====== 阶段二: 使用单一通用 {config.model_type} 模型生成多步预测文件 ======")
    
    # 1. 加载训练好的单一模型
    if not os.path.exists(config.model_save_path):
        print(f"错误: 未找到模型文件 {config.model_save_path}。请先运行训练。")
        return

    if config.model_type == 'LSTM':
        model = LSTMModel(config.input_size, config.output_size, config.hidden_size, config.dropout, config.predict_window).to(config.device)
    else:
        model = GRUModel(config.input_size, config.output_size, config.hidden_size, config.dropout, config.predict_window).to(config.device)
    model.load_state_dict(torch.load(config.model_save_path))
    model.eval()
    print("单一通用模型加载成功。")

    # 2. 加载全量数据并生成预测
    all_ratio_df = get_sw_level1_ratio(frequency='weekly').set_index('trade_date').drop(columns=['000300.SH'])
    all_momentum_df = get_sw_level1_momentum(frequency='weekly').set_index('trade_date').drop(columns=['000300.SH'])

    common_index = all_ratio_df.index.intersection(all_momentum_df.index)
    all_ratio_df = all_ratio_df.loc[common_index]
    all_momentum_df = all_momentum_df.loc[common_index]
    sector_codes = get_sw_level1_codes()

    # 新建每步的预测DataFrame列表
    pred_ratios_list = [pd.DataFrame(index=all_ratio_df.index, columns=all_ratio_df.columns) for _ in range(config.predict_window)]
    pred_momentums_list = [pd.DataFrame(index=all_momentum_df.index, columns=all_momentum_df.columns) for _ in range(config.predict_window)]

    with torch.no_grad():
        for code in tqdm(sector_codes, desc="为所有板块生成多步预测"):
            if code not in all_ratio_df.columns: continue

            ratio_series = all_ratio_df[code]
            momentum_series = all_momentum_df[code]
            
            
            for i in range(len(momentum_series) - config.time_window - config.predict_window + 1):
                # 预测窗口内每一步的日期
                predict_dates = momentum_series.index[i + config.time_window : i + config.time_window + config.predict_window]
                ratio_mean, ratio_std = ratio_series[i:(i + config.time_window)].mean(), ratio_series[i:(i + config.time_window)].std()
                momentum_mean, momentum_std = momentum_series[i:(i + config.time_window)].mean(), momentum_series[i:(i + config.time_window)].std()
                ratio_norm = (ratio_series[i:(i + config.time_window)] - ratio_mean) / (ratio_std + 1e-8)
                momentum_norm = (momentum_series[i:(i + config.time_window)] - momentum_mean) / (momentum_std + 1e-8)
                feature = np.column_stack((
                    ratio_norm,
                    momentum_norm
                ))
                input_tensor = torch.FloatTensor(feature).unsqueeze(0).to(config.device)
                prediction = model(input_tensor).squeeze(0).cpu().numpy()  # shape: (predict_window, output_size)
                if config.predict_window > 1:
                    # 修正 ratio 第一维
                    prediction[0][0] = np.mean([
                        feature[-1, 0],  # 输入序列最后一步
                        prediction[0][0],  # 预测第一步
                        prediction[1][0]   # 预测第二步
                    ])
                    # 修正 momentum 第二维
                    prediction[0][1] = np.mean([
                        feature[-1, 1],
                        prediction[0][1],
                        prediction[1][1]
                    ])
                for step in range(config.predict_window):
                    # 反标准化
                    pred_ratios_list[step].loc[predict_dates[step], code] = prediction[step][0] * ratio_std + ratio_mean
                    pred_momentums_list[step].loc[predict_dates[step], code] = prediction[step][1] * momentum_std + momentum_mean

    # 3. 保存每一步的结果
    for step in range(config.predict_window):
        x_pred_filename = os.path.join(config.results_save_dir, f'x_pred_step{step+1}_{config.model_type.lower()}_weekly.csv')
        y_pred_filename = os.path.join(config.results_save_dir, f'y_pred_step{step+1}_{config.model_type.lower()}_weekly.csv')
        pred_ratios_list[step].dropna(how='all', inplace=True)
        pred_momentums_list[step].dropna(how='all', inplace=True)
        pred_ratios_list[step].to_csv(x_pred_filename)
        pred_momentums_list[step].to_csv(y_pred_filename)
        print(f"第{step+1}步预测文件已保存:\n{x_pred_filename}\n{y_pred_filename}")

# ...existing code...

if __name__ == "__main__":
    set_seed(1000)
    cfg = Config()
    
    # 确保目录存在
    if not os.path.exists(os.path.dirname(cfg.model_save_path)):
        os.makedirs(os.path.dirname(cfg.model_save_path))
    if not os.path.exists(cfg.results_save_dir):
        os.makedirs(cfg.results_save_dir)
        
    # 训练单一通用模型
    train_single_general_model(cfg)
    
    # 使用训练好的模型生成预测
    run_prediction_with_general_model(cfg)