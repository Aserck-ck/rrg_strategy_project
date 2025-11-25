import sys
import gc
import matplotlib
import pandas as pd
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
# Ensure project root is in path
from rrg_stock_a.models.GAT_GRU import GRU_GAT_RRGPredictor
from dataset import prepare_sw1_dataloaders_pro, prepare_sw1_backtest_data
from rrg_stock_a.models.module import set_seed,efficient_sequence_correlation,exp_loss
from train_sw1 import get_config
from market_data.sw_level1.sw_level1_functions import get_sw_level1_info

matplotlib.rcParams['font.family'] = 'SimHei'  # 或其他支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
np.set_printoptions(threshold=np.inf)

def test_model():
    sys.path.append('/')
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            gc.collect()
            print("GPU缓存已清理")
        except Exception as e:
            print(f"缓存清理失败: {e}")

    config = get_config()
    time_window = config.time_window
    predict_window = config.predict_window
    batch_size = config.batch_size
    z_score_feature = config.z_score_feature
    log_feature = config.log_feature
    origin_feature = config.origin_feature
    stamp_feature = config.stamp_feature
    sizes= [128,256]
    model_name = [f'sw_level1_gat_gru{predict_window}_{size}' for size in sizes]
    # model_path = f"./gnn_models/{model_name}_best_model.pth"
    model_path = [f"./gnn_models/checkpoint/{name}_best_model.pth" for name in model_name]
    checkpoint_path = f"./gnn_models/checkpoint/{model_name[0]}_checkpoint.pth"


    model = [torch.load(path, weights_only=False, map_location=device) for path in model_path]
    best_val_loss = torch.load(checkpoint_path)["best_val_loss"]
    print(f"模型 {model_path[0]} 已加载，上一轮训练最小损失: {best_val_loss}")



    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, train_sampler = (
        prepare_sw1_dataloaders_pro(time_window, predict_window, batch_size,
                                    z_score_feature, log_feature, origin_feature, stamp_feature))
    dataset_len = len(test_dataset)
    for i in range(dataset_len-1,-1,-100):
        test_data,_=test_dataset[i]
        test_data =test_data.float()
        test_data = test_data.unsqueeze(0)
        test_data=test_data.to(device)
        test_in = test_data[:,:time_window].clone()
        target = test_data[:,time_window:,:,:2]
        preds = []
        with torch.no_grad():
            for model_x in model:
                preds.append(model_x(test_in))
        preds = torch.stack(preds,dim=-1)
        preds=preds.mean(dim=-1)
        # preds = torch.cat([test_in[:,-1:,:,:2],preds],dim=1)
        preds=preds.cpu().detach().numpy()
        raw_df = test_dataset.get_raw_data(['ratio','momentum'],i)
        std_df = test_dataset.get_std(['ratio_chg5', 'momentum_chg5'],i)
        test_data=test_data.cpu().detach().numpy()
        for j in range(31):


            # raw_df['ratio'].iloc[-predict_window, j]=raw_df['ratio'].iloc[-predict_window:, j].mean()
            # raw_df['momentum'].iloc[-predict_window, j] = raw_df['momentum'].iloc[-predict_window:, j].mean()
            point_x_start = raw_df['ratio'].iloc[-predict_window-1, j]
            point_x_middle = np.mean((raw_df['ratio'].iloc[-predict_window*2:-predict_window, j]+100)*(preds[0,:,j,0]*std_df['ratio_chg5'].iloc[j]+1)-100)
            point_x_end = (raw_df['ratio'].iloc[-predict_window-1, j]+100)*(preds[0,-1,j,0]*std_df['ratio_chg5'].iloc[j]+1)-100
            point_x = (raw_df['ratio'].iloc[-predict_window*2:-predict_window, j]+100)*(preds[0,:,j,0]*std_df['ratio_chg5'].iloc[j]+1)-100
            point_x=np.insert(point_x,0,raw_df['ratio'].iloc[-predict_window-1, j])

            point_y_start = raw_df['momentum'].iloc[-predict_window-1, j]
            point_y_middle = np.mean((raw_df['momentum'].iloc[-predict_window*2:-predict_window, j]+100)*(preds[0,:,j,1]*std_df['momentum_chg5'].iloc[j]+1)-100)
            point_y_end = (raw_df['momentum'].iloc[-predict_window-1, j]+100)*(preds[0,-1,j,1]*std_df['momentum_chg5'].iloc[j]+1)-100
            point_y = (raw_df['momentum'].iloc[-predict_window*2:-predict_window, j]+100)*(preds[0,:,j,1]*std_df['momentum_chg5'].iloc[j]+1)-100
            point_y=np.insert(point_y,0,raw_df['momentum'].iloc[-predict_window-1, j])
            # test_data[0, -predict_window, j, 0] = mean_x
            # test_data[0, -predict_window, j, 1] = mean_y
            # preds[0, -predict_window, j, 0] = pred_x
            # preds[0, -predict_window, j, 1] = pred_y
            plot_start=100

            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 12))

            axes[0,0].set_title("5日变化率预测原数据")
            pred_x = np.insert(np.array(preds[0, :, j, 0]),0,test_data[0,time_window-1,j,0])
            axes[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1, label='Threshold 0')
            axes[0,0].axvline(x=time_window-1, color='black', linestyle='--', alpha=0.7, linewidth=1)
            axes[0,0].plot(range(plot_start,time_window + predict_window),
                     test_data[0,plot_start:,j,0], label='origin_ratio_chg')
            axes[0,0].plot(range(time_window-1, time_window + predict_window),
                     pred_x, label='predict_ratio_chg', linestyle='--')
            pred_y =np.insert(np.array(preds[0, :, j, 1]),0,test_data[0,time_window-1,j,1])
            axes[0,0].plot(range(plot_start,time_window + predict_window),
                     test_data[0,plot_start:,j,1], label='origin_momentum_chg')
            axes[0,0].plot(range(time_window-1,time_window + predict_window),
                     pred_y, label='predict_momentum_chg', linestyle='--')

            axes[0,1].set_title("5日xy预测原数据")
            axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1, label='Threshold 100')
            axes[0,1].plot(range(plot_start, time_window + predict_window),
                     raw_df['ratio'].iloc[plot_start:, j], label='origin_ratio')
            axes[0,1].plot(range(time_window-1,time_window + predict_window),
                     point_x, label='predict_ratio', linestyle='--')

            axes[0,1].plot(range(plot_start, time_window + predict_window),
                     raw_df['momentum'].iloc[plot_start:, j], label='origin_momentum')
            axes[0,1].plot(range(time_window-1,time_window + predict_window),
                     point_y, label='predict_momentum', linestyle='--')

            axes[0,2].set_title("5日xy预测平均")
            axes[0,2].axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1, label='Threshold 100')
            axes[0,2].plot(range(plot_start,time_window+predict_window),
                     raw_df['ratio'].iloc[plot_start:,j],label='origin_ratio')
            axes[0,2].plot([time_window - 1, time_window + predict_window / 2, time_window + predict_window],
                     [point_x_start, point_x_middle, point_x_end], label='predict_ratio', linestyle='--')
            axes[0,2].plot(range(plot_start,time_window+predict_window),
                     raw_df['momentum'].iloc[plot_start:,j],label='origin_momentum')
            axes[0,2].plot([time_window-1,time_window+predict_window/2,time_window+predict_window],
                     [point_y_start,point_y_middle,point_y_end], label='predict_momentum', linestyle='--')

            axes[1,0].set_title("5日rrg预测原数据")
            axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
            axes[1,0].axvline(x=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
            axes[1,0].plot(raw_df['ratio'].iloc[-20:time_window, j],
                     raw_df['momentum'].iloc[-20:time_window, j])
            axes[1,0].plot(raw_df['ratio'].iloc[time_window - 1:, j],
                     raw_df['momentum'].iloc[time_window - 1:, j], label='origin')

            axes[1,0].plot(point_x,
                     point_y, label='predict', linestyle='--')

            axes[1,1].set_title("5日rrg预测平均")
            axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
            axes[1,1].axvline(x=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
            axes[1,1].plot(raw_df['ratio'].iloc[-20:time_window, j],
                     raw_df['momentum'].iloc[-20:time_window, j])
            axes[1,1].plot(raw_df['ratio'].iloc[time_window-1:, j],
                     raw_df['momentum'].iloc[time_window-1:, j], label='origin')
            axes[1,1].plot([point_x_start, point_x_middle,point_x_end],
                     [point_y_start, point_y_middle,point_y_end], label='predict_mean', linestyle='--')
            axes[1,1].plot([point_x_start,point_x_end],
                     [point_y_start,point_y_end], label='predict',linestyle='--')

            plt.legend()
            plt.savefig(f"./graphs/test_graphs/rrg/graph{j}_{i}.png")
            plt.close()
            print(f"图片: graph{j}_{i}.png 已存储")


def test_trail():
    sys.path.append('/')
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            gc.collect()
            print("GPU缓存已清理")
        except Exception as e:
            print(f"缓存清理失败: {e}")

    config = get_config()
    time_window = config.time_window
    predict_window = config.predict_window

    z_score_feature = config.z_score_feature
    log_feature = config.log_feature
    origin_feature = config.origin_feature
    stamp_feature = config.stamp_feature
    sizes = [128, 256, 200]
    model_name = [f'sw_level1_gat_gru{predict_window}_{size}' for size in sizes]
    # model_path = f"./gnn_models/{model_name}_best_model.pth"
    model_path = [f"./gnn_models/checkpoint/{name}_best_model.pth" for name in model_name]
    checkpoint_path = f"./gnn_models/checkpoint/{model_name[0]}_checkpoint.pth"

    model = [torch.load(path, weights_only=False, map_location=device) for path in model_path]
    best_val_loss = torch.load(checkpoint_path)["best_val_loss"]
    print(f"模型 {model_path[0]} 已加载，上一轮训练最小损失: {best_val_loss}")

    test_dataset = (
        prepare_sw1_backtest_data(time_window, predict_window, z_score_feature,
                                  log_feature, origin_feature, stamp_feature))
    etf_info_df = get_sw_level1_info()
    dataset_len = len(test_dataset)
    raw_df = test_dataset.get_raw_data_all(['ratio', 'momentum'], 0)
    date_index = raw_df['ratio'].index[time_window:]
    date_index1= raw_df['ratio'].index[time_window-1:]

    x_pred_df = pd.DataFrame(index=date_index, columns=raw_df['ratio'].columns, data=np.zeros((date_index.shape[0],31)))
    y_pred_df = pd.DataFrame(index=date_index, columns=raw_df['ratio'].columns, data=np.zeros((date_index.shape[0],31)))
    pred_cnt = np.zeros(date_index.shape[0])
    print(f"测试集长度：{dataset_len}")
    x_pkl = []
    y_pkl = []
    with torch.no_grad():
        for i in range(dataset_len):
            print(f"now processing data {i}: {date_index[i]}")
            test_data, _ = test_dataset[i]
            test_data = test_data.float()
            test_data = test_data.unsqueeze(0)
            test_data = test_data.to(device)
            test_in = test_data[:, :time_window].clone()
            preds = []

            for model_x in model:
                preds.append(model_x(test_in))
            preds = torch.stack(preds, dim=-1)
            preds = preds.mean(dim=-1)
            # preds = torch.cat([test_in[:,-1:,:,:2],preds],dim=1)
            preds = preds.cpu().detach().numpy()
            pred_cnt[i:i + predict_window] += 1

            std_df = test_dataset.get_std(['ratio_chg5', 'momentum_chg5'], i)
            test_data = test_data.cpu().detach().numpy()
            x_pkl5 = pd.DataFrame(index = date_index1[i:i+ predict_window+1],columns=raw_df['ratio'].columns)
            y_pkl5 = pd.DataFrame(index=date_index1[i:i + predict_window+1], columns=raw_df['ratio'].columns)
            x_pkl1 = pd.DataFrame(index=date_index1[i+1:i + predict_window + 1], columns=raw_df['ratio'].columns)
            y_pkl1 = pd.DataFrame(index=date_index1[i+1:i + predict_window + 1], columns=raw_df['ratio'].columns)

            x_pkl5.iloc[0]=raw_df['ratio'].iloc[i + time_window -1]
            y_pkl5.iloc[0]=raw_df['momentum'].iloc[i + time_window -1]
            for j in range(31):
                col = raw_df['ratio'].columns[j]
                etf_name = etf_info_df[etf_info_df['ts_code'] == col]['industry_name'].values[0]

                point_x = (raw_df['ratio'].iloc[i + time_window - predict_window:i + time_window, j] + 100) * (
                        preds[0, :, j, 0] * std_df['ratio_chg5'].iloc[j] + 1) - 100
                x_pkl1.iloc[:,j]=preds[0, :, j, 0] * std_df['ratio_chg5'].iloc[j]
                x_pred_df.iloc[i:i + predict_window, j] += point_x.to_numpy()

                point_y = (raw_df['momentum'].iloc[i + time_window - predict_window:i + time_window, j] + 100) * (
                        preds[0, :, j, 1] * std_df['momentum_chg5'].iloc[j] + 1) - 100
                y_pkl1.iloc[:,j]=preds[0, :, j, 1] * std_df['momentum_chg5'].iloc[j]
                y_pred_df.iloc[i:i + predict_window, j] += point_y.to_numpy()

                x_pkl5.iloc[1:predict_window+1,j] = point_x.to_numpy()
                y_pkl5.iloc[1:predict_window+1,j] = point_y.to_numpy()
            x_pkl.append((x_pkl1,x_pkl5))
            y_pkl.append((y_pkl1,y_pkl5))

    with open(f"./analyse/x_predict_all.pkl", 'wb') as f:
        pickle.dump(x_pkl, f)
    with open(f"./analyse/y_predict_all.pkl", 'wb') as f:
        pickle.dump(y_pkl, f)

    for col in x_pred_df.columns:
        x_pred_df[col]/=pred_cnt
        y_pred_df[col]/=pred_cnt

    x_pred_df.to_csv('./analyse/x_pred_all.csv')
    y_pred_df.to_csv('./analyse/y_pred_all.csv')

def plot_trail():
    sys.path.append('/')
    config = get_config()
    time_window = config.time_window
    predict_window = config.predict_window

    z_score_feature = config.z_score_feature
    log_feature = config.log_feature
    origin_feature = config.origin_feature
    stamp_feature = config.stamp_feature


    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, train_sampler = (
        prepare_sw1_dataloaders_pro(time_window, predict_window, 10,
                                    z_score_feature, log_feature, origin_feature, stamp_feature))
    etf_info_df = get_sw_level1_info()
    dataset_len = len(test_dataset)
    raw_df = test_dataset.get_raw_data_all(['ratio', 'momentum'], 0)
    date_index = raw_df['ratio'].index[time_window:]
    x_pred_df = pd.read_csv('./x_pred.csv',index_col=0,parse_dates=True)
    y_pred_df = pd.read_csv('./y_pred.csv',index_col=0,parse_dates=True)



    for col in x_pred_df.columns:
        plt.figure(figsize=(20,9))
        plt.plot(date_index,raw_df['ratio'].loc[date_index,col],label='ratio_origin')
        plt.plot(date_index, raw_df['momentum'].loc[date_index, col], label='momentum_origin')
        plt.plot(date_index, x_pred_df[col], label='ratio_pred', linestyle='--')
        plt.plot(date_index, y_pred_df[col], label='momentum_pred', linestyle='--')
        plt.legend()
        plt.savefig(f'./graphs/trail/{col}.png')
        plt.close()
        print(f'./graphs/trail/{col}.png saved')

def test_loss():
    sys.path.append('/')
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            gc.collect()
            print("GPU缓存已清理")
        except Exception as e:
            print(f"缓存清理失败: {e}")

    time_window = 200
    predict_window = 5
    epochs = 100
    patience = 12
    batch_size = 12
    lr = 2e-4
    model_name = f'sw_level1_gat_gru{predict_window}'
    model_path = f"./gnn_models/checkpoint/{model_name}_best_model.pth"
    checkpoint_path = f"./gnn_models/checkpoint/{model_name}_checkpoint.pth"

    model = torch.load(model_path, weights_only=False, map_location=device)
    best_val_loss = torch.load(checkpoint_path)["best_val_loss"]
    print(f"模型 {model_path} 已加载，上一轮训练最小损失: {best_val_loss}")

    config = get_config()
    time_window = config.time_window
    predict_window = config.predict_window
    batch_size = config.batch_size
    z_score_feature = config.z_score_feature
    log_feature = config.log_feature
    origin_feature = config.origin_feature
    stamp_feature = config.stamp_feature
    predict_dim =config.d_out

    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, train_sampler = (
        prepare_sw1_dataloaders_pro(time_window, predict_window, batch_size,
                                    z_score_feature, log_feature, origin_feature, stamp_feature))
    dataset_len = len(test_dataset)

    loss_function = nn.MSELoss()
    model.eval()
    ic_sum = 0.0
    tot_val_loss_sum_=0.0
    seq_loss_sum = 0.0
    mean_loss_sum = 0.0
    tail_loss_sum = 0.0
    val_batches_processed_rank=0
    with torch.no_grad():
        with tqdm(test_loader, desc=f"[测试]") as pbar:
            for i, (batch_x, _) in enumerate(pbar):
                batch_x = batch_x.to(device)
                batch_x = batch_x.float()
                inputs = batch_x[:, :time_window, ]
                targets = batch_x[:, time_window:, :, :predict_dim]

                preds = model(inputs)
                seq_loss = loss_function(preds, targets)

                pred_mean = preds.mean(dim=1)
                tar_mean = targets.mean(dim=1)

                mean_loss = loss_function(pred_mean, tar_mean)
                tail_loss = exp_loss(preds, targets, loss_function)

                single_loss = seq_loss*min(1,seq_loss.item())+tail_loss*min(1,tail_loss.item())

                tot_val_loss_sum_ += single_loss.item()
                seq_loss_sum += seq_loss.item()
                mean_loss_sum += mean_loss.item()
                tail_loss_sum += tail_loss.item()

                preds = torch.cat([batch_x[:, -predict_window:, :, :predict_dim], preds], dim=1)
                targets = torch.cat([batch_x[:, -predict_window:, :, :predict_dim], targets], dim=1)
                ic = efficient_sequence_correlation(preds, targets, 1)
                ic_sum += ic.item()

                val_batches_processed_rank += 1
                pbar.set_postfix({
                    '平均损失': f'{tot_val_loss_sum_ / (i + 1):.6f}',
                    '平均相关系数': f'{ic_sum / (i + 1):.6f}',
                    'seq_loss': f'{seq_loss_sum / (i + 1):.6f}',
                    'mean_loss': f'{mean_loss_sum / (i + 1):.6f}',
                    'tail_loss': f'{tail_loss_sum / (i + 1):.6f}',
                })
        # preds = torch.cat([test_in[:,-1:,:,:2],preds],dim=1)
        # preds = preds.cpu().detach().numpy()
        # test_data = test_data.cpu().detach().numpy()


    val_batches_processed_rank/=dataset_len
    print(val_batches_processed_rank)
    ### loss: 0.2942853276922026


if __name__ == "__main__":
    set_seed(1000)
    # test_loss()
    # test_model()
    test_trail()
    # plot_trail()
