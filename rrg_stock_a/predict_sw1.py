import sys
import gc
import matplotlib


import matplotlib.pyplot as plt
import numpy as np
import torch
# Ensure project root is in path

from dataset import prepare_sw1_predict_dataset
from rrg_stock_a.models.module import set_seed
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

    z_score_feature = config.z_score_feature
    log_feature = config.log_feature
    origin_feature = config.origin_feature
    stamp_feature = config.stamp_feature
    sizes = [128, 256]
    model_name = [f'sw_level1_gat_gru{predict_window}_{size}' for size in sizes]
    # model_path = f"./gnn_models/{model_name}_best_model.pth"
    model_path = [f"./gnn_models/checkpoint/{name}_best_model.pth" for name in model_name]
    checkpoint_path = f"./gnn_models/checkpoint/{model_name[0]}_checkpoint.pth"

    model = [torch.load(path, weights_only=False, map_location=device) for path in model_path]
    best_val_loss = torch.load(checkpoint_path)["best_val_loss"]
    print(f"模型 {model_path[0]} 已加载，上一轮训练最小损失: {best_val_loss}")



    test_dataset = (
        prepare_sw1_predict_dataset(time_window, z_score_feature, log_feature, origin_feature, stamp_feature))

    dataset_len = len(test_dataset)
    print(f"测试集长度：{dataset_len}")
    for i in [0,dataset_len-1]:
        test_data,_=test_dataset[i]
        test_data =test_data.float()
        test_data = test_data.unsqueeze(0)
        test_data=test_data.to(device)
        test_in = test_data[:,:time_window].clone()
        preds = []
        with torch.no_grad():
            for model_x in model:
                preds.append(model_x(test_in))
        preds = torch.stack(preds, dim=-1)
        preds = preds.mean(dim=-1)
        preds=preds.cpu().detach().numpy()
        raw_df = test_dataset.get_raw_data(['ratio','momentum'],i)
        std_df = test_dataset.get_std(['ratio_chg5', 'momentum_chg5'],i)
        test_data=test_data.cpu().detach().numpy()
        for j in range(31):


            # raw_df['ratio'].iloc[-predict_window, j]=raw_df['ratio'].iloc[-predict_window:, j].mean()
            # raw_df['momentum'].iloc[-predict_window, j] = raw_df['momentum'].iloc[-predict_window:, j].mean()
            point_x_start = raw_df['ratio'].iloc[-1, j]
            point_x_middle = np.mean((raw_df['ratio'].iloc[-predict_window:, j]+100)*(preds[0,:,j,0]*std_df['ratio_chg5'].iloc[j]+1)-100)
            point_x_end = (raw_df['ratio'].iloc[-1, j]+100)*(preds[0,-1,j,0]*std_df['ratio_chg5'].iloc[j]+1)-100
            point_x = (raw_df['ratio'].iloc[-predict_window:, j]+100)*(preds[0,:,j,0]*std_df['ratio_chg5'].iloc[j]+1)-100
            point_x=np.insert(point_x,0,raw_df['ratio'].iloc[-1, j])

            point_y_start = raw_df['momentum'].iloc[-1, j]
            point_y_middle = np.mean((raw_df['momentum'].iloc[-predict_window:, j]+100)*(preds[0,:,j,1]*std_df['momentum_chg5'].iloc[j]+1)-100)
            point_y_end = (raw_df['momentum'].iloc[-1, j]+100)*(preds[0,-1,j,1]*std_df['momentum_chg5'].iloc[j]+1)-100
            point_y = (raw_df['momentum'].iloc[-predict_window:, j]+100)*(preds[0,:,j,1]*std_df['momentum_chg5'].iloc[j]+1)-100
            point_y=np.insert(point_y,0,raw_df['momentum'].iloc[-1, j])
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
            axes[0,0].plot(range(plot_start,time_window),
                     test_data[0,plot_start:time_window,j,0], label='origin_ratio_chg')
            axes[0,0].plot(range(time_window-1, time_window + predict_window),
                     pred_x, label='predict_ratio_chg', linestyle='--')
            pred_y =np.insert(np.array(preds[0, :, j, 1]),0,test_data[0,time_window-1,j,1])
            axes[0,0].plot(range(plot_start,time_window),
                     test_data[0,plot_start:time_window,j,1], label='origin_momentum_chg')
            axes[0,0].plot(range(time_window-1,time_window + predict_window),
                     pred_y, label='predict_momentum_chg', linestyle='--')

            axes[0,1].set_title("5日xy预测原数据")
            axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1, label='Threshold 100')
            axes[0,1].plot(range(plot_start, time_window),
                     raw_df['ratio'].iloc[plot_start:time_window, j], label='origin_ratio')
            axes[0,1].plot(range(time_window-1,time_window + predict_window),
                     point_x, label='predict_ratio', linestyle='--')

            axes[0,1].plot(range(plot_start, time_window ),
                     raw_df['momentum'].iloc[plot_start:time_window, j], label='origin_momentum')
            axes[0,1].plot(range(time_window-1,time_window + predict_window),
                     point_y, label='predict_momentum', linestyle='--')

            axes[0,2].set_title("5日xy预测平均")
            axes[0,2].axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1, label='Threshold 100')
            axes[0,2].plot(range(plot_start,time_window),
                     raw_df['ratio'].iloc[plot_start:time_window,j],label='origin_ratio')
            axes[0,2].plot([time_window - 1, time_window + predict_window / 2, time_window + predict_window],
                     [point_x_start, point_x_middle, point_x_end], label='predict_ratio', linestyle='--')
            axes[0,2].plot(range(plot_start,time_window),
                     raw_df['momentum'].iloc[plot_start:time_window,j],label='origin_momentum')
            axes[0,2].plot([time_window-1,time_window+predict_window/2,time_window+predict_window],
                     [point_y_start,point_y_middle,point_y_end], label='predict_momentum', linestyle='--')

            axes[1,0].set_title("5日rrg预测原数据")
            axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
            axes[1,0].axvline(x=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
            axes[1,0].plot(raw_df['ratio'].iloc[-20:time_window, j],
                     raw_df['momentum'].iloc[-20:time_window, j], label='origin')
            axes[1,0].plot(point_x,
                     point_y, label='predict', linestyle='--')

            axes[1,1].set_title("5日rrg预测平均")
            axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
            axes[1,1].axvline(x=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
            axes[1,1].plot(raw_df['ratio'].iloc[-20:time_window, j],
                     raw_df['momentum'].iloc[-20:time_window, j], label='origin')
            axes[1,1].plot([point_x_start, point_x_middle,point_x_end],
                     [point_y_start, point_y_middle,point_y_end], label='predict_mean', linestyle='--')
            axes[1,1].plot([point_x_start,point_x_end],
                     [point_y_start,point_y_end], label='predict',linestyle='--')

            plt.legend()
            plt.savefig(f"./graphs/test_graphs/rrg_predict/graph{j}_{i}.png")
            plt.close()
            # plt.show()
            print(f"图片: graph{j}_{i}.png 已存储")

def plot_rrg():
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
    sizes = [128,200, 256]
    model_name = [f'sw_level1_gat_gru{predict_window}_{size}' for size in sizes]
    # model_path = f"./gnn_models/{model_name}_best_model.pth"
    model_path = [f"./gnn_models/checkpoint/{name}_best_model.pth" for name in model_name]
    checkpoint_path = f"./gnn_models/checkpoint/{model_name[0]}_checkpoint.pth"

    model = [torch.load(path, weights_only=False, map_location=device) for path in model_path]
    best_val_loss = torch.load(checkpoint_path)["best_val_loss"]
    print(f"模型 {model_path[0]} 已加载，上一轮训练最小损失: {best_val_loss}")


    test_dataset = (
        prepare_sw1_predict_dataset(time_window, z_score_feature, log_feature, origin_feature, stamp_feature))
    etf_info_df = get_sw_level1_info()
    dataset_len = len(test_dataset)
    print(f"测试集长度：{dataset_len}")
    for i in [0,dataset_len-1]:
        test_data,_=test_dataset[i]
        test_data =test_data.float()
        test_data = test_data.unsqueeze(0)
        test_data=test_data.to(device)
        test_in = test_data[:,:time_window].clone()
        preds = []
        with torch.no_grad():
            for model_x in model:
                preds.append(model_x(test_in))
        preds = torch.stack(preds, dim=-1)
        preds = preds.mean(dim=-1)
        # preds = torch.cat([test_in[:,-1:,:,:2],preds],dim=1)
        preds=preds.cpu().detach().numpy()
        raw_df = test_dataset.get_raw_data(['ratio','momentum'],i)
        date = raw_df['ratio'].index[-1]
        date = date.date()
        std_df = test_dataset.get_std(['ratio_chg5', 'momentum_chg5'],i)
        test_data=test_data.cpu().detach().numpy()
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 16))
        axes=[axes]
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
        axes[0].axvline(x=0, color='black', linestyle='--', alpha=0.7, linewidth=1)

        for j in range(31):
            col = raw_df['ratio'].columns[j]
            etf_name = etf_info_df[etf_info_df['ts_code'] == col]['industry_name'].values[0]
            point_x_start = raw_df['ratio'].iloc[-1, j]
            point_x_middle = np.mean((raw_df['ratio'].iloc[-predict_window:, j]+100)*(preds[0,:,j,0]*std_df['ratio_chg5'].iloc[j]+1)-100)
            point_x_end = (raw_df['ratio'].iloc[-1, j]+100)*(preds[0,-1,j,0]*std_df['ratio_chg5'].iloc[j]+1)-100
            point_x = (raw_df['ratio'].iloc[-predict_window:, j]+100)*(preds[0,:,j,0]*std_df['ratio_chg5'].iloc[j]+1)-100
            point_x=np.insert(point_x,0,raw_df['ratio'].iloc[-1, j])

            point_y_start = raw_df['momentum'].iloc[-1, j]
            point_y_middle = np.mean((raw_df['momentum'].iloc[-predict_window:, j]+100)*(preds[0,:,j,1]*std_df['momentum_chg5'].iloc[j]+1)-100)
            point_y_end = (raw_df['momentum'].iloc[-1, j]+100)*(preds[0,-1,j,1]*std_df['momentum_chg5'].iloc[j]+1)-100
            point_y = (raw_df['momentum'].iloc[-predict_window:, j]+100)*(preds[0,:,j,1]*std_df['momentum_chg5'].iloc[j]+1)-100
            point_y=np.insert(point_y,0,raw_df['momentum'].iloc[-1, j])

            line,= axes[0].plot(raw_df['ratio'].iloc[-5:time_window, j],
                     raw_df['momentum'].iloc[-5:time_window, j])
            plt.text(raw_df['ratio'].iloc[time_window-1, j],
                     raw_df['momentum'].iloc[time_window-1, j],
                     etf_name)
            color = line.get_color()
            axes[0].plot(point_x,
                     point_y, linestyle='--', color=color)

        plt.legend()
        plt.savefig(f"./graphs/test_graphs/rrg_predict/graph{date}.png")
        plt.close()
        # plt.show()
        print(f"图片: graph{date}.png 已存储")

if __name__ == "__main__":
    set_seed(1000)
    # test_loss()
    # test_model()
    plot_rrg()

