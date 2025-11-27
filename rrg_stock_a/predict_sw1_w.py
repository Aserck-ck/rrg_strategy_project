import sys
import gc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import prepare_sw1_predict_weekly
from models.module import set_seed
from models.GAT_GRU_pro import Stacked_GRU_GAT_Predictor_rg
from train_sw1_rolling import get_config
from market_data.sw_level1.sw_level1_functions import get_sw_level1_info

matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
np.set_printoptions(threshold=np.inf)

def predict_weekly_rrg(predict_window=1, hidden_size=128):
    set_seed(1000)
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
    z_score_feature = config.z_score_feature
    log_feature = config.log_feature
    origin_feature = config.origin_feature
    stamp_feature = config.stamp_feature
    input_dim = len(z_score_feature) + len(log_feature) + len(origin_feature)

    model_name = f'sw_level1_w_stack_rg{predict_window}_{hidden_size}_rolling'
    model_path = f"./gnn_models/checkpoint/{model_name}_checkpoint.pth"

    model = Stacked_GRU_GAT_Predictor_rg(
        num_industries=config.num_industries,
        input_features=input_dim,
        output_features=config.d_out,
        hidden_size=hidden_size,
        gat_heads=config.d_out*2,
        num_gat_layers=config.num_gat_layers,
        num_gru_layers=config.num_gru_layers,
        pred_len=predict_window,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_dataset = prepare_sw1_predict_weekly(
        lookback_window=config.time_window,
        predict_window=predict_window,
        z_score_feature=config.z_score_feature, log_feature=config.log_feature,
        origin_feature=config.origin_feature, stamp_feature=config.stamp_feature,
        dataset='sw1'
    )
    etf_info_df = get_sw_level1_info()
    dataset_len = len(test_dataset)
    print(f"测试集长度：{dataset_len}")

    for i in [dataset_len - predict_window - 1, dataset_len - 1]:
        test_data, _ = test_dataset[i]
        test_data = test_data.float().unsqueeze(0).to(device)
        test_in = test_data[:, :time_window].clone()
        with torch.no_grad():
            preds = model(test_in)
        preds = preds.cpu().detach().numpy()  # shape: (1, pred_len, num_industries, 2)
        raw_df = test_dataset.get_raw_data(['ratio', 'momentum'], i)
        std_df = test_dataset.get_std(['ratio', 'momentum'], i)
        date = raw_df['ratio'].index[-1]
        date = date.date()
        test_data = test_data.cpu().detach().numpy()

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 16))
        axes = [axes]
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
        axes[0].axvline(x=0, color='black', linestyle='--', alpha=0.7, linewidth=1)

        for j in range(config.num_industries):
            col = raw_df['ratio'].columns[j]
            etf_name = etf_info_df[etf_info_df['ts_code'] == col]['industry_name'].values[0]
            # 画过去5个点的真实RRG轨迹（实线）
            line = axes[0].plot(
                raw_df['ratio'].iloc[-5-predict_window:, j],
                raw_df['momentum'].iloc[-5-predict_window:, j],
                linestyle='-', marker=None
            )[0]
            color = line.get_color()
            plt.text(raw_df['ratio'].iloc[-1, j],
                     raw_df['momentum'].iloc[-1, j],
                     etf_name)
            # 预测轨迹（虚线，marker='o'，颜色与真实轨迹一致）
            ratio_last = raw_df['ratio'].iloc[-1, j]
            momentum_last = raw_df['momentum'].iloc[-1, j]
            pred_ratio = preds[0, :, j, 0] * std_df['ratio'].iloc[j]
            pred_momentum = preds[0, :, j, 1] * std_df['momentum'].iloc[j]
            # 修正第一个预测点
            if predict_window > 1:
                ratio_first_pred = np.mean([ratio_last, pred_ratio[0], pred_ratio[1]])
                momentum_first_pred = np.mean([momentum_last, pred_momentum[0], pred_momentum[1]])
                point_x = [ratio_last, ratio_first_pred] + list(pred_ratio[1:])
                point_y = [momentum_last, momentum_first_pred] + list(pred_momentum[1:])
            else:
                point_x = [ratio_last] + list(pred_ratio)
                point_y = [momentum_last] + list(pred_momentum)
            axes[0].plot(point_x, point_y, linestyle='--', color=color, marker='o')

        plt.legend()
        plt.savefig(f"./graphs/test_graphs/rrg_predict_w/weekly_graph_{date}_predlen{predict_window}.png")
        plt.close()
        print(f"图片: weekly_graph_{date}_predlen{predict_window}.png 已存储")

if __name__ == "__main__":
    predict_weekly_rrg(predict_window=2, hidden_size=256)