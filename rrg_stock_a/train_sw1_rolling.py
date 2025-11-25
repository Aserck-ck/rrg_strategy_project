import sys
import time
import gc
import os
from torch import nn
import pandas as pd
import torch
import json
from tqdm import tqdm
# Ensure project root is in path
import numpy as np
# from GAT_GRU import GRU_GAT_RRGPredictor
from models.GAT_GRU_pro import Stacked_GRU_GAT_Predictor, GRU_GAT_RRGPredictor,Stacked_GRU_GAT_Predictor_rg
from dataset import prepare_sw1_dataloaders_weekly,prepare_sw1_backtest_weekly
from models.module import set_seed,efficient_sequence_correlation,exp_loss





def train_model(continue_training,train_size,train_step,e_x=21, Model = None, 
                train_ratio=0.5, val_ratio=0.4, test_ratio=0.1):

    sys.path.append('../')
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
    predict_window = config.predict_window[train_step]
    epochs = config.epochs
    patience = config.patience
    batch_size = config.batch_size
    lr = config.lr
    min_lr = config.min_lr
    z_score_feature=config.z_score_feature
    log_feature= config.log_feature
    origin_feature=config.origin_feature
    stamp_feature=config.stamp_feature
    hidden_size = config.hidden_size[train_size]
    dropout = config.dropout[train_size]

    start_time = time.time()
    print(f"start training: size:{hidden_size}, predict: {predict_window}, dropout: {dropout}")

    continue_training = continue_training
    # model_name='sw_level1_gat_gru'
    if Model == Stacked_GRU_GAT_Predictor:
        model_name = f'sw_level1_w_stack{predict_window}_{hidden_size}_rolling'

    elif Model == GRU_GAT_RRGPredictor:
        model_name = f'sw_level1_w_gat_gru{predict_window}_{hidden_size}_rolling'

    elif Model == Stacked_GRU_GAT_Predictor_rg:
        model_name = f'sw_level1_w_stack_rg{predict_window}_{hidden_size}_rolling'

    model_path=f"./gnn_models/checkpoint/{model_name}_best_model.pth"
    checkpoint_path = f"./gnn_models/checkpoint/{model_name}_checkpoint.pth"

    training_params = {
        'time_window': time_window,
        'predict_window': predict_window,
        'epochs': epochs,
        'patience': patience,
        'batch_size': batch_size,
        'lr': lr,
        'min_lr': min_lr,
        'z_score_feature': z_score_feature,
        'log_feature': log_feature,
        'origin_feature': origin_feature,
        'stamp_feature': stamp_feature,
        'hidden_size': hidden_size,
        'd_out': config.d_out,
        'num_industries': config.num_industries,
        'num_gat_layers': config.num_gat_layers,
        'num_gru_layers': config.num_gru_layers,
        'model_name': model_name,
        'dropout': dropout,
    }

    model = None
    best_val_loss = float('inf')
    pre_val_loss = -1
    base_epochs = epochs//2

    if continue_training:
        try:
            model = torch.load(model_path, weights_only=False, map_location=device)
            pre_val_loss = torch.load(checkpoint_path, weights_only=False)["best_val_loss"]
            
            epochs_trained=0
            try:
                epochs_trained = torch.load(checkpoint_path, weights_only=False).get('epoch', 0)
                print(f"已训练轮数: {epochs_trained}")
            except Exception as e:
                print(f"无法获取已训练轮数: {e}")

            epochs = min(max(int(0.4 * (epochs-epochs_trained) + 0.6 * base_epochs),epochs//4+int(pre_val_loss*10/predict_window)),epochs)
                # 确保最少训练几轮，例如5轮

            lr = lr * (0.8 + 0.2*epochs_trained / config.epochs)
            print(f"模型 {model_path} 已加载，上一轮训练最小损失: {pre_val_loss}")
        except Exception as e:
            print(f"加载模型失败: {e}")

            
    if model is None:
        input_dim = len(z_score_feature)+len(log_feature)+len(origin_feature)
        model = Model(
            num_industries=config.num_industries,
            input_features=input_dim,
            output_features=config.d_out,
            hidden_size=hidden_size,
            gat_heads=config.d_out*2,
            num_gat_layers=config.num_gat_layers,
            num_gru_layers=config.num_gru_layers,
            pred_len=predict_window,
            dropout=dropout,
        ).to(device)
        best_val_loss = float('inf')


    loss_function = nn.MSELoss()


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=(8e-4 + 1e-6*hidden_size),
    )



    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, train_sampler = (
        prepare_sw1_dataloaders_weekly(time_window, predict_window, batch_size,
                                    z_score_feature, log_feature, origin_feature, stamp_feature,
                                    dataset='sw1',train_ratio=train_ratio,val_ratio=val_ratio,test_ratio=test_ratio))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader), epochs=epochs,
        pct_start=0.1, div_factor=10
    )
    # scheduler=DynamicWarmupCosineScheduler(
    #     optimizer=optimizer,
    #     warmup_steps=20,
    #     total_steps=len(train_loader),
    #     initial_lr_ratio=lr,
    #     min_lr_ratio=1e-6,
    #     max_lr_decay=0.90,
    #     min_lr_decay=0.95,
    #     min_cycle=100,
    # )

    rollback_cnt=0
    rollback=0
    predict_dim=config.d_out
    prepare_time = time.time()
    print(f"已完成训练前准备，用时: {prepare_time-start_time}")

    for epoch_idx in range(epochs):

        model.train()
        # train_loader.sampler.set_epoch(epoch_idx)

        train_sampler.set_epoch_seed(epoch_idx * 1919+e_x)

        loss_sum=0.0
        ic_sum=0.0

        iter_cnt=0
        loss_sum = 0.0
        seq_loss_sum = 0.0
        mean_loss_sum = 0.0
        tail_loss_sum = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch_idx + 1}/{epochs} [训练]") as pbar:

            for i, (batch_x, batch_stamp) in enumerate(pbar):
                batch_x = batch_x.to(device)
                batch_stamp=batch_stamp.to(device)
                batch_x = batch_x.float()
                inputs = batch_x[:, :time_window,]
                stamps = batch_stamp[:, :time_window,]
                targets = batch_x[:, time_window:, :, :predict_dim]

                preds = model(inputs,stamps, top_k=10)
                seq_loss = loss_function(preds, targets)

                # single_loss = single_loss - 0.1 * single_loss.item() * ic
                pred_mean = preds.mean(dim=1)
                tar_mean = targets.mean(dim=1)

                mean_loss = loss_function(pred_mean, tar_mean)
                tail_loss = exp_loss(preds, targets, loss_function)
                single_loss= seq_loss


                loss_sum += single_loss.item()

                seq_loss_sum += seq_loss.item()
                mean_loss_sum += mean_loss.item()
                tail_loss_sum += tail_loss.item()

                preds = torch.cat([batch_x[:, -predict_window:, :, :predict_dim], preds], dim=1)
                targets = torch.cat([batch_x[:, -predict_window:, :, :predict_dim], targets], dim=1)
                ic = efficient_sequence_correlation(preds, targets, 1)
                ic_sum += ic.item()
                optimizer.zero_grad()
                # single_loss = single_loss - 0.1 * single_loss.item() * ic
                single_loss.backward()
                optimizer.step()
                scheduler.step()


                del preds
                # print(preds)
                iter_cnt+=1
                pbar.set_postfix({
                    'LR': f"{optimizer.param_groups[0]['lr']:.6f}",
                    '平均损失': f'{loss_sum / (i + 1):.6f}',
                    '平均相关系数': f'{ic_sum / (i + 1):.6f}',
                    'seq_loss': f'{seq_loss_sum / (i + 1):.6f}',
                    'mean_loss': f'{mean_loss_sum / (i + 1):.6f}',
                    'tail_loss': f'{tail_loss_sum / (i + 1):.6f}',
                })
        model.eval()
        train_loss = loss_sum/iter_cnt
        tot_val_loss_sum_ = 0.0
        ic_sum=0.0
        loss_sum = 0.0
        seq_loss_sum = 0.0
        mean_loss_sum = 0.0
        tail_loss_sum = 0.0
        val_batches_processed_rank = 0
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Epoch {epoch_idx + 1}/{epochs} [验证]") as pbar:
                for i, (batch_x, batch_stamp) in enumerate(pbar):
                    batch_x = batch_x.to(device)
                    batch_stamp = batch_stamp.to(device)
                    batch_x = batch_x.float()
                    inputs = batch_x[:, :time_window, ]
                    stamps = batch_stamp[:, :time_window, ]
                    targets = batch_x[:, time_window:, :, :predict_dim]

                    preds = model(inputs, stamps, top_k=5)
                    seq_loss = loss_function(preds, targets)

                    pred_mean = preds.mean(dim=1)
                    tar_mean = targets.mean(dim=1)

                    mean_loss = loss_function(pred_mean, tar_mean)
                    tail_loss = exp_loss(preds, targets, loss_function)

                    single_loss = seq_loss

                    tot_val_loss_sum_ += single_loss.item()
                    seq_loss_sum += seq_loss.item()
                    mean_loss_sum += mean_loss.item()
                    tail_loss_sum += tail_loss.item()

                    preds = torch.cat([batch_x[:, -predict_window:, :, :predict_dim],preds], dim=1)
                    targets = torch.cat([batch_x[:, -predict_window:, :, :predict_dim],targets], dim=1)
                    ic = efficient_sequence_correlation(preds, targets, 1)
                    ic_sum += ic.item()

                    val_batches_processed_rank+=1
                    pbar.set_postfix({
                        '平均损失': f'{tot_val_loss_sum_ / (i + 1):.6f}',
                        '平均相关系数': f'{ic_sum / (i + 1):.6f}',
                        'seq_loss': f'{seq_loss_sum / (i + 1):.6f}',
                        'mean_loss': f'{mean_loss_sum / (i + 1):.6f}',
                        'tail_loss': f'{tail_loss_sum / (i + 1):.6f}',
                    })

        tot_val_loss_sum_ = tot_val_loss_sum_/val_batches_processed_rank


        if tot_val_loss_sum_ < best_val_loss :
            rollback_cnt = 0
            torch.save(model, model_path)
            best_val_loss = tot_val_loss_sum_
            save_val_loss = best_val_loss

                
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': save_val_loss,
                'training_params': training_params, # <--- 保存这个精确的参数字典
                'epoch': epoch_idx + 1
            }
            torch.save(checkpoint,checkpoint_path)
            print(f"模型已存储")
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    gc.collect()
                    print("GPU缓存已清理")
                except Exception as e:
                    print(f"缓存清理失败: {e}")
        else:
            rollback_cnt += 1
            print(f"回滚计数: {rollback_cnt}")

            if rollback_cnt == patience:
                checkpoint = torch.load(checkpoint_path, weights_only=False)

                # 恢复模型权重
                model.load_state_dict(checkpoint['model_state_dict'])

                # 恢复优化器状态
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                rollback_cnt = 0
                rollback += 1
                if rollback >=2:
                    return


                print(f"模型回滚成功")
    


def run_single_prediction(models, full_dataset, predict_idx, device, config, val_losses):
    """使用加载的模型对特定索引的数据点进行单次预测。"""
    with torch.no_grad():

        test_data, stamps = full_dataset[predict_idx]
        test_data = test_data.float().unsqueeze(0).to(device)
        stamps = stamps.unsqueeze(0).to(device)
        test_in = test_data[:, :config.time_window].clone()
        stamps_in = stamps[:, :config.time_window].clone()
        preds = []
        single_preds = []
        for i,model_x in enumerate(models):
            res = model_x(test_in,stamps_in,top_k=5)
            preds.append(res* ((1/val_losses[i])/sum(1/v for v in val_losses)))
            single_preds.append(res)
        preds = torch.stack(preds, dim=-1)
        preds = preds.sum(dim=-1)
            # preds = torch.cat([test_in[:,-1:,:,:2],preds],dim=1)
        if preds.shape[1]>1:
            preds = torch.cat([test_in[:, -1:, :, :config.d_out], preds[:, :2]], dim=1)
            preds = preds.mean(dim=1, keepdim=True)

        preds = preds.cpu().detach().numpy()

    std_df = full_dataset.get_std(['ratio', 'momentum'], predict_idx)
    
    final_pred_ratio_normalized = preds[0, 0, :, 0]
    final_pred_momentum_normalized = preds[0, 0, :, 1]
    
    # 反标准化
    final_pred_ratio = final_pred_ratio_normalized * std_df['ratio'].values
    final_pred_momentum = final_pred_momentum_normalized * std_df['momentum'].values
    individual_predictions = []
    for pred_normalized in single_preds:
        smoothed_individual = torch.cat([test_in[:, -1:, :, :config.d_out], pred_normalized[:, :2]], dim=1)
        smoothed_individual = smoothed_individual.mean(dim=1, keepdim=True).cpu().detach().numpy()
            
        individual_ratio = smoothed_individual[0, 0, :, 0] * std_df['ratio'].values
        individual_momentum = smoothed_individual[0, 0, :, 1] * std_df['momentum'].values
        individual_predictions.append((individual_ratio, individual_momentum))

    # 返回平滑并反标准化后的预测值向量
    return (final_pred_ratio, final_pred_momentum), individual_predictions

def walk_forward_finetune(train_step_idx, continue_rolling=True, skip_first=False, model_type =None):
    """执行前向展开分析的主函数 (滚动窗口 + 增量微调)。"""
    set_seed(1000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- 开始前向展开分析 (滚动窗口 + 增量微调) ---")

    config = get_config()
    train_window_size, validation_weeks, retrain_interval = 500, 150, 20

    sizes = config.hidden_size
    predict_window = config.predict_window[train_step_idx]

    output_dir = './analyse/rolling/'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    progress_file = os.path.join(output_dir, f'{model_type}_pred{predict_window}_walk_forward_progress.json')

    config = get_config()
    time_window = config.time_window
    predict_window = config.predict_window[train_step_idx]

    z_score_feature = config.z_score_feature
    log_feature = config.log_feature
    origin_feature = config.origin_feature
    stamp_feature = config.stamp_feature
    sizes = config.hidden_size
    if model_type == 'stack':
        Model = Stacked_GRU_GAT_Predictor
        
    elif model_type == 'gat_gru':
        Model = GRU_GAT_RRGPredictor

    elif model_type == 'stack_rg':
        Model = Stacked_GRU_GAT_Predictor_rg

    model_name = [f'sw_level1_w_{model_type}{predict_window}_{size}_rolling' for size in sizes]
    # model_path = f"./gnn_models/{model_name}_best_model.pth"
    model_path = [f"./gnn_models/checkpoint/{name}_best_model.pth" for name in model_name]
    checkpoint_path = [f"./gnn_models/checkpoint/{name}_checkpoint.pth" for name in model_name]
    
    # 为每个模型创建独立的预测文件路径
    x_pred_paths = {size: os.path.join(output_dir, f'{model_type}_x_pred{predict_window}_walk_forward_{size}.csv') for size in sizes}
    y_pred_paths = {size: os.path.join(output_dir, f'{model_type}_y_pred{predict_window}_walk_forward_{size}.csv') for size in sizes}
    

    x_pred_path = os.path.join(output_dir, f'{model_type}_x_pred{predict_window}_walk_forward.csv')
    y_pred_path = os.path.join(output_dir, f'{model_type}_y_pred{predict_window}_walk_forward.csv')
    full_dataset = prepare_sw1_backtest_weekly(
        lookback_window=config.time_window,
        predict_window=config.predict_window[train_step_idx],
        z_score_feature=config.z_score_feature, log_feature=config.log_feature,
        origin_feature=config.origin_feature, stamp_feature=config.stamp_feature,
        dataset='sw1'
    )
    total_samples = len(full_dataset)
    print(f"完整数据集加载完毕，总样本数: {total_samples}")

    raw_data_info = full_dataset.get_raw_data_all(['ratio'], 0)
    all_dates, industry_codes = raw_data_info['ratio'].index, raw_data_info['ratio'].columns
    
    loop_start_idx = train_window_size
    prediction_start_date_idx = loop_start_idx + validation_weeks + config.time_window

    if prediction_start_date_idx >= len(all_dates):
        print("错误：数据量不足以进行任何预测。")
        return
    
    x_pred_dfs = {}
    y_pred_dfs = {}
    progress_data = {}
    if continue_rolling and os.path.exists(progress_file) and os.path.exists(x_pred_path):
        print("发现历史进度，尝试加载...")
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            last_completed_i = progress_data['last_completed_outer_loop_idx']
            start_i = last_completed_i + retrain_interval
            
            x_pred_df = pd.read_csv(x_pred_path, index_col=0, parse_dates=True)
            y_pred_df = pd.read_csv(y_pred_path, index_col=0, parse_dates=True)
            for size in sizes:
                x_pred_dfs[size] = pd.read_csv(x_pred_paths[size], index_col=0, parse_dates=True)
                y_pred_dfs[size] = pd.read_csv(y_pred_paths[size], index_col=0, parse_dates=True)
            
            print(f"加载成功。从迭代索引 {start_i} 继续。")
        except Exception as e:
            print(f"加载历史进度失败: {e}。将从头开始。")
            start_i = loop_start_idx
            x_pred_df = None # 强制重新创建
            x_pred_dfs = {} # 强制重新创建
            progress_data = {}
            continue_rolling = False
    else:
        print("未发现历史进度，将从头开始。")
        start_i = loop_start_idx
        x_pred_df = None  # 强制重新创建
        x_pred_dfs = {} # 强制重新创建
        progress_data = {}
        continue_rolling = False
    continue_training = continue_rolling 
    # --- 初始化预测DataFrame ---
    if 'x_pred_df' not in locals() or x_pred_df is None:
        prediction_start_date_idx = loop_start_idx + validation_weeks + config.time_window
        if prediction_start_date_idx >= len(all_dates):
            print("错误：数据量不足以进行任何预测。")
            return
        predict_dates = all_dates[prediction_start_date_idx:]
        x_pred_df = pd.DataFrame(index=predict_dates, columns=industry_codes, dtype=float)
        y_pred_df = pd.DataFrame(index=predict_dates, columns=industry_codes, dtype=float)
        for size in sizes:
            x_pred_dfs[size] = pd.DataFrame(index=predict_dates, columns=industry_codes, dtype=float)
            y_pred_dfs[size] = pd.DataFrame(index=predict_dates, columns=industry_codes, dtype=float)
    

    predict_dates = all_dates[prediction_start_date_idx:]

    if 'x_pred_df' not in locals() or x_pred_df is None:
        prediction_start_date_idx = loop_start_idx + validation_weeks + config.time_window
        if prediction_start_date_idx >= len(all_dates):
            print("错误：数据量不足以进行任何预测。")
            return
        predict_dates = all_dates[prediction_start_date_idx:]
        x_pred_df = pd.DataFrame(index=predict_dates, columns=industry_codes, dtype=float)
        y_pred_df = pd.DataFrame(index=predict_dates, columns=industry_codes, dtype=float)


    for i in range(start_i, total_samples, retrain_interval):
        train_end_idx = i
        if i > start_i:
            continue_training = True
        if i > start_i:
            skip_first = False
        train_start_idx = max(0, train_end_idx - train_window_size)
        train_ratio = total_samples-train_start_idx
        val_end_idx = i + validation_weeks
        val_ratio = total_samples - train_end_idx
        predict_start_idx = val_end_idx
        test_ratio = total_samples - predict_start_idx
        predict_end_idx = min(predict_start_idx + retrain_interval, total_samples)
        

        if val_end_idx >= total_samples:
            print("验证集触及数据末尾，分析结束。")
            break
        
        print(f"\n===== 迭代开始: {all_dates[i + config.time_window - 1]} =====")
        print(f"训练周期: {train_start_idx} -> {train_end_idx-1} | 验证周期: {train_end_idx} -> {val_end_idx-1}")
        if not skip_first:
            if continue_training:
                val_losses = [torch.load(check_path, weights_only=False)["best_val_loss"] for check_path in checkpoint_path]
            for size_idx in range(len(sizes)):
                if continue_training and val_losses[size_idx] < 0.05:
                    print(f"跳过模型大小 {sizes[size_idx]} 的训练，因验证损失较低: {val_losses[size_idx]}")
                    continue
                elif continue_training and val_losses[size_idx] >= np.mean(val_losses)*5:
                    print(f"重新训练模型大小 {sizes[size_idx]}，因验证损失过高: {val_losses[size_idx]}")
                    train_model(
                        False, size_idx, train_step_idx,
                        15, Model,
                        train_ratio, val_ratio, test_ratio,
                    )
                else:
                    train_model(
                        continue_training, size_idx, train_step_idx,
                        15, Model,
                        train_ratio, val_ratio, test_ratio,
                    )
        

        models = [torch.load(path, weights_only=False, map_location=device) for path in model_path]
        val_losses = [torch.load(check_path, weights_only=False)["best_val_loss"] for check_path in checkpoint_path]

        print(f"模型{model_name}加载完毕，上一轮验证损失: {val_losses}")

        print(f"预测周期: {predict_start_idx} -> {predict_end_idx-1}")
        for predict_idx in range(predict_start_idx, predict_end_idx):
            pred_date_idx = predict_idx + config.time_window
            if pred_date_idx >= len(all_dates): continue

            pred_date = all_dates[pred_date_idx]
            if pred_date not in x_pred_df.index: continue

            (pred_ratio, pred_momentum),individual_preds = run_single_prediction(models, full_dataset, predict_idx, device, config, val_losses)
        
            x_pred_df.loc[pred_date] = pred_ratio
            y_pred_df.loc[pred_date] = pred_momentum

            for idx, size in enumerate(sizes):
                pred_ratio, pred_momentum = individual_preds[idx]
                x_pred_dfs[size].loc[pred_date] = pred_ratio
                y_pred_dfs[size].loc[pred_date] = pred_momentum

        print("正在保存当前进度...")
        x_pred_df.to_csv(x_pred_path)
        y_pred_df.to_csv(y_pred_path)
        for size in sizes:
            x_pred_dfs[size].to_csv(x_pred_paths[size])
            y_pred_dfs[size].to_csv(y_pred_paths[size])

        model_performance = {name: loss for name, loss in zip(model_name, val_losses)}
        if 'performance_history' not in progress_data:
            progress_data['performance_history'] = {}
        
        # 将当前循环的性能数据添加到历史记录中，使用循环索引 i 作为键
        progress_data['performance_history'][str(i)] = model_performance
        
        # 更新最后完成的循环索引
        progress_data['last_completed_outer_loop_idx'] = i
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=4) # 使用indent参数使json文件更易读
         
        print(f"进度已保存。已完成索引 {i} 的迭代。")

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

   

class Config():
    def __init__(self,z_score_feature  ,log_feature ,origin_feature ,stamp_feature ,time_window ,
                 predict_window ,epochs ,patience ,batch_size ,lr ,
                 min_lr ,d_out,hidden_size, num_gat_layers,num_gru_layers,num_industries,dropout=0.2):
        self.z_score_feature=z_score_feature
        self.log_feature=log_feature
        self.origin_feature=origin_feature
        self.stamp_feature=stamp_feature
        self.time_window=time_window
        self.predict_window=predict_window
        self.epochs=epochs
        self.patience=patience
        self.batch_size=batch_size
        self.lr=lr
        self.min_lr=min_lr
        self.d_out=d_out
        self.hidden_size = hidden_size
        self.num_gat_layers = num_gat_layers
        self.num_gru_layers = num_gru_layers
        self.num_industries = num_industries
        self.dropout = []
        for size in hidden_size:
            self.dropout.append(np.sqrt(size/32)*0.1)

def get_config():
    #

    config=Config(
        z_score_feature=['ratio', 'momentum','ratio_chg5', 'momentum_chg5',  'close5', 'radius'],
        log_feature = ['distance'],
        origin_feature = ['cos', 'sin', 'rank'],
        stamp_feature = ['rank', ],
        time_window = 52,
        predict_window = [2,3,4,1],
        epochs = 20,
        patience = 6,
        batch_size = 7,
        lr = 2e-4,
        min_lr = 1e-5,
        d_out=2,
        num_industries=31,
        hidden_size= [320],
        num_gat_layers=2,
        num_gru_layers=3,
    )
    return config


if __name__ == "__main__":
    set_seed(1000)
    # for train_step in range(1,3):
    #     for train_size in range(3):
    #         train_model(False, train_size, train_step, 15)
    # train_model(False, 0, 3, 15)
    # train_ratio=0.5
    # val_ratio=0.2
    # test_ratio=0.3
    # for train_step in [0,1]:
    #     for train_size in [0,1,2]:
    #         train_model(False, train_size, train_step, 15, Stacked_GRU_GAT_Predictor,
    #                     train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)

    walk_forward_finetune(0, continue_rolling=False, skip_first=False, model_type='stack_rg')
