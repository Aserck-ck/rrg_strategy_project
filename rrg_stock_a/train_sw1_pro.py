import sys
import time
import gc

from torch import nn

import torch
from tqdm import tqdm
# Ensure project root is in path
import numpy as np
# from GAT_GRU import GRU_GAT_RRGPredictor
from models.GAT_GRU_pro import Stacked_GRU_GAT_Predictor
from dataset import prepare_sw1_dataloaders_weekly
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
    dropout = config.dropout

    start_time = time.time()
    print(f"start training: size:{hidden_size}, predict: {predict_window}")

    continue_training = continue_training
    # model_name='sw_level1_gat_gru'
    model_name = f'sw_level1_w_stack{predict_window}_{hidden_size}'
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
    if continue_training:
        try:
            model = torch.load(model_path, weights_only=False, map_location=device)
            best_val_loss = torch.load(checkpoint_path)["best_val_loss"]
            epochs_trained = torch.load(checkpoint_path).get('epoch', 0)
            epochs = min(epochs-epochs_trained+5, epochs)
            print(f"模型 {model_path} 已加载，上一轮训练最小损失: {best_val_loss}")
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
        weight_decay=1e-3,
    )



    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, train_sampler = (
        prepare_sw1_dataloaders_weekly(time_window, predict_window, batch_size,
                                    z_score_feature, log_feature, origin_feature, stamp_feature,
                                    dataset='sw1',train_ratio=train_ratio,val_ratio=val_ratio,test_ratio=test_ratio))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        steps_per_epoch=len(train_loader), epochs=epochs,
        pct_start=0.03, div_factor=10
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
                single_loss= seq_loss+tail_loss*min(tail_loss.item(),1)


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

                    single_loss = seq_loss+tail_loss*min(tail_loss.item(),1)

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
        loss_gap=abs(train_loss-tot_val_loss_sum_)
        if tot_val_loss_sum_ < best_val_loss:
            rollback_cnt = 0
            torch.save(model, model_path)
            best_val_loss = tot_val_loss_sum_
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_loss_gap':best_val_loss,
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
                checkpoint = torch.load(checkpoint_path, map_location=device)

                # 恢复模型权重
                model.load_state_dict(checkpoint['model_state_dict'])

                # 恢复优化器状态
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                rollback_cnt = 0
                rollback += 1
                if rollback >=2:
                    return


                print(f"模型回滚成功")


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
        self.dropout = dropout

def get_config():
    #

    config=Config(
        z_score_feature=['ratio', 'momentum','ratio_chg5', 'momentum_chg5',  'close5', 'radius'],
        log_feature = ['distance'],
        origin_feature = ['cos', 'sin', 'rank'],
        stamp_feature = ['rank', ],
        time_window = 52,
        predict_window = [2,3,4,5],
        epochs = 40,
        patience = 8,
        batch_size = 10,
        lr = 2e-4,
        min_lr = 1e-5,
        d_out=2,
        num_industries=31,
        hidden_size= [128,200,256,360,512],
        num_gat_layers=2,
        num_gru_layers=3,
        dropout=0.25,
    )
    return config


if __name__ == "__main__":
    set_seed(1000)
    # for train_step in range(1,3):
    #     for train_size in range(3):
    #         train_model(False, train_size, train_step, 15)
    # train_model(False, 0, 3, 15)
    train_ratio=0.5
    val_ratio=0.2
    test_ratio=0.3
    for train_step in [0,1]:
        for train_size in [0,1,2]:
            train_model(False, train_size, train_step, 15, Stacked_GRU_GAT_Predictor,
                        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)

