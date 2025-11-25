import sys
import time
import gc

from torch import nn

import torch
from tqdm import tqdm
# Ensure project root is in path
import numpy as np
# from GAT_GRU import GRU_GAT_RRGPredictor
from models.GAT_GRU import GRU_GAT_RRGPredictor
from dataset import prepare_sw1_dataloaders_pro
from models.module import set_seed,efficient_sequence_correlation,exp_loss





def train_model(continue_training,e_x=21):

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
    predict_window = config.predict_window
    epochs = config.epochs
    patience = config.patience
    batch_size = config.batch_size
    lr = config.lr
    min_lr = config.min_lr
    z_score_feature=config.z_score_feature
    log_feature= config.log_feature
    origin_feature=config.origin_feature
    stamp_feature=config.stamp_feature
    start_time = time.time()

    continue_training = continue_training
    # model_name='sw_level1_gat_gru'
    model_name = f'sw_level1_gat_gru{predict_window}_{config.hidden_size}'
    model_path=f"./gnn_models/checkpoint/{model_name}_best_model.pth"
    checkpoint_path = f"./gnn_models/checkpoint/{model_name}_checkpoint.pth"


    if continue_training:
        model = torch.load(model_path, weights_only=False, map_location=device)
        best_val_loss = torch.load(checkpoint_path)["best_val_loss"]*1.05
        print(f"模型 {model_path} 已加载，上一轮训练最小损失: {best_val_loss}")
    else:
        input_dim = len(z_score_feature)+len(log_feature)+len(origin_feature)
        model = GRU_GAT_RRGPredictor(
            num_industries=config.num_industries,
            input_features=input_dim,
            output_features=config.d_out,
            hidden_size=config.hidden_size,
            num_gat_layers=config.num_gat_layers,
            num_gru_layers=config.num_gru_layers,
            pred_len=predict_window
        ).to(device)
        best_val_loss = float('inf')
        best_loss_gap = 0.1

    loss_function = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=1e-3,
    )



    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, train_sampler = (
        prepare_sw1_dataloaders_pro(time_window, predict_window, batch_size,
                                    z_score_feature, log_feature, origin_feature, stamp_feature))
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

            for i, (batch_x, _) in enumerate(pbar):
                batch_x = batch_x.to(device)
                batch_x = batch_x.float()
                inputs = batch_x[:, :time_window,]
                targets = batch_x[:, time_window:, :, :predict_dim]

                preds = model(inputs)
                seq_loss = loss_function(preds, targets)

                # single_loss = single_loss - 0.1 * single_loss.item() * ic
                pred_mean = preds.mean(dim=1)
                tar_mean = targets.mean(dim=1)

                mean_loss = loss_function(pred_mean, tar_mean)
                tail_loss = exp_loss(preds, targets, loss_function)
                single_loss= seq_loss*min(1,seq_loss.item())+tail_loss*min(1,tail_loss.item())


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
                # 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_loss_gap':best_val_loss,

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
            if tot_val_loss_sum_ > best_val_loss*1.2:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr']/2,min_lr)
            if rollback_cnt == patience:
                model = torch.load(model_path, weights_only=False, map_location=device)
                rollback_cnt = 0
                for param_group in optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr']/2,min_lr/2)
                    min_lr = min_lr/2

                print(f"模型回滚成功")


class Config():
    def __init__(self,z_score_feature  ,log_feature ,origin_feature ,stamp_feature ,time_window ,
                 predict_window ,epochs ,patience ,batch_size ,lr ,
                 min_lr ,d_out,hidden_size, num_gat_layers,num_gru_layers,num_industries):
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

def get_config():
    #

    config=Config(
        z_score_feature=['ratio_chg5', 'momentum_chg5', 'close5',
                                            'ratio', 'momentum', 'radius'],
        log_feature = ['distance'],
        origin_feature = ['cos', 'sin', 'rank'],
        stamp_feature = ['rank', ],
        time_window = 252,
        predict_window = 5,
        epochs = 20,
        patience = 8,
        batch_size = 12,
        lr = 2e-4,
        min_lr = 1e-5,
        d_out=2,
        num_industries=31,
        hidden_size=200,
        num_gat_layers=4,
        num_gru_layers=3,
    )
    return config


if __name__ == "__main__":
    set_seed(1000)
    train_model(False,15)
