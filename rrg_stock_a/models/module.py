import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
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



import torch
import random
from torch.utils.data import Sampler

def set_seed(seed: int, rank: int = 0):
    """
    Sets the random seed for reproducibility across all relevant libraries.

    Args:
        seed (int): The base seed value.
        rank (int): The process rank, used to ensure different processes have
                    different seeds, which can be important for data loading.
    """
    actual_seed = seed + rank
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)
        # The two lines below can impact performance, so they are often
        # reserved for final experiments where reproducibility is critical.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SequentialNonOverlappingSampler(Sampler):
    def __init__(self, dataset_length, seq_len, batch_size, shuffle=True):
        """
        顺序非重叠采样器
        Args:
            dataset_length: 数据集总长度
            seq_len: 序列长度
            batch_size: 批次大小
            shuffle: 是否打乱批次顺序
        """
        super(SequentialNonOverlappingSampler, self).__init__()
        self.dataset_length = dataset_length
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.py_rng = random.Random(1000)


        # 验证参数有效性
        if dataset_length < seq_len * (batch_size-1):
            raise ValueError(f"数据集长度({dataset_length})不足以容纳一个批次({seq_len * batch_size})")

        # 计算每个批次第一条数据的起始下标范围
        self.max_start_idx = dataset_length - seq_len * batch_size
        if self.max_start_idx<=0:
            self.max_start_idx=dataset_length-1
        # 生成所有可能的批次起始位置
        self.batch_starts = list(range(0, self.max_start_idx + 1))

    def _get_batch_indices(self, start_idx):
        """根据起始索引生成一个批次的索引列表"""
        batch_indices = []

        for i in range(self.batch_size):
            # 顺序取样本，间隔seq_len确保不重叠
            sample_start = start_idx + i * self.seq_len
            batch_indices.append(sample_start)

        return batch_indices
    def set_epoch_seed(self, epoch: int):
        """
        Sets a new seed for the random sampler for each epoch. This is crucial
        for reproducibility in distributed training.

        Args:
            epoch (int): The current epoch number.
        """
        epoch_seed = 1000 + epoch
        self.py_rng.seed(epoch_seed)
        self.rng_seed = epoch_seed

    def __iter__(self):
        # 如果需要打乱，先打乱批次起始位置
        if self.shuffle:
            self.py_rng.shuffle(self.batch_starts)

        # 为每个批次起始位置生成索引
        for start_idx in self.batch_starts:
            batch_indices = self._get_batch_indices(start_idx)
            if self.shuffle:
                self.py_rng.shuffle(batch_indices)

            yield batch_indices

    def __len__(self):
        return len(self.batch_starts)



import torch


@torch.jit.export
def calculate_vector_projection_matrix_optimized(x, current_timestep):
    """
    优化版的向量投影关系矩阵计算 - 完全向量化
    """
    batch_size, seq_len, num_nodes, input_features = x.shape

    # 使用当前时间步的坐标
    current_coords = x[:, current_timestep, :, :2]  # (batch_size, num_nodes, 2)

    # 扩展坐标矩阵以计算所有节点对
    coords_expanded_i = current_coords.unsqueeze(2)  # (batch_size, num_nodes, 1, 2)
    coords_expanded_j = current_coords.unsqueeze(1)  # (batch_size, 1, num_nodes, 2)

    # 计算点积和范数
    dot_products = torch.sum(coords_expanded_i * coords_expanded_j, dim=-1)  # (batch_size, num_nodes, num_nodes)
    norms_i = torch.norm(coords_expanded_i, dim=-1)  # (batch_size, num_nodes, 1)
    norms_j = torch.norm(coords_expanded_j, dim=-1)  # (batch_size, 1, num_nodes)

    geometric_mean = torch.sqrt(norms_i * norms_j)
    # 计算投影长度
    projection_lengths = dot_products/ (geometric_mean+1e-8)   # 避免除零

    # 使用较长的向量长度进行归一化
    max_norms = torch.max(norms_i, norms_j)

    normalized_projections = projection_lengths / (max_norms + 1e-8)

    # 设置对角线为1（自连接）
    identity = torch.eye(num_nodes, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
    projection_matrix = torch.clamp(normalized_projections, -1.0, 1.0)
    projection_matrix = torch.where(identity.bool(), torch.ones_like(projection_matrix), projection_matrix)


    return projection_matrix.detach()

@torch.jit.export
def calculate_trajectory_similarity_matrix_optimized(x, current_timestep, window_size=10):
    """
    优化版的轨迹相似性矩阵计算 - 完全向量化
    """
    batch_size, seq_len, num_nodes, input_features = x.shape

    # 确定窗口起始位置
    start_timestep = max(0, current_timestep - window_size + 1)
    if start_timestep == current_timestep:
        start_timestep = max(0, current_timestep - 1)

    # 提取窗口内的坐标数据
    window_coords = x[:, start_timestep:current_timestep + 1, :, :2]  # (batch_size, window_len, num_nodes, 2)
    window_len = window_coords.shape[1]

    if window_len < 2:
        # 窗口太小，返回零矩阵
        return torch.zeros(batch_size, num_nodes, num_nodes, device=x.device)

    # 计算方向向量（相邻时间步的差值）
    dir_vectors = window_coords[:, 1:] - window_coords[:, :-1]  # (batch_size, window_len-1, num_nodes, 2)

    # 归一化方向向量
    dir_norms = torch.norm(dir_vectors, dim=-1, keepdim=True)  # (batch_size, window_len-1, num_nodes, 1)
    dir_vectors_normalized = dir_vectors / (dir_norms + 1e-8)  # 避免除零

    # 扩展以计算所有节点对
    dir_i = dir_vectors_normalized.unsqueeze(3)  # (batch_size, window_len-1, num_nodes, 1, 2)
    dir_j = dir_vectors_normalized.unsqueeze(2)  # (batch_size, window_len-1, 1, num_nodes, 2)

    # 计算余弦相似度
    cos_similarities = torch.sum(dir_i * dir_j, dim=-1)  # (batch_size, window_len-1, num_nodes, num_nodes)

    # 创建有效掩码（排除零向量）
    valid_mask = (dir_norms.squeeze(-1) > 1e-8).unsqueeze(2).unsqueeze(3)  # (batch_size, window_len-1, 1, 1)
    squeezed = valid_mask.squeeze(-2).squeeze(-2)
    # 使用广播机制创建两个视图
    # 第一个视图：形状 (a, b, x, 1)
    view1 = squeezed.unsqueeze(-1)
    # 第二个视图：形状 (a, b, 1, x)
    view2 = squeezed.unsqueeze(-2)
    # 使用广播和逻辑与操作得到结果
    valid_mask = view1 & view2
    # valid_mask = valid_mask.expand(-1, -1, num_nodes, num_nodes)
    # 计算平均相似度
    valid_cos_similarities = torch.where(valid_mask, cos_similarities, torch.zeros_like(cos_similarities))
    valid_counts = valid_mask.sum(dim=1)  # (batch_size, num_nodes, num_nodes)

    # 计算平均值
    sum_similarities = valid_cos_similarities.sum(dim=1)  # (batch_size, num_nodes, num_nodes)
    avg_similarities = torch.where(
        valid_counts > 0,
        sum_similarities / valid_counts,
        torch.zeros_like(sum_similarities)
    )

    # 设置对角线为1（自连接）
    identity = torch.eye(num_nodes, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
    similarity_matrix = torch.clamp(avg_similarities, -1.0, 1.0)
    similarity_matrix = torch.where(identity.bool(), torch.ones_like(similarity_matrix), similarity_matrix)

    return similarity_matrix.detach()

@torch.jit.export
def calculate_quadrant_matrix_optimized(x, current_timestep):
    """
    优化版的象限关系矩阵计算 - 完全向量化
    """
    batch_size, seq_len, num_nodes, input_features = x.shape

    # 使用当前时间步的数据
    current_data = x[:, current_timestep]  # (batch_size, num_nodes, input_features)

    # 提取RS和动量
    rs_data = current_data[:, :, 0]  # (batch_size, num_nodes)
    momentum_data = current_data[:, :, 1]  # (batch_size, num_nodes)

    # 确定每个行业的象限
    # 使用向量化条件判断
    quadrants = torch.zeros(batch_size, num_nodes, dtype=torch.long, device=x.device)

    # 领先象限: RS>100 and Momentum>100
    lead_mask = (rs_data > 0) & (momentum_data > 0)
    quadrants[lead_mask] = 0

    # 改善象限: RS<100 and Momentum>100
    improve_mask = (rs_data < 0) & (momentum_data > 0)
    quadrants[improve_mask] = 1

    # 落后象限: RS<100 and Momentum<100
    lag_mask = (rs_data < 0) & (momentum_data < 0)
    quadrants[lag_mask] = 2

    # 转弱象限: RS>100 and Momentum<100
    weaken_mask = (rs_data > 0) & (momentum_data < 0)
    quadrants[weaken_mask] = 3

    # 构建象限关系矩阵
    quadrants_i = quadrants.unsqueeze(2)  # (batch_size, num_nodes, 1)
    quadrants_j = quadrants.unsqueeze(1)  # (batch_size, 1, num_nodes)

    quadrant_matrix = torch.stack([quadrants_i.expand(-1, -1, num_nodes),
                                   quadrants_j.expand(-1, num_nodes, -1)], dim=-1)

    return quadrant_matrix.detach()


@torch.jit.export
def calculate_vector_momentum_projection_matrix_optimized(x, current_timestep, weights=[0.6, 0.4]):
    """
    优化版的向量投影关系矩阵计算 - 支持多维特征融合
    weights: 一个列表，[位置权重, 速度权重]
    """
    batch_size, seq_len, num_nodes, input_features = x.shape

    # --- 位置平面计算 ---
    coords_pos = x[:, current_timestep, :, :2]  # (batch_size, num_nodes, 2)
    norms_sq_pos = (coords_pos ** 2).sum(dim=-1, keepdim=True) # (batch_size, num_nodes, 1)
    dot_product_pos = torch.bmm(coords_pos, coords_pos.transpose(1, 2)) # (batch_size, num_nodes, num_nodes)
    projection_matrix_pos = dot_product_pos / (norms_sq_pos.transpose(1, 2) + 1e-9)

    # --- 速度平面计算 ---
    coords_vel = x[:, current_timestep, :, 2:4] # (batch_size, num_nodes, 2)
    norms_sq_vel = (coords_vel ** 2).sum(dim=-1, keepdim=True) # (batch_size, num_nodes, 1)
    dot_product_vel = torch.bmm(coords_vel, coords_vel.transpose(1, 2)) # (batch_size, num_nodes, num_nodes)
    projection_matrix_vel = dot_product_vel / (norms_sq_vel.transpose(1, 2) + 1e-9)

    # --- 加权融合 ---
    projection_matrix = weights[0] * projection_matrix_pos + weights[1] * projection_matrix_vel
    
    # 保持原有的归一化和对角线处理
    identity = torch.eye(num_nodes, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
    projection_matrix = torch.clamp(projection_matrix, -1.0, 1.0)
    projection_matrix = torch.where(identity.bool(), torch.ones_like(projection_matrix), projection_matrix)

    return projection_matrix.detach()

@torch.jit.export
def calculate_momentum_trajectory_similarity_matrix_optimized(x, current_timestep, window_size=10, weights=[0.6, 0.4]):
    """
    优化版的轨迹相似性矩阵计算 - 支持多维特征融合
    weights: 一个列表，[位置权重, 速度权重]
    """
    batch_size, seq_len, num_nodes, input_features = x.shape

    if current_timestep < 1:
        return torch.zeros(batch_size, num_nodes, num_nodes, device=x.device)

    start_timestep = max(0, current_timestep - window_size + 1)
    
    # --- 内部辅助函数，用于计算单个平面的相似度 ---
    def _calculate_similarity_for_plane(features):
        # features: (batch_size, window_len, num_nodes, 2)
        window_len = features.shape[1]
        if window_len < 2:
            return torch.zeros(batch_size, num_nodes, num_nodes, device=x.device)
        
        # (batch, num_nodes, window_len, 2) -> (batch, num_nodes, window_len*2)
        flat_trajectories = features.permute(0, 2, 1, 3).flatten(start_dim=2)
        norm_trajectories = F.normalize(flat_trajectories, p=2, dim=-1)
        return torch.bmm(norm_trajectories, norm_trajectories.transpose(1, 2))

    # --- 位置平面计算 ---
    coords_pos = x[:, start_timestep:current_timestep + 1, :, :2]
    similarity_matrix_pos = _calculate_similarity_for_plane(coords_pos)

    # --- 速度平面计算 ---
    coords_vel = x[:, start_timestep:current_timestep + 1, :, 2:4]
    similarity_matrix_vel = _calculate_similarity_for_plane(coords_vel)

    # --- 加权融合 ---
    avg_similarities = weights[0] * similarity_matrix_pos + weights[1] * similarity_matrix_vel

    # 保持原有的对角线处理
    identity = torch.eye(num_nodes, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
    similarity_matrix = torch.clamp(avg_similarities, -1.0, 1.0)
    similarity_matrix = torch.where(identity.bool(), torch.ones_like(similarity_matrix), similarity_matrix)

    return similarity_matrix.detach()


# 使用示例和性能测试
# if __name__ == "__main__":
#     # 生成示例数据
#     batch_size = 16
#     seq_len = 200
#     num_nodes = 31
#     input_features = 8
#
#     # 使用GPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     x = torch.randn(batch_size, seq_len, num_nodes, input_features, device=device)
#
#     # 测试当前时间步
#     current_timestep = 25
#
#     # 性能测试
#     import time

    # 优化前版本（如果可用）
    # start_time = time.time()
    # projection_matrix_old = calculate_vector_projection_matrix(x, current_timestep)
    # similarity_matrix_old = calculate_trajectory_similarity_matrix(x, current_timestep, window_size=10)
    # quadrant_matrix_old = calculate_quadrant_matrix(x, current_timestep)
    # old_time = time.time() - start_time

    # print(f"优化前计算时间: {old_time:.4f}秒")
    # 优化后版本
    # start_time = time.time()
    # projection_matrix_new = calculate_vector_projection_matrix_optimized(x, current_timestep)
    # similarity_matrix_new = calculate_trajectory_similarity_matrix_optimized(x, current_timestep, window_size=10)
    # quadrant_matrix_new = calculate_quadrant_matrix_optimized(x, current_timestep)
    # new_time = time.time() - start_time


    # print(f"优化后计算时间: {new_time:.4f}秒")
    # print(f"加速比: {old_time / new_time:.2f}x")

    # 验证结果一致性
    # projection_diff = torch.abs(projection_matrix_old - projection_matrix_new).max()
    # similarity_diff = torch.abs(similarity_matrix_old - similarity_matrix_new).max()
    # quadrant_diff = torch.abs(quadrant_matrix_old.float() - quadrant_matrix_new.float()).max()

    # print(f"投影矩阵最大差异: {projection_diff:.6f}")
    # print(f"相似性矩阵最大差异: {similarity_diff:.6f}")
    # print(f"象限矩阵最大差异: {quadrant_diff:.6f}")
@torch.jit.export
def exp_loss(pred, target, loss_function):
    seq_len = pred.shape[1]
    loss=0.0
    for i in range(seq_len):
        loss += loss_function(pred[:,i],target[:,i])*np.exp(i-seq_len+1)
    return loss


@torch.jit.export
def efficient_sequence_correlation(x, y, seq_dim=1):
    x_perm = x.transpose(seq_dim, -1)
    y_perm = y.transpose(seq_dim, -1)

    x_centered = x_perm - x_perm.mean(dim=-1, keepdim=True)
    y_centered = y_perm - y_perm.mean(dim=-1, keepdim=True)

    # 使用Pearson相关系数的标准公式
    numerator = (x_centered * y_centered).sum(dim=-1)
    denominator = torch.sqrt((x_centered ** 2).sum(dim=-1) * (y_centered ** 2).sum(dim=-1))

    epsilon = 1e-8
    correlation = numerator / (denominator + epsilon)

    return correlation.mean()


import math
from torch.optim.lr_scheduler import _LRScheduler
import warnings
class DynamicWarmupCosineScheduler(_LRScheduler):
    """
    动态调整最大最小学习率的预热余弦调度器
    """

    def __init__(self, optimizer, warmup_steps, total_steps,
                 initial_lr_ratio=0.1, min_lr_ratio=1e-7,
                 max_lr_decay=0.95, min_lr_decay=0.98,min_cycle=21):
        """
        参数:
        - max_lr_decay: 最大学习率衰减因子 (每个调整周期)
        - min_lr_decay: 最小学习率衰减因子 (每个调整周期)
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cosine_steps = min(min_cycle,(total_steps-warmup_steps)//5)
        self.initial_lr_ratio = initial_lr_ratio
        self.min_lr_ratio = min_lr_ratio

        # 动态调整参数
        self.max_lr_decay = max_lr_decay
        self.min_lr_decay = min_lr_decay

        # 记录基础学习率和当前调整周期
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.adjustment_count = 0

        super().__init__(optimizer)

    def decay(self):
        self.max_lr_decay *= 0.99  # 逐渐减缓衰减
        self.min_lr_decay *= 0.995
        self.adjustment_count += 1
        self.cosine_steps += 1
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("...")

        if self.last_epoch < self.warmup_steps:
            # 预热阶段：线性增长
            progress = self.last_epoch / self.warmup_steps
            return [base_lr * (self.initial_lr_ratio + (1 - self.initial_lr_ratio) * progress)
                    for base_lr in self.base_lrs]
        else:

            cosine_progress = (self.last_epoch - self.warmup_steps) / self.cosine_steps
            current_max_lr_decay = self.max_lr_decay ** self.adjustment_count
            current_min_lr_decay = self.min_lr_decay ** self.adjustment_count

            cosine_value = 0.5 * (1 + math.cos(math.pi * cosine_progress))

            return [base_lr * (current_max_lr_decay +
                               (self.min_lr_ratio * current_min_lr_decay - current_max_lr_decay) *
                               (1 - cosine_value))
                    for base_lr in self.base_lrs]


