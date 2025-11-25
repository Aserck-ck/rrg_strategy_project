import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.module import (
    calculate_momentum_trajectory_similarity_matrix_optimized,
    calculate_quadrant_matrix_optimized,
    calculate_vector_momentum_projection_matrix_optimized, )

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeAwareGATLayer(nn.Module):
    """优化版的时间感知图注意力层，使用连续的轨迹相似性矩阵"""

    def __init__(self, in_features, out_features, heads=2, dropout=0.2):
        super(TimeAwareGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads

        # 线性变换参数
        self.W = nn.Linear(in_features, out_features * heads, bias=False)

        # 注意力机制参数 - 使用更高效的多头注意力实现
        self.a = nn.Parameter(torch.empty(size=(heads, 2 * out_features, 1)))
        # 投影矩阵变换 - 使用线性层处理连续投影值
        self.projection_transform = nn.Linear(1, out_features, bias=False)

        # 轨迹相似性变换 - 使用线性层处理连续相似性值
        self.similarity_transform = nn.Linear(1, out_features, bias=False)

        # 象限一致性偏置
        self.quadrant_bias = nn.Parameter(torch.zeros(4, 4))

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(dropout)

        self.head_concat = nn.Linear(out_features * heads, out_features, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.data)
        nn.init.xavier_uniform_(self.similarity_transform.weight)
        nn.init.zeros_(self.quadrant_bias)

    def forward(self, h, projection_matrix=None, similarity_matrix=None, quadrant_matrix=None, top_k=15):
        """
        h: (batch_size, num_nodes, in_features)
        projection_matrix: (batch_size, num_nodes, num_nodes) 邻接矩阵
        similarity_matrix: (batch_size, num_nodes, num_nodes) 轨迹相似性矩阵 [-1, 1]
        quadrant_matrix: (batch_size, num_nodes, num_nodes, 2) 象限关系矩阵
        """
        batch_size, num_nodes, _ = h.shape

        Wh = self.W(h)  # (batch_size, num_nodes, out_features * heads)
        Wh = Wh.view(batch_size, num_nodes, self.heads, self.out_features)

        Wh_heads = Wh.permute(0, 2, 1, 3)

        Wh_i = Wh_heads.unsqueeze(3)  # (batch_size, heads, num_nodes, 1, out_features)
        Wh_j = Wh_heads.unsqueeze(2)  # (batch_size, heads, 1, num_nodes, out_features)

        Wh_concat = torch.cat([Wh_i.expand(-1, -1, -1, num_nodes, -1),
                               Wh_j.expand(-1, -1, num_nodes, -1, -1)], dim=-1)

        e = torch.einsum('bhijk,hkl->bhij', Wh_concat, self.a).squeeze(-1)
        e = self.leakyrelu(e)
        if projection_matrix is not None:

            projection_expanded = projection_matrix.unsqueeze(1).expand(-1, self.heads, -1,
                                                                        -1)  # (batch_size, heads, num_nodes, num_nodes)

            projection_flat = projection_expanded.reshape(-1, 1)
            projection_transformed = self.projection_transform(projection_flat)

            projection_transformed = projection_transformed.reshape(batch_size, self.heads, num_nodes, num_nodes,
                                                                    self.out_features)

            Wh_i_expanded = Wh_heads.unsqueeze(2).expand(-1, -1, num_nodes, -1,
                                                         -1)  # (batch_size, heads, num_nodes, num_nodes, out_features)
            projection_bias = torch.einsum('bhijk,bhijk->bhij', Wh_i_expanded, projection_transformed)
            e = e + projection_bias

        if similarity_matrix is not None:

            similarity_expanded = similarity_matrix.unsqueeze(1).expand(-1, self.heads, -1,
                                                                        -1)  # (batch_size, heads, num_nodes, num_nodes)

            similarity_flat = similarity_expanded.reshape(-1, 1)
            similarity_transformed = self.similarity_transform(similarity_flat)

            similarity_transformed = similarity_transformed.reshape(batch_size, self.heads, num_nodes, num_nodes,
                                                                    self.out_features)


            Wh_i_expanded = Wh_heads.unsqueeze(2).expand(-1, -1, num_nodes, -1,
                                                         -1)  # (batch_size, heads, num_nodes, num_nodes, out_features)
            similarity_bias = torch.einsum('bhijk,bhijk->bhij', Wh_i_expanded, similarity_transformed)
            e = e + similarity_bias

        # 添加象限一致性偏置 - 向量化处理
        if quadrant_matrix is not None:
            # 批量获取象限偏置
            quadrant_bias = self.quadrant_bias[
                quadrant_matrix[:, :, :, 0],  # (batch_size, num_nodes, num_nodes)
                quadrant_matrix[:, :, :, 1]  # (batch_size, num_nodes, num_nodes)
            ]  # (batch_size, num_nodes, num_nodes)


            quadrant_bias = quadrant_bias.unsqueeze(1).expand(-1, self.heads, -1,
                                                              -1)  # (batch_size, heads, num_nodes, num_nodes)
            e = e + quadrant_bias
        masked_e = torch.full_like(e, -1e9)


         # 1. Select Top-K neighbors
        top_scores, top_indices = torch.topk(e, top_k, dim=-1)
        masked_e.scatter_(-1, top_indices, top_scores)
            
        # 2. Select Bottom-K neighbors
        # We use topk on the negative scores to find the minimums
        bottom_scores_neg, bottom_indices = torch.topk(-e, top_k, dim=-1)
        bottom_scores = -bottom_scores_neg
        masked_e.scatter_(-1, bottom_indices, bottom_scores)
        attention = F.softmax(masked_e, dim=-1)
        # attention = F.softmax(e, dim=-1)  # (batch_size, heads, num_nodes, num_nodes)
        attention = self.dropout(attention)

        h_prime = torch.einsum('bhij,bhjk->bhik', attention, Wh_heads)  # (batch_size, heads, num_nodes, out_features)

        if self.heads > 1:

            h_prime = h_prime.permute(0, 2, 1, 3).contiguous().view(batch_size, num_nodes,
                                                                    -1)  # (batch_size, num_nodes, heads * out_features)
            h_prime = self.head_concat(h_prime)
        else:
            h_prime = h_prime.squeeze(1)  # (batch_size, num_nodes, out_features)

        return F.elu(h_prime)


class MultiScaleGRU(nn.Module):
    """更激进的多尺度GRU优化版本，尝试并行处理"""

    def __init__(self, input_size, hidden_size, num_layers=2, scales=[1, 3, 5], dropout=0.1):
        super(MultiScaleGRU, self).__init__()
        self.scales = scales
        self.hidden_size = hidden_size
        self.num_scales = len(scales)

        # 为每个尺度创建独立的GRU，但使用相同参数初始化
        self.grus = nn.ModuleList([
            nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            for _ in scales
        ])

        # 初始化所有GRU参数相同
        for i in range(1, len(self.grus)):
            self.grus[i].load_state_dict(self.grus[0].state_dict())

        # 尺度注意力机制
        self.scale_attention = nn.Linear(hidden_size * self.num_scales, self.num_scales)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (batch_size, seq_len, input_size)
        """
        batch_size, seq_len, input_size = x.shape

        # 准备所有尺度的输入
        scale_inputs = []
        max_len = 0

        for scale in self.scales:
            if scale > 1:
                indices = torch.arange(0, seq_len, scale, device=x.device)
                x_scale = x.index_select(1, indices)
                target_len = (seq_len + scale - 1) // scale
                if x_scale.size(1) < target_len:
                    pad_len = target_len - x_scale.size(1)
                    x_scale = F.pad(x_scale, (0, 0, 0, pad_len, 0, 0))
            else:
                x_scale = x

            scale_inputs.append(x_scale)
            max_len = max(max_len, x_scale.size(1))

        # 将所有输入填充到相同长度以便并行处理
        padded_inputs = []
        for inp in scale_inputs:
            if inp.size(1) < max_len:
                inp = F.pad(inp, (0, 0, 0, max_len - inp.size(1), 0, 0))
            padded_inputs.append(inp)

        # 堆叠所有输入
        stacked_inputs = torch.stack(padded_inputs, dim=0)  # (num_scales, batch_size, max_len, input_size)
        stacked_inputs = stacked_inputs.transpose(0, 1)  # (batch_size, num_scales, max_len, input_size)

        # 重塑以便并行处理
        batch_size, num_scales, max_len, input_size = stacked_inputs.shape
        stacked_inputs_flat = stacked_inputs.reshape(batch_size * num_scales, max_len, input_size)

        # 并行处理所有尺度
        scale_outputs_flat, _ = self.grus[0](stacked_inputs_flat)  # 使用第一个GRU，但参数相同

        # 恢复形状并取最后一个时间步
        scale_outputs = scale_outputs_flat.reshape(batch_size, num_scales, max_len, self.hidden_size)
        scale_outputs_last = scale_outputs[:, :, -1, :]  # (batch_size, num_scales, hidden_size)

        # 合并多尺度特征
        combined = scale_outputs_last.reshape(batch_size, -1)  # (batch_size, hidden_size * num_scales)

        # 尺度注意力权重
        scale_weights = F.softmax(self.scale_attention(combined), dim=-1)  # (batch_size, num_scales)

        # 加权融合
        final_output = torch.einsum('bsh,bs->bh', scale_outputs_last, scale_weights)

        return self.dropout(final_output)



class SpatioTemporalBlock(nn.Module):
    """
    一个时空块，处理一个时间序列片段，并返回其最终的摘要表示。
    """
    def __init__(self, input_features, hidden_size, gat_heads, num_gru_layers, dropout=0.2):
        super(SpatioTemporalBlock, self).__init__()
        
        self.gat_layer = TimeAwareGATLayer(
            input_features,
            hidden_size,
            heads=gat_heads,
            dropout=dropout
        )
        
        self.gru = MultiScaleGRU(
            hidden_size,
            hidden_size,
            num_layers=num_gru_layers,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        # 残差连接和LayerNorm
        self.proj = nn.Linear(input_features, hidden_size) if input_features != hidden_size else nn.Identity()

    def forward(self, x_chunk, proj_chunk, sim_chunk, quad_chunk, top_k=15):
        """
        x_chunk: 输入时空张量块 (batch, chunk_len, num_nodes, input_features)
        *_chunk: 对应的图结构矩阵块
        """
        batch_size, chunk_len, num_nodes, _ = x_chunk.shape
        
        # 1. 空间建模 (GAT)
        spatial_features = []
        for t in range(chunk_len):
            h_t = checkpoint(
                self.gat_layer, 
                x_chunk[:, t, :, :],
                proj_chunk[:, t], 
                sim_chunk[:, t], 
                quad_chunk[:, t],
                top_k,
                use_reentrant=False
            )
            spatial_features.append(h_t)
        
        spatial_temporal_out = torch.stack(spatial_features, dim=1) # (batch, chunk_len, num_nodes, hidden_size)

        # 2. 时间建模 (GRU) - 向量化
        # (B, S, N, H) -> (B*N, S, H)
        gru_input = spatial_temporal_out.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, chunk_len, -1)
        
        # GRU 返回最后一个时间步的摘要
        gru_summary = self.gru(gru_input) # (B*N, H)
        
        # (B*N, H) -> (B, N, H)
        final_summary = gru_summary.view(batch_size, num_nodes, -1)
        
        return self.dropout(final_summary)


class Stacked_GRU_GAT_Predictor(nn.Module):
    """
    堆叠式的GRU-GAT RRG预测模型，按时间块顺序处理。
    """
    def __init__(self,
                 num_industries=32,
                 input_features=8,
                 output_features=2,
                 hidden_size=128,
                 gat_heads=4,
                 num_gru_layers=2,
                 num_gat_layers=2,
                 pred_len=5,
                 dropout=0.2):

        super(Stacked_GRU_GAT_Predictor, self).__init__()
        self.pred_len = pred_len
        self.num_stacked_layers = num_gat_layers
        self.hidden_size = hidden_size

        self.feature_encoder = nn.Linear(input_features, hidden_size)
        
        self.stacked_blocks = nn.ModuleList()
        current_features = hidden_size
        for _ in range(self.num_stacked_layers):
            self.stacked_blocks.append(
                SpatioTemporalBlock(
                    input_features=current_features,
                    hidden_size=hidden_size,
                    gat_heads=gat_heads,
                    num_gru_layers=num_gru_layers,
                    dropout=dropout
                )
            )
            # 第一个块之后，输入特征维度都是hidden_size
            current_features = hidden_size

        # 为每个块的残差连接添加LayerNorm
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(self.num_stacked_layers)])

        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, pred_len * output_features)
        )
        self.output_features = output_features

    def forward(self, x, stamps=None, ablation_mode=False,top_k=15):
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # 确保序列长度可以被块数整除
        if seq_len % self.num_stacked_layers != 0:
            raise ValueError(f"Sequence length ({seq_len}) must be divisible by num_stacked_layers ({self.num_stacked_layers})")
        
        chunk_len = seq_len // self.num_stacked_layers

        # 预计算所有图结构矩阵
        with torch.no_grad():
            proj_mats = torch.stack([calculate_vector_momentum_projection_matrix_optimized(x, t) for t in range(seq_len)], dim=1)
            sim_mats = torch.stack([calculate_momentum_trajectory_similarity_matrix_optimized(x, t, window_size=10) for t in range(seq_len)], dim=1)
            quad_mats = torch.stack([calculate_quadrant_matrix_optimized(x, t) for t in range(seq_len)], dim=1)
        
        if ablation_mode:
            # 如果是消融模式，用无意义的0矩阵替换真实的图结构信息
            print("--- Running in Ablation Mode: Graph structure info is zeroed out. ---")
            proj_mats = torch.zeros_like(proj_mats)
            sim_mats = torch.zeros_like(sim_mats)
            quad_mats = torch.zeros_like(quad_mats)
        # 1. 初始特征编码
        x_processed = self.feature_encoder(x)

        # 2. 依次通过时间块进行处理
        block_summary = None
        for i in range(self.num_stacked_layers):
            start_idx = i * chunk_len
            end_idx = (i + 1) * chunk_len
            
            x_chunk = x_processed[:, start_idx:end_idx]
            x_chunk = x_chunk.clone()
            proj_chunk = proj_mats[:, start_idx:end_idx]
            sim_chunk = sim_mats[:, start_idx:end_idx]
            quad_chunk = quad_mats[:, start_idx:end_idx]

            if block_summary is not None:
                # 残差连接：将上一个块的摘要与当前块第一个时间步的特征相加 使用LayerNorm稳定训练
                x_chunk[:, 0, :, :] = self.layer_norms[i-1](x_chunk[:, 0, :, :] + block_summary)

            # 通过时空块处理当前块
            block = self.stacked_blocks[i]
            block_summary = block(x_chunk, proj_chunk, sim_chunk, quad_chunk, top_k=top_k)

        # 3. 使用最后一个块的最终摘要进行预测
        final_representation = block_summary # (batch, num_nodes, hidden_size)
        predictions = self.prediction_head(final_representation)

        # 4. 重塑输出
        predictions = predictions.view(batch_size, num_nodes, self.pred_len, self.output_features)
        predictions = predictions.permute(0, 2, 1, 3)

        return predictions
    
class Stacked_GRU_GAT_Predictor_rg(nn.Module):
    """
    堆叠式的GRU-GAT RRG预测模型，按时间块顺序处理。
    """
    def __init__(self,
                 num_industries=32,
                 input_features=8,
                 output_features=2,
                 hidden_size=128,
                 gat_heads=4,
                 num_gru_layers=2,
                 num_gat_layers=2,
                 pred_len=5,
                 dropout=0.2):

        super(Stacked_GRU_GAT_Predictor_rg, self).__init__()
        self.pred_len = pred_len
        self.num_stacked_layers = num_gat_layers
        self.hidden_size = hidden_size

        self.feature_encoder = nn.Linear(input_features, hidden_size)
        
        self.stacked_blocks = nn.ModuleList()
        current_features = hidden_size
        for _ in range(self.num_stacked_layers):
            self.stacked_blocks.append(
                SpatioTemporalBlock(
                    input_features=current_features,
                    hidden_size=hidden_size,
                    gat_heads=gat_heads,
                    num_gru_layers=num_gru_layers,
                    dropout=dropout
                )
            )
            # 第一个块之后，输入特征维度都是hidden_size
            current_features = hidden_size

        # 为每个块的残差连接添加LayerNorm
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(self.num_stacked_layers)])

        self.output_features = output_features
        self.predict_heat = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, output_features)  # 预测RS和动量
            ) for _ in range(pred_len)
        ])

        self.output_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_features, hidden_size//2),
                nn.ELU(),
                nn.Linear(hidden_size//2, hidden_size),
                nn.Dropout(dropout)
            ) for _ in range(pred_len - 1)
        ])

    def forward(self, x, stamps=None, ablation_mode=False,top_k=15):
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # 确保序列长度可以被块数整除
        if seq_len % self.num_stacked_layers != 0:
            raise ValueError(f"Sequence length ({seq_len}) must be divisible by num_stacked_layers ({self.num_stacked_layers})")
        
        chunk_len = seq_len // self.num_stacked_layers

        # 预计算所有图结构矩阵
        with torch.no_grad():
            proj_mats = torch.stack([calculate_vector_momentum_projection_matrix_optimized(x, t) for t in range(seq_len)], dim=1)
            sim_mats = torch.stack([calculate_momentum_trajectory_similarity_matrix_optimized(x, t, window_size=10) for t in range(seq_len)], dim=1)
            quad_mats = torch.stack([calculate_quadrant_matrix_optimized(x, t) for t in range(seq_len)], dim=1)
        
        if ablation_mode:
            # 如果是消融模式，用无意义的0矩阵替换真实的图结构信息
            print("--- Running in Ablation Mode: Graph structure info is zeroed out. ---")
            proj_mats = torch.zeros_like(proj_mats)
            sim_mats = torch.zeros_like(sim_mats)
            quad_mats = torch.zeros_like(quad_mats)
        # 1. 初始特征编码
        x_processed = self.feature_encoder(x)

        # 2. 依次通过时间块进行处理
        block_summary = None
        for i in range(self.num_stacked_layers):
            start_idx = i * chunk_len
            end_idx = (i + 1) * chunk_len
            
            x_chunk = x_processed[:, start_idx:end_idx]
            x_chunk = x_chunk.clone()
            proj_chunk = proj_mats[:, start_idx:end_idx]
            sim_chunk = sim_mats[:, start_idx:end_idx]
            quad_chunk = quad_mats[:, start_idx:end_idx]

            if block_summary is not None:
                # 残差连接：将上一个块的摘要与当前块第一个时间步的特征相加 使用LayerNorm稳定训练
                x_chunk[:, 0, :, :] = self.layer_norms[i-1](x_chunk[:, 0, :, :] + block_summary)

            # 通过时空块处理当前块
            block = self.stacked_blocks[i]
            block_summary = block(x_chunk, proj_chunk, sim_chunk, quad_chunk, top_k=top_k)

        # 3. 使用最后一个块的最终摘要进行预测
        hidden_state = block_summary # (batch, num_nodes, hidden_size)

        outputs = []
        for step in range(self.pred_len):
            step_pred = self.predict_heat[step](hidden_state) # (batch, num_nodes, output_features)
            outputs.append(step_pred.unsqueeze(1)) # (batch, 1, num_nodes, output_features)
            if step < self.pred_len - 1:
                step_pred_encoded = self.output_encoder[step](step_pred) # (batch, num_nodes, hidden_size)
                hidden_state = hidden_state + step_pred_encoded * 0.2  # 小权重更新

        predictions = torch.cat(outputs, dim=1) # (batch, pred_len, num_nodes, output_features)

        return predictions


class GRU_GAT_RRGPredictor(nn.Module):
    """简化的GRU+GAT RRG轨迹预测模型"""

    def __init__(self,
                 num_industries=32,
                 input_features=8,  # RS, 动量, RS变化率, 动量变化率, 极坐标距离, sin, cos, rank
                 output_features=2,  # RS, momentum
                 hidden_size=128,
                 gat_heads=4,
                 num_gat_layers=2,
                 num_gru_layers=2,
                 pred_len=5,
                 dropout=0.2):

        super(GRU_GAT_RRGPredictor, self).__init__()

        self.num_industries = num_industries
        self.hidden_size = hidden_size
        self.pred_len = pred_len
        self.num_gat_layers = num_gat_layers
        self.num_gru_layers = num_gru_layers

        # 特征编码层
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_features, hidden_size),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        self.output_encoder = nn.Sequential(
            nn.Linear(output_features, hidden_size),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        # GAT层 - 空间关系建模
        self.gat_layers = nn.ModuleList()
        gat_input_size = hidden_size
        for i in range(num_gat_layers):
            self.gat_layers.append(
                TimeAwareGATLayer(
                    gat_input_size,
                    hidden_size,
                    heads=gat_heads,
                    dropout=dropout
                )
            )
            gat_input_size = hidden_size

        # GRU层 - 时间序列建模
        self.gru = MultiScaleGRU(
            hidden_size,
            hidden_size,
            num_layers=num_gru_layers,
            dropout=dropout
        )

        # 时空融合层
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

        # 输出层 - 直接预测RS和动量
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(64, output_features)  # 预测RS和动量
            ) for _ in range(pred_len)
        ])

        self.dropout = nn.Dropout(dropout)



    def forward(self, x, stamps=None, top_k=15):
        """
        x: (batch_size, seq_len, num_industries, input_features)
        adj_matrix: (num_industries, num_industries)
        similarity_matrix : (num_industries, num_industries)
        quadrant_matrix: (num_industries, num_industries, 2)
        """
        batch_size, seq_len, num_industries, input_features = x.shape

        # 特征编码
        x_encoded = self.feature_encoder(x)  # (batch_size, seq_len, num_industries, hidden_size)

        temporal_features = []
        grad_step=seq_len//self.num_gat_layers
        # 对每个时间步处理

        with torch.no_grad():
            projection_matrix_new = torch.stack([calculate_vector_momentum_projection_matrix_optimized(x, t) for t in range(seq_len)], dim=1)
            similarity_matrix_new = torch.stack([calculate_momentum_trajectory_similarity_matrix_optimized(x, t, window_size=10) for t in range(seq_len)], dim=1)
            quadrant_matrix_new = torch.stack([calculate_quadrant_matrix_optimized(x, t) for t in range(seq_len)], dim=1)

        for t in range(seq_len):

            # (batch_size, num_industries, hidden_size)
            if (t+1) % grad_step:
                with torch.no_grad():
                    h_t = self.gat_layers[(t // grad_step) % self.num_gat_layers]( 
                            x_encoded[:, t, :, :].clone(),
                            projection_matrix_new[:, t], 
                            similarity_matrix_new[:, t], 
                            quadrant_matrix_new[:, t],
                            top_k,
                        )
            else:
                h_t = self.gat_layers[(t // grad_step) % self.num_gat_layers]( 
                            x_encoded[:, t, :, :].clone(),
                            projection_matrix_new[:, t], 
                            similarity_matrix_new[:, t], 
                            quadrant_matrix_new[:, t],
                            top_k,
                        )
            temporal_features.append(h_t)
        spatial_temporal = torch.stack(temporal_features, dim=1)  # (batch_size, seq_len, num_industries, hidden_size)

        # 对每个行业进行时间序列建模
        industry_predictions = []

        for i in range(num_industries):
            industry_seq = spatial_temporal[:, :, i, :]  # (batch_size, seq_len, hidden_size)

            # GRU时间序列建模
            gru_out = self.gru(industry_seq)  # (batch_size, hidden_size)

            # 时空特征融合
            spatial_feat = spatial_temporal[:, -1, i, :]  # 最后一个时间步的空间特征
            fusion_weight = self.fusion_gate(torch.cat([gru_out, spatial_feat], dim=-1))
            fused_feat = fusion_weight * gru_out + (1 - fusion_weight) * spatial_feat

            # 多步预测
            industry_preds = []
            hidden = fused_feat

            for step in range(self.pred_len):
                pred = self.output_layers[step](hidden)
                industry_preds.append(pred.unsqueeze(1))

                # 使用预测更新隐藏状态（自回归）
                if step < self.pred_len - 1:
                    # 将预测结果编码用于下一步
                    pred_encoded = self.output_encoder(pred).squeeze(1)
                    hidden = hidden + pred_encoded * 0.2  # 小权重更新

            industry_predictions.append(torch.cat(industry_preds, dim=1))

        # 组合所有行业的预测
        predictions = torch.stack(industry_predictions, dim=2)  # (batch_size, pred_len, num_industries, 2)

        return predictions

# 使用示例
# if __name__ == "__main__":
#     # 模型参数
#     num_industries = 32
#     seq_len = 200
#     pred_len = 5
#     batch_size = 16
#     input_features = 7  # RS, 动量, RS变化率, 动量变化率, 极坐标距离, sin, cos
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # 创建模型
#     model = GRU_GAT_RRGPredictor(
#         num_industries=num_industries,
#         input_features=input_features,
#         pred_len=pred_len
#     ).to(device)
#
#     # 示例输入
#     x = torch.randn(batch_size, seq_len, num_industries, input_features, device=device)
#     # adj_matrix = torch.randn(batch_size,num_industries, num_industries, device=device)  # 全连接图
#     # similarity_matrix = torch.randn(batch_size,num_industries, num_industries, device=device)  # 领先滞后关系
#     # quadrant_matrix = torch.randint(0, 4, (batch_size,num_industries, num_industries, 2), device=device)  # 象限关系
#     import time
#     # 前向传播
#     # with torch.no_grad():
#     if 1:
#         start_time = time.time()
#         output = model(x)
#         new_time = time.time() - start_time
#
#         print(f"优化后计算时间: {new_time:.4f}秒")
#         print(f"输入形状: {x.shape}")
#         print(f"输出形状: {output.shape}")  # 应该是 (batch_size, pred_len, num_industries, 2)
#
#         # 参数统计
#         total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         print(f"总参数量: {total_params:,}")