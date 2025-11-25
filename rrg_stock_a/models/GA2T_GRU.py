import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from rrg_stock_a.models.module import (
    calculate_trajectory_similarity_matrix_optimized,
    calculate_quadrant_matrix_optimized,
    calculate_vector_projection_matrix_optimized, )

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

    def forward(self, h, projection_matrix=None, similarity_matrix=None, quadrant_matrix=None):
        """
        h: (batch_size, num_nodes, in_features)
        projection_matrix: (batch_size, num_nodes, num_nodes) 邻接矩阵
        similarity_matrix: (batch_size, num_nodes, num_nodes) 轨迹相似性矩阵 [-1, 1]
        quadrant_matrix: (batch_size, num_nodes, num_nodes, 2) 象限关系矩阵
        """
        batch_size, num_nodes, _ = h.shape

        # 线性变换 - 保持原始形状
        Wh = self.W(h)  # (batch_size, num_nodes, out_features * heads)
        Wh = Wh.view(batch_size, num_nodes, self.heads, self.out_features)

        # 重塑为多头格式: (batch_size, heads, num_nodes, out_features)
        Wh_heads = Wh.permute(0, 2, 1, 3)

        # 计算基础注意力分数 - 向量化多头计算
        Wh_i = Wh_heads.unsqueeze(3)  # (batch_size, heads, num_nodes, 1, out_features)
        Wh_j = Wh_heads.unsqueeze(2)  # (batch_size, heads, 1, num_nodes, out_features)

        # 拼接特征并计算注意力 - 使用广播避免显式扩展
        Wh_concat = torch.cat([Wh_i.expand(-1, -1, -1, num_nodes, -1),
                               Wh_j.expand(-1, -1, num_nodes, -1, -1)], dim=-1)

        # 使用einsum进行高效矩阵乘法
        e = torch.einsum('bhijk,hkl->bhij', Wh_concat, self.a).squeeze(-1)
        e = self.leakyrelu(e)
        if projection_matrix is not None:
            # 重塑投影矩阵以匹配多头格式
            projection_expanded = projection_matrix.unsqueeze(1).expand(-1, self.heads, -1,
                                                                        -1)  # (batch_size, heads, num_nodes, num_nodes)

            # 首先将投影矩阵重塑为(batch_size*heads*num_nodes*num_nodes, 1)
            projection_flat = projection_expanded.reshape(-1, 1)
            projection_transformed = self.projection_transform(projection_flat)
            # 重塑回原始形状
            projection_transformed = projection_transformed.reshape(batch_size, self.heads, num_nodes, num_nodes,
                                                                    self.out_features)

            # 计算投影偏置 - 使用einsum提高效率
            Wh_i_expanded = Wh_heads.unsqueeze(2).expand(-1, -1, num_nodes, -1,
                                                         -1)  # (batch_size, heads, num_nodes, num_nodes, out_features)
            projection_bias = torch.einsum('bhijk,bhijk->bhij', Wh_i_expanded, projection_transformed)
            e = e + projection_bias
        # 添加轨迹相似性偏置 - 向量化处理连续值
        if similarity_matrix is not None:
            # 重塑相似性矩阵以匹配多头格式
            similarity_expanded = similarity_matrix.unsqueeze(1).expand(-1, self.heads, -1,
                                                                        -1)  # (batch_size, heads, num_nodes, num_nodes)

            # 使用线性变换处理连续相似性值
            # 首先将相似性矩阵重塑为(batch_size*heads*num_nodes*num_nodes, 1)
            similarity_flat = similarity_expanded.reshape(-1, 1)
            similarity_transformed = self.similarity_transform(similarity_flat)
            # 重塑回原始形状
            similarity_transformed = similarity_transformed.reshape(batch_size, self.heads, num_nodes, num_nodes,
                                                                    self.out_features)

            # 计算相似性偏置 - 使用einsum提高效率
            Wh_i_expanded = Wh_heads.unsqueeze(2).expand(-1, -1, num_nodes, -1,
                                                         -1)  # (batch_size, heads, num_nodes, num_nodes, out_features)
            similarity_bias = torch.einsum('bhijk,bhijk->bhij', Wh_i_expanded, similarity_transformed)
            e = e + similarity_bias

        # 添加象限一致性偏置 - 向量化处理
        if quadrant_matrix is not None:
            # 批量获取象限偏置
            quadrant_bias = self.quadrant_bias[
                quadrant_matrix[:, :, :, 0],  # 源节点象限 (batch_size, num_nodes, num_nodes)
                quadrant_matrix[:, :, :, 1]  # 目标节点象限 (batch_size, num_nodes, num_nodes)
            ]  # (batch_size, num_nodes, num_nodes)

            # 重塑以匹配多头格式
            quadrant_bias = quadrant_bias.unsqueeze(1).expand(-1, self.heads, -1,
                                                              -1)  # (batch_size, heads, num_nodes, num_nodes)
            e = e + quadrant_bias

        # 注意力权重计算
        attention = F.softmax(e, dim=-1)  # (batch_size, heads, num_nodes, num_nodes)
        attention = self.dropout(attention)

        # 应用注意力权重 - 使用einsum提高效率
        h_prime = torch.einsum('bhij,bhjk->bhik', attention, Wh_heads)  # (batch_size, heads, num_nodes, out_features)

        # 合并多头输出
        if self.heads > 1:
            # 使用线性变换合并多头，而不是简单平均
            h_prime = h_prime.permute(0, 2, 1, 3).contiguous().view(batch_size, num_nodes,
                                                                    -1)  # (batch_size, num_nodes, heads * out_features)
            h_prime = self.head_concat(h_prime)
        else:
            h_prime = h_prime.squeeze(1)  # (batch_size, num_nodes, out_features)

        return F.elu(h_prime)


import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    """线性时间感知注意力层，针对大规模关系矩阵优化内存使用"""

    def __init__(self, in_features, out_features, heads=2, dropout=0.2, feature_dim=64):
        super(LinearAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.feature_dim = feature_dim

        # 线性变换参数
        self.W = nn.Linear(in_features, out_features * heads, bias=False)

        # 随机特征映射 - 用于线性注意力
        self.random_projection = nn.Parameter(
            torch.randn(in_features, feature_dim) * 0.1
        )

        # 注意力机制参数 - 使用线性注意力形式
        self.a_q = nn.Linear(feature_dim * 2, out_features, bias=False)
        self.a_k = nn.Linear(feature_dim * 2, out_features, bias=False)

        # 关系矩阵变换 - 使用更高效的方式
        self.projection_transform = nn.Linear(1, out_features, bias=False)
        self.similarity_transform = nn.Linear(1, out_features, bias=False)

        # 象限一致性偏置
        self.quadrant_bias = nn.Parameter(torch.zeros(4, 4))

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(dropout)

        # 输出变换
        self.output_proj = nn.Linear(out_features * heads, out_features, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_q.weight)
        nn.init.xavier_uniform_(self.a_k.weight)
        nn.init.xavier_uniform_(self.projection_transform.weight)
        nn.init.xavier_uniform_(self.similarity_transform.weight)
        nn.init.zeros_(self.quadrant_bias)

    def random_feature_map(self, x):
        """随机特征映射 - 实现线性注意力"""
        # x: [batch_size, num_nodes, in_features]
        projection = torch.einsum('bnd,df->bnf', x, self.random_projection)

        # 使用随机傅里叶特征
        cos_features = torch.cos(projection)
        sin_features = torch.sin(projection)

        return torch.cat([cos_features, sin_features], dim=-1)

    def forward(self, h, projection_matrix=None, similarity_matrix=None, quadrant_matrix=None):
        """
        h: (batch_size, num_nodes, in_features)
        projection_matrix: (batch_size, num_nodes, num_nodes) 邻接矩阵
        similarity_matrix: (batch_size, num_nodes, num_nodes) 轨迹相似性矩阵 [-1, 1]
        quadrant_matrix: (batch_size, num_nodes, num_nodes, 2) 象限关系矩阵
        """
        batch_size, num_nodes, _ = h.shape

        # 应用随机特征映射
        phi_h = self.random_feature_map(h)  # [batch_size, num_nodes, feature_dim*2]

        # 线性变换生成Q和K
        Q = self.a_q(phi_h)  # [batch_size, num_nodes, out_features]
        K = self.a_k(phi_h)  # [batch_size, num_nodes, out_features]

        # 线性注意力计算: softmax(Q) * (K^T * V) 的高效近似
        # 这里V就是原始特征h的变换
        Wh = self.W(h)  # [batch_size, num_nodes, out_features * heads]
        Wh = Wh.view(batch_size, num_nodes, self.heads, self.out_features)

        # 计算注意力权重（线性复杂度）
        attention_weights = self.compute_efficient_attention_weights(
            Q, K, projection_matrix, similarity_matrix, quadrant_matrix, batch_size, num_nodes
        )

        # 应用注意力
        h_prime = torch.einsum('bhij,bhjk->bhik', attention_weights, Wh)

        # 合并多头
        h_prime = h_prime.permute(0, 2, 1, 3).contiguous().view(
            batch_size, num_nodes, -1
        )
        h_prime = self.output_proj(h_prime)

        return F.elu(h_prime)

    def compute_efficient_attention_weights(self, Q, K, projection_matrix, similarity_matrix, quadrant_matrix,
                                            batch_size, num_nodes):
        """高效计算注意力权重"""
        # 基础线性注意力分数
        attention_scores = torch.bmm(Q, K.transpose(1, 2))  # [batch_size, num_nodes, num_nodes]
        attention_scores = attention_scores.unsqueeze(1).expand(-1, self.heads, -1, -1)

        # 添加投影矩阵偏置（如果提供）
        if projection_matrix is not None:
            # 使用低内存方式处理投影矩阵
            projection_bias = self.compute_projection_bias(projection_matrix, batch_size, num_nodes)
            attention_scores = attention_scores + projection_bias

        # 添加相似性矩阵偏置（如果提供）
        if similarity_matrix is not None:
            # 使用低内存方式处理相似性矩阵
            similarity_bias = self.compute_similarity_bias(similarity_matrix, batch_size, num_nodes)
            attention_scores = attention_scores + similarity_bias

        # 添加象限一致性偏置（如果提供）
        if quadrant_matrix is not None:
            quadrant_bias = self.quadrant_bias[
                quadrant_matrix[:, :, :, 0],  # 源节点象限
                quadrant_matrix[:, :, :, 1]  # 目标节点象限
            ]
            quadrant_bias = quadrant_bias.unsqueeze(1).expand(-1, self.heads, -1, -1)
            attention_scores = attention_scores + quadrant_bias

        # 应用softmax和dropout
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        return attention_weights

    def compute_projection_bias(self, projection_matrix, batch_size, num_nodes):
        """计算投影矩阵偏置 - 内存优化版本"""
        # 重塑投影矩阵并使用线性变换
        projection_flat = projection_matrix.reshape(batch_size * num_nodes * num_nodes, 1)
        projection_transformed = self.projection_transform(projection_flat)
        projection_transformed = projection_transformed.reshape(batch_size, num_nodes, num_nodes, self.out_features)

        # 计算偏置 - 使用更高效的方式
        # 这里我们简化计算，只使用投影矩阵本身作为偏置
        projection_bias = projection_matrix.unsqueeze(1).expand(-1, self.heads, -1, -1)
        return projection_bias

    def compute_similarity_bias(self, similarity_matrix, batch_size, num_nodes):
        """计算相似性矩阵偏置 - 内存优化版本"""
        # 重塑相似性矩阵并使用线性变换
        similarity_flat = similarity_matrix.reshape(batch_size * num_nodes * num_nodes, 1)
        similarity_transformed = self.similarity_transform(similarity_flat)
        similarity_transformed = similarity_transformed.reshape(batch_size, num_nodes, num_nodes, self.out_features)

        # 计算偏置 - 使用更高效的方式
        # 这里我们简化计算，只使用相似性矩阵本身作为偏置
        similarity_bias = similarity_matrix.unsqueeze(1).expand(-1, self.heads, -1, -1)
        return similarity_bias




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


class GRU_GA2T_RRGPredictor(nn.Module):
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

        super(GRU_GA2T_RRGPredictor, self).__init__()

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



    def forward(self, x):
        """
        x: (batch_size, seq_len, num_industries, input_features)
        adj_matrix: (num_industries, num_industries)
        similarity_matrix : (num_industries, num_industries)
        quadrant_matrix: (num_industries, num_industries, 2)
        """
        batch_size, seq_len, num_industries, input_features = x.shape

        # 特征编码
        x_encoded = self.feature_encoder(x)  # (batch_size, seq_len, num_industries, hidden_size)

        # 添加位置编码


        # 存储时空特征
        temporal_features = []
        grad_step=seq_len//self.num_gat_layers
        # 对每个时间步处理
        projection_matrix_new = torch.zeros((batch_size, seq_len, num_industries, num_industries),device=x.device)
        similarity_matrix_new = torch.zeros((batch_size, seq_len, num_industries, num_industries),device=x.device)
        quadrant_matrix_new = torch.zeros((batch_size, seq_len, num_industries, num_industries, 2),device=x.device).int()
        with torch.no_grad():
            for t in range(seq_len):
                projection_matrix_new[:, t] = calculate_vector_projection_matrix_optimized(x, t)
                similarity_matrix_new[:, t] = calculate_trajectory_similarity_matrix_optimized(x, t,
                                                                                         window_size=10)
                quadrant_matrix_new[:, t] = calculate_quadrant_matrix_optimized(x, t)

        for t in range(seq_len):

            h_t = x_encoded[:, t, :, :].clone()  # (batch_size, num_industries, hidden_size)
            # GAT空间关系建模
            if (t+1) % grad_step:
                with torch.no_grad():
                    h_t = self.gat_layers[t // grad_step](h_t, projection_matrix_new[:, t], similarity_matrix_new[:, t],
                                    quadrant_matrix_new[:, t])
                    h_t = self.dropout(h_t)
            else:
                h_t = self.gat_layers[t // grad_step](h_t, projection_matrix_new[:, t],
                                                            similarity_matrix_new[:, t], quadrant_matrix_new[:, t])
                h_t = self.dropout(h_t)
            temporal_features.append(h_t)

        # del projection_matrix_new, similarity_matrix_new, quadrant_matrix_new
        #
        # torch.cuda.empty_cache()
        # 堆叠时间维度
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