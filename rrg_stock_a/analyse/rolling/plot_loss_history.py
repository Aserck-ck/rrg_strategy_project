import json
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import numpy as np


def plot_loss_from_json(json_path):
    """
    读取滚动训练的进度json文件，并绘制验证损失的折线图。

    参数:
    json_path (str): 进度json文件的路径。
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 文件未找到 -> {json_path}")
        return
    except json.JSONDecodeError:
        print(f"错误: 文件格式不正确，无法解析JSON -> {json_path}")
        return

    # --- 使用pandas进行数据转换，更健壮、更简洁 ---
    # 将 performance_history 字典转换为DataFrame
    # orient='index' 会让字典的键（如'500', '520'）成为DataFrame的行索引
    df = pd.DataFrame.from_dict(data['performance_history'], orient='index')

    # 将字符串索引（'500'）转换为数值类型，以便正确排序和绘图
    df.index = pd.to_numeric(df.index)

    # 按索引排序，确保X轴是递增的
    df = df.sort_index()

    # --- 设置绘图样式 ---
    matplotlib.rcParams['font.family'] = 'SimHei'  # 支持中文显示
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
    
    # plt.style.use('seaborn-v0_8-whitegrid') # 使用一个美观的样式
    fig, ax = plt.subplots(figsize=(16, 9))

    # --- 绘制每一条折线 ---
    # 遍历DataFrame的每一列（即每个模型）
    for model_name in df.columns:
        # plot会自动忽略NaN值，这对于数据缺失的情况非常友好
        ax.plot(df.index, df[model_name], marker='o', linestyle='-', label=model_name)

    # --- 美化图表 ---
    ax.set_title('滚动训练验证损失变化曲线 (Rolling Training Validation Loss)', fontsize=18, pad=20)
    ax.set_xlabel('滚动训练数据点 (Outer Loop Index)', fontsize=12)
    ax.set_ylabel('验证集损失 (Validation Loss)', fontsize=12)
    
    # 优化图例显示
    legend = ax.legend(title='模型/配置', fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.setp(legend.get_title(), fontsize=12)

    # 优化刻度
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # 调整布局以防止图例被截断
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # 为右侧的图例留出空间
    
    # 显示网格
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    # 显示图表
    plt.show()


if __name__ == '__main__':
    # 指定您的json文件路径
    progress_file_path = 'd:\\files\\qt\\guangfa\\rrg_stock_a\\analyse\\rolling\\stack_rg_pred2_walk_forward_progress.json'
    plot_loss_from_json(progress_file_path)