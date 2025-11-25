import pandas as pd
import numpy as np
# 不再需要 import cvxpy as cp
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
# 假设您的数据获取函数在这个路径下
from market_data.sw_level1.sw_level1_functions import get_sw_level1_codes, get_sw_level1_data

# 使用 'agg' backend 来避免在服务器环境中出现GUI问题
plt.switch_backend('agg')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
warnings.filterwarnings('ignore')

# --- 1. 数据获取 (与之前相同) ---
def get_sw1_daily_data(start_date, end_date):
    """
    获取所有申万一级行业在指定期间的日线数据
    """
    print("正在获取申万一级行业日线数据...")
    ts_codes = get_sw_level1_codes()
    
    all_data = []
    for code in ts_codes:
        try:
            # 确保 get_sw_level1_data 返回的DataFrame有 'trade_date' 和 'ts_code' 列
            index_df = get_sw_level1_data(code)
            
            # 筛选日期
            index_df = index_df[(index_df['trade_date'] >= start_date) & (index_df['trade_date'] <= end_date)]
            
            # 【核心修正】在设置索引前，先删除基于 ts_code 和 trade_date 的重复行
            index_df = index_df.drop_duplicates(subset=['ts_code', 'trade_date'])
            
            # 设置多级索引
            index_df = index_df.set_index(['ts_code', 'trade_date'])
            
            if not index_df.empty:
                all_data.append(index_df)
        except Exception as e:
            print(f"获取代码 {code} 数据时出错: {e}")

    if not all_data:
        return pd.DataFrame()
        
    # 合并时，索引会自然对齐
    return pd.concat(all_data)

# --- 2. 因子构建与面板数据准备 (与之前相同) ---
class PanelDataBuilder:
    def __init__(self, start_date, end_date, future_return_window=1):
        self.start_date = start_date
        self.end_date = end_date
        self.future_return_window = future_return_window
        self.panel_df = None

    def _get_weekly_data(self, daily_data):
        if daily_data.empty:
            return pd.DataFrame()
        
        # daily_data 已经有了 (ts_code, trade_date) 的 MultiIndex
        # 我们直接在 level='trade_date' 上进行重采样
        # 【核心修改】不再 reset_index()，直接返回带有 MultiIndex 的 DataFrame
        weekly_data = daily_data.groupby(level='ts_code').resample('W-FRI', level='trade_date').last()
        
        print(f"周度数据准备完成，共 {len(weekly_data)} 条记录。")
        return weekly_data

    def _calculate_factors(self, df):
        print("开始计算因子...")
        # 【核心修改】df 是一个带有 MultiIndex (ts_code, trade_date) 的 DataFrame
        # 我们在这里执行唯一的一次 groupby 操作
        
        # 创建一个列表来存储所有计算出的因子Series
        factor_list = []

        # A. 动量因子 (Momentum)
        # 对整个DataFrame按ts_code分组，然后计算pct_change
        factor_list.append(df.groupby(level='ts_code')['close'].pct_change(4).rename('MOM_4W'))
        # factor_list.append(df.groupby(level='ts_code')['close'].pct_change(12).rename('MOM_12W'))
        # factor_list.append(df.groupby(level='ts_code')['close'].pct_change(52).rename('MOM_52W'))
        
        # B. 价值因子 (Value)
        factor_list.append((1 / df.groupby(level='ts_code')['pe'].rolling(4).mean().droplevel(0)).rename('EP'))


        # C. 波动率因子 (Volatility)
        weekly_pct_change = df.groupby(level='ts_code')['close'].pct_change()
        factor_list.append(weekly_pct_change.groupby(level='ts_code').rolling(12).std().droplevel(0).rename('VOL_12W'))

        # D. 流动性因子 (Liquidity)
        factor_list.append(df.groupby(level='ts_code')['amount'].rolling(4).mean().droplevel(0).rename('AMOUNT_4W_AVG'))
        
        # 计算未来收益率 (Y)
        factor_list.append(df.groupby(level='ts_code')['close'].pct_change(self.future_return_window).shift(-self.future_return_window).rename('FUT_RET'))
        
        # 使用 pd.concat 将所有因子Series一次性合并成一个DataFrame
        # axis=1 表示按列合并
        factors = pd.concat(factor_list, axis=1)
        
        # 在所有计算和合并完成后，才进行 reset_index
        factors = factors.reset_index()
        # 重命名 trade_date 为 date
        factors = factors.rename(columns={'trade_date': 'date'})
        print("因子计算完成。")
        return factors

    def _standardize_factors(self, df):
        print("开始截面标准化因子...")
        factor_names = self.get_factor_names(df)
        df_std = df.set_index(['date', 'ts_code'])
        
        def z_score(group):
            return (group - group.mean()) / group.std()

        df_std[factor_names] = df_std.groupby('date')[factor_names].transform(z_score)
        
        print("因子标准化完成。")
        return df_std.reset_index()

    def build(self):
        # 因子计算需要约1年历史数据，滚动回归需要约1-2年，所以多获取一些
        fetch_start_date = (pd.to_datetime(self.start_date) - timedelta(days=365*3)).strftime('%Y-%m-%d')
        daily_data = get_sw1_daily_data(fetch_start_date, self.end_date)
        
        if daily_data.empty:
            print("未能获取到任何日度数据，构建失败。")
            return pd.DataFrame()

        weekly_data = self._get_weekly_data(daily_data)
        factors_df = self._calculate_factors(weekly_data)
        panel_df_std = self._standardize_factors(factors_df)
        
        # dropna() 会删除任何包含NaN的行，这会消耗掉部分早期数据
        final_panel = panel_df_std.dropna().sort_values(by=['date', 'ts_code'])
        
        self.panel_df = final_panel
        print(f"最终面板数据构建完成，分析周期：{self.start_date} 到 {self.end_date}")
        print(f"数据维度 (行, 列): {self.panel_df.shape}")
        print(f"包含的因子: {self.get_factor_names()}")
        return self.panel_df

    def get_factor_names(self, df=None):
        if df is None: df = self.panel_df
        if df is None: return []
        # 移除了BP因子
        return [col for col in df.columns if col not in ['date', 'ts_code', 'FUT_RET', 'BP']]

# --- 3. 【重大修改】使用滚动窗口Lasso回归替代TV-PR ---
class RollingLassoAnalyzer:
    def __init__(self, panel_df, factor_names):
        self.panel_df = panel_df
        self.factor_names = factor_names
        # 筛选出在分析周期内的数据
        self.analysis_df = panel_df[panel_df['date'] >= pd.to_datetime(ANALYSIS_START_DATE)].copy()
        self.dates = sorted(self.analysis_df['date'].unique())
        self.T = len(self.dates)
        self.K = len(factor_names)
        self.beta_hat = None

    def solve(self, window_size=104, alpha=0.01):
        """
        执行滚动窗口Lasso回归
        :param window_size: 滚动窗口大小（周数），例如104周(2年)
        :param alpha: Lasso回归的正则化强度，相当于之前的 lambda_l1
        """
        print(f"开始执行滚动Lasso回归... 窗口大小: {window_size}周, Alpha: {alpha}")
        
        all_betas = []
        
        # 遍历每个分析期的时间点
        for i, current_date in enumerate(self.dates):
            if i % 20 == 0:
                print(f"进度: {i+1}/{self.T} ({current_date.date()})")

            # 定义训练数据的窗口期
            window_end_date = current_date - timedelta(days=1) # 训练数据不包含当天
            window_start_date = window_end_date - timedelta(weeks=window_size)
            
            # 提取窗口期内的训练数据
            train_data = self.panel_df[
                (self.panel_df['date'] >= window_start_date) & 
                (self.panel_df['date'] <= window_end_date)
            ]
            
            if train_data.shape[0] < self.K * 5: # 确保有足够的训练样本
                # 如果数据不足，使用上一次的beta值，或者用0向量
                last_beta = all_betas[-1] if all_betas else np.zeros(self.K)
                all_betas.append(last_beta)
                continue

            X_train = train_data[self.factor_names].values
            Y_train = train_data['FUT_RET'].values
            
            # 训练Lasso模型
            model = Lasso(alpha=alpha, fit_intercept=False) # 因子已经标准化，无需截距
            model.fit(X_train, Y_train)
            
            # 保存当前估计的因子收益率
            all_betas.append(model.coef_)

        # 将结果整理成 (K, T) 的矩阵，与之前格式保持一致
        self.beta_hat = np.array(all_betas).T
        print("滚动回归求解完成！")
        return self.beta_hat

    def plot_factor_premia(self, save_path='factor_premia.png'):
        if self.beta_hat is None:
            print("没有可供可视化的因子收益率。")
            return

        print(f"正在绘制因子收益率图像并保存至 {save_path}...")
        K, T = self.beta_hat.shape
        num_cols = 3
        num_rows = (K + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 4 * num_rows), sharex=True)
        axes = axes.flatten()

        for i in range(K):
            ax = axes[i]
            ax.plot(self.dates, self.beta_hat[i, :], label=self.factor_names[i])
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
            ax.set_title(f'因子收益率: {self.factor_names[i]}')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=20)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        for i in range(K, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print("图像保存完毕。")

    # 【新增】绘制因子权重构成图
    def plot_factor_weight_composition(self, save_path='factor_weight_composition.png'):
        """
        绘制因子权重构成随时间变化的堆叠面积图
        """
        if self.beta_hat is None:
            print("没有因子权重，无法绘制构成图。")
            return

        print(f"正在绘制因子权重构成图并保存至 {save_path}...")
        
        # 1. 计算每个时间点上，各因子beta绝对值之和
        abs_betas = np.abs(self.beta_hat)
        total_abs_beta_by_time = np.sum(abs_betas, axis=0)
        
        # 2. 计算相对权重（占比），避免除以0
        # np.divide的where参数可以防止0/0的NaN出现
        relative_weights = np.divide(
            abs_betas, 
            total_abs_beta_by_time, 
            out=np.zeros_like(abs_betas), 
            where=(total_abs_beta_by_time != 0)
        )
        
        # 3. 绘制堆叠面积图
        fig, ax = plt.subplots(figsize=(16, 8))
        
        ax.stackplot(self.dates, relative_weights, labels=self.factor_names, alpha=0.8)
        
        ax.set_title('因子相对重要性（权重）随时间变化图', fontsize=16)
        ax.set_ylabel('因子权重占比', fontsize=12)
        ax.set_xlabel('日期', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1) # 权重占比和为1
        
        # 格式化X轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate() # 自动调整日期标签以防重叠
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print("因子权重构成图保存完毕。")

    def run_backtest(self, save_path='strategy_net_value.png'):
        if self.beta_hat is None:
            print("没有因子收益率，无法进行回测。")
            return

        print("开始执行因子轮动策略回测...")
        portfolio_returns = []
        
        for t, date in enumerate(self.dates):
            current_data = self.analysis_df[self.analysis_df['date'] == date]
            if current_data.empty:
                continue
            
            X_t = current_data[self.factor_names].values
            beta_t = self.beta_hat[:, t]
            predicted_returns = X_t @ beta_t
            
            pred_series = pd.Series(predicted_returns, index=current_data['ts_code'])
            pred_series = pred_series.dropna()
            
            q = 5
            labels = [f'G{i}' for i in range(1, q + 1)]
            try:
                groups = pd.qcut(pred_series, q, labels=labels, duplicates='drop')
            except ValueError:
                portfolio_returns.append(0)
                continue

            long_portfolio = groups[groups == f'G{q}'].index
            short_portfolio = groups[groups == 'G1'].index
            
            actual_returns = current_data.set_index('ts_code')['FUT_RET']
            
            long_return = actual_returns.reindex(long_portfolio).mean()
            short_return = actual_returns.reindex(short_portfolio).mean()
            
            if pd.notna(long_return) and pd.notna(short_return):
                period_return = long_return - short_return
                portfolio_returns.append(period_return)
            else:
                portfolio_returns.append(0)

        valid_dates = self.dates[:len(portfolio_returns)]
        net_value = (1 + pd.Series(portfolio_returns, index=valid_dates)).cumprod()
        
        plt.figure(figsize=(12, 6))
        net_value.plot()
        plt.title('滚动Lasso因子轮动策略净值曲线')
        plt.ylabel('Net Value')
        plt.grid(True, alpha=0.5)
        plt.savefig(save_path)
        plt.close()
        print(f"回测完成，净值曲线图已保存至 {save_path}")
        
        if net_value.empty:
            print("无法计算投资组合表现。")
            return
        annual_return = (net_value.iloc[-1] ** (52 / len(net_value)) - 1) * 100
        annual_vol = net_value.pct_change().std() * np.sqrt(52) * 100
        sharpe_ratio = (annual_return / annual_vol) if annual_vol > 0 else 0
        print(f"年化收益率: {annual_return:.2f}%")
        print(f"年化波动率: {annual_vol:.2f}%")
        print(f"夏普比率: {sharpe_ratio:.2f}")

# --- 4. 主程序入口 ---
if __name__ == "__main__":
    # --- 参数设置 ---
    ANALYSIS_START_DATE = '2010-01-01'
    ANALYSIS_END_DATE = '2025-11-04'
    # 滚动Lasso超参数
    ROLLING_WINDOW = 52  # 滚动窗口大小，52周约为1年
    LASSO_ALPHA = 0.00005    # Lasso正则化强度

    # --- 步骤1: 构建面板数据 ---
    builder = PanelDataBuilder(start_date=ANALYSIS_START_DATE, end_date=ANALYSIS_END_DATE)
    panel_data = builder.build()

    if panel_data is not None and not panel_data.empty:
        # --- 步骤2: 运行滚动Lasso分析 ---
        analyzer = RollingLassoAnalyzer(panel_df=panel_data, factor_names=builder.get_factor_names())
        beta_hat = analyzer.solve(window_size=ROLLING_WINDOW, alpha=LASSO_ALPHA)

        if beta_hat is not None:
            # --- 步骤3: 可视化与回测 ---
            analyzer.plot_factor_premia(save_path='multifactor/sw1_factor_premia_rolling_lasso.png')
            # 【新增】调用新的绘图函数
            analyzer.plot_factor_weight_composition(save_path='multifactor/sw1_factor_weight_composition.png')
            analyzer.run_backtest(save_path='multifactor/sw1_strategy_net_value_rolling_lasso.png')
    else:
        print("未能成功构建面板数据，程序终止。")
