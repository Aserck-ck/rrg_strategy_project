import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import cvxpy as cp
import warnings
# 假设您的数据获取函数在这个路径下
from market_data.sw_level1.sw_level1_functions import get_sw_level1_codes, get_sw_level1_data

# --- Matplotlib and Warnings Setup ---
plt.switch_backend('agg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# --- 1. Data Acquisition (Unaltered) ---
def get_sw1_daily_data(start_date, end_date):
    print("正在获取申万一级行业日线数据...")
    ts_codes = get_sw_level1_codes()
    all_data = []
    for code in ts_codes:
        try:
            index_df = get_sw_level1_data(code)
            index_df = index_df[(index_df['trade_date'] >= start_date) & (index_df['trade_date'] <= end_date)]
            index_df = index_df.drop_duplicates(subset=['ts_code', 'trade_date'])
            index_df = index_df.set_index(['ts_code', 'trade_date'])
            if not index_df.empty:
                all_data.append(index_df)
        except Exception as e:
            print(f"获取代码 {code} 数据时出错: {e}")
    if not all_data: return pd.DataFrame()
    return pd.concat(all_data)

# --- 2. Panel Data Builder (Unaltered) ---
class PanelDataBuilder:
    def __init__(self, start_date, end_date, future_return_window=1):
        self.start_date = start_date
        self.end_date = end_date
        self.future_return_window = future_return_window
        self.panel_df = None

    def _get_weekly_data(self, daily_data):
        if daily_data.empty: return pd.DataFrame()
        weekly_data = daily_data.groupby(level='ts_code').resample('W-FRI', level='trade_date').last()
        print(f"周度数据准备完成，共 {len(weekly_data)} 条记录。")
        return weekly_data

    def _calculate_factors(self, df):
        print("开始计算因子...")
        factor_list = []
        factor_list.append(df.groupby(level='ts_code')['close'].pct_change(12).rename('MOM_12W'))
        factor_list.append((1 / df.groupby(level='ts_code')['pe'].rolling(4).mean().droplevel(0)).rename('EP'))
        weekly_pct_change = df.groupby(level='ts_code')['close'].pct_change()
        factor_list.append(weekly_pct_change.groupby(level='ts_code').rolling(12).std().droplevel(0).rename('VOL_12W'))
        factor_list.append(df.groupby(level='ts_code')['amount'].rolling(4).mean().droplevel(0).rename('AMOUNT_4W_AVG'))
        factor_list.append(df.groupby(level='ts_code')['close'].pct_change(self.future_return_window).shift(-self.future_return_window).rename('FUT_RET'))
        factors = pd.concat(factor_list, axis=1).reset_index().rename(columns={'trade_date': 'date'})
        print("因子计算完成。")
        return factors

    def _standardize_factors(self, df):
        print("开始截面标准化因子...")
        factor_names = self.get_factor_names(df)
        df_std = df.set_index(['date', 'ts_code'])
        def z_score(group): return (group - group.mean()) / group.std()
        df_std[factor_names] = df_std.groupby('date')[factor_names].transform(z_score)
        print("因子标准化完成。")
        return df_std.reset_index()

    def build(self):
        fetch_start_date = (pd.to_datetime(self.start_date) - timedelta(days=365*3)).strftime('%Y-%m-%d')
        daily_data = get_sw1_daily_data(fetch_start_date, self.end_date)
        if daily_data.empty:
            print("未能获取到任何日度数据，构建失败。")
            return pd.DataFrame()
        weekly_data = self._get_weekly_data(daily_data)
        factors_df = self._calculate_factors(weekly_data)
        panel_df_std = self._standardize_factors(factors_df)
        self.panel_df = panel_df_std.dropna().sort_values(by=['date', 'ts_code'])
        print(f"最终面板数据构建完成，分析周期：{self.start_date} 到 {self.end_date}")
        print(f"数据维度 (行, 列): {self.panel_df.shape}")
        print(f"包含的因子: {self.get_factor_names()}")
        return self.panel_df

    def get_factor_names(self, df=None):
        if df is None: df = self.panel_df
        if df is None: return []
        return [col for col in df.columns if col not in ['date', 'ts_code', 'FUT_RET']]

# --- 3. TV-PR Analyzer with Full Visualization ---
class TVPRLassoAnalyzer:
    def __init__(self, panel_df, factor_names):
        self.panel_df = panel_df
        self.factor_names = factor_names
        self.analysis_df = panel_df[panel_df['date'] >= pd.to_datetime(ANALYSIS_START_DATE)].copy()
        self.dates = sorted(self.analysis_df['date'].unique())
        self.T = len(self.dates)
        self.K = len(factor_names)
        # 【新增】用于存储回测过程中的beta和日期
        self.backtest_betas = []
        self.backtest_dates = []

    def _solve_tvpr_for_period(self, start_date, end_date, lambda1, lambda2):
        window_df = self.analysis_df[(self.analysis_df['date'] >= start_date) & (self.analysis_df['date'] <= end_date)]
        dates_in_window = sorted(window_df['date'].unique())
        T_window = len(dates_in_window)
        if T_window < 10: return None
        X_list = [window_df[window_df['date'] == d][self.factor_names].values for d in dates_in_window]
        Y_list = [window_df[window_df['date'] == d]['FUT_RET'].values for d in dates_in_window]
        beta = cp.Variable((self.K, T_window))
        objective = sum(cp.sum_squares(X_list[i] @ beta[:, i] - Y_list[i]) for i in range(T_window))
        problem = cp.Problem(cp.Minimize(objective + lambda1 * cp.sum(cp.abs(beta)) + lambda2 * cp.sum(cp.abs(cp.diff(beta, k=1, axis=1)))))
        try:
            problem.solve(solver='SCS', verbose=False)
            return beta.value[:, -1] if beta.value is not None else None
        except Exception as e:
            print(f"  -> CVXPY 求解错误: {e}")
            return None

    def run_rolling_window_backtest(self, window_size=104, lambda1=0.01, lambda2=0.5, save_path='strategy_net_value_tvpr_rolling.png'):
        print(f"开始执行滚动窗口TV-PR回测... 窗口大小: {window_size}周")
        portfolio_returns = []
        self.backtest_betas = []
        self.backtest_dates = []

        for t in range(window_size, self.T):
            current_date = self.dates[t]
            training_end_date = self.dates[t-1]
            training_start_date = self.dates[t - window_size]
            print(f"回测进度: {t-window_size+1}/{self.T-window_size} | 决策日期: {current_date.date()} | 训练窗口: {training_start_date.date()} to {training_end_date.date()}")

            latest_beta = self._solve_tvpr_for_period(training_start_date, training_end_date, lambda1, lambda2)
            
            # 存储当期的beta和日期
            self.backtest_dates.append(current_date)
            if latest_beta is None:
                print("  -> 模型求解失败，本期收益计为0，因子收益记为0")
                portfolio_returns.append(0)
                self.backtest_betas.append(np.zeros(self.K))
                continue
            
            self.backtest_betas.append(latest_beta)
            
            current_data = self.analysis_df[self.analysis_df['date'] == current_date]
            if current_data.empty:
                portfolio_returns.append(0)
                continue
            
            X_t = current_data[self.factor_names].values
            predicted_returns = X_t @ latest_beta
            pred_series = pd.Series(predicted_returns, index=current_data['ts_code']).dropna()

            try:
                groups = pd.qcut(pred_series, 5, labels=False, duplicates='drop')
                long_return = current_data.set_index('ts_code')['FUT_RET'].reindex(pred_series.index[groups == 4]).mean()
                short_return = current_data.set_index('ts_code')['FUT_RET'].reindex(pred_series.index[groups == 0]).mean()
                period_return = (long_return - short_return) if pd.notna(long_return) and pd.notna(short_return) else 0
                portfolio_returns.append(period_return)
            except (ValueError, IndexError):
                portfolio_returns.append(0)

        if not portfolio_returns:
            print("回测未能产生任何收益数据。")
            return
        
        # 转换存储的beta为numpy array
        self.backtest_betas = np.array(self.backtest_betas).T # Shape: (K, T_backtest)

        returns_series = pd.Series(portfolio_returns, index=self.backtest_dates)
        net_value = (1 + returns_series).cumprod()
        
        plt.figure(figsize=(12, 6)); net_value.plot(); plt.title(f'TV-PR 滚动窗口回测净值曲线 (l1={lambda1}, l2={lambda2})'); plt.ylabel('Net Value'); plt.grid(True, alpha=0.5); plt.savefig(save_path); plt.close()
        
        if net_value.empty: return
        annual_return = (net_value.iloc[-1] ** (52 / len(net_value)) - 1) * 100
        annual_vol = net_value.pct_change().std() * np.sqrt(52) * 100
        sharpe_ratio = (annual_return / annual_vol) if annual_vol > 0 else 0
        print("\n--- 滚动窗口回测结果 (无前视偏差) ---")
        print(f"年化收益率: {annual_return:.2f}%, 年化波动率: {annual_vol:.2f}%, 夏普比率: {sharpe_ratio:.2f}")

    # --- 新增的可视化函数 ---
    def plot_factor_premia(self, save_path='factor_premia.png'):
        print(f"正在绘制因子收益（Beta）时序图并保存至 {save_path}...")
        K, T = self.backtest_betas.shape
        num_cols = 2
        num_rows = (K + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 4 * num_rows), sharex=True)
        axes = axes.flatten()
        for i in range(K):
            axes[i].plot(self.backtest_dates, self.backtest_betas[i, :], label=self.factor_names[i])
            axes[i].axhline(0, color='gray', linestyle='--', linewidth=0.8)
            axes[i].set_title(f'时变因子收益: {self.factor_names[i]}')
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='x', rotation=20)
        for i in range(K, len(axes)): fig.delaxes(axes[i])
        plt.tight_layout(); plt.savefig(save_path); plt.close()

    def plot_factor_weight_composition(self, save_path='factor_weight_composition.png'):
        print(f"正在绘制因子暴露（相对重要性）图并保存至 {save_path}...")
        abs_betas = np.abs(self.backtest_betas)
        total_abs_beta = np.sum(abs_betas, axis=0)
        relative_weights = np.divide(abs_betas, total_abs_beta, out=np.zeros_like(abs_betas), where=(total_abs_beta != 0))
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.stackplot(self.backtest_dates, relative_weights, labels=self.factor_names, alpha=0.8)
        ax.set_title('因子暴露（相对重要性）随时间变化图', fontsize=16)
        ax.set_ylabel('因子权重占比'); ax.set_xlabel('日期'); ax.legend(loc='upper left'); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)
        fig.autofmt_xdate()
        plt.tight_layout(); plt.savefig(save_path); plt.close()

    def calculate_benchmark_factor_returns(self):
        print("正在计算基准因子收益（通过横截面回归）...")
        all_benchmark_betas = []
        for date in self.backtest_dates:
            current_data = self.analysis_df[self.analysis_df['date'] == date]
            if len(current_data) < self.K + 1:
                all_benchmark_betas.append(np.zeros(self.K))
                continue
            X = current_data[self.factor_names].values
            Y = current_data['FUT_RET'].values
            model = LinearRegression(fit_intercept=False)
            model.fit(X, Y)
            all_benchmark_betas.append(model.coef_)
        return pd.DataFrame(all_benchmark_betas, index=self.backtest_dates, columns=self.factor_names)

    def plot_cumulative_factor_returns(self, benchmark_returns, save_path='cumulative_returns_comparison.png'):
        print(f"正在绘制累计收益对比（拟合结果）图并保存至 {save_path}...")
        num_factors = self.K
        num_cols = 2
        num_rows = (num_factors + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows), sharex=True)
        axes = axes.flatten()
        for i, factor_name in enumerate(self.factor_names):
            ax = axes[i]
            benchmark_net_value = (1 + benchmark_returns[factor_name]).cumprod()
            ax.plot(benchmark_net_value.index, benchmark_net_value, color='black', linewidth=1.5, label='基准因子收益')
            model_beta_series = pd.Series(self.backtest_betas[i, :], index=self.backtest_dates)
            model_net_value = (1 + model_beta_series).cumprod()
            ax.plot(model_net_value.index, model_net_value, color='deeppink', label='模型估计收益')
            ax.set_title(f'拟合结果: {factor_name}'); ax.grid(True, alpha=0.3); ax.axhline(1, color='gray', linestyle='--', linewidth=0.8); ax.set_ylabel('累计收益 (净值)')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))
        for i in range(num_factors, len(axes)): fig.delaxes(axes[i])
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]); plt.savefig(save_path); plt.close()

# --- 4. Main Execution Block ---
if __name__ == "__main__":
    ANALYSIS_START_DATE = '2020-01-01' 
    ANALYSIS_END_DATE = '2025-11-04'
    ROLLING_WINDOW_SIZE = 52
    LAMBDA_1 = 0.01
    LAMBDA_2 = 0.01

    builder = PanelDataBuilder(start_date=ANALYSIS_START_DATE, end_date=ANALYSIS_END_DATE)
    panel_data = builder.build()

    if panel_data is not None and not panel_data.empty:
        analyzer = TVPRLassoAnalyzer(panel_df=panel_data, factor_names=builder.get_factor_names())
        
        # 1. 运行核心回测，此函数会填充 backtest_betas 和 backtest_dates
        analyzer.run_rolling_window_backtest(
            window_size=ROLLING_WINDOW_SIZE,
            lambda1=LAMBDA_1, 
            lambda2=LAMBDA_2,
            save_path='multifactor/sw1_strategy_net_value_tvpr_rolling.png'
        )
        
        # 2. 检查是否有结果，然后生成所有附加图表
        if analyzer.backtest_betas is not None and len(analyzer.backtest_betas) > 0:
            print("\n回测完成，开始生成分析图表...")
            
            # 2.1 绘制各因子收益曲线
            analyzer.plot_factor_premia(save_path='multifactor/sw1_factor_premia_tvpr.png')
            
            # 2.2 绘制因子暴露（相对重要性）
            analyzer.plot_factor_weight_composition(save_path='multifactor/sw1_factor_weight_composition_tvpr.png')
            
            # 2.3 绘制拟合结果（模型 vs 基准）
            benchmark_rets = analyzer.calculate_benchmark_factor_returns()
            analyzer.plot_cumulative_factor_returns(benchmark_rets, save_path='multifactor/sw1_cumulative_returns_comparison_tvpr.png')
            
            print("所有分析图表生成完毕。")
        else:
            print("未能从回测中获取有效的因子收益数据，无法生成图表。")
    else:
        print("未能成功构建面板数据，程序终止。")