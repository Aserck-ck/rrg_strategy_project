import sys
import gc
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import torch
# Ensure project root is in path

from market_data.sw_level1.sw_level1_functions import get_sw_level1_info

matplotlib.rcParams['font.family'] = 'SimHei'  # 或其他支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
np.set_printoptions(threshold=np.inf)
import pickle
import os
import pandas as pd

import numpy as np
import pandas as pd
from scipy import stats


def calculate_performance_metrics_with_costs_included(strategy_net, benchmark_net, portfolios,
                                                      buy_rate=0.0003, sell_rate=0.0015,
                                                      risk_free_rate=0.02, trading_days=52):
    """
    计算策略相对于基准的量化指标（交易成本已直接计入收益曲线）

    参数:
    strategy_net: 策略净值序列 (DataFrame或Series)
    benchmark_net: 基准净值序列 (DataFrame或Series)
    portfolios: 投资组合仓位记录 (DataFrame)
    buy_rate: 买入费率
    sell_rate: 卖出费率
    risk_free_rate: 无风险利率 (年化)
    trading_days: 年化交易日数

    返回:
    metrics_dict: 包含所有指标的字典
    adjusted_net: 考虑交易成本后的净值序列
    """

    # 确保数据对齐
    common_index = strategy_net.index.intersection(benchmark_net.index)
    strategy_net = strategy_net.loc[common_index]
    benchmark_net = benchmark_net.loc[common_index]
    portfolios = portfolios.loc[common_index]
    # 计算交易成本和换手率
    turnover_stats = calculate_turnover_and_costs(portfolios, buy_rate, sell_rate)
    # 计算考虑交易成本后的净值曲线
    adjusted_net = apply_trading_costs_to_net(strategy_net, portfolios, buy_rate, sell_rate)

    # 使用调整后的净值计算所有指标
    strategy_returns = adjusted_net.pct_change().dropna()
    benchmark_returns = benchmark_net.pct_change().dropna()
    excess_returns = strategy_returns - benchmark_returns

    # 计算累计收益
    total_strategy_return = (adjusted_net.iloc[-1] / adjusted_net.iloc[0]) - 1 - turnover_stats['total_cost']
    total_benchmark_return = (benchmark_net.iloc[-1] / benchmark_net.iloc[0]) - 1
    total_excess_return = total_strategy_return - total_benchmark_return

    # 年化收益率
    days = len(adjusted_net)
    annualized_strategy_return = (1 + total_strategy_return) ** (trading_days / days) - 1
    annualized_benchmark_return = (1 + total_benchmark_return) ** (trading_days / days) - 1
    annualized_excess_return = annualized_strategy_return - annualized_benchmark_return

    # 年化波动率
    annualized_strategy_vol = strategy_returns.std() * np.sqrt(trading_days)
    annualized_benchmark_vol = benchmark_returns.std() * np.sqrt(trading_days)

    # 夏普比率
    strategy_sharpe = (annualized_strategy_return - risk_free_rate) / annualized_strategy_vol
    benchmark_sharpe = (annualized_benchmark_return - risk_free_rate) / annualized_benchmark_vol

    # 最大回撤
    def calculate_max_drawdown(net_values):
        peak = net_values.expanding().max()
        drawdown = (net_values - peak) / peak
        return drawdown.min()

    strategy_max_drawdown = calculate_max_drawdown(adjusted_net)
    benchmark_max_drawdown = calculate_max_drawdown(benchmark_net)

    # 计算Alpha和Beta
    covariance = np.cov(strategy_returns.values.flatten(), benchmark_returns.values.flatten())[0, 1]
    benchmark_variance = np.var(benchmark_returns.values.flatten())
    beta = covariance / benchmark_variance

    # Alpha计算
    alpha = annualized_strategy_return - (risk_free_rate + beta * (annualized_benchmark_return - risk_free_rate))

    # 信息比率
    tracking_error = excess_returns.std() * np.sqrt(trading_days)
    information_ratio = annualized_excess_return / tracking_error if tracking_error != 0 else np.nan

    # 胜率 (策略跑赢基准的天数比例)
    win_rate = (excess_returns > 0).sum() / len(excess_returns) if len(excess_returns) > 0 else 0

    # Calmar比率 (年化收益/最大回撤)
    strategy_calmar = annualized_strategy_return / abs(strategy_max_drawdown) if strategy_max_drawdown != 0 else np.nan
    benchmark_calmar = annualized_benchmark_return / abs(
        benchmark_max_drawdown) if benchmark_max_drawdown != 0 else np.nan

    # 索提诺比率 (只考虑下行风险)
    downside_returns = strategy_returns[strategy_returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(trading_days) if len(downside_returns) > 0 else np.nan
    sortino_ratio = (
                            annualized_strategy_return - risk_free_rate) / downside_vol if downside_vol != 0 and downside_vol is not np.nan else np.nan

    # 计算成本对收益的影响
    cost_impact = turnover_stats['total_cost'] / (strategy_net.iloc[-1] / strategy_net.iloc[0] - 1)

    # 整理结果
    metrics = {
        '累计收益-策略': f"{total_strategy_return:.2%}",
        '累计收益-基准': f"{total_benchmark_return:.2%}",
        '累计超额收益': f"{total_excess_return:.2%}",
        '年化收益-策略': f"{annualized_strategy_return:.2%}",
        '年化收益-基准': f"{annualized_benchmark_return:.2%}",
        '年化超额收益': f"{annualized_excess_return:.2%}",
        '年化波动-策略': f"{annualized_strategy_vol:.2%}",
        '年化波动-基准': f"{annualized_benchmark_vol:.2%}",
        '夏普比率-策略': f"{strategy_sharpe:.4f}",
        '夏普比率-基准': f"{benchmark_sharpe:.4f}",
        '最大回撤-策略': f"{strategy_max_drawdown:.2%}",
        '最大回撤-基准': f"{benchmark_max_drawdown:.2%}",
        'Alpha': f"{alpha:.4f}",
        'Beta': f"{beta:.4f}",
        '信息比率': f"{information_ratio:.4f}",
        '胜率': f"{win_rate:.2%}",
        'Calmar比率-策略': f"{strategy_calmar:.4f}",
        'Calmar比率-基准': f"{benchmark_calmar:.4f}",
        '索提诺比率': f"{sortino_ratio:.4f}",
        '跟踪误差': f"{tracking_error:.2%}",
        '年化换手率': f"{turnover_stats['annual_turnover']:.2%}",
        '总交易成本': f"{turnover_stats['total_cost']:.4f}",
        '平均单次换手率': f"{turnover_stats['avg_turnover']:.2%}",
        '调仓频率': '周度',
        '交易成本对收益影响': f"{cost_impact:.2%}"
    }

    return metrics, adjusted_net


def apply_trading_costs_to_net(strategy_net, portfolios, buy_rate=0.0003, sell_rate=0.0015):
    """
    将交易成本直接应用到净值曲线中
    模拟真实的交易成本对净值的影响
    """
    # 复制原始净值
    adjusted_net = strategy_net.copy()

    # 计算每周的换手率和交易成本
    for i in range(1, len(portfolios)):
        prev_portfolio = portfolios.iloc[i - 1]
        curr_portfolio = portfolios.iloc[i]

        # 计算换手率
        turnover = abs(curr_portfolio - prev_portfolio).sum() / 2

        if turnover > 0:  # 只有在有交易时才应用成本
            # 计算交易成本
            buy_cost = np.where(curr_portfolio > prev_portfolio,
                                (curr_portfolio - prev_portfolio) * buy_rate, 0).sum()
            sell_cost = np.where(curr_portfolio < prev_portfolio,
                                 (prev_portfolio - curr_portfolio) * sell_rate, 0).sum()
            total_cost = buy_cost + sell_cost

            # 将交易成本应用到净值中
            # 使用前一日的净值作为基础计算成本影响
            cost_factor = 1 - total_cost
            adjusted_net.iloc[i] = adjusted_net.iloc[i] * cost_factor

    return adjusted_net


def calculate_turnover_and_costs(portfolios, buy_rate=0.0003, sell_rate=0.0015):
    """
    计算换手率和交易成本
    """
    turnover_rates = []
    total_costs = 0

    for i in range(1, len(portfolios)):
        prev_portfolio = portfolios.iloc[i - 1]
        curr_portfolio = portfolios.iloc[i]

        # 计算换手率
        turnover = abs(curr_portfolio - prev_portfolio).sum() / 2
        turnover_rates.append(turnover)

        # 计算交易成本
        buy_cost = np.where(curr_portfolio > prev_portfolio,
                            (curr_portfolio - prev_portfolio) * buy_rate, 0).sum()
        sell_cost = np.where(curr_portfolio < prev_portfolio,
                             (prev_portfolio - curr_portfolio) * sell_rate, 0).sum()

        total_costs += buy_cost + sell_cost

    if len(turnover_rates) > 0:
        avg_turnover = np.mean(turnover_rates)
        annual_turnover = avg_turnover * 52  # 周度调仓，年化换手率
    else:
        avg_turnover = 0
        annual_turnover = 0

    return {
        'annual_turnover': annual_turnover,
        'total_cost': total_costs,
        'avg_turnover': avg_turnover,
        'turnover_rates': turnover_rates
    }


def plot_result():
    # 在你的绘图代码中使用（修改后）
    columns = test.pct_change_df.columns
    for col in columns[-1:]:
        plt.figure(figsize=(14, 7))

        # 计算包含交易成本的指标和净值
        metrics, adjusted_net = calculate_performance_metrics_with_costs_included(
            test.sum_net['net_value'],
            test.benchmark_net[col],
            test.portfolios,
            buy_rate=0.0003,
            sell_rate=0.0015
        )

        plt.title(f"{col} - 周度调仓策略 (交易成本: 万3/千1.5)")

        # 绘制三条曲线：原始策略、考虑成本后的策略、基准
        plt.plot(test.sum_net.index, test.sum_net['net_value'],
                 label="策略: 原始净值", linewidth=2, alpha=0.7, linestyle='--')
        plt.plot(adjusted_net.index, adjusted_net,
                 label="策略: 含交易成本净值", linewidth=2, color='red')
        plt.plot(test.benchmark_net.index, test.benchmark_net[col],
                 label='基准', linewidth=2, alpha=0.7, color='green')

        # 在图上添加关键指标
        textstr = '\n'.join([
            f'年化收益(含成本): {metrics["年化收益-策略"]}',
            f'年化收益(基准): {metrics["年化收益-基准"]}',
            f'夏普比率(含成本): {metrics["夏普比率-策略"]}',
            f'最大回撤(含成本): {metrics["最大回撤-策略"]}',
            f'Alpha: {metrics["Alpha"]}, Beta: {metrics["Beta"]}',
            f'年化换手率: {metrics["年化换手率"]}',
            f'总交易成本: {metrics["总交易成本"]}',
            f'成本对收益影响: {metrics["交易成本对收益影响"]}'
        ])

        # 在图的右上角添加文本框
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                       verticalalignment='top', bbox=props)

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylabel('净值')
        plt.xlabel('日期')
        plt.tight_layout()
        plt.show()

        # 打印完整指标表格
        print(f"\n=== {col} 详细性能指标 (含交易成本) ===")

        # 创建更美观的表格显示
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['指标', '数值'])

        # 分组显示指标
        basic_metrics = metrics_df.iloc[:9]  # 基本收益指标
        risk_metrics = metrics_df.iloc[9:18]  # 风险指标
        cost_metrics = metrics_df.iloc[18:]  # 成本相关指标

        print("\n【基本收益指标】")
        print(basic_metrics.to_string(index=False))

        print("\n【风险调整指标】")
        print(risk_metrics.to_string(index=False))

        print("\n【交易成本指标】")
        print(cost_metrics.to_string(index=False))

        sharpe = float(metrics['夏普比率-策略'])
        max_dd = float(metrics['最大回撤-策略'].strip('%')) / 100
        annual_return = float(metrics['年化收益-策略'].strip('%')) / 100

        # 计算交易成本对累计收益的影响
        original_total_return = (test.sum_net['net_value'].iloc[-1] / test.sum_net['net_value'].iloc[0]) - 1
        adjusted_total_return = (adjusted_net.iloc[-1] / adjusted_net.iloc[0]) - 1 - float(metrics["总交易成本"])
        cost_impact_pct = (original_total_return - adjusted_total_return) / original_total_return * 100

        print(f"\n【交易成本影响分析】")
        print(f"原始累计收益: {original_total_return:.5%}")
        print(f"扣除成本后累计收益: {adjusted_total_return:.5%}")
        print(f"交易成本侵蚀收益: {cost_impact_pct:.5f}%")


def calc_net_values(index_pct_chg):
    '''Calculate net values from a series of percentage change. Return a pandas series object.'''
    temp = index_pct_chg + 1
    return np.cumprod(temp)


class BackTest(object):
    def __init__(self, start_date, end_date, ratio_pred_df, momentum_pred_df, ratio_weekly_df, momentum_weekly_df,
                 pct_change_df):
        '''Initialize trade dates.'''
        self.start_date = start_date
        self.end_date = end_date

        '''Read etf and index data.'''
        self.ratio_df = ratio_pred_df.query('trade_date >= @start_date & trade_date <= @end_date')
        self.momentum_df = momentum_pred_df.query('trade_date >= @start_date & trade_date <= @end_date')
        self.ratio_weekly_df = ratio_weekly_df.query('trade_date >= @start_date & trade_date <= @end_date')
        self.momentum_weekly_df = momentum_weekly_df.query('trade_date >= @start_date & trade_date <= @end_date')
        self.pct_change_df = pct_change_df.query('trade_date >= @start_date & trade_date <= @end_date')

        # 设置日期索引并确保数据类型一致
        self._setup_dataframes()

        # 确保日期排序
        self._sort_dataframes()

    def _setup_dataframes(self):
        """设置数据框索引并确保数据类型一致"""
        # 确保所有数据框的索引都是相同的日期类型
        for df_name in ['ratio_df', 'momentum_df', 'ratio_weekly_df', 'momentum_weekly_df', 'pct_change_df']:
            df = getattr(self, df_name)
            # 如果'trade_date'是列而不是索引，则设置为索引
            if 'trade_date' in df.columns:
                df = df.set_index('trade_date')
            # 确保索引是日期类型
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            setattr(self, df_name, df)

        # 打印索引类型和长度以便调试
        print("数据框索引类型和长度:")
        for df_name in ['ratio_df', 'momentum_df', 'ratio_weekly_df', 'momentum_weekly_df', 'pct_change_df']:
            df = getattr(self, df_name)
            print(f"{df_name}: {type(df.index)}, 长度: {len(df)}")
            if len(df) > 0:
                print(f"  第一个日期: {df.index[0]}, 最后一个日期: {df.index[-1]}")

    def _sort_dataframes(self):
        """确保所有数据框按日期排序"""
        for df_name in ['ratio_df', 'momentum_df', 'ratio_weekly_df', 'momentum_weekly_df', 'pct_change_df']:
            df = getattr(self, df_name)
            setattr(self, df_name, df.sort_index())

    def strategy_net_value(self):
        self.buy_rate = 0  # 0% commission fee for buy orders
        self.sell_rate = 0  # 0% commission fee for sell orders

        # 确保有足够的数据点
        if len(self.pct_change_df) < 3:
            print("错误: 数据点不足，需要至少3个交易日")
            return

        # 使用实际的日期索引
        self.trade_dates = self.pct_change_df.index[1:]  # 从第4个日期开始

        # 初始化净值数据框
        self.net_values = pd.DataFrame(
            np.ones((len(self.pct_change_df), len(self.pct_change_df.columns))),
            index=self.pct_change_df.index,
            columns=self.pct_change_df.columns
        )
        self.sum_net = pd.DataFrame(
            np.ones((len(self.pct_change_df), 1)),
            index=self.pct_change_df.index,
            columns=['net_value']
        )
        self.portfolios = pd.DataFrame(
            0.0,
            columns=self.ratio_df.columns,
            index=self.pct_change_df.index
        )
        self.strategy_name = 'Buy when in the second quadrant'

        # 修正循环逻辑
        for i in range(3, len(self.trade_dates)):
            current_date = self.trade_dates[i]
            prev_date = self.trade_dates[i - 1]  # 注意索引调整
            prev_prev_date = self.trade_dates[i - 2]  # 注意索引调整
            prev_prev_date2 = self.trade_dates[i - 3]

            print(f"处理日期: {current_date} (索引 {i})")

            portfolio = []

            try:
                # 使用.loc按日期访问数据
                # overheat
                condition1 = (self.momentum_df.loc[current_date] - self.momentum_weekly_df.loc[prev_date] >
                              self.momentum_weekly_df.loc[prev_date] - self.momentum_weekly_df.loc[prev_prev_date])
                condition2 = self.momentum_df.loc[current_date] > 0
                condition3 = self.momentum_df.loc[current_date] > self.momentum_df.loc[prev_date] + 1

                condition1l = (self.momentum_weekly_df.loc[prev_date] - self.momentum_weekly_df.loc[prev_prev_date] >
                               self.momentum_weekly_df.loc[prev_prev_date] - self.momentum_weekly_df.loc[
                                   prev_prev_date2])
                condition2l = self.momentum_weekly_df.loc[prev_date] > self.momentum_weekly_df.loc[prev_prev_date] + 0.5
                condition3l = self.momentum_weekly_df.loc[prev_date] > 0

                # Reverse

                condition4f = self.momentum_weekly_df.loc[current_date] > self.momentum_weekly_df.loc[prev_date] + 0.5
                condition5f = self.momentum_weekly_df.loc[current_date] > self.ratio_weekly_df.loc[current_date] - 1
                condition6f = self.ratio_weekly_df.loc[current_date] < 101

                condition4 = self.momentum_df.loc[current_date] > self.momentum_df.loc[prev_date] + 0.5
                condition5 = self.momentum_df.loc[current_date] > self.ratio_df.loc[current_date] - 1
                condition6 = self.ratio_df.loc[current_date] < 100
                # condition5l = self.momentum_df.loc[current_date] > self.momentum_weekly_df.loc[prev_date]+0.5

                ratio_w_smooth = self.ratio_weekly_df.rolling(3, min_periods=1).mean()
                momentum_w_smooth = self.momentum_weekly_df.rolling(3, min_periods=1).mean()
                condition4l = ratio_w_smooth.loc[prev_date] < 100
                condition5l = momentum_w_smooth.loc[prev_date] > 100
                condition6l = momentum_w_smooth.loc[prev_date] > momentum_w_smooth.loc[prev_prev_date]

                # leading

                condition7f = (self.momentum_weekly_df.loc[current_date] > self.momentum_weekly_df.loc[
                    current_date].median()) & (self.ratio_weekly_df.loc[current_date] > self.ratio_weekly_df.loc[
                    current_date].median())
                condition8f = self.momentum_weekly_df.loc[current_date] - self.momentum_weekly_df.loc[
                    prev_date] > max(
                    (self.momentum_weekly_df.loc[current_date] - self.momentum_weekly_df.loc[prev_date]).median(), 0)
                condition9f = self.ratio_weekly_df.loc[current_date] - self.ratio_weekly_df.loc[prev_date] > max(
                    (self.ratio_weekly_df.loc[current_date] - self.ratio_weekly_df.loc[prev_date]).median(), 0)

                condition7 = (self.ratio_df.loc[current_date] > self.ratio_df.loc[current_date].median()) & (
                            self.momentum_df.loc[current_date] > self.momentum_df.loc[current_date].median())
                condition8 = self.momentum_df.loc[current_date] - self.momentum_df.loc[prev_date] > max(
                    (self.momentum_df.loc[current_date] - self.momentum_df.loc[prev_date]).median(), 1)
                condition9 = self.ratio_df.loc[current_date] - self.ratio_df.loc[prev_date] > max(
                    (self.ratio_df.loc[current_date] - self.ratio_df.loc[prev_date]).median(), 1)

                condition7l = (self.momentum_weekly_df.loc[prev_date] > self.momentum_weekly_df.loc[
                    prev_date].median()) & (self.ratio_weekly_df.loc[prev_date] > self.ratio_weekly_df.loc[
                    prev_date].median())
                condition8l = self.momentum_weekly_df.loc[prev_date] - self.momentum_weekly_df.loc[
                    prev_prev_date] > max(
                    (self.momentum_weekly_df.loc[prev_date] - self.momentum_weekly_df.loc[prev_prev_date]).max()-1, 2)
                condition9l = self.ratio_weekly_df.loc[prev_date] - self.ratio_weekly_df.loc[prev_prev_date] > max(
                    (self.ratio_weekly_df.loc[prev_date] - self.ratio_weekly_df.loc[prev_prev_date]).max()-1, 2)

                combined_condition1 = condition1 & condition2 & condition3
                combined_condition1l = condition1l & condition2l & condition3l
                combined_condition2f = condition4f & condition5f & condition6f
                combined_condition2 = condition4 & condition5 & condition6
                combined_condition2l = condition4l & condition5l & condition6l
                combined_condition3 = condition7 & condition8 & condition9
                combined_condition3f = condition7f & condition8f & condition9f
                combined_condition3l = condition7l & condition8l & condition9l

                # combined_condition =  (combined_condition2 & combined_condition2l) | (combined_condition3 | combined_condition3l)
                combined_condition = combined_condition3l
                # combined_condition = combined_condition2f
                # combined_condition = (~combined_condition2l & combined_condition3l) | (combined_condition2l & ~combined_condition3l)
                # 获取满足条件的列索引
                column_indices = np.where(combined_condition)[0]
                selected_columns = self.ratio_weekly_df.columns[column_indices]

                # 将选中的列索引添加到portfolio中
                portfolio.extend(selected_columns)

                num_stocks = len(portfolio)

                print(f"选中的资产数量为: {num_stocks} : {portfolio}, ")

                if num_stocks == 0:
                    self.portfolios.loc[current_date] = 0
                else:
                    # 等权分配投资组合
                    self.portfolios.loc[current_date, portfolio] = 1 / num_stocks

            except KeyError as e:
                print(f"KeyError 在日期 {current_date}: {e}")
                print(f"prev_date: {prev_date} 是否在 ratio_weekly_df 中: {prev_date in self.ratio_weekly_df.index}")
                print(
                    f"prev_date: {prev_date} 是否在 momentum_weekly_df 中: {prev_date in self.momentum_weekly_df.index}")
                print(
                    f"current_date: {current_date} 是否在 ratio_weekly_df 中: {current_date in self.ratio_weekly_df.index}")
                print(
                    f"current_date: {current_date} 是否在 momentum_weekly_df 中: {current_date in self.momentum_weekly_df.index}")
                continue
            except Exception as e:
                print(f"错误 在日期 {current_date}: {e}")
                continue

            # 计算每日P&L
            try:
                if i > 0:  # 从第4个日期开始计算
                    prev_net_date = prev_date  # 注意索引调整
                    pct_change = (self.pct_change_df.loc[current_date] * self.portfolios.loc[current_date])
                    self.net_values.loc[current_date] = self.net_values.loc[prev_net_date] * (1 + pct_change)
                    total_pct_change = (self.pct_change_df.loc[current_date] * self.portfolios.loc[current_date]).sum()
                    self.sum_net.loc[current_date] = self.sum_net.loc[prev_net_date] * (1 + total_pct_change)
            except Exception as e:
                print(f"计算净值错误 在日期 {current_date}: {e}")
                continue

    def benchmark_net_value(self):
        # 简单的基准净值计算 - 等权持有所有资产
        self.benchmark_net = pd.DataFrame(
            (1 + self.pct_change_df).cumprod(),
            index=self.pct_change_df.index,
            columns=self.pct_change_df.columns
        )


steps = 2
# x_pred_df = pd.read_csv(f'./analyse/x_pred_mean{steps}_weekly.csv',index_col=0,parse_dates=True)
# y_pred_df = pd.read_csv(f'./analyse/y_pred_mean{steps}_weekly.csv',index_col=0,parse_dates=True)
x_pred_df = pd.read_csv(f'./analyse/sw2_x_pred_mean{steps}_weekly.csv', index_col=0, parse_dates=True)
y_pred_df = pd.read_csv(f'./analyse/sw2_y_pred_mean{steps}_weekly.csv', index_col=0, parse_dates=True)
# x_pred_df = pd.read_csv(f'./analyse/x_pred_single_lstm_weekly.csv',index_col=0,parse_dates=True)
# y_pred_df = pd.read_csv(f'./analyse/y_pred_single_lstm_weekly.csv',index_col=0,parse_dates=True)

from market_data.sw_level2.sw_level2_jdk_functions import get_sw_level2_ratio, get_sw_level2_momentum


ratio_df = get_sw_level2_ratio(ts_code='all', frequency='daily').set_index('trade_date').drop(columns=['000300.SH'])
momentum_df = get_sw_level2_momentum(ts_code='all', frequency='daily').set_index('trade_date').drop(
    columns=['000300.SH'])

ratio_w_df = get_sw_level2_ratio(ts_code='all', frequency='weekly').set_index('trade_date').drop(columns=['000300.SH'])
momentum_w_df = get_sw_level2_momentum(ts_code='all', frequency='weekly').set_index('trade_date').drop(
    columns=['000300.SH'])

from market_data.sw_level2.sw_level2_functions import *

ts_codes = get_sw_level2_codes()
close_df = pd.DataFrame(index=ratio_df.index, columns=ts_codes)
amt_df = pd.DataFrame(index=ratio_df.index, columns=ts_codes)
for i, ts_code in enumerate(ts_codes):
    index_df = get_sw_level2_data(ts_code).set_index('trade_date')
    close_df[ts_code] = index_df['close']
    amt_df[ts_code] = index_df['amount'].rolling(5).sum() / 100000000

from market_data.market_index.market_index_functions import *

hs300 = get_market_index_data('000300.SH').set_index('trade_date')
close_df['000300.SH'] = hs300['close']
close_df = close_df.dropna()
amt_df['000300.SH'] = hs300['amount']
amt_df = amt_df.dropna()
start_date = x_pred_df.index[-350]
end_date = x_pred_df.index[-2]
data_df = pd.DataFrame(index=momentum_df.index, columns=momentum_df.columns)

spx_weekly_df = close_df.resample('W-FRI').ffill().query('trade_date >= @start_date & trade_date <= @end_date')
spx_df = close_df.query('trade_date >= @start_date & trade_date <= @end_date')
data_df = spx_weekly_df
pct_change_df = data_df.pct_change().dropna()
start_date1 = x_pred_df.index[-350]

test = BackTest(start_date1, end_date, x_pred_df + 100, y_pred_df + 100, ratio_w_df, momentum_w_df, pct_change_df)
test.strategy_net_value()
test.benchmark_net_value()

plot_result()