import pandas as pd
import numpy as np

def calc_jdk_daily(data_df: pd.DataFrame, ratio_rolling: int=20, ratio_momentum: int=9, total_rolling: int=12):
    '''Calculate JDK RS Ratio and JDK RS Momentum from data_df.
    data_df should be a df with the index trade dates, 
    and the first column should be benchmark values; 
    other columns should be sector values.'''
    # Compute relative performance of each sector w.r.t. the benchmark
    relative_df = pd.DataFrame(index=data_df.index, columns=data_df.columns[:])
    benchmark_name = data_df.columns[0]
    for sector_name in data_df.columns[:]:
        relative_df[sector_name] = data_df[sector_name] / data_df[benchmark_name]
    
    # Compute JDK RS Ratio
    jdk_rs_ratio = pd.DataFrame(index=relative_df.index, columns=data_df.columns[:])
    for sector_name in relative_df.columns:
        jdk_rs_ratio[sector_name] = 100 + \
            (relative_df[sector_name] - relative_df[sector_name].rolling(ratio_rolling).mean()) / \
            relative_df[sector_name].rolling(ratio_rolling).mean() * 100
        
    jdk_rs_ratio = jdk_rs_ratio.dropna(axis=0, how='all')
    
    jdk_rs_ratio = jdk_rs_ratio.rolling(total_rolling).mean().dropna(axis=0, how='all')
   
    # Compute JDK RS momentum
    jdk_rs_momentum = pd.DataFrame(index=jdk_rs_ratio.index, columns=jdk_rs_ratio.columns)
    for sector_name in jdk_rs_ratio.columns:
        jdk_rs_momentum[sector_name] = 100 + \
            (jdk_rs_ratio[sector_name] - jdk_rs_ratio[sector_name].rolling(ratio_momentum).mean())
    jdk_rs_momentum = jdk_rs_momentum.dropna(axis=0, how='all')
        
    return jdk_rs_ratio, jdk_rs_momentum
    
def calc_jdk_daily_no_rolling(data_df: pd.DataFrame, ratio_rolling: int=20, ratio_momentum: int=9, total_rolling: int=12):
    '''Calculate JDK RS Ratio and JDK RS Momentum from data_df.
    data_df should be a df with the index trade dates, 
    and the first column should be benchmark values; 
    other columns should be sector values.'''
    # Compute relative performance of each sector w.r.t. the benchmark
    relative_df = data_df[data_df.columns[1:]] 
    
    # Compute JDK RS Ratio
    jdk_rs_ratio = pd.DataFrame(index=relative_df.index, columns=data_df.columns[:])
    for sector_name in relative_df.columns:
        jdk_rs_ratio[sector_name] = 100 + \
            (relative_df[sector_name] - relative_df[sector_name].rolling(ratio_rolling).mean()) / \
            relative_df[sector_name].rolling(ratio_rolling).mean() * 100
        
    jdk_rs_ratio = jdk_rs_ratio.dropna(axis=0, how='all')
    
    jdk_rs_ratio = jdk_rs_ratio.rolling(total_rolling).mean().dropna(axis=0, how='all')
   
    # Compute JDK RS momentum
    jdk_rs_momentum = pd.DataFrame(index=jdk_rs_ratio.index, columns=jdk_rs_ratio.columns)
    for sector_name in jdk_rs_ratio.columns:
        jdk_rs_momentum[sector_name] = 100 + \
            (jdk_rs_ratio[sector_name] - jdk_rs_ratio[sector_name].rolling(ratio_momentum).mean())
    jdk_rs_momentum = jdk_rs_momentum.dropna(axis=0, how='all')
        
    return jdk_rs_ratio, jdk_rs_momentum

def calc_jdk_weekly(data_df: pd.DataFrame, on: str, ratio_rolling: int=20, ratio_momentum: int=9, total_rolling: int=12):
    '''Calculate weekly JDK RS Ratio and JDK RS Momentum from data_df.
    data_df should be a df with the index trade dates, 
    and the first column should be benchmark values; 
    other columns should be sector values. on specifies 
    which date in a week to represent the data of the whole week.
    on can range from 'MON' to 'FRI'.'''
    
    # Convert daily close to weekly close price based on specified weekday
    data_df = data_df.resample('W-'+on).ffill()

    # Compute relative performance of each sector w.r.t. the benchmark
    relative_df = pd.DataFrame(index=data_df.index, columns=data_df.columns[:])
    benchmark_name = data_df.columns[0]
    for sector_name in data_df.columns[:]:
        relative_df[sector_name] = data_df[sector_name] / data_df[benchmark_name]
    
    # Compute JDK RS Ratio
    jdk_rs_ratio = pd.DataFrame(index=relative_df.index, columns=data_df.columns[:])
    for sector_name in relative_df.columns:
        jdk_rs_ratio[sector_name] = 100 + \
            (relative_df[sector_name] - relative_df[sector_name].rolling(ratio_rolling).mean()) / \
            relative_df[sector_name].rolling(ratio_rolling).mean() * 100
        
    jdk_rs_ratio = jdk_rs_ratio.dropna(axis=0, how='all')
    
    jdk_rs_ratio = jdk_rs_ratio.rolling(total_rolling).mean().dropna(axis=0, how='all')
   
    # Compute JDK RS momentum
    jdk_rs_momentum = pd.DataFrame(index=jdk_rs_ratio.index, columns=jdk_rs_ratio.columns)
    for sector_name in jdk_rs_ratio.columns:
        jdk_rs_momentum[sector_name] = 100 + \
            (jdk_rs_ratio[sector_name] - jdk_rs_ratio[sector_name].rolling(ratio_momentum).mean())
    jdk_rs_momentum = jdk_rs_momentum.dropna(axis=0, how='all')
        
    return jdk_rs_ratio, jdk_rs_momentum
