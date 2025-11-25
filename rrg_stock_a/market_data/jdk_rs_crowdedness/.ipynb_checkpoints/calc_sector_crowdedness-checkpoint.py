import pandas as pd
import numpy as np

def calc_sector_rank_daily(sector_amount_df, period=250):
    '''Compute sector crowdedness quantile value for daily amount.
    Period is 250 days by default. sector_amount_df has index trade_date and columns 
    ts_codes with values the amount of trading.'''
    # Compute the amount ratio for each sector
    sector_amount_ratio_df = (sector_amount_df.T / sector_amount_df.sum(axis=1).T).T
    
    # Compute the quantile value within period for each date
    sector_amount_ratio_rank_df = sector_amount_ratio_df.rolling(period).apply(lambda x: \
                                (list(np.argsort(x)).index(period-1)+1) / period).dropna()
    
    return sector_amount_ratio_rank_df

def calc_sector_percentile_daily(sector_amount_df, period=750):
    '''Compute sector crowdedness percentile value w.r.t. the maximum within a period on a daily basis.
    Period is 750 days by default. sector_amount_df has index trade_date and columns 
    ts_codes with values the amount of trading.'''
    # Compute the amount ratio for each sector
    sector_amount_ratio_df = (sector_amount_df.T / sector_amount_df.sum(axis=1).T).T
    
    # Compute the percentile w.r.t. the maximum value within period for each date
    sector_amount_ratio_percentile_df = (sector_amount_ratio_df / \
                                         sector_amount_ratio_df.rolling(period).max()).dropna()
    
    return sector_amount_ratio_percentile_df

def calc_sector_rank_weekly(sector_amount_df, on, period=50):
    '''Compute sector crowdedness quantile value for weekly amount.
    Period is 250 days (50 weeks) by default. sector_amount_df has index trade_date 
    and columns ts_codes with values the amount of trading. on specifies 
    which date in a week to represent the data of the whole week.
    on can range from 'MON' to 'FRI'.'''
    
    # Convert data to weekly frequency by summing 5-day amounts
    sector_amount_df = sector_amount_df.rolling(5).sum().dropna().resample('W-'+on).ffill()
    
    # Compute the amount ratio for each sector
    sector_amount_ratio_df = (sector_amount_df.T / sector_amount_df.sum(axis=1).T).T
    
    # Compute the quantile value within period for each date
    sector_amount_ratio_rank_df = sector_amount_ratio_df.rolling(period).apply(lambda x: \
                                (list(np.argsort(x)).index(period-1)+1) / period).dropna()
    
    return sector_amount_ratio_rank_df

def calc_sector_percentile_weekly(sector_amount_df, on, period=150):
    '''Compute sector crowdedness percentile value w.r.t. 
    the maximum within a period on a weekly basis.
    Period is 750 days by default. sector_amount_df has index trade_date and columns 
    ts_codes with values the amount of trading. on specifies 
    which date in a week to represent the data of the whole week.
    on can range from 'MON' to 'FRI'.'''
    
    # Convert data to weekly frequency by summing 5-day amounts
    sector_amount_df = sector_amount_df.rolling(5).sum().dropna().resample('W-'+on).ffill()
    
    # Compute the amount ratio for each sector
    sector_amount_ratio_df = (sector_amount_df.T / sector_amount_df.sum(axis=1).T).T
    
    # Compute the percentile w.r.t. the maximum value within period for each date
    sector_amount_ratio_percentile_df = (sector_amount_ratio_df / \
                                         sector_amount_ratio_df.rolling(period).max()).dropna()
    
    return sector_amount_ratio_percentile_df

def calc_sector_crowdedness(sector_percentile_df, sector_rank_df, threshold_percentile=0.9, threshold_rank=0.95):
    '''Compute sector crowdedness using values from calc_sector_percentile
    and calc_sector_rank. The thresholds are 0.9 and 0.95 by default.
    Note that the format of two input df's must match.'''
    # Fix the start date for calculation
    start_date = max(sector_percentile_df.index[0], sector_rank_df.index[0])

    sector_percentile_df = \
        sector_percentile_df.reset_index().query('trade_date >= @start_date').set_index('trade_date')
    sector_rank_df = \
        sector_rank_df.reset_index().query('trade_date >= @start_date').set_index('trade_date')
    
    # Assign 1 to dates when both conditions are satisfied and 0 otherwise
    sector_crowdedness_df = \
    pd.DataFrame(np.where((sector_percentile_df > threshold_percentile) & (sector_rank_df > threshold_rank), 1, 0), \
                 index=sector_rank_df.index, columns=sector_rank_df.columns)
    return sector_crowdedness_df
