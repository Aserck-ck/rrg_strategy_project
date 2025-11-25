import pandas as pd
import os

def get_sw_level1_rank(ts_code='all', frequency='daily'):
    '''Retrieve crowdedness rank of SW level1 sectors. By default retrieve all. 
    If a specific stock data is wanted, input the corresponding
    wind_code. Frequency is by default daily, and weekly frequency can also 
    be admitted.'''
    file_path = os.path.abspath(os.path.dirname(__file__))

    # Return the full table
    if ts_code == 'all':
        if frequency == 'daily':
            data_path = os.path.join(file_path, 'crowdedness', 'rank_daily.csv')
            rank_df = pd.read_csv(data_path)
            rank_df['trade_date'] = pd.to_datetime(rank_df['trade_date'], format='%Y-%m-%d')
            return rank_df
        elif frequency == 'weekly':
            data_path = os.path.join(file_path, 'crowdedness', 'rank_weekly.csv')
            rank_df = pd.read_csv(data_path)
            rank_df['trade_date'] = pd.to_datetime(rank_df['trade_date'], format='%Y-%m-%d')
            return rank_df
        else:
            raise ValueError('Data with such frequency is not availabe.')
    else:
        # Return a specific stock
        if frequency == 'daily':
            data_path = os.path.join(file_path, 'crowdedness', 'rank_daily.csv')
            rank_df = pd.read_csv(data_path)
            rank_df['trade_date'] = pd.to_datetime(rank_df['trade_date'], format='%Y-%m-%d')
            etf_rank_df = pd.DataFrame({'trade_date': rank_df['trade_date'], ts_code: rank_df[ts_code]})
            return etf_rank_df
        elif frequency == 'weekly':
            data_path = os.path.join(file_path, 'crowdedness', 'rank_weekly.csv')
            rank_df = pd.read_csv(data_path)
            rank_df['trade_date'] = pd.to_datetime(rank_df['trade_date'], format='%Y-%m-%d')
            etf_rank_df = pd.DataFrame({'trade_date': rank_df['trade_date'], ts_code: rank_df[ts_code]})
            return etf_rank_df
        else:
            raise ValueError('Data with such frequency is not availabe.')

def get_sw_level1_percentile(ts_code='all', frequency='daily'):
    '''Retrieve crowdedness percentile of SW level 1 sectors. By default retrieve all. 
    If a specific stock data is wanted, input the corresponding
    wind_code. Frequency is by default daily, and weekly frequency can also 
    be admitted.'''
    file_path = os.path.abspath(os.path.dirname(__file__))

    # Return the full table
    if ts_code == 'all':
        if frequency == 'daily':
            data_path = os.path.join(file_path, 'crowdedness', 'percentile_daily.csv')
            percentile_df = pd.read_csv(data_path)
            percentile_df['trade_date'] = pd.to_datetime(percentile_df['trade_date'], format='%Y-%m-%d')
            return percentile_df
        elif frequency == 'weekly':
            data_path = os.path.join(file_path, 'crowdedness', 'percentile_weekly.csv')
            percentile_df = pd.read_csv(data_path)
            percentile_df['trade_date'] = pd.to_datetime(percentile_df['trade_date'], format='%Y-%m-%d')
            return percentile_df
        else:
            raise ValueError('Data with such frequency is not availabe.')
    else:
        # Return a specific stock
        if frequency == 'daily':
            data_path = os.path.join(file_path, 'crowdedness', 'percentile_daily.csv')
            percentile_df = pd.read_csv(data_path)
            percentile_df['trade_date'] = pd.to_datetime(percentile_df['trade_date'], format='%Y-%m-%d')
            etf_percentile_df = pd.DataFrame({'trade_date': percentile_df['trade_date'], ts_code: percentile_df[ts_code]})
            return etf_percentile_df
        elif frequency == 'weekly':
            data_path = os.path.join(file_path, 'crowdedness', 'percentile_weekly.csv')
            percentile_df = pd.read_csv(data_path)
            percentile_df['trade_date'] = pd.to_datetime(percentile_df['trade_date'], format='%Y-%m-%d')
            etf_percentile_df = pd.DataFrame({'trade_date': percentile_df['trade_date'], ts_code: percentile_df[ts_code]})
            return etf_percentile_df
        else:
            raise ValueError('Data with such frequency is not availabe.')

def get_sw_level1_crowdedness(ts_code='all', frequency='daily'):
    '''Retrieve crowdedness of SW level 1 sectors. By default retrieve all. 
    If a specific stock data is wanted, input the corresponding
    wind_code. Frequency is by default daily, and weekly frequency can also 
    be admitted.'''
    file_path = os.path.abspath(os.path.dirname(__file__))

    # Return the full table
    if ts_code == 'all':
        if frequency == 'daily':
            data_path = os.path.join(file_path, 'crowdedness', 'crowdedness_daily.csv')
            crowdedness_df = pd.read_csv(data_path)
            crowdedness_df['trade_date'] = pd.to_datetime(crowdedness_df['trade_date'], format='%Y-%m-%d')
            return crowdedness_df
        elif frequency == 'weekly':
            data_path = os.path.join(file_path, 'crowdedness', 'crowdedness_weekly.csv')
            crowdedness_df = pd.read_csv(data_path)
            crowdedness_df['trade_date'] = pd.to_datetime(crowdedness_df['trade_date'], format='%Y-%m-%d')
            return crowdedness_df
        else:
            raise ValueError('Data with such frequency is not availabe.')
    else:
        # Return a specific stock
        if frequency == 'daily':
            data_path = os.path.join(file_path, 'crowdedness', 'crowdedness_daily.csv')
            crowdedness_df = pd.read_csv(data_path)
            crowdedness_df['trade_date'] = pd.to_datetime(crowdedness_df['trade_date'], format='%Y-%m-%d')
            etf_crowdedness_df = pd.DataFrame({'trade_date': crowdedness_df['trade_date'], ts_code: crowdedness_df[ts_code]})
            return etf_crowdedness_df
        elif frequency == 'weekly':
            data_path = os.path.join(file_path, 'crowdedness', 'crowdedness_weekly.csv')
            crowdedness_df = pd.read_csv(data_path)
            crowdedness_df['trade_date'] = pd.to_datetime(crowdedness_df['trade_date'], format='%Y-%m-%d')
            etf_crowdedness_df = pd.DataFrame({'trade_date': crowdedness_df['trade_date'], ts_code: crowdedness_df[ts_code]})
            return etf_crowdedness_df
        else:
            raise ValueError('Data with such frequency is not availabe.')