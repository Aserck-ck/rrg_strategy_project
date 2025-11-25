import pandas as pd
import os

def get_sw_level2_ratio(ts_code='all', frequency='daily',get_abs=False):
    '''Retrieve JDK RS ratio of SW level2 sectors. By default retrieve all
    JDK ratios. If a specific stock data is wanted, input the corresponding
    wind_code. Frequency is by default daily, and weekly frequency can also 
    be admitted.'''
    file_path = os.path.abspath(os.path.dirname(__file__))

    # Return the full table
    if ts_code == 'all':
        if frequency == 'daily':
            if get_abs:
                data_path = os.path.join(file_path, 'jdk_rs', 'abs_ratio_daily.csv')
            else:
                data_path = os.path.join(file_path, 'jdk_rs', 'ratio_daily.csv')
            ratio_df = pd.read_csv(data_path)
            ratio_df['trade_date'] = pd.to_datetime(ratio_df['trade_date'], format='%Y-%m-%d')
            return ratio_df
        elif frequency == 'weekly':
            data_path = os.path.join(file_path, 'jdk_rs', 'ratio_weekly.csv')
            ratio_df = pd.read_csv(data_path)
            ratio_df['trade_date'] = pd.to_datetime(ratio_df['trade_date'], format='%Y-%m-%d')
            return ratio_df
        else:
            raise ValueError('Data with such frequency is not availabe.')
    else:
        # Return a specific stock
        if frequency == 'daily':
            if get_abs:
                data_path = os.path.join(file_path, 'jdk_rs', 'abs_ratio_daily.csv')
            else:
                data_path = os.path.join(file_path, 'jdk_rs', 'ratio_daily.csv')
            ratio_df = pd.read_csv(data_path)
            ratio_df['trade_date'] = pd.to_datetime(ratio_df['trade_date'], format='%Y-%m-%d')
            etf_ratio_df = pd.DataFrame({'trade_date': ratio_df['trade_date'], ts_code: ratio_df[ts_code]})
            return etf_ratio_df
        elif frequency == 'weekly':
            data_path = os.path.join(file_path, 'jdk_rs', 'ratio_weekly.csv')
            ratio_df = pd.read_csv(data_path)
            ratio_df['trade_date'] = pd.to_datetime(ratio_df['trade_date'], format='%Y-%m-%d')
            etf_ratio_df = pd.DataFrame({'trade_date': ratio_df['trade_date'], ts_code: ratio_df[ts_code]})
            return etf_ratio_df
        else:
            raise ValueError('Data with such frequency is not availabe.')

def get_sw_level2_momentum(ts_code='all', frequency='daily',get_abs=False):
    '''Retrieve JDK RS momentum of SW level2 sectors. By default retrieve all
    JDK momentum. If a specific stock data is wanted, input the corresponding
    wind_code. Frequency is by default daily, and weekly frequency can also 
    be admitted.'''
    file_path = os.path.abspath(os.path.dirname(__file__))

    # Return the full table
    if ts_code == 'all':
        if frequency == 'daily':
            data_path = os.path.join(file_path, 'jdk_rs', 'momentum_daily.csv')
            momentum_df = pd.read_csv(data_path)
            momentum_df['trade_date'] = pd.to_datetime(momentum_df['trade_date'], format='%Y-%m-%d')
            return momentum_df
        elif frequency == 'weekly':
            data_path = os.path.join(file_path, 'jdk_rs', 'momentum_weekly.csv')
            momentum_df = pd.read_csv(data_path)
            momentum_df['trade_date'] = pd.to_datetime(momentum_df['trade_date'], format='%Y-%m-%d')
            return momentum_df
        else:
            raise ValueError('Data with such frequency is not availabe.')
    else:
        # Return a specific stock
        if frequency == 'daily':
            data_path = os.path.join(file_path, 'jdk_rs', 'momentum_daily.csv')
            momentum_df = pd.read_csv(data_path)
            momentum_df['trade_date'] = pd.to_datetime(momentum_df['trade_date'], format='%Y-%m-%d')
            etf_momentum_df = pd.DataFrame({'trade_date': momentum_df['trade_date'], ts_code: momentum_df[ts_code]})
            return etf_momentum_df
        elif frequency == 'weekly':
            data_path = os.path.join(file_path, 'jdk_rs', 'momentum_weekly.csv')
            momentum_df = pd.read_csv(data_path)
            momentum_df['trade_date'] = pd.to_datetime(momentum_df['trade_date'], format='%Y-%m-%d')
            etf_momentum_df = pd.DataFrame({'trade_date': momentum_df['trade_date'], ts_code: momentum_df[ts_code]})
            return etf_momentum_df
        else:
            raise ValueError('Data with such frequency is not availabe.')
