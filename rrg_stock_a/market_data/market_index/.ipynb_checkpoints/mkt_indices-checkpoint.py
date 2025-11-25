# Import the tushare API
import tushare as ts
ts.set_token('a2959f9452045935e19bd4b2e9765ec057b54a4bcb0f17f401628abb')
pro = ts.pro_api()

import pandas as pd
from datetime import datetime, timedelta
import time

# This is the absolute path to market_indices package
path = '/Users/jieyanzhu/Desktop/量化学习/market_data/market_indices/'
# ------------------------------------------

def get_path():
    return path

def get_index_codes(path=path):
    '''Return a list of all market index codes.'''
    return list(pd.read_csv(path+'indices_info.csv')['ts_code'][:])

def get_index_daily(index_code, start_date=datetime(2000, 1, 1), end_date=datetime.today(), path=path):
    '''Return daily data of a market index specified by an index_code.
    The date range is 2000-01-01 to datetime.today() by default.'''
    df = pd.read_csv(path+index_code+'.csv')
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df.query('trade_date >= @start_date & trade_date <= @end_date')

def update_index(index_code, end_date, path=path):
    '''Update a specific market index.'''
    index_df = get_index_daily(index_code)
    
    # Calculate the start_date for update
    start_date = index_df.iloc[-1]['trade_date'] + timedelta(days=1)
    
    # Query up-to-date data from tushare
    new_data_df = pro.index_daily(ts_code=index_code, start_date=datetime.strftime(start_date, '%Y%m%d'),
                               end_date=datetime.strftime(end_date, '%Y%m%d'), fields= 'ts_code,trade_date,close,open,high,low,pre_close,change,pct_chg,vol,amount').sort_values('trade_date').set_index('ts_code')
    new_data_df['trade_date'] = pd.to_datetime(new_data_df['trade_date'], format='%Y%m%d')
    new_data_df.rename(columns={'pct_chg': 'pct_change'})
    
    # Append new data to the original CSV file
    new_data_df.to_csv(path+index_code+'.csv', mode='a', header=False)
    
    return None

def update_mkt_indices(end_date=datetime.today()):
    '''Update market indices data.
    Note that end_date must be a datetime.datetime object.
    end_date is datetime.today() by default.'''
    
    # Access all index codes and their data correspondingly
    index_codes = get_index_codes()
    
    # Update each file
    for i, index_code in enumerate(index_codes):
        try:
            update_index(index_code, end_date)
        except:
            print('Failed to update '+index_code+'.csv file.')
            return -1
        print('Updating market indices data: '+str(i+1)+' / '+str(len(index_codes))+'.', end='\r')
        time.sleep(0.1)
    
    print('All market indices data are up-to-date.')
    return None