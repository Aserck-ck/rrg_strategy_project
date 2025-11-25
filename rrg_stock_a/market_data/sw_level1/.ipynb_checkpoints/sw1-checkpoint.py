# Import the tushare API
import tushare as ts
ts.set_token('a2959f9452045935e19bd4b2e9765ec057b54a4bcb0f17f401628abb')
pro = ts.pro_api()

import pandas as pd
from datetime import datetime, timedelta
import time

# This is the absolute path to sw_L1 package
path = '/Users/jieyanzhu/Desktop/量化学习/market_data/sw_L1/'
# ------------------------------------------

def get_path():
    return path

def get_sw1_codes(path=path):
    '''Return a list of all SW L1 industry codes.'''
    return list(pd.read_csv(path+'sw_info_L1.csv')['index_code'][:])

def get_sw1_daily(sw_code, start_date=datetime(2000, 1, 1), end_date=datetime.today(), path=path):
    '''Return daily data of an SW L1 undustry specified by an sw_code.
    The date range is 2000-01-01 to datetime.today() by default.'''
    df = pd.read_csv(path+sw_code+'.csv')
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df.query('trade_date >= @start_date & trade_date <= @end_date')

def update_industry(sw_code, end_date, path=path):
    '''Update a specific industry.'''
    industry_df = get_sw1_daily(sw_code)
    
    # Calculate the start_date for update
    start_date = industry_df.iloc[-1]['trade_date'] + timedelta(days=1)
    
    # Query up-to-date data from tushare
    new_data_df = pro.sw_daily(ts_code=sw_code, start_date=datetime.strftime(start_date, '%Y%m%d'),
                               end_date=datetime.strftime(end_date, '%Y%m%d'), fields='ts_code,trade_date,name,open,low,high,close,change,pct_change,vol,amount,pe,pb').sort_values('trade_date').set_index('ts_code')
    new_data_df['trade_date'] = pd.to_datetime(new_data_df['trade_date'], format='%Y%m%d')
    
    # Append new data to the original CSV file
    new_data_df.to_csv(path+sw_code+'.csv', mode='a', header=False)
    
    return None

def update_sw1(end_date=datetime.today()):
    '''Update SW L1 industry data.
    Note that end_date must be a datetime.datetime object.
    end_date is datetime.today() by default.'''
    
    # Access all SW L1 industry codes and their data correspondingly
    sw_codes = get_sw1_codes()
    
    # Update each file
    for i, sw_code in enumerate(sw_codes):
        try:
            update_industry(sw_code, end_date)
        except:
            print('Failed to update '+sw_code+'.csv file.')
            return -1
        print('Updating SW L1 data: '+str(i+1)+' / '+str(len(sw_codes))+'.', end='\r')
        time.sleep(0.1)
    
    print('All SW L1 data are up-to-date.')
    return None