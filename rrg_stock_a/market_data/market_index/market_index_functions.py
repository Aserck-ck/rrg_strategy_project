import pandas as pd
from datetime import datetime, timedelta
import os
import time

# Import the tushare API
import tushare as ts


# Set the path to data files (index data, sector data, stock data, etc.)
data_path = os.path.abspath(os.path.dirname(__file__))

def get_market_index_data_path():
    return data_path

def get_market_index_info():
    '''Get the information about the market indices.'''
    filename = os.path.join(data_path, 'market_index_info.csv')
    index_info_df = pd.read_csv(filename)
    return index_info_df

def get_market_index_codes():
    '''Return a list of all market index codes.'''
    info_df = get_market_index_info()
    return list(info_df['ts_code'])

def get_market_index_data(ts_code: str):
    '''Return daily data of a market index specified by ts_code.'''
    filename = os.path.join(data_path, ts_code+'.csv')
    data_df = pd.read_csv(filename)

    # Change the data type of 'trade_date' column to datetime
    data_df['trade_date'] = pd.to_datetime(data_df['trade_date'])
    return data_df

def update_market_index_data(ts_code: str, end_date: datetime=datetime.today()):
    '''ts_code specifies the type index data for update. 
    ts_code can be looked up with get_market_index_info().
    end_date is the last trade day for data updated;
    end_date needs to be a datetime object and is today by default.'''
    # Get the start_date of the update
    ts.set_token('a2959f9452045935e19bd4b2e9765ec057b54a4bcb0f17f401628abb')
    pro = ts.pro_api()
    filename = os.path.join(data_path, ts_code+'.csv')
    old_index_data_df = pd.read_csv(filename)

    old_index_data_df['trade_date'] = pd.to_datetime(old_index_data_df['trade_date'])
    start_date = old_index_data_df.iloc[-1]['trade_date'] + timedelta(days=1)

    # If start_date is larger than end_date, then there is no need to update
    if start_date > end_date:
        print('No need to update as the start_date is larger than the end_date.')
        return None
    
    # Query up-to-date data from tushare
    new_data_df = pro.index_daily(ts_code=ts_code, start_date=datetime.strftime(start_date, '%Y%m%d'), \
                end_date=datetime.strftime(end_date, '%Y%m%d'), \
                fields= 'ts_code,trade_date,close,open,high,low,pre_close,change,pct_chg,vol,amount')
    new_data_df['trade_date'] = pd.to_datetime(new_data_df['trade_date'], format='%Y%m%d')
    new_data_df = new_data_df.sort_values('trade_date')
    new_data_df.rename(columns={'pct_chg': 'pct_change'})

    # Multiply vol and amount by 1000 to reflect the actual value
    new_data_df['amount'] = new_data_df['amount'] * 1000
    new_data_df['vol'] = new_data_df['vol'] * 1000
    
    # Append new data to the original CSV file
    new_data_df.to_csv(filename, mode='a', header=False, index=False)
    return None

def update_all_market_index_data(end_date: datetime=datetime.today()):
    '''Update all market index data.
    end_date is the last trade day for data updated;
    end_date is datetime.today() by default.'''
    # Access all index codes and their data correspondingly
    ts.set_token('a2959f9452045935e19bd4b2e9765ec057b54a4bcb0f17f401628abb')
    pro = ts.pro_api()
    ts_codes = get_market_index_codes()
    num_index = len(ts_codes)
    
    # Update each file
    for i, ts_code in enumerate(ts_codes):
        try:
            update_market_index_data(ts_code, end_date)
        except:
            raise ConnectionError('Failed to update '+ts_code+'.csv file.')
            
        print('Updating market index data: '+str(i+1)+' / '+str(num_index)+'.', end='\r')
        time.sleep(0.1)
    
    print('Done updating all market index data.')
    return None
