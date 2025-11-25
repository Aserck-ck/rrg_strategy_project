import pandas as pd
from datetime import datetime, timedelta
import os
import time

# Import the tushare API
import tushare as ts
ts.set_token('a2959f9452045935e19bd4b2e9765ec057b54a4bcb0f17f401628abb')
pro = ts.pro_api()

# Set the path to data files (index data, sector data, stock data, etc.)
data_path = os.path.abspath(os.path.dirname(__file__))

def get_sw_level1_data_path():
    '''Return the absolute path to the directory containing all data files.'''
    return data_path

def get_sw_level1_info():
    '''Get the information about the sw level 1 sectors.'''
    filename = os.path.join(data_path, 'sw_level1_info.csv')
    sector_info_df = pd.read_csv(filename)
    return sector_info_df

def get_sw_level1_codes():
    '''Return a list of all sw level 1 sector codes.'''
    info_df = get_sw_level1_info()
    return list(info_df['ts_code'])

def get_sw_level1_data(ts_code: str):
    '''Return daily data of an sw level 1 sector specified by an sw_code. 
    ts_code can be looked up with get_sw_level1_info(). 
    Trade dates are from old to new.'''
    filename = os.path.join(data_path, ts_code+'.csv')
    data_df = pd.read_csv(filename)

    # Change the data type of 'trade_date' column to datetime
    data_df['trade_date'] = pd.to_datetime(data_df['trade_date'])
    return data_df

def update_sw_level1_data(ts_code: str, end_date: datetime=datetime.today()):
    '''Update sw level1 data specified by ts_code. 
    ts_code can be looked up with get_sw_level1_info(). 
    end_date is today by default.'''
    # Get the start_date of the update
    filename = os.path.join(data_path, ts_code+'.csv')
    old_data_df = pd.read_csv(filename)

    old_data_df['trade_date'] = pd.to_datetime(old_data_df['trade_date'])
    start_date = old_data_df.iloc[-1]['trade_date'] + timedelta(days=1)

    # If start_date is larger than end_date, then there is no need to update
    if start_date > end_date:
        print('No need to update as the start_date is larger than the end_date.')
        return None
    
    # Query up-to-date data from tushare
    new_data_df = pro.sw_daily(ts_code=ts_code, start_date=datetime.strftime(start_date, '%Y%m%d'), \
                end_date=datetime.strftime(end_date, '%Y%m%d'), \
                fields='ts_code,trade_date,name,open,low,high,close,change,pct_change,vol,amount,pe,pb')
    new_data_df['trade_date'] = pd.to_datetime(new_data_df['trade_date'], format='%Y%m%d')
    new_data_df = new_data_df.sort_values('trade_date')

    # Multiply vol and amount by 10000 to reflect the actual value
    new_data_df['amount'] = new_data_df['amount'] * 10000
    new_data_df['vol'] = new_data_df['vol'] * 10000

    # Append new data to the original CSV file
    new_data_df.to_csv(filename, mode='a', header=False, index=False)
    return None

def update_all_sw_level1_data(end_date: datetime=datetime.today()):
    '''Update all sw level 1 data.
    end_date is the last trade day for data updated;
    end_date is datetime.today() by default.'''
    # Access all SW L1 industry codes
    ts_codes = get_sw_level1_codes()
    num_sectors = len(ts_codes)
    
    # Iterate to update each csv file.
    for i, ts_code in enumerate(ts_codes):
        try:
            update_sw_level1_data(ts_code, end_date=end_date)
        except Exception as e:
            print(f'Failed to update {ts_code}.csv file. error: {e}')
            # raise ConnectionError('Failed to update '+ts_code+'.csv file.')
        
        print('Updating SW level 1 data: '+str(i+1)+' / '+str(num_sectors)+'.', end='\r')
        time.sleep(0.1)
    
    print('Done updating all SW level 1 data.')
    return None
