# Compute the JDK_RS for S&P sectorss
import pandas as pd
from datetime import datetime

from sw_level1.sw_level1_functions import get_sw_level1_codes, get_sw_level1_data, update_all_sw_level1_data
from market_index.market_index_functions import get_market_index_data, update_market_index_data

from jdk_rs_crowdedness.calc_jdk_rs import calc_jdk_daily, calc_jdk_weekly
from jdk_rs_crowdedness.calc_sector_crowdedness import calc_sector_rank_daily, calc_sector_percentile_daily, \
calc_sector_rank_weekly, calc_sector_percentile_weekly, calc_sector_crowdedness

import os

if __name__ == '__main__':
    # Update all files needed
    try:
        update_all_sw_level1_data()
    except Exception as e:
        print(f"update failed: {e}")
    try:
        update_market_index_data('000300.SH')
    except Exception as e:
        print(f"update failed: {e}")

    # The path
    file_path = os.path.abspath(os.path.dirname(__file__))

    ts_codes = get_sw_level1_codes()
    sector_df = get_sw_level1_data(ts_codes[0])
    index_df = get_market_index_data('000300.SH')

    start_date = datetime(2005, 1, 4)
    end_date = min(sector_df.iloc[-1]['trade_date'], index_df.iloc[-1]['trade_date'])

    # Retrieve index data
    index_df = \
        get_market_index_data('000300.SH').query('trade_date >= @start_date & trade_date <= @end_date').reset_index(drop=True)
    data_df = pd.DataFrame({'000300.SH': list(index_df['close'])}, index=index_df['trade_date'])

    # Retrieve sector data and merge data into data_df (a df containing all close data)
    for ts_code in ts_codes:
        sector_df = \
            get_sw_level1_data(ts_code).query('trade_date >= @start_date & trade_date <= @end_date').reset_index(drop=True)
        # Use close price
        data_df[ts_code] = list(sector_df['close'])

    # Calculate JDK RS
    # Choose Friday as the date of the week when computing weekly jdk rs
    jdk_rs_ratio_daily, jdk_rs_momentum_daily = calc_jdk_daily(data_df)
    jdk_rs_ratio_weekly, jdk_rs_momentum_weekly = calc_jdk_weekly(data_df, 'FRI')

    # Save data
    jdk_rs_ratio_daily.to_csv(os.path.join(file_path, 'sw_level1/jdk_rs/ratio_daily.csv'), index=True)
    jdk_rs_momentum_daily.to_csv(os.path.join(file_path, 'sw_level1/jdk_rs/momentum_daily.csv'), index=True)

    jdk_rs_ratio_weekly.to_csv(os.path.join(file_path, 'sw_level1/jdk_rs/ratio_weekly.csv'), index=True)
    jdk_rs_momentum_weekly.to_csv(os.path.join(file_path, 'sw_level1/jdk_rs/momentum_weekly.csv'), index=True)

    # Load data and calculate crowdedness
    data_df = get_sw_level1_data(ts_codes[0])
    data_df = data_df.set_index('trade_date')
    sector_amount_df = pd.DataFrame(list(data_df['amount']), columns=[ts_codes[0]], index=data_df.index)
    for ts_code in ts_codes[1:]:
        data_df = get_sw_level1_data(ts_code)
        data_df = data_df.set_index('trade_date')
        sector_amount_df = sector_amount_df.join(data_df['amount'], how='left')
        sector_amount_df = sector_amount_df.rename(columns={'amount': ts_code})

    sector_amount_ratio_rank = calc_sector_rank_daily(sector_amount_df)
    sector_amount_ratio_percentile = calc_sector_percentile_daily(sector_amount_df)
    sector_amount_ratio_rank_weekly = calc_sector_rank_weekly(sector_amount_df, 'FRI')
    sector_amount_ratio_percentile_weekly = calc_sector_percentile_weekly(sector_amount_df, 'FRI')

    sector_crowdedness = \
        calc_sector_crowdedness(sector_amount_ratio_percentile, sector_amount_ratio_rank)
    sector_crowdedness_weekly = \
        calc_sector_crowdedness(sector_amount_ratio_percentile_weekly, sector_amount_ratio_rank_weekly)

    # Save data
    sector_amount_ratio_rank.to_csv(os.path.join(file_path, 'sw_level1/crowdedness/rank_daily.csv'), index=True)
    sector_amount_ratio_rank_weekly.to_csv(os.path.join(file_path, 'sw_level1/crowdedness/rank_weekly.csv'), index=True)

    sector_amount_ratio_percentile.to_csv(os.path.join(file_path, 'sw_level1/crowdedness/percentile_daily.csv'), index=True)
    sector_amount_ratio_percentile_weekly.to_csv(os.path.join(file_path, 'sw_level1/crowdedness/percentile_weekly.csv'), index=True)

    sector_crowdedness.to_csv(os.path.join(file_path, 'sw_level1/crowdedness/crowdedness_daily.csv'), index=True)
    sector_crowdedness_weekly.to_csv(os.path.join(file_path, 'sw_level1/crowdedness/crowdedness_weekly.csv'), index=True)