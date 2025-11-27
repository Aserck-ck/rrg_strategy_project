import numpy as np
import pandas as pd

import torch


from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import random
from market_data.sw_level2.sw_level2_jdk_functions import get_sw_level2_ratio, get_sw_level2_momentum
from market_data.sw_level1.sw_level1_jdk_functions import get_sw_level1_ratio,get_sw_level1_momentum
from market_data.sw_level1.sw_level1_crowdedness_functions import get_sw_level1_rank,get_sw_level1_percentile
from market_data.sw_level2.sw_level2_crowdedness_functions import get_sw_level2_rank,get_sw_level2_percentile
from models.module import SequentialNonOverlappingSampler
# from us_data.level2_etf.level2_etf_jdk_functions import get_level2_etf_ratio, get_level2_etf_momentum
# from us_data.level2_etf.level2_etf_functions import get_level2_etf_codes, get_level2_etf_data
def create_inout_sequences(input_data, time_window):
    inout_seq = []
    for j in range(input_data.shape[1]):
        for i in range(input_data.shape[0]-time_window):
            inout_seq.append((torch.Tensor(input_data[i:i+time_window, j].reshape(-1, 1)), \
                            torch.Tensor(np.array([input_data[i+time_window, j]]).reshape(-1, 1))))
    return inout_seq


class FactorDataset(Dataset):
    def __init__(self, data: list, lookback_window: int, predict_window: int, batch_size: int, rand=True):
        self.data = data
        self.lookback_window = lookback_window
        self.predict_window = predict_window
        self.py_rng = random.Random(1000)
        self.rng_seed = 1000
        self.window = lookback_window + predict_window
        self.batch_size = batch_size
        self.symbols={}
        self.n_samples = 0
        self.d_in = len(data)
        self.rand = rand

        if not self.check_data():
            raise ValueError("数据条目不匹配")
        self.prepare_data()


    def check_data(self):
        if self.d_in==1:
            return True
        data_shape=self.data[0].shape
        data_columns= self.data[0].columns
        self.n_samples = len(self.data[0].index)
        for data_df in self.data[1:]:
            if data_df.shape != data_shape:
                return False
            if data_df.columns.all() != data_columns.all():
                return False

        return True

    def prepare_data(self):
        self.symbols = {key:[] for key in self.data[0].columns}

        for data_df in self.data:
            for column in data_df.columns:
                self.symbols[column].append(data_df[column].values)

        for ts_code in self.symbols.keys():
            data_list=self.symbols[ts_code]
            self.symbols[ts_code] = np.array(data_list).T



    def set_epoch_seed(self, epoch: int):
        """
        Sets a new seed for the random sampler for each epoch. This is crucial
        for reproducibility in distributed training.

        Args:
            epoch (int): The current epoch number.
        """
        epoch_seed = 1000 + epoch
        self.py_rng.seed(epoch_seed)
        self.rng_seed = epoch_seed

    def set_idx_seed(self, idx: int):
        idx_seed = self.rng_seed+idx
        self.py_rng.seed(idx_seed)

    def __len__(self) -> int:
        """Returns the number of samples per epoch."""
        return self.n_samples - self.window + 1

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves a random sample from the dataset.

        Note: The `idx` argument is ignored. Instead, a random index is drawn
        from the pre-computed `self.indices` list using `self.py_rng`. This
        ensures random sampling over the entire dataset for each call.

        Args:
            idx (int): Ignored.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x_tensor (torch.Tensor): The normalized feature tensor.
        """
        # Select a random sample from the entire pool of indices.
        self.set_idx_seed(idx)
        keys = list(self.symbols.keys())
        num = min(self.batch_size, len(keys))  # 确保不超过字典大小
        if self.rand:
            sample_symbol = self.py_rng.sample(keys, num)
        else:
            sample_symbol=keys
        x = []

        for symbol in sample_symbol:
            x.append(self.symbols[symbol][idx:idx+self.window])
        x = np.array(x)
        x_tensor = torch.from_numpy(x)

        return x_tensor





class SWDataset(Dataset):
    def __init__(self, data: dict, lookback_window: int, predict_window: int,
                 z_score_feature: list, log_feature: list, origin_feature: list, stamp_feature: list):
        self.data = data
        self.lookback_window = lookback_window
        self.predict_window = predict_window
        self.py_rng = random.Random(1000)
        self.rng_seed = 1000
        self.window = lookback_window + predict_window
        self.symbols = {}
        self.stamps = {}
        self.n_samples = 0
        self.d_in = len(data)

        self.z_score_feature = z_score_feature
        self.log_feature = log_feature
        self.origin_feature = origin_feature
        self.stamp_feature = stamp_feature

        if not self.check_data():
            for key, data_df in self.data.items():
                print(f'{key}: {data_df.shape}')
            raise ValueError("数据条目不匹配")
        self.prepare_data()


    def check_data(self):
        if self.d_in==1:
            return True
        first_key = next(iter(self.data))
        data_shape=self.data[first_key].shape
        data_columns= self.data[first_key].columns
        self.n_samples = len(self.data[first_key].index)
        for key, data_df in self.data.items():
            if data_df.shape != data_shape:
                return False
            if data_df.columns.all() != data_columns.all():
                return False

        return True

    def prepare_data(self):
        first_key = next(iter(self.data))
        self.symbols = {key: [] for key in self.data[first_key].columns}
        self.stamps = {key: [] for key in self.data[first_key].columns}


        # z_score_feature
        for data_df in [self.data[key] for key in self.z_score_feature]:
            for column in data_df.columns:
                self.symbols[column].append(data_df[column].values)

        # log_feature
        for data_df in [self.data[key] for key in self.log_feature]:
            log_data_df = np.log(data_df + 1)
            for column in log_data_df.columns:
                self.symbols[column].append(log_data_df[column].values)

        # origin_feature
        for data_df in [self.data[key] for key in self.origin_feature]:
            for column in data_df.columns:
                self.symbols[column].append(data_df[column].values)

        # stamp_feature
        dates = self.data[first_key].index
        years = dates.year.values.astype(int) % 4
        months = dates.month.values.astype(int)
        days = dates.day.values.astype(int)
        for idx,column in enumerate(self.data[first_key].columns):
            self.stamps[column].append(years)
            self.stamps[column].append(months)
            self.stamps[column].append(days)
            self.stamps[column].append(np.repeat(idx,years.shape[0]))

        for data_df in [self.data[key] for key in self.stamp_feature]:
            for column in data_df.columns:
                rank = data_df[column].values*100
                self.stamps[column].append(rank)

        for ts_code in self.symbols.keys():
            data_list=self.symbols[ts_code]
            stamp_list = self.stamps[ts_code]
            self.symbols[ts_code] = np.array(data_list).T
            self.stamps[ts_code] = np.array(stamp_list).T


    def get_raw_data(self, keys:list, idx:int):
        raw_data = {}
        for key in keys:
            raw_data[key]=self.data[key].iloc[idx:idx+self.window]

        return raw_data

    def get_raw_data_all(self, keys:list, idx:int):
        raw_data = {}
        for key in keys:
            raw_data[key]=self.data[key].iloc[idx:]

        return raw_data

    def get_std(self, keys:list, idx:int):
        raw_std = {}
        for key in keys:
            raw_std[key] = self.data[key].iloc[idx:idx+self.lookback_window].std()

        return raw_std

    def set_epoch_seed(self, epoch: int):
        """
        Sets a new seed for the random sampler for each epoch. This is crucial
        for reproducibility in distributed training.

        Args:
            epoch (int): The current epoch number.
        """
        epoch_seed = 1000 + epoch
        self.py_rng.seed(epoch_seed)
        self.rng_seed = epoch_seed

    def set_idx_seed(self, idx: int):
        idx_seed = self.rng_seed+idx
        self.py_rng.seed(idx_seed)

    def __len__(self) -> int:
        """Returns the number of samples per epoch."""
        return self.n_samples - self.window + 1

    def __getitem__(self, idx: int):
        """

        Args:
            idx (int):

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - x_tensor (torch.Tensor): The normalized feature tensor.
        """
        # Select a random sample from the entire pool of indices.
        if idx<0:
            idx = self.n_samples - self.window + 1 + idx

        self.set_idx_seed(idx)
        keys = list(self.symbols.keys())
        sample_symbol = keys
        x = []
        y = []
        for symbol in sample_symbol:
            data_seq = self.symbols[symbol][idx:idx + self.window].copy()
            for i in range(len(self.z_score_feature)):
                data_seq[:,i]=data_seq[:,i]/np.std(data_seq[:self.lookback_window,i])
            x.append(data_seq)
            y.append(self.stamps[symbol][idx:idx + self.window].copy())
        x = np.array(x)
        y = np.array(y)

        x_tensor = torch.from_numpy(x)
        x_tensor = x_tensor.transpose(0,1)

        y_tensor = torch.from_numpy(y)
        y_tensor = y_tensor.transpose(0, 1)

        return x_tensor, y_tensor


# prepare_sw1_dataloaders(200,10,12)

from market_data.sw_level1.sw_level1_functions import *


def prepare_sw1_dataloaders_weekly(lookback_window, predict_window, batch_size,
                                z_score_feature, log_feature, origin_feature, stamp_feature,train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,dataset='sw1'):

    if dataset =='sw1':
        ratio_df = get_sw_level1_ratio(ts_code='all', frequency='weekly').set_index('trade_date')
        momentum_df = get_sw_level1_momentum(ts_code='all', frequency='weekly').set_index('trade_date')
        rank_df = get_sw_level1_rank(ts_code='all',frequency='weekly').set_index('trade_date')
        num_ind=31
        ts_codes = get_sw_level1_codes()
        get_sw_data_func = get_sw_level1_data
    elif dataset =='sw2':
        ratio_df = get_sw_level2_ratio(ts_code='all', frequency='weekly').set_index('trade_date')
        momentum_df = get_sw_level2_momentum(ts_code='all', frequency='weekly').set_index('trade_date')
        rank_df = get_sw_level2_rank(ts_code='all',frequency='weekly').set_index('trade_date')
        num_ind=124
        ts_codes = get_sw_level2_codes()
        get_sw_data_func = get_sw_level2_data
    elif dataset == 'etf':
        raise NotImplementedError("etf数据集尚未实现")
    else:
        raise ValueError("dataset参数错误，仅支持'sw1'或'sw2'")

    ratio_df.drop(columns=['000300.SH'], inplace=True)
    momentum_df.drop(columns=['000300.SH'], inplace=True)
    


    hs300 = get_market_index_data('000300.SH').set_index('trade_date')
    hs300 = hs300['close']
    close_df = pd.DataFrame(index=hs300.index, columns=ts_codes)
        
    for i, ts_code in enumerate(ts_codes):
        index_df = get_sw_data_func(ts_code).set_index('trade_date')
        close_df[ts_code] = index_df['close']

    close_df = close_df.resample('W-FRI').ffill()
    hs300_w = hs300.resample('W-FRI').ffill()
    close_df = (close_df-close_df.shift(predict_window))/close_df.shift(predict_window)
    hs300_w = (hs300_w-hs300_w.shift(predict_window))/hs300_w.shift(predict_window)
    close5_df = pd.DataFrame(index=close_df.index, columns=close_df.columns)
    for col in close_df.columns:
        close5_df[col] = (close_df[col]-hs300_w) * 100 - 100

    close5_df = close5_df.dropna()

    close_rank_df = (1 - close5_df.rank(axis=1, method='min', ascending=False) / num_ind)
    row_min = close5_df.min(axis=1)
    row_max = close5_df.max(axis=1)

    relative_return_df = close5_df.copy()
    for idx in close5_df.index:
        r_min = row_min[idx]
        r_max = row_max[idx]

        # 如果所有收益率相同，设为0
        if r_max == r_min:
            relative_return_df.loc[idx] = 0
        else:
            # 分别处理正收益率和负收益率
            day_returns = close5_df.loc[idx]

            # 找到正收益率的最大值和负收益率的最小值
            positive_returns = day_returns[day_returns > 0]
            negative_returns = day_returns[day_returns < 0]

            # 如果有正收益率
            if len(positive_returns) > 0:
                pos_max = positive_returns.max()
                # 正收益率缩放到(0, 1]
                positive_mask = day_returns > 0
                relative_return_df.loc[idx, positive_mask] = day_returns[positive_mask] / pos_max

            # 如果有负收益率
            if len(negative_returns) > 0:
                neg_min = negative_returns.min()
                # 负收益率缩放到[-1, 0)
                negative_mask = day_returns < 0
                relative_return_df.loc[idx, negative_mask] = day_returns[negative_mask] / abs(neg_min)

            # 收益率为0的设为0
            zero_mask = day_returns == 0
            relative_return_df.loc[idx, zero_mask] = 0

    ratio_chg1_df = (ratio_df-ratio_df.shift(1))/ratio_df.shift(1)
    ratio_chg5_df = (ratio_df-ratio_df.shift(predict_window))/ratio_df.shift(predict_window)
    momentum_chg1_df = (momentum_df - momentum_df.shift(1)) / momentum_df.shift(1)
    momentum_chg5_df = (momentum_df - momentum_df.shift(predict_window)) / momentum_df.shift(predict_window)

    distance_df = np.sqrt((ratio_df-100)**2+(momentum_df-100)**2)
    radius_df = np.arctan2(momentum_df - 100, ratio_df - 100)

    distance5_df = np.sqrt((ratio_chg5_df*100) ** 2 + (momentum_chg5_df*100) ** 2)
    radius5_df = np.arctan2(momentum_chg5_df*100, ratio_chg5_df*100)

    data_length = min(len(momentum_chg5_df.dropna()),len(rank_df.dropna()))
    bound_all = data_length
    bound_train = int(data_length * (1-train_ratio))
    bound_val = int(data_length * (1-train_ratio - val_ratio))
    if train_ratio>1:
        bound_all = train_ratio
        bound_train = val_ratio
        bound_val = test_ratio

    ratio_df = ratio_df-100
    momentum_df = momentum_df-100

    radius5_sin_df = np.sin(radius5_df)
    radius5_cos_df = np.cos(radius5_df)

    data_list = [ratio_df, momentum_df, distance5_df, radius5_cos_df, radius5_sin_df,
                 ratio_chg5_df, momentum_chg5_df, rank_df, close5_df, radius5_df,
                 relative_return_df, close_rank_df, radius_df, distance_df]

    train_list = [x.iloc[-bound_all:-bound_train] for x in data_list]
    val_list = [x.iloc[-bound_train-lookback_window:-bound_val] for x in data_list]
    test_list = [x.iloc[-bound_val-lookback_window:] for x in data_list]

    feature_list = ['ratio', 'momentum', 'distance5', 'cos', 'sin',
                    'ratio_chg5', 'momentum_chg5', 'rank', 'close5', 'radius5',
                    'relative', 'close_rank', 'radius', 'distance']

    train_data = {x: y for _, (x, y) in enumerate(zip(feature_list,train_list))}
    val_data = {x: y for _, (x, y) in enumerate(zip(feature_list, val_list))}
    test_data = {x: y for _, (x, y) in enumerate(zip(feature_list, test_list))}


    train_dataset = SWDataset(train_data, lookback_window, predict_window,
                              z_score_feature, log_feature, origin_feature, stamp_feature)

    val_dataset = SWDataset(val_data, lookback_window, predict_window,
                            z_score_feature, log_feature, origin_feature, stamp_feature)
    test_dataset = SWDataset(test_data, lookback_window, predict_window,
                             z_score_feature, log_feature, origin_feature, stamp_feature)


    # print(len(train_dataset))
    # tt=train_dataset[2790]
    # print(train_data)
    batch_size1 = (len(train_dataset) - 1) // (lookback_window+predict_window)
    batch_size1 = min(batch_size, max(batch_size1, 1))
    batch_size2 = (len(val_dataset) - 1) // (lookback_window+predict_window)
    batch_size2 = min(max(batch_size2, 1), batch_size)
    batch_size3 = (len(test_dataset) - 1) // (lookback_window+predict_window)
    batch_size3 = min(max(batch_size3, 1), batch_size)
    print(f"训练集batch_size: {batch_size1}, 验证集batch_size: {batch_size2}")
    train_sampler = SequentialNonOverlappingSampler(
        dataset_length=len(train_dataset),
        seq_len=lookback_window+predict_window,
        batch_size=batch_size1,
        shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,  # 关键：使用batch_sampler
        num_workers=4,

    )
    # t1 = next(iter(train_sampler))
    val_sampler = SequentialNonOverlappingSampler(
        dataset_length=len(val_dataset),
        seq_len=lookback_window + predict_window,
        batch_size=batch_size2,
        shuffle=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,  # 关键：使用batch_sampler
        num_workers=4,


    )
    test_sampler = SequentialNonOverlappingSampler(
        dataset_length=len(test_dataset),
        seq_len=lookback_window + predict_window,
        batch_size=batch_size3,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,  # 关键：使用batch_sampler
        num_workers=4,

    )

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=max(1, batch_size//3), shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=max(1, batch_size//3), shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, train_sampler


def prepare_sw1_backtest_weekly(lookback_window, predict_window,
                                z_score_feature, log_feature, origin_feature, stamp_feature,
                                dataset='sw1'):


    if dataset =='sw1':
        ratio_df = get_sw_level1_ratio(ts_code='all', frequency='weekly').set_index('trade_date')
        momentum_df = get_sw_level1_momentum(ts_code='all', frequency='weekly').set_index('trade_date')
        rank_df = get_sw_level1_rank(ts_code='all',frequency='weekly').set_index('trade_date')
        num_ind=31
        ts_codes = get_sw_level1_codes()
        get_sw_data_func = get_sw_level1_data
    elif dataset =='sw2':
        ratio_df = get_sw_level2_ratio(ts_code='all', frequency='weekly').set_index('trade_date')
        momentum_df = get_sw_level2_momentum(ts_code='all', frequency='weekly').set_index('trade_date')
        rank_df = get_sw_level2_rank(ts_code='all',frequency='weekly').set_index('trade_date')
        num_ind=124
        ts_codes = get_sw_level2_codes()
        get_sw_data_func = get_sw_level2_data
    else:
        raise ValueError("dataset参数错误，仅支持'sw1'或'sw2'")
    
    ratio_df.drop(columns=['000300.SH'], inplace=True)
    momentum_df.drop(columns=['000300.SH'], inplace=True)
    
    hs300 = get_market_index_data('000300.SH').set_index('trade_date')
    hs300 = hs300['close']
    close_df = pd.DataFrame(index=hs300.index, columns=ts_codes)
    for i, ts_code in enumerate(ts_codes):
        index_df = get_sw_data_func(ts_code).set_index('trade_date')
        close_df[ts_code] = index_df['close']

    close_df = close_df.resample('W-FRI').ffill()
    hs300_w = hs300.resample('W-FRI').ffill()
    close_df = (close_df - close_df.shift(predict_window)) / close_df.shift(predict_window)
    hs300_w = (hs300_w - hs300_w.shift(predict_window)) / hs300_w.shift(predict_window)
    close5_df = pd.DataFrame(index=close_df.index, columns=close_df.columns)
    for col in close_df.columns:
        close5_df[col] = (close_df[col] - hs300_w) * 100 - 100

    close5_df = close5_df.dropna()

    close_rank_df = (1 - close5_df.rank(axis=1, method='min', ascending=False) / num_ind)
    row_min = close5_df.min(axis=1)
    row_max = close5_df.max(axis=1)

    relative_return_df = close5_df.copy()
    for idx in close5_df.index:
        r_min = row_min[idx]
        r_max = row_max[idx]

        # 如果所有收益率相同，设为0
        if r_max == r_min:
            relative_return_df.loc[idx] = 0
        else:
            # 分别处理正收益率和负收益率
            day_returns = close5_df.loc[idx]

            # 找到正收益率的最大值和负收益率的最小值
            positive_returns = day_returns[day_returns > 0]
            negative_returns = day_returns[day_returns < 0]

            # 如果有正收益率
            if len(positive_returns) > 0:
                pos_max = positive_returns.max()
                # 正收益率缩放到(0, 1]
                positive_mask = day_returns > 0
                relative_return_df.loc[idx, positive_mask] = day_returns[positive_mask] / pos_max

            # 如果有负收益率
            if len(negative_returns) > 0:
                neg_min = negative_returns.min()
                # 负收益率缩放到[-1, 0)
                negative_mask = day_returns < 0
                relative_return_df.loc[idx, negative_mask] = day_returns[negative_mask] / abs(neg_min)

            # 收益率为0的设为0
            zero_mask = day_returns == 0
            relative_return_df.loc[idx, zero_mask] = 0

    ratio_chg1_df = (ratio_df-ratio_df.shift(1))/ratio_df.shift(1)
    ratio_chg5_df = (ratio_df-ratio_df.shift(predict_window))/ratio_df.shift(predict_window)
    momentum_chg1_df = (momentum_df - momentum_df.shift(1)) / momentum_df.shift(1)
    momentum_chg5_df = (momentum_df - momentum_df.shift(predict_window)) / momentum_df.shift(predict_window)

    distance_df = np.sqrt((ratio_df-100)**2+(momentum_df-100)**2)
    radius_df = np.arctan2(momentum_df - 100, ratio_df - 100)

    distance5_df = np.sqrt((ratio_chg5_df*100) ** 2 + (momentum_chg5_df*100) ** 2)
    radius5_df = np.arctan2(momentum_chg5_df*100, ratio_chg5_df*100)

    ratio_df = ratio_df-100
    momentum_df = momentum_df-100

    

    radius5_sin_df = np.sin(radius5_df)
    radius5_cos_df = np.cos(radius5_df)

    data_list = [ratio_df, momentum_df, distance5_df, radius5_cos_df, radius5_sin_df,
                 ratio_chg5_df, momentum_chg5_df, rank_df, close5_df, radius5_df,
                 relative_return_df, close_rank_df, radius_df, distance_df]
    data_length = len(momentum_chg5_df.dropna())
    for data in data_list:
        data_length = min(data_length, len(data.dropna()))
        
    bound_all = data_length
    train_list = [x.iloc[-bound_all:] for x in data_list]

    feature_list = ['ratio', 'momentum', 'distance5', 'cos', 'sin',
                    'ratio_chg5', 'momentum_chg5', 'rank', 'close5', 'radius5',
                    'relative', 'close_rank', 'radius', 'distance']

    train_data = {x: y for _, (x, y) in enumerate(zip(feature_list,train_list))}

    train_dataset = SWDataset(train_data, lookback_window, predict_window,
                              z_score_feature, log_feature, origin_feature, stamp_feature)

    return train_dataset

def prepare_sw1_predict_weekly(lookback_window, predict_window,
                                z_score_feature, log_feature, origin_feature, stamp_feature,
                                dataset='sw1'):


    if dataset =='sw1':
        ratio_df = get_sw_level1_ratio(ts_code='all', frequency='weekly').set_index('trade_date')
        momentum_df = get_sw_level1_momentum(ts_code='all', frequency='weekly').set_index('trade_date')
        rank_df = get_sw_level1_rank(ts_code='all',frequency='weekly').set_index('trade_date')
        num_ind=31
        ts_codes = get_sw_level1_codes()
        get_sw_data_func = get_sw_level1_data
    elif dataset =='sw2':
        ratio_df = get_sw_level2_ratio(ts_code='all', frequency='weekly').set_index('trade_date')
        momentum_df = get_sw_level2_momentum(ts_code='all', frequency='weekly').set_index('trade_date')
        rank_df = get_sw_level2_rank(ts_code='all',frequency='weekly').set_index('trade_date')
        num_ind=124
        ts_codes = get_sw_level2_codes()
        get_sw_data_func = get_sw_level2_data
    else:
        raise ValueError("dataset参数错误，仅支持'sw1'或'sw2'")
    
    ratio_df.drop(columns=['000300.SH'], inplace=True)
    momentum_df.drop(columns=['000300.SH'], inplace=True)
    
    hs300 = get_market_index_data('000300.SH').set_index('trade_date')
    hs300 = hs300['close']
    close_df = pd.DataFrame(index=hs300.index, columns=ts_codes)
    for i, ts_code in enumerate(ts_codes):
        index_df = get_sw_data_func(ts_code).set_index('trade_date')
        close_df[ts_code] = index_df['close']

    close_df = close_df.resample('W-FRI').ffill()
    hs300_w = hs300.resample('W-FRI').ffill()
    close_df = (close_df - close_df.shift(predict_window)) / close_df.shift(predict_window)
    hs300_w = (hs300_w - hs300_w.shift(predict_window)) / hs300_w.shift(predict_window)
    close5_df = pd.DataFrame(index=close_df.index, columns=close_df.columns)
    for col in close_df.columns:
        close5_df[col] = (close_df[col] - hs300_w) * 100 - 100

    close5_df = close5_df.dropna()

    close_rank_df = (1 - close5_df.rank(axis=1, method='min', ascending=False) / num_ind)
    row_min = close5_df.min(axis=1)
    row_max = close5_df.max(axis=1)

    relative_return_df = close5_df.copy()
    for idx in close5_df.index:
        r_min = row_min[idx]
        r_max = row_max[idx]

        # 如果所有收益率相同，设为0
        if r_max == r_min:
            relative_return_df.loc[idx] = 0
        else:
            # 分别处理正收益率和负收益率
            day_returns = close5_df.loc[idx]

            # 找到正收益率的最大值和负收益率的最小值
            positive_returns = day_returns[day_returns > 0]
            negative_returns = day_returns[day_returns < 0]

            # 如果有正收益率
            if len(positive_returns) > 0:
                pos_max = positive_returns.max()
                # 正收益率缩放到(0, 1]
                positive_mask = day_returns > 0
                relative_return_df.loc[idx, positive_mask] = day_returns[positive_mask] / pos_max

            # 如果有负收益率
            if len(negative_returns) > 0:
                neg_min = negative_returns.min()
                # 负收益率缩放到[-1, 0)
                negative_mask = day_returns < 0
                relative_return_df.loc[idx, negative_mask] = day_returns[negative_mask] / abs(neg_min)

            # 收益率为0的设为0
            zero_mask = day_returns == 0
            relative_return_df.loc[idx, zero_mask] = 0

    ratio_chg1_df = (ratio_df-ratio_df.shift(1))/ratio_df.shift(1)
    ratio_chg5_df = (ratio_df-ratio_df.shift(predict_window))/ratio_df.shift(predict_window)
    momentum_chg1_df = (momentum_df - momentum_df.shift(1)) / momentum_df.shift(1)
    momentum_chg5_df = (momentum_df - momentum_df.shift(predict_window)) / momentum_df.shift(predict_window)

    distance_df = np.sqrt((ratio_df-100)**2+(momentum_df-100)**2)
    radius_df = np.arctan2(momentum_df - 100, ratio_df - 100)

    distance5_df = np.sqrt((ratio_chg5_df*100) ** 2 + (momentum_chg5_df*100) ** 2)
    radius5_df = np.arctan2(momentum_chg5_df*100, ratio_chg5_df*100)

    ratio_df = ratio_df-100
    momentum_df = momentum_df-100

    

    radius5_sin_df = np.sin(radius5_df)
    radius5_cos_df = np.cos(radius5_df)

    data_list = [ratio_df, momentum_df, distance5_df, radius5_cos_df, radius5_sin_df,
                 ratio_chg5_df, momentum_chg5_df, rank_df, close5_df, radius5_df,
                 relative_return_df, close_rank_df, radius_df, distance_df]
    data_length = len(momentum_chg5_df.dropna())
    for data in data_list:
        data_length = min(data_length, len(data.dropna()))
        
    bound_all = data_length
    train_list = [x.iloc[-bound_all:] for x in data_list]

    feature_list = ['ratio', 'momentum', 'distance5', 'cos', 'sin',
                    'ratio_chg5', 'momentum_chg5', 'rank', 'close5', 'radius5',
                    'relative', 'close_rank', 'radius', 'distance']

    train_data = {x: y for _, (x, y) in enumerate(zip(feature_list,train_list))}

    train_dataset = SWDataset(train_data, lookback_window, 0,
                              z_score_feature, log_feature, origin_feature, stamp_feature)

    return train_dataset

def prepare_sw1_predict_dataset(lookback_window, z_score_feature, log_feature, origin_feature, stamp_feature):
    ratio_df = get_sw_level1_ratio(ts_code='all', frequency='daily').set_index('trade_date')
    momentum_df = get_sw_level1_momentum(ts_code='all', frequency='daily').set_index('trade_date')
    rank_df = get_sw_level1_rank(ts_code='all', frequency='daily').set_index('trade_date')
    percentile_df = get_sw_level1_percentile(ts_code='all', frequency='daily').set_index('trade_date')

    ratio_df.drop(columns=['000300.SH'], inplace=True)
    momentum_df.drop(columns=['000300.SH'], inplace=True)
    ts_codes = get_sw_level1_codes()
    close_df = pd.DataFrame(index=ratio_df.index, columns=ts_codes)
    for i, ts_code in enumerate(ts_codes):
        index_df = get_sw_level1_data(ts_code).set_index('trade_date')
        close_df[ts_code] = index_df['close']

    bound_all = 5000
    bound_train = 2000
    bound_val = lookback_window
    bound_test = 5

    hs300 = get_market_index_data('000300.SH').set_index('trade_date')
    hs300 = hs300['close']
    close5_df = (close_df - close_df.shift(5)) / close_df.shift(5) * 100 + 100
    hs300_chg5 = (hs300 - hs300.shift(5)) / hs300.shift(5) * 100 + 100

    for col in close5_df.columns:
        close5_df[col] = close5_df[col] / hs300_chg5 * 100 - 100
    # close5_df = close5_df.rolling(5).mean()
    close5_df = close5_df.dropna()

    close5_df = close5_df.iloc[-bound_all:]

    close_rank_df = (1 - close5_df.rank(axis=1, method='min', ascending=False) / 31)
    row_min = close5_df.min(axis=1)
    row_max = close5_df.max(axis=1)

    relative_return_df = close5_df.copy()
    for idx in close5_df.index:
        r_min = row_min[idx]
        r_max = row_max[idx]

        # 如果所有收益率相同，设为0
        if r_max == r_min:
            relative_return_df.loc[idx] = 0
        else:
            # 分别处理正收益率和负收益率
            day_returns = close5_df.loc[idx]

            # 找到正收益率的最大值和负收益率的最小值
            positive_returns = day_returns[day_returns > 0]
            negative_returns = day_returns[day_returns < 0]

            # 如果有正收益率
            if len(positive_returns) > 0:
                pos_max = positive_returns.max()
                # 正收益率缩放到(0, 1]
                positive_mask = day_returns > 0
                relative_return_df.loc[idx, positive_mask] = day_returns[positive_mask] / pos_max

            # 如果有负收益率
            if len(negative_returns) > 0:
                neg_min = negative_returns.min()
                # 负收益率缩放到[-1, 0)
                negative_mask = day_returns < 0
                relative_return_df.loc[idx, negative_mask] = day_returns[negative_mask] / abs(neg_min)

            # 收益率为0的设为0
            zero_mask = day_returns == 0
            relative_return_df.loc[idx, zero_mask] = 0

    ratio_df = ratio_df.iloc[-bound_all - 10:]
    momentum_df = momentum_df.iloc[-bound_all - 10:]
    rank_df = rank_df.iloc[-bound_all - 10:]
    percentile_df = percentile_df.iloc[-bound_all - 10:]

    ratio_chg1_df = (ratio_df - ratio_df.shift(1)) / ratio_df.shift(1)
    ratio_chg5_df = (ratio_df - ratio_df.shift(5)) / ratio_df.shift(5)
    momentum_chg1_df = (momentum_df - momentum_df.shift(1)) / momentum_df.shift(1)
    momentum_chg5_df = (momentum_df - momentum_df.shift(5)) / momentum_df.shift(5)

    ratio_chg1_df = ratio_chg1_df.iloc[-bound_all:]
    ratio_chg5_df = ratio_chg5_df.iloc[-bound_all:]
    momentum_chg1_df = momentum_chg1_df.iloc[-bound_all:]
    momentum_chg5_df = momentum_chg5_df.iloc[-bound_all:]

    distance_df = np.sqrt((ratio_df - 100) ** 2 + (momentum_df - 100) ** 2)
    radius_df = np.arctan2(momentum_df - 100, ratio_df - 100)

    distance5_df = np.sqrt((ratio_chg5_df * 100) ** 2 + (momentum_chg5_df * 100) ** 2)
    radius5_df = np.arctan2(momentum_chg5_df * 100, ratio_chg5_df * 100)

    distance5_df = distance5_df.iloc[-bound_all:]
    radius5_df = radius5_df.iloc[-bound_all:]
    ratio_df = ratio_df.iloc[-bound_all:] - 100
    momentum_df = momentum_df.iloc[-bound_all:] - 100
    rank_df = rank_df.iloc[-bound_all:]

    radius5_sin_df = np.sin(radius5_df)
    radius5_cos_df = np.cos(radius5_df)

    data_list = [ratio_df, momentum_df, distance5_df, radius5_cos_df, radius5_sin_df,
                 ratio_chg5_df, momentum_chg5_df, rank_df, close5_df, radius5_df,
                 relative_return_df, close_rank_df, radius_df, distance_df]


    test_list = [x.iloc[-bound_val-4:] for x in data_list]

    feature_list = ['ratio', 'momentum', 'distance5', 'cos', 'sin',
                    'ratio_chg5', 'momentum_chg5', 'rank', 'close5', 'radius5',
                    'relative', 'close_rank', 'radius', 'distance']
    test_data = {x: y for _, (x, y) in enumerate(zip(feature_list, test_list))}

    test_dataset = SWDataset(test_data, lookback_window, 0,
                             z_score_feature, log_feature, origin_feature, stamp_feature)

    return test_dataset

from market_data.sw_level2.sw_level2_functions import *
from market_data.market_index.market_index_functions import*
def prepare_sw2_dataloaders(lookback_window, predict_window, batch_size,
                                z_score_feature, log_feature, origin_feature, stamp_feature):

    ratio_df = get_sw_level2_ratio(ts_code='all', frequency='daily').set_index('trade_date')
    momentum_df = get_sw_level2_momentum(ts_code='all', frequency='daily').set_index('trade_date')
    rank_df = get_sw_level2_rank(ts_code='all', frequency='daily').set_index('trade_date')
    percentile_df = get_sw_level2_percentile(ts_code='all', frequency='daily').set_index('trade_date')

    ratio_df.drop(columns=['000300.SH'], inplace=True)
    momentum_df.drop(columns=['000300.SH'], inplace=True)
    ts_codes = get_sw_level2_codes()
    ratio_df=ratio_df[ts_codes]
    momentum_df = momentum_df[ts_codes]
    rank_df=rank_df[ts_codes]
    percentile_df=percentile_df[ts_codes]

    close_df = pd.DataFrame(index=ratio_df.index, columns=ts_codes)
    for i, ts_code in enumerate(ts_codes):
        index_df = get_sw_level2_data(ts_code).set_index('trade_date')
        close_df[ts_code] = index_df['close']

    bound_all = 1950
    bound_train = 800
    bound_val = 300
    bound_test = 5

    hs300 = get_market_index_data('000300.SH').set_index('trade_date')
    hs300 = hs300['close']
    close5_df = (close_df - close_df.shift(5)) / close_df.shift(5) * 100 + 100
    hs300_chg5 = (hs300 - hs300.shift(5)) / hs300.shift(5) * 100 + 100

    for col in close5_df.columns:
        close5_df[col] = close5_df[col] / hs300_chg5 * 100 - 100
    # close5_df = close5_df.rolling(5).mean()
    close5_df = close5_df.dropna()

    close5_df = close5_df.iloc[-bound_all:]

    close_rank_df = (1 - close5_df.rank(axis=1, method='min', ascending=False) / 31)
    row_min = close5_df.min(axis=1)
    row_max = close5_df.max(axis=1)

    relative_return_df = close5_df.copy()
    for idx in close5_df.index:
        r_min = row_min[idx]
        r_max = row_max[idx]

        # 如果所有收益率相同，设为0
        if r_max == r_min:
            relative_return_df.loc[idx] = 0
        else:
            # 分别处理正收益率和负收益率
            day_returns = close5_df.loc[idx]

            # 找到正收益率的最大值和负收益率的最小值
            positive_returns = day_returns[day_returns > 0]
            negative_returns = day_returns[day_returns < 0]

            # 如果有正收益率
            if len(positive_returns) > 0:
                pos_max = positive_returns.max()
                # 正收益率缩放到(0, 1]
                positive_mask = day_returns > 0
                relative_return_df.loc[idx, positive_mask] = day_returns[positive_mask] / pos_max

            # 如果有负收益率
            if len(negative_returns) > 0:
                neg_min = negative_returns.min()
                # 负收益率缩放到[-1, 0)
                negative_mask = day_returns < 0
                relative_return_df.loc[idx, negative_mask] = day_returns[negative_mask] / abs(neg_min)

            # 收益率为0的设为0
            zero_mask = day_returns == 0
            relative_return_df.loc[idx, zero_mask] = 0

    ratio_df = ratio_df.iloc[-bound_all - 10:]
    momentum_df = momentum_df.iloc[-bound_all - 10:]
    rank_df = rank_df.iloc[-bound_all - 10:]
    percentile_df = percentile_df.iloc[-bound_all - 10:]

    ratio_chg1_df = (ratio_df - ratio_df.shift(1)) / ratio_df.shift(1)
    ratio_chg5_df = (ratio_df - ratio_df.shift(5)) / ratio_df.shift(5)
    momentum_chg1_df = (momentum_df - momentum_df.shift(1)) / momentum_df.shift(1)
    momentum_chg5_df = (momentum_df - momentum_df.shift(5)) / momentum_df.shift(5)

    ratio_chg1_df = ratio_chg1_df.iloc[-bound_all:]
    ratio_chg5_df = ratio_chg5_df.iloc[-bound_all:]
    momentum_chg1_df = momentum_chg1_df.iloc[-bound_all:]
    momentum_chg5_df = momentum_chg5_df.iloc[-bound_all:]

    distance_df = np.sqrt((ratio_df - 100) ** 2 + (momentum_df - 100) ** 2)
    radius_df = np.arctan2(momentum_df - 100, ratio_df - 100)

    distance5_df = np.sqrt((ratio_chg5_df * 100) ** 2 + (momentum_chg5_df * 100) ** 2)
    radius5_df = np.arctan2(momentum_chg5_df * 100, ratio_chg5_df * 100)

    distance5_df = distance5_df.iloc[-bound_all:]
    radius5_df = radius5_df.iloc[-bound_all:]
    ratio_df = ratio_df.iloc[-bound_all:] - 100
    momentum_df = momentum_df.iloc[-bound_all:] - 100
    rank_df = rank_df.iloc[-bound_all:]

    radius5_sin_df = np.sin(radius5_df)
    radius5_cos_df = np.cos(radius5_df)

    data_list = [ratio_df, momentum_df, distance5_df, radius5_cos_df, radius5_sin_df,
                 ratio_chg5_df, momentum_chg5_df, rank_df, close5_df, radius5_df,
                 relative_return_df, close_rank_df, radius_df, distance_df]

    train_list = [x.iloc[-bound_all:-bound_train] for x in data_list]
    val_list = [x.iloc[-bound_train-lookback_window:-bound_val] for x in data_list]
    test_list = [x.iloc[-bound_val-lookback_window:] for x in data_list]

    feature_list = ['ratio', 'momentum', 'distance5', 'cos', 'sin',
                    'ratio_chg5', 'momentum_chg5', 'rank', 'close5', 'radius5',
                    'relative', 'close_rank', 'radius', 'distance']

    train_data = {x: y for _, (x, y) in enumerate(zip(feature_list, train_list))}
    val_data = {x: y for _, (x, y) in enumerate(zip(feature_list, val_list))}
    test_data = {x: y for _, (x, y) in enumerate(zip(feature_list, test_list))}
    window = lookback_window+predict_window
    train_dataset = SWDataset(train_data, lookback_window, predict_window,
                              z_score_feature, log_feature, origin_feature, stamp_feature)

    val_dataset = SWDataset(val_data, lookback_window, predict_window,
                            z_score_feature, log_feature, origin_feature, stamp_feature)
    test_dataset = SWDataset(test_data, lookback_window, predict_window,
                             z_score_feature, log_feature, origin_feature, stamp_feature)

    # print(len(train_dataset))
    # tt=train_dataset[2790]
    # print(train_data)
    batch_size1 = (len(train_dataset) - 1) // window
    batch_size1 = min(batch_size, max(batch_size1,1))
    batch_size2 = (len(val_dataset) - 1) // window
    batch_size2 = min(max(batch_size2,1), batch_size)
    batch_size3 = (len(test_dataset) - 1) // window
    batch_size3 = min(max(batch_size3,1), batch_size)

    print(f"训练集batch_size: {batch_size1}, 验证集batch_size: {batch_size2}")
    train_sampler = SequentialNonOverlappingSampler(
        dataset_length=len(train_dataset),
        seq_len=lookback_window + predict_window,
        batch_size=batch_size1,
        shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,  # 关键：使用batch_sampler
        num_workers=4,

    )
    # t1 = next(iter(train_sampler))
    val_sampler = SequentialNonOverlappingSampler(
        dataset_length=len(val_dataset),
        seq_len=lookback_window + predict_window,
        batch_size=batch_size2,
        shuffle=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,  # 关键：使用batch_sampler
        num_workers=4,

    )
    test_sampler = SequentialNonOverlappingSampler(
        dataset_length=len(test_dataset),
        seq_len=lookback_window + predict_window,
        batch_size=batch_size3,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,  # 关键：使用batch_sampler
        num_workers=4,

    )

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=max(1, batch_size//3), shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=max(1, batch_size//3), shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, train_sampler




def prepare_sw1_backtest_data(lookback_window, predict_window,
                                z_score_feature, log_feature, origin_feature, stamp_feature):


    ratio_df = get_sw_level1_ratio(ts_code='all', frequency='daily').set_index('trade_date')
    momentum_df = get_sw_level1_momentum(ts_code='all', frequency='daily').set_index('trade_date')
    rank_df = get_sw_level1_rank(ts_code='all',frequency='daily').set_index('trade_date')
    percentile_df = get_sw_level1_percentile(ts_code='all',frequency='daily').set_index('trade_date')

    ratio_df.drop(columns=['000300.SH'], inplace=True)
    momentum_df.drop(columns=['000300.SH'], inplace=True)
    ts_codes = get_sw_level1_codes()
    close_df = pd.DataFrame(index=ratio_df.index, columns=ts_codes)
    for i, ts_code in enumerate(ts_codes):
        index_df = get_sw_level1_data(ts_code).set_index('trade_date')
        close_df[ts_code] = index_df['close']

    bound_all=4000


    hs300 = get_market_index_data('000300.SH').set_index('trade_date')
    hs300 = hs300['close']
    close5_df = (close_df - close_df.shift(predict_window)) / close_df.shift(predict_window) * 100 + 100
    hs300_chg5 = (hs300 - hs300.shift(predict_window)) / hs300.shift(predict_window) * 100 + 100

    for col in close5_df.columns:
        close5_df[col] = close5_df[col] / hs300_chg5 * 100 - 100
    close5_df = close5_df.rolling(predict_window).mean()
    close5_df = close5_df.dropna()

    close5_df = close5_df.iloc[-bound_all:]

    close_rank_df = (1 - close5_df.rank(axis=1, method='min', ascending=False) / 31)
    row_min = close5_df.min(axis=1)
    row_max = close5_df.max(axis=1)

    relative_return_df = close5_df.copy()
    for idx in close5_df.index:
        r_min = row_min[idx]
        r_max = row_max[idx]

        # 如果所有收益率相同，设为0
        if r_max == r_min:
            relative_return_df.loc[idx] = 0
        else:
            # 分别处理正收益率和负收益率
            day_returns = close5_df.loc[idx]

            # 找到正收益率的最大值和负收益率的最小值
            positive_returns = day_returns[day_returns > 0]
            negative_returns = day_returns[day_returns < 0]

            # 如果有正收益率
            if len(positive_returns) > 0:
                pos_max = positive_returns.max()
                # 正收益率缩放到(0, 1]
                positive_mask = day_returns > 0
                relative_return_df.loc[idx, positive_mask] = day_returns[positive_mask] / pos_max

            # 如果有负收益率
            if len(negative_returns) > 0:
                neg_min = negative_returns.min()
                # 负收益率缩放到[-1, 0)
                negative_mask = day_returns < 0
                relative_return_df.loc[idx, negative_mask] = day_returns[negative_mask] / abs(neg_min)

            # 收益率为0的设为0
            zero_mask = day_returns == 0
            relative_return_df.loc[idx, zero_mask] = 0


    ratio_df=ratio_df.iloc[-bound_all-10:]
    momentum_df=momentum_df.iloc[-bound_all-10:]
    rank_df=rank_df.iloc[-bound_all-10:]
    percentile_df=percentile_df.iloc[-bound_all-10:]

    ratio_chg1_df = (ratio_df-ratio_df.shift(1))/ratio_df.shift(1)
    ratio_chg5_df = (ratio_df-ratio_df.shift(predict_window))/ratio_df.shift(predict_window)
    momentum_chg1_df = (momentum_df - momentum_df.shift(1)) / momentum_df.shift(1)
    momentum_chg5_df = (momentum_df - momentum_df.shift(predict_window)) / momentum_df.shift(predict_window)

    ratio_chg1_df = ratio_chg1_df.iloc[-bound_all:]
    ratio_chg5_df = ratio_chg5_df.iloc[-bound_all:]
    momentum_chg1_df = momentum_chg1_df.iloc[-bound_all:]
    momentum_chg5_df = momentum_chg5_df.iloc[-bound_all:]

    distance_df = np.sqrt((ratio_df-100)**2+(momentum_df-100)**2)
    radius_df = np.arctan2(momentum_df - 100, ratio_df - 100)

    distance5_df = np.sqrt((ratio_chg5_df*100) ** 2 + (momentum_chg5_df*100) ** 2)
    radius5_df = np.arctan2(momentum_chg5_df*100, ratio_chg5_df*100)

    distance5_df = distance5_df.iloc[-bound_all:]
    radius5_df = radius5_df.iloc[-bound_all:]
    ratio_df = ratio_df.iloc[-bound_all:]-100
    momentum_df = momentum_df.iloc[-bound_all:]-100
    rank_df = rank_df.iloc[-bound_all:]

    radius5_sin_df = np.sin(radius5_df)
    radius5_cos_df = np.cos(radius5_df)

    data_list = [ratio_df, momentum_df, distance5_df, radius5_cos_df, radius5_sin_df,
                 ratio_chg5_df, momentum_chg5_df, rank_df, close5_df, radius5_df,
                 relative_return_df, close_rank_df, radius_df, distance_df]

    all_list = [x.iloc[-bound_all:] for x in data_list]


    feature_list = ['ratio', 'momentum', 'distance5', 'cos', 'sin',
                    'ratio_chg5', 'momentum_chg5', 'rank', 'close5', 'radius5',
                    'relative', 'close_rank', 'radius', 'distance']

    all_data = {x: y for _, (x, y) in enumerate(zip(feature_list,all_list))}



    backtest_dataset = SWDataset(all_data, lookback_window, predict_window,
                              z_score_feature, log_feature, origin_feature, stamp_feature)




    return backtest_dataset


if __name__ == "__main__":
    from train_sw1_pro import get_config
    config = get_config()
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, train_sampler=(
        prepare_sw1_dataloaders_weekly(config.time_window, config.predict_window[-1], config.batch_size,
                                    config.z_score_feature, config.log_feature, config.origin_feature, config.stamp_feature))



