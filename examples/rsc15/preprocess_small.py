# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:20:12 2015

@author: Balázs Hidasi
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# PATH_TO_ORIGINAL_DATA = '/path/to/clicks/dat/file/'
PATH_TO_ORIGINAL_DATA = '/media/andreu/DATA/datasets/sequences/recsys_challenge/yoochoose/'
# PATH_TO_PROCESSED_DATA = '/path/to/store/processed/data/'
PATH_TO_PROCESSED_DATA = '/media/andreu/DATA/datasets/sequences/RSC15-Hidasi/preprocess_out/'

data = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'yoochoose-clicks.dat', sep=',', header=None, usecols=[0, 1, 2], dtype={0: np.int32, 1: str, 2: np.int64})
data.columns = ['SessionId', 'TimeStr', 'ItemId']
data['Time'] = data.TimeStr.apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ'))
del(data['TimeStr'])

# remove sessions of length 1
session_lengths = data.groupby('SessionId').size()
data = data[np.in1d(data.SessionId, session_lengths[session_lengths>1].index)]

# remove items appearing less than 5 times
item_supports = data.groupby('ItemId').size()
data = data[np.in1d(data.ItemId, item_supports[item_supports>=5].index)]
session_lengths = data.groupby('SessionId').size()

# after deleting items, remove again sessions of length 1
data = data[np.in1d(data.SessionId, session_lengths[session_lengths>=2].index)]

# set last day for validation, day before the last day for training
tmax = data.Time.max()
session_max_times = data.groupby('SessionId').Time.max()
session_train = session_max_times[(session_max_times < tmax-timedelta(days=1)) and (session_max_times >= tmax-timedelta(days=2))].index
session_valid = session_max_times[session_max_times >= tmax - timedelta(days=1)].index
train = data[np.in1d(data.SessionId, session_train)]
valid = data[np.in1d(data.SessionId, session_valid)]

# remove from valid all items that are not seen in train
valid = valid[np.in1d(valid.ItemId, train.ItemId)]

# after deleting items in valid, remove again valid sessions of length 1
tslength = valid.groupby('SessionId').size()
valid = valid[np.in1d(valid.SessionId, tslength[tslength >= 2].index)]

# info + save (full) train and test
print('Small train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
train.to_csv(PATH_TO_PROCESSED_DATA + 'rsc15_small_train.txt', sep='\t', index=False)
print('Small validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(), valid.ItemId.nunique()))
valid.to_csv(PATH_TO_PROCESSED_DATA + 'rsc15_small_valid.txt', sep='\t', index=False)