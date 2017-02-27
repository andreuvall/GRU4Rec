# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:20:12 2015

@author: Balázs Hidasi
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

PATH_TO_ORIGINAL_DATA = '/path/to/clicks/dat/file/'
PATH_TO_PROCESSED_DATA = '/path/to/store/processed/data/'

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

# set last day for test, previous days for (full) training
tmax = data.Time.max()
session_max_times = data.groupby('SessionId').Time.max()
session_train = session_max_times[session_max_times < tmax-timedelta(days=1)].index
session_test = session_max_times[session_max_times >= tmax-timedelta(days=1)].index
train = data[np.in1d(data.SessionId, session_train)]
test = data[np.in1d(data.SessionId, session_test)]

# remove from test all items that are not seen in (full) train
test = test[np.in1d(test.ItemId, train.ItemId)]

# after deleting items in test, remove again test sessions of length 1
tslength = test.groupby('SessionId').size()
test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]

# info + save (full) train and test
print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
train.to_csv(PATH_TO_PROCESSED_DATA + 'rsc15_train_full.txt', sep='\t', index=False)
print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
test.to_csv(PATH_TO_PROCESSED_DATA + 'rsc15_test.txt', sep='\t', index=False)

# set the last training day for validation, previous days for training
tmax = train.Time.max()
session_max_times = train.groupby('SessionId').Time.max()
session_train = session_max_times[session_max_times < tmax-timedelta(days=1)].index
session_valid = session_max_times[session_max_times >= tmax-timedelta(days=1)].index
train_tr = train[np.in1d(train.SessionId, session_train)]
valid = train[np.in1d(train.SessionId, session_valid)]

# remove from validation all items that are not seen in train
valid = valid[np.in1d(valid.ItemId, train_tr.ItemId)]

# after deleting items in validation, remove again validation sessions of length 1
tslength = valid.groupby('SessionId').size()
valid = valid[np.in1d(valid.SessionId, tslength[tslength>=2].index)]

# info + save train and validation
print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(), train_tr.ItemId.nunique()))
train_tr.to_csv(PATH_TO_PROCESSED_DATA + 'rsc15_train_tr.txt', sep='\t', index=False)
print('Validation set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(), valid.ItemId.nunique()))
valid.to_csv(PATH_TO_PROCESSED_DATA + 'rsc15_train_valid.txt', sep='\t', index=False)
