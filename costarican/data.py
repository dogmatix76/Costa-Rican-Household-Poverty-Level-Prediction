import os
from urllib.request import urlretrieve

import pandas as pd

COSTA_TRAIN_URL = 'https://www.kaggle.com/c/costa-rican-household-poverty-prediction/download/train.csv'
COSTA_TEST_URL = 'https://www.kaggle.com/c/costa-rican-household-poverty-prediction/download/test.csv'
# create a function to download the data
def get_fremont_data(train_filename='train.csv', test_filename='test.csv' train_url= COSTA_TRAIN_URL, test_url= COSTA_TEST_URL, force_download=False):
    """ Download and cache the fremont data
	Parameters
	==========
	train_filename and test_filename : string(optional)
	   location to save the data
    train_url and test_url: string(optional)
        web location of the data
    force_download : boolean(optional)
        if True, force redownload of data
    returns
    =======
    train and test : pandas.Dataframe
        The fremont bridge data
    """
    if force_download or not os.path.exists(filename):
        urlretrieve(train_url, train_filename)
        urlretrieve(test_url, test_filename)
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test
