import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split


def make_dataset(df, test_ratio=0.20):

    array_winter, array_summer = split_winter_summer(df)

    array_winter_block = permute_rows(make_two_weeks_block(array_winter), seed=10)
    array_summer_block = permute_rows(make_two_weeks_block(array_summer), seed=97)

    min_nb_samples = min(array_winter_block.shape[0], array_summer_block.shape[0])

    array_winter_block = array_winter_block[:min_nb_samples]
    array_summer_block = array_summer_block[:min_nb_samples]

    y_winter = np.zeros(array_winter_block.shape[0])
    y_summer = np.ones(array_summer_block.shape[0])

    X = np.concatenate([array_winter_block, array_summer_block], axis=0)
    y = np.concatenate([y_winter, y_summer], axis=0)

    X, y = permute_X_y_rows(X, y, seed=1)
    X_train, X_test, y_train, y_test = train_test_split(
                                            X, 
                                            y, 
                                            test_size=test_ratio, 
                                            random_state=91
                                        )

    return X_train, y_train, X_test, y_test


def split_winter_summer(df):


    df['date'] = pd.to_datetime(df['date'])

    df_winter = df[
        (df['date'].dt.month == 12) & (df['date'].dt.day >= 21) |
        (df['date'].dt.month <= 3) & (df['date'].dt.day <= 20)
    ]

    df_summer = df[
        (df['date'].dt.month == 6) & (df['date'].dt.day >= 21) |
        (df['date'].dt.month == 7) |
        (df['date'].dt.month == 8) |
        (df['date'].dt.month == 9) & (df['date'].dt.day <= 20)
    ]

    return df_winter.transpose().values[1:], df_summer.transpose().values[1:] # Drop date line

def make_two_weeks_block(arr, block_size=24*7*2):

    n_samples, n_timestamps = arr.shape
    n_blocs = n_timestamps // block_size
    arr_reshaped = arr[:, :n_blocs * block_size].reshape(n_samples * n_blocs, block_size)

    return np.array(arr_reshaped, dtype=np.float32)


def permute_rows(arr, seed=None):

    if seed is not None:
        np.random.seed(seed)

    indices = np.random.permutation(arr.shape[0])
    
    return arr[indices, :]


def permute_X_y_rows(X, y, seed=None):

    if seed is not None:
        np.random.seed(seed)

    indices = np.random.permutation(X.shape[0])
    
    return X[indices, :], y[indices]

