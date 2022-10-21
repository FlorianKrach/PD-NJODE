"""
author: Florian Krach
"""

import numpy as np  # large arrays and matrices, functions
from sklearn.model_selection import train_test_split
import sys
import sklearn.linear_model as lm

import data_utils
from configs import config
sys.path.append("../")

try:
    from telegram_notifications import send_bot_message as SBM
except Exception:
    from config import SendBotMessage as SBM


def train_linreg(data_dict, use_midprice_only=True, last_x_times=None,
                 test_size=0.2, seed=398, batch_size=50):
    """
    train a LinReg model for regression on the LOB dataset
    Args:
        data_dict: str, name of data_dict to use
        use_midprice_only: whether only the mid-price or also the other prices
            and volumes included in the data are used
        last_x_times: None or int, the number of last time steps of the data to
            use as predictors, if None: all timesteps of the dataset are used,
            if 0: a constant prediction from last value is used instead of
            LinReg model
        test_size:
        seed:

    Returns:

    """
    dataset, dataset_id = data_utils._get_dataset_name_id_from_dict(
        data_dict=data_dict)
    dataset_id = int(dataset_id)
    dataset_metadata = data_utils.load_metadata(
        stock_model_name=dataset, time_id=dataset_id)
    eval_predict_steps = dataset_metadata['eval_predict_steps']
    train_idx, val_idx = train_test_split(
        np.arange(dataset_metadata["nb_paths"]), test_size=test_size,
        random_state=seed, shuffle=False)
    data_train = data_utils.LOBDataset(time_id=dataset_id, idx=train_idx)
    data_val = data_utils.LOBDataset(time_id=dataset_id, idx=val_idx)

    if last_x_times is None:
        last_x_times = data_train.samples.shape[2]
    if last_x_times == 0:
        Y_test = data_val.eval_samples[:, 0, eval_predict_steps[0]-1]
        Y_test_pred = data_val.samples[:, 0, -1]
    else:
        X_train = data_train.samples[:,:, -last_x_times:]
        if use_midprice_only:
            X_train = X_train[:, 0, :]
        else:
            X_train = X_train.reshape((len(X_train), -1))
        Y_train = data_train.eval_samples[:, 0, eval_predict_steps[0]-1]

        X_test = data_val.samples[:,:, -last_x_times:]
        if use_midprice_only:
            X_test = X_test[:, 0, :]
        else:
            X_test = X_test.reshape((len(X_test), -1))
        Y_test = data_val.eval_samples[:, 0, eval_predict_steps[0]-1]

        linreg = lm.LinearRegression().fit(X_train, Y_train)
        Y_test_pred = linreg.predict(X_test)

    if batch_size is None:
        batch_size = len(Y_test)
    mse = []
    for i in range(len(Y_test)//batch_size + (len(Y_test)%batch_size > 0)):
        mse.append(np.mean((Y_test[i*batch_size:(i+1)*batch_size] -
                    Y_test_pred[i*batch_size:(i+1)*batch_size])**2))
    mse = np.mean(mse)

    print("LinReg model for LOB:\ndataset: {}, use-midprice-only: {}, "
          "last-x-times: {}\ntest-MSE: {}".format(
        data_dict, use_midprice_only, last_x_times, mse))



if __name__ == '__main__':
    train_linreg(data_dict="LOB_dict2", use_midprice_only=False)
    train_linreg(data_dict="LOB_dict3", use_midprice_only=False)
    train_linreg(data_dict="LOB_dict3", use_midprice_only=True)
    train_linreg(data_dict="LOB_dict3", use_midprice_only=True, last_x_times=10)
    train_linreg(data_dict="LOB_dict3", use_midprice_only=True, last_x_times=1)
    train_linreg(data_dict="LOB_dict3", use_midprice_only=True, last_x_times=0)
    print("="*80)
    train_linreg(data_dict="LOB_dict_K_2", use_midprice_only=False)
    train_linreg(data_dict="LOB_dict_K_3", use_midprice_only=False)
    train_linreg(data_dict="LOB_dict_K_3", use_midprice_only=True)
    train_linreg(data_dict="LOB_dict_K_3", use_midprice_only=True, last_x_times=10)
    train_linreg(data_dict="LOB_dict_K_3", use_midprice_only=True, last_x_times=1)
    train_linreg(data_dict="LOB_dict_K_3", use_midprice_only=True, last_x_times=0)
    print("="*80)
    train_linreg(data_dict="LOB_dict_K_6", use_midprice_only=False)
    train_linreg(data_dict="LOB_dict_K_6", use_midprice_only=True)
    train_linreg(data_dict="LOB_dict_K_6", use_midprice_only=True, last_x_times=10)
    train_linreg(data_dict="LOB_dict_K_6", use_midprice_only=True, last_x_times=1)
    train_linreg(data_dict="LOB_dict_K_6", use_midprice_only=True, last_x_times=0)



    pass
