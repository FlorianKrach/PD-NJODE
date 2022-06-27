"""
author: Florian Krach & Calypso Herrera

data utilities for creating and loading synthetic test datasets
"""


# =====================================================================================================================
import numpy as np
import json, os, time, sys
from torch.utils.data import Dataset
import torch
import socket
import copy
import pandas as pd
from absl import app
from absl import flags
from datetime import datetime
import wget
from zipfile import ZipFile
import scipy.stats

import config
import stock_model


# =====================================================================================================================
FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_params", None,
                    "name of the dict with data hyper-params")
flags.DEFINE_string("dataset_name", None,
                    "name of the dataset to generate")
flags.DEFINE_integer("seed", 0,
                     "seed for making dataset generation reproducible")

hyperparam_default = config.hyperparam_default
_STOCK_MODELS = stock_model.STOCK_MODELS
data_path = config.data_path
training_data_path = config.training_data_path


# =====================================================================================================================
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_dataset_overview(training_data_path=config.training_data_path):
    data_overview = '{}dataset_overview.csv'.format(
        training_data_path)
    makedirs(training_data_path)
    if not os.path.exists(data_overview):
        df_overview = pd.DataFrame(
            data=None, columns=['name', 'id', 'description'])
    else:
        df_overview = pd.read_csv(data_overview, index_col=0)
    return df_overview, data_overview


def create_dataset(
        stock_model_name="BlackScholes", 
        hyperparam_dict=hyperparam_default,
        seed=0):
    """
    create a synthetic dataset using one of the stock-models
    :param stock_model_name: str, name of the stockmodel, see _STOCK_MODELS
    :param hyperparam_dict: dict, contains all needed parameters for the model
    :param seed: int, random seed for the generation of the dataset
    :return: str (path where the dataset is saved), int (time_id to identify
                the dataset)
    """
    df_overview, data_overview = get_dataset_overview()

    np.random.seed(seed=seed)
    hyperparam_dict['model_name'] = stock_model_name
    obs_perc = hyperparam_dict['obs_perc']
    masked = False
    if "masked" in hyperparam_dict and hyperparam_dict['masked'] is not None:
        masked = True
        if isinstance(hyperparam_dict['masked'], float):
            masked_lambda = hyperparam_dict['masked']
        else:
            raise ValueError("please provide a float (poisson lambda) "
                             "in hyperparam_dict['masked']")
    stockmodel = _STOCK_MODELS[stock_model_name](**hyperparam_dict)
    # stock paths shape: [nb_paths, dim, time_steps]
    stock_paths, dt = stockmodel.generate_paths()
    size = stock_paths.shape
    observed_dates = np.random.random(size=(size[0], size[2]))
    if "X_dependent_observation_prob" in hyperparam_dict:
        print("use X_dependent_observation_prob")
        prob_f = eval(hyperparam_dict["X_dependent_observation_prob"])
        obs_perc = prob_f(stock_paths)
    observed_dates = (observed_dates < obs_perc)*1
    observed_dates[:, 0] = 1
    nb_obs = np.sum(observed_dates[:, 1:], axis=1)
    if masked:
        mask = np.zeros(shape=size)
        mask[:,:,0] = 1
        for i in range(size[0]):
            for j in range(1, size[2]):
                if observed_dates[i,j] == 1:
                    amount = min(1+np.random.poisson(masked_lambda), size[1])
                    observed = np.random.choice(size[1], amount, replace=False)
                    mask[i, observed, j] = 1
        observed_dates = mask

    # time_id = int(time.time())
    time_id = 1
    if len(df_overview) > 0:
        time_id = np.max(df_overview["id"].values) + 1
    file_name = '{}-{}'.format(stock_model_name, time_id)
    if stock_model_name in ["FBM"]:
        file_name = '{}[h={}]-{}'.format(
            stock_model_name,hyperparam_dict['hurst'], time_id)
    path = '{}{}/'.format(training_data_path, file_name)
    hyperparam_dict['dt'] = dt
    desc = json.dumps(hyperparam_dict, sort_keys=True)
    if os.path.exists(path):
        print('Path already exists - abort')
        raise ValueError
    df_app = pd.DataFrame(
        data=[[stock_model_name, time_id, desc]],
        columns=['name', 'id', 'description']
    )
    df_overview = pd.concat([df_overview, df_app],
                            ignore_index=True)
    df_overview.to_csv(data_overview)

    os.makedirs(path)
    with open('{}data.npy'.format(path), 'wb') as f:
        np.save(f, stock_paths)
        np.save(f, observed_dates)
        np.save(f, nb_obs)
    with open('{}metadata.txt'.format(path), 'w') as f:
        json.dump(hyperparam_dict, f, sort_keys=True)

    # stock_path dimension: [nb_paths, dimension, time_steps]
    return path, time_id


def create_combined_dataset(
        stock_model_names=("BlackScholes", "OrnsteinUhlenbeck"),
        hyperparam_dicts=(hyperparam_default, hyperparam_default),
        seed=0):
    """
    create a synthetic dataset using one of the stock-models
    :param stock_model_names: list of str, each str is a name of a stockmodel,
            see _STOCK_MODELS
    :param hyperparam_dicts: list of dict, each dict contains all needed
            parameters for the model
    :param seed: int, random seed for the generation of the dataset
    :return: str (path where the dataset is saved), int (time_id to identify
                the dataset)
    """
    df_overview, data_overview = get_dataset_overview()

    assert len(stock_model_names) == len(hyperparam_dicts)
    np.random.seed(seed=seed)

    # start to create paths from first model
    filename = 'combined_{}'.format(stock_model_names[0])
    maturity = hyperparam_dicts[0]['maturity']
    hyperparam_dicts[0]['model_name'] = stock_model_names[0]
    obs_perc = hyperparam_dicts[0]['obs_perc']
    stockmodel = _STOCK_MODELS[stock_model_names[0]](**hyperparam_dicts[0])
    stock_paths, dt = stockmodel.generate_paths()
    last = stock_paths[:, :, -1]

    # for every other model, add the paths created with this model starting at
    #   last point of previous model
    for i in range(1, len(stock_model_names)):
        dt_last = dt
        assert hyperparam_dicts[i]['dimension'] == \
               hyperparam_dicts[i-1]['dimension']
        assert hyperparam_dicts[i]['nb_paths'] == \
               hyperparam_dicts[i-1]['nb_paths']
        filename += '_{}'.format(stock_model_names[i])
        maturity += hyperparam_dicts[i]['maturity']
        hyperparam_dicts[i]['model_name'] = stock_model_names[i]
        stockmodel = _STOCK_MODELS[stock_model_names[i]](**hyperparam_dicts[i])
        _stock_paths, dt = stockmodel.generate_paths(start_X=last)
        assert dt_last == dt
        last = _stock_paths[:, :, -1]
        stock_paths = np.concatenate(
            [stock_paths, _stock_paths[:, :, 1:]], axis=2
        )

    size = stock_paths.shape
    observed_dates = np.random.random(size=(size[0], size[2]))
    observed_dates = (observed_dates < obs_perc)*1
    nb_obs = np.sum(observed_dates[:, 1:], axis=1)

    time_id = 1
    if len(df_overview) >0:
        time_id = np.max(df_overview["id"].values) + 1
    file_name = '{}-{}'.format(filename, time_id)
    path = '{}{}/'.format(training_data_path, file_name)
    if os.path.exists(path):
        print('Path already exists - abort')
        raise ValueError

    metadata = {'dt': dt, 'maturity': maturity,
                'dimension': hyperparam_dicts[0]['dimension'],
                'nb_paths': hyperparam_dicts[0]['nb_paths'],
                'model_name': 'combined',
                'stock_model_names': stock_model_names,
                'hyperparam_dicts': hyperparam_dicts}
    desc = json.dumps(metadata, sort_keys=True)

    df_app = pd.DataFrame(
        data=[[filename, time_id, desc]],
        columns=['name', 'id', 'description']
    )
    df_overview = pd.concat([df_overview, df_app],
                            ignore_index=True)
    df_overview.to_csv(data_overview)

    os.makedirs(path)
    with open('{}data.npy'.format(path), 'wb') as f:
        np.save(f, stock_paths)
        np.save(f, observed_dates)
        np.save(f, nb_obs)
    with open('{}metadata.txt'.format(path), 'w') as f:
        json.dump(metadata, f, sort_keys=True)

    return path, time_id



def _get_datasetname(time_id):
    df_overview, data_overview = get_dataset_overview()
    vals = df_overview.loc[df_overview["id"] == time_id, "name"].values
    if len(vals) >= 1:
        return vals[0]
    return None


def _get_time_id(stock_model_name="BlackScholes", time_id=None,
                 path=training_data_path):
    """
    if time_id=None, get the time id of the newest dataset with the given name
    :param stock_model_name: str
    :param time_id: None or int
    :return: int, time_id
    """
    if time_id is None:
        df_overview, _ = get_dataset_overview(path)
        if "FBM" in stock_model_name:
            hurst = float(stock_model_name.split("h=")[1][:-1])
            df_overview = df_overview.loc[
                (df_overview["name"] == "FBM") &
                (df_overview["description"].apply(
                    lambda x: '"hurst": {}'.format(hurst) in x))]
        else:
            df_overview = df_overview.loc[
                df_overview["name"] == stock_model_name]
        if len(df_overview) > 0:
            time_id = np.max(df_overview["id"].values)
        else:
            time_id = None
    return time_id


def load_metadata(stock_model_name="BlackScholes", time_id=None):
    """
    load the metadata of a dataset specified by its name and id
    :return: dict (with hyperparams of the dataset)
    """
    time_id = _get_time_id(stock_model_name=stock_model_name, time_id=time_id)
    path = '{}{}-{}/'.format(training_data_path, stock_model_name, int(time_id))
    with open('{}metadata.txt'.format(path), 'r') as f:
        hyperparam_dict = json.load(f)
    return hyperparam_dict


def load_dataset(stock_model_name="BlackScholes", time_id=None):
    """
    load a saved dataset by its name and id
    :param stock_model_name: str, name
    :param time_id: int, id
    :return: np.arrays of stock_paths, observed_dates, number_observations
                dict of hyperparams of the dataset
    """
    time_id = _get_time_id(stock_model_name=stock_model_name, time_id=time_id)
    path = '{}{}-{}/'.format(training_data_path, stock_model_name, int(time_id))

    with open('{}data.npy'.format(path), 'rb') as f:
        stock_paths = np.load(f)
        observed_dates = np.load(f)
        nb_obs = np.load(f)
    with open('{}metadata.txt'.format(path), 'r') as f:
        hyperparam_dict = json.load(f)

    return stock_paths, observed_dates, nb_obs, hyperparam_dict


class IrregularDataset(Dataset):
    """
    class for iterating over a dataset
    """
    def __init__(self, model_name, time_id=None, idx=None):
        stock_paths, observed_dates, nb_obs, hyperparam_dict = load_dataset(
            stock_model_name=model_name, time_id=time_id)
        if idx is None:
            idx = np.arange(hyperparam_dict['nb_paths'])
        self.metadata = hyperparam_dict
        self.stock_paths = stock_paths[idx]
        self.observed_dates = observed_dates[idx]
        self.nb_obs = nb_obs[idx]

    def __len__(self):
        return len(self.nb_obs)

    def __getitem__(self, idx):
        if type(idx) == int:
            idx = [idx]
        # stock_path dimension: [BATCH_SIZE, DIMENSION, TIME_STEPS]
        return {"idx": idx, "stock_path": self.stock_paths[idx], 
                "observed_dates": self.observed_dates[idx], 
                "nb_obs": self.nb_obs[idx], "dt": self.metadata['dt']}


def _get_func(name):
    """
    transform a function given as str to a python function
    :param name: str, correspond to a function,
            supported: 'exp', 'power-x' (x the wanted power)
    :return: numpy fuction
    """
    if name in ['exp', 'exponential']:
        return np.exp
    if 'power-' in name:
        x = float(name.split('-')[1])
        def pow(input):
            return np.power(input, x)
        return pow
    else:
        return None


def _get_X_with_func_appl(X, functions, axis):
    """
    apply a list of functions to the paths in X and append X by the outputs
    along the given axis
    :param X: np.array, with the data,
    :param functions: list of functions to be applied
    :param axis: int, the data_dimension (not batch and not time dim) along
            which the new paths are appended
    :return: np.array
    """
    Y = X
    for f in functions:
        Y = np.concatenate([Y, f(X)], axis=axis)
    return Y


def CustomCollateFnGen(func_names=None):
    """
    a function to get the costume collate function that can be used in
    torch.DataLoader with the wanted functions applied to the data as new
    dimensions
    -> the functions are applied on the fly to the dataset, and this additional
    data doesn't have to be saved

    :param func_names: list of str, with all function names, see _get_func
    :return: collate function, int (multiplication factor of dimension before
                and after applying the functions)
    """
    # get functions that should be applied to X, additionally to identity 
    functions = []
    if func_names is not None:
        for func_name in func_names:
            f = _get_func(func_name)
            if f is not None:
                functions.append(f)
    mult = len(functions) + 1

    def custom_collate_fn(batch):
        dt = batch[0]['dt']
        stock_paths = np.concatenate([b['stock_path'] for b in batch], axis=0)
        observed_dates = np.concatenate([b['observed_dates'] for b in batch],
                                        axis=0)
        masked = False
        mask = None
        if len(observed_dates.shape) == 3:
            masked = True
            mask = observed_dates
            observed_dates = observed_dates.max(axis=1)
        nb_obs = torch.tensor(
            np.concatenate([b['nb_obs'] for b in batch], axis=0))

        # here axis=1, since we have elements of dim
        #    [batch_size, data_dimension] => add as new data_dimensions
        start_X = torch.tensor(
            _get_X_with_func_appl(stock_paths[:, :, 0], functions, axis=1), 
            dtype=torch.float32
        )
        X = []
        if masked:
            M = []
            start_M = torch.tensor(mask[:,:,0], dtype=torch.float32).repeat(
                (1,mult))
        else:
            M = None
            start_M = None
        times = []
        time_ptr = [0]
        obs_idx = []
        current_time = 0.
        counter = 0
        for t in range(1, observed_dates.shape[-1]):
            current_time += dt
            if observed_dates[:, t].sum() > 0:
                times.append(current_time)
                for i in range(observed_dates.shape[0]):
                    if observed_dates[i, t] == 1:
                        counter += 1
                        # here axis=0, since only 1 dim (the data_dimension),
                        #    i.e. the batch-dim is cummulated outside together
                        #    with the time dimension
                        X.append(_get_X_with_func_appl(stock_paths[i, :, t], 
                                                       functions, axis=0))
                        if masked:
                            M.append(np.tile(mask[i, :, t], reps=mult))
                        obs_idx.append(i)
                time_ptr.append(counter)

        assert len(obs_idx) == observed_dates[:, 1:].sum()
        if masked:
            M = torch.tensor(M, dtype=torch.float32)
        res = {'times': np.array(times), 'time_ptr': np.array(time_ptr),
               'obs_idx': torch.tensor(obs_idx, dtype=torch.long),
               'start_X': start_X, 'n_obs_ot': nb_obs,
               'X': torch.tensor(X, dtype=torch.float32),
               'true_paths': stock_paths, 'observed_dates': observed_dates,
               'true_mask': mask,
               'M': M, 'start_M': start_M}
        return res

    return custom_collate_fn, mult


def main(arg):
    """
    function to generate datasets
    """
    del arg
    if FLAGS.dataset_name:
        dataset_name = FLAGS.dataset_name
        print('dataset_name: {}'.format(dataset_name))
    else:
        raise ValueError("Please provide --dataset_name")
    if FLAGS.dataset_params:
        dataset_params = eval("config."+FLAGS.dataset_params)
        print('dataset_params: {}'.format(dataset_params))
    else:
        raise ValueError("Please provide --dataset_params")
    if "combined_" in dataset_name:
        smn = dataset_name.split("_")[1:]
        create_combined_dataset(
            stock_model_names=smn, hyperparam_dicts=dataset_params,
            seed=FLAGS.seed)
    else:
        create_dataset(
            stock_model_name=dataset_name, hyperparam_dict=dataset_params,
            seed=FLAGS.seed)


if __name__ == '__main__':
    app.run(main)



    pass
