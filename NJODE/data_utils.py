"""
author: Florian Krach & Calypso Herrera

data utilities for creating and loading synthetic test datasets
"""


# =====================================================================================================================
import numpy as np
import json, os, time
from torch.utils.data import Dataset
import torch
import copy
import pandas as pd
from absl import app
from absl import flags
import wget
from zipfile import ZipFile

from configs import config
import synthetic_datasets


# =====================================================================================================================
FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_params", None,
                    "name of the dict with data hyper-params")
flags.DEFINE_string("dataset_name", None,
                    "name of the dataset to generate")
flags.DEFINE_integer("seed", 0,
                     "seed for making dataset generation reproducible")

hyperparam_default = config.hyperparam_default
_STOCK_MODELS = synthetic_datasets.DATASETS
data_path = config.data_path
training_data_path = config.training_data_path


# =====================================================================================================================
def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_dataset_overview(training_data_path=training_data_path):
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
            it can also contain additional options for dataset generation:
                - masked    None, float or array of floats. if None: no mask is
                            used; if float: lambda of the poisson distribution;
                            if array of floats: gives the bernoulli probability
                            for each coordinate to be observed
                - timelag_in_dt_steps   None or int. if None: no timelag used;
                            if int: number of (dt) steps by which the 1st
                            coordinate is shifted to generate the 2nd coord.,
                            this is used to generate mask accordingly (such that
                            second coord. is observed whenever the information
                            is already known from first coord.)
                - timelag_shift1    bool, if True: observe the second coord.
                            additionally only at one step after the observation
                            times of the first coord. with the given prob., if
                            False: observe the second coord. additionally at all
                            times within timelag_in_dt_steps after observation
                            times of first coordinate, at each time with given
                            probability (in masked); default: True
                - X_dependent_observation_prob   not given or str, if given:
                            string that can be evaluated to a function that is
                            applied to the generated paths to get the
                            observation probability for each coordinate
                - obs_scheme   dict, if given: specifies the observation scheme
                - obs_noise    dict, if given: add noise to the observations
                            the dict needs the following keys: 'distribution'
                            (defining the distribution of the noise), and keys
                            for the parameters of the distribution (depending on
                            the used distribution); supported distributions
                            {'normal'}. Be aware that the noise needs to be
                            centered for the model to be able to learn the
                            correct dynamics.

    :param seed: int, random seed for the generation of the dataset
    :return: str (path where the dataset is saved), int (time_id to identify
                the dataset)
    """
    df_overview, data_overview = get_dataset_overview()

    np.random.seed(seed=seed)
    hyperparam_dict['model_name'] = stock_model_name
    original_desc = json.dumps(hyperparam_dict, sort_keys=True)
    obs_perc = hyperparam_dict['obs_perc']
    obs_scheme = None
    if "obs_scheme" in hyperparam_dict:
        obs_scheme = hyperparam_dict["obs_scheme"]
    masked = False
    masked_lambda = None
    mask_probs = None
    timelag_in_dt_steps = None
    timelag_shift1 = True
    if "masked" in hyperparam_dict and hyperparam_dict['masked'] is not None:
        masked = True
        if isinstance(hyperparam_dict['masked'], float):
            masked_lambda = hyperparam_dict['masked']
        elif isinstance(hyperparam_dict['masked'], (tuple, list)):
            mask_probs = hyperparam_dict['masked']
            assert len(mask_probs) == hyperparam_dict['dimension']
        else:
            raise ValueError("please provide a float (poisson lambda) "
                             "in hyperparam_dict['masked']")
        if "timelag_in_dt_steps" in hyperparam_dict:
            timelag_in_dt_steps = hyperparam_dict["timelag_in_dt_steps"]
        if "timelag_shift1" in hyperparam_dict:
            timelag_shift1 = hyperparam_dict["timelag_shift1"]

    stockmodel = _STOCK_MODELS[stock_model_name](**hyperparam_dict)
    # stock paths shape: [nb_paths, dim, time_steps]
    stock_paths, dt = stockmodel.generate_paths()
    size = stock_paths.shape
    if obs_scheme is None:
        observed_dates = np.random.random(size=(size[0], size[2]))
        if "X_dependent_observation_prob" in hyperparam_dict:
            print("use X_dependent_observation_prob")
            prob_f = eval(hyperparam_dict["X_dependent_observation_prob"])
            obs_perc = prob_f(stock_paths)
        observed_dates = (observed_dates < obs_perc)*1
        observed_dates[:, 0] = 1
        nb_obs = np.sum(observed_dates[:, 1:], axis=1)
    else:
        if obs_scheme["name"] == "NJODE3-Example4.9":
            """
            implements the observation scheme from Example 4.9 in the NJODE3 
            paper based in 1st coordinate of the process (in case there are more 
            coordinates).
            """
            print("use observation scheme: NJODE3-Example4.9")
            observed_dates = np.zeros(shape=(size[0], size[2]))
            observed_dates[:, 0] = 1
            p = obs_scheme["p"]
            eta = obs_scheme["eta"]
            for i in range(size[0]):
                x0 = stock_paths[i, 0, 0]
                last_observation = x0
                last_obs_time = 0
                for j in range(1, size[2]):
                    v1 = np.random.binomial(1, 1/(j-last_obs_time), 1)
                    v3 = np.random.binomial(1, p, 1)
                    v2 = np.random.normal(0, eta, 1)
                    m = v1*(
                            last_observation+v2 >= stockmodel.next_cond_exp(
                        x0, j*dt, j*dt)) + (1-v1)*v3
                    observed_dates[i, j] = m
                    if m == 1:
                        last_observation = stock_paths[i, 0, j]
                        last_obs_time = j
            nb_obs = np.ones(shape=(size[0],))*size[2]
        else:
            raise ValueError("obs_scheme {} not implemented".format(
                obs_scheme["name"]))
    if masked:
        mask = np.zeros(shape=size)
        mask[:,:,0] = 1
        for i in range(size[0]):
            for j in range(1, size[2]):
                if observed_dates[i,j] == 1:
                    if masked_lambda is not None:
                        amount = min(1+np.random.poisson(masked_lambda),
                                     size[1])
                        observed = np.random.choice(
                            size[1], amount, replace=False)
                        mask[i, observed, j] = 1
                    elif mask_probs is not None:
                        for k in range(size[1]):
                            mask[i, k, j] = np.random.binomial(1, mask_probs[k])
        if timelag_in_dt_steps is not None:
            mask_shift = np.zeros_like(mask[:,0,:])
            mask_shift[:,timelag_in_dt_steps:] = mask[:,0,:-timelag_in_dt_steps]
            if timelag_shift1:
                mask_shift1 = np.zeros_like(mask[:,1,:])
                mask_shift1[:,0] = 1
                mask_shift1[:, 2:] = mask[:, 1, 1:-1]
                mask[:,1,:] = np.maximum(mask_shift1, mask_shift)
            else:
                mult = copy.deepcopy(mask[:,0,:])
                for i in range(1, timelag_in_dt_steps):
                    mult[:,i:] = np.maximum(mult[:,i:], mask[:,0,:-i])
                mask1 = mult*np.random.binomial(1, mask_probs[1], mult.shape)
                mask[:,1,:] = np.maximum(mask1, mask_shift)
        observed_dates = mask
    if "obs_noise" in hyperparam_dict:
        obs_noise_dict = hyperparam_dict["obs_noise"]
        if obs_noise_dict["distribution"] == "normal":
            obs_noise = np.random.normal(
                loc=obs_noise_dict["loc"],
                scale=obs_noise_dict["scale"],
                size=size)
            if 'noise_at_start' in obs_noise_dict and \
                    obs_noise_dict['noise_at_start']:
                pass
            else:
                obs_noise[:,:,0] = 0
        else:
            raise ValueError("obs_noise distribution {} not implemented".format(
                obs_noise_dict["distribution"]))
    else:
        obs_noise = None

    # time_id = int(time.time())
    time_id = 1
    if len(df_overview) > 0:
        time_id = np.max(df_overview["id"].values) + 1
    file_name = '{}-{}'.format(stock_model_name, time_id)
    path = '{}{}/'.format(training_data_path, file_name)
    hyperparam_dict['dt'] = dt
    desc = json.dumps(hyperparam_dict, sort_keys=True)
    if os.path.exists(path):
        print('Path already exists - abort')
        raise ValueError
    df_app = pd.DataFrame(
        data=[[stock_model_name, time_id, original_desc]],
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
        if obs_noise is not None:
            np.save(f, obs_noise)
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


def create_LOB_dataset(hyperparam_dict=hyperparam_default,
                       seed=0):
    """
    create Limit Order Book (LOB) datasets.
    Args:
        hyperparam_dict: dict, with all needed hyperparams
        seed: int, the seed for any random numbers
    Returns: path to dataset, id of dataset
    """
    if "which_raw_data" in hyperparam_dict and \
        hyperparam_dict["which_raw_data"] in [
        "ADA_1min", "ADA_1sec", "ADA_5min", "BTC_1min", "BTC_1sec", "BTC_5min",
        "ETH_1min", "ETH_1sec", "ETH_5min",]:
        df_raw = get_rawLOB_dataset2(hyperparam_dict, seed)
    else:
        df_raw = get_rawLOB_dataset1(hyperparam_dict, seed)

    df_overview, data_overview = get_dataset_overview()
    model_name = "LOB"

    np.random.seed(seed=seed)
    hyperparam_dict['model_name'] = model_name
    original_desc = json.dumps(hyperparam_dict, sort_keys=True)
    level = hyperparam_dict["LOB_level"]
    amount_obs = hyperparam_dict["amount_obs"]
    eval_predict_steps = hyperparam_dict["eval_predict_steps"]
    use_volume = hyperparam_dict["use_volume"]
    normalize = hyperparam_dict["normalize"]
    start_at_0 = True
    if "start_at_0" in hyperparam_dict:
        start_at_0 = hyperparam_dict["start_at_0"]
    shift = hyperparam_dict["shift"]
    max_pred_step = int(np.max(eval_predict_steps))
    hyperparam_dict["max_pred_step"] = max_pred_step
    length = amount_obs+max_pred_step

    bpc = []
    apc = []
    bvc = []
    avc = []
    for i in range(max(1, level)):
        bpc.append("bid_price_{}".format(i+1))
        apc.append("ask_price_{}".format(i+1))
        bvc.append("bid_amount_{}".format(i+1))
        avc.append("ask_amount_{}".format(i+1))

    df_raw.set_index("time", inplace=True)
    df_raw = df_raw[bpc+apc+bvc+avc]
    df_raw.drop_duplicates(keep="first", inplace=True)
    df_raw.reset_index(inplace=True)
    df_raw["time"] = pd.to_datetime(
        df_raw["time"], format="%Y-%m-%d %H:%M:%S.%f", utc=True)
    df_raw["time"] = df_raw["time"].apply(lambda x: x.timestamp())
    if normalize:
        price_mean = np.mean(df_raw[bpc+apc].values)
        price_std = np.std(df_raw[bpc+apc].values)
        vol_mean = np.mean(df_raw[bvc+avc].values)
        vol_std = np.std(df_raw[bvc+avc].values)
        df_raw[bpc+apc] = (df_raw[bpc+apc] - price_mean) / price_std
        df_raw[bvc+avc] = (df_raw[bvc+avc] - vol_mean) / vol_std
    else:
        price_mean = 0.
        price_std = 1.
        vol_mean = 0.
        vol_std = 1.
    df_raw["mid_price"] = (df_raw[apc[0]] + df_raw[bpc[0]]) / 2
    # dt = df_raw["time"].diff().min(skipna=True)
    dt = df_raw["time"].diff().median(skipna=True)
    hyperparam_dict["dt"] = dt
    hyperparam_dict["price_mean"] = price_mean
    hyperparam_dict["price_std"] = price_std
    hyperparam_dict["vol_mean"] = vol_mean
    hyperparam_dict["vol_std"] = vol_std

    # make samples
    # l = int(len(df_raw)/length)
    l = int((len(df_raw)-length)/shift)+1
    hyperparam_dict["nb_paths"] = l
    cols = ["mid_price"]
    if level > 0:
        cols += bpc+apc
        if use_volume:
            cols += bvc+avc
    dim = len(cols)
    samples = np.zeros(shape=(l, dim, amount_obs))
    times = np.zeros(shape=(l, amount_obs))
    eval_samples = np.zeros(shape=(l, dim, max_pred_step))
    eval_times = np.zeros(shape=(l, max_pred_step))

    # split up data into samples
    for i in range(l):
        # df_ = df_raw.iloc[i*length:(i+1)*length, :]
        df_ = df_raw.iloc[i*shift:i*shift+length, :]
        df = df_.iloc[:amount_obs, :]
        df_eval = df_.iloc[amount_obs:amount_obs+max_pred_step, :]
        samples[i, :, :] = np.transpose(df[cols].values)
        times[i, :] = df["time"].values - df["time"].values[0]
        eval_samples[i, :, :] = np.transpose(df_eval[cols].values)
        eval_times[i, :] = df_eval["time"].values - df["time"].values[0]

    # generate labels
    eval_labels = np.zeros(shape=(l, len(eval_predict_steps)))
    thresholds = []
    for i, k in enumerate(eval_predict_steps):
        m_minus = np.mean(samples[:, 0, -k:]*price_std + price_mean, axis=1)
        m_plus = np.mean(eval_samples[:, 0, :k]*price_std + price_mean, axis=1)
        pctc = (m_plus - m_minus) / m_minus
        threshold = np.quantile(pctc, q=2/3)
        thresholds.append(threshold)
        eval_labels[pctc > threshold, i] = 1
        eval_labels[pctc < -threshold, i] = -1
        print("steps ahead: ", k)
        print("amount label 1: {}".format(np.sum(eval_labels[:, i] == 1)))
        print("amount label 0: {}".format(np.sum(eval_labels[:, i] == 0)))
        print("amount label -1: {}".format(np.sum(eval_labels[:, i] == -1)))
    hyperparam_dict["thresholds"] = thresholds

    # shift samples and eval samples s.t. they start at 0
    if start_at_0:
        eval_samples -= np.repeat(samples[:, :, 0:1], axis=2,
                                  repeats=eval_samples.shape[2])
        samples -= np.repeat(samples[:, :, 0:1], axis=2,
                             repeats=samples.shape[2])

    # save the dataset
    time_id = 1
    if len(df_overview) > 0:
        time_id = np.max(df_overview["id"].values) + 1
    file_name = '{}-{}'.format(model_name, time_id)
    path = '{}{}/'.format(training_data_path, file_name)
    desc = json.dumps(hyperparam_dict, sort_keys=True)
    if os.path.exists(path):
        print('Path already exists - abort')
        raise ValueError
    df_app = pd.DataFrame(
        data=[[model_name, time_id, original_desc]],
        columns=['name', 'id', 'description']
    )
    df_overview = pd.concat([df_overview, df_app],
                            ignore_index=True)
    df_overview.to_csv(data_overview)

    os.makedirs(path)
    with open('{}data.npy'.format(path), 'wb') as f:
        np.save(f, samples)
        np.save(f, times)
        np.save(f, eval_samples)
        np.save(f, eval_times)
        np.save(f, eval_labels)
    with open('{}metadata.txt'.format(path), 'w') as f:
        json.dump(hyperparam_dict, f, sort_keys=True)

    # stock_path dimension: [nb_paths, dimension, time_steps]
    return path, time_id


def get_rawLOB_dataset2(
        hyperparam_dict=hyperparam_default,
        seed=0):
    raw_data_path = config.LOB_data_path2
    makedirs(raw_data_path)
    np.random.seed(seed=seed)
    level = hyperparam_dict["LOB_level"]

    # load raw data and preprocess
    raw_data_dir = "{}{}.csv".format(
        raw_data_path, hyperparam_dict["which_raw_data"])
    if not os.path.exists(raw_data_dir):
        print("raw LOB ({}) data not found -> dowloading ...".format(
            hyperparam_dict["which_raw_data"]))
        zip_file = wget.download(
            "https://polybox.ethz.ch/index.php/s/JJ2eRMB3JmMTVzr/download",
            training_data_path)
        print("extracting zip ...")
        with ZipFile(zip_file, 'r') as zipObj:
            zipObj.extractall(path=training_data_path)
        print("removing zip ...")
        os.remove(zip_file)
        print("download complete!")
    df_raw = pd.read_csv(raw_data_dir, index_col=0)

    for i in range(max(1, level)):
        df_raw["bid_amount_{}".format(i+1)] = \
            df_raw["bids_limit_notional_{}".format(i)]
        df_raw["ask_amount_{}".format(i+1)] = \
            df_raw["asks_limit_notional_{}".format(i)]
        df_raw["bid_price_{}".format(i+1)] = \
            df_raw["midpoint"] + \
            df_raw["midpoint"]*df_raw["bids_distance_{}".format(i)]
        df_raw["ask_price_{}".format(i+1)] = \
            df_raw["midpoint"] + \
            df_raw["midpoint"]*df_raw["asks_distance_{}".format(i)]
    df_raw["time"] = df_raw["system_time"]

    return df_raw


def get_rawLOB_dataset1(
        hyperparam_dict=hyperparam_default,
        seed=0):
    raw_data_path = config.LOB_data_path
    np.random.seed(seed=seed)

    # load raw data and preprocess
    raw_data_dir = "{}sample.csv".format(raw_data_path)
    if not os.path.exists(raw_data_dir):
        print("raw LOB data not found -> dowloading ...")
        zip_file = wget.download(
            " >>> put here the link to a raw dataset <<< ",
            training_data_path)
        print("extracting zip ...")
        with ZipFile(zip_file, 'r') as zipObj:
            zipObj.extractall(path=training_data_path)
        print("removing zip ...")
        os.remove(zip_file)
        print("download complete!")
    df_raw = pd.read_csv(raw_data_dir)

    return df_raw


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
        df_overview = df_overview.loc[
            df_overview["name"] == stock_model_name]
        if len(df_overview) > 0:
            time_id = np.max(df_overview["id"].values)
        else:
            time_id = None
    return time_id


def _get_dataset_name_id_from_dict(data_dict):
    if isinstance(data_dict, str):
        data_dict = eval("config."+data_dict)
    desc = json.dumps(data_dict, sort_keys=True)
    df_overview, _ = get_dataset_overview()
    which = df_overview.loc[df_overview["description"] == desc].index
    if len(which) == 0:
        ValueError("the given dataset does not exist yet, please generate it "
                   "first using data_utils.py. \ndata_dict: {}".format(
            data_dict))
    elif len(which) > 1:
        print("WARNING: multiple datasets match the description, returning the "
              "last one. To uniquely identify the wanted dataset, please "
              "provide the dataset_id instead of the data_dict.")
    return list(df_overview.loc[which[-1], ["name", "id"]].values)


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

    if stock_model_name == "LOB":
        with open('{}data.npy'.format(path), 'rb') as f:
            samples = np.load(f)
            times = np.load(f)
            eval_samples = np.load(f)
            eval_times = np.load(f)
            eval_labels = np.load(f)
        with open('{}metadata.txt'.format(path), 'r') as f:
            hyperparam_dict = json.load(f)
        return samples, times, eval_samples, eval_times, eval_labels, \
               hyperparam_dict

    with open('{}metadata.txt'.format(path), 'r') as f:
        hyperparam_dict = json.load(f)
    with open('{}data.npy'.format(path), 'rb') as f:
        stock_paths = np.load(f)
        observed_dates = np.load(f)
        nb_obs = np.load(f)
        if "obs_noise" in hyperparam_dict:
            obs_noise = np.load(f)
        else:
            obs_noise = None

    return stock_paths, observed_dates, nb_obs, hyperparam_dict, obs_noise


class IrregularDataset(Dataset):
    """
    class for iterating over a dataset
    """
    def __init__(self, model_name, time_id=None, idx=None):
        stock_paths, observed_dates, nb_obs, hyperparam_dict, obs_noise = \
            load_dataset(stock_model_name=model_name, time_id=time_id)
        if idx is None:
            idx = np.arange(hyperparam_dict['nb_paths'])
        self.metadata = hyperparam_dict
        self.stock_paths = stock_paths[idx]
        self.observed_dates = observed_dates[idx]
        self.nb_obs = nb_obs[idx]
        self.obs_noise = obs_noise

    def __len__(self):
        return len(self.nb_obs)

    def __getitem__(self, idx):
        if type(idx) == int:
            idx = [idx]
        if self.obs_noise is None:
            obs_noise = None
        else:
            obs_noise = self.obs_noise[idx]
        # stock_path dimension: [BATCH_SIZE, DIMENSION, TIME_STEPS]
        return {"idx": idx, "stock_path": self.stock_paths[idx], 
                "observed_dates": self.observed_dates[idx], 
                "nb_obs": self.nb_obs[idx], "dt": self.metadata['dt'],
                "obs_noise": obs_noise}


class LOBDataset(Dataset):
    def __init__(self, time_id, idx=None):
        samples, times, eval_samples, eval_times, eval_labels, \
        hp_dict = load_dataset(
            stock_model_name="LOB", time_id=time_id)
        if idx is None:
            idx = np.arange(len(samples))
        self.metadata = hp_dict
        self.samples = samples[idx]
        self.times = times[idx]
        self.eval_samples = eval_samples[idx]
        self.eval_times = eval_times[idx]
        self.eval_labels = eval_labels[idx]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if type(idx) == int:
            idx = [idx]
        # stock_path dimension: [BATCH_SIZE, DIMENSION, TIME_STEPS]
        return {"idx": idx, "samples": self.samples[idx],
                "times": self.times[idx],
                "eval_samples": self.eval_samples[idx],
                "eval_times": self.eval_times[idx],
                "eval_labels": self.eval_labels[idx],
                "amount_obs": self.metadata["amount_obs"],
                "eval_predict_steps": self.metadata["eval_predict_steps"],
                "max_pred_step": self.metadata["max_pred_step"],
                "dt": self.metadata['dt']}


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
        try:
            return eval(name)
        except Exception:
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
        obs_noise = np.concatenate([b['obs_noise'] for b in batch], axis=0)
        if obs_noise[0] is None:
            obs_noise = None
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
        sp = stock_paths[:, :, 0]
        if obs_noise is not None:
            sp = stock_paths[:, :, 0] + obs_noise[:, :, 0]
        start_X = torch.tensor(
            _get_X_with_func_appl(sp, functions, axis=1),
            dtype=torch.float32)
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
                        sp = stock_paths[i, :, t]
                        if obs_noise is not None:
                            sp = stock_paths[i, :, t] + obs_noise[i, :, t]
                        X.append(_get_X_with_func_appl(sp, functions, axis=0))
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
               'X': torch.tensor(np.array(X), dtype=torch.float32),
               'true_paths': stock_paths, 'observed_dates': observed_dates,
               'true_mask': mask, 'obs_noise': obs_noise,
               'M': M, 'start_M': start_M}
        return res

    return custom_collate_fn, mult


def LOBCollateFnGen(data_type="train", use_eval_on_train=True,
                    train_classifier=False):
    """
    a function to get the costume collate function that can be used in
    torch.DataLoader with the wanted functions applied to the data as new
    dimensions
    -> the functions are applied on the fly to the dataset, and this additional
    data doesn't have to be saved

    :param data_type: one of {"train", "test"}
    :param use_eval_on_train: bool, whether to use the eval parts of the samples
    :param train_classifier: bool, whether a classifier is trained

    :return: collate function
    """

    def custom_collate_fn(batch):
        amount_obs = batch[0]['amount_obs']
        eval_predict_steps = batch[0]['eval_predict_steps']
        max_pred_step = batch[0]['max_pred_step']
        samples = np.concatenate([b['samples'] for b in batch], axis=0)
        times = np.concatenate([b['times'] for b in batch], axis=0)
        eval_samples = np.concatenate([b['eval_samples'] for b in batch],axis=0)
        eval_times = np.concatenate([b['eval_times'] for b in batch], axis=0)
        eval_labels = np.concatenate([b['eval_labels'] for b in batch], axis=0)

        if data_type == "train":
            if use_eval_on_train and not train_classifier:
                samples = np.concatenate([samples, eval_samples], axis=2)
                times = np.concatenate([times, eval_times], axis=1)
                nb_obs = amount_obs+max_pred_step
                predict_times = None
                predict_vals = None
                predict_labels = None
            else:
                nb_obs = amount_obs
                pred_indices = [x-1 for x in eval_predict_steps]
                predict_times = eval_times[:, pred_indices]
                predict_vals = eval_samples[:, :, pred_indices]
                predict_labels = torch.tensor(eval_labels+1, dtype=torch.long)
            max_time = np.max(times[:, -1])
        else:
            nb_obs = amount_obs
            pred_indices = [x-1 for x in eval_predict_steps]
            predict_times = eval_times[:, pred_indices]
            predict_vals = eval_samples[:, :, pred_indices]
            predict_labels = torch.tensor(eval_labels+1, dtype=torch.long)
            max_time = np.max(predict_times[:, -1])

        start_X = torch.tensor(samples[:, :, 0], dtype=torch.float32)
        X = []
        time_ptr = [0]
        obs_idx = []
        counter = 0
        all_times = []
        curr_times = copy.deepcopy(times[:, 1])
        curr_time_ptr = np.ones_like(curr_times)

        ti = time.time()
        while np.any(curr_time_ptr < nb_obs):
            next_t = np.min(curr_times)
            all_times.append(next_t)
            which = curr_times == next_t
            for i, w in enumerate(which):
                if w == 1:
                    counter += 1
                    X.append(samples[i, :, int(curr_time_ptr[i])])
                    obs_idx.append(i)
                    curr_time_ptr[i] += 1
                    if curr_time_ptr[i] < nb_obs:
                        curr_times[i] = times[i, int(curr_time_ptr[i])]
                    else:
                        curr_times[i] = np.infty
            time_ptr.append(counter)
        # print("collate time: {}".format(time.time()-ti))

        assert len(obs_idx) == (nb_obs-1)*samples.shape[0]
        # predict_labels has values in {0,1,2} (which represent {-1,0,1})
        res = {'times': np.array(all_times), 'time_ptr': np.array(time_ptr),
               'obs_idx': torch.tensor(obs_idx, dtype=torch.long),
               'start_X': start_X,
               'n_obs_ot': torch.ones(size=(start_X.shape[0],))*nb_obs,
               'X': torch.tensor(np.array(X), dtype=torch.float32),
               'true_samples': samples, 'true_times': times,
               'true_eval_samples': eval_samples, 'true_eval_times': eval_times,
               'true_eval_labels': eval_labels,
               'predict_times': predict_times, 'predict_vals': predict_vals,
               'predict_labels': predict_labels,
               'max_time': max_time, 'coord_to_compare': [0],
               'true_mask': None, 'M': None, 'start_M': None}
        return res

    return custom_collate_fn


def LOBCollateFnGen2():
    """
    a function to get the costume collate function that can be used in
    torch.DataLoader with the wanted functions applied to the data as new
    dimensions
    -> the functions are applied on the fly to the dataset, and this additional
    data doesn't have to be saved

    :return: collate function
    """

    def custom_collate_fn(batch):
        amount_obs = batch[0]['amount_obs']
        eval_predict_steps = batch[0]['eval_predict_steps']
        max_pred_step = batch[0]['max_pred_step']
        samples = np.concatenate([b['samples'] for b in batch], axis=0)
        samples = np.transpose(samples[:, 1:, :], axes=(0, 2, 1))
        samples = np.expand_dims(samples, axis=1)
        times = np.concatenate([b['times'] for b in batch], axis=0)
        eval_samples = np.concatenate([b['eval_samples'] for b in batch],axis=0)
        eval_times = np.concatenate([b['eval_times'] for b in batch], axis=0)
        eval_labels = np.concatenate([b['eval_labels'] for b in batch], axis=0)

        predict_labels = torch.tensor(eval_labels+1, dtype=torch.long)

        res = {'samples': torch.tensor(samples),
               'true_labels': eval_labels,
               'labels': predict_labels,
               }
        return res

    return custom_collate_fn


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
    elif "LOB" in dataset_name:
        create_LOB_dataset(hyperparam_dict=dataset_params, seed=FLAGS.seed)
    else:
        create_dataset(
            stock_model_name=dataset_name, hyperparam_dict=dataset_params,
            seed=FLAGS.seed)


if __name__ == '__main__':
    app.run(main)



    pass
