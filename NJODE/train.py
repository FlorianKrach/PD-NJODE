"""
author: Florian Krach & Calypso Herrera

implementation of the training (and evaluation) of NJ-ODE
"""

# =====================================================================================================================
from typing import List

import torch  # machine learning
import torch.nn as nn
import tqdm  # process bar for iterations
import numpy as np  # large arrays and matrices, functions
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os, sys
import pandas as pd  # data analysis and manipulation
import json  # storing and exchanging data
import time
import socket
import matplotlib  # plots
import matplotlib.colors
from torch.backends import cudnn
import gc
import scipy.stats as stats
import warnings

from configs import config
import models
import data_utils
sys.path.append("../")
import baselines.GRU_ODE_Bayes.models_gru_ode_bayes as models_gru_ode_bayes

try:
    from telegram_notifications import send_bot_message as SBM
except Exception:
    from configs.config import SendBotMessage as SBM


# =====================================================================================================================
# check whether running on computer or server
if 'ada-' not in socket.gethostname():
    SERVER = False
    N_CPUS = 1
    SEND = False
else:
    SERVER = True
    N_CPUS = 1
    SEND = True
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
print(socket.gethostname())
print('SERVER={}'.format(SERVER))


# ==============================================================================
# Global variables
CHAT_ID = config.CHAT_ID
ERROR_CHAT_ID = config.ERROR_CHAT_ID

data_path = config.data_path
saved_models_path = config.saved_models_path
flagfile = config.flagfile

METR_COLUMNS: List[str] = [
    'epoch', 'train_time', 'val_time', 'train_loss', 'val_loss',
    'optimal_val_loss', 'test_loss', 'optimal_test_loss', 'evaluation_mean_diff']
default_ode_nn = ((50, 'tanh'), (50, 'tanh'))
default_readout_nn = ((50, 'tanh'), (50, 'tanh'))
default_enc_nn = ((50, 'tanh'), (50, 'tanh'))

ANOMALY_DETECTION = False
N_DATASET_WORKERS = 0
USE_GPU = False

# =====================================================================================================================
# Functions
makedirs = config.makedirs

def update_metric_df_to_new_version(df, path):
    """
    update the metric file to the new version, using the updated column names
    """
    if 'val_loss' not in df.columns:
        df = df.rename(columns={
            'eval_time': 'val_time', 'eval_loss': 'val_loss',
            'optimal_eval_loss': 'optimal_val_loss'})
        df.to_csv(path)
    return df


def train(
        anomaly_detection=None, n_dataset_workers=None, use_gpu=None,
        nb_cpus=None, send=None, gpu_num=0,
        model_id=None, epochs=100, batch_size=100, save_every=1,
        learning_rate=0.001, test_size=0.2, seed=398,
        hidden_size=10, bias=True, dropout_rate=0.1,
        ode_nn=default_ode_nn, readout_nn=default_readout_nn,
        enc_nn=default_enc_nn, use_rnn=False,
        solver="euler", weight=0.5, weight_decay=1.,
        dataset='BlackScholes', dataset_id=None, data_dict=None,
        plot=True, paths_to_plot=(0,),
        saved_models_path=saved_models_path,
        DEBUG=0,
        **options
):


    """
    training function for NJODE model (models.NJODE),
    the model is automatically saved in the model-save-path with the given
    model id, also all evaluations of the model are saved there

    :param anomaly_detection: used to pass on FLAG from parallel_train
    :param n_dataset_workers: used to pass on FLAG from parallel_train
    :param use_gpu: used to pass on FLAG from parallel_train
    :param nb_cpus: used to pass on FLAG from parallel_train
    :param send: used to pass on FLAG from parallel_train
    :param model_id: None or int, the id to save (or load if it already exists)
            the model, if None: next biggest unused id will be used
    :param epochs: int, number of epochs to train, each epoch is one cycle
            through all (random) batches of the training data
    :param batch_size: int
    :param save_every: int, defined number of epochs after each of which the
            model is saved and plotted if wanted. whenever the model has a new
            best eval-loss it is also saved, independent of this number (but not
            plotted)
    :param learning_rate: float
    :param test_size: float in (0,1), the percentage of samples to use for the
            test set (here there exists only a test set, since there is no extra
            evaluation)
    :param seed: int, seed for the random splitting of the dataset into train
            and test
    :param hidden_size: see models.NJODE
    :param bias: see models.NJODE
    :param dropout_rate: float
    :param ode_nn: see models.NJODE
    :param readout_nn: see models.NJODE
    :param enc_nn: see models.NJODE
    :param use_rnn: see models.NJODE
    :param solver: see models.NJODE
    :param weight: see models.NJODE
    :param weight_decay: see models.NJODE
    :param dataset: str, which dataset to use, supported: {'BlackScholes',
            'Heston', 'OrnsteinUhlenbeck'}. The corresponding dataset already
            needs to exist (create it first using data_utils.create_dataset)
    :param dataset_id: int or None, the id of the dataset to be used, if None,
            the latest generated dataset of the given name will be used
    :param data_dict: None, str or dict, if not None, the inputs dataset and
            dataset_id are overwritten. if str, the dict from config.py with the
            given name is loaded.
            from the dataset_overview.csv file, the
            dataset with a matching description is loaded.
    :param plot: bool, whether to plot
    :param paths_to_plot: list of ints, which paths of the test-set should be
            plotted
    :param saved_models_path: str, where to save the models
    :param DEBUG: int, if >0, then the model is in debug mode
    :param options: kwargs, used keywords:
        'test_data_dict'    None, str or dict, if no None, this data_dict is
                        used to define the dataset for plot_only and
                        evaluation (if evaluate=True)
        'func_appl_X'   list of functions (as str, see data_utils)
                        to apply to X
        'masked'        bool, whether the data is masked (i.e. has
                        incomplete observations)
        'save_extras'   bool, dict of options for saving the plots
        'plot_variance' bool, whether to plot also variance
        'std_factor'    float, the factor by which the std is multiplied
                        when plotting the variance
        'parallel'      bool, used by parallel_train.parallel_training
        'resume_training'   bool, used by parallel_train.parallel_training
        'plot_only'     bool, whether the model is used only to plot after
                        initiating or loading (i.e. no training) and exit
                        afterwards (used by demo)
        'ylabels'       list of str, see plot_one_path_with_pred()
        'legendlabels'  list of str, see plot_one_path_with_pred()
        'plot_same_yaxis'   bool, whether to plot the same range on y axis
                        for all dimensions
        'plot_obs_prob' bool, whether to plot the observation probability
        'plot_true_var' bool, whether to plot the true variance, if available
                        default: True
        'which_loss'    default: 'standard', see models.LOSS_FUN_DICT for
                        choices. suggested: 'easy' or 'very_easy'
        'loss_quantiles'    None or np.array, if loss is 'quantile', then
                        this is the array of quantiles to use
        'residual_enc_dec'  bool, whether resNNs are used for encoder and
                        readout NN, used by models.NJODE. the provided value
                        is overwritten by 'residual_enc' & 'residual_dec' if
                        they are provided. default: False
        'residual_enc'  bool, whether resNNs are used for encoder NN,
                        used by models.NJODE.
                        default: True if use_rnn=False, else: False (this is
                        for backward compatibility)
        'residual_dec'  bool, whether resNNs are used for readout NN,
                        used by models.NJODE. default: True
        'use_y_for_ode' bool, whether to use y (after jump) or x_impute for
                        the ODE as input, only in masked case, default: True
        'use_current_y_for_ode' bool, whether to use the current y as input
                        to the ode. this should make the training more
                        stable in the case of long windows without
                        observations (cf. literature about stable output
                        feedback). default: False
        'coord_wise_tau'    bool, whether to use a coordinate wise tau
        'input_sig'     bool, whether to use the signature as input
        'level'         int, level of the signature that is used
        'input_current_t'   bool, whether to additionally input current time
                        to the ODE function f, default: False
        'enc_input_t'   bool, whether to use the time as input for the
                        encoder network. default: False
        'train_readout_only'    bool, whether to only train the readout
                        network
        'training_size' int, if given and smaller than
                        dataset_size*(1-test_size), then this is the umber
                        of samples used for the training set (randomly
                        selected out of original training set)
        'evaluate'      bool, whether to evaluate the model in the test set
                        (i.e. not only compute the val_loss, but also
                        compute the mean difference between the true and the
                        predicted paths comparing at each time point)
        'load_best'     bool, whether to load the best checkpoint instead of
                        the last checkpoint when loading the model. Mainly
                        used for evaluating model at the best checkpoint.
        'gradient_clip' float, if provided, then gradient values are clipped
                        by the given value
        'clamp'         float, if provided, then output of model is clamped
                        to +/- the given value
        'other_model'   one of {'GRU_ODE_Bayes', randomizedNJODE};
                        the specifieed model is trained instead of the
                        controlled ODE-RNN model.
                        Other options/inputs might change or lose their
                        effect.
        'use_observation_as_input'  bool, whether to use the observations as
                        input to the model or whether to only use them for
                        the loss function (this can be used to make the
                        model learn to predict well far into the future).
                        can be a float in (0,1) to use an observation with
                        this probability as input. can also be a string
                        defining a function (when evaluated) that takes the
                        current epoch as input and returns a bool whether to
                        use the current observation as input (this can be a
                        random function, i.e. the output can depend on
                        sampling a random variable). default: true
        'val_use_observation_as_input'  bool, None, float or str, same as
                        'use_observation_as_input', but for the validation
                        set. default: None, i.e. same as for training set
        'ode_input_scaling_func'    None or str in {'id', 'tanh'}, the
                        function used to scale inputs to the neuralODE.
                        default: tanh
        'use_cond_exp'  bool, whether to use the conditional expectation
                        as reference for model evaluation, default: True
        'eval_use_true_paths'   bool, whether to use the true paths for
                        evaluation (instead of the conditional expectation)
                        default: False
        'which_val_loss'   str, see models.LOSS_FUN_DICT for choices, which
                        loss to use for evaluation, default: 'very_easy'
        'input_coords'  list of int or None, which coordinates to use as
                        input. overwrites the setting from dataset_metadata.
                        if None, then all coordinates are used.
        'output_coords' list of int or None, which coordinates to use as
                        output. overwrites the setting from
                        dataset_metadata. if None, then all coordinates are
                        used.
        'signature_coords'  list of int or None, which coordinates to use as
                        signature coordinates. overwrites the setting from
                        dataset_metadata. if None, then all input
                        coordinates are used.
        'compute_variance'   None, bool or str, if None, then no variance
                        computation is done. if bool, then the (marginal)
                        variance is computed. if str "covariance", then the
                        covariance matrix is computed. ATTENTION: the model
                        output corresponds to the square root of the
                        variance (or the Cholesky decomposition of the
                        covariance matrix, respectivel), so if W is the
                        model's output corresponding to the variance, then
                        the models variance estimate is V=W^T*W or W^2,
                        depending whether the covariance or marginal
                        variance is estimated.
                        default: None
        'var_weight'    float, weight of the variance loss term in the loss
                        function, default: 1
        'input_var_t_helper'    bool, whether to use 1/sqrt(Delta_t) as
                        additional input to the ODE function f. this should help
                        to better learn the variance of the process.
                        default: False
        'which_var_loss'   None or int, which loss to use for the variance loss
                        term. default: None, which leads to using default choice
                        of the main loss function (which aligns with structure
                        of main loss function as far as reasonable). see
                        models.LOSS_FUN_DICT for choices (currently in {1,2,3}).
        'plot_only_evaluate' bool, whether to evaluate the model when in
                        plot_only mode
        'plot_error_dist'   None or dict, if not None, then the kwargs for
                        the plot_error_distribution function

            -> 'GRU_ODE_Bayes' has the following extra options with the
                names 'GRU_ODE_Bayes'+<option_name>, for the following list
                of possible choices for <options_name>:
                '-mixing'   float, default: 0.0001, weight of the 2nd loss
                            term of GRU-ODE-Bayes
                '-solver'   one of {"euler", "midpoint", "dopri5"}, default:
                            "euler"
                '-impute'   bool, default: False,
                            whether to impute the last parameter
                            estimation of the p_model for the next ode_step
                            as input. the p_model maps (like the
                            readout_map) the hidden state to the
                            parameter estimation of the normal distribution.
                '-logvar'   bool, default: True, wether to use logarithmic
                            (co)variace -> hardcodinng positivity constraint
                '-full_gru_ode'     bool, default: True,
                                    whether to use the full GRU cell
                                    or a smaller version, see GRU-ODE-Bayes
                '-p_hidden'         int, default: hidden_size, size of the
                                    inner hidden layer of the p_model
                '-prep_hidden'      int, default: hidden_size, in the
                                    observational cell (i.e. jumps) a prior
                                    matrix multiplication transforms the
                                    input to have the size
                                    prep_hidden * input_size
                '-cov_hidden'       int, default: hidden_size, size of the
                                    inner hidden layer of the covariate_map.
                                    the covariate_map is used as a mapping
                                    to get the initial h (for controlled
                                    ODE-RNN this is done by the encoder)
            -> 'randomizedNJODE' has same options as NJODE
    """

    global ANOMALY_DETECTION, USE_GPU, SEND, N_CPUS, N_DATASET_WORKERS
    if anomaly_detection is not None:
        ANOMALY_DETECTION = anomaly_detection
    if use_gpu is not None:
        USE_GPU = use_gpu
    if send is not None:
        SEND = send
    if nb_cpus is not None:
        N_CPUS = nb_cpus
    if n_dataset_workers is not None:
        N_DATASET_WORKERS = n_dataset_workers

    initial_print = "model-id: {}\n".format(model_id)

    use_cond_exp = True
    if 'use_cond_exp' in options:
        use_cond_exp = options['use_cond_exp']
    eval_use_true_paths = False
    if 'eval_use_true_paths' in options:
        eval_use_true_paths = options['eval_use_true_paths']

    masked = False
    if 'masked' in options and 'other_model' not in options:
        masked = options['masked']

    if ANOMALY_DETECTION:
        # allow backward pass to print the traceback of the forward operation
        #   if it fails, "nan" backward computation produces error
        torch.autograd.set_detect_anomaly(True)
        torch.manual_seed(0)
        np.random.seed(0)
        # set seed and deterministic to make reproducible
        cudnn.deterministic = True

    # set number of CPUs
    torch.set_num_threads(N_CPUS)

    # get the device for torch
    if USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(gpu_num))
        torch.cuda.set_device(gpu_num)
        initial_print += '\nusing GPU'
    else:
        device = torch.device("cpu")
        initial_print += '\nusing CPU'

    # load dataset-metadata
    if data_dict is not None:
        dataset, dataset_id = data_utils._get_dataset_name_id_from_dict(
            data_dict=data_dict)
        dataset_id = int(dataset_id)
    else:
        if dataset is None:
            dataset = data_utils._get_datasetname(time_id=dataset_id)
        dataset_id = int(data_utils._get_time_id(stock_model_name=dataset,
                                                 time_id=dataset_id))
    dataset_metadata = data_utils.load_metadata(stock_model_name=dataset,
                                                time_id=dataset_id)

    # get input and output coordinates of the dataset
    input_coords = None
    output_coords = None
    signature_coords = None
    if "input_coords" in options:
        input_coords = options["input_coords"]
    elif "input_coords" in dataset_metadata:
        input_coords = dataset_metadata["input_coords"]
    if "output_coords" in options:
        output_coords = options["output_coords"]
    elif "output_coords" in dataset_metadata:
        output_coords = dataset_metadata["output_coords"]
    if "signature_coords" in options:
        signature_coords = options["signature_coords"]
    elif "signature_coords" in dataset_metadata:
        signature_coords = dataset_metadata["signature_coords"]
    if input_coords is None:
        input_size = dataset_metadata['dimension']
        input_coords = np.arange(input_size)
    else:
        input_size = len(input_coords)
    if output_coords is None:
        output_size = dataset_metadata['dimension']
        output_coords = np.arange(output_size)
    else:
        output_size = len(output_coords)
    if signature_coords is None:
        signature_coords = input_coords
    loss_quantiles = None
    if 'loss_quantiles' in options:
        loss_quantiles = options['loss_quantiles']

    initial_print += '\ninput_coords: {}\noutput_coords: {}'.format(
        input_coords, output_coords)
    initial_print += '\ninput_size: {}\noutput_size: {}'.format(
        input_size, output_size)
    initial_print += '\nsignature_coords: {}'.format(signature_coords)
    dimension = dataset_metadata['dimension']
    T = dataset_metadata['maturity']
    delta_t = dataset_metadata['dt']  # copy metadata
    original_output_dim = output_size
    original_input_dim = input_size

    # load raw data
    train_idx, val_idx = train_test_split(
        np.arange(dataset_metadata["nb_paths"]), test_size=test_size,
        random_state=seed)
    # --> get subset of training samples if wanted
    if 'training_size' in options:
        train_set_size = options['training_size']
        if train_set_size < len(train_idx):
            train_idx = np.random.choice(
                train_idx, train_set_size, replace=False
            )
    data_train = data_utils.IrregularDataset(
        model_name=dataset, time_id=dataset_id, idx=train_idx)
    data_val = data_utils.IrregularDataset(
        model_name=dataset, time_id=dataset_id, idx=val_idx)
    test_data_dict = None
    if 'test_data_dict' in options:
        test_data_dict = options['test_data_dict']
    if test_data_dict is not None:
        test_ds, test_ds_id = data_utils._get_dataset_name_id_from_dict(
            data_dict=test_data_dict)
        test_ds_id = int(test_ds_id)
        data_test = data_utils.IrregularDataset(
            model_name=test_ds, time_id=test_ds_id, idx=None)

    # get functions to apply to the paths in X
    if 'func_appl_X' in options:  # list of functions to apply to the paths in X
        initial_print += '\napply functions to X'
        functions = options['func_appl_X']
        collate_fn, mult = data_utils.CustomCollateFnGen(functions)
        input_size = input_size * mult
        output_size = output_size * mult
        input_coords = np.concatenate(
            [np.array(input_coords)+dimension*i for i in range(mult)])
        output_coords = np.concatenate(
            [np.array(output_coords)+dimension*i for i in range(mult)])
        initial_print += '\nnew input_coords: {}'.format(input_coords)
        initial_print += '\nnew output_coords: {}'.format(output_coords)
    else:
        functions = None
        collate_fn, mult = data_utils.CustomCollateFnGen(None)
        mult = 1

    # get variance or covariance coordinates if wanted
    compute_variance = None
    var_size = 0
    if 'compute_variance' in options:
        if functions is not None:
            warnings.warn(
                "function application to X and concurrent variance/covariance "
                "computation might lead to problems! Use carefully!",
                UserWarning)
        compute_variance = options['compute_variance']
        if compute_variance == 'covariance':
            var_size = output_size**2
            initial_print += '\ncompute covariance of size {}'.format(var_size)
        elif compute_variance not in [None, False]:
            compute_variance = 'variance'
            var_size = output_size
            initial_print += '\ncompute (marginal) variance of size {}'.format(
                var_size)
        else:
            compute_variance = None
            initial_print += '\nno variance computation'
            var_size = 0
        # the models variance output is the Cholesky decomposition of the
        #   covariance matrix or the square root of the marginal variance.
        # for Y being the entire model output, the variance output is
        #   W=Y[:,-var_size:]
        output_size += var_size

    # get data-loader for training
    dl = DataLoader(  # class to iterate over training data
        dataset=data_train, collate_fn=collate_fn,
        shuffle=True, batch_size=batch_size, num_workers=N_DATASET_WORKERS)
    dl_val = DataLoader(  # class to iterate over validation data
        dataset=data_val, collate_fn=collate_fn,
        shuffle=False, batch_size=len(data_val), num_workers=N_DATASET_WORKERS)
    stockmodel = data_utils._STOCK_MODELS[
        dataset_metadata['model_name']](**dataset_metadata)
    if test_data_dict is not None:
        dl_test = DataLoader(  # class to iterate over test data
            dataset=data_test, collate_fn=collate_fn,
            shuffle=False, batch_size=len(data_test),
            num_workers=N_DATASET_WORKERS)
        testset_metadata = data_utils.load_metadata(
            stock_model_name=test_ds, time_id=test_ds_id)
        stockmodel_test = data_utils._STOCK_MODELS[
            testset_metadata['model_name']](**testset_metadata)
    else:
        dl_test = dl_val
        stockmodel_test = stockmodel
        testset_metadata = dataset_metadata
    if loss_quantiles is not None:
        stockmodel.set_quantiles(loss_quantiles)
        stockmodel_test.set_quantiles(loss_quantiles)

    # get additional plotting information
    plot_variance = False
    std_factor = 1  # factor with which the std is multiplied
    if (functions is not None and mult > 1) or (compute_variance is not None):
        if 'plot_variance' in options:
            plot_variance = options['plot_variance']
        if 'std_factor' in options:
            std_factor = options['std_factor']
    plot_true_var = True
    if 'plot_true_var' in options:
        plot_true_var = options['plot_true_var']
    ylabels = None
    if 'ylabels' in options:
        ylabels = options['ylabels']
    legendlabels = None
    if 'legendlabels' in options:
        legendlabels = options['legendlabels']
    plot_same_yaxis = False
    if 'plot_same_yaxis' in options:
        plot_same_yaxis = options['plot_same_yaxis']
    plot_obs_prob = False
    if 'plot_obs_prob' in options:
        plot_obs_prob = options["plot_obs_prob"]

    # validation loss function
    which_val_loss = 'very_easy'
    if 'which_loss' in options:
        which_val_loss = options['which_loss']
    if 'which_val_loss' in options:
        which_val_loss = options['which_val_loss']
    assert which_val_loss in models.LOSS_FUN_DICT

    # get optimal eval loss
    #   -> if other functions are applied to X, then only the original X is used
    #      in the computation of the optimal eval loss, except for the case of
    #      functions=["power-2"] and the model has implemented the loss
    #      computation for the power-2 case
    print(dataset_metadata)
    plot_only = False
    if 'plot_only' in options:
        plot_only = options['plot_only']
    if use_cond_exp and not plot_only:
        if compute_variance is not None:
            warnings.warn(
                "optimal loss might be wrong, since the conditional "
                "variance is also learned, which is not accounted for in "
                "computation of the optimal loss",
                UserWarning)
        store_cond_exp = True
        if dl_val != dl_test:
            store_cond_exp = False
        if (functions is not None and functions == ["power-2"] and
                stockmodel.loss_comp_for_pow2_implemented):
            corrected_string = "(for X and X^2) "
            opt_val_loss = compute_optimal_val_loss(
                dl_val, stockmodel, delta_t, T, mult=1,
                store_cond_exp=store_cond_exp, return_var=True,
                which_loss=which_val_loss)
        else:
            if functions is not None and len(functions) > 0:
                initial_print += '\nWARNING: optimal loss computation for ' \
                                 'power=2 not implemented for this model'
                corrected_string = "(corrected: only original X used) "
            else:
                corrected_string = ""
            opt_val_loss = compute_optimal_val_loss(
                dl_val, stockmodel, delta_t, T, mult=mult,
                store_cond_exp=store_cond_exp, return_var=False,
                which_loss=which_val_loss)
        initial_print += '\noptimal {}val-loss (achieved by true cond exp): ' \
                     '{:.5f}'.format(corrected_string, opt_val_loss)
    else:
        opt_val_loss = np.nan

    # get params_dict
    params_dict = {  # create a dictionary of the wanted parameters
        'input_size': input_size, 'epochs': epochs,
        'hidden_size': hidden_size, 'output_size': output_size, 'bias': bias,
        'ode_nn': ode_nn, 'readout_nn': readout_nn, 'enc_nn': enc_nn,
        'use_rnn': use_rnn,
        'dropout_rate': dropout_rate, 'batch_size': batch_size,
        'solver': solver, 'dataset': dataset, 'dataset_id': dataset_id,
        'data_dict': data_dict,
        'learning_rate': learning_rate, 'test_size': test_size, 'seed': seed,
        'weight': weight, 'weight_decay': weight_decay,
        'optimal_val_loss': opt_val_loss, 'options': options}
    desc = json.dumps(params_dict, sort_keys=True)

    # add additional values to params_dict (not to be shown in the description)
    params_dict['input_coords'] = input_coords
    params_dict['output_coords'] = output_coords
    params_dict['signature_coords'] = signature_coords
    params_dict['compute_variance'] = compute_variance
    params_dict['var_size'] = var_size

    # get overview file
    resume_training = False
    if ('parallel' in options and options['parallel'] is False) or \
            ('parallel' not in options):
        model_overview_file_name = '{}model_overview.csv'.format(
            saved_models_path
        )
        makedirs(saved_models_path)
        if not os.path.exists(model_overview_file_name):
            df_overview = pd.DataFrame(data=None, columns=['id', 'description'])
            max_id = 0
        else:
            df_overview = pd.read_csv(model_overview_file_name, index_col=0)  # read model overview csv file
            max_id = np.max(df_overview['id'].values)

        # get model_id, model params etc.
        if model_id is None:
            model_id = max_id + 1
        if model_id not in df_overview['id'].values:  # create new model ID
            initial_print += '\nnew model_id={}'.format(model_id)
            df_ov_app = pd.DataFrame([[model_id, desc]],
                                     columns=['id', 'description'])
            df_overview = pd.concat([df_overview, df_ov_app], ignore_index=True)
            df_overview.to_csv(model_overview_file_name)
        else:
            initial_print += '\nmodel_id already exists -> resume training'  # resume training if model already exists
            resume_training = True
            desc = (df_overview['description'].loc[
                df_overview['id'] == model_id]).values[0]
            params_dict = json.loads(desc)
    initial_print += '\nmodel params:\n{}'.format(desc)
    if 'resume_training' in options and options['resume_training'] is True:
        resume_training = True

    # get all needed paths
    model_path = '{}id-{}/'.format(saved_models_path, model_id)
    makedirs(model_path)
    model_path_save_last = '{}last_checkpoint/'.format(model_path)
    model_path_save_best = '{}best_checkpoint/'.format(model_path)
    makedirs(model_path_save_last)
    makedirs(model_path_save_best)
    model_metric_file = '{}metric_id-{}.csv'.format(model_path, model_id)
    plot_save_path = '{}plots/'.format(model_path)
    if 'save_extras' in options:
        save_extras = options['save_extras']
    else:
        save_extras = {}

    # get the model & optimizer
    if 'other_model' not in options:  # take NJODE model if not specified otherwise
        model = models.NJODE(**params_dict)  # get NJODE model class from
        model_name = 'NJODE'
    elif options['other_model'] == "randomizedNJODE":
        model_name = 'randomizedNJODE'
        epochs = 1
        model = models.randomizedNJODE(**params_dict)
    elif options['other_model'] == "NJmodel":
        model_name = 'NJmodel'
        params_dict["hidden_size"] = output_size
        model = models.NJmodel(**params_dict)
    elif options['other_model'] == "GRU_ODE_Bayes":  # see train documentation
        model_name = 'GRU-ODE-Bayes'
        # get parameters for GRU-ODE-Bayes model
        hidden_size = params_dict['hidden_size']
        mixing = 0.0001
        if 'GRU_ODE_Bayes-mixing' in options:
            mixing = options['GRU_ODE_Bayes-mixing']
        solver = 'euler'
        if 'GRU_ODE_Bayes-solver' in options:
            solver = options['GRU_ODE_Bayes-solver']
        impute = False
        if 'GRU_ODE_Bayes-impute' in options:
            impute = options['GRU_ODE_Bayes-impute']
        logvar = True
        if 'GRU_ODE_Bayes-logvar' in options:
            logvar = options['GRU_ODE_Bayes-logvar']
        full_gru_ode = True
        if 'GRU_ODE_Bayes-full_gru_ode' in options:
            full_gru_ode = options['GRU_ODE_Bayes-full_gru_ode']
        p_hidden = hidden_size
        if 'GRU_ODE_Bayes-p_hidden' in options:
            p_hidden = options['GRU_ODE_Bayes-p_hidden']
        prep_hidden = hidden_size
        if 'GRU_ODE_Bayes-prep_hidden' in options:
            prep_hidden = options['GRU_ODE_Bayes-prep_hidden']
        cov_hidden = hidden_size
        if 'GRU_ODE_Bayes-cov_hidden' in options:
            cov_hidden = options['GRU_ODE_Bayes-cov_hidden']

        model = models_gru_ode_bayes.NNFOwithBayesianJumps(  # import GRU ODE model
            input_size=params_dict['input_size'],
            hidden_size=params_dict['hidden_size'],
            p_hidden=p_hidden, prep_hidden=prep_hidden,
            bias=params_dict['bias'],
            cov_size=params_dict['input_size'], cov_hidden=cov_hidden,
            logvar=logvar, mixing=mixing,
            dropout_rate=params_dict['dropout_rate'],
            full_gru_ode=full_gru_ode, solver=solver, impute=impute,
        )
    else:
        raise ValueError("Invalid argument for (option) parameter 'other_model'."
                         "Please check docstring for correct use.")
    train_readout_only = False
    if 'train_readout_only' in options:
        train_readout_only = options['train_readout_only']
    model.to(device)  # pass model to CPU/GPU
    if not train_readout_only:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=0.0005)
    else:
        optimizer = torch.optim.Adam(
            model.readout_map.parameters(), lr=learning_rate,
            weight_decay=0.0005)
    gradient_clip = None
    if 'gradient_clip' in options:
        gradient_clip = options["gradient_clip"]

    # load saved model if wanted/possible
    best_val_loss = np.infty
    metr_columns = METR_COLUMNS
    if resume_training:
        initial_print += '\nload saved model ...'
        try:
            if 'load_best' in options and options['load_best']:
                models.get_ckpt_model(model_path_save_best, model, optimizer,
                                      device)
            else:
                models.get_ckpt_model(model_path_save_last, model, optimizer,
                                      device)
            df_metric = pd.read_csv(model_metric_file, index_col=0)
            df_metric = update_metric_df_to_new_version(
                df_metric, model_metric_file)
            best_val_loss = np.min(df_metric['val_loss'].values)
            model.epoch += 1
            model.weight_decay_step()
            initial_print += '\nepoch: {}, weight: {}'.format(
                model.epoch, model.weight)
        except Exception as e:
            initial_print += '\nloading model failed -> initiate new model'
            initial_print += '\nException:\n{}'.format(e)
            resume_training = False
    if not resume_training:
        initial_print += '\ninitiate new model ...'
        df_metric = pd.DataFrame(columns=metr_columns)

    # ---------- plot only option ------------
    if plot_only:
        print("!! plot only mode !!")
        batch = next(iter(dl_test))
        model.epoch -= 1
        initial_print += '\nplotting ...'
        plot_filename = 'demo-plot_epoch-{}'.format(model.epoch)
        plot_filename = plot_filename + '_path-{}.pdf'
        plot_error_dist = None
        if "plot_error_dist" in options:
            plot_error_dist = options["plot_error_dist"]
        ref_model_to_use = None
        if "ref_model_to_use" in options:
            ref_model_to_use = options["ref_model_to_use"]
        curr_opt_loss, c_model_test_loss, ed_paths = plot_one_path_with_pred(
            device, model, batch, stockmodel_test,
            testset_metadata['dt'], testset_metadata['maturity'],
            path_to_plot=paths_to_plot, save_path=plot_save_path,
            filename=plot_filename, plot_variance=plot_variance,
            plot_true_var=plot_true_var,
            functions=functions, std_factor=std_factor,
            use_cond_exp=use_cond_exp, output_coords=output_coords,
            model_name=model_name, save_extras=save_extras, ylabels=ylabels,
            legendlabels=legendlabels, input_coords=input_coords,
            same_yaxis=plot_same_yaxis, plot_obs_prob=plot_obs_prob,
            dataset_metadata=testset_metadata, reuse_cond_exp=True,
            loss_quantiles=loss_quantiles, which_loss=which_val_loss,
            plot_error_dist=plot_error_dist, ref_model_to_use=ref_model_to_use)
        eval_msd = None
        if "plot_only_evaluate" in options and options["plot_only_evaluate"]:
            print("evaluate model ...")
            eval_msd = evaluate_model(
                model=model, dl_test=dl_test, device=device,
                stockmodel_test=stockmodel_test,
                testset_metadata=testset_metadata,
                mult=mult, use_cond_exp=use_cond_exp,
                eval_use_true_paths=eval_use_true_paths)
        if SEND:
            files_to_send = []
            caption = "{} - id={}".format(model_name, model_id)
            for i in paths_to_plot:
                files_to_send.append(
                    os.path.join(plot_save_path, plot_filename.format(i)))
            if ed_paths is not None:
                files_to_send += ed_paths
            SBM.send_notification(
                text='finished plot-only: {}, id={}\n'
                     'optimal test loss: {}\n'
                     'current test loss: {}\n'
                     'evaluation metric (test set): {}, epoch: {}\n\n{}'.format(
                    model_name, model_id, curr_opt_loss,
                    c_model_test_loss, eval_msd, model.epoch, desc),
                chat_id=config.CHAT_ID,
                files=files_to_send,
                text_for_files=caption
            )
        initial_print += '\noptimal test-loss (with current weight={:.5f}): ' \
                         '{:.5f}'.format(model.weight, curr_opt_loss)
        initial_print += '\nmodel test-loss (with current weight={:.5f}): ' \
                         '{:.5f}'.format(model.weight, c_model_test_loss)
        if eval_msd is not None:
            initial_print += '\nevaluation metric (test set): {:.3e}'.format(
                eval_msd)
        print(initial_print)
        return 0

    # ---------------- TRAINING ----------------
    skip_training = True
    if model.epoch <= epochs:  # check if it already trained the requested number of epochs
        skip_training = False

        # send notification
        if SEND:
            SBM.send_notification(
                text='start training - model id={}'.format(model_id),
                chat_id=config.CHAT_ID)
        initial_print += '\n\nmodel overview:'
        print(initial_print)
        print(model, '\n')

        # compute number of parameters
        nr_params = 0
        for name, param in model.named_parameters():
            skip = False
            for p_name in ['gru_debug', 'classification_model']:
                if p_name in name:
                    skip = True
            if not skip:
                nr_params += param.nelement()  # count number of parameters
        print('# parameters={}\n'.format(nr_params))

        # compute number of trainable params
        nr_trainable_params = 0
        for pg in optimizer.param_groups:
            for p in pg['params']:
                nr_trainable_params += p.nelement()
        print('# trainable parameters={}\n'.format(nr_trainable_params))
        print('start training ...')

    metric_app = []
    while model.epoch <= epochs:
        t = time.time()  # return the time in seconds since the epoch
        model.train()  # set model in train mode (e.g. BatchNorm)
        if 'other_model' in options and \
                options['other_model'] == "randomizedNJODE":
            linreg_X = []
            linreg_y = []
        for i, b in tqdm.tqdm(enumerate(dl)):  # iterate over the dataloader
            optimizer.zero_grad()  # reset the gradient
            times = b["times"]  # Produce instance of byte type instead of str type
            time_ptr = b["time_ptr"]  # pointer
            X = b["X"].to(device)
            M = b["M"]
            if M is not None:
                M = M.to(device)
            start_M = b["start_M"]
            if start_M is not None:
                start_M = start_M.to(device)

            start_X = b["start_X"].to(device)
            obs_idx = b["obs_idx"]
            n_obs_ot = b["n_obs_ot"].to(device)

            if 'other_model' not in options or model_name == "NJmodel":
                hT, loss = model(
                    times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx,
                    delta_t=delta_t, T=T, start_X=start_X, n_obs_ot=n_obs_ot,
                    return_path=False, get_loss=True, M=M, start_M=start_M,)
            elif options['other_model'] == "randomizedNJODE":
                linreg_X_, linreg_y_ = model.get_Xy_reg(
                    times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx,
                    delta_t=delta_t, T=T, start_X=start_X, n_obs_ot=n_obs_ot,
                    return_path=False, M=M, start_M=start_M)
                linreg_X += linreg_X_
                linreg_y += linreg_y_
                loss = torch.tensor(0.)
                continue
            elif options['other_model'] == "GRU_ODE_Bayes":
                if M is None:
                    M = torch.ones_like(X)
                hT, loss, _, _ = model(
                    times, time_ptr, X, M, obs_idx, delta_t, T, start_X,
                    return_path=False, smoother=False)
            else:
                raise ValueError
            loss.backward()  # compute gradient of each weight regarding loss function
            if gradient_clip is not None:
                nn.utils.clip_grad_value_(
                    model.parameters(), clip_value=gradient_clip)
            optimizer.step()  # update weights by ADAM optimizer
            if ANOMALY_DETECTION:
                print(r"current loss: {}".format(loss.detach().cpu().numpy()))
            if DEBUG:
                print("DEBUG MODE: stop training after first batch")
                break
        if 'other_model' in options and \
                options['other_model'] == "randomizedNJODE":
            linreg_X = np.stack(linreg_X, axis=0)
            linreg_y = np.stack(linreg_y, axis=0)
            print("OLS to fit readout-map ...")
            model.readout_map.fit(linreg_X, linreg_y)
        train_time = time.time() - t  # difference between current time and start time

        # -------- evaluation --------
        print("evaluating ...")
        t = time.time()
        batch = None
        with torch.no_grad():  # no gradient needed for evaluation
            loss_val = 0
            loss_val_corrected = 0
            num_obs = 0
            eval_msd = 0
            model.eval()  # set model in evaluation mode
            for i, b in enumerate(dl_val):  # iterate over dataloader for validation set
                times = b["times"]
                time_ptr = b["time_ptr"]
                X = b["X"].to(device)
                M = b["M"]
                if M is not None:
                    M = M.to(device)
                start_M = b["start_M"]
                if start_M is not None:
                    start_M = start_M.to(device)

                start_X = b["start_X"].to(device)
                obs_idx = b["obs_idx"]
                n_obs_ot = b["n_obs_ot"].to(device)
                true_paths = b["true_paths"]
                true_mask = b["true_mask"]

                if 'other_model' not in options or \
                        model_name in ("randomizedNJODE", "NJmodel"):
                    hT, c_loss = model(
                        times, time_ptr, X, obs_idx, delta_t, T, start_X,
                        n_obs_ot, return_path=False, get_loss=True, M=M,
                        start_M=start_M, which_loss=which_val_loss,)
                elif options['other_model'] == "GRU_ODE_Bayes":
                    if M is None:
                        M = torch.ones_like(X)
                    hT, c_loss, _, _ = model(
                        times, time_ptr, X, M, obs_idx, delta_t, T, start_X,
                        return_path=False, smoother=False,)
                else:
                    raise ValueError
                loss_val += c_loss.detach().cpu().numpy()
                num_obs += 1  # count number of observations

                # if functions are applied or var loss is computed, also compute
                #   the loss when only using the coordinates where function was
                #   not applied & without computing the variance loss
                #   -> this can be compared to the optimal-eval-loss
                if (mult is not None and mult > 1) or (compute_variance is not None):
                    if 'other_model' not in options:
                        hT_corrected, c_loss_corrected = model(
                            times, time_ptr, X, obs_idx, delta_t, T, start_X,
                            n_obs_ot, return_path=False, get_loss=True, M=M,
                            start_M=start_M, which_loss=which_val_loss,
                            dim_to=original_output_dim,
                            compute_variance_loss=False,)
                    loss_val_corrected += c_loss_corrected.detach().cpu().numpy()

            # mean squared difference evaluation
            if 'evaluate' in options and options['evaluate']:
                eval_msd = evaluate_model(
                    model=model, dl_test=dl_test, device=device,
                    stockmodel_test=stockmodel_test,
                    testset_metadata=testset_metadata,
                    mult=mult, use_cond_exp=use_cond_exp,
                    eval_use_true_paths=eval_use_true_paths)

            val_time = time.time() - t
            loss_val = loss_val / num_obs
            loss_val_corrected /= num_obs
            eval_msd = eval_msd / num_obs
            train_loss = loss.detach().cpu().numpy()
            print_str = "epoch {}, weight={:.5f}, train-loss={:.5f}, " \
                        "optimal-val-loss={:.5f}, val-loss={:.5f}, ".format(
                model.epoch, model.weight, train_loss, opt_val_loss, loss_val)
            if (mult is not None and mult > 1) or (compute_variance is not None):
                print_str += "\ncorrected (i.e. without additional dims of " \
                             "funct_appl_X)-val-loss={:.5f}, ".format(
                    loss_val_corrected)
            print(print_str)

        curr_metric = [model.epoch, train_time, val_time, train_loss,
                               loss_val, opt_val_loss, None, None]
        if 'evaluate' in options and options['evaluate']:
            curr_metric.append(eval_msd)
            print("evaluation mean square difference (test set): {:.5f}".format(
                eval_msd))
        else:
            curr_metric.append(None)
        metric_app.append(curr_metric)

        # save model
        if model.epoch % save_every == 0:
            if plot:
                batch = next(iter(dl_test))
                print('plotting ...')
                plot_filename = 'epoch-{}'.format(model.epoch)
                plot_filename = plot_filename + '_path-{}.pdf'
                curr_opt_test_loss, c_model_test_loss, _ = plot_one_path_with_pred(
                    device=device, model=model, batch=batch,
                    stockmodel=stockmodel, delta_t=delta_t, T=T,
                    path_to_plot=paths_to_plot, save_path=plot_save_path,
                    filename=plot_filename, plot_variance=plot_variance,
                    plot_true_var=plot_true_var,
                    functions=functions, std_factor=std_factor,
                    model_name=model_name, save_extras=save_extras,
                    ylabels=ylabels, use_cond_exp=use_cond_exp,
                    legendlabels=legendlabels, reuse_cond_exp=True,
                    output_coords=output_coords, input_coords=input_coords,
                    same_yaxis=plot_same_yaxis, plot_obs_prob=plot_obs_prob,
                    dataset_metadata=dataset_metadata,
                    loss_quantiles=loss_quantiles, which_loss=which_val_loss)
                print('optimal test-loss (with current weight={:.5f}): '
                      '{:.5f}'.format(model.weight, curr_opt_test_loss))
                print('model test-loss (with current weight={:.5f}): '
                      '{:.5f}'.format(model.weight, c_model_test_loss))
                curr_metric[-2] = curr_opt_test_loss
                curr_metric[-3] = c_model_test_loss
            print('save model ...')
            df_m_app = pd.DataFrame(data=metric_app, columns=metr_columns)
            df_metric = pd.concat([df_metric, df_m_app], ignore_index=True)
            df_metric.to_csv(model_metric_file)
            models.save_checkpoint(model, optimizer, model_path_save_last,
                                   model.epoch)
            metric_app = []
            print('saved!')
        if loss_val < best_val_loss:
            print('save new best model: last-best-loss: {:.5f}, '
                  'new-best-loss: {:.5f}, epoch: {}'.format(
                best_val_loss, loss_val, model.epoch))
            df_m_app = pd.DataFrame(data=metric_app, columns=metr_columns)
            df_metric = pd.concat([df_metric, df_m_app], ignore_index=True)
            df_metric.to_csv(model_metric_file)
            models.save_checkpoint(model, optimizer, model_path_save_last,
                                   model.epoch)
            models.save_checkpoint(model, optimizer, model_path_save_best,
                                   model.epoch)
            metric_app = []
            best_val_loss = loss_val
            print('saved!')
        print("-"*100)

        model.epoch += 1
        model.weight_decay_step()

    # send notification
    if SEND and not skip_training:
        files_to_send = [model_metric_file]
        caption = "{} - id={}".format(model_name, model_id)
        if plot:
            for i in paths_to_plot:
                files_to_send.append(
                    os.path.join(plot_save_path, plot_filename.format(i)))
        SBM.send_notification(
            text='finished training: {}, id={}\n\n{}'.format(
                model_name, model_id, desc),
            chat_id=config.CHAT_ID,
            files=files_to_send,
            text_for_files=caption)

    # delete model & free memory
    del model, dl, dl_val, data_train, data_val, dl_test
    gc.collect()

    return 0


def compute_optimal_val_loss(
        dl_val, stockmodel, delta_t, T, mult=None,
        store_cond_exp=False, return_var=False, which_loss='easy'):
    """
    compute optimal evaluation loss (with the true cond. exp.) on the
    test-dataset
    :param dl_val: torch.DataLoader, used for the validation dataset
    :param stockmodel: stock_model.StockModel instance
    :param delta_t: float, the time_delta
    :param T: float, the terminal time
    :param mult: None or int, the factor by which the dimension is multiplied
    :param store_cond_exp: bool, whether to store the conditional expectation
    :return: float (optimal loss)
    """
    opt_loss = 0
    num_obs = 0
    for i, b in enumerate(dl_val):
        times = b["times"]
        time_ptr = b["time_ptr"]
        X = b["X"].detach().cpu().numpy()
        start_X = b["start_X"].detach().cpu().numpy()
        obs_idx = b["obs_idx"].detach().cpu().numpy()
        n_obs_ot = b["n_obs_ot"].detach().cpu().numpy()
        M = b["M"]
        if M is not None:
            M = M.detach().cpu().numpy()
        num_obs += 1
        opt_loss += stockmodel.get_optimal_loss(
            times, time_ptr, X, obs_idx, delta_t, T, start_X, n_obs_ot, M=M,
            mult=mult, store_and_use_stored=store_cond_exp,
            return_var=return_var, which_loss=which_loss)
    return opt_loss / num_obs


def evaluate_model(
        model, dl_test, device, stockmodel_test, testset_metadata,
        mult, use_cond_exp, eval_use_true_paths):
    """
    evaluate the model on the test set

    Args:
        model:
        dl_test:
        device:
        stockmodel_test:
        testset_metadata:
        mult:
        use_cond_exp:
        eval_use_true_paths:

    Returns: evaluation metric

    """
    eval_msd = 0.
    for i, b in enumerate(dl_test):
        times = b["times"]
        time_ptr = b["time_ptr"]
        X = b["X"].to(device)
        M = b["M"]
        if M is not None:
            M = M.to(device)
        start_M = b["start_M"]
        if start_M is not None:
            start_M = start_M.to(device)
        start_X = b["start_X"].to(device)
        obs_idx = b["obs_idx"]
        n_obs_ot = b["n_obs_ot"].to(device)
        true_paths = b["true_paths"]
        true_mask = b["true_mask"]

        if use_cond_exp and not eval_use_true_paths:
            true_paths = None
            true_mask = None
        _eval_msd = model.evaluate(
            times=times, time_ptr=time_ptr, X=X,
            obs_idx=obs_idx,
            delta_t=testset_metadata["dt"],
            T=testset_metadata["maturity"],
            start_X=start_X, n_obs_ot=n_obs_ot,
            stockmodel=stockmodel_test, return_paths=False, M=M,
            start_M=start_M, true_paths=true_paths,
            true_mask=true_mask, mult=mult,
            use_stored_cond_exp=True, )
        eval_msd += _eval_msd

    return eval_msd

def compute_prediction_errors(
        model_pred, model_t, true_paths, true_times, output_coords,
        original_out_dim, eval_times):
    """
    compute the error distribution of the model predictions
    Args:
        model_pred: np.array, the model predictions,
            shape: (time_steps, bs, dim)
        model_t: np.array, the time points of the model predictions,
            shape: (time_steps,)
        true_paths: np.array, the true paths, shape: (bs, dim, time_steps)
        true_times: np.array, the time points of the true paths,
            shape: (time_steps,)
        output_coords: list of int, the coordinates corresponding to the model
            output in the extended X (after function applications)
        original_out_dim: int, the original dimension of the output (i.e.,
            without function applications)
        eval_times: list of float, the times where to evaluate the error

    Returns:

    """

    if len(output_coords) > original_out_dim:
        output_coords = output_coords[:original_out_dim]
    eval_times = sorted(eval_times)

    # get the model predictions at the evaluation times
    ind = np.searchsorted(model_t, eval_times, side='right') - 1
    model_pred_eval = model_pred[:, :, :original_out_dim][ind]

    # get the true paths at the evaluation times
    ind = np.searchsorted(true_times, eval_times, side='right') - 1
    true_paths = np.transpose(true_paths, (2, 0, 1))
    true_paths_eval = true_paths[:, :, output_coords][ind]

    # compute the errors
    error = model_pred_eval - true_paths_eval
    return error


def plot_error_distribution(
        true_paths, true_times, model_preds, model_ts, output_coords,
        original_out_dim, eval_times, colors, model_names=None, save_path='',
        filename='error_dist_plot_{}.pdf', coord_names=None):

    # replace special times ("mid", "end") by the actual times
    T = true_times[-1]
    _evl_times = []
    for t in eval_times:
        if t == "mid":
            _evl_times.append(T/2)
        elif t in ["end", "last"]:
            _evl_times.append(T)
        else:
            _evl_times.append(t)
    eval_times = _evl_times

    # compute the errors for each model, shape: (model, eval_times, bs, dim)
    errors = [compute_prediction_errors(
        model_pred, model_t, true_paths, true_times, output_coords,
        original_out_dim, eval_times) for model_pred, model_t in
              zip(model_preds, model_ts)]

    # get error statistics
    nperrors = np.array(errors)
    mean_errors = np.mean(nperrors, axis=2)
    std_errors = np.std(nperrors, axis=2)
    cols = ["model", "eval_time", "coord", "mean", "std"]
    dat = []
    for i, model_name in enumerate(model_names):
        for j, eval_time in enumerate(eval_times):
            for k, coord in enumerate(output_coords):
                dat.append([model_name, eval_time, coord, mean_errors[i, j, k],
                            std_errors[i, j, k]])
    df = pd.DataFrame(columns=cols, data=dat)
    err_stats_file = save_path + "error_stats.csv"
    df.to_csv(err_stats_file)
    paths = [err_stats_file]

    # plot the error distribution
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    for d in range(original_out_dim):
        plt.figure()
        boxplots = []
        nb_groups = len(errors)
        # loop over the models
        for i, error in enumerate(errors):
            boxplots.append(plt.boxplot(
                # make boxplot for each evaluation time
                [x[:,d] for x in error],
                positions=np.array(np.arange(len(eval_times)))*nb_groups+i*0.8,
                sym='', widths=0.6))

        for i, bp in enumerate(boxplots):
            set_box_color(bp, colors[i])
            plt.plot([], c=colors[i], label=model_names[i])
        plt.legend()

        plt.xticks(np.arange(len(eval_times))*nb_groups+(nb_groups-1)*0.4,
                   eval_times)
        # plt.xlim(-1, len(eval_times) * nb_groups + 1)
        cn = d
        if coord_names is not None:
            cn = coord_names[output_coords[d]]
        plt.xlabel('Evaluation Time')
        plt.ylabel('Prediction Error')
        plt.title('Error distribution for coordinate {}'.format(cn))
        plt.tight_layout()
        plt.savefig(filename.format(d))
        plt.close()
        paths.append(filename.format(d))

    return paths


def plot_one_path_with_pred(
        device, model, batch, stockmodel, delta_t, T,
        path_to_plot=(0,), save_path='', filename='plot_{}.pdf',
        plot_variance=False, plot_true_var=True,
        functions=None, std_factor=1,
        model_name=None, ylabels=None,
        legendlabels=None,
        save_extras={'bbox_inches': 'tight', 'pad_inches': 0.01},
        use_cond_exp=True, same_yaxis=False,
        plot_obs_prob=False, dataset_metadata=None,
        reuse_cond_exp=True, output_coords=None,
        loss_quantiles=None, input_coords=None,
        which_loss='easy',
        plot_error_dist=None, ref_model_to_use=None,
):
    """
    plot one path of the stockmodel together with optimal cond. exp. and its
    prediction by the model
    :param device: torch.device, to use for computations
    :param model: models.NJODE instance
    :param batch: the batch from where to take the paths
    :param stockmodel: stock_model.StockModel instance, used to compute true
            cond. exp.
    :param delta_t: float
    :param T: float
    :param path_to_plot: list of ints, which paths to plot (i.e. which elements
            oof the batch)
    :param save_path: str, the path where to save the plot
    :param filename: str, the filename for the plot, should have one insertion
            possibility to put the path number
    :param plot_variance: bool, whether to plot the variance, if supported by
            functions (i.e. square has to be applied)
    :param plot_true_var: bool, whether to plot the true variance
    :param functions: list of functions (as str), the functions applied to X
    :param std_factor: float, the factor by which std is multiplied
    :param model_name: str or None, name used for model in plots
    :param ylabels: None or list of str of same length as dimension of X
    :param legendlabels: None or list of str of length 4 or 5 (labels of
        i) true path, ii) our model, iii) true cond. exp., iv) observed values,
        v) true values at observation times (only if noisy observations are
        used))
    :param save_extras: dict with extra options for saving plot
    :param use_cond_exp: bool, whether to plot the conditional expectation
    :param same_yaxis: bool, whether to plot all coordinates with same range on
        y-axis
    :param plot_obs_prob: bool, whether to plot the probability of an
        observation for all times
    :param dataset_metadata: needed if plot_obs_prob=true, the metadata of the
        used dataset to extract the observation probability
    :param reuse_cond_exp: bool, whether to reuse the conditional expectation
        from the last computation
    :param output_coords: None or list of ints, the coordinates corresponding to
        the model output
    :param loss_quantiles: None or list of floats, the quantiles to plot for the
        loss
    :param input_coords: None or list of ints, the coordinates corresponding to
        the model input
    :param which_loss: str, the loss function to use for the computation
    :param plot_error_dists: None or dict, the kwargs for plotting the error
        distribution
    :param ref_model_to_use: None or str, the reference models to use in the
        plots, None uses the standard reference model, usually the conditional
        expectation

    :return: optimal loss
    """
    if model_name is None or model_name == "NJODE":
        model_name = 'our model'

    prop_cycle = plt.rcParams['axes.prop_cycle']  # change style of plot?
    colors = prop_cycle.by_key()['color']
    std_color = list(matplotlib.colors.to_rgb(colors[1])) + [0.5]
    std_color2 = list(matplotlib.colors.to_rgb(colors[2])) + [0.5]

    makedirs(save_path)  # create a directory

    times = batch["times"]
    time_ptr = batch["time_ptr"]
    X = batch["X"].to(device)
    M = batch["M"]
    if M is not None:
        M = M.to(device)
    start_X = batch["start_X"].to(device)
    start_M = batch["start_M"]
    if start_M is not None:
        start_M = start_M.to(device)
    obs_idx = batch["obs_idx"]
    n_obs_ot = batch["n_obs_ot"].to(device)
    true_X = batch["true_paths"]
    # dim does not take the function applications into account
    bs, dim, time_steps = true_X.shape
    if output_coords is None:
        output_coords = list(range(dim))
        out_dim = dim
    else:
        # if output_coords is given, then they also include the function
        #   applications
        mult = len(functions)+1 if functions is not None else 1
        out_dim = int(len(output_coords)/mult)
    true_M = batch["true_mask"]
    observed_dates = batch['observed_dates']
    if "obs_noise" in batch:
        obs_noise = batch["obs_noise"]
    else:
        obs_noise = None
    path_t_true_X = np.linspace(0., T, int(np.round(T / delta_t)) + 1)

    model.eval()  # put model in evaluation mode
    res = model.get_pred(
        times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx, delta_t=delta_t,
        T=T, start_X=start_X, M=M, start_M=start_M, n_obs_ot=n_obs_ot,
        which_loss=which_loss)
    path_y_pred = res['pred'].detach().cpu().numpy()
    path_t_pred = res['pred_t']
    current_model_loss = res['loss'].detach().cpu().numpy()

    # get variance path
    if plot_variance and (functions is not None) and ('power-2' in functions):
        which = np.argmax(np.array(functions) == 'power-2')+1
        y2 = path_y_pred[:, :, (out_dim * which):(out_dim * (which + 1))]
        path_var_pred = y2 - np.power(path_y_pred[:, :, 0:out_dim], 2)
        if np.any(path_var_pred < 0):
            print('WARNING: some predicted cond. variances below 0 -> clip')
            path_var_pred = np.maximum(0, path_var_pred)
        path_std_pred = np.sqrt(path_var_pred)
    elif plot_variance and (model.compute_variance is not None):
        path_var_pred = res["pred_var"].detach().cpu().numpy()
        if model.compute_variance == "variance":
            path_var_pred = path_var_pred[:, :, 0:out_dim]**2
        elif model.compute_variance == "covariance":
            d = int(np.sqrt(path_var_pred.shape[2]))
            path_var_pred = path_var_pred.reshape(
                path_y_pred.shape[0], path_y_pred.shape[1], d, d)
            path_var_pred = np.matmul(
                path_var_pred.transpose(0,1,3,2), path_var_pred)
            path_var_pred = np.diagonal(path_var_pred, axis1=2, axis2=3)
            path_var_pred = path_var_pred[:, :, 0:out_dim]
        else:
            raise ValueError("compute_variance {} not implemented".format(
                model.compute_variance))
        path_std_pred = np.sqrt(path_var_pred)
    else:
        plot_variance = False
    path_var_true = None
    if use_cond_exp:
        if M is not None:
            M = M.detach().cpu().numpy()
        if (functions is not None and functions == ["power-2"] and
                stockmodel.loss_comp_for_pow2_implemented):
            X_ = X.detach().cpu().numpy()
            start_X_ = start_X.detach().cpu().numpy()
        else:
            X_ = X.detach().cpu().numpy()[:, :dim]
            start_X_ = start_X.detach().cpu().numpy()[:, :dim]
            if M is not None:
                M = M[:, :dim]
        res_sm = stockmodel.compute_cond_exp(
            times, time_ptr, X_, obs_idx.detach().cpu().numpy(),
            delta_t, T, start_X_, n_obs_ot.detach().cpu().numpy(),
            return_path=True, get_loss=True, weight=model.weight,
            M=M, store_and_use_stored=reuse_cond_exp,
            return_var=(plot_variance and plot_true_var),
            which_loss=which_loss, ref_model=ref_model_to_use)
        opt_loss, path_t_true, path_y_true = res_sm[:3]
        if plot_variance and len(res_sm) > 3:
            path_var_true = res_sm[3]

        # get additional reference model predictions for error distribution
        #   plots
        if (plot_error_dist is not None
                and "additional_ref_models" in plot_error_dist):
            add_model_preds = []
            add_model_ts = []
            for ref_model in plot_error_dist["additional_ref_models"]:
                res_sm_add = stockmodel.compute_cond_exp(
                    times, time_ptr, X_, obs_idx.detach().cpu().numpy(),
                    delta_t, T, start_X_, n_obs_ot.detach().cpu().numpy(),
                    return_path=True, get_loss=True, weight=model.weight,
                    M=M, store_and_use_stored=False,
                    return_var=plot_variance,
                    which_loss=which_loss, ref_model=ref_model)
                _, t, pred = res_sm_add[:3]
                add_model_preds.append(pred)
                add_model_ts.append(t)
    else:
        opt_loss = 0

    # plot the error distribution
    err_dist_paths = None
    if plot_error_dist is not None:
        names = ["model", "cond. exp."]
        if "additional_ref_models" in plot_error_dist:
            names += plot_error_dist["additional_ref_models"]
        if "model_names" not in plot_error_dist:
            plot_error_dist["model_names"] = names
        err_dist_filename = "{}error_distribution_plot_coord{}.pdf".format(
            save_path, "{}")
        model_preds = [path_y_pred,]
        model_ts = [path_t_pred,]
        if use_cond_exp:
            model_preds.append(path_y_true)
            model_ts.append(path_t_true)
        if "additional_ref_models" in plot_error_dist:
            model_preds += add_model_preds
            model_ts += add_model_ts
        err_dist_paths = plot_error_distribution(
            true_paths=true_X, true_times=path_t_true_X,
            model_preds=model_preds, model_ts=model_ts,
            output_coords=output_coords,
            original_out_dim=out_dim, eval_times=plot_error_dist["eval_times"],
            colors=colors, model_names=plot_error_dist["model_names"],
            save_path=save_path, filename=err_dist_filename,
            coord_names=ylabels)

    for i in path_to_plot:
        fig, axs = plt.subplots(dim, sharex=True)
        if dim == 1:
            axs = [axs]
        outcoord_ind = -1
        unobserved_coord = False
        for j in range(dim):
            if j in output_coords:
                outcoord_ind += 1
            # get the true_X at observed dates
            path_t_obs = []
            path_X_obs = []
            if obs_noise is not None:
                path_O_obs = []
            for k, od in enumerate(observed_dates[i]):
                if od == 1:
                    if true_M is None or (true_M is not None and
                                          true_M[i, j, k]==1):
                        path_t_obs.append(path_t_true_X[k])
                        path_X_obs.append(true_X[i, j, k])
                        if obs_noise is not None:
                            path_O_obs.append(
                                true_X[i, j, k]+obs_noise[i, j, k])
            path_t_obs = np.array(path_t_obs)
            path_X_obs = np.array(path_X_obs)
            if obs_noise is not None:
                path_O_obs = np.array(path_O_obs)

            # get the legend labels
            lab0 = legendlabels[0] if legendlabels is not None else 'true path'
            lab1 = legendlabels[1] if legendlabels is not None else model_name
            lab2 = legendlabels[2] if legendlabels is not None \
                else 'true conditional expectation'
            lab3 = legendlabels[3] if legendlabels is not None else 'observed'

            axs[j].plot(path_t_true_X, true_X[i, j, :], label=lab0,
                        color=colors[0])
            if obs_noise is not None:
                axs[j].scatter(path_t_obs, path_O_obs, label=lab3,
                               color=colors[0])
                # axs[j].scatter(recr_t, recr_X[i, :, j],label='observed recr.',
                #                color="black", marker="x")
                lab4 = legendlabels[4] if legendlabels is not None \
                    else 'true value at obs time'
                axs[j].scatter(path_t_obs, path_X_obs,
                               label=lab4,
                               color=colors[2], marker='*')
            else:
                facecolors = colors[0]
                if input_coords is not None and j not in input_coords:
                    facecolors = 'none'
                    lab3 = '(un)observed'
                    unobserved_coord = True
                axs[j].scatter(path_t_obs, path_X_obs, label=lab3,
                               color=colors[0], facecolors=facecolors)
            if j in output_coords:
                if loss_quantiles is not None:
                    for iq, q in enumerate(loss_quantiles):
                        axs[j].plot(
                            path_t_pred, path_y_pred[:, i, outcoord_ind, iq],
                            label="q{} - {}".format(q, lab1),
                            color=list(matplotlib.colors.to_rgb(colors[1]))+
                                  [2*min(q, 1-q)])
                else:
                    axs[j].plot(path_t_pred, path_y_pred[:, i, outcoord_ind],
                                label=lab1, color=colors[1])
                if plot_variance:
                    axs[j].fill_between(
                        path_t_pred,
                        path_y_pred[:, i, outcoord_ind] - std_factor *
                        path_std_pred[:, i, outcoord_ind],
                        path_y_pred[:, i, outcoord_ind] + std_factor *
                        path_std_pred[:, i, outcoord_ind],
                        color=std_color)
                if use_cond_exp:
                    if loss_quantiles is not None:
                        for iq, q in enumerate(loss_quantiles):
                            axs[j].plot(
                                path_t_true, path_y_true[:, i, outcoord_ind,iq],
                                label="q{} - {}".format(q, lab2),
                                linestyle=':',
                                color=list(matplotlib.colors.to_rgb(colors[2]))+
                                      [2*min(q, 1-q)])
                    else:
                        axs[j].plot(path_t_true, path_y_true[:,i,outcoord_ind],
                                    label=lab2, linestyle=':', color=colors[2])
                    if plot_variance and path_var_true is not None:
                        axs[j].fill_between(
                            path_t_true,
                            path_y_true[:, i, outcoord_ind] - std_factor *
                            np.sqrt(path_var_true[:, i, outcoord_ind]),
                            path_y_true[:, i, outcoord_ind] + std_factor *
                            np.sqrt(path_var_true[:, i, outcoord_ind]),
                            color=std_color2)
            if plot_obs_prob and dataset_metadata is not None:
                ax2 = axs[j].twinx()
                if "X_dependent_observation_prob" in dataset_metadata:
                    prob_f = eval(
                        dataset_metadata["X_dependent_observation_prob"])
                    obs_perc = prob_f(true_X[:, :, :])[i]
                elif "obs_scheme" in dataset_metadata:
                    obs_scheme = dataset_metadata["obs_scheme"]
                    if obs_scheme["name"] == "NJODE3-Example4.9":
                        obs_perc = np.ones_like(path_t_true_X)
                        x0 = true_X[i, 0, 0]
                        p = obs_scheme["p"]
                        eta = obs_scheme["eta"]
                        last_observation = x0
                        last_obs_time = 0
                        for k, t in enumerate(path_t_true_X[1:]):
                            q = 1/(k+1-last_obs_time)
                            normal_prob = stats.norm.sf(
                                stockmodel.next_cond_exp(
                                    x0, (k+1)*delta_t, (k+1)*delta_t),
                                scale=eta, loc=last_observation)
                            obs_perc[k+1] = q*normal_prob + (1-q)*p
                            if observed_dates[i, k+1] == 1:
                                last_observation = true_X[i, 0, k+1]
                                last_obs_time = k+1
                else:
                    obs_perc = dataset_metadata['obs_perc']
                    obs_perc = np.ones_like(path_t_true_X) * obs_perc
                ax2.plot(path_t_true_X, obs_perc, color="red",
                         label="observation probability")
                ax2.set_ylim(-0.1, 1.1)
                ax2.set_ylabel("observation probability")
                axs[j].set_ylabel("$X$")
                ax2.legend()
                # axs[j].set_xlabel("$t$")
            if ylabels:
                axs[j].set_ylabel(ylabels[j])
            if same_yaxis:
                low = np.min(true_X[i, :, :])
                high = np.max(true_X[i, :, :])
                if obs_noise is not None:
                    low = min(low, np.min(true_X[i]+obs_noise[i]))
                    high = max(high, np.max(true_X[i]+obs_noise[i]))
                eps = (high - low)*0.05
                axs[j].set_ylim([low-eps, high+eps])

        if unobserved_coord:
            handles, labels = axs[-1].get_legend_handles_labels()
            l, = axs[-1].plot(
                [], [], color=colors[0], label='(un)observed',
                linestyle='none', marker="o", fillstyle="right")

            handles[-1] = l
            labels[-1] = 'unobserved/observed'
            axs[-1].legend(handles, labels)
        else:
            axs[-1].legend()
        plt.xlabel('$t$')
        save = os.path.join(save_path, filename.format(i))
        plt.savefig(save, **save_extras)
        plt.close()

    return opt_loss, current_model_loss, err_dist_paths


