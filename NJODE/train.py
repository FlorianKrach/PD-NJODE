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

from configs import config
import models
import data_utils
sys.path.append("../")
import GRU_ODE_Bayes.models_gru_ode_bayes as models_gru_ode_bayes

try:
    from telegram_notifications import send_bot_message as SBM
except Exception:
    from config import SendBotMessage as SBM


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
    'epoch', 'train_time', 'eval_time', 'train_loss', 'eval_loss',
    'optimal_eval_loss']
default_ode_nn = ((50, 'tanh'), (50, 'tanh'))
default_readout_nn = ((50, 'tanh'), (50, 'tanh'))
default_enc_nn = ((50, 'tanh'), (50, 'tanh'))

ANOMALY_DETECTION = False
N_DATASET_WORKERS = 0
USE_GPU = False

# =====================================================================================================================
# Functions
makedirs = config.makedirs


def train(
        anomaly_detection=None, n_dataset_workers=None, use_gpu=None,
        nb_cpus=None, send=None,
        model_id=None, epochs=100, batch_size=100, save_every=1,
        learning_rate=0.001, test_size=0.2, seed=398,
        hidden_size=10, bias=True, dropout_rate=0.1,
        ode_nn=default_ode_nn, readout_nn=default_readout_nn,
        enc_nn=default_enc_nn, use_rnn=False,
        solver="euler", weight=0.5, weight_decay=1.,
        dataset='BlackScholes', dataset_id=None, data_dict=None,
        plot=True, paths_to_plot=(0,),
        saved_models_path=saved_models_path,
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
    :param options: kwargs, used keywords:
            'test_data_dict'    None, str or dict, if no None, this data_dict is
                            used to define the dataset for testing and
                            evaluation
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
            'plot_same_yaxis'   bool, whether to plot the same range on y axis
                            for all dimensions
            'plot_obs_prob' bool, whether to plot the observation probability
            'which_loss'    'standard' or 'easy', used by models.NJODE
            'residual_enc_dec'  bool, whether resNNs are used for encoder and
                            readout NN, used by models.NJODE, default True
            'use_y_for_ode' bool, whether to use y (after jump) or x_impute for
                            the ODE as input, only in masked case, default: True
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
                            (i.e. not only compute the eval_loss, but also
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
                            Other options/inputs might change or loose their
                            effect.
            'ode_input_scaling_func'    None or str in {'id', 'tanh'}, the
                            function used to scale inputs to the neuralODE.
                            default: tanh
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
        gpu_num = 0
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
    input_size = dataset_metadata['dimension']
    dimension = dataset_metadata['dimension']
    output_size = input_size
    T = dataset_metadata['maturity']
    delta_t = dataset_metadata['dt']  # copy metadata

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
        data_val = data_utils.IrregularDataset(
            model_name=test_ds, time_id=test_ds_id, idx=None)

    # get data-loader for training
    if 'func_appl_X' in options:  # list of functions to apply to the paths in X
        functions = options['func_appl_X']
        collate_fn, mult = data_utils.CustomCollateFnGen(functions)
        input_size = input_size * mult
        output_size = output_size * mult
    else:
        functions = None
        collate_fn, mult = data_utils.CustomCollateFnGen(None)
        mult = 1

    dl = DataLoader(  # class to iterate over training data
        dataset=data_train, collate_fn=collate_fn,
        shuffle=True, batch_size=batch_size, num_workers=N_DATASET_WORKERS)
    dl_val = DataLoader(  # class to iterate over validation data
        dataset=data_val, collate_fn=collate_fn,
        shuffle=False, batch_size=len(data_val), num_workers=N_DATASET_WORKERS)

    # get additional plotting information
    plot_variance = False
    std_factor = 1  # factor with which the std is multiplied
    if functions is not None and mult > 1:
        if 'plot_variance' in options:
            plot_variance = options['plot_variance']
        if 'std_factor' in options:
            std_factor = options['std_factor']
    ylabels = None
    if 'ylabels' in options:
        ylabels = options['ylabels']
    plot_same_yaxis = False
    if 'plot_same_yaxis' in options:
        plot_same_yaxis = options['plot_same_yaxis']
    plot_obs_prob = False
    if 'plot_obs_prob' in options:
        plot_obs_prob = options["plot_obs_prob"]

    # get optimal eval loss
    #   -> if other functions are applied to X, then only the original X is used
    #      in the computation of the optimal eval loss
    print(dataset_metadata)
    stockmodel = data_utils._STOCK_MODELS[
        dataset_metadata['model_name']](**dataset_metadata)
    if use_cond_exp:
        opt_eval_loss = compute_optimal_eval_loss(
            dl_val, stockmodel, delta_t, T, mult=mult)
        initial_print += '\noptimal eval loss (achieved by true cond exp): ' \
                     '{:.5f}'.format(opt_eval_loss)
    else:
        opt_eval_loss = np.nan
    if 'other_model' in options:
        opt_eval_loss = np.nan

    # get params_dict
    params_dict = {  # create a dictionary of the wanted parameters
        'input_size': input_size, 'epochs': epochs,
        'hidden_size': hidden_size, 'output_size': output_size, 'bias': bias,
        'ode_nn': ode_nn, 'readout_nn': readout_nn, 'enc_nn': enc_nn,
        'use_rnn': use_rnn,
        'dropout_rate': dropout_rate, 'batch_size': batch_size,
        'solver': solver, 'dataset': dataset, 'dataset_id': dataset_id,
        'learning_rate': learning_rate, 'test_size': test_size, 'seed': seed,
        'weight': weight, 'weight_decay': weight_decay,
        'optimal_eval_loss': opt_eval_loss, 'options': options}
    desc = json.dumps(params_dict, sort_keys=True)  # serialize to a JSON formatted str

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
    best_eval_loss = np.infty
    if 'evaluate' in options and options['evaluate']:
        metr_columns = METR_COLUMNS + ['evaluation_mean_diff']
    else:
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
            best_eval_loss = np.min(df_metric['eval_loss'].values)
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
    if 'plot_only' in options and options['plot_only']:
        for i, b in enumerate(dl_val):
            batch = b
        model.epoch -= 1
        initial_print += '\nplotting ...'
        plot_filename = 'demo-plot_epoch-{}'.format(model.epoch)
        plot_filename = plot_filename + '_path-{}.pdf'
        curr_opt_loss = plot_one_path_with_pred(
            device, model, batch, stockmodel, delta_t, T,
            path_to_plot=paths_to_plot, save_path=plot_save_path,
            filename=plot_filename, plot_variance=plot_variance,
            functions=functions, std_factor=std_factor,
            model_name=model_name, save_extras=save_extras, ylabels=ylabels,
            same_yaxis=plot_same_yaxis, plot_obs_prob=plot_obs_prob,
            dataset_metadata=dataset_metadata)
        if SEND:
            files_to_send = []
            caption = "{} - id={}".format(model_name, model_id)
            for i in paths_to_plot:
                files_to_send.append(
                    os.path.join(plot_save_path, plot_filename.format(i)))
            SBM.send_notification(
                text='finished plot-only: {}, id={}\n\n{}'.format(
                    model_name, model_id, desc),
                chat_id=config.CHAT_ID,
                files=files_to_send,
                text_for_files=caption
            )
        initial_print += '\noptimal eval-loss (with current weight={:.5f}): ' \
                         '{:.5f}'.format(model.weight, curr_opt_loss)
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
                    return_path=False, get_loss=True, M=M, start_M=start_M)
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
                print(r"current loss: {}".format(loss.detach().numpy()))
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
                if plot:
                    batch = b
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
                        start_M=start_M, which_loss='standard')
                elif options['other_model'] == "GRU_ODE_Bayes":
                    if M is None:
                        M = torch.ones_like(X)
                    hT, c_loss, _, _ = model(
                        times, time_ptr, X, M, obs_idx, delta_t, T, start_X,
                        return_path=False, smoother=False,)
                else:
                    raise ValueError
                loss_val += c_loss.detach().numpy()
                num_obs += 1  # count number of observations

                # if functions are applied, also compute the loss when only
                #   using the coordinates where function was not applied
                #   -> this can be compared to the optimal-eval-loss
                if mult is not None and mult > 1:
                    if 'other_model' not in options:
                        hT_corrected, c_loss_corrected = model(
                            times, time_ptr, X, obs_idx, delta_t, T, start_X,
                            n_obs_ot, return_path=False, get_loss=True, M=M,
                            start_M=start_M, which_loss='standard',
                            dim_to=dimension)
                    loss_val_corrected += c_loss_corrected.detach().numpy()

                # mean squared difference evaluation
                if 'evaluate' in options and options['evaluate']:
                    if use_cond_exp:
                        true_paths = None
                        true_mask = None
                    _eval_msd = model.evaluate(
                        times=times, time_ptr=time_ptr, X=X,
                        obs_idx=obs_idx, delta_t=delta_t, T=T,
                        start_X=start_X, n_obs_ot=n_obs_ot,
                        stockmodel=stockmodel, return_paths=False, M=M,
                        start_M=start_M, true_paths=true_paths,
                        true_mask=true_mask, mult=mult)
                    eval_msd += _eval_msd

            eval_time = time.time() - t
            loss_val = loss_val / num_obs
            loss_val_corrected /= num_obs
            eval_msd = eval_msd / num_obs
            train_loss = loss.detach().numpy()
            print_str = "epoch {}, weight={:.5f}, train-loss={:.5f}, " \
                        "optimal-eval-loss={:.5f}, eval-loss={:.5f}, ".format(
                model.epoch, model.weight, train_loss, opt_eval_loss, loss_val)
            if mult is not None and mult > 1:
                print_str += "\ncorrected(i.e. without additional dims of " \
                             "funct_appl_X)-eval-loss={:.5f}, ".format(
                    loss_val_corrected)
            print(print_str)
        if 'evaluate' in options and options['evaluate']:
            metric_app.append([model.epoch, train_time, eval_time, train_loss,
                               loss_val, opt_eval_loss, eval_msd])
            print("evaluation mean square difference={:.5f}".format(
                eval_msd))
        else:
            metric_app.append([model.epoch, train_time, eval_time, train_loss,
                               loss_val, opt_eval_loss])

        # save model
        if model.epoch % save_every == 0:
            if plot:
                print('plotting ...')
                plot_filename = 'epoch-{}'.format(model.epoch)
                plot_filename = plot_filename + '_path-{}.pdf'
                curr_opt_loss = plot_one_path_with_pred(
                    device=device, model=model, batch=batch,
                    stockmodel=stockmodel, delta_t=delta_t, T=T,
                    path_to_plot=paths_to_plot, save_path=plot_save_path,
                    filename=plot_filename, plot_variance=plot_variance,
                    functions=functions, std_factor=std_factor,
                    model_name=model_name, save_extras=save_extras,
                    ylabels=ylabels, use_cond_exp=use_cond_exp,
                    same_yaxis=plot_same_yaxis, plot_obs_prob=plot_obs_prob,
                    dataset_metadata=dataset_metadata)
                print('optimal eval-loss (with current weight={:.5f}): '
                      '{:.5f}'.format(model.weight, curr_opt_loss))
            print('save model ...')
            df_m_app = pd.DataFrame(data=metric_app, columns=metr_columns)
            df_metric = pd.concat([df_metric, df_m_app], ignore_index=True)
            df_metric.to_csv(model_metric_file)
            models.save_checkpoint(model, optimizer, model_path_save_last,
                                   model.epoch)
            metric_app = []
            print('saved!')
        if loss_val < best_eval_loss:
            print('save new best model: last-best-loss: {:.5f}, '
                  'new-best-loss: {:.5f}, epoch: {}'.format(
                best_eval_loss, loss_val, model.epoch))
            df_m_app = pd.DataFrame(data=metric_app, columns=metr_columns)
            df_metric = pd.concat([df_metric, df_m_app], ignore_index=True)
            df_metric.to_csv(model_metric_file)
            models.save_checkpoint(model, optimizer, model_path_save_last,
                                   model.epoch)
            models.save_checkpoint(model, optimizer, model_path_save_best,
                                   model.epoch)
            metric_app = []
            best_eval_loss = loss_val
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
    del model, dl, dl_val, data_train, data_val
    gc.collect()

    return 0


def compute_optimal_eval_loss(dl_val, stockmodel, delta_t, T, mult=None):
    """
    compute optimal evaluation loss (with the true cond. exp.) on the
    test-dataset
    :param dl_val: torch.DataLoader, used for the validation dataset
    :param stockmodel: stock_model.StockModel instance
    :param delta_t: float, the time_delta
    :param T: float, the terminal time
    :return: float (optimal loss)
    """
    opt_loss = 0
    num_obs = 0
    for i, b in enumerate(dl_val):
        times = b["times"]
        time_ptr = b["time_ptr"]
        X = b["X"].detach().numpy()
        start_X = b["start_X"].detach().numpy()
        obs_idx = b["obs_idx"].detach().numpy()
        n_obs_ot = b["n_obs_ot"].detach().numpy()
        M = b["M"]
        if M is not None:
            M = M.detach().numpy()
        num_obs += 1
        opt_loss += stockmodel.get_optimal_loss(
            times, time_ptr, X, obs_idx, delta_t, T, start_X, n_obs_ot, M=M,
            mult=mult)
    return opt_loss / num_obs


def plot_one_path_with_pred(
        device, model, batch, stockmodel, delta_t, T,
        path_to_plot=(0,), save_path='', filename='plot_{}.pdf',
        plot_variance=False, functions=None, std_factor=1,
        model_name=None, ylabels=None,
        save_extras={'bbox_inches': 'tight', 'pad_inches': 0.01},
        use_cond_exp=True, same_yaxis=False,
        plot_obs_prob=False, dataset_metadata=None,
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
    :param functions: list of functions (as str), the functions applied to X
    :param std_factor: float, the factor by which std is multiplied
    :param model_name: str or None, name used for model in plots
    :param ylabels: None or list of str of same length as dimension of X
    :param save_extras: dict with extra options for saving plot
    :param use_cond_exp: bool, whether to plot the conditional expectation
    :param same_yaxis: bool, whether to plot all coordinates with same range on
        y-axis
    :param plot_obs_prob: bool, whether to plot the probability of an
        observation for all times
    :param dataset_metadata: needed if plot_obs_prob=true, the metadata of the
        used dataset to extract the observation probability
    :return: optimal loss
    """
    if model_name is None or model_name == "NJODE":
        model_name = 'our model'

    prop_cycle = plt.rcParams['axes.prop_cycle']  # change style of plot?
    colors = prop_cycle.by_key()['color']
    std_color = list(matplotlib.colors.to_rgb(colors[1])) + [0.5]

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
    bs, dim, time_steps = true_X.shape
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
        T=T, start_X=start_X, M=M, start_M=start_M)
    path_y_pred = res['pred'].detach().numpy()
    path_t_pred = res['pred_t']

    # get variance path
    if plot_variance and (functions is not None) and ('power-2' in functions):
        which = np.argmax(np.array(functions) == 'power-2')+1
        y2 = path_y_pred[:, :, (dim * which):(dim * (which + 1))]
        path_var_pred = y2 - np.power(path_y_pred[:, :, 0:dim], 2)
        if np.any(path_var_pred < 0):
            print('WARNING: some predicted cond. variances below 0 -> clip')
            path_var_pred = np.maximum(0, path_var_pred)
        path_std_pred = np.sqrt(path_var_pred)
    else:
        plot_variance = False
    if use_cond_exp:
        if M is not None:
            M = M.detach().numpy()
        opt_loss, path_t_true, path_y_true = stockmodel.compute_cond_exp(
            times, time_ptr, X.detach().numpy(), obs_idx.detach().numpy(),
            delta_t, T, start_X.detach().numpy(), n_obs_ot.detach().numpy(),
            return_path=True, get_loss=True, weight=model.weight,
            M=M,)
    else:
        opt_loss = 0

    for i in path_to_plot:
        fig, axs = plt.subplots(dim)
        if dim == 1:
            axs = [axs]
        for j in range(dim):
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

            axs[j].plot(path_t_true_X, true_X[i, j, :], label='true path',
                        color=colors[0])
            if obs_noise is not None:
                axs[j].scatter(path_t_obs, path_O_obs, label='observed',
                               color=colors[0])
                axs[j].scatter(path_t_obs, path_X_obs,
                               label='true value at obs time',
                               color=colors[2], marker='*')
            else:
                axs[j].scatter(path_t_obs, path_X_obs, label='observed',
                               color=colors[0])
            axs[j].plot(path_t_pred, path_y_pred[:, i, j],
                        label=model_name, color=colors[1])
            if plot_variance:
                axs[j].fill_between(
                    path_t_pred,
                    path_y_pred[:, i, j] - std_factor * path_std_pred[:, i, j],
                    path_y_pred[:, i, j] + std_factor * path_std_pred[:, i, j],
                    color=std_color)
            if use_cond_exp:
                axs[j].plot(path_t_true, path_y_true[:, i, j],
                            label='true conditional expectation',
                            linestyle=':', color=colors[2])
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
                axs[j].set_ylabel("X")
                ax2.legend()
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

        axs[-1].legend()
        plt.xlabel('$t$')
        save = os.path.join(save_path, filename.format(i))
        plt.savefig(save, **save_extras)
        plt.close()

    return opt_loss


