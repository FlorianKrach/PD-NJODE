"""
author: Florian Krach
"""

# =====================================================================================================================
from typing import List

import torch  # machine learning
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
import sklearn

from configs import config
import models
import data_utils
sys.path.append("../")

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

METR_COLUMNS = ['epoch', 'train_time', 'eval_time', 'train_loss',
                'eval_loss', 'mse_eval_loss', 'classification_eval_loss']

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
        output_midprice_only=True, use_eval_on_train=True,
        classifier_nn=None,
        solver="euler", weight=0.5, weight_decay=1.,
        dataset_id=None, data_dict=None, plot=True, paths_to_plot=(0,),
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
    :param output_midprice_only: bool, whether the model output is only the
            midprice or the same as the input
    :param use_eval_on_train: bool, whether to use the eval parts on the train
            dataset for training (=> filtering type of framework)
    :param classifier_nn: None or network dict (as ode_nn)
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
            'plot_only'     bool, whether to only plot from model instead of
                            training it
            'plot_errors'   bool, whether to only plot errors from model
            'save_extras'   bool, dict of options for saving the plots
            'parallel'      bool, used by parallel_train.parallel_training
            'resume_training'   bool, used by parallel_train.parallel_training
            'ylabels'       list of str, see plot_one_path_with_pred()
            'plot_same_yaxis'   bool, whether to plot the same range on y axis
                            for all dimensions
            'which_loss'    'standard' or 'easy', used by models.NJODE
            'classifier_loss_weight'    float, weighting for classification loss
            'residual_enc_dec'  bool, whether resNNs are used for encoder and
                                readout NN, used by models.NJODE, default True
            'input_sig'     bool, whether to use the signature as input
            'level'         int, level of the signature that is used
            'use_sig_for_classifier'    bool, whether to use the signature as
                            additional input for the classifier network,
                            default: False
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
            'other_model'   one of {'GRU_ODE_Bayes'}; the specifieed model is
                            trained instead of the controlled ODE-RNN model.
                            Other options/inputs might change or loose their
                            effect. The saved_models_path is changed to
                            "{...}<model-name>-saved_models/" instead of
                            "{...}saved_models/".
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
        dataset = "LOB"
        dataset_id = int(data_utils._get_time_id(
            stock_model_name=dataset, time_id=dataset_id))
    dataset_metadata = data_utils.load_metadata(
        stock_model_name=dataset, time_id=dataset_id)
    eval_predict_steps = dataset_metadata['eval_predict_steps']
    use_volume = dataset_metadata['use_volume']
    input_size = dataset_metadata['LOB_level']*2*(1+use_volume) + 1
    dimension = input_size
    output_size = input_size
    delta_t = dataset_metadata['dt']
    normalizing_price_mean = dataset_metadata["price_mean"]
    normalizing_price_std = dataset_metadata["price_std"]
    thresholds = dataset_metadata["thresholds"]
    dim_to = None
    if output_midprice_only:
        dim_to = 1
    train_classifier = False
    classifier_dict = None
    if classifier_nn is not None:
        train_classifier = True
        classifier_dict = {
            "nn_desc": classifier_nn, 'input_size': hidden_size,
            'output_size': 3, 'dropout_rate': dropout_rate, 'bias': bias,
            'residual': False}

    # load raw data
    train_idx, val_idx = train_test_split(
        np.arange(dataset_metadata["nb_paths"]), test_size=test_size,
        random_state=seed, shuffle=False)
    # --> get subset of training samples if wanted
    if 'training_size' in options:
        train_set_size = options['training_size']
        if train_set_size < len(train_idx):
            train_idx = np.random.choice(
                train_idx, train_set_size, replace=False)
    data_train = data_utils.LOBDataset(time_id=dataset_id, idx=train_idx)
    data_val = data_utils.LOBDataset(time_id=dataset_id, idx=val_idx)

    # get data-loader
    collate_fn = data_utils.LOBCollateFnGen(
        data_type="train", use_eval_on_train=use_eval_on_train,
        train_classifier=train_classifier)
    collate_fn_val = data_utils.LOBCollateFnGen(data_type="test")

    dl = DataLoader(
        dataset=data_train, collate_fn=collate_fn,
        shuffle=True, batch_size=batch_size, num_workers=N_DATASET_WORKERS)
    dl_val = DataLoader(
        dataset=data_val, collate_fn=collate_fn_val,
        shuffle=False, batch_size=batch_size,
        num_workers=N_DATASET_WORKERS)

    # get additional plotting information
    ylabels = None
    if 'ylabels' in options:
        ylabels = options['ylabels']
    plot_same_yaxis = False
    if 'plot_same_yaxis' in options:
        plot_same_yaxis = options['plot_same_yaxis']

    initial_print += "\n{}".format(dataset_metadata)

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
        "output_midprice_only": output_midprice_only,
        'use_eval_on_train': use_eval_on_train,
        'classifier_nn': classifier_nn, 'classifier_dict': classifier_dict,
        'options': options}
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
            df_overview = pd.read_csv(model_overview_file_name, index_col=0)
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
            initial_print += '\nmodel_id already exists -> resume training'
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
    model = models.NJODE(**params_dict)  # get NJODE model class from
    model_name = 'NJODE'
    model.to(device)  # pass model to CPU/GPU
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=0.0005)

    # load saved model if wanted/possible
    best_eval_loss = np.infty
    if 'evaluate' in options and options['evaluate']:
        metr_columns = METR_COLUMNS + [
            'evaluation_mse', 'ref_evaluation_mse', 'evaluation_f1score']
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
            if 'evaluation_mse' in df_metric.columns:
                best_eval_loss = np.min(df_metric['evaluation_mse'].values)
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
    stop = False
    if 'plot_errors' in options and options['plot_errors']:
        stop = True
        with torch.no_grad():
            eval_msd = 0
            ref_eval_msd = 0
            predicted_vals = np.zeros((len(data_val), dimension))
            true_vals = np.zeros((len(data_val), dimension))
            predicted_labels = np.zeros(len(data_val))
            true_labels = np.zeros(len(data_val))
            cbs = 0
            model.eval()
            for i, b in enumerate(dl_val):
                times = b["times"]
                time_ptr = b["time_ptr"]
                X = b["X"].to(device)
                start_X = b["start_X"].to(device)
                bs = start_X.shape[0]
                obs_idx = b["obs_idx"]
                n_obs_ot = b["n_obs_ot"]
                T = b["max_time"]
                predict_times = b["predict_times"]
                predict_vals = b["predict_vals"]
                true_eval_labels = b["true_eval_labels"]
                samples = b["true_samples"]
                coord_to_compare = b["coord_to_compare"]
                pred_labels = b["predict_labels"]

                # mean squared error evaluation
                if 'evaluate' in options and options['evaluate']:
                    _eval_msd, _ref_eval_msd, _eval_f1score, _, _, _pred_vals, \
                    _pred_labels = model.evaluate_LOB(
                        times=times, time_ptr=time_ptr, X=X,
                        obs_idx=obs_idx, delta_t=delta_t, T=T,
                        start_X=start_X, n_obs_ot=n_obs_ot,
                        return_paths=True, predict_times=predict_times,
                        true_predict_vals=predict_vals,
                        coord_to_compare=coord_to_compare,
                        true_predict_labels=true_eval_labels,
                        true_samples=samples,
                        normalizing_mean=normalizing_price_mean,
                        normalizing_std=normalizing_price_std,
                        eval_predict_steps=eval_predict_steps,
                        thresholds=thresholds, predict_labels=pred_labels)
                    eval_msd += _eval_msd
                    ref_eval_msd += _ref_eval_msd
                    predicted_vals[cbs:cbs+bs] = _pred_vals
                    true_vals[cbs:cbs+bs] = predict_vals[:, :, 0]
                    true_labels[cbs:cbs+bs] = true_eval_labels[:, 0]
                    if model.classifier is not None:
                        predicted_labels[cbs:cbs+bs] = _pred_labels
                    cbs += bs
            pred_errors = predicted_vals[:, coord_to_compare] - \
                          true_vals[:, coord_to_compare]
            pred_errors = pred_errors[:, 0]
        plot_filename = 'error-distr-plot_epoch-{}_id-{}.pdf'.format(
            model.epoch, model_id)
        fig, axs = plt.subplots(ncols=2)
        axs[0].hist(pred_errors, bins=100, density=True)
        axs[1].boxplot(pred_errors)
        save_f = os.path.join(plot_save_path, plot_filename)
        plt.savefig(save_f, **save_extras)
        plt.close()

        data = []
        for which in [None, -1, 0, 1]:
            if which is not None:
                vals_t = true_vals[true_labels==which, coord_to_compare]
                vals_p = predicted_vals[true_labels==which, coord_to_compare]
            else:
                vals_t = true_vals[:, coord_to_compare]
                vals_p = predicted_vals[:, coord_to_compare]
            mean_t = np.mean(vals_t)
            mean_p = np.mean(vals_p)
            data.append([which, mean_t, mean_p])
        df_means = pd.DataFrame(
            data=data, columns=["which", "true_mean", "pred_mean"])
        means_file = "{}true_pred_means.csv".format(plot_save_path)
        df_means.to_csv(means_file)

        if SEND:
            files_to_send = [save_f, means_file]
            caption = "{} - id={}".format(model_name, model_id)
            SBM.send_notification(
                text='finished errors-plot-only: {}, id={}\n\n{}'.format(
                    model_name, model_id, desc),
                chat_id=config.CHAT_ID,
                files=files_to_send,
                text_for_files=caption)

    if 'plot_only' in options and options['plot_only']:
        stop = True
        batch = None
        for i, b in enumerate(dl_val):
            batch = b
            break
        model.epoch -= 1
        initial_print += '\nplotting ...'
        plot_filename = 'demo-plot_epoch-{}'.format(model.epoch)
        plot_filename = plot_filename + '_path-{}.pdf'
        plot_one_path_with_pred(
            device=device, model=model, batch=batch, delta_t=delta_t,
            T=batch["max_time"],
            path_to_plot=paths_to_plot, save_path=plot_save_path,
            filename=plot_filename,
            model_name=model_name, save_extras=save_extras, ylabels=ylabels,
            same_yaxis=plot_same_yaxis,
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
                text_for_files=caption)
        print(initial_print)
    if stop:
        return

    # ---------------- TRAINING ----------------
    skip_training = True
    if model.epoch <= epochs:
        skip_training = False

        # send notification
        if SEND:
            SBM.send_notification(
                text='start training - model id={}'.format(model_id),
                chat_id=config.CHAT_ID
            )
        initial_print += '\n\nmodel overview:'
        print(initial_print)
        print(model, '\n')

        # compute number of parameters
        nr_params = 0
        for name, param in model.named_parameters():
            skip = False
            for p_name in []:
                if p_name in name:
                    skip = True
            if not skip:
                nr_params += param.nelement()  # count number of parameters

        print('# parameters={}\n'.format(nr_params))
        print('start training ...')
    metric_app = []
    while model.epoch <= epochs:
        t = time.time()
        model.train()
        for i, b in tqdm.tqdm(enumerate(dl)):
            optimizer.zero_grad()
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
            n_obs_ot = b["n_obs_ot"]
            T = b["max_time"]
            pred_labels = b["predict_labels"]

            hT, loss = model(
                times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx,
                delta_t=delta_t, T=T, start_X=start_X, n_obs_ot=n_obs_ot,
                return_path=False, get_loss=True, M=M, start_M=start_M,
                dim_to=dim_to, predict_labels=pred_labels)
            if train_classifier:
                _loss, _mse_loss, _cl_loss = loss
                loss = _loss
            loss.backward()
            optimizer.step()
        train_time = time.time() - t

        # -------- evaluation --------
        t = time.time()
        batch = None
        with torch.no_grad():
            loss_val = 0
            mse_loss_val = 0
            cl_loss_val = 0
            num_obs = 0
            eval_msd = 0
            ref_eval_msd = 0
            eval_f1score = 0
            predicted_vals = np.zeros((len(data_val), dimension))
            true_vals = np.zeros((len(data_val), dimension))
            predicted_labels = np.zeros(len(data_val))
            true_labels = np.zeros(len(data_val))
            cbs = 0
            model.eval()
            for i, b in enumerate(dl_val):
                if plot and batch is None:
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
                bs = start_X.shape[0]
                obs_idx = b["obs_idx"]
                n_obs_ot = b["n_obs_ot"]
                T = b["max_time"]
                predict_times = b["predict_times"]
                predict_vals = b["predict_vals"]
                true_eval_labels = b["true_eval_labels"]
                samples = b["true_samples"]
                coord_to_compare = b["coord_to_compare"]
                pred_labels = b["predict_labels"]

                hT, c_loss = model(
                    times, time_ptr, X, obs_idx, delta_t, T, start_X,
                    n_obs_ot, return_path=False, get_loss=True, M=M,
                    start_M=start_M, which_loss='standard', dim_to=dim_to,
                    predict_labels=pred_labels)
                if train_classifier:
                    c_loss, c_mse_loss, c_cl_loss = c_loss
                    c_mse_loss = c_mse_loss.detach().numpy()
                    c_cl_loss = c_cl_loss.detach().numpy()
                else:
                    c_mse_loss, c_cl_loss = np.nan, np.nan
                loss_val += c_loss.detach().numpy()
                mse_loss_val += c_mse_loss
                cl_loss_val += c_cl_loss
                num_obs += 1  # count number of observations

                # mean squared error evaluation
                if 'evaluate' in options and options['evaluate']:
                    _eval_msd, _ref_eval_msd, _eval_f1score, _, _, _pred_vals, \
                    _pred_labels = model.evaluate_LOB(
                            times=times, time_ptr=time_ptr, X=X,
                            obs_idx=obs_idx, delta_t=delta_t, T=T,
                            start_X=start_X, n_obs_ot=n_obs_ot,
                            return_paths=True, predict_times=predict_times,
                            true_predict_vals=predict_vals,
                            coord_to_compare=coord_to_compare,
                            true_predict_labels=true_eval_labels,
                            true_samples=samples,
                            normalizing_mean=normalizing_price_mean,
                            normalizing_std=normalizing_price_std,
                            eval_predict_steps=eval_predict_steps,
                            thresholds=thresholds, predict_labels=pred_labels)
                    eval_msd += _eval_msd
                    ref_eval_msd += _ref_eval_msd
                    predicted_vals[cbs:cbs+bs] = _pred_vals
                    true_vals[cbs:cbs+bs] = predict_vals[:, :, 0]
                    true_labels[cbs:cbs+bs] = true_eval_labels[:, 0]
                    if model.classifier is not None:
                        predicted_labels[cbs:cbs+bs] = _pred_labels
                    cbs += bs
                    if _eval_f1score is not None:
                        eval_f1score += _eval_f1score

            eval_time = time.time() - t
            loss_val = loss_val / num_obs
            mse_loss_val = mse_loss_val / num_obs
            cl_loss_val = cl_loss_val / num_obs
            eval_msd = eval_msd / num_obs
            ref_eval_msd /= num_obs
            eval_f1score = eval_f1score / num_obs
            train_loss = loss.detach().numpy()
            print_str = "epoch {}, weight={:.5f}, train-loss={:.5f}, " \
                        "eval-loss={:.5f}, mse-eval-loss={:.5f}, " \
                        "classification-eval-loss={:.5f},".format(
                model.epoch, model.weight, train_loss, loss_val, mse_loss_val,
                cl_loss_val)
            print(print_str)

            # compute overall f1-score and classification report
            if 'evaluate' in options and options['evaluate'] and \
                    model.classifier is not None:
                eval_f1score = sklearn.metrics.f1_score(
                    true_labels, predicted_labels,
                    average="weighted")
                print("classification report \n",
                      sklearn.metrics.classification_report(
                          true_labels, predicted_labels))

        if 'evaluate' in options and options['evaluate']:
            metric_app.append([model.epoch, train_time, eval_time, train_loss,
                               loss_val, mse_loss_val, cl_loss_val,
                               eval_msd, ref_eval_msd, eval_f1score])
            eval_string = "evaluation mean square differences: k={}, " \
                          "eval_mse={:.5f}, (ref_eval_mse={:.5f})".format(
                eval_predict_steps[0], eval_msd, ref_eval_msd)
            eval_string_f1 = "evaluation f1 scores: k={}, " \
                             "eval_f1-score={:.5f}".format(
                eval_predict_steps[0], eval_f1score)
            print(eval_string)
            print(eval_string_f1)
            loss_to_compare = eval_msd
        else:
            metric_app.append([model.epoch, train_time, eval_time, train_loss,
                               loss_val, mse_loss_val, cl_loss_val])
            loss_to_compare = loss_val

        # save model
        if model.epoch % save_every == 0:
            if plot:
                print('plotting ...')
                plot_filename = 'epoch-{}'.format(model.epoch)
                plot_filename = plot_filename + '_path-{}.pdf'
                plot_one_path_with_pred(
                    device=device, model=model, batch=batch,
                    delta_t=delta_t, T=batch["max_time"],
                    path_to_plot=paths_to_plot, save_path=plot_save_path,
                    filename=plot_filename,
                    model_name=model_name, save_extras=save_extras,
                    ylabels=ylabels,
                    same_yaxis=plot_same_yaxis,
                    dataset_metadata=dataset_metadata)
            print('save model ...')
            df_m_app = pd.DataFrame(data=metric_app, columns=metr_columns)
            df_metric = pd.concat([df_metric, df_m_app], ignore_index=True)
            df_metric.to_csv(model_metric_file)
            models.save_checkpoint(model, optimizer, model_path_save_last,
                                   model.epoch)
            metric_app = []
            print('saved!')
        if loss_to_compare < best_eval_loss:
            print('save new best model: last-best-loss: {:.5f}, '
                  'new-best-loss: {:.5f}, epoch: {}'.format(
                best_eval_loss, loss_to_compare, model.epoch))
            df_m_app = pd.DataFrame(data=metric_app, columns=metr_columns)
            df_metric = pd.concat([df_metric, df_m_app], ignore_index=True)
            df_metric.to_csv(model_metric_file)
            models.save_checkpoint(model, optimizer, model_path_save_last,
                                   model.epoch)
            models.save_checkpoint(model, optimizer, model_path_save_best,
                                   model.epoch)
            metric_app = []
            best_eval_loss = loss_to_compare
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
            text='finished LOB training: {}, id={}\n\n{}'.format(
                model_name, model_id, desc),
            chat_id=config.CHAT_ID,
            files=files_to_send,
            text_for_files=caption)

    # delete model & free memory
    del model, dl, dl_val, data_train, data_val
    gc.collect()


def plot_one_path_with_pred(
        device, model, batch, delta_t, T,
        path_to_plot=(0,), save_path='', filename='plot_{}.pdf',
        model_name=None, ylabels=None,
        save_extras={'bbox_inches': 'tight', 'pad_inches': 0.01},
        same_yaxis=False, dataset_metadata=None, dims_to=1,
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
    :param model_name: str or None, name used for model in plots
    :param ylabels: None or list of str of same length as dimension of X
    :param save_extras: dict with extra options for saving plot
    :param same_yaxis: bool, whether to plot all coordinates with same range on
        y-axis
    :param dataset_metadata: needed if plot_obs_prob=true, the metadata of the
        used dataset to extract the observation probability
    :param dims_to: int or None, up to which coordinate to plot
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
    true_X = batch["true_samples"]
    bs, dim, time_steps = true_X.shape
    true_M = batch["true_mask"]
    path_t_true_X = batch["true_times"]
    predict_times = batch["predict_times"]
    predict_vals = batch["predict_vals"]
    if dims_to is not None:
        dim = dims_to

    model.eval()  # put model in evaluation mode
    res = model.get_pred(
        times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx, delta_t=delta_t,
        T=T, start_X=start_X, M=M, start_M=start_M)
    path_y_pred = res['pred'].detach().numpy()
    path_t_pred = res['pred_t']

    for i in path_to_plot:
        fig, axs = plt.subplots(dim)
        if dim == 1:
            axs = [axs]
        for j in range(dim):
            path_t_obs = path_t_true_X[i]
            # path_X_obs = true_X[i, j, :]

            axs[j].plot(path_t_obs, true_X[i, j, :], label='true path',
                        color=colors[0])
            # axs[j].scatter(path_t_obs, path_X_obs, label='observed',
            #                color=colors[0])
            axs[j].plot(path_t_pred, path_y_pred[:, i, j],
                        label=model_name, color=colors[1])
            axs[j].scatter(
                predict_times[i, :], predict_vals[i, j, :], label='eval points',
                color=colors[2])

            low = np.min(path_t_obs)
            high = np.max(predict_times[i, :])
            eps = (high - low)*0.05
            axs[j].set_xlim([low-eps, high+eps])
            last_ind = np.argmin(np.abs(path_t_pred - high))
            low_y = min(np.min(path_y_pred[:last_ind+1, i, j]),
                        np.min(predict_vals[i, j, :]))
            high_y = max(np.max(path_y_pred[:last_ind+1, i, j]),
                         np.max(predict_vals[i, j, :]))
            eps_y = (high_y - low_y)*0.05
            axs[j].set_ylim([low_y-eps_y, high_y+eps_y])

            if ylabels:
                axs[j].set_ylabel(ylabels[j])
            if same_yaxis:
                low = np.min(true_X[i, :, :])
                high = np.max(true_X[i, :, :])
                eps = (high - low)*0.05
                axs[j].set_ylim([low-eps, high+eps])

        axs[-1].legend()
        plt.xlabel('$t$')
        save = os.path.join(save_path, filename.format(i))
        plt.savefig(save, **save_extras)
        plt.close()



