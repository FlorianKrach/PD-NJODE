"""
author: Florian Krach
"""

# =====================================================================================================================

import torch  # machine learning
import tqdm  # process bar for iterations
import numpy as np  # large arrays and matrices, functions
from torch.utils.data import DataLoader
import torch.utils.data as tdata
from sklearn.model_selection import train_test_split
import os, sys
import pandas as pd  # data analysis and manipulation
import json  # storing and exchanging data
import time
import socket
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
print(socket.gethostname())
print('SERVER={}'.format(SERVER))


# ==============================================================================
# Global variables
CHAT_ID = config.CHAT_ID
ERROR_CHAT_ID = config.ERROR_CHAT_ID

data_path = config.data_path
saved_models_path = "{}saved_models_LOB/".format(data_path)
flagfile = config.flagfile

METR_COLUMNS = ['epoch', 'train_time', 'eval_time', 'train_loss',
                'classification_eval_loss']

default_ode_nn = ((50, 'tanh'), (50, 'tanh'))
default_readout_nn = ((50, 'tanh'), (50, 'tanh'))
default_enc_nn = ((50, 'tanh'), (50, 'tanh'))

ANOMALY_DETECTION = False
N_DATASET_WORKERS = 0
USE_GPU = False


# =====================================================================================================================
# Functions
makedirs = config.makedirs


def get_dataset_and_model(load_model_id, saved_models_path, device, load_best,
                          new_classifier_nn):
    # get all needed paths
    model_path = '{}id-{}/'.format(saved_models_path, load_model_id)
    model_path_save_last = '{}last_checkpoint/'.format(model_path)
    model_path_save_best = '{}best_checkpoint/'.format(model_path)
    poststring = "best" if load_best else "last"
    save_dataset_path = "{}retrain_dataset_at_{}/".format(model_path, poststring)
    file_name = "{}data.npy".format(save_dataset_path)

    # load the params_dict of pretrained saved model
    model_overview_file_name = '{}model_overview.csv'.format(
        saved_models_path)
    df_overview = pd.read_csv(model_overview_file_name, index_col=0)
    if load_model_id not in df_overview['id'].values:
        raise ValueError("please provide a model_id of an existing model")
    desc = (df_overview['description'].loc[
        df_overview['id'] == load_model_id]).values[0]
    params_dict = json.loads(desc)

    # load datasets metadata
    if "data_dict" in params_dict:
        dataset, dataset_id = data_utils._get_dataset_name_id_from_dict(
            data_dict=params_dict["data_dict"])
        dataset_id = int(dataset_id)
    else:
        dataset = "LOB"
        dataset_id = int(data_utils._get_time_id(
            stock_model_name=dataset, time_id=params_dict["dataset_id"]))
    dataset_metadata = data_utils.load_metadata(
        stock_model_name=dataset, time_id=dataset_id)
    use_volume = dataset_metadata['use_volume']
    input_size = dataset_metadata['LOB_level']*2*(1+use_volume) + 1
    output_size = input_size
    delta_t = dataset_metadata['dt']
    dim_to = None
    if params_dict["output_midprice_only"]:
        dim_to = 1
    train_classifier = False
    if params_dict["classifier_nn"] is not None:
        train_classifier = True
        classifier_dict = {
            "nn_desc": params_dict["classifier_nn"],
            'input_size': params_dict["hidden_size"],
            'output_size': 3, 'dropout_rate': params_dict["dropout_rate"],
            'bias': params_dict["bias"],
            'residual': False}
        params_dict["classifier_dict"] = classifier_dict
    if "options" not in params_dict:
        params_dict["options"] = {}
    use_sig_for_classifier = False
    if ("use_sig_for_classifier" in params_dict["options"] and
        params_dict["options"]["use_sig_for_classifier"]) or \
            ("use_sig_for_classifier" in params_dict and
             params_dict["use_sig_for_classifier"]):
        use_sig_for_classifier = True
    params_dict["input_size"] = input_size
    params_dict["output_size"] = output_size
    params_dict["options"] = params_dict

    model = models.NJODE(**params_dict)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params_dict["learning_rate"],
        weight_decay=0.0005)

    if load_best:
        models.get_ckpt_model(model_path_save_best, model, optimizer, device)
    else:
        models.get_ckpt_model(model_path_save_last, model, optimizer, device)

    # check whether dataset already exists
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            train_samples = np.load(f)
            train_labels = np.load(f)
            test_samples = np.load(f)
            test_labels = np.load(f)
        print("loaded retrain dataset at: {}".format(file_name))
    # otherwise generate the dataset
    else:
        print("retrain dataset does not exist -> generate ...")
        # load raw data
        train_idx, val_idx = train_test_split(
            np.arange(dataset_metadata["nb_paths"]),
            test_size=params_dict["test_size"],
            random_state=params_dict["seed"], shuffle=False)
        # --> get subset of training samples if wanted
        if 'training_size' in params_dict["options"]:
            train_set_size = params_dict["options"]['training_size']
            if train_set_size < len(train_idx):
                train_idx = np.random.choice(
                    train_idx, train_set_size, replace=False)
        data_train = data_utils.LOBDataset(time_id=dataset_id, idx=train_idx)
        data_val = data_utils.LOBDataset(time_id=dataset_id, idx=val_idx)

        # get data-loader
        collate_fn = data_utils.LOBCollateFnGen(
            data_type="train",
            use_eval_on_train=params_dict["use_eval_on_train"],
            train_classifier=train_classifier)
        collate_fn_val = data_utils.LOBCollateFnGen(data_type="test")

        dl = DataLoader(
            dataset=data_train, collate_fn=collate_fn,
            shuffle=False, batch_size=params_dict["batch_size"],
            num_workers=N_DATASET_WORKERS)
        dl_val = DataLoader(
            dataset=data_val, collate_fn=collate_fn_val,
            shuffle=False, batch_size=params_dict["batch_size"],
            num_workers=N_DATASET_WORKERS)

        # train dataset
        hs = []
        sigs = []
        labels = []
        model.eval()
        for i, b in tqdm.tqdm(enumerate(dl)):
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
            h_at_last_obs, sig_at_last_obs = model(
                times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx,
                delta_t=delta_t, T=T, start_X=start_X, n_obs_ot=n_obs_ot,
                return_path=False, get_loss=False, M=M, start_M=start_M,
                dim_to=dim_to, predict_labels=None, until_T=False,
                return_at_last_obs=True)
            hs.append(h_at_last_obs.detach().numpy())
            if sig_at_last_obs is not None:
                sigs.append(sig_at_last_obs.detach().numpy())
            labels.append(pred_labels[:,0].detach().numpy())
        train_samples = np.concatenate(hs, axis=0)
        if len(sigs) > 0 and use_sig_for_classifier:
            sig_samples = np.concatenate(sigs, axis=0)
            train_samples = np.concatenate([train_samples, sig_samples], axis=1)
        train_labels = np.concatenate(labels, axis=0)

        # test dataset
        hs = []
        sigs = []
        labels = []
        model.eval()
        for i, b in tqdm.tqdm(enumerate(dl_val)):
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
            h_at_last_obs, sig_at_last_obs = model(
                times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx,
                delta_t=delta_t, T=T, start_X=start_X, n_obs_ot=n_obs_ot,
                return_path=False, get_loss=False, M=M, start_M=start_M,
                dim_to=dim_to, predict_labels=None, until_T=False,
                return_at_last_obs=True)
            hs.append(h_at_last_obs.detach().numpy())
            if sig_at_last_obs is not None:
                sigs.append(sig_at_last_obs.detach().numpy())
            labels.append(pred_labels[:,0].detach().numpy())
        test_samples = np.concatenate(hs, axis=0)
        if len(sigs) > 0 and use_sig_for_classifier:
            sig_samples = np.concatenate(sigs, axis=0)
            test_samples = np.concatenate([test_samples, sig_samples], axis=1)
        test_labels = np.concatenate(labels, axis=0)

        # save the dataset
        makedirs(save_dataset_path)
        with open(file_name, 'wb') as f:
            np.save(f, train_samples)
            np.save(f, train_labels)
            np.save(f, test_samples)
            np.save(f, test_labels)
        print("saved retrain dataset at: {}".format(file_name))

    # if wanted, replace the classifier NN
    if new_classifier_nn is not None:
        classifier_dict = {
            "nn_desc": new_classifier_nn["nn_desc"],
            'input_size': params_dict["hidden_size"],
            'output_size': 3,
            'dropout_rate': new_classifier_nn["dropout_rate"],
            'bias': new_classifier_nn["bias"],
            'residual': False}
        params_dict["classifier_dict"] = classifier_dict
        model.get_classifier(classifier_dict)
        model.classifier.apply(models.init_weights)

    return model, params_dict, train_samples, train_labels, test_samples, \
           test_labels



def train(
        anomaly_detection=None, n_dataset_workers=None, use_gpu=None,
        nb_cpus=None, send=None,
        model_id=None, saved_models_path=saved_models_path,
        load_model_id=None, load_saved_models_path=saved_models_path,
        epochs=100, batch_size=100, save_every=1, learning_rate=None,
        load_model_load_best=True, new_classifier_nn=None,
        **options
):
    """
    retrain the classifier of an NJODE model (with classifier) for LOB data

    Args:
        anomaly_detection:  used to pass on FLAG from parallel_train
        n_dataset_workers:  used to pass on FLAG from parallel_train
        use_gpu:            used to pass on FLAG from parallel_train
        nb_cpus:            used to pass on FLAG from parallel_train
        send:               used to pass on FLAG from parallel_train
        model_id:           None or int, the id to save (or load if it already
                            exists) the model, if None: next biggest unused id
                            will be used
        saved_models_path:  str, where to save the models
        load_model_id:      int, the id of the previously trained and savd
                            model, which should be loaded to retrain its
                            classifier
        load_saved_models_path: str, where the pretrained model is saved
        epochs:             int, number of epochs to retrain the model
                            classifier
        batch_size:         int, batch size for retraining
        save_every:         int, see train.py
        learning_rate:      float or None, learning rate for retraining, if
                            None: the learning rate of the pretrained model is
                            used
        load_model_load_best:   bool, whether to load best or last state of
                            pretrained model
        new_classifier_nn:  dict or None, possibility to train a new
                            classifier NN on top of the trained model, the dict
                            needs the keys: 'nn_desc', 'dropout_rate', 'bias'
        **options: kwargs, supported keywords are:
            load_best:      bool, whether to load best (or last) saved model,
                            default: False
            evaluate:       bool, whether to evaluate the model (via f1-score),
                            default: False
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

    initial_print = "load-model-id: {}\n".format(load_model_id)

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

    # get model, optimizer and raw data
    model, params_dict, \
    train_samples, train_labels, test_samples, test_labels = \
        get_dataset_and_model(
            load_model_id, load_saved_models_path, device, load_model_load_best,
            new_classifier_nn)
    train_samples = torch.from_numpy(train_samples).float()
    train_labels = torch.from_numpy(train_labels).long()
    test_samples = torch.from_numpy(test_samples).float()
    test_labels = torch.from_numpy(test_labels).long()
    if learning_rate is None:
        learning_rate = params_dict["learning_rate"]
    assert model.classifier is not None, \
        "only model with a classifier can be retrained"
    optimizer = torch.optim.Adam(
        model.classifier.parameters(), lr=learning_rate, weight_decay=0.0005)

    # get dataloader
    batch_sampler = tdata.BatchSampler(
        tdata.RandomSampler(range(len(train_samples)), replacement=False),
        batch_size=batch_size, drop_last=False)
    # test_batch_sampler = tdata.BatchSampler(
    #     tdata.SequentialSampler(range(len(test_samples))),
    #     batch_size=batch_size, drop_last=False)

    # get params_dict
    params_dict = {  # create a dictionary of the wanted parameters
        "load_model_id": load_model_id,
        "load_saved_models_path": load_saved_models_path,
        'epochs': epochs, 'batch_size': batch_size,
        'learning_rate': learning_rate, 'load_best': load_model_load_best,
        'new_classifier_nn': new_classifier_nn,
        'options': options}
    desc = json.dumps(params_dict, sort_keys=True)

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
    initial_print += '\nmodel-id: {}'.format(model_id)
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

    # load saved model if wanted/possible
    best_eval_loss = np.infty
    if 'evaluate' in options and options['evaluate']:
        metr_columns = METR_COLUMNS + ['evaluation_f1score']
    else:
        metr_columns = METR_COLUMNS
    if resume_training:
        initial_print += '\nload saved model ...'
        try:
            if 'load_best' in options and options['load_best']:
                models.get_ckpt_model(
                    model_path_save_best, model, optimizer, device)
            else:
                models.get_ckpt_model(
                    model_path_save_last, model, optimizer, device)
            df_metric = pd.read_csv(model_metric_file, index_col=0)
            best_eval_loss = np.min(
                df_metric['classification_eval_loss'].values)
            model.retrain_epoch += 1
            model.weight_decay_step()
            initial_print += '\nepoch: {}'.format(model.retrain_epoch)
        except Exception as e:
            initial_print += '\nloading retrained model failed -> ' \
                             'start from pretrained model'
            initial_print += '\nException:\n{}'.format(e)
            resume_training = False
            model.retrain_epoch = 1
    if not resume_training:
        initial_print += '\ninitiate new model ...'
        df_metric = pd.DataFrame(columns=metr_columns)

    # ---------------- TRAINING ----------------
    skip_training = True
    if model.retrain_epoch <= epochs:
        skip_training = False
        if SEND:
            SBM.send_notification(
                text='start retraining classifier of load-model-id {} '
                     '- model id={}'.format(load_model_id, model_id),
                chat_id=config.CHAT_ID)
        initial_print += '\n\nmodel overview (classifier network):'
        print(initial_print)
        print(model.classifier, '\n')

        # compute number of parameters
        nr_params = 0
        for name, param in model.classifier.named_parameters():
            skip = False
            for p_name in []:
                if p_name in name:
                    skip = True
            if not skip:
                nr_params += param.nelement()  # count number of parameters

        print('# parameters={}\n'.format(nr_params))
        print('start training ...')
    metric_app = []
    while model.retrain_epoch <= epochs:
        t = time.time()
        model.train()
        for b in tqdm.tqdm(batch_sampler):
            optimizer.zero_grad()
            x = train_samples[b]
            y = train_labels[b]
            loss, _ = model.forward_classifier(x,y)
            loss.backward()
            optimizer.step()
        train_time = time.time() - t

        # -------- evaluation --------
        t = time.time()
        batch = None
        with torch.no_grad():
            model.eval()
            x = test_samples
            y = test_labels
            cl_loss, cl_out = model.forward_classifier(x,y)
            cl_loss_val = cl_loss.detach().numpy()

            # compute overall f1-score and classification report
            if 'evaluate' in options and options['evaluate']:
                class_probs = model.SM(cl_out).detach().numpy()
                classes = np.argmax(class_probs, axis=1)
                eval_f1score = sklearn.metrics.f1_score(
                    y, classes, average="weighted")
                cl_report = sklearn.metrics.classification_report(y, classes)

        eval_time = time.time() - t
        train_loss = loss.detach().numpy()
        print_str = "epoch {}, weight={:.5f}, train-loss={:.5f}, " \
                    "classification-eval-loss={:.5f},".format(
            model.retrain_epoch, model.weight, train_loss, cl_loss_val)
        print(print_str)

        if 'evaluate' in options and options['evaluate']:
            metric_app.append(
                [model.retrain_epoch, train_time, eval_time, train_loss,
                 cl_loss_val, eval_f1score])
            print("eval_f1-score={:.5f}".format(eval_f1score))
            print("classification report \n", cl_report)
        else:
            metric_app.append([model.retrain_epoch, train_time, eval_time,
                               train_loss, cl_loss_val])
        # save model
        if model.retrain_epoch % save_every == 0:
            print('save model ...')
            df_m_app = pd.DataFrame(data=metric_app, columns=metr_columns)
            df_metric = pd.concat([df_metric, df_m_app], ignore_index=True)
            df_metric.to_csv(model_metric_file)
            models.save_checkpoint(model, optimizer, model_path_save_last,
                                   model.epoch, model.retrain_epoch)
            metric_app = []
            print('saved!')
        if cl_loss_val < best_eval_loss:
            print('save new best model: last-best-loss: {:.5f}, '
                  'new-best-loss: {:.5f}, epoch: {}'.format(
                best_eval_loss, cl_loss_val, model.retrain_epoch))
            df_m_app = pd.DataFrame(data=metric_app, columns=metr_columns)
            df_metric = pd.concat([df_metric, df_m_app], ignore_index=True)
            df_metric.to_csv(model_metric_file)
            models.save_checkpoint(model, optimizer, model_path_save_last,
                                   model.epoch, model.retrain_epoch)
            models.save_checkpoint(model, optimizer, model_path_save_best,
                                   model.epoch, model.retrain_epoch)
            metric_app = []
            best_eval_loss = cl_loss_val
            print('saved!')
        print("-"*100)
        model.retrain_epoch += 1

    # send notification
    if SEND and not skip_training:
        files_to_send = [model_metric_file]
        caption = "{} - id={}".format("retrained-NJODE-classifier", model_id)
        SBM.send_notification(
            text='finished retraining on LOB: id={}, '
                 'load-model-id={}\n\n{}'.format(model_id, load_model_id, desc),
            chat_id=config.CHAT_ID,
            files=files_to_send,
            text_for_files=caption)

    # delete model & free memory
    del model, train_samples, train_labels, test_samples, test_labels
    gc.collect()






if __name__ == '__main__':
    pass
