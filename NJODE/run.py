"""
author: Florian Krach & Calypso Herrera

code for running the experiments
"""

# =====================================================================================================================
import numpy as np
import os
import pandas as pd
import json
import socket
import matplotlib
from joblib import Parallel, delayed
from absl import app
from absl import flags

from configs import config
import extras
from train_switcher import train_switcher

try:
    from telegram_notifications import send_bot_message as SBM
except Exception:
    from config import SendBotMessage as SBM


# =====================================================================================================================
# FLAGS
FLAGS = flags.FLAGS

flags.DEFINE_string("params", None, "name of the params list (in config.py) to "
                                    "use for parallel run")
flags.DEFINE_string("model_ids", None,
                    "name of list of model ids (in config.py) to run or list "
                    "of model ids")
flags.DEFINE_integer("first_id", None, "First id of the given list / "
                                       "to start training of")
flags.DEFINE_bool("DEBUG", False, "whether to run parallel in debug mode")
flags.DEFINE_string("saved_models_path", config.saved_models_path,
                    "path where the models are saved")
flags.DEFINE_string("overwrite_params", None,
                    "name of dict to use for overwriting params")
flags.DEFINE_string("get_overview", None,
                    "name of the dict (in config.py) defining input for "
                    "extras.get_training_overview")
flags.DEFINE_string("plot_paths", None,
                    "name of the dict (in config.py) defining input for "
                    "extras.plot_paths_from_checkpoint")
flags.DEFINE_string("crossval", None,
                    "name of the dict (in config.py) defining input for "
                    "extras.get_cross_validation")
flags.DEFINE_string("plot_conv_study", None,
                    "name of the dict (in config.py) defining input for "
                    "extras.plot_convergence_study")



flags.DEFINE_bool("USE_GPU", False, "whether to use GPU for training")
flags.DEFINE_bool("ANOMALY_DETECTION", False,
                  "whether to run in torch debug mode")
flags.DEFINE_integer("N_DATASET_WORKERS", 0,
                     "number of processes that generate batches in parallel")

# check whether running on computer or server
if 'ada-' not in socket.gethostname():
    SERVER = False
    flags.DEFINE_integer("NB_JOBS", 1,
                         "nb of parallel jobs to run  with joblib")
    flags.DEFINE_integer("NB_CPUS", 1, "nb of CPUs used by each training")
    flags.DEFINE_bool("SEND", False, "whether to send with telegram bot")
else:
    SERVER = True
    flags.DEFINE_integer("NB_JOBS", 24,
                         "nb of parallel jobs to run  with joblib")
    flags.DEFINE_integer("NB_CPUS", 2, "nb of CPUs used by each training")
    flags.DEFINE_bool("SEND", True, "whether to send with telegram bot")
    matplotlib.use('Agg')

print(socket.gethostname())
print('SERVER={}'.format(SERVER))


# =====================================================================================================================
# Functions
def get_parameter_array(param_dict):
    """
    helper function to get a list of parameter-list with all combinations of
    parameters specified in a parameter-dict

    :param param_dict: dict with parameters
    :return: 2d-array with the parameter combinations
    """
    return config.get_parameter_array(param_dict)


def parallel_training(params=None, model_ids=None, nb_jobs=1, first_id=None,
                      saved_models_path=config.saved_models_path,
                      overwrite_params=None):
    """
    function for parallel training, based on train.train
    :param params: a list of param_dicts, each dict corresponding to one model
            that should be trained, can be None if model_ids is given
            (then unused)
            all kwargs needed for train.train have to be in each dict
            -> giving the params together with first_id, they can be used to
                restart parallel training (however, the saved params for all
                models where the model_id already existed will be used instead
                of the params in this list, so that no unwanted errors are
                produced by mismatching. whenever a model_id didn't exist yet
                the params of the list are used to make a new one)
            -> giving params without first_id, all param_dicts will be used to
                initiate new models
    :param model_ids: list of ints, the model ids to use (only those for which a
            model was already initiated and its description was saved to the
            model_overview.csv file will be used)
            -> used to restart parallel training of certain model_ids after the
                training was stopped
    :param nb_jobs: int, the number of CPUs to use parallelly
    :param first_id: int or None, the model_id corresponding to the first
            element of params list
    :param saved_models_path: str, path to saved models
    :param overwrite_params: None or dict with key the param name to be
            overwritten and value the new value for this param. can bee  used to
            continue the training of a stored model, where some params should be
            changed (e.g. the number of epochs to train longer)
    :return:
    """
    if params is not None and 'saved_models_path' in params[0]:
        saved_models_path = params[0]['saved_models_path']
    model_overview_file_name = '{}model_overview.csv'.format(
        saved_models_path)
    config.makedirs(saved_models_path)
    if not os.path.exists(model_overview_file_name):
        df_overview = pd.DataFrame(data=None, columns=['id', 'description'])
        max_id = 0
    else:
        df_overview = pd.read_csv(model_overview_file_name, index_col=0)
        max_id = np.max(df_overview['id'].values)

    # get model_id, model params etc. for each param
    if model_ids is None and params is None:
        return 0
    if model_ids is None:  # if no model id is specified, start new model
        if first_id is None:
            model_id = max_id + 1
        else:
            model_id = first_id
        for i, param in enumerate(params):  # iterate through all specified parameter settings
            if model_id in df_overview['id'].values:  # resume training if taken id is specified as first model id
                desc = (df_overview['description'].loc[
                    df_overview['id'] == model_id]).values[0]
                params_dict = json.loads(desc)
                params_dict['resume_training'] = True
                params_dict['model_id'] = model_id
                if overwrite_params:
                    for k, v in overwrite_params.items():
                        params_dict[k] = v
                    desc = json.dumps(params_dict, sort_keys=True)
                    df_overview.loc[
                        df_overview['id'] == model_id, 'description'] = desc
                    df_overview.to_csv(model_overview_file_name)
                params[i] = params_dict
            else:  # if new model id, create new training
                desc = json.dumps(param, sort_keys=True)
                df_ov_app = pd.DataFrame([[model_id, desc]],
                                         columns=['id', 'description'])
                df_overview = pd.concat([df_overview, df_ov_app],
                                        ignore_index=True)
                df_overview.to_csv(model_overview_file_name)
                params_dict = json.loads(desc)
                params_dict['resume_training'] = False
                params_dict['model_id'] = model_id
                params[i] = params_dict
            model_id += 1
    else:
        params = []
        for model_id in model_ids:
            if model_id not in df_overview['id'].values:
                print("model_id={} does not exist yet -> skip".format(model_id))
            else:
                desc = (df_overview['description'].loc[
                    df_overview['id'] == model_id]).values[0]
                params_dict = json.loads(desc)
                params_dict['model_id'] = model_id
                params_dict['resume_training'] = True
                if overwrite_params:
                    for k, v in overwrite_params.items():
                        params_dict[k] = v
                    desc = json.dumps(params_dict, sort_keys=True)
                    df_overview.loc[
                        df_overview['id'] == model_id, 'description'] = desc
                    df_overview.to_csv(model_overview_file_name)
                params.append(params_dict)

    for param in params:
        param['parallel'] = True

    if FLAGS.SEND:
        SBM.send_notification(
            text='start parallel training - \nparams:'
                 '\n\n{}'.format(params),
            chat_id=config.CHAT_ID
        )

    if FLAGS.DEBUG:
        results = Parallel(n_jobs=nb_jobs)(delayed(train_switcher)(
            anomaly_detection=FLAGS.ANOMALY_DETECTION,
            n_dataset_workers=FLAGS.N_DATASET_WORKERS, use_gpu=FLAGS.USE_GPU,
            nb_cpus=FLAGS.NB_CPUS, send=FLAGS.SEND, **param)
                                           for param in params)
        if FLAGS.SEND:
            SBM.send_notification(
                text='finished parallel training - \nparams:'
                     '\n\n{}'.format(params),
                chat_id=config.CHAT_ID
            )
    else:
        try:
            results = Parallel(n_jobs=nb_jobs)(delayed(train_switcher)(
                anomaly_detection=FLAGS.ANOMALY_DETECTION,
                n_dataset_workers=FLAGS.N_DATASET_WORKERS, use_gpu=FLAGS.USE_GPU,
                nb_cpus=FLAGS.NB_CPUS, send=FLAGS.SEND, **param)
                                               for param in params)
            if FLAGS.SEND:
                SBM.send_notification(
                    text='finished parallel training - \nparams:'
                         '\n\n{}'.format(params),
                    chat_id=config.CHAT_ID
                )
        except Exception as e:
            if FLAGS.SEND:
                SBM.send_notification(
                    text='error in parallel training - \nerror:'
                         '\n\n{}'.format(e),
                    chat_id=config.ERROR_CHAT_ID
                )
            else:
                print('error:\n\n{}'.format(e))


def main(arg):
    """
    function to run parallel training with flags from command line
    """
    del arg
    params_list = None
    model_ids = None
    nb_jobs = FLAGS.NB_JOBS
    if FLAGS.params:
        params_list = eval("config."+FLAGS.params)
        nb_jobs = min(FLAGS.NB_JOBS, len(params_list))
        print('combinations: {}'.format(len(params_list)))
    elif FLAGS.model_ids:
        try:
            model_ids = eval("config."+FLAGS.model_ids)
        except Exception:
            model_ids = eval(FLAGS.model_ids)
        nb_jobs = min(FLAGS.NB_JOBS, len(model_ids))
        print('combinations: {}'.format(len(model_ids)))
    overwrite_params = None
    if FLAGS.overwrite_params:
        try:
            overwrite_params = eval("config."+FLAGS.overwrite_params)
        except Exception:
            overwrite_params = eval(FLAGS.overwrite_params)
    get_training_overview_dict = None
    if FLAGS.get_overview:
        get_training_overview_dict = eval("config."+FLAGS.get_overview)
    plot_paths_dict = None
    if FLAGS.plot_paths:
        plot_paths_dict = eval("config."+FLAGS.plot_paths)
    crossval = None
    if FLAGS.crossval:
        crossval = eval("config."+FLAGS.crossval)
    plot_conv_study = None
    if FLAGS.plot_conv_study:
        plot_conv_study = eval("config."+FLAGS.plot_conv_study)
    print('nb_jobs: {}'.format(nb_jobs))
    if params_list is not None or model_ids is not None:
        parallel_training(
            params=params_list, model_ids=model_ids,
            first_id=FLAGS.first_id, nb_jobs=nb_jobs,
            saved_models_path=FLAGS.saved_models_path,
            overwrite_params=overwrite_params
        )
    if get_training_overview_dict is not None:
        extras.get_training_overview(
            send=FLAGS.SEND, **get_training_overview_dict)
    if plot_paths_dict is not None:
        extras.plot_paths_from_checkpoint(
            send=FLAGS.SEND, **plot_paths_dict)
    if crossval is not None:
        extras.get_cross_validation(send=FLAGS.SEND, **crossval)
    if plot_conv_study is not None:
        extras.plot_convergence_study(send=FLAGS.SEND, **plot_conv_study)


if __name__ == '__main__':
    app.run(main)

