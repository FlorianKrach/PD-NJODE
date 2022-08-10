"""
author: Florian Krach
"""

from config_NJODE1 import *

import numpy as np
import torch
import os
import pandas as pd
from sklearn.model_selection import ParameterGrid
import socket

if 'ada-' not in socket.gethostname():
    SERVER = False
else:
    SERVER = True

# ==============================================================================
# Global variables
CHAT_ID = "XXX"
ERROR_CHAT_ID = "XXX"

data_path = '../data/'
training_data_path = '{}training_data/'.format(data_path)
LOB_data_path = '{}LOB-raw_data/'.format(training_data_path)
LOB_data_path2 = '{}LOB-raw_data2/'.format(training_data_path)
LOB2_data_path = '{}LOB2-raw_data/'.format(training_data_path)
saved_models_path = '{}saved_models/'.format(data_path)
flagfile = "{}flagfile.tmp".format(data_path)

# ==============================================================================
# GLOBAL CLASSES
class SendBotMessage:
    def __init__(self):
        pass

    @staticmethod
    def send_notification(text, *args, **kwargs):
        print(text)


# ==============================================================================
# FUNCTIONS
def get_parameter_array(param_dict):
    """
    helper function to get a list of parameter-list with all combinations of
    parameters specified in a parameter-dict

    :param param_dict: dict with parameters
    :return: 2d-array with the parameter combinations
    """
    param_combs_dict_list = list(ParameterGrid(param_dict))
    return param_combs_dict_list


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)



# ==============================================================================
# DATASET DICTS
# ------------------------------------------------------------------------------
poisson_pp_dict = {
    'model_name': 'PoissonPointProcess',
    'poisson_lambda': 3.,
    'nb_paths': 20000, 'nb_steps': 100,
    'S0': 0, 'maturity': 1., 'dimension': 1,
    'obs_perc': 0.1,
    'scheme': 'euler', 'return_vol': False
}

# ------------------------------------------------------------------------------
FBM_1_dict = {
    'model_name': "FBM",
    'nb_paths': 100000, 'nb_steps': 100,
    'S0': 0, 'maturity': 1., 'dimension': 1,
    'obs_perc': 0.1,
    'return_vol': False, 'hurst': 0.05,
    'FBMmethod': "daviesharte"
}
FBM_2_dict = {
    'model_name': "FBM",
    'nb_paths': 100000, 'nb_steps': 100,
    'S0': 0, 'maturity': 1., 'dimension': 1,
    'obs_perc': 0.1,
    'return_vol': False, 'hurst': 0.25,
    'FBMmethod': "daviesharte"
}
FBM_3_dict = {
    'model_name': "FBM",
    'nb_paths': 100000, 'nb_steps': 100,
    'S0': 0, 'maturity': 1., 'dimension': 1,
    'obs_perc': 0.1,
    'return_vol': False, 'hurst': 0.75,
    'FBMmethod': "daviesharte"
}
FBM_4_dict = {
    'model_name': "FBM",
    'nb_paths': 100000, 'nb_steps': 100,
    'S0': 0, 'maturity': 1., 'dimension': 1,
    'obs_perc': 0.1,
    'return_vol': False, 'hurst': 0.95,
    'FBMmethod': "daviesharte"
}

# ------------------------------------------------------------------------------
BM_2D_dict = {
    'nb_paths': 20000, 'nb_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'alpha_sq': 0.9, 'masked': 0.,
    'dimension': 2,
}
BM_2D_dict_2 = {
    'nb_paths': 20000, 'nb_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'alpha_sq': 0.5, 'masked': 0.25,
    'dimension': 2,
}

# ------------------------------------------------------------------------------
BM_and_Var_dict = {
    'nb_paths': 20000, 'nb_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'dimension': 2,
}
BM_dict = {
    'nb_paths': 20000, 'nb_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'dimension': 1,
}

# ------------------------------------------------------------------------------
BS_dep_intensity_dict = {
    'drift': 2., 'volatility': 0.3,
    'nb_paths': 20000, 'nb_steps': 100,
    'S0': 1, 'maturity': 1., 'dimension': 1,
    'obs_perc': 0.1,
    'scheme': 'euler', 'return_vol': False,
    'X_dependent_observation_prob':
        "lambda x: 0.05 + 0.4 * np.tanh(np.abs(np.mean(x, axis=1))/10)"
}

# ------------------------------------------------------------------------------
LOB_dict1 = {
    'model_name': "LOB",
    'LOB_level': 10, 'amount_obs': 100, 'eval_predict_steps': [10,],
    'use_volume': True, 'normalize': True, 'shift': 110,
}

LOB_dict1_2 = {
    'model_name': "LOB", "start_at_0": False,
    'LOB_level': 10, 'amount_obs': 100, 'eval_predict_steps': [10,],
    'use_volume': True, 'normalize': True, 'shift': 110,
}

LOB_dict2 = {
    'model_name': "LOB",
    'LOB_level': 10, 'amount_obs': 100, 'eval_predict_steps': [10,],
    'use_volume': True, 'normalize': False, 'shift': 110,
}

LOB_dict3 = {
    'model_name': "LOB",
    'LOB_level': 10, 'amount_obs': 100, 'eval_predict_steps': [10,],
    'use_volume': False, 'normalize': False, 'shift': 110,
}

LOB_dict3_2 = {
    'model_name': "LOB", "start_at_0": False,
    'LOB_level': 10, 'amount_obs': 100, 'eval_predict_steps': [10,],
    'use_volume': False, 'normalize': False, 'shift': 110,
}

LOB_dict_K_1 = {
    'model_name': "LOB", "which_raw_data": "BTC_1sec",
    'LOB_level': 10, 'amount_obs': 100, 'eval_predict_steps': [10,],
    'use_volume': True, 'normalize': True, 'shift': 110,
}

LOB_dict_K_1_2 = {
    'model_name': "LOB", "which_raw_data": "BTC_1sec",
    "start_at_0": False,
    'LOB_level': 10, 'amount_obs': 100, 'eval_predict_steps': [10,],
    'use_volume': True, 'normalize': True, 'shift': 110,
}

LOB_dict_K_2 = {
    'model_name': "LOB", "which_raw_data": "BTC_1sec",
    'LOB_level': 10, 'amount_obs': 100, 'eval_predict_steps': [10,],
    'use_volume': True, 'normalize': False, 'shift': 110,
}

LOB_dict_K_3 = {
    'model_name': "LOB", "which_raw_data": "BTC_1sec",
    'LOB_level': 10, 'amount_obs': 100, 'eval_predict_steps': [10,],
    'use_volume': False, 'normalize': False, 'shift': 110,
}

LOB_dict_K_3_2 = {
    'model_name': "LOB", "which_raw_data": "BTC_1sec",
    "start_at_0": False,
    'LOB_level': 10, 'amount_obs': 100, 'eval_predict_steps': [10,],
    'use_volume': False, 'normalize': False, 'shift': 110,
}

LOB_dict_K_4 = {
    'model_name': "LOB", "which_raw_data": "ETH_1sec",
    'LOB_level': 10, 'amount_obs': 100, 'eval_predict_steps': [10,],
    'use_volume': True, 'normalize': True, 'shift': 110,
}

LOB_dict_K_4_2 = {
    'model_name': "LOB", "which_raw_data": "ETH_1sec",
    "start_at_0": False,
    'LOB_level': 10, 'amount_obs': 100, 'eval_predict_steps': [10,],
    'use_volume': True, 'normalize': True, 'shift': 110,
}

LOB_dict_K_6 = {
    'model_name': "LOB", "which_raw_data": "ETH_1sec",
    'LOB_level': 10, 'amount_obs': 100, 'eval_predict_steps': [10,],
    'use_volume': False, 'normalize': False, 'shift': 110,
}

LOB_dict_K_6_2 = {
    'model_name': "LOB", "which_raw_data": "ETH_1sec",
    "start_at_0": False,
    'LOB_level': 10, 'amount_obs': 100, 'eval_predict_steps': [10,],
    'use_volume': False, 'normalize': False, 'shift': 110,
}



# ==============================================================================
# TRAINING PARAM DICTS
ode_nn = ((50, 'tanh'), (50, 'tanh'))
readout_nn = ((50, 'tanh'), (50, 'tanh'))
enc_nn = ((50, 'tanh'), (50, 'tanh'))

# ------------------------------------------------------------------------------
# --- Poisson Point Process
PPP_models_path = "{}saved_models_PPP/".format(data_path)
param_dict_poissonpp1 = {
    'epochs': [200],
    'batch_size': [200],
    'save_every': [5],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [10],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [ode_nn],
    'readout_nn': [readout_nn],
    'enc_nn': [enc_nn],
    'use_rnn': [False],
    'func_appl_X': [[]],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'data_dict': ['poisson_pp_dict'],
    'plot': [True],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'saved_models_path': [PPP_models_path],
}
param_list_poissonpp1 = get_parameter_array(param_dict=param_dict_poissonpp1)

plot_paths_ppp_dict = {
    'model_ids': [2,], 'saved_models_path': PPP_models_path, 'which': 'best',
    'paths_to_plot': [3,],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}

# ------------------------------------------------------------------------------
# --- Fractional Brownian Motion
FBM_models_path = "{}saved_models_FBM/".format(data_path)
param_list_FBM1 = []
param_dict_FBM1_1 = {
    'epochs': [200],
    'batch_size': [200],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [10, 50],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [ode_nn],
    'readout_nn': [readout_nn],
    'enc_nn': [enc_nn],
    'use_rnn': [False],
    'func_appl_X': [[]],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'data_dict': ['FBM_1_dict'],
    'plot': [True],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'saved_models_path': [FBM_models_path],
}
param_list_FBM1 += get_parameter_array(param_dict=param_dict_FBM1_1)

for size in [50, 200]:
    _nn = ((size, 'tanh'), (size, 'tanh'))
    param_dict_FBM1_2 = {
        'epochs': [200],
        'batch_size': [50, 200],
        'save_every': [1],
        'learning_rate': [0.001],
        'test_size': [0.2],
        'seed': [398],
        'hidden_size': [10, 50],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [_nn],
        'readout_nn': [_nn],
        'enc_nn': [_nn],
        'use_rnn': [False],
        'func_appl_X': [[]],
        'solver': ["euler"],
        'weight': [0.5],
        'weight_decay': [1.],
        'input_sig': [True],
        'level': [2, 3],
        'data_dict': ['FBM_1_dict'],
        'which_loss': ['standard', 'easy'],
        'plot': [True],
        'evaluate': [True],
        'paths_to_plot': [(0,1,2,3,4,)],
        'saved_models_path': [FBM_models_path],
    }
    param_list_FBM1 += get_parameter_array(param_dict=param_dict_FBM1_2)

for _nn in [((200, 'tanh'), (200, 'tanh')), ]:
    param_dict_FBM1_3 = {
        'epochs': [200],
        'batch_size': [200],
        'save_every': [1],
        'learning_rate': [0.001],
        'test_size': [0.2],
        'seed': [398],
        'hidden_size': [50, 100],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [_nn],
        'readout_nn': [_nn, None],
        'enc_nn': [_nn],
        'use_rnn': [True,],
        'func_appl_X': [[]],
        'solver': ["euler"],
        'weight': [0.5],
        'weight_decay': [1.],
        'input_sig': [True, False],
        'level': [3],
        'data_dict': ['FBM_1_dict'],
        'which_loss': ['easy'],
        'plot': [True],
        'evaluate': [True],
        'paths_to_plot': [(0,1,2,3,4,)],
        'saved_models_path': [FBM_models_path],
    }
    param_list_FBM1 += get_parameter_array(param_dict=param_dict_FBM1_3)


for _nn in [((200, 'tanh'), (200, 'tanh')), ]:
    param_dict_FBM1_4 = {
        'epochs': [200],
        'batch_size': [200],
        'save_every': [1],
        'learning_rate': [0.001],
        'test_size': [0.2],
        'seed': [398],
        'hidden_size': [50, 100],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [_nn],
        'readout_nn': [_nn],
        'enc_nn': [_nn],
        'use_rnn': [False,],
        'func_appl_X': [[]],
        'solver': ["euler"],
        'weight': [0.5],
        'weight_decay': [1.],
        'input_sig': [False],
        'level': [3],
        'data_dict': ['FBM_1_dict'],
        'which_loss': ['easy'],
        'plot': [True],
        'evaluate': [True],
        'paths_to_plot': [(0,1,2,3,4,)],
        'saved_models_path': [FBM_models_path],
    }
    param_list_FBM1 += get_parameter_array(param_dict=param_dict_FBM1_4)

for _nn in [((200, 'tanh'), (200, 'tanh')), ]:
    param_dict_FBM1_5 = {
        'epochs': [200],
        'batch_size': [200],
        'save_every': [1],
        'learning_rate': [0.001],
        'test_size': [0.2],
        'seed': [398],
        'hidden_size': [50,],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [_nn],
        'readout_nn': [_nn],
        'enc_nn': [_nn],
        'use_rnn': [True,],
        'func_appl_X': [[]],
        'solver': ["euler"],
        'weight': [0.5],
        'weight_decay': [1.],
        'input_sig': [True],
        'level': [1,2,4,5,6,7,8,9,10],
        'data_dict': ['FBM_1_dict'],
        'which_loss': ['easy'],
        'plot': [True],
        'evaluate': [True],
        'paths_to_plot': [(0,1,2,3,4,)],
        'saved_models_path': [FBM_models_path],
    }
    param_list_FBM2 = get_parameter_array(param_dict=param_dict_FBM1_5)
    param_list_FBM1 += get_parameter_array(param_dict=param_dict_FBM1_5)


overview_dict_FBM1 = dict(
    ids_from=1, ids_to=len(param_list_FBM1),
    path=FBM_models_path,
    params_extract_desc=('dataset', 'network_size', 'nb_layers',
                         'activation_function_1', 'use_rnn',
                         'readout_nn', 'dropout_rate',
                         'hidden_size', 'batch_size', 'which_loss',
                         'input_sig', 'level'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "evaluation_mean_diff",
         "evaluation_mean_diff", "evaluation_mean_diff_min"),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
    ),
    sortby=["evaluation_mean_diff_min"],
)

plot_paths_FBM_dict = {
    'model_ids': [33, 34, 35, 41, 43, 50],
    'saved_models_path': FBM_models_path,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4,5],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}

# ------------------------------------------------------------------------------
# --- 2d Brownian Motion with correlation and correct cond. exp.
BM2D_models_path = "{}saved_models_BM2D/".format(data_path)
param_list_BM2D_1 = []

for size in [100]:
    for act in ['tanh', 'relu']:
        _nn = ((size, act),)
        param_dict_BM2D_1_2 = {
            'epochs': [200],
            'batch_size': [200],
            'save_every': [1],
            'learning_rate': [0.001],
            'test_size': [0.2],
            'seed': [398],
            'hidden_size': [size,],
            'bias': [True],
            'dropout_rate': [0.1],
            'ode_nn': [_nn],
            'readout_nn': [_nn, None],
            'enc_nn': [_nn],
            'use_rnn': [True],
            'func_appl_X': [[]],
            'solver': ["euler"],
            'weight': [0.5],
            'weight_decay': [1.],
            'input_sig': [True],
            'level': [2, ],
            'dataset': ["BM2DCorr", ],
            'dataset_id': [None],
            'which_loss': ['easy',],
            'coord_wise_tau': [False,],
            'use_y_for_ode': [True],
            'masked': [True],
            'plot': [True],
            'evaluate': [True],
            'paths_to_plot': [(0,1,2,3,4,)],
            'plot_same_yaxis': [True],
            'saved_models_path': [BM2D_models_path],
        }
        param_list_BM2D_1 += get_parameter_array(param_dict=param_dict_BM2D_1_2)

overview_dict_BM2D_1 = dict(
    ids_from=1, ids_to=len(param_list_BM2D_1),
    path=BM2D_models_path,
    params_extract_desc=('dataset', 'network_size', 'readout_nn',
                         'activation_function_1',
                         'hidden_size', 'batch_size', 'which_loss',
                         'input_sig', 'level', 'coord_wise_tau',
                         'use_y_for_ode'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "evaluation_mean_diff",
         "evaluation_mean_diff", "evaluation_mean_diff_min"),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
    ),
    sortby=["evaluation_mean_diff_min"],
)

plot_paths_BMV2D_dict = {
    'model_ids': [1,2,3,4], 'saved_models_path': BM2D_models_path,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4,5],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}



# ------------------------------------------------------------------------------
# --- BM and Var
BMandVar_models_path = "{}saved_models_BMandVar/".format(data_path)
_nn = ((50, 'tanh'),)
param_dict_BMandVar_1 = {
    'epochs': [200],
    'batch_size': [200],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [50],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'readout_nn': [None],
    'enc_nn': [_nn],
    'func_appl_X': [[]],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'dataset': ["BMandVar",],
    'dataset_id': [None],
    'which_loss': ['easy',],
    'coord_wise_tau': [False,],
    'use_y_for_ode': [True, ],
    'use_rnn': [False],
    'masked': [False],
    'plot': [True],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'plot_same_yaxis': [True],
    'saved_models_path': [BMandVar_models_path],
}
param_list_BMandVar_1 = get_parameter_array(param_dict=param_dict_BMandVar_1)

param_dict_BMandVar_2 = {
    'epochs': [200],
    'batch_size': [200],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [50],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'readout_nn': [None],
    'enc_nn': [_nn],
    'func_appl_X': [["power-2"]],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'dataset': ["BM",],
    'dataset_id': [None],
    'which_loss': ['easy',],
    'coord_wise_tau': [False,],
    'use_y_for_ode': [True, ],
    'use_rnn': [False],
    'masked': [False],
    'plot': [True],
    'plot_variance': [True],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'plot_same_yaxis': [True],
    'saved_models_path': [BMandVar_models_path],
}
param_list_BMandVar_1 += get_parameter_array(param_dict=param_dict_BMandVar_2)

plot_paths_BMVar_dict = {
    'model_ids': [1,2,], 'saved_models_path': BMandVar_models_path,
    'which': 'best', 'paths_to_plot': [2,],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}



# ------------------------------------------------------------------------------
# --- BS with dependent observation intensity
DepIntensity_models_path = "{}saved_models_DepIntensity/".format(data_path)
_nn = ((50, 'tanh'), (50, 'tanh'),)
param_dict_DepIntensity_1 = {
    'epochs': [200],
    'batch_size': [200],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [50],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'readout_nn': [_nn],
    'enc_nn': [_nn],
    'func_appl_X': [[]],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'dataset': ["BlackScholes"],
    'dataset_id': [None],
    'which_loss': ['easy',],
    'coord_wise_tau': [False,],
    'use_y_for_ode': [True, ],
    'use_rnn': [False],
    'masked': [False],
    'plot': [True],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'plot_same_yaxis': [True],
    'saved_models_path': [DepIntensity_models_path],
}
param_list_DepIntensity_1 = get_parameter_array(
    param_dict=param_dict_DepIntensity_1)

param_dict_DepIntensity_2 = {
    'epochs': [200],
    'batch_size': [200],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [50],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'readout_nn': [_nn],
    'enc_nn': [_nn],
    'func_appl_X': [[]],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'dataset': ["BlackScholes"],
    'dataset_id': [None],
    'which_loss': ['easy',],
    'coord_wise_tau': [False,],
    'use_y_for_ode': [True, ],
    'use_rnn': [True],
    'input_sig': [True],
    'level': [3,],
    'masked': [False],
    'plot': [True],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'plot_same_yaxis': [True],
    'saved_models_path': [DepIntensity_models_path],
}
param_list_DepIntensity_1 += get_parameter_array(
    param_dict=param_dict_DepIntensity_2)

plot_paths_DepIntensity_dict = {
    'model_ids': [1,2], 'saved_models_path': DepIntensity_models_path,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4,5,6,7,8,],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},
    'plot_obs_prob': True}


# ------------------------------------------------------------------------------
# --- physionet
param_list_physio = []
physio_models_path = "{}saved_models_PhysioNet/".format(data_path)
_nn = ((50, 'tanh'),)
param_dict_physio_2 = {
    'epochs': [175],
    'batch_size': [50],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'hidden_size': [50,],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'readout_nn': [_nn],
    'enc_nn': [_nn],
    'use_rnn': [True,],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'input_sig': [True],
    'level': [2],
    'dataset': ["physionet"],
    'dataset_id': [None],
    'which_loss': ['easy'],
    'quantization': [0.016],
    'n_samples': [8000],
    'saved_models_path': [physio_models_path],
}
param_list_physio += get_parameter_array(param_dict=param_dict_physio_2)*5
param_list_physio2 = get_parameter_array(param_dict=param_dict_physio_2)*5

overview_dict_physio = dict(
    ids_from=1, ids_to=len(param_list_physio),
    path=physio_models_path,
    params_extract_desc=('dataset', 'network_size', 'nb_layers',
                         'activation_function_1', 'use_rnn',
                         'readout_nn', 'dropout_rate',
                         'hidden_size', 'batch_size', 'which_loss',
                         'input_sig', 'level', ),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "eval_metric_2",
         "eval_metric_2", "evaluation_mse_min"),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
    ),
    sortby=["evaluation_mse_min"],
)

crossval_dict_physio = dict(
    path=physio_models_path, early_stop_after_epoch=0,
    params_extract_desc=(
        'dataset', 'network_size', 'dropout_rate', 'hidden_size',
        'activation_function_1', 'input_sig', ),
    param_combinations=(
        {'network_size': 50, 'hidden_size': 50, 'input_sig': True,},),
    val_test_params_extract=(("max", "epoch", "epoch", "epochs_trained"),
                             ("min", "eval_metric",
                              "eval_metric", "eval_metric_min"),
                             ("min", "eval_metric_2",
                              "eval_metric_2", "eval_metric_2_min"),
                             ("last", "eval_metric_2",
                              "eval_metric_2", "eval_metric_2_last"),
                             ("min", "train_loss",
                              "eval_metric_2", "eval_metric_2_eval_min"),
                             ),
    target_col=('eval_metric_min', 'eval_metric_2_min',
                'eval_metric_2_last', 'eval_metric_2_eval_min')
)


# ------------------------------------------------------------------------------
# --- climate
param_list_climate = []
climate_models_path = "{}saved_models_Climate/".format(data_path)
for _nn in [((50, 'tanh'),), ((200, 'tanh'), (200, 'tanh')), ]:
    param_dict_climate_1 = {
        'epochs': [200],
        'batch_size': [100],
        'save_every': [1],
        'learning_rate': [0.001],
        'test_size': [0.2],
        'seed': [398],
        'hidden_size': [50, 100],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [_nn],
        'readout_nn': [_nn],
        'enc_nn': [_nn],
        'use_rnn': [True,],
        'solver': ["euler"],
        'weight': [0.5],
        'weight_decay': [1.],
        'input_sig': [False, True],
        'level': [2],
        'dataset': ["climate"],
        'data_index': [0, 1, 2, 3, 4],
        'which_loss': ['easy'],
        'delta_t': [0.1],
        'saved_models_path': [climate_models_path],
    }
    param_list_climate += get_parameter_array(param_dict=param_dict_climate_1)

overview_dict_climate = dict(
    ids_from=1, ids_to=len(param_list_climate),
    path=FBM_models_path,
    params_extract_desc=('data_index', 'network_size', 'nb_layers',
                         'activation_function_1', 'use_rnn',
                         'readout_nn', 'dropout_rate',
                         'hidden_size', 'batch_size', 'which_loss',
                         'input_sig', 'level', ),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "eval_metric",
         "test_metric", "test_metric_eval_min"),
        ("min", "eval_metric", "eval_metric", "eval_metric_min"),
        ("min", "test_metric", "test_metric", "test_metric_min"),
        ("min", "eval_loss", "test_metric", "test_metric_eval_loss_min"),
    ),
    sortby=["data_index", "test_metric_eval_min"],
)

crossval_dict_climate = dict(
    path=climate_models_path, early_stop_after_epoch=0,
    params_extract_desc=(
        'dataset', 'network_size', 'dropout_rate', 'hidden_size',
        'activation_function_1', 'input_sig', ),
    param_combinations=(
        {'network_size': 50, 'hidden_size': 50, 'input_sig': False,},
        {'network_size': 50, 'hidden_size': 50, 'input_sig': True,},
        {'network_size': 50, 'hidden_size': 100, 'input_sig': False,},
        {'network_size': 50, 'hidden_size': 100, 'input_sig': True,},
        {'network_size': 200, 'hidden_size': 50, 'input_sig': False,},
        {'network_size': 200, 'hidden_size': 50, 'input_sig': True,},
        {'network_size': 200, 'hidden_size': 100, 'input_sig': False,},
        {'network_size': 200, 'hidden_size': 100, 'input_sig': True,},),
)


# ------------------------------------------------------------------------------
# --- LOB 1
param_list_LOB = []
LOB_models_path = "{}saved_models_LOB/".format(data_path)
_nn = ((50, 'tanh'),)


param_dict_LOB1 = {
    'epochs': [50],
    'batch_size': [50],
    'save_every': [1],
    'learning_rate': [0.01,],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [100],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'readout_nn': [_nn],
    'enc_nn': [_nn],
    'use_rnn': [False, True],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'input_sig': [False, True],
    'level': [2],
    'use_sig_for_classifier': [False],
    'data_dict': ["LOB_dict1", "LOB_dict2", "LOB_dict3"],
    'which_loss': ['easy'],
    'plot': [True],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'output_midprice_only': [False,],
    'use_eval_on_train': [False],
    'residual_enc_dec': [True,],
    'classifier_nn': [_nn],
    'classifier_loss_weight': [1.,],
    'saved_models_path': [LOB_models_path],
}
param_list_LOB1 = get_parameter_array(param_dict=param_dict_LOB1)
param_list_LOB += param_list_LOB1

param_dict_LOB2 = {
    'epochs': [50],
    'batch_size': [50],
    'save_every': [1],
    'learning_rate': [0.01,],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [100],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'readout_nn': [_nn],
    'enc_nn': [_nn],
    'use_rnn': [False, True],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'input_sig': [True],
    'level': [2],
    'use_sig_for_classifier': [True],
    'data_dict': ["LOB_dict1", "LOB_dict2", "LOB_dict3"],
    'which_loss': ['easy'],
    'plot': [True],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'output_midprice_only': [False,],
    'use_eval_on_train': [False],
    'residual_enc_dec': [True,],
    'classifier_nn': [_nn],
    'classifier_loss_weight': [1.,],
    'saved_models_path': [LOB_models_path],
}
param_list_LOB2 = get_parameter_array(param_dict=param_dict_LOB2)
param_list_LOB += param_list_LOB2

param_dict_LOB3 = {
    'epochs': [50],
    'batch_size': [50],
    'save_every': [1],
    'learning_rate': [0.01,],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [100],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'readout_nn': [_nn],
    'enc_nn': [_nn],
    'use_rnn': [False],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'input_sig': [True],
    'level': [2],
    'use_sig_for_classifier': [False],
    'data_dict': ["LOB_dict3"],
    'which_loss': ['easy'],
    'plot': [True],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'output_midprice_only': [False, True],
    'use_eval_on_train': [False],
    'residual_enc_dec': [True,],
    'classifier_nn': [None, _nn],
    'classifier_loss_weight': [1.,],
    'saved_models_path': [LOB_models_path],
}
param_list_LOB3 = get_parameter_array(param_dict=param_dict_LOB3)
param_list_LOB += param_list_LOB3

overview_dict_LOB = dict(
    ids_from=1, ids_to=len(param_list_LOB),
    path=LOB_models_path,
    params_extract_desc=('dataset', 'dataset_id', 'data_dict',
                         'network_size', 'nb_layers',
                         'activation_function_1', 'use_rnn',
                         'classifier_nn', 'dropout_rate', 'learning_rate',
                         'hidden_size', 'batch_size', 'which_loss',
                         'input_sig', 'level', 'output_midprice_only',
                         'use_sig_for_classifier',
                         'use_eval_on_train', 'classifier_loss_weight'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "ref_evaluation_mse", "ref_evaluation_mse",
         "ref_evaluation_mse_min"),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
        ("min", "evaluation_mse", "evaluation_mse", "evaluation_mse_min"),
        ("max", "evaluation_f1score", "evaluation_f1score",
         "evaluation_f1score_max"),
    ),
    sortby=["dataset_id", "evaluation_f1score_max"],
)


# ------------------------------------------------------------------------------
# --- retrain_LOB
param_list_retrainLOB = []
LOB_models_path = "{}saved_models_LOB/".format(data_path)
retrainLOB_models_path = "{}saved_models_retrain_LOB/".format(data_path)

param_dict_retrainLOB1 = {
    'epochs': [200],
    'batch_size': [50],
    'save_every': [1],
    'learning_rate': [0.001,],
    'dataset': ["retrain_LOB"],
    'evaluate': [True],
    'saved_models_path': [retrainLOB_models_path],
    'load_model_id': [17, 18, 9, 10, 11, 12, 19, 20, 21, 22],
    'load_saved_models_path': [LOB_models_path],
    'load_model_load_best': [True, False],
}
param_list_retrainLOB1 = get_parameter_array(param_dict=param_dict_retrainLOB1)
param_list_retrainLOB += param_list_retrainLOB1

param_list_retrainLOB2 = []
nns = [((200, 'tanh'),(200, 'tanh'),),
       ((200, 'tanh'),(200, 'tanh'),(200, 'tanh'),(200, 'tanh')),]
for nn in nns:
    new_classifier_nn1 = {'nn_desc': nn, 'dropout_rate': 0.1, 'bias': True}
    param_dict_retrainLOB2 = {
        'epochs': [1000],
        'batch_size': [50],
        'save_every': [1],
        'learning_rate': [0.001,],
        'dataset': ["retrain_LOB"],
        'evaluate': [True],
        'saved_models_path': [retrainLOB_models_path],
        'load_model_id': [17, 18, 9, 10, 11, 12, 19, 20, 21, 22],
        'load_saved_models_path': [LOB_models_path],
        'load_model_load_best': [True, False],
        'new_classifier_nn': [new_classifier_nn1],
    }
    param_list_retrainLOB2 += get_parameter_array(
        param_dict=param_dict_retrainLOB2)
param_list_retrainLOB += param_list_retrainLOB2

overview_dict_retrainLOB = dict(
    ids_from=1, ids_to=len(param_list_retrainLOB),
    path=retrainLOB_models_path,
    params_extract_desc=(
        'dataset', 'load_model_id', 'load_model_load_best',
        'new_classifier_nn'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "classification_eval_loss", "classification_eval_loss",
         "classification_eval_loss_min"),
        ("max", "evaluation_f1score", "evaluation_f1score",
         "evaluation_f1score_max"),
        ("min", "classification_eval_loss", "evaluation_f1score",
         "f1score_at_eval_loss_min"),
    ),
    sortby=["evaluation_f1score_max"],
)


# ------------------------------------------------------------------------------
# --- LOB 2: Kaggle datasets
param_list_LOB_K = []
LOB_K_models_path = "{}saved_models_LOB_K/".format(data_path)
_nn = ((50, 'tanh'),)

param_dict_LOB_K_1 = {
    'epochs': [50],
    'batch_size': [50],
    'save_every': [1],
    'learning_rate': [0.01,],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [100],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'readout_nn': [_nn],
    'enc_nn': [_nn],
    'use_rnn': [False],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'input_sig': [True],
    'level': [2],
    'use_sig_for_classifier': [False],
    'data_dict': ["LOB_dict_K_3", "LOB_dict_K_6"],
    'which_loss': ['easy'],
    'plot': [True],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'output_midprice_only': [False],
    'use_eval_on_train': [False],
    'residual_enc_dec': [True,],
    'classifier_nn': [_nn],
    'classifier_loss_weight': [1.,],
    'saved_models_path': [LOB_K_models_path],
}
param_list_LOB_K_1 = get_parameter_array(param_dict=param_dict_LOB_K_1)
param_list_LOB_K += param_list_LOB_K_1


overview_dict_LOB_K = dict(
    ids_from=1, ids_to=len(param_list_LOB_K),
    path=LOB_K_models_path,
    params_extract_desc=('dataset', 'dataset_id', 'data_dict',
                         'network_size', 'nb_layers',
                         'activation_function_1', 'use_rnn',
                         'classifier_nn', 'dropout_rate', 'learning_rate',
                         'hidden_size', 'batch_size', 'which_loss',
                         'input_sig', 'level', 'output_midprice_only',
                         'use_sig_for_classifier',
                         'use_eval_on_train', 'classifier_loss_weight'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "ref_evaluation_mse", "ref_evaluation_mse",
         "ref_evaluation_mse_min"),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
        ("min", "evaluation_mse", "evaluation_mse", "evaluation_mse_min"),
        ("max", "evaluation_f1score", "evaluation_f1score",
         "evaluation_f1score_max"),
    ),
    sortby=["data_dict", "evaluation_f1score_max"],
)

plot_paths_LOB_K_dict = {
    'model_ids': [1,2], 'saved_models_path': LOB_K_models_path,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4,5,6,7,8,],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},
    'LOB_plot_errors': True,}


# ------------------------------------------------------------------------------
# --- retrain_LOB_K
param_list_retrainLOB_K = []
LOB_K_models_path = "{}saved_models_LOB_K/".format(data_path)
retrainLOB_K_models_path = "{}saved_models_retrain_LOB_K/".format(data_path)

param_dict_retrainLOB_K_1 = {
    'epochs': [200],
    'batch_size': [50],
    'save_every': [1],
    'learning_rate': [0.001,],
    'dataset': ["retrain_LOB"],
    'evaluate': [True],
    'saved_models_path': [retrainLOB_K_models_path],
    'load_model_id': [1,2],
    'load_saved_models_path': [LOB_K_models_path],
    'load_model_load_best': [True, False],
}
param_list_retrainLOB_K_1 = get_parameter_array(param_dict=param_dict_retrainLOB_K_1)
param_list_retrainLOB_K += param_list_retrainLOB_K_1

param_list_retrainLOB_K_2 = []
param_list_retrainLOB_K_4 = []
nns = [((200, 'tanh'),(200, 'tanh'),),
       ((200, 'tanh'),(200, 'tanh'),(200, 'tanh'),(200, 'tanh'))]
for nn in nns:
    new_classifier_nn1 = {'nn_desc': nn, 'dropout_rate': 0.1, 'bias': True}
    param_dict_retrainLOB_K_2 = {
        'epochs': [1000],
        'batch_size': [50],
        'save_every': [1],
        'learning_rate': [0.001,],
        'dataset': ["retrain_LOB"],
        'evaluate': [True],
        'saved_models_path': [retrainLOB_K_models_path],
        'load_model_id': [1, 2],
        'load_saved_models_path': [LOB_K_models_path],
        'load_model_load_best': [True, False],
        'new_classifier_nn': [new_classifier_nn1],
    }
    param_list_retrainLOB_K_2 += get_parameter_array(
        param_dict=param_dict_retrainLOB_K_2)
param_list_retrainLOB_K += param_list_retrainLOB_K_2


overview_dict_retrainLOB_K = dict(
    ids_from=1, ids_to=len(param_list_retrainLOB_K),
    path=retrainLOB_K_models_path,
    params_extract_desc=(
        'dataset', 'load_model_id', 'load_model_load_best',
        'new_classifier_nn'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "classification_eval_loss", "classification_eval_loss",
         "classification_eval_loss_min"),
        ("max", "evaluation_f1score", "evaluation_f1score",
         "evaluation_f1score_max"),
        ("min", "classification_eval_loss", "evaluation_f1score",
         "f1score_at_eval_loss_min"),
    ),
    sortby=['load_model_id', "evaluation_f1score_max"],
)


# ------------------------------------------------------------------------------
# --- LOB no shift to 0
param_list_LOB_n = []
LOB_n_models_path = "{}saved_models_LOB_n/".format(data_path)
_nn = ((50, 'tanh'),)

param_dict_LOB_n_1 = {
    'epochs': [50],
    'batch_size': [50],
    'save_every': [1],
    'learning_rate': [0.01,],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [100],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'readout_nn': [_nn],
    'enc_nn': [_nn],
    'use_rnn': [False],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'input_sig': [True],
    'level': [2],
    'use_sig_for_classifier': [False],
    'data_dict': ["LOB_dict3_2", "LOB_dict_K_3_2", "LOB_dict_K_6_2"],
    'which_loss': ['easy'],
    'plot': [True],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'output_midprice_only': [False],
    'use_eval_on_train': [False],
    'residual_enc_dec': [True,],
    'classifier_nn': [_nn],
    'classifier_loss_weight': [1.,],
    'saved_models_path': [LOB_n_models_path],
}
param_list_LOB_n_1 = get_parameter_array(param_dict=param_dict_LOB_n_1)
param_list_LOB_n += param_list_LOB_n_1

overview_dict_LOB_n = dict(
    ids_from=1, ids_to=len(param_list_LOB_n),
    path=LOB_n_models_path,
    params_extract_desc=('dataset', 'dataset_id', 'data_dict',
                         'network_size', 'nb_layers',
                         'activation_function_1', 'use_rnn',
                         'classifier_nn', 'dropout_rate', 'learning_rate',
                         'hidden_size', 'batch_size', 'which_loss',
                         'input_sig', 'level', 'output_midprice_only',
                         'use_sig_for_classifier',
                         'use_eval_on_train', 'classifier_loss_weight'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "ref_evaluation_mse", "ref_evaluation_mse",
         "ref_evaluation_mse_min"),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
        ("min", "evaluation_mse", "evaluation_mse", "evaluation_mse_min"),
        ("max", "evaluation_f1score", "evaluation_f1score",
         "evaluation_f1score_max"),
    ),
    sortby=["data_dict", "evaluation_f1score_max"],
)

# --- retrain_LOB_n
param_list_retrainLOB_n = []
LOB_n_models_path = "{}saved_models_LOB_n/".format(data_path)
retrainLOB_n_models_path = "{}saved_models_retrain_LOB_n/".format(data_path)

param_dict_retrainLOB_n_1 = {
    'epochs': [200],
    'batch_size': [50],
    'save_every': [1],
    'learning_rate': [0.001,],
    'dataset': ["retrain_LOB"],
    'evaluate': [True],
    'saved_models_path': [retrainLOB_n_models_path],
    'load_model_id': [1, 2, 3],
    'load_saved_models_path': [LOB_n_models_path],
    'load_model_load_best': [True, False],
}
param_list_retrainLOB_n_1 = get_parameter_array(param_dict=param_dict_retrainLOB_n_1)
param_list_retrainLOB_n += param_list_retrainLOB_n_1


param_list_retrainLOB_n_2 = []
nns = [((200, 'tanh'),(200, 'tanh'),),
       ((200, 'tanh'),(200, 'tanh'),(200, 'tanh'),(200, 'tanh')),]
for nn in nns:
    new_classifier_nn1 = {'nn_desc': nn, 'dropout_rate': 0.1, 'bias': True}
    param_dict_retrainLOB_n_2 = {
        'epochs': [1000],
        'batch_size': [50],
        'save_every': [1],
        'learning_rate': [0.001,],
        'dataset': ["retrain_LOB"],
        'evaluate': [True],
        'saved_models_path': [retrainLOB_n_models_path],
        'load_model_id': [1, 2, 3],
        'load_saved_models_path': [LOB_n_models_path],
        'load_model_load_best': [True, False],
        'new_classifier_nn': [new_classifier_nn1],
    }
    param_list_retrainLOB_n_2 += get_parameter_array(
        param_dict=param_dict_retrainLOB_n_2)
param_list_retrainLOB_n += param_list_retrainLOB_n_2


overview_dict_retrainLOB_n = dict(
    ids_from=1, ids_to=len(param_list_retrainLOB_n),
    path=retrainLOB_n_models_path,
    params_extract_desc=(
        'dataset', 'load_model_id', 'load_model_load_best',
        'new_classifier_nn'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "classification_eval_loss", "classification_eval_loss",
         "classification_eval_loss_min"),
        ("max", "evaluation_f1score", "evaluation_f1score",
         "evaluation_f1score_max"),
        ("min", "classification_eval_loss", "evaluation_f1score",
         "f1score_at_eval_loss_min"),
    ),
    sortby=['load_model_id', "evaluation_f1score_max"],
)



if __name__ == '__main__':
    pass
