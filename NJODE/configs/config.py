"""
author: Florian Krach
"""

from configs.config_NJODE1 import *
from configs.config_LOB import *
from configs.config_randomizedNJODE import *
from configs.config_NJmodel import *
from configs.config_NJODE3 import *

import numpy as np
import socket

from configs.config_utils import get_parameter_array, makedirs, \
    SendBotMessage, data_path, training_data_path

if 'ada-' not in socket.gethostname():
    SERVER = False
else:
    SERVER = True

# ==============================================================================
# Global variables
CHAT_ID = "-587067467"
ERROR_CHAT_ID = "-725470544"

SendBotMessage = SendBotMessage
makedirs = makedirs

flagfile = "{}flagfile.tmp".format(data_path)

saved_models_path = '{}saved_models/'.format(data_path)




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
    'model_name': "BM2DCorr",
    'nb_paths': 20000, 'nb_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'alpha_sq': 0.9, 'masked': 0.,
    'dimension': 2,
}
BM_2D_dict_2 = {
    'model_name': "BM2DCorr",
    'nb_paths': 20000, 'nb_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'alpha_sq': 0.5, 'masked': 0.25,
    'dimension': 2,
}

# ------------------------------------------------------------------------------
BM_and_Var_dict = {
    'model_name': "BMandVar",
    'nb_paths': 20000, 'nb_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'dimension': 2,
}
BM_dict = {
    'model_name': "BM",
    'nb_paths': 20000, 'nb_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'dimension': 1,
}

# ------------------------------------------------------------------------------
BS_dep_intensity_dict = {
    'model_name': "BlackScholes",
    'drift': 2., 'volatility': 0.3,
    'nb_paths': 20000, 'nb_steps': 100,
    'S0': 1, 'maturity': 1., 'dimension': 1,
    'obs_perc': 0.1,
    'scheme': 'euler', 'return_vol': False,
    'X_dependent_observation_prob':
        "lambda x: 0.05 + 0.4 * np.tanh(np.abs(np.mean(x, axis=1))/10)"
}

# ------------------------------------------------------------------------------
DP_dict1 = {
    'model_name': "DoublePendulum",
    'start_alpha_mean': np.pi, 'start_alpha_std': 0.2, 'length': 1.,
    'mass_ratio': 1., 'nb_paths': 20000, 'maturity': 10.,
    'sampling_step_size': 0.1, 'sampling_nb_steps': 100,
    'use_every_n': 1,
    'dimension': 4, 'obs_perc': 0.1,
}

DP_dict2 = {
    'model_name': "DoublePendulum",
    'start_alpha_mean': np.pi, 'start_alpha_std': 0.2, 'length': 1.,
    'mass_ratio': 1., 'nb_paths': 20000, 'maturity': 2.5,
    'sampling_step_size': 0.025, 'sampling_nb_steps': 100,
    'use_every_n': 1,
    'dimension': 4, 'obs_perc': 0.25,
}

DP_dict3 = {
    'model_name': "DoublePendulum",
    'start_alpha_mean': np.pi, 'start_alpha_std': 0.2, 'length': 1.,
    'mass_ratio': 1., 'nb_paths': 20000, 'maturity': 2.5,
    'sampling_step_size': 0.025, 'sampling_nb_steps': 100,
    'use_every_n': 1,
    'dimension': 4, 'obs_perc': 0.1,
}

DP_dict3_test = {
    'model_name': "DoublePendulum",
    'start_alpha_mean': np.pi, 'start_alpha_std': 0.2, 'length': 1.,
    'mass_ratio': 1., 'nb_paths': 4000, 'maturity': 2.5,
    'sampling_step_size': 0.025, 'sampling_nb_steps': 100,
    'use_every_n': 1,
    'dimension': 4, 'obs_perc': 0.0,
}

# ------------------------------------------------------------------------------
BM_Filter_dict = {
    'model_name': "BMFiltering",
    'nb_paths': 40000, 'nb_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'alpha': 1, 'masked': (1, 0.1),
    'dimension': 2,
}
BM_Filter_dict_1 = {
    'model_name': "BMFiltering",
    'nb_paths': 40000, 'nb_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'alpha': 1, 'masked': (1, 0.25),
    'dimension': 2,
}
BM_Filter_dict_testdata = {
    'model_name': "BMFiltering",
    'nb_paths': 4000, 'nb_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'alpha': 1, 'masked': (1., 0.),
    'dimension': 2,
}

# ------------------------------------------------------------------------------
BM_TimeLag_dict_1 = {
    'model_name': "BMwithTimeLag",
    'nb_paths': 20000, 'nb_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'alpha_in_dt_steps': 19, 'masked': (1, 0.5),
    'timelag_in_dt_steps': 19,
    'dimension': 2,
}
BM_TimeLag_dict_2 = {
    'model_name': "BMwithTimeLag",
    'nb_paths': 20000, 'nb_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'alpha_in_dt_steps': 19, 'masked': (1, 0.1),
    'timelag_in_dt_steps': 19, 'timelag_shift1': False,
    'dimension': 2,
}
BM_TimeLag_dict_testdata = {
    'model_name': "BMwithTimeLag",
    'nb_paths': 4000, 'nb_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'alpha_in_dt_steps': 19, 'masked': (1., 0.),
    'timelag_in_dt_steps': 19,
    'dimension': 2,
}



# ==============================================================================
# TRAINING PARAM DICTS
# ------------------------------------------------------------------------------
ode_nn = ((50, 'tanh'), (50, 'tanh'))
readout_nn = ((50, 'tanh'), (50, 'tanh'))
enc_nn = ((50, 'tanh'), (50, 'tanh'))
lstm_enc_nn = (("lstm", 50),)
# NOTE: when using an lstm encoder, the hidden_size has to be 2* lstm output
#   size, so here 100

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

plot_paths_BM2D_dict = {
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
    # 'input_sig': [True],
    # 'level': [2,],
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
    # 'input_sig': [True],
    # 'level': [2,],
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
    'data_dict': ['BS_dep_intensity_dict'],
    'dataset_id': [None],
    'which_loss': ['easy',],
    'coord_wise_tau': [False,],
    'use_y_for_ode': [True, ],
    'use_rnn': [False],
    # 'input_sig': [True],
    # 'level': [2,],
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
    'data_dict': ['BS_dep_intensity_dict'],
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
# --- Double Pendulum (chaotic system) dataset
DP_models_path = "{}saved_models_DoublePendulum/".format(data_path)
param_list_DP = []
for _nn in [((200, 'tanh'),), ((200, 'relu'),),]:
    param_dict_DP_1 = {
        'epochs': [200],
        'batch_size': [200],
        'save_every': [1],
        'learning_rate': [0.001],
        'test_size': [0.2],
        'seed': [398],
        'hidden_size': [50, 100, 200, 400],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [_nn],
        'readout_nn': [_nn],
        'enc_nn': [_nn],
        'use_rnn': [False, True,],
        'func_appl_X': [[]],
        'solver': ["euler"],
        'weight': [0.5],
        'weight_decay': [1.],
        'input_sig': [False, True],
        'level': [3],
        'data_dict': ['DP_dict1', 'DP_dict2'],
        'which_loss': ['easy'],
        'plot': [True],
        'evaluate': [True],
        'paths_to_plot': [(0,1,2,3,4,)],
        'saved_models_path': [DP_models_path],
    }
    param_list_DP1 = get_parameter_array(param_dict=param_dict_DP_1)
    param_list_DP += param_list_DP1

param_list_DP2 = []
for _nn in [((200, 'tanh'),), ]:
    param_dict_DP_2 = {
        'epochs': [200],
        'batch_size': [200],
        'save_every': [1],
        'learning_rate': [0.001],
        'test_size': [0.2],
        'seed': [398],
        'hidden_size': [400,],
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
        'input_sig': [False],
        'level': [3],
        'data_dict': ['DP_dict3'],
        'which_loss': ['easy'],
        'plot': [True],
        'evaluate': [True],
        'paths_to_plot': [(0,1,2,3,4,)],
        'saved_models_path': [DP_models_path],
    }
    param_list_DP2 += get_parameter_array(param_dict=param_dict_DP_2)
    param_list_DP += param_list_DP2

overview_dict_DP = dict(
    ids_from=1, ids_to=len(param_list_DP),
    path=DP_models_path,
    params_extract_desc=('data_dict', 'network_size', 'readout_nn',
                         'activation_function_1',
                         'hidden_size', 'batch_size', 'which_loss',
                         'use_rnn', 'input_sig', 'level',),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "evaluation_mean_diff",
         "evaluation_mean_diff", "evaluation_mean_diff_min"),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
    ),
    sortby=['data_dict', "evaluation_mean_diff_min"],
)

plot_paths_DP_dict = {
    'model_ids': [65, 14], 'saved_models_path': DP_models_path,
    'which': 'best', 'paths_to_plot': [4,5,6,7,8,],
    'ylabels': ["$\\alpha_1$", "$\\alpha_2$", "$p_1$", "$p_2$"],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}


# ------------------------------------------------------------------------------
# --- Brownian Motion Filtering Problem
BMFilter_models_path = "{}saved_models_BMFiltering/".format(data_path)
param_list_BMFilter_1 = []

for size in [100,]:
    for act in ['tanh',]:
        _nn = ((size, act), )
        param_dict_BMFilter_1_2 = {
            'epochs': [200],
            'batch_size': [200],
            'save_every': [1],
            'learning_rate': [1e-3, ],
            'test_size': [0.2],
            'seed': [398],
            'hidden_size': [200,],
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
            'input_sig': [True],
            'level': [2,],
            'data_dict': ["BM_Filter_dict_1"],
            'test_data_dict': ['BM_Filter_dict_testdata',],
            'which_loss': ['easy',],
            'use_y_for_ode': [True],
            'masked': [True],
            'plot': [True],
            'evaluate': [True],
            'paths_to_plot': [(0,1,2,3,4,)],
            'plot_same_yaxis': [True],
            'saved_models_path': [BMFilter_models_path],
        }
        param_list_BMFilter_1 += get_parameter_array(
            param_dict=param_dict_BMFilter_1_2)


overview_dict_BMFilter_1 = dict(
    ids_from=1, ids_to=len(param_list_BMFilter_1),
    path=BMFilter_models_path,
    params_extract_desc=('data_dict', 'network_size', 'readout_nn',
                         'activation_function_1', 'learning_rate',
                         'hidden_size', 'batch_size', 'which_loss',
                         'use_rnn', 'input_sig', 'level', ),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "evaluation_mean_diff",
         "evaluation_mean_diff", "evaluation_mean_diff_min"),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
    ),
    sortby=["evaluation_mean_diff_min"],
)

plot_paths_BMFilter_dict = {
    'model_ids': [1], 'saved_models_path': BMFilter_models_path,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4,5],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}


# ------------------------------------------------------------------------------
# --- Brownian Motion with TimeLag
BMTimeLag_models_path = "{}saved_models_BMTimeLag/".format(data_path)
param_list_BMTimeLag = []

for _nn in [((200, 'relu'),),]:
    param_dict_BMTimeLag_1_2 = {
        'epochs': [200],
        'batch_size': [200],
        'save_every': [1],
        'learning_rate': [1e-3, ],
        'test_size': [0.2],
        'seed': [398],
        'hidden_size': [400,],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [_nn],
        'readout_nn': [_nn, None],
        'enc_nn': [_nn, ],
        'use_rnn': [True,],
        'func_appl_X': [[]],
        'solver': ["euler"],
        'weight': [0.5],
        'weight_decay': [1.],
        'input_sig': [True, False],
        'level': [2,],
        'data_dict': ["BM_TimeLag_dict_1", "BM_TimeLag_dict_2"],
        'test_data_dict': ['BM_TimeLag_dict_testdata',],
        'which_loss': ['easy',],
        'residual_enc_dec': [False,],
        'ode_input_scaling_func': ['tanh'],
        'enc_input_t': [True,],
        'coord_wise_tau': [True,],
        'use_y_for_ode': [True],
        'masked': [True],
        'plot': [True],
        'evaluate': [True],
        'paths_to_plot': [(0,1,2,3,4,)],
        'plot_same_yaxis': [True],
        'saved_models_path': [BMTimeLag_models_path],
    }
    param_list_BMTimeLag += get_parameter_array(
        param_dict=param_dict_BMTimeLag_1_2)

overview_dict_BMTimeLag = dict(
    ids_from=1, ids_to=len(param_list_BMTimeLag),
    path=BMTimeLag_models_path,
    params_extract_desc=('data_dict', 'readout_nn', 'ode_nn',
                         'learning_rate',
                         'hidden_size', 'batch_size', 'which_loss',
                         'use_rnn', 'input_sig', 'level', ),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "evaluation_mean_diff",
         "evaluation_mean_diff", "evaluation_mean_diff_min"),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
    ),
    sortby=["evaluation_mean_diff_min"],
)

plot_paths_BMTimeLag_dict = {
    'model_ids': [2],
    'saved_models_path': BMTimeLag_models_path,
    'which': 'best', 'paths_to_plot': [0,4,],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}








if __name__ == '__main__':
    pass
