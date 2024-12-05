"""
author: Florian Krach

This file contains all configs to run the experiments for learning ODEs.
"""

import numpy as np

from configs.config_utils import get_parameter_array, get_dataset_overview, \
    makedirs, data_path, training_data_path



# ==============================================================================
# DATASET DICTS
# ------------------------------------------------------------------------------
DP_dict3 = {
    'model_name': "DoublePendulum",
    'start_alpha_mean': np.pi, 'start_alpha_std': 0.2, 'length': 1.,
    'mass_ratio': 1., 'nb_paths': 20000, 'maturity': 2.5,
    'sampling_nb_steps': 100,
    'use_every_n': 1,
    'dimension': 4, 'obs_perc': 0.1,
}

DP_dict3_test = {
    'model_name': "DoublePendulum",
    'start_alpha_mean': np.pi, 'start_alpha_std': 0.2, 'length': 1.,
    'mass_ratio': 1., 'nb_paths': 4000, 'maturity': 2.5,
    'sampling_nb_steps': 100,
    'use_every_n': 1,
    'dimension': 4, 'obs_perc': 0.0,
}

DP_dict3_2_test = {
    'model_name': "DoublePendulum",
    'start_alpha_mean': np.pi, 'start_alpha_std': 0.2, 'length': 1.,
    'mass_ratio': 1., 'nb_paths': 4000, 'maturity': 5,
    'sampling_nb_steps': 2*100,
    'use_every_n': 1,
    'dimension': 4, 'obs_perc': 0.0,
}

DP_dict4 = {
    'model_name': "DoublePendulum",
    'start_alpha_mean': np.pi, 'start_alpha_std': 0.2, 'length': 1.,
    'mass_ratio': 1., 'nb_paths': 100000, 'maturity': 2.5,
    'sampling_nb_steps': 100,
    'use_every_n': 1,
    'dimension': 4, 'obs_perc': 0.25,
}

DP_dict4_test = {
    'model_name': "DoublePendulum",
    'start_alpha_mean': np.pi, 'start_alpha_std': 0.2, 'length': 1.,
    'mass_ratio': 1., 'nb_paths': 4000, 'maturity': 2.5,
    'sampling_nb_steps': 100,
    'use_every_n': 1,
    'dimension': 4, 'obs_perc': 0.0,
}


# --- long-term predictions
BS_LT_dict = {
    'model_name': "BlackScholes",
    'drift': 2., 'volatility': 0.3,
    'nb_paths': 20000, 'nb_steps': 100,
    'S0': 1, 'maturity': 1., 'dimension': 1,
    'obs_perc': 0.1,
}

BS_LT_dict1 = {
    'model_name': "BlackScholes",
    'drift': 2., 'volatility': 0.3,
    'nb_paths': 20000, 'nb_steps': 100,
    'S0': 1, 'maturity': 1., 'dimension': 1,
    'obs_perc': 0.4,
}

BS_LT_dict_test = {
    'model_name': "BlackScholes",
    'drift': 2., 'volatility': 0.3,
    'nb_paths': 4000, 'nb_steps': 100,
    'S0': 1, 'maturity': 1., 'dimension': 1,
    'obs_perc': 0.0,
}

BS_LT_dict2 = {
    'model_name': "BlackScholes",
    'drift': 2., 'volatility': 0.3,
    'nb_paths': 20000, 'nb_steps': 100,
    'S0': 1, 'maturity': 1., 'dimension': 1,
    'obs_perc': 0.1,
    'sine_coeff': 2 * np.pi
}

BS_LT_dict2_test = {
    'model_name': "BlackScholes",
    'drift': 2., 'volatility': 0.3,
    'nb_paths': 4000, 'nb_steps': 100,
    'S0': 1, 'maturity': 1., 'dimension': 1,
    'obs_perc': 0.0,
    'sine_coeff': 2 * np.pi
}






# ==============================================================================
# TRAINING PARAM DICTS
# ------------------------------------------------------------------------------
# --- Double Pendulum (chaotic system) dataset
DP_models_path = "{}saved_models_ODEexp_DoublePendulum/".format(data_path)
param_list_ODE_DP = []
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
        'test_data_dict': ['DP_dict3_test'],
        'which_loss': ['easy'],
        'plot': [True],
        'evaluate': [True],
        'paths_to_plot': [(0,1,2,3,4,)],
        'saved_models_path': [DP_models_path],
        'use_current_y_for_ode': [False, True],
        'use_observation_as_input': [
            False, True, 0.5, 0.75, 0.25,
            "lambda x: np.random.random(1) < 1-x/200",
            "lambda x: np.random.random(1) < 1-x/100",
        ],
        'val_use_observation_as_input': [False,],
        'eval_use_true_paths': [True],

    }
    param_list_ODE_DP += get_parameter_array(param_dict=param_dict_DP_2)


for _nn in [((200, 'tanh'),), ((400, 'tanh'),),
            ((200, 'tanh'),(200, 'tanh')), ((200, 'relu'),(200, 'tanh')),]:
    param_dict_DP_2 = {
        'epochs': [300],
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
        'data_dict': ['DP_dict4'],
        'test_data_dict': ['DP_dict3_test'],
        'which_loss': ['easy'],
        'plot': [True],
        'evaluate': [True],
        'paths_to_plot': [(0,1,2,3,4,)],
        'saved_models_path': [DP_models_path],
        'use_current_y_for_ode': [True],
        'use_observation_as_input': [
            "lambda x: np.random.random() < 1-x/100",
        ],
        'val_use_observation_as_input': [False,],
        'eval_use_true_paths': [True],

    }
    param_list_ODE_DP += get_parameter_array(param_dict=param_dict_DP_2)



overview_dict_ODE_DP = dict(
    ids_from=1, ids_to=len(param_list_ODE_DP),
    path=DP_models_path,
    params_extract_desc=('data_dict', 'test_data_dict',
                         'network_size', 'readout_nn',
                         'activation_function_1',
                         'hidden_size', 'batch_size', 'which_loss',
                         'use_rnn', 'input_sig', 'level',
                         'use_current_y_for_ode',
                         'use_observation_as_input',
                         'val_use_observation_as_input'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "evaluation_mean_diff",
         "evaluation_mean_diff", "evaluation_mean_diff_min"),
        ("min", "val_loss", "val_loss", "val_loss_min"),
    ),
    sortby=['data_dict', "evaluation_mean_diff_min"],
)

plot_paths_ODE_DP_dict = {
    'model_ids': [2, 15], 'saved_models_path': DP_models_path,
    'which': 'best', 'paths_to_plot': [4,5,6,7,8,],
    'ylabels': ["$\\alpha_1$", "$\\alpha_2$", "$p_1$", "$p_2$"],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}


# ------------------------------------------------------------------------------
# --- long-term predictions
BS_LT_models_path = "{}saved_models_BS_LongTerm/".format(data_path)
param_list_BS_LT = []
_nn = ((100, 'tanh'),)
for ds, ds_test in [('BS_LT_dict', 'BS_LT_dict_test'),
                    ('BS_LT_dict2', 'BS_LT_dict2_test'),
                    ('BS_LT_dict1', 'BS_LT_dict_test')]:
    param_dict_BS_LT = {
        'epochs': [200],
        'batch_size': [200],
        'save_every': [1],
        'learning_rate': [0.001],
        'test_size': [0.2],
        'seed': [398],
        'hidden_size': [100,],
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
        'level': [3],
        'data_dict': [ds],
        'test_data_dict': [ds_test],
        'which_loss': ['easy'],
        'plot': [True],
        'evaluate': [True],
        'paths_to_plot': [(0,1,2,3,4,)],
        'saved_models_path': [BS_LT_models_path],
        'use_current_y_for_ode': [False, True],
        'use_observation_as_input': [
            True,
            "lambda x: np.random.random(1) < 1-x/100",
        ],
        'val_use_observation_as_input': [False,],
    }
    param_list_BS_LT += get_parameter_array(param_dict=param_dict_BS_LT)

overview_dict_BS_LT = dict(
    ids_from=1, ids_to=len(param_list_BS_LT),
    path=BS_LT_models_path,
    params_extract_desc=('data_dict', 'test_data_dict',
                         'network_size', 'readout_nn',
                         'activation_function_1',
                         'hidden_size', 'batch_size', 'which_loss',
                         'use_rnn', 'input_sig', 'level',
                         'use_current_y_for_ode',
                         'use_observation_as_input',
                         'val_use_observation_as_input'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "evaluation_mean_diff",
         "evaluation_mean_diff", "evaluation_mean_diff_min"),
        ("min", "val_loss", "val_loss", "val_loss_min"),
        ("min", "val_loss", "evaluation_mean_diff",
         "evaluation_mean_diff_at_min_val_loss"),
    ),
    sortby=['data_dict', "evaluation_mean_diff_min"],
)

plot_paths_BS_LT_dict = {
    'model_ids': [1,4,5,8,9,12], 'saved_models_path': BS_LT_models_path,
    'which': 'best', 'paths_to_plot': [4,5,6,7,8,],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}


# ------------------------------------------------------------------------------
# --- long-term predictions training in physionet dataset
physionetLT_models_path = "{}saved_models_LongTerm_physionet/".format(data_path)
param_list_physioLT = []

_nn = ((50, 'tanh'),)
param_dict_physioLT = {
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
    'saved_models_path': [physionetLT_models_path],
    'use_current_y_for_ode': [False, True],
    'use_observation_as_input': [
        "lambda x: np.random.random(1) < 1-x/350",
        "lambda x: np.random.random(1) < 1-(x-100)/150",
        "lambda x: np.random.random(1) < 1-min(x-75,50)/200",
    ],
    'val_use_observation_as_input': [True,],
}

param_list_physioLT += get_parameter_array(param_dict=param_dict_physioLT) # *5

overview_dict_physioLT = dict(
    ids_from=1, ids_to=len(param_list_physioLT),
    path=physionetLT_models_path,
    params_extract_desc=('dataset', 'network_size', 'nb_layers',
                         'activation_function_1', 'use_rnn',
                         'readout_nn', 'dropout_rate',
                         'hidden_size', 'batch_size', 'which_loss',
                         'use_observation_as_input', 'level', 'input_sig'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "eval_metric_2", "eval_metric_2", "evaluation_mse_min"),
        ("min", "val_loss", "val_loss", "val_loss_min"),
        ("min", "train_loss", "eval_metric_2", "eval_metric_2_train_min"),
    ),
    sortby=["evaluation_mse_min"],
)






if __name__ == '__main__':
    pass
