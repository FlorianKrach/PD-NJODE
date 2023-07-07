"""
author: Florian Krach

This file contains all configs to run the experiments from the NJODE3 paper.
"""

import numpy as np

from configs.config_utils import get_parameter_array, get_dataset_overview, \
    makedirs, data_path, training_data_path



# ==============================================================================
# DATASET DICTS
# ------------------------------------------------------------------------------
BS_dep_obs_dict = {
    'model_name': "BlackScholes",
    'drift': 2., 'volatility': 0.3,
    'nb_paths': 20000, 'nb_steps': 100,
    'S0': 1, 'maturity': 1., 'dimension': 1,
    'obs_perc': None,
    'obs_scheme': {'name': "NJODE3-Example4.9", "p": 0.1, "eta": 3},
    'scheme': 'euler', 'return_vol': False,
}

BM_NoisyObs_dict = {
    'model_name': "BMNoisyObs",
    'nb_paths': 20000, 'nb_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'dimension': 1,
    'obs_noise': {'distribution': 'normal', 'scale': 0.5, 'loc': 0.,
                  'noise_at_start': False},
}

# ==============================================================================
# TRAINING PARAM DICTS
# ------------------------------------------------------------------------------
# --- BS with dependent observations as in NJODE3 Example4.9
DepObs_models_path = "{}saved_models_DepObservations/".format(data_path)
_nn = ((50, 'tanh'), (50, 'tanh'),)
param_dict_DepObs_1 = {
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
    'data_dict': ['BS_dep_obs_dict'],
    'dataset_id': [None],
    'which_loss': ['easy',],
    'coord_wise_tau': [False,],
    'use_y_for_ode': [True, ],
    'use_rnn': [True],
    'input_sig': [True],
    'level': [3,],
    'masked': [False],
    'residual_enc_dec': [True, False,],
    'plot': [True],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'plot_same_yaxis': [True],
    'plot_obs_prob': [True],
    'saved_models_path': [DepObs_models_path],
}
param_list_DepObs_1 = get_parameter_array(
    param_dict=param_dict_DepObs_1)

plot_paths_DepObs_dict = {
    'model_ids': [1,2], 'saved_models_path': DepObs_models_path,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4,5,6,7,8,],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},
    'plot_obs_prob': True}


# ------------------------------------------------------------------------------
# --- Brownian Motion with Noisy Observations
BM_NoisyObs_models_path = "{}saved_models_BMNoisyObs/".format(data_path)
param_list_BM_NoisyObs = []

for size in [100]:
    for act in ['tanh', 'relu']:
        _nn = ((size, act),)
        param_dict_BM_NoisyObs = {
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
            'residual_enc_dec': [True, False, ],
            'func_appl_X': [[]],
            'solver': ["euler"],
            'weight': [0.5],
            'weight_decay': [1.],
            'input_sig': [True],
            'level': [3, ],
            'data_dict': ["BM_NoisyObs_dict", ],
            'which_loss': ['easy', 'noisy_obs'],
            'coord_wise_tau': [False,],
            'use_y_for_ode': [True],
            'masked': [False],
            'plot': [True],
            'evaluate': [True],
            'paths_to_plot': [(0,1,2,3,4,)],
            'plot_same_yaxis': [True],
            'saved_models_path': [BM_NoisyObs_models_path],
        }
        param_list_BM_NoisyObs += get_parameter_array(
            param_dict=param_dict_BM_NoisyObs)

overview_dict_BM_NoisyObs = dict(
    ids_from=1, ids_to=len(param_list_BM_NoisyObs),
    path=BM_NoisyObs_models_path,
    params_extract_desc=('data_dict', 'network_size', 'readout_nn',
                         'activation_function_1',
                         'hidden_size', 'batch_size', 'which_loss',
                         'input_sig', 'level', 'coord_wise_tau',
                         'use_y_for_ode', 'residual_enc_dec'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "evaluation_mean_diff",
         "evaluation_mean_diff", "evaluation_mean_diff_min"),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
    ),
    sortby=["evaluation_mean_diff_min"],
)

plot_paths_BM_NoisyObs_dict = {
    'model_ids': [14, 13], 'saved_models_path': BM_NoisyObs_models_path,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4,5],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}





if __name__ == '__main__':
    pass
