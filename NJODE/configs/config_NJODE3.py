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
    'residual_dec': [True, False,],
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
            'residual_dec': [True, False, ],
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
                         'use_y_for_ode', 'residual_dec'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "evaluation_mean_diff",
         "evaluation_mean_diff", "evaluation_mean_diff_min"),
        ("min", "val_loss", "val_loss", "val_loss_min"),
    ),
    sortby=["evaluation_mean_diff_min"],
)

plot_paths_BM_NoisyObs_dict = {
    'model_ids': [14, 13], 'saved_models_path': BM_NoisyObs_models_path,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4,5],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}



# ------------------------------------------------------------------------------
# --- physionet
param_list_physio_N3 = []
physio_models_path_N3 = "{}saved_models_PhysioNet_NJODE3/".format(data_path)
_nn = ((50, 'tanh'),)

param_dict_physio_N3_1 = {
    'epochs': [100],
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
    'which_loss': ['easy', 'noisy_obs'],
    'quantization': [0.016],
    'n_samples': [8000],
    'saved_models_path': [physio_models_path_N3],
    'obs_noise': [{"std_factor": 0.0, "seed": 333},
                  {"std_factor": 0.2, "seed": 333},
                  {"std_factor": 0.4, "seed": 333},
                  {"std_factor": 0.6, "seed": 333},
                  {"std_factor": 0.8, "seed": 333},
                  {"std_factor": 1.0, "seed": 333},],
}
param_list_physio_N3 += get_parameter_array(param_dict=param_dict_physio_N3_1)

overview_dict_physio_N3 = dict(
    ids_from=1, ids_to=len(param_list_physio_N3),
    path=physio_models_path_N3,
    params_extract_desc=('dataset', 'network_size', 'nb_layers',
                         'activation_function_1', 'use_rnn',
                         'readout_nn', 'dropout_rate',
                         'hidden_size', 'batch_size', 'which_loss',
                         'input_sig', 'level', 'obs_noise',
                         'obs_noise-std_factor'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "eval_metric_2",
         "eval_metric_2", "evaluation_mse_min"),
        ("min", "val_loss", "val_loss", "val_loss_min"),
    ),
    sortby=["evaluation_mse_min"],
)

plot_loss_comparison_physio_N3 = dict(
    filename="{}training_overview-ids-{}-{}.csv".format(
                physio_models_path_N3, 1, len(param_list_physio_N3)),
    param_combinations=({'which_loss': 'easy'}, {'which_loss': 'noisy_obs'}),
    labels=('original', 'noise-adapted'),
    outfile="{}loss_comparison.pdf".format(physio_models_path_N3),
    xcol='obs_noise-std_factor', ycol='evaluation_mse_min',
    xlabel="std. factor of observation noise",
    ylabel="Evaluation MSE",
    logx=False, logy=False,)


# ------------------------------------------------------------------------------
# --- climate
param_list_climate_N3 = []
climate_models_path_N3 = "{}saved_models_Climate/".format(data_path)
for _nn in [((50, 'tanh'),), ]:
    param_dict_climate_1 = {
        'epochs': [200],
        'batch_size': [100],
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
        'use_rnn': [False, True,],
        'solver': ["euler"],
        'weight': [0.5],
        'weight_decay': [1.],
        'input_sig': [False, True],
        'level': [2],
        'dataset': ["climate"],
        'data_index': [0, 1, 2, 3, 4],
        'which_loss': ['easy', 'noisy_obs'],
        'delta_t': [0.1],
        'saved_models_path': [climate_models_path_N3],
    }
    param_list_climate_N3 += get_parameter_array(param_dict=param_dict_climate_1)

overview_dict_climate_N3 = dict(
    ids_from=1, ids_to=len(param_list_climate_N3),
    path=climate_models_path_N3,
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
        ("min", "val_loss", "test_metric", "test_metric_val_loss_min"),
    ),
    sortby=["data_index", "test_metric_eval_min"],
)

crossval_dict_climate_N3 = dict(
    path=climate_models_path_N3, early_stop_after_epoch=100,
    params_extract_desc=(
        'dataset', 'network_size', 'dropout_rate', 'hidden_size',
        'activation_function_1', 'input_sig', 'use_rnn', 'which_loss', ),
    param_combinations=(
        {'network_size': 50, 'hidden_size': 50, 'input_sig': False,
            'use_rnn': False, 'which_loss': 'easy',},
        {'network_size': 50, 'hidden_size': 50, 'input_sig': True,
            'use_rnn': False, 'which_loss': 'easy',},
        {'network_size': 50, 'hidden_size': 50, 'input_sig': False,
            'use_rnn': True, 'which_loss': 'easy',},
        {'network_size': 50, 'hidden_size': 50, 'input_sig': True,
            'use_rnn': True, 'which_loss': 'easy',},
        {'network_size': 50, 'hidden_size': 50, 'input_sig': False,
            'use_rnn': False, 'which_loss': 'noisy_obs',},
        {'network_size': 50, 'hidden_size': 50, 'input_sig': True,
            'use_rnn': False, 'which_loss': 'noisy_obs',},
        {'network_size': 50, 'hidden_size': 50, 'input_sig': False,
            'use_rnn': True, 'which_loss': 'noisy_obs',},
        {'network_size': 50, 'hidden_size': 50, 'input_sig': True,
            'use_rnn': True, 'which_loss': 'noisy_obs',},
    ),
)



if __name__ == '__main__':
    pass
