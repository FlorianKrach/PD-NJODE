"""
author: Florian Krach
"""


import numpy as np

from configs.config_utils import get_parameter_array, get_dataset_overview, \
    makedirs, data_path, training_data_path



# ==============================================================================
#                        randomizedNJODE - DATASET DICTS
# ==============================================================================
BlackScholes_dict = {
    'model_name': "BlackScholes",
    'drift': 2., 'volatility': 0.3, 'mean': 4, 'poisson_lambda': 3.,
    'speed': 2., 'correlation': 0.5, 'nb_paths': 20000, 'nb_steps': 100,
    'S0': 1, 'maturity': 1., 'dimension': 1,
    'obs_perc': 0.1,
    'scheme': 'euler', 'return_vol': False, 'v0': 1, 'hurst':0.75,
    'FBMmethod':"daviesharte"
}


# ==============================================================================
#                        randomizedNJODE - TRAINING PARAM DICTS
# ==============================================================================
# ------------------------------------------------------------------------------
# --- training with SGD (only readout) on BM and BMwithVar
randNJODE_models_path = "{}saved_models_randNJODE/".format(data_path)
_nn = ((50, 'tanh'),)
param_dict_randNJODE1 = {
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
    'dataset': ["BM", "BMandVar"],
    'dataset_id': [None],
    'which_loss': ['easy',],
    'coord_wise_tau': [False,],
    'use_y_for_ode': [True,],
    'use_rnn': [False],
    # 'input_sig': [True],
    # 'level': [2,],
    'masked': [False],
    'plot': [True],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'plot_same_yaxis': [True],
    'train_readout_only': [True],
    'saved_models_path': [randNJODE_models_path],
}
param_list_randNJODE_1 = get_parameter_array(param_dict=param_dict_randNJODE1)

# ------------------------------------------------------------------------------
# --- training with OLS on BM and BMwithVar
param_dict_randNJODE2 = {
    'epochs': [200],
    'batch_size': [200],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [50, 200],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'enc_nn': [_nn],
    'func_appl_X': [[]],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'dataset': ["BM", "BMandVar"],
    'dataset_id': [None],
    'which_loss': ['easy',],
    'coord_wise_tau': [False,],
    'use_y_for_ode': [True,],
    'use_rnn': [False],
    # 'input_sig': [True],
    # 'level': [2,],
    'masked': [False],
    'plot': [True],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'plot_same_yaxis': [True],
    'other_model': ['randomizedNJODE'],
    'saved_models_path': [randNJODE_models_path],
}
param_list_randNJODE_2 = get_parameter_array(param_dict=param_dict_randNJODE2)

plot_paths_randNJODE_dict = {
    'model_ids': [2, 5, 6], 'saved_models_path': randNJODE_models_path,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4,],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}


# ------------------------------------------------------------------------------
# --- training with OLS on DoublePendulum
param_dict_randNJODE3 = {
    'epochs': [200],
    'batch_size': [200],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [50, 200],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'enc_nn': [_nn],
    'func_appl_X': [[]],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'dataset': ["DoublePendulum",],
    'dataset_id': [None],
    'which_loss': ['easy',],
    'coord_wise_tau': [False,],
    'use_y_for_ode': [True,],
    'use_rnn': [False],
    # 'input_sig': [True],
    # 'level': [2,],
    'masked': [False],
    'plot': [True],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'plot_same_yaxis': [True],
    'other_model': ['randomizedNJODE'],
    'saved_models_path': [randNJODE_models_path],
}
param_list_randNJODE_3 = get_parameter_array(param_dict=param_dict_randNJODE3)


# ------------------------------------------------------------------------------
# --- training with OLS on BlackScholes
randNJODE_BS_models_path = "{}saved_models_randNJODE_BS/".format(data_path)
param_dict_randNJODE_BS = {
    'epochs': [200],
    'batch_size': [200],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [50, 200],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'enc_nn': [_nn],
    'func_appl_X': [[]],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'data_dict': ["BlackScholes_dict",],
    'dataset_id': [None],
    'which_loss': ['easy',],
    'coord_wise_tau': [False,],
    'use_y_for_ode': [True,],
    'use_rnn': [False, True],
    # 'input_sig': [True],
    # 'level': [2,],
    'masked': [False],
    'plot': [True],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'plot_same_yaxis': [True],
    'other_model': ['randomizedNJODE'],
    'saved_models_path': [randNJODE_BS_models_path],
}
param_list_randNJODE_BS = get_parameter_array(param_dict=param_dict_randNJODE_BS)


overview_dict_randNJODE_BS = dict(
    ids_from=1, ids_to=len(param_list_randNJODE_BS),
    path=randNJODE_BS_models_path,
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



plot_paths_randNJODE_dict_BS = {
    'model_ids': [1,2,3,4], 'saved_models_path': randNJODE_BS_models_path,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4,],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}


if __name__ == '__main__':
    pass
