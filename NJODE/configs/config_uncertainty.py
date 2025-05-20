"""
author: Florian Krach

This file contains all configs to run the experiments for uncertainty computations.
"""

import numpy as np

from configs.config_utils import get_parameter_array, get_dataset_overview, \
    makedirs, data_path, training_data_path


# ------------------------------------------------------------------------------
# --- physionet
param_list_physio_U = []
physio_models_path_Unc = "{}saved_models_PhysioNet_Uncertainty/".format(data_path)
_nn = ((50, 'tanh'),)

param_dict_physio_U_1 = {
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
    'which_loss': ['easy',],
    'quantization': [0.016],
    'n_samples': [8000],
    'saved_models_path': [physio_models_path_Unc],
    'random_state': [1,2,3,4,5],
    'seed': [1,2,3,4,5],
    'compute_variance': ['variance'],
    'var_weight': [1.],
}
param_list_physio_U += get_parameter_array(param_dict=param_dict_physio_U_1)

overview_dict_physio_U = dict(
    ids_from=1, ids_to=len(param_list_physio_U),
    path=physio_models_path_Unc,
    params_extract_desc=('dataset', 'network_size', 'nb_layers',
                         'activation_function_1', 'use_rnn',
                         'readout_nn', 'dropout_rate',
                         'hidden_size', 'batch_size', 'which_loss',
                         'input_sig', 'level', 'compute_variance', 'var_weight',
                         'random_state', 'seed'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "eval_metric_2",
         "eval_metric_2", "evaluation_mse_min"),
        ("min", "val_loss", "val_loss", "val_loss_min"),
    ),
    sortby=["evaluation_mse_min"],
)

eval_physio_U_dict = {
    'model_ids': list(range(1,26)), 'saved_models_path': physio_models_path_Unc,
    'which': 'best', 'paths_to_plot': None,}




# ------------------------------------------------------------------------------
# --- BM with var

BM_dict_U = {
    'model_name': "BM",
    'nb_paths': 20000, 'nb_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'dimension': 1,
}
BM_models_path_Unc = "{}saved_models_BM_Uncertainty/".format(data_path)
_nn = ((50, 'relu'),(50, 'relu'))
param_dict_BMandVar = {
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
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'data_dict': ["BM_dict_U",],
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
    'saved_models_path': [BM_models_path_Unc],
    'compute_variance': ['variance'],
    'var_weight': [10.],
    'which_var_loss': [1,2,3],
}
param_list_BMandVar_U = get_parameter_array(param_dict=param_dict_BMandVar)


