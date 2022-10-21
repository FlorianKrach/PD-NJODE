"""
author: Florian Krach

This file contains all configs to run the experiments from the first paper
"""
import numpy as np

from configs.config_utils import get_parameter_array, get_dataset_overview, \
    makedirs, data_path, training_data_path



# ==============================================================================
# -------------------- NJODE 1 - Dataset Dicts ---------------------------------
# ==============================================================================

# ------------------------------------------------------------------------------
# default hp dict for generating Black-Scholes, Ornstein-Uhlenbeck and Heston
hyperparam_default = {
    'drift': 2., 'volatility': 0.3, 'mean': 4, 'poisson_lambda': 3.,
    'speed': 2., 'correlation': 0.5, 'nb_paths': 20000, 'nb_steps': 100,
    'S0': 1, 'maturity': 1., 'dimension': 1,
    'obs_perc': 0.1,
    'scheme': 'euler', 'return_vol': False, 'v0': 1, 'hurst':0.75,
    'FBMmethod':"daviesharte"
}


# ------------------------------------------------------------------------------
# Heston without Feller condition
HestonWOFeller_dict1 = {
    'drift': 2., 'volatility': 3., 'mean': 1.,
    'speed': 2., 'correlation': 0.5, 'nb_paths': 20000, 'nb_steps': 100,
    'S0': 1, 'maturity': 1., 'dimension': 1,
    'obs_perc': 0.1,
    'scheme': 'euler', 'return_vol': False, 'v0': 0.5,
}
HestonWOFeller_dict2 = {
    'drift': 2., 'volatility': 3., 'mean': 1.,
    'speed': 2., 'correlation': 0.5, 'nb_paths': 20000, 'nb_steps': 100,
    'S0': 1, 'maturity': 1., 'dimension': 2,
    'obs_perc': 0.1,
    'scheme': 'euler', 'return_vol': True, 'v0': 0.5,
}


# ------------------------------------------------------------------------------
# Combined Ornstein-Uhlenback + Black-Scholes dataset
combined_OU_BS_dataset_dict1 = {
    'drift': 2., 'volatility': 0.3, 'mean': 10, 'poisson_lambda': 3.,
    'speed': 2., 'correlation': 0.5, 'nb_paths': 20000, 'nb_steps': 50,
    'S0': 1, 'maturity': 0.5, 'dimension': 1,
    'obs_perc': 0.1,
    'scheme': 'euler', 'return_vol': False, 'v0': 1, 'hurst':0.75,
    'FBMmethod':"daviesharte"
}
combined_OU_BS_dataset_dict2 = {
    'drift': 2., 'volatility': 0.3, 'mean': 10, 'poisson_lambda': 3.,
    'speed': 2., 'correlation': 0.5, 'nb_paths': 20000, 'nb_steps': 50,
    'S0': 1, 'maturity': 0.5, 'dimension': 1,
    'obs_perc': 0.1,
    'scheme': 'euler', 'return_vol': False, 'v0': 1, 'hurst':0.75,
    'FBMmethod':"daviesharte"
}
combined_OU_BS_dataset_dicts = [combined_OU_BS_dataset_dict1,
                                combined_OU_BS_dataset_dict2]

# ------------------------------------------------------------------------------
# sine-drift Black-Scholes dataset
sine_BS_dataset_dict1 = {
    'drift': 2., 'volatility': 0.3, 'mean': 4, 'poisson_lambda': 3.,
    'speed': 2., 'correlation': 0.5, 'nb_paths': 20000, 'nb_steps': 100,
    'S0': 1, 'maturity': 1., 'dimension': 1,
    'obs_perc': 0.1,
    'scheme': 'euler', 'return_vol': False, 'v0': 1, 'hurst':0.75,
    'FBMmethod': "daviesharte",
    'sine_coeff': 2 * np.pi
}
sine_BS_dataset_dict2 = {
    'drift': 2., 'volatility': 0.3, 'mean': 4, 'poisson_lambda': 3.,
    'speed': 2., 'correlation': 0.5, 'nb_paths': 20000, 'nb_steps': 100,
    'S0': 1, 'maturity': 1., 'dimension': 1,
    'obs_perc': 0.1,
    'scheme': 'euler', 'return_vol': False, 'v0': 1, 'hurst':0.75,
    'FBMmethod': "daviesharte",
    'sine_coeff': 4 * np.pi
}




# ==============================================================================
# -------------------- NJODE 1 - Training Dicts --------------------------------
# ==============================================================================

ode_nn = ((50, 'tanh'), (50, 'tanh'))
readout_nn = ((50, 'tanh'), (50, 'tanh'))
enc_nn = ((50, 'tanh'), (50, 'tanh'))

# ------------------------------------------------------------------------------
# --- Black-Scholes (geom. Brownian Motion), Heston and Ornstein-Uhlenbeck
param_dict1 = {
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
    'dataset': ["BlackScholes", "Heston", "OrnsteinUhlenbeck"],
    'dataset_id': [None],
    'plot': [True],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)]
}
params_list1 = get_parameter_array(param_dict=param_dict1)


# ------------------------------------------------------------------------------
# convergence analysis
path_heston = '{}conv-study-Heston-saved_models/'.format(data_path)
training_size = [int(100 * 2 ** x) for x in np.linspace(1, 7, 7)]
network_size = [int(5 * 2 ** x) for x in np.linspace(1, 6, 6)]
ode_nn = [((size, 'tanh'), (size, 'tanh')) for size in network_size]
params_list_convstud_Heston = []
for _ode_nn in ode_nn:
    param_dict_convstud_Heston = {
        'epochs': [100],
        'batch_size': [20],
        'save_every': [10],
        'learning_rate': [0.001],
        'test_size': [0.2],
        'training_size': training_size,
        'seed': [398],
        'hidden_size': [10],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [_ode_nn],
        'readout_nn': [_ode_nn],
        'enc_nn': [_ode_nn],
        'use_rnn': [False],
        'func_appl_X': [[]],
        'solver': ["euler"],
        'weight': [0.5],
        'weight_decay': [1.],
        'dataset': ["Heston"],
        'dataset_id': [None],
        'plot': [True],
        'paths_to_plot': [(0,)],
        'saved_models_path': [path_heston],
        'evaluate': [True]
    }
    params_list_convstud_Heston += get_parameter_array(
        param_dict=param_dict_convstud_Heston)
params_list_convstud_Heston *= 5

plot_conv_stud_heston_dict1 = dict(
    path=path_heston, x_axis="training_size", x_log=True, y_log=True,
    save_path=path_heston)
plot_conv_stud_heston_dict2 = dict(
    path=path_heston, x_axis="network_size", x_log=True, y_log=True,
    save_path=path_heston)


path_BS = '{}conv-study-BS-saved_models/'.format(data_path)
params_list_convstud_BS = []
for _ode_nn in ode_nn:
    param_dict_convstud_BS = {
        'epochs': [100],
        'batch_size': [20],
        'save_every': [10],
        'learning_rate': [0.001],
        'test_size': [0.2],
        'training_size': training_size,
        'seed': [398],
        'hidden_size': [10],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [_ode_nn],
        'readout_nn': [_ode_nn],
        'enc_nn': [_ode_nn],
        'use_rnn': [False],
        'func_appl_X': [[]],
        'solver': ["euler"],
        'weight': [0.5],
        'weight_decay': [1.],
        'dataset': ["BlackScholes"],
        'dataset_id': [None],
        'plot': [True],
        'paths_to_plot': [(0,)],
        'saved_models_path': [path_BS],
        'evaluate': [True]
    }
    params_list_convstud_BS += get_parameter_array(
        param_dict=param_dict_convstud_BS)
params_list_convstud_BS *= 5

plot_conv_stud_BS_dict1 = dict(
    path=path_BS, x_axis="training_size", x_log=True, y_log=True,
    save_path=path_BS)
plot_conv_stud_BS_dict2 = dict(
    path=path_BS, x_axis="network_size", x_log=True, y_log=True,
    save_path=path_BS)


path_OU = '{}conv-study-OU-saved_models/'.format(data_path)
params_list_convstud_OU = []
for _ode_nn in ode_nn:
    param_dict_convstud_OU = {
        'epochs': [100],
        'batch_size': [20],
        'save_every': [10],
        'learning_rate': [0.001],
        'test_size': [0.2],
        'training_size': training_size,
        'seed': [398],
        'hidden_size': [10],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [_ode_nn],
        'readout_nn': [_ode_nn],
        'enc_nn': [_ode_nn],
        'use_rnn': [False],
        'func_appl_X': [[]],
        'solver': ["euler"],
        'weight': [0.5],
        'weight_decay': [1.],
        'dataset': ["OrnsteinUhlenbeck"],
        'dataset_id': [None],
        'plot': [True],
        'paths_to_plot': [(0,)],
        'saved_models_path': [path_OU],
        'evaluate': [True]
    }
    params_list_convstud_OU += get_parameter_array(
        param_dict=param_dict_convstud_OU)
params_list_convstud_OU *= 5

plot_conv_stud_OU_dict1 = dict(
    path=path_OU, x_axis="training_size", x_log=True, y_log=True,
    save_path=path_OU)
plot_conv_stud_OU_dict2 = dict(
    path=path_OU, x_axis="network_size", x_log=True, y_log=True,
    save_path=path_OU)


# ------------------------------------------------------------------------------
# training of Heston without Feller
df_overview, filename = get_dataset_overview()
data_ids = []
for index, row in df_overview.iterrows():
    if 'HestonWOFeller' in row['name']:
        data_ids.append(row['id'])
path_HestonWOF = '{}saved_models_HestonWOFeller/'.format(data_path)
ode_nn = ((50, 'tanh'), (50, 'tanh'))
param_dict_HestonWOFeller = {
    'epochs': [200],
    'batch_size': [100],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [10],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [ode_nn],
    'readout_nn': [ode_nn],
    'enc_nn': [ode_nn],
    'use_rnn': [False],
    'func_appl_X': [[]],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'dataset': ['HestonWOFeller'],
    'dataset_id': data_ids,
    'plot': [True],
    'paths_to_plot': [(0, 1, 2, 3, 4,)],
    'evaluate': [True],
    'saved_models_path': [path_HestonWOF],
}
params_list_HestonWOFeller = get_parameter_array(
    param_dict=param_dict_HestonWOFeller)


# ------------------------------------------------------------------------------
# training of Combined stock models
path_combined_OU_BS = '{}saved_models_combined_OU_BS/'.format(data_path)
_ode_nn = ((100, 'tanh'), (100, 'tanh'))
param_dict_combined_OU_BS = {
    'epochs': [200],
    'batch_size': [100],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [10],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_ode_nn],
    'readout_nn': [_ode_nn],
    'enc_nn': [_ode_nn],
    'use_rnn': [False],
    'func_appl_X': [[]],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'dataset': ['combined_OrnsteinUhlenbeck_BlackScholes'],
    'plot': [True],
    'paths_to_plot': [(0, 1, 2, 3, 4,)],
    'evaluate': [True],
    'saved_models_path': [path_combined_OU_BS],
}
params_list_combined_OU_BS = get_parameter_array(
    param_dict=param_dict_combined_OU_BS)


# ------------------------------------------------------------------------------
# training on sine-drift BS dataset
path_sine_BS = '{}saved_models_sine_BS/'.format(data_path)
df_overview, filename = get_dataset_overview()
data_ids = []
for index, row in df_overview.iterrows():
    if 'sine_BlackScholes' in row['name']:
        data_ids.append(row['id'])
_ode_nn = ((400, 'tanh'), (400, 'tanh'))
param_dict_sine_BS = {
    'epochs': [100],
    'batch_size': [100],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [10],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_ode_nn],
    'readout_nn': [_ode_nn],
    'enc_nn': [_ode_nn],
    'use_rnn': [False],
    'func_appl_X': [[]],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'dataset': ["sine_BlackScholes"],
    'dataset_id': data_ids,
    'plot': [True],
    'paths_to_plot': [(0, 1, 2, 3, 4,)],
    'evaluate': [True],
    'saved_models_path': [path_sine_BS],
}
params_list_sine_BS = get_parameter_array(param_dict=param_dict_sine_BS)


# ------------------------------------------------------------------------------
# training of GRU-ODE-Bayes on 3 standard datasets
path_GRU_comparison = '{}saved_models_GRUODEBayes-comparison/'.format(data_path)
params_list_GRUODEBayes = []
param_dict_GRUODEBayes = {
    'epochs': [100],
    'batch_size': [20],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [50, 100],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [None],
    'readout_nn': [None],
    'enc_nn': [None],
    'use_rnn': [False],
    'func_appl_X': [[]],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'dataset': ["BlackScholes", "Heston", "OrnsteinUhlenbeck"],
    'dataset_id': [None],
    'plot': [True],
    'paths_to_plot': [(0, 1, 2, 3, 4,)],
    'evaluate': [True],
    'other_model': ['GRU_ODE_Bayes'],
    'saved_models_path': [path_GRU_comparison],
    'GRU_ODE_Bayes-impute': [True, False],
    'GRU_ODE_Bayes-logvar': [True, False],
    'GRU_ODE_Bayes-mixing': [0.0001, 0.5],
}
params_list_GRUODEBayes += get_parameter_array(
    param_dict=param_dict_GRUODEBayes)

# for comparison: NJ-ODE
ode_nn = ((50, 'tanh'), (50, 'tanh'))
param_dict_GRUODEBayes2 = {
    'epochs': [100],
    'batch_size': [20],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [10],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [ode_nn],
    'readout_nn': [ode_nn],
    'enc_nn': [ode_nn],
    'use_rnn': [False],
    'func_appl_X': [[]],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'dataset': ["BlackScholes", "Heston", "OrnsteinUhlenbeck"],
    'dataset_id': [None],
    'plot': [True],
    'paths_to_plot': [(0, 1, 2, 3, 4,)],
    'evaluate': [True],
    'saved_models_path': [path_GRU_comparison],
}
params_list_GRUODEBayes += get_parameter_array(param_dict=param_dict_GRUODEBayes2)

overview_dict_GRUODEBayes = dict(
    path=path_GRU_comparison,
    params_extract_desc=('dataset', "other_model",
                         'network_size', 'training_size',
                         'hidden_size', "GRU_ODE_Bayes-mixing",
                         "GRU_ODE_Bayes-logvar", "GRU_ODE_Bayes-impute",
                         "GRU_ODE_Bayes-mixing"),
    val_test_params_extract=(("max", "epoch", "epoch", "epochs_trained"),
                             ("min", "evaluation_mean_diff",
                              "evaluation_mean_diff", "eval_metric_min"),
                             ("last", "evaluation_mean_diff",
                              "evaluation_mean_diff", "eval_metric_last"),
                             ("average", "evaluation_mean_diff",
                              "evaluation_mean_diff", "eval_metric_average")
                             ),
    sortby=['dataset', 'eval_metric_min'], )


# ------------------------------------------------------------------------------
# training on climate dataset for cross validation
path_NJODE1_climate = '{}saved_models_NJODE1-climate/'.format(data_path)
params_list_NJODE1_climate = []
_ode_nn = ((50, 'tanh'), (50, 'tanh'))
param_dict_NJODE1_climate1 = {
    'epochs': [200],
    'batch_size': [100],
    'save_every': [1],
    'learning_rate': [0.001],
    'hidden_size': [10],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_ode_nn],
    'readout_nn': [_ode_nn],
    'enc_nn': [_ode_nn],
    'use_rnn': [False],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'dataset': ["climate"],
    'data_index': [0, 1, 2, 3, 4],
    'delta_t': [0.1],
    'saved_models_path': [path_NJODE1_climate],
}
params_list_NJODE1_climate += get_parameter_array(
    param_dict=param_dict_NJODE1_climate1)

_ode_nn = ((400, 'tanh'), (400, 'tanh'))
param_dict_NJODE1_climate2 = {
    'epochs': [200],
    'batch_size': [100],
    'save_every': [1],
    'learning_rate': [0.001],
    'hidden_size': [50],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_ode_nn],
    'readout_nn': [_ode_nn],
    'enc_nn': [_ode_nn],
    'use_rnn': [False],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'dataset': ["climate"],
    'data_index': [0, 1, 2, 3, 4],
    'delta_t': [0.1],
    'saved_models_path': [path_NJODE1_climate],
}
params_list_NJODE1_climate += get_parameter_array(
    param_dict=param_dict_NJODE1_climate2)

# for comparison: GRU-ODE-Bayes with suggested hyper-params
param_dict_NJODE1_climate3 = {
    'epochs': [50],
    'batch_size': [100],
    'save_every': [1],
    'learning_rate': [0.001],
    'hidden_size': [50],
    'bias': [True],
    'dropout_rate': [0.2],
    'ode_nn': [None],
    'readout_nn': [None],
    'enc_nn': [None],
    'use_rnn': [False],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [0, 1, 2, 3, 4],
    'dataset': ["climate"],
    'data_index': [1],
    'delta_t': [0.1],
    'other_model': ['GRU_ODE_Bayes'],
    'GRU_ODE_Bayes-impute': [False],
    'GRU_ODE_Bayes-logvar': [True],
    'GRU_ODE_Bayes-mixing': [1e-4],
    'GRU_ODE_Bayes-p_hidden': [25],
    'GRU_ODE_Bayes-prep_hidden': [10],
    'GRU_ODE_Bayes-cov_hidden': [50],
    'saved_models_path': [path_NJODE1_climate],
}
params_list_NJODE1_climate += get_parameter_array(
    param_dict=param_dict_NJODE1_climate3)


overview_dict_NJODE1_climate = dict(
    params_extract_desc=('dataset', 'network_size', 'dropout_rate',
                         'hidden_size', 'data_index'),
    val_test_params_extract=(("max", "epoch", "epoch", "epochs_trained"),
                             ("min", "eval_metric",
                              "eval_metric", "eval_metric_min"),
                             ("min", "test_metric",
                              "test_metric", "test_metric_min"),
                             ("min", "eval_metric",
                              "test_metric", "test_metric_evaluation_min"),
                             ("min", "eval_loss",
                              "test_metric", "test_metric_eval_loss_min"),
                             ), )

crossval_dict_NJODE1_climate = dict(
    path=path_NJODE1_climate, early_stop_after_epoch=100,
    save_path='{}cross_val.csv'.format(path_NJODE1_climate),
    param_combinations=(
        {'hidden_size': 10, 'dropout_rate': 0.1},
        {'hidden_size': 50, 'dropout_rate': 0.1}), )



# ------------------------------------------------------------------------------
# training on physionet dataset for cross validation
path_NJODE1_physionet = '{}saved_models_NJODE1_physionet_comparison/'.format(data_path)
network_size = [50, 200]
ode_nn = [((size, 'tanh'), (size, 'tanh')) for size in network_size]
params_list_NJODE1_physionet = []
for _ode_nn in ode_nn:
    param_dict_NJODE1_physionet = {
        'epochs': [175],
        'batch_size': [50],
        'save_every': [1],
        'learning_rate': [0.001],
        'hidden_size': [41],
        'bias': [True],
        'dropout_rate': [0.1],
        'ode_nn': [_ode_nn],
        'readout_nn': [_ode_nn],
        'enc_nn': [_ode_nn],
        'use_rnn': [False],
        'solver': ["euler"],
        'weight': [0.5],
        'weight_decay': [1.],
        'dataset': ["physionet"],
        'quantization': [0.016],
        'n_samples': [8000],
        'saved_models_path': [path_NJODE1_physionet],
    }
    params_list_NJODE1_physionet += get_parameter_array(
        param_dict=param_dict_NJODE1_physionet)
params_list_NJODE1_physionet *= 5

overview_dict_NJODE1_physionet = dict(
    path=path_NJODE1_physionet,
    params_extract_desc=('dataset', 'network_size', 'dropout_rate',
                         'hidden_size', 'data_index'),
    val_test_params_extract=(("max", "epoch", "epoch", "epochs_trained"),
                             ("min", "eval_metric",
                              "eval_metric", "eval_metric_min"),
                             ("min", "eval_metric_2",
                              "eval_metric_2", "eval_metric_2_min"),
                             ), )

crossval_dict_NJODE1_physionet = dict(
    path=path_NJODE1_physionet,
    save_path='{}cross_val.csv'.format(path_NJODE1_physionet),
    param_combinations=({'network_size': 50},
                        {'network_size': 200},),
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
                'eval_metric_2_last', 'eval_metric_2_eval_min'), )


