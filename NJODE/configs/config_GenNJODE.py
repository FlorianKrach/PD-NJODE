"""
author: Florian Krach

This file contains all configs to run the experiments for the Generative NJODE.
"""
import copy
import os.path

import numpy as np

from configs.config_utils import get_parameter_array, get_dataset_overview, \
    makedirs, data_path, training_data_path

################################################################################
# -------- Geometric Brownian Motion (GBM)
################################################################################

# ------------------------------------------------------------------------------
# datasets configs
# ------------------------------------------------------------------------------

# --- joint instantaneous training set ---
GBM_joint_training_dict = {
    'model_name': "BlackScholesGenCoeff",
    'nb_paths': 20000, 'nb_steps': 100,
    'maturity': 1.,
    'obs_perc': 0.1, # 0.1
    'dimension': 1, 'drift': 2., 'volatility': 0.3, 'S0': 1.,
    'special_dataset_features': 'X_inc', 'divide_by_t': True,
    'input_coords': [1], 'output_coords': [0],}

# only one step ahead (full observations)
GBM_joint_training_dict2 = {
    'model_name': "BlackScholesGenCoeff",
    'nb_paths': 20000, 'nb_steps': 100,
    'maturity': 1.,
    'obs_perc': 1., # 0.1
    'dimension': 1, 'drift': 2., 'volatility': 0.3, 'S0': 1.,
    'special_dataset_features': 'X_inc', 'divide_by_t': True,
    'input_coords': [1], 'output_coords': [0],}

# --- individual instantaneous training sets ---
GBM_drift_lim_training_dict = {
    'model_name': "BlackScholes_Xinc_lim",
    'nb_paths': 20000, 'nb_steps': 100,
    'maturity': 1.,
    'obs_perc': 0.1, # 0.1
    'dimension': 1, 'drift': 2., 'volatility': 0.3, 'S0': 1.,
    'special_dataset_features': 'X_inc', 'divide_by_t': True,
    'input_coords': [1], 'output_coords': [0],}

GBM_vol_lim_training_dict = {
    'model_name': "BlackScholes_Z_lim",
    'nb_paths': 20000, 'nb_steps': 100,
    'maturity': 1.,
    'obs_perc': 0.1, # 0.1
    'dimension': 1, 'drift': 2., 'volatility': 0.3, 'S0': 1.,
    'special_dataset_features': 'Z', 'divide_by_t': True,
    'input_coords': [2], 'output_coords': [1],}

# --- individual baseline training sets ---
GBM_drift_training_dict = {
    'model_name': "BlackScholes",
    'nb_paths': 20000, 'nb_steps': 100,
    'maturity': 1.,
    'obs_perc': 0.1, # 0.1
    'dimension': 1, 'drift': 2., 'volatility': 0.3, 'S0': 1.,}

GBM_vol_training_dict = {
    'model_name': "BlackScholes_Z",
    'nb_paths': 20000, 'nb_steps': 100,
    'maturity': 1.,
    'obs_perc': 0.1, # 0.1
    'dimension': 1, 'drift': 2., 'volatility': 0.3, 'S0': 1.,
    'special_dataset_features': 'Z', 'divide_by_t': False,
    'input_coords': [0,2], 'output_coords': [1],}


# ------------------------------------------------------------------------------
# training configs
# ------------------------------------------------------------------------------
# --- joint training of instantaneous coefficients ---
models_path_GenNJODE_GBM = "{}saved_models_GenNJODE_GBM/".format(data_path)
_nn = ((50, 'relu'),)
param_list_GenNJODE_GBM = []

param_dict_GenNJODE_GBM1 = {
    'epochs': [200],
    'batch_size': [200],
    'input_coords': [[1]],
    'output_coords': [[0]],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [1],  # for dataset splitting
    'training_seed': [2],
    'hidden_size': [100],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'readout_nn': [_nn],
    'enc_nn': [_nn],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'data_dict': ["GBM_joint_training_dict",],
    'dataset_id': [None],
    'which_loss': ['gen_coeffs',],
    'use_y_for_ode': [False, ],
    'use_rnn': [True],
    'input_sig': [False],
    'level': [2,],
    'masked': [False],
    'plot': [True],
    'plot_vola': [True],
    'square_model_output': [False],
    'coord_wise_tau': [False, ],
    'evaluate': [True],
    'use_cond_exp': [True,],
    'paths_to_plot': [(0,1,2,3,4,)],
    'saved_models_path': [models_path_GenNJODE_GBM],
    'residual_enc_dec': [True],  # True
    'ode_input_scaling_func': ["identity"],
    'compute_variance': ['volatility'],
    'input_var_t_helper': [False],
}
param_list_GenNJODE_GBM1 = get_parameter_array(
    param_dict=param_dict_GenNJODE_GBM1)
param_list_GenNJODE_GBM += param_list_GenNJODE_GBM1


# --- joint training of baseline coefficients ---
param_dict_GenNJODE_GBM2 = {
    'epochs': [200],
    'batch_size': [200],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [1],  # for dataset splitting
    'training_seed': [2],
    'hidden_size': [100],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'readout_nn': [_nn],
    'enc_nn': [_nn],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'data_dict': ["GBM_drift_training_dict",],
    'dataset_id': [None],
    'which_loss': ['easy',],
    'use_y_for_ode': [False, ],
    'use_rnn': [True],
    'input_sig': [False],
    'level': [2,],
    'masked': [False],
    'plot': [True],
    'plot_variance': [True],
    'square_model_output': [False],
    'coord_wise_tau': [False, ],
    'evaluate': [True],
    'use_cond_exp': [True,],
    'paths_to_plot': [(0,1,2,3,4,)],
    'saved_models_path': [models_path_GenNJODE_GBM],
    'residual_enc_dec': [True],
    'ode_input_scaling_func': ["identity"],
    'compute_variance': ['covariance'],
    'input_var_t_helper': [True],
    'var_weight': [1.],
    'which_var_loss': [1],
}
param_list_GenNJODE_GBM2 = get_parameter_array(
    param_dict=param_dict_GenNJODE_GBM2)
param_list_GenNJODE_GBM += param_list_GenNJODE_GBM2


# --- individual training of instantaneous coefficients ---
param_dict_GenNJODE_GBM3_1 = {
    'epochs': [200],
    'batch_size': [200],
    'input_coords': [[1]],
    'output_coords': [[0]],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [1],  # for dataset splitting
    'training_seed': [2],
    'hidden_size': [100],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'readout_nn': [_nn],
    'enc_nn': [_nn],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'data_dict': ["GBM_drift_lim_training_dict",],
    'dataset_id': [None],
    'which_loss': ['drift_lim',],
    'use_y_for_ode': [False, ],
    'use_rnn': [True],
    'input_sig': [False],
    'level': [2,],
    'masked': [False],
    'plot': [True],
    'plot_vola': [False],
    'square_model_output': [False],
    'coord_wise_tau': [False, ],
    'evaluate': [True],
    'use_cond_exp': [True,],
    'paths_to_plot': [(0,1,2,3,4,)],
    'saved_models_path': [models_path_GenNJODE_GBM],
    'residual_enc_dec': [True],  # True
    'ode_input_scaling_func': ["identity"],
    'compute_variance': [False],
    'input_var_t_helper': [False],
}

param_dict_GenNJODE_GBM3_2 = {
    'epochs': [200],
    'batch_size': [200],
    'input_coords': [[2]],
    'output_coords': [[1]],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [1],  # for dataset splitting
    'training_seed': [2],
    'hidden_size': [100],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'readout_nn': [_nn],
    'enc_nn': [_nn],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'data_dict': ["GBM_vol_lim_training_dict",],
    'dataset_id': [None],
    'which_loss': ['vola_lim',],
    'use_y_for_ode': [False, ],
    'use_rnn': [True],
    'input_sig': [False],
    'level': [2,],
    'masked': [False],
    'plot': [True],
    'plot_vola': [False],
    'square_model_output': [True],
    'coord_wise_tau': [False, ],
    'evaluate': [True],
    'use_cond_exp': [True,],
    'paths_to_plot': [(0,1,2,3,4,)],
    'saved_models_path': [models_path_GenNJODE_GBM],
    'residual_enc_dec': [True],  # True
    'ode_input_scaling_func': ["identity"],
    'compute_variance': [False],
    'input_var_t_helper': [False],
}
param_list_GenNJODE_GBM3_1 = get_parameter_array(
    param_dict=param_dict_GenNJODE_GBM3_1)
param_list_GenNJODE_GBM3_2 = get_parameter_array(
    param_dict=param_dict_GenNJODE_GBM3_2)
param_list_GenNJODE_GBM3 = param_list_GenNJODE_GBM3_1 + \
    param_list_GenNJODE_GBM3_2
param_list_GenNJODE_GBM += param_list_GenNJODE_GBM3


# --- individual training of baseline coefficients ---
param_dict_GenNJODE_GBM4_1 = {
    'epochs': [200],
    'batch_size': [200],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [1],  # for dataset splitting
    'training_seed': [2],
    'hidden_size': [100],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'readout_nn': [_nn],
    'enc_nn': [_nn],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'data_dict': ["GBM_drift_training_dict",],
    'dataset_id': [None],
    'which_loss': ['easy',],
    'use_y_for_ode': [False, ],
    'use_rnn': [True],
    'input_sig': [False],
    'level': [2,],
    'masked': [False],
    'plot': [True],
    'plot_vola': [False],
    'square_model_output': [False],
    'coord_wise_tau': [False, ],
    'evaluate': [True],
    'use_cond_exp': [True,],
    'paths_to_plot': [(0,1,2,3,4,)],
    'saved_models_path': [models_path_GenNJODE_GBM],
    'residual_enc_dec': [True],  # True
    'ode_input_scaling_func': ["identity"],
    'compute_variance': [False],
    'input_var_t_helper': [False],
}

param_dict_GenNJODE_GBM4_2 = {
    'epochs': [200],
    'batch_size': [200],
    'input_coords': [[0, 2]],
    'output_coords': [[1]],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [1],  # for dataset splitting
    'training_seed': [2],
    'hidden_size': [100],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'readout_nn': [_nn],
    'enc_nn': [_nn],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'data_dict': ["GBM_vol_training_dict",],
    'dataset_id': [None],
    'which_loss': ['vola',],
    'use_y_for_ode': [False, ],
    'use_rnn': [True],
    'input_sig': [False],
    'level': [2,],
    'masked': [False],
    'plot': [True],
    'plot_vola': [False],
    'square_model_output': [True],
    'coord_wise_tau': [False, ],
    'evaluate': [True],
    'use_cond_exp': [True,],
    'paths_to_plot': [(0,1,2,3,4,)],
    'saved_models_path': [models_path_GenNJODE_GBM],
    'residual_enc_dec': [True],  # True
    'ode_input_scaling_func': ["identity"],
    'compute_variance': [False],
    'input_var_t_helper': [True],
}
param_list_GenNJODE_GBM4_1 = get_parameter_array(
    param_dict=param_dict_GenNJODE_GBM4_1)
param_list_GenNJODE_GBM4_2 = get_parameter_array(
    param_dict=param_dict_GenNJODE_GBM4_2)
param_list_GenNJODE_GBM4 = param_list_GenNJODE_GBM4_1 + \
    param_list_GenNJODE_GBM4_2
param_list_GenNJODE_GBM += param_list_GenNJODE_GBM4


# --- joint instantaneous coefficients (save later) ---
param_dict_GenNJODE_GBM1_2 = {
    'epochs': [200],
    'batch_size': [200],
    'input_coords': [[1]],
    'output_coords': [[0]],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [1],  # for dataset splitting
    'training_seed': [2],
    'hidden_size': [100],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'readout_nn': [_nn],
    'enc_nn': [_nn],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'data_dict': ["GBM_joint_training_dict",],
    'dataset_id': [None],
    'which_loss': ['gen_coeffs',],
    'use_y_for_ode': [False, ],
    'use_rnn': [True],
    'input_sig': [False],
    'level': [2,],
    'masked': [False],
    'plot': [True],
    'plot_vola': [True],
    'square_model_output': [False],
    'coord_wise_tau': [False, ],
    'evaluate': [True],
    'use_cond_exp': [True,],
    'paths_to_plot': [(0,1,2,3,4,)],
    'saved_models_path': [models_path_GenNJODE_GBM],
    'residual_enc_dec': [True],  # True
    'ode_input_scaling_func': ["identity"],
    'compute_variance': ['volatility'],
    'input_var_t_helper': [False],
    'save_best_from_epoch': [100],
}
param_list_GenNJODE_GBM1_2 = get_parameter_array(
    param_dict=param_dict_GenNJODE_GBM1_2)
param_list_GenNJODE_GBM += param_list_GenNJODE_GBM1_2


# --- individual instantaneous coefficients (save later) ---
param_dict_GenNJODE_GBM3_3 = {
    'epochs': [200],
    'batch_size': [200],
    'input_coords': [[2]],
    'output_coords': [[1]],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [1],  # for dataset splitting
    'training_seed': [2],
    'hidden_size': [100],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'readout_nn': [_nn],
    'enc_nn': [_nn],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'data_dict': ["GBM_vol_lim_training_dict",],
    'dataset_id': [None],
    'which_loss': ['vola_lim',],
    'use_y_for_ode': [False, ],
    'use_rnn': [True],
    'input_sig': [False],
    'level': [2,],
    'masked': [False],
    'plot': [True],
    'plot_vola': [False],
    'square_model_output': [True],
    'coord_wise_tau': [False, ],
    'evaluate': [True],
    'use_cond_exp': [True,],
    'paths_to_plot': [(0,1,2,3,4,)],
    'saved_models_path': [models_path_GenNJODE_GBM],
    'residual_enc_dec': [True],  # True
    'ode_input_scaling_func': ["identity"],
    'compute_variance': [False],
    'input_var_t_helper': [False],
    'save_best_from_epoch': [100],
}
param_list_GenNJODE_GBM3_3 = get_parameter_array(
    param_dict=param_dict_GenNJODE_GBM3_3)
param_list_GenNJODE_GBM3 += param_list_GenNJODE_GBM3_3
param_list_GenNJODE_GBM += param_list_GenNJODE_GBM3_3


# --- joint instantaneous coefficients - 1-step ahead predictions ---
param_dict_GenNJODE_GBM1_3 = {
    'epochs': [200],
    'batch_size': [200],
    'input_coords': [[1]],
    'output_coords': [[0]],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [1],  # for dataset splitting
    'training_seed': [2],
    'hidden_size': [100],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'readout_nn': [_nn],
    'enc_nn': [_nn],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'data_dict': ["GBM_joint_training_dict2",],
    'dataset_id': [None],
    'which_loss': ['gen_coeffs',],
    'use_y_for_ode': [False, ],
    'use_rnn': [True],
    'input_sig': [False],
    'level': [2,],
    'masked': [False],
    'plot': [True],
    'plot_vola': [True],
    'square_model_output': [False],
    'coord_wise_tau': [False, ],
    'evaluate': [True],
    'use_cond_exp': [True,],
    'paths_to_plot': [(0,1,2,3,4,)],
    'saved_models_path': [models_path_GenNJODE_GBM],
    'residual_enc_dec': [True],  # True
    'ode_input_scaling_func': ["identity"],
    'compute_variance': ['volatility'],
    'input_var_t_helper': [False],
    'save_best_from_epoch': [100],
}
param_list_GenNJODE_GBM1_3 = get_parameter_array(
    param_dict=param_dict_GenNJODE_GBM1_3)
param_list_GenNJODE_GBM += param_list_GenNJODE_GBM1_3


# --- overview of the training runs ---
overview_dict_GBM_genNJODE = dict(
    ids_from=1, ids_to=len(param_list_GenNJODE_GBM),
    path=models_path_GenNJODE_GBM,
    params_extract_desc=(
        'data_dict', 'network_size', 'readout_nn',
        'activation_function_1', 'hidden_size', 'batch_size', 'which_loss',
        'input_sig', 'level', 'coord_wise_tau', 'use_rnn', 'compute_variance',
        'use_y_for_ode', 'residual_enc_dec', 'ode_input_scaling_func',
        'input_var_t_helper', 'var_weight', 'which_var_loss',
        'save_best_from_epoch'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        # ("min", "evaluation_mean_diff",
        #  "evaluation_mean_diff", "evaluation_mean_diff_min"),
        ("min", "val_loss", "val_loss", "val_loss_min"),
        ("min", "val_loss", "evaluation_mean_diff",
         "evaluation_mean_diff_at_val_loss_min"),
    ),
    sortby=["val_loss_min"],
)


# ------------------------------------------------------------------------------
# path generation configs
# ------------------------------------------------------------------------------
# --- generation of paths with joint instantaneous coefficient estimation ---
save_gen_paths_path = os.path.join(models_path_GenNJODE_GBM, "generation1_1/")
generation_dict_GBM1_1_0 = dict(
    model_drift_path=models_path_GenNJODE_GBM,
    model_drift_id=1, model_drift_params=param_list_GenNJODE_GBM[0],
    model_drift_inputs=["X_inc", "X"],
    model_diffusion_path=None, model_diffusion_id=None,
    model_diffusion_params=None,
    model_diffusion_inputs=None,
    joint_model=True, load_best=True,
    start_X=np.array([1,]),
    init_times=None, init_X=None, start_M=None, init_M=None,
    training_data_dict="GBM_joint_training_dict", T=None, delta_t=None,
    nb_samples_gen=5000, max_paths_to_plot=1000,
    steps_ahead_prediction=0,
    instantaneous_coeff_estimation=True,
    drift_predicts_inc=True, return_coeffs=True,
    save_gen_paths_path=save_gen_paths_path,
    save_paths=False, plot_paths=True, plot_real_paths=True,
    gen_seed=3,
    estimate_GBM_params=True,
    plot_which_coeff_paths=(0,1,2,3,4),
    mu_function=lambda x,t: x*GBM_joint_training_dict['drift'],
    sigma_function=lambda x,t: x*GBM_joint_training_dict['volatility'],
    plot_dist_at_t_indices=(-1,), plot_dist_num_bins=50,
)

generation_dict_GBM1_1 = dict(
    model_drift_path=models_path_GenNJODE_GBM,
    model_drift_id=7, model_drift_params=param_list_GenNJODE_GBM[6],
    model_drift_inputs=["X_inc", "X"],
    model_diffusion_path=None, model_diffusion_id=None,
    model_diffusion_params=None,
    model_diffusion_inputs=None,
    joint_model=True, load_best=True,
    start_X=np.array([1,]),
    init_times=None, init_X=None, start_M=None, init_M=None,
    training_data_dict="GBM_joint_training_dict", T=None, delta_t=None,
    nb_samples_gen=5000, max_paths_to_plot=1000,
    steps_ahead_prediction=0,
    instantaneous_coeff_estimation=True,
    drift_predicts_inc=True, return_coeffs=True,
    save_gen_paths_path=save_gen_paths_path,
    save_paths=False, plot_paths=True, plot_real_paths=True,
    gen_seed=3,
    estimate_GBM_params=True,
    plot_which_coeff_paths=(0,1,2,3,4),
    mu_function=lambda x,t: x*GBM_joint_training_dict['drift'],
    sigma_function=lambda x,t: x*GBM_joint_training_dict['volatility'],
    plot_dist_at_t_indices=(50, -1,), plot_dist_num_bins=50,
)

# generate paths with some given start sequence
save_gen_paths_path = os.path.join(models_path_GenNJODE_GBM, "generation1_2/")
generation_dict_GBM1_2 = dict(
    model_drift_path=models_path_GenNJODE_GBM,
    model_drift_id=7, model_drift_params=param_list_GenNJODE_GBM[6],
    model_drift_inputs=["X_inc", "X"],
    model_diffusion_path=None, model_diffusion_id=None,
    model_diffusion_params=None,
    model_diffusion_inputs=None,
    joint_model=True, load_best=True,
    start_X=np.array([1,]),
    init_times=np.array([0.01*(i+1) for i in range(3*18+1)]),
    init_X=np.array(
        [1.0422280406354794, 1.0840698844267571, 1.1303867417174764,
         1.1466465501082568, 1.2579007746940876, 1.2679963869064905,
         1.223523650884504, 1.1826552860414932, 1.2660142222683544,
         1.2936363339150716, 1.3629876751910053, 1.3429938059550643,
         1.3764882090127855, 1.3968880400802894, 1.4009801030788294,
         1.475060890420235, 1.5158837723918606, 1.5371271327871587,
         1.5528858887171326, 1.6462720452005795, 1.7116880889955381,
         1.7386262223355775, 1.7323961274293356, 1.7588984822308626,
         1.722776667121706, 1.820948227419956, 1.8811155381712243,
         1.909577978862829, 1.8962677402883057, 1.9458712304666994,
         1.991190755244843, 1.9572712816632472, 1.986337441353595,
         1.984512375021064, 2.0342102946346365, 2.0599149746595526,
         2.193018911183549, 2.09857732057984, 2.079897452715525,
         2.210520057315031, 2.222251071877223, 2.3037417843172157,
         2.499984628371491, 2.6149329831564456, 2.6498985953667864,
         2.684146247777057, 2.7643726143605756, 2.87252864369634,
         2.9465526273926828, 2.997586726970451, 3.083519543346865,
         3.112657686223388, 3.2838949332510268, 3.2519294496495315,
         3.513919692425278,]).reshape(-1,1),
    start_M=None, init_M=None,
    training_data_dict="GBM_joint_training_dict", T=None, delta_t=None,
    nb_samples_gen=5000, max_paths_to_plot=1000,
    steps_ahead_prediction=0,
    instantaneous_coeff_estimation=True,
    drift_predicts_inc=True, return_coeffs=True,
    save_gen_paths_path=save_gen_paths_path,
    save_paths=False, plot_paths=True, plot_real_paths=False,
    gen_seed=3,
    estimate_GBM_params=True,
    plot_which_coeff_paths=(0,1,2,3,4),
    mu_function=lambda x,t: x*GBM_joint_training_dict['drift'],
    sigma_function=lambda x,t: x*GBM_joint_training_dict['volatility'],
    plot_dist_at_t_indices=(-1,), plot_dist_num_bins=50,
)

# generate paths where model was trained for 1-step ahead prediction only
generation_dict_GBM1_3 = dict(
    model_drift_path=models_path_GenNJODE_GBM,
    model_drift_id=9, model_drift_params=param_list_GenNJODE_GBM[8],
    model_drift_inputs=["X_inc", "X"],
    model_diffusion_path=None, model_diffusion_id=None,
    model_diffusion_params=None,
    model_diffusion_inputs=None,
    joint_model=True, load_best=True,
    start_X=np.array([1,]),
    init_times=None, init_X=None, start_M=None, init_M=None,
    training_data_dict="GBM_joint_training_dict2", T=None, delta_t=None,
    nb_samples_gen=5000, max_paths_to_plot=1000,
    steps_ahead_prediction=0,
    instantaneous_coeff_estimation=True,
    drift_predicts_inc=True, return_coeffs=True,
    save_gen_paths_path=save_gen_paths_path,
    save_paths=False, plot_paths=True, plot_real_paths=True,
    gen_seed=3,
    estimate_GBM_params=True,
    plot_which_coeff_paths=(0,1,2,3,4),
    mu_function=lambda x,t: x*GBM_joint_training_dict2['drift'],
    sigma_function=lambda x,t: x*GBM_joint_training_dict2['volatility'],
    plot_dist_at_t_indices=(-1,), plot_dist_num_bins=50,
)


# --- generation of paths with joint baseline coefficient estimation ---
save_gen_paths_path = os.path.join(models_path_GenNJODE_GBM, "generation2_1/")
generation_dict_GBM2_1 = dict(
    model_drift_path=models_path_GenNJODE_GBM,
    model_drift_id=2, model_drift_params=param_list_GenNJODE_GBM[1],
    model_drift_inputs=["X"],
    model_diffusion_path=None, model_diffusion_id=None,
    model_diffusion_params=None,
    model_diffusion_inputs=None,
    joint_model=True, load_best=True,
    start_X=np.array([1,]),
    init_times=None, init_X=None, start_M=None, init_M=None,
    training_data_dict="GBM_drift_training_dict", T=None, delta_t=None,
    nb_samples_gen=5000, max_paths_to_plot=1000,
    steps_ahead_prediction=1,
    instantaneous_coeff_estimation=False,
    drift_predicts_inc=False, return_coeffs=True,
    save_gen_paths_path=save_gen_paths_path,
    save_paths=False, plot_paths=True, plot_real_paths=True,
    gen_seed=3,
    estimate_GBM_params=True,
    plot_which_coeff_paths=(0,1,2,3,4),
    mu_function=lambda x,t: x*GBM_drift_training_dict['drift'],
    sigma_function=lambda x,t: x*GBM_drift_training_dict['volatility'],
    plot_dist_at_t_indices=(-1,), plot_dist_num_bins=50,
)


# --- generation of paths with individual instantaneous coefficient estimation ---
save_gen_paths_path = os.path.join(models_path_GenNJODE_GBM, "generation3_1/")
generation_dict_GBM3_1 = dict(
    model_drift_path=models_path_GenNJODE_GBM,
    model_drift_id=3, model_drift_params=param_list_GenNJODE_GBM[2],
    model_drift_inputs=["X_inc", "X"],
    model_diffusion_path=models_path_GenNJODE_GBM,
    model_diffusion_id=4, model_diffusion_params=param_list_GenNJODE_GBM[3],
    model_diffusion_inputs=["Z", "X"],
    joint_model=False, load_best=True,
    start_X=np.array([1,]),
    init_times=None, init_X=None, start_M=None, init_M=None,
    training_data_dict="GBM_drift_lim_training_dict", T=None, delta_t=None,
    nb_samples_gen=5000, max_paths_to_plot=1000,
    steps_ahead_prediction=0,
    instantaneous_coeff_estimation=True,
    drift_predicts_inc=True, return_coeffs=True,
    save_gen_paths_path=save_gen_paths_path,
    save_paths=False, plot_paths=True, plot_real_paths=True,
    gen_seed=3,
    estimate_GBM_params=True,
    plot_which_coeff_paths=(0,1,2,3,4),
    mu_function=lambda x,t: x*GBM_drift_lim_training_dict['drift'],
    sigma_function=lambda x,t: x*GBM_drift_lim_training_dict['volatility'],
    plot_dist_at_t_indices=(-1,), plot_dist_num_bins=50,
)
save_gen_paths_path = os.path.join(models_path_GenNJODE_GBM, "generation3_2/")
generation_dict_GBM3_2 = dict(
    model_drift_path=models_path_GenNJODE_GBM,
    model_drift_id=3, model_drift_params=param_list_GenNJODE_GBM[2],
    model_drift_inputs=["X_inc", "X"],
    model_diffusion_path=models_path_GenNJODE_GBM,
    model_diffusion_id=8, model_diffusion_params=param_list_GenNJODE_GBM[7],
    model_diffusion_inputs=["Z", "X"],
    joint_model=False, load_best=True,
    start_X=np.array([1,]),
    init_times=None, init_X=None, start_M=None, init_M=None,
    training_data_dict="GBM_drift_lim_training_dict", T=None, delta_t=None,
    nb_samples_gen=5000, max_paths_to_plot=1000,
    steps_ahead_prediction=0,
    instantaneous_coeff_estimation=True,
    drift_predicts_inc=True, return_coeffs=True,
    save_gen_paths_path=save_gen_paths_path,
    save_paths=False, plot_paths=True, plot_real_paths=True,
    gen_seed=3,
    estimate_GBM_params=True,
    plot_which_coeff_paths=(0,1,2,3,4),
    mu_function=lambda x,t: x*GBM_drift_lim_training_dict['drift'],
    sigma_function=lambda x,t: x*GBM_drift_lim_training_dict['volatility'],
    plot_dist_at_t_indices=(-1,), plot_dist_num_bins=50,
)


# --- generation of paths with individual instantaneous coefficient estimation ---
save_gen_paths_path = os.path.join(models_path_GenNJODE_GBM, "generation4_1/")
generation_dict_GBM4_1 = dict(
    model_drift_path=models_path_GenNJODE_GBM,
    model_drift_id=5, model_drift_params=param_list_GenNJODE_GBM[4],
    model_drift_inputs=["X"],
    model_diffusion_path=models_path_GenNJODE_GBM,
    model_diffusion_id=6, model_diffusion_params=param_list_GenNJODE_GBM[5],
    model_diffusion_inputs=["Z", "X"],
    joint_model=False, load_best=True,
    start_X=np.array([1,]),
    init_times=None, init_X=None, start_M=None, init_M=None,
    training_data_dict="GBM_drift_training_dict", T=None, delta_t=None,
    nb_samples_gen=5000, max_paths_to_plot=1000,
    steps_ahead_prediction=1,
    instantaneous_coeff_estimation=False,
    drift_predicts_inc=False, return_coeffs=True,
    save_gen_paths_path=save_gen_paths_path,
    save_paths=False, plot_paths=True, plot_real_paths=True,
    gen_seed=3,
    estimate_GBM_params=True,
    plot_which_coeff_paths=(0,1,2,3,4),
    mu_function=lambda x,t: x*GBM_drift_training_dict['drift'],
    sigma_function=lambda x,t: x*GBM_drift_training_dict['volatility'],
    plot_dist_at_t_indices=(-1,), plot_dist_num_bins=50,
)




################################################################################
# -------- Ornstein-Uhlenbeck
################################################################################

# --- joint instantaneous training set ---
OU_joint_training_dict = {
    'model_name': "OrnsteinUhlenbeckGenCoeff",
    'nb_paths': 20000, 'nb_steps': 100,
    'maturity': 1.,
    'obs_perc': 0.1, # 0.1
    'dimension': 1,
    'volatility': 1., 'mean': 3., 'speed': 2.,
    'S0': 1.,
    'special_dataset_features': 'X_inc', 'divide_by_t': True,
    'input_coords': [1], 'output_coords': [0],}

models_path_GenNJODE_OU = "{}saved_models_GenNJODE_OU/".format(data_path)
_nn = ((50, 'relu'),)
param_list_GenNJODE_OU = []

param_dict_GenNJODE_OU1 = {
    'epochs': [200],
    'batch_size': [200],
    'input_coords': [[1]],
    'output_coords': [[0]],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [1],  # for dataset splitting
    'training_seed': [2],
    'hidden_size': [100],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'readout_nn': [_nn],
    'enc_nn': [_nn],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'data_dict': ["OU_joint_training_dict",],
    'dataset_id': [None],
    'which_loss': ['gen_coeffs',],
    'use_y_for_ode': [False, ],
    'use_rnn': [True],
    'input_sig': [False],
    'level': [2,],
    'masked': [False],
    'plot': [True],
    'plot_vola': [True],
    'square_model_output': [False],
    'coord_wise_tau': [False, ],
    'evaluate': [True],
    'use_cond_exp': [True,],
    'paths_to_plot': [(0,1,2,3,4,)],
    'saved_models_path': [models_path_GenNJODE_OU],
    'residual_enc_dec': [True],  # True
    'ode_input_scaling_func': ["identity"],
    'compute_variance': ['volatility'],
    'input_var_t_helper': [False],
    'save_best_from_epoch': [0, 100],
}
param_list_GenNJODE_OU1 = get_parameter_array(
    param_dict=param_dict_GenNJODE_OU1)
param_list_GenNJODE_OU += param_list_GenNJODE_OU1

save_gen_paths_path = os.path.join(models_path_GenNJODE_OU, "generation1_1/")
generation_dict_OU1_1 = dict(
    model_drift_path=models_path_GenNJODE_OU,
    model_drift_id=1, model_drift_params=param_list_GenNJODE_OU[0],
    model_drift_inputs=["X_inc", "X"],
    model_diffusion_path=None, model_diffusion_id=None,
    model_diffusion_params=None,
    model_diffusion_inputs=None,
    joint_model=True, load_best=True,
    start_X=np.array([1,]),
    init_times=None, init_X=None, start_M=None, init_M=None,
    training_data_dict="OU_joint_training_dict", T=None, delta_t=None,
    nb_samples_gen=5000, max_paths_to_plot=1000,
    steps_ahead_prediction=0,
    instantaneous_coeff_estimation=True,
    drift_predicts_inc=True, return_coeffs=True,
    save_gen_paths_path=save_gen_paths_path,
    save_paths=False, plot_paths=True, plot_real_paths=True,
    gen_seed=3,
    estimate_OU_params=True,
    plot_which_coeff_paths=(0,1,2,3,4),
    mu_function=lambda x,t: -OU_joint_training_dict['speed'] * (x - OU_joint_training_dict['mean']),
    sigma_function=lambda x,t: OU_joint_training_dict['volatility'] * np.ones_like(x),
    plot_dist_at_t_indices=(50, -1,), plot_dist_num_bins=50,
)

generation_dict_OU1_2 = copy.deepcopy(generation_dict_OU1_1)
generation_dict_OU1_2['model_drift_id'] = 2






if __name__ == '__main__':
    pass
