"""
author: Florian Krach

This file contains all configs to run the experiments for parameter filtering
using an input-output NJODE model.
"""


import numpy as np

from configs.config_utils import get_parameter_array, get_dataset_overview, \
    makedirs, data_path, training_data_path


# ==============================================================================
# DATASET DICTS
# ------------------------------------------------------------------------------
PF_BSUP_dict1 = {
    'model_name': "BlackScholesUncertainParams",
    'nb_paths': 20000, 'maturity': 1,
    'nb_steps': 100, 'obs_perc': 0.1,
    'dimension': 3, 'S0': 1,
    'drift_dist': {'dist': 'fixed', 'params': {'value': 0.05,},},
    'volatility_dist': {'dist': 'uniform', 'params': {'low': 0.05, 'high': 0.5},},
    'input_coords': [0], 'output_coords': [1,2],
}

PF_BSUP_dict1_test = {
    'model_name': "BlackScholesUncertainParams",
    'nb_paths': 5000, 'maturity': 1,
    'nb_steps': 100, 'obs_perc': 0.1,
    'dimension': 3, 'S0': 1,
    'drift_dist': {'dist': 'fixed', 'params': {'value': 0.05,},},
    'volatility_dist': {'dist': 'uniform', 'params': {'low': 0.05, 'high': 0.5},},
    'input_coords': [0], 'output_coords': [1,2],
    'test': True,
}


PF_BSUP_dict2 = {
    'model_name': "BlackScholesUncertainParams",
    'nb_paths': 100000, 'maturity': 1,
    'nb_steps': 100, 'obs_perc': 0.1,
    'dimension': 3, 'S0': 1,
    'drift_dist': {'dist': 'normal', 'params': {'loc': 0.05, 'scale': 0.1},},
    'volatility_dist': {'dist': 'uniform', 'params': {'low': 0.05, 'high': 0.5},},
    'input_coords': [0], 'output_coords': [1,2],
    'num_particles': 1000,
}

PF_BSUP_dict2_test = {
    'model_name': "BlackScholesUncertainParams",
    'nb_paths': 5000, 'maturity': 1,
    'nb_steps': 100, 'obs_perc': 0.1,
    'dimension': 3, 'S0': 1,
    'drift_dist': {'dist': 'normal', 'params': {'loc': 0.05, 'scale': 0.1},},
    'volatility_dist': {'dist': 'uniform', 'params': {'low': 0.05, 'high': 0.5},},
    'input_coords': [0], 'output_coords': [1,2],
    'num_particles': 1000,
}


for i, p in enumerate([0.01, 0.025, 0.05, 0.1, 0.25, 0.5]):
    d1 = {
        'model_name': "BlackScholesUncertainParams",
        'nb_paths': 100000, 'maturity': 1,
        'nb_steps': 1000, 'obs_perc': p,
        'dimension': 3, 'S0': 1,
        'drift_dist': {'dist': 'normal', 'params': {'loc': 0.05, 'scale': 0.1},},
        'volatility_dist': {'dist': 'uniform', 'params': {'low': 0.05, 'high': 0.5},},
        'input_coords': [0], 'output_coords': [1,2],
        'num_particles': 1000,
    }
    d2 = {
        'model_name': "BlackScholesUncertainParams",
        'nb_paths': 5000, 'maturity': 1,
        'nb_steps': 1000, 'obs_perc': p,
        'dimension': 3, 'S0': 1,
        'drift_dist': {'dist': 'normal', 'params': {'loc': 0.05, 'scale': 0.1},},
        'volatility_dist': {'dist': 'uniform', 'params': {'low': 0.05, 'high': 0.5},},
        'input_coords': [0], 'output_coords': [1,2],
        'num_particles': 1000,
    }
    exec("PF_BSUP_dict_CS1_{}=d1".format(i + 1))
    exec("PF_BSUP_dict_CS1_test_{}=d2".format(i + 1))


for i, T in enumerate([0.1, 1, 2, 5, 10, 20]):
    d1 = {
        'model_name': "BlackScholesUncertainParams",
        'nb_paths': 100000, 'maturity': T,
        'nb_steps': 100, 'obs_perc': 0.1,
        'dimension': 3, 'S0': 1,
        'drift_dist': {'dist': 'normal', 'params': {'loc': 0.05, 'scale': 0.1},},
        'volatility_dist': {'dist': 'uniform', 'params': {'low': 0.05, 'high': 0.5},},
        'input_coords': [0], 'output_coords': [1,2],
        'num_particles': 1000,
    }
    d2 = {
        'model_name': "BlackScholesUncertainParams",
        'nb_paths': 5000, 'maturity': T,
        'nb_steps': 100, 'obs_perc': 0.1,
        'dimension': 3, 'S0': 1,
        'drift_dist': {'dist': 'normal', 'params': {'loc': 0.05, 'scale': 0.1},},
        'volatility_dist': {'dist': 'uniform', 'params': {'low': 0.05, 'high': 0.5},},
        'input_coords': [0], 'output_coords': [1,2],
        'num_particles': 1000,
    }
    exec("PF_BSUP_dict_CS2_{}=d1".format(i + 1))
    exec("PF_BSUP_dict_CS2_test_{}=d2".format(i + 1))



# ------------------------------------------------------------------------------
PF_BMwUD_dict1 = {
    'model_name': "BMwithUncertainDrift",
    'nb_paths': 20000, 'maturity': 1,
    'nb_steps': 100, 'obs_perc': 0.1,
    'dimension': 2, 'S0': 0.,
    'volatility': 0.2,
    'drift_mean':  0.05, 'drift_std': 0.1,
    'input_coords': [0], 'output_coords': [1],
}

PF_BMwUD_dict1_test = {
    'model_name': "BMwithUncertainDrift",
    'nb_paths': 5000, 'maturity': 1,
    'nb_steps': 100, 'obs_perc': 0.1,
    'dimension': 2, 'S0': 0.,
    'volatility': 0.2,
    'drift_mean':  0.05, 'drift_std': 0.1,
    'input_coords': [0], 'output_coords': [1],
}

# ------------------------------------------------------------------------------
IO_BM_Filter_dict_1 = {
    'model_name': "BMFiltering",
    'nb_paths': 40000, 'nb_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'alpha': 1, 'dimension': 2,
    'input_coords': [0], 'output_coords': [1],
    'IO_version': True,
}
IO_BM_Filter_dict_1_test = {
    'model_name': "BMFiltering",
    'nb_paths': 4000, 'nb_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'alpha': 1, 'dimension': 2,
    'input_coords': [0], 'output_coords': [1],
    'IO_version': True,
}

# ------------------------------------------------------------------------------
IO_BS_dict = {
    'model_name': "BlackScholes",
    'drift': 2., 'volatility': 0.3,
    'nb_paths': 20000, 'nb_steps': 100,
    'S0': 1, 'maturity': 1., 'dimension': 1,
    'obs_perc': 0.1,
}

IO_BS_dict_test = {
    'model_name': "BlackScholes",
    'drift': 2., 'volatility': 0.3,
    'nb_paths': 4000, 'nb_steps': 100,
    'S0': 1, 'maturity': 1., 'dimension': 1,
    'obs_perc': 0.1,
}


# ------------------------------------------------------------------------------
PF_CIR_dict1 = {
    'model_name': "CIRUncertainParams",
    'nb_paths': 100000, 'maturity': 1,
    'nb_steps': 100, 'obs_perc': 0.1,
    'dimension': 4, 'S0': 1,
    'a_dist': {'dist': 'uniform', 'params': {'low': 0.2, 'high': 2},},
    'b_dist': {'dist': 'uniform', 'params': {'low': 1., 'high': 5.},},
    'volatility_dist': {'dist': 'uniform', 'params': {'low': 0.05, 'high': 0.5},},
    'sin_coeff': None,
    'input_coords': [0], 'output_coords': [1,2,3],
    'num_particles': 1000,
}

PF_CIR_dict1_test = {
    'model_name': "CIRUncertainParams",
    'nb_paths': 4000, 'maturity': 1,
    'nb_steps': 100, 'obs_perc': 0.1,
    'dimension': 4, 'S0': 1,
    'a_dist': {'dist': 'uniform', 'params': {'low': 0.2, 'high': 2},},
    'b_dist': {'dist': 'uniform', 'params': {'low': 1., 'high': 5.},},
    'volatility_dist': {'dist': 'uniform', 'params': {'low': 0.05, 'high': 0.5},},
    'sin_coeff': None,
    'input_coords': [0], 'output_coords': [1,2,3],
    'num_particles': 1000,
}


PF_CIR_dict2 = {
    'model_name': "CIRUncertainParams",
    'nb_paths': 100000, 'maturity': 1,
    'nb_steps': 100, 'obs_perc': 0.1,
    'dimension': 4, 'S0': 1,
    'a_dist': {'dist': 'uniform', 'params': {'low': 0.2, 'high': 2},},
    'b_dist': {'dist': 'uniform', 'params': {'low': 1., 'high': 5.},},
    'volatility_dist': {'dist': 'uniform', 'params': {'low': 0.05, 'high': 0.5},},
    'sin_coeff': 2*np.pi,
    'input_coords': [0], 'output_coords': [1,2,3],
    'num_particles': 1000,
}

PF_CIR_dict2_test = {
    'model_name': "CIRUncertainParams",
    'nb_paths': 4000, 'maturity': 1,
    'nb_steps': 100, 'obs_perc': 0.1,
    'dimension': 4, 'S0': 1,
    'a_dist': {'dist': 'uniform', 'params': {'low': 0.2, 'high': 2},},
    'b_dist': {'dist': 'uniform', 'params': {'low': 1., 'high': 5.},},
    'volatility_dist': {'dist': 'uniform', 'params': {'low': 0.05, 'high': 0.5},},
    'sin_coeff': 2*np.pi,
    'input_coords': [0], 'output_coords': [1,2,3],
    'num_particles': 1000,
}


PF_CIR_dict3 = {
    'model_name': "CIRUncertainParams",
    'nb_paths': 100000, 'maturity': 1,
    'nb_steps': 100, 'obs_perc': 0.1,
    'dimension': 4, 'S0': 1,
    'a_dist': {'dist': 'uniform', 'params': {'low': 2., 'high': 3.},},
    'b_dist': {'dist': 'uniform', 'params': {'low': 1., 'high': 2.},},
    'volatility_dist': {'dist': 'uniform', 'params': {'low': 1., 'high': 2.},},
    'sin_coeff': None,
    'input_coords': [0], 'output_coords': [1,2,3],
    'num_particles': 1000,
}

PF_CIR_dict3_test = {
    'model_name': "CIRUncertainParams",
    'nb_paths': 4000, 'maturity': 1,
    'nb_steps': 100, 'obs_perc': 0.1,
    'dimension': 4, 'S0': 1,
    'a_dist': {'dist': 'uniform', 'params': {'low': 2., 'high': 3.},},
    'b_dist': {'dist': 'uniform', 'params': {'low': 1., 'high': 2.},},
    'volatility_dist': {'dist': 'uniform', 'params': {'low': 1., 'high': 2.},},
    'sin_coeff': None,
    'input_coords': [0], 'output_coords': [1,2,3],
    'num_particles': 1000,
}


PF_CIR_dict4 = {
    'model_name': "CIRUncertainParams",
    'nb_paths': 100000, 'maturity': 1,
    'nb_steps': 100, 'obs_perc': 0.1,
    'dimension': 4, 'S0': 1,
    'a_dist': {'dist': 'uniform', 'params': {'low': 2., 'high': 3.},},
    'b_dist': {'dist': 'uniform', 'params': {'low': 1., 'high': 2.},},
    'volatility_dist': {'dist': 'uniform', 'params': {'low': 1., 'high': 2.},},
    'sin_coeff': 2*np.pi,
    'input_coords': [0], 'output_coords': [1,2,3],
    'num_particles': 1000,
}

PF_CIR_dict4_test = {
    'model_name': "CIRUncertainParams",
    'nb_paths': 4000, 'maturity': 1,
    'nb_steps': 100, 'obs_perc': 0.1,
    'dimension': 4, 'S0': 1,
    'a_dist': {'dist': 'uniform', 'params': {'low': 2., 'high': 3.},},
    'b_dist': {'dist': 'uniform', 'params': {'low': 1., 'high': 2.},},
    'volatility_dist': {'dist': 'uniform', 'params': {'low': 1., 'high': 2.},},
    'sin_coeff': 2*np.pi,
    'input_coords': [0], 'output_coords': [1,2,3],
    'num_particles': 1000,
}


# ------------------------------------------------------------------------------
IO_BMClass_dict = {
    'model_name': "BMClassification",
    'nb_paths': 40000, 'nb_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'alpha': 0., 'dimension': 2,
    'input_coords': [0], 'output_coords': [1],
}

IO_BMClass_dict_test = {
    'model_name': "BMClassification",
    'nb_paths': 4000, 'nb_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'alpha': 0., 'dimension': 2,
    'input_coords': [0], 'output_coords': [1],
}


# ==============================================================================
# TRAINING PARAM DICTS
# ------------------------------------------------------------------------------
# --- Black Scholes with Uncertain Parameters
PF_BSUP_models_path = "{}saved_models_BSUncertainParams/".format(data_path)
param_list_PF_BSUP = []

for size in [100]:
    for act in ['tanh', 'relu']:
        _nn = ((size, act),)
        param_dict_PF_BSUP = {
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
            'use_rnn': [True, False],
            'residual_enc_dec': [True, False,],
            'func_appl_X': [[]],
            'solver': ["euler"],
            'weight': [0.5],
            'weight_decay': [1.],
            'input_sig': [True, False],
            'level': [3, ],
            'data_dict': ["PF_BSUP_dict1", ],
            'test_data_dict': ["PF_BSUP_dict1_test", ],
            'which_loss': ['IO', ],
            'coord_wise_tau': [False,],
            'use_y_for_ode': [False],
            'masked': [False],
            'plot': [True],
            'evaluate': [True],
            'paths_to_plot': [(0,1,2,3,4,)],
            'plot_same_yaxis': [False],
            'use_cond_exp': [True],
            'saved_models_path': [PF_BSUP_models_path],
        }
        param_list_PF_BSUP += get_parameter_array(
            param_dict=param_dict_PF_BSUP)

overview_dict_PF_BSUP = dict(
    ids_from=1, ids_to=len(param_list_PF_BSUP),
    path=PF_BSUP_models_path,
    params_extract_desc=('data_dict', 'network_size', 'readout_nn',
                         'activation_function_1',
                         'hidden_size', 'batch_size', 'which_loss',
                         'input_sig', 'level', 'coord_wise_tau',
                         'use_rnn',
                         'use_y_for_ode', 'residual_enc_dec'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "evaluation_mean_diff",
         "evaluation_mean_diff", "evaluation_mean_diff_min"),
        ("min", "val_loss", "val_loss", "val_loss_min"),
        ("min", "val_loss", "evaluation_mean_diff",
         "evaluation_mean_diff_at_val_loss_min"),
    ),
    sortby=["val_loss_min"],
)

plot_paths_PF_BSUP_dict = {
    'model_ids': [], 'saved_models_path': PF_BSUP_models_path,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4,5],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}

# ----------
PF_BSUP_models_path2 = "{}saved_models_BSUncertainParams2/".format(data_path)
param_list_PF_BSUP2 = []

for size in [100]:
    for act in ['tanh', 'relu']:
        _nn = ((size, act),)
        param_dict_PF_BSUP2 = {
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
            'use_rnn': [True, False],
            'residual_enc_dec': [True, False,],
            'func_appl_X': [[]],
            'solver': ["euler"],
            'weight': [0.5],
            'weight_decay': [1.],
            'input_sig': [True, False],
            'level': [3, ],
            'data_dict': ["PF_BSUP_dict2", ],
            'test_data_dict': ["PF_BSUP_dict2_test", ],
            'which_loss': ['IO', ],
            'coord_wise_tau': [False,],
            'use_y_for_ode': [False],
            'masked': [False],
            'plot': [True],
            'evaluate': [True],
            'paths_to_plot': [(0,1,2,3,4,)],
            'plot_same_yaxis': [False],
            'use_cond_exp': [True],
            'saved_models_path': [PF_BSUP_models_path2],
        }
        param_list_PF_BSUP2 += get_parameter_array(
            param_dict=param_dict_PF_BSUP2)

overview_dict_PF_BSUP2 = dict(
    ids_from=1, ids_to=len(param_list_PF_BSUP2),
    path=PF_BSUP_models_path2,
    params_extract_desc=('data_dict', 'network_size', 'readout_nn',
                         'activation_function_1',
                         'hidden_size', 'batch_size', 'which_loss',
                         'input_sig', 'level', 'coord_wise_tau',
                         'use_rnn',
                         'use_y_for_ode', 'residual_enc_dec'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "evaluation_mean_diff",
         "evaluation_mean_diff", "evaluation_mean_diff_min"),
        ("min", "val_loss", "val_loss", "val_loss_min"),
        ("min", "val_loss", "evaluation_mean_diff",
         "evaluation_mean_diff_at_val_loss_min"),
    ),
    sortby=["val_loss_min"],
)

plot_paths_PF_BSUP_dict2 = {
    'model_ids': [17], 'saved_models_path': PF_BSUP_models_path2,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4,5],
    'ylabels': ['$X_t$', '$\\mu$', '$\\sigma$'],
    'legendlabels': ['true path','I/O NJODE','PF','observed'],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},
    'plot_error_dist': {
        'model_names': ['I/O NJODE', 'PF',
                        'financial estimator'],
        'additional_ref_models': ['financial estimator'],
        'eval_times': [0.5, 1.]
    },
    'plot_only_evaluate': True,
}

plot_paths_PF_BSUP_dict2_1 = {
    'model_ids': [17], 'saved_models_path': PF_BSUP_models_path2,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4,5],
    'ylabels': ['$X_t$', '$\\mu$', '$\\sigma$'],
    'legendlabels': ['true path','I/O NJODE','financial estimator','observed'],
    'ref_model_to_use': 'financial estimator',
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},
    'plot_only_evaluate': False,
}



# ----------
# convergence study
PF_BSUP_models_path_CS = "{}saved_models_BSUncertainParams_CS/".format(data_path)
param_list_PF_BSUP_CS = []

for CS in [1,2]:
    for dataset_id in range(1, 7):
        dsd = "PF_BSUP_dict_CS{}_{}".format(CS, dataset_id)
        dsd_test = "PF_BSUP_dict_CS{}_test_{}".format(CS, dataset_id)
        _nn = ((100, "relu"),)
        param_dict_PF_BSUP_CS = {
            'epochs': [200],
            'batch_size': [200],
            'save_every': [1],
            'learning_rate': [0.001],
            'test_size': [0.2],
            'seed': [398],
            'hidden_size': [200,],
            'bias': [True],
            'dropout_rate': [0.1],
            'ode_nn': [_nn],
            'readout_nn': [_nn,],
            'enc_nn': [_nn],
            'use_rnn': [True,],
            'residual_enc_dec': [True,],
            'func_appl_X': [[]],
            'solver': ["euler"],
            'weight': [0.5],
            'weight_decay': [1.],
            'input_sig': [True,],
            'level': [3, ],
            'data_dict': [dsd, ],
            'test_data_dict': [dsd_test, ],
            'which_loss': ['IO', ],
            'coord_wise_tau': [False,],
            'use_y_for_ode': [False],
            'masked': [False],
            'plot': [True],
            'evaluate': [True],
            'paths_to_plot': [(0,1,2,3,4,)],
            'plot_same_yaxis': [False],
            'use_cond_exp': [True],
            'saved_models_path': [PF_BSUP_models_path_CS],
        }
        param_list_PF_BSUP_CS += get_parameter_array(
            param_dict=param_dict_PF_BSUP_CS)

overview_dict_PF_BSUP_CS = dict(
    ids_from=1, ids_to=len(param_list_PF_BSUP_CS),
    path=PF_BSUP_models_path_CS,
    params_extract_desc=('data_dict', 'network_size', 'readout_nn',
                         'activation_function_1',
                         'hidden_size', 'batch_size', 'which_loss',
                         'input_sig', 'level', 'coord_wise_tau',
                         'use_rnn',
                         'use_y_for_ode', 'residual_enc_dec'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "evaluation_mean_diff",
         "evaluation_mean_diff", "evaluation_mean_diff_min"),
        ("min", "val_loss", "val_loss", "val_loss_min"),
        ("min", "val_loss", "evaluation_mean_diff",
         "evaluation_mean_diff_at_val_loss_min"),
    ),
    sortby=["val_loss_min"],
)

plot_paths_PF_BSUP_CS_dict = {
    'model_ids': list(range(1,11)), 'saved_models_path': PF_BSUP_models_path_CS,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4],
    'ylabels': ['$X_t$', '$\\mu$', '$\\sigma$'],
    'legendlabels': ['true path','I/O NJODE','PF','observed'],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},
    'plot_error_dist': {
        'model_names': ['I/O NJODE', 'PF',
                        'financial estimator'],
        'additional_ref_models': ['financial estimator'],
        'eval_times': ["mid", "end"]
    },
    'plot_only_evaluate': False,
}




# ------------------------------------------------------------------------------
# --- Brownian Motion with Uncertain Drift
PF_BMwUD_models_path = "{}saved_models_BMwUncertainDrift/".format(data_path)
param_list_PF_BMwUD = []

for size in [100]:
    for act in ['tanh', 'relu']:
        _nn = ((size, act),)
        param_dict_PF_BMwUD = {
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
            'use_rnn': [True, False],
            'residual_enc_dec': [True, False,],
            'func_appl_X': [[]],
            'solver': ["euler"],
            'weight': [0.5],
            'weight_decay': [1.],
            'input_sig': [True, False],
            'level': [3, ],
            'data_dict': ["PF_BMwUD_dict1", ],
            'test_data_dict': ["PF_BMwUD_dict1_test", ],
            'which_loss': ['IO', ],
            'coord_wise_tau': [False,],
            'use_y_for_ode': [False],
            'masked': [False],
            'plot': [True],
            'evaluate': [True],
            'paths_to_plot': [(0,1,2,3,4,)],
            'plot_same_yaxis': [False],
            'use_cond_exp': [True],
            'saved_models_path': [PF_BMwUD_models_path],
        }
        # param_list_PF_BMwUD += get_parameter_array(
        #     param_dict=param_dict_PF_BMwUD)


# BM uncertain drift, with variance prediction
_nn = ((100, 'tanh'),)
param_dict_PF_BMwUD = {
    'epochs': [200],
    'batch_size': [200],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [100, 200],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'readout_nn': [_nn,],
    'enc_nn': [_nn],
    'use_rnn': [True, False],
    'residual_enc_dec': [True,],
    'func_appl_X': [["power-2"]],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'input_sig': [True, False],
    'level': [3, ],
    'data_dict': ["PF_BMwUD_dict1", ],
    'test_data_dict': ["PF_BMwUD_dict1_test", ],
    'which_loss': ['IO', ],
    'coord_wise_tau': [False,],
    'use_y_for_ode': [False],
    'masked': [False],
    'plot': [True],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'plot_variance': [True],
    'plot_same_yaxis': [False],
    'use_cond_exp': [True],
    'saved_models_path': [PF_BMwUD_models_path],
}
param_list_PF_BMwUD += get_parameter_array(
    param_dict=param_dict_PF_BMwUD)

overview_dict_PF_BMwUD = dict(
    ids_from=1, ids_to=len(param_list_PF_BMwUD),
    path=PF_BMwUD_models_path,
    params_extract_desc=('data_dict', 'network_size', 'readout_nn',
                         'activation_function_1', 'func_appl_X',
                         'hidden_size', 'batch_size', 'which_loss',
                         'input_sig', 'level', 'coord_wise_tau',
                         'use_rnn',
                         'use_y_for_ode', 'residual_enc_dec'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "evaluation_mean_diff",
         "evaluation_mean_diff", "evaluation_mean_diff_min"),
        ("min", "val_loss", "val_loss", "val_loss_min"),
        ("min", "val_loss", "evaluation_mean_diff",
         "evaluation_mean_diff_at_val_loss_min"),
    ),
    sortby=["val_loss_min"],
)

plot_paths_PF_BMwUD_dict = {
    'model_ids': [1], 'saved_models_path': PF_BMwUD_models_path,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4,5],
    'ylabels': ['$X_t$', '$\\mu$'],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},
    'plot_error_dist': {
        'model_names': ['I/O NJODE', 'true cond. expectation'],
        'eval_times': [1.]
    },
    'plot_only_evaluate': False,
}


# ------------------------------------------------------------------------------
# --- Brownian Motion Filtering
IO_BMFilter_models_path = "{}saved_models_IO_BMFilter/".format(data_path)
param_list_IO_BMFilter = []

_nn = ((100, 'tanh'),)
param_dict_IO_BMFilter = {
    'epochs': [200],
    'batch_size': [200],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [100, 200],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'readout_nn': [_nn,],
    'enc_nn': [_nn],
    'use_rnn': [True, False],
    'residual_enc_dec': [True,],
    'func_appl_X': [[]],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'input_sig': [True, False],
    'level': [3, ],
    'data_dict': ["IO_BM_Filter_dict_1", ],
    'test_data_dict': ["IO_BM_Filter_dict_1_test", ],
    'which_loss': ['very_easy', ],
    'coord_wise_tau': [False,],
    'use_y_for_ode': [False],
    'masked': [False],
    'plot': [True],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'plot_variance': [False],
    'plot_same_yaxis': [False],
    'use_cond_exp': [True],
    'saved_models_path': [IO_BMFilter_models_path],
}
param_list_IO_BMFilter += get_parameter_array(
    param_dict=param_dict_IO_BMFilter)

overview_dict_IO_BMFilter = dict(
    ids_from=1, ids_to=len(param_list_IO_BMFilter),
    path=IO_BMFilter_models_path,
    params_extract_desc=('data_dict', 'network_size', 'readout_nn',
                         'activation_function_1', 'func_appl_X',
                         'hidden_size', 'batch_size', 'which_loss',
                         'input_sig', 'level', 'coord_wise_tau',
                         'use_rnn',
                         'use_y_for_ode', 'residual_enc_dec'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "evaluation_mean_diff",
         "evaluation_mean_diff", "evaluation_mean_diff_min"),
        ("min", "val_loss", "val_loss", "val_loss_min"),
        ("min", "val_loss", "evaluation_mean_diff",
         "evaluation_mean_diff_at_val_loss_min"),
    ),
    sortby=["val_loss_min"],
)

plot_paths_IO_BMFilter_dict = {
    'model_ids': [1], 'saved_models_path': IO_BMFilter_models_path,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4,5],
    'ylabels': ['$Y_t$', '$X_t$'],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}


# ------------------------------------------------------------------------------
# --- Black Scholes: speed of learning jumps
IO_BS_learn_jumps_model_path = "{}saved_models_BS_learn_jumps/".format(data_path)
param_list_IO_BS_LJ = []

for size in [100]:
    for act in ['tanh', 'relu']:
        _nn = ((size, act),)
        param_dict_IO_BS_LJ = {
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
            'readout_nn': [_nn,],
            'enc_nn': [_nn],
            'use_rnn': [True],
            'residual_enc_dec': [True,],
            'func_appl_X': [[]],
            'solver': ["euler"],
            'weight': [0.5],
            'weight_decay': [1.],
            'input_sig': [True],
            'level': [3, ],
            'data_dict': ["IO_BS_dict", ],
            'test_data_dict': ["IO_BS_dict_test", ],
            'which_loss': ['IO', "easy"],
            'which_val_loss': ['jump'],
            'coord_wise_tau': [False,],
            'use_y_for_ode': [False],
            'masked': [False],
            'plot': [True],
            'evaluate': [True],
            'paths_to_plot': [(0,1,2,3,4,)],
            'plot_same_yaxis': [False],
            'use_cond_exp': [True],
            'saved_models_path': [IO_BS_learn_jumps_model_path],
        }
        param_list_IO_BS_LJ += get_parameter_array(
            param_dict=param_dict_IO_BS_LJ)

overview_dict_IO_BS_LJ = dict(
    ids_from=1, ids_to=len(param_list_IO_BS_LJ),
    path=IO_BS_learn_jumps_model_path,
    params_extract_desc=('data_dict', 'network_size', 'readout_nn',
                         'activation_function_1',
                         'hidden_size', 'batch_size', 'which_loss',
                         'input_sig', 'level', 'coord_wise_tau',
                         'use_rnn',
                         'use_y_for_ode', 'residual_enc_dec'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "evaluation_mean_diff",
         "evaluation_mean_diff", "evaluation_mean_diff_min"),
        ("min", "val_loss", "val_loss", "val_loss_min"),
        ("min", "val_loss", "evaluation_mean_diff",
         "evaluation_mean_diff_at_val_loss_min"),
    ),
    sortby=["val_loss_min"],
)

plot_loss_IO_BS_LJ = {
    'model_ids': [4, 3], 'saved_models_path': IO_BS_learn_jumps_model_path,
    'logy': True, 'relative_error': False,
    'ylab': '$\\Psi_{\\operatorname{jump}}$',
    'names': ['original loss ($L^1$ aspect)', 'I/O loss (pure $L^2$)'],
    'filename': 'loss_comparison.pdf',
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}


# ------------------------------------------------------------------------------
# --- CIR with uncertain parameters
PF_CIR_models_path = "{}saved_models_CIRUncertainParams/".format(data_path)
param_list_PF_CIR = []

for ds, dst in [("PF_CIR_dict1", "PF_CIR_dict1_test"),
                ("PF_CIR_dict2", "PF_CIR_dict2_test"),
                ("PF_CIR_dict3", "PF_CIR_dict3_test"),
                ("PF_CIR_dict4", "PF_CIR_dict4_test")]:
    for act in ['tanh', 'relu']:
        _nn = ((size, act),)
        param_dict_PF_CIR = {
            'epochs': [200],
            'batch_size': [200],
            'save_every': [1],
            'learning_rate': [0.001],
            'test_size': [0.2],
            'seed': [398],
            'hidden_size': [200,],
            'bias': [True],
            'dropout_rate': [0.1],
            'ode_nn': [_nn],
            'readout_nn': [_nn],
            'enc_nn': [_nn],
            'use_rnn': [True, False],
            'residual_enc_dec': [True, False,],
            'func_appl_X': [[]],
            'solver': ["euler"],
            'weight': [0.5],
            'weight_decay': [1.],
            'input_sig': [True, False],
            'level': [3, ],
            'data_dict': [ds, ],
            'test_data_dict': [dst, ],
            'which_loss': ['IO', ],
            'coord_wise_tau': [False,],
            'use_y_for_ode': [False],
            'masked': [False],
            'plot': [True],
            'evaluate': [True],
            'paths_to_plot': [(0,1,2,3,4,)],
            'plot_same_yaxis': [False],
            'use_cond_exp': [True],
            'saved_models_path': [PF_CIR_models_path],
        }
        param_list_PF_CIR += get_parameter_array(
            param_dict=param_dict_PF_CIR)

overview_dict_PF_CIR = dict(
    ids_from=1, ids_to=len(param_list_PF_CIR),
    path=PF_CIR_models_path,
    params_extract_desc=('data_dict', 'network_size', 'readout_nn',
                         'activation_function_1',
                         'hidden_size', 'batch_size', 'which_loss',
                         'input_sig', 'level', 'coord_wise_tau',
                         'use_rnn',
                         'use_y_for_ode', 'residual_enc_dec'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "evaluation_mean_diff",
         "evaluation_mean_diff", "evaluation_mean_diff_min"),
        ("min", "val_loss", "val_loss", "val_loss_min"),
        ("min", "val_loss", "evaluation_mean_diff",
         "evaluation_mean_diff_at_val_loss_min"),
        ("min", "val_loss", "test_loss",
         "test_loss_at_val_loss_min"),
    ),
    sortby=['data_dict', "val_loss_min"],
)

plot_paths_PF_CIR_dict = {
    'model_ids': [11, 27, 43, 59], 'saved_models_path': PF_CIR_models_path,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4,5],
    'ylabels': ['$X_t$', '$a$', '$b$', '$\\sigma$'],
    'legendlabels': ['true path','I/O NJODE','PF','observed'],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},
    'plot_error_dist': {
        'model_names': ['I/O NJODE', 'PF',],
        'eval_times': [0.5, 1.]
    },
    'plot_only_evaluate': True,
}


# ------------------------------------------------------------------------------
# --- Brownian Motion Classification
IO_BMClass_models_path = "{}saved_models_BMClassification/".format(data_path)
param_list_IO_BMClass = []

for ds, dst in [("IO_BMClass_dict", "IO_BMClass_dict_test"),]:
    for act in ['tanh', 'relu']:
        _nn = ((size, act),)
        param_dict_IO_BMClass = {
            'epochs': [200],
            'batch_size': [200],
            'save_every': [1],
            'learning_rate': [0.001],
            'test_size': [0.2],
            'seed': [398],
            'hidden_size': [200,],
            'bias': [True],
            'dropout_rate': [0.1],
            'ode_nn': [_nn],
            'readout_nn': [_nn],
            'enc_nn': [_nn],
            'use_rnn': [True, False],
            'residual_enc_dec': [True, False,],
            'func_appl_X': [[]],
            'solver': ["euler"],
            'weight': [0.5],
            'weight_decay': [1.],
            'input_sig': [True, False],
            'level': [3, ],
            'data_dict': [ds, ],
            'test_data_dict': [dst, ],
            'which_loss': ['IO', ],
            'coord_wise_tau': [False,],
            'use_y_for_ode': [False],
            'masked': [False],
            'plot': [True],
            'evaluate': [True],
            'paths_to_plot': [(0,1,2,3,4,)],
            'plot_same_yaxis': [False],
            'use_cond_exp': [True],
            'saved_models_path': [IO_BMClass_models_path],
        }
        param_list_IO_BMClass += get_parameter_array(
            param_dict=param_dict_IO_BMClass)

overview_dict_IO_BMClass = dict(
    ids_from=1, ids_to=len(param_list_IO_BMClass),
    path=IO_BMClass_models_path,
    params_extract_desc=('data_dict', 'network_size', 'readout_nn',
                         'activation_function_1',
                         'hidden_size', 'batch_size', 'which_loss',
                         'input_sig', 'level', 'coord_wise_tau',
                         'use_rnn',
                         'use_y_for_ode', 'residual_enc_dec'),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "evaluation_mean_diff",
         "evaluation_mean_diff", "evaluation_mean_diff_min"),
        ("min", "val_loss", "val_loss", "val_loss_min"),
        ("min", "val_loss", "evaluation_mean_diff",
         "evaluation_mean_diff_at_val_loss_min"),
        ("min", "val_loss", "test_loss",
         "test_loss_at_val_loss_min"),
    ),
    sortby=['data_dict', "val_loss_min"],
)

plot_paths_IO_BMClass_dict = {
    'model_ids': [9, 12], 'saved_models_path': IO_BMClass_models_path,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4,5],
    'ylabels': ['$W_t$', '$S_t$'],
    'legendlabels': ['true path','I/O NJODE','true conditional expectation','observed'],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},
    'plot_only_evaluate': True,
}



if __name__ == '__main__':
    pass
