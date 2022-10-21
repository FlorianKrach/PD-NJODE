"""
author: Florian Krach

this file contains the configs for the Limit Order Book (LOB) experiments
"""

import numpy as np

from configs.config_utils import get_parameter_array, get_dataset_overview, \
    makedirs, data_path, training_data_path


# ==============================================================================
# Global variables
LOB_data_path = '{}LOB-raw_data/'.format(training_data_path)
LOB_data_path2 = '{}LOB-raw_data2/'.format(training_data_path)
LOB2_data_path = '{}LOB2-raw_data/'.format(training_data_path)


# ==============================================================================
#                           LOB - DATASET DICTS
# ==============================================================================
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
#                        LOB - TRAINING PARAM DICTS
# ==============================================================================
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
