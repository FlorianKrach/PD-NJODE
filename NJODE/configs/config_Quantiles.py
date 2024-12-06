"""
author: Florian Krach
"""


import numpy as np

from configs.config_utils import get_parameter_array, get_dataset_overview, \
    makedirs, data_path, training_data_path


# ==============================================================================
# DATASET DICTS
# ------------------------------------------------------------------------------
BM_Quantiles = {
    'model_name': "BMandQuantiles",
    'nb_paths': 20000, 'nb_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'dimension': 1,
}

BM_Quantiles_test = {
    'model_name': "BMandQuantiles",
    'nb_paths': 4000, 'nb_steps': 100,
    'maturity': 1., 'obs_perc': 0.1,
    'dimension': 1,
}


# ==============================================================================
# TRAINING PARAM DICTS
# ------------------------------------------------------------------------------
# --- BM with Quantiles dataset
BMQ_models_path = "{}saved_models_BMandQuantiles/".format(data_path)
param_list_BMQ = []
for _nn in [((200, 'tanh'),), ]:
    param_dict_BMQ = {
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
        'use_rnn': [True, False],
        'func_appl_X': [[]],
        'solver': ["euler"],
        'weight': [0.5],
        'weight_decay': [1.],
        'input_sig': [False, True],
        'level': [3],
        'residual_enc_dec': [False, True],
        'data_dict': ['BM_Quantiles'],
        'test_data_dict': ['BM_Quantiles_test'],
        'which_loss': ['quantile_jump', 'quantile'],
        'loss_quantiles': [[0.1, 0.5, 0.9]],
        'which_val_loss': ['quantile'],
        'plot': [True],
        'evaluate': [True],
        'paths_to_plot': [(0,1,2,3,4,)],
        'saved_models_path': [BMQ_models_path],
        'use_current_y_for_ode': [False, ],
        'use_y_for_ode': [False, ],
        'use_observation_as_input': [
            True,
            "lambda x: np.random.random(1) < 1-x/200",
        ],
    }
    param_list_BMQ += get_parameter_array(param_dict=param_dict_BMQ)

overview_dict_BMQ = dict(
    ids_from=1, ids_to=len(param_list_BMQ),
    path=BMQ_models_path,
    params_extract_desc=('data_dict', 'test_data_dict',
                         'network_size', 'readout_nn',
                         'activation_function_1',
                         'hidden_size', 'batch_size', 'which_loss',
                         'use_rnn', 'input_sig', 'level',
                         'residual_enc_dec',
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

plot_paths_BMQ_dict = {
    'model_ids': [9, 25, 29], 'saved_models_path': BMQ_models_path,
    'which': 'best', 'paths_to_plot': [4,5,6,7,8,],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}



if __name__ == '__main__':
    pass
