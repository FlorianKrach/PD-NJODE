"""
author: Florian Krach
"""


from configs.config_utils import get_parameter_array, get_dataset_overview, \
    makedirs, data_path, training_data_path



# ==============================================================================
#                        NJmodel - TRAINING PARAM DICTS
# ==============================================================================




# ------------------------------------------------------------------------------
# --- training of NJmodel on BM and BS
_nn = ((100, 'tanh'),)
NJmodel_models_path = "{}saved_models_NJmodel/".format(data_path)

param_dict_NJmodel = {
    'epochs': [200],
    'batch_size': [200],
    'save_every': [1],
    'learning_rate': [0.01, 0.001],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [None,],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [None],
    'readout_nn': [None],
    'enc_nn': [_nn],
    'func_appl_X': [[]],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'dataset': [
        "BlackScholes",
    ],
    'dataset_id': [None],
    'which_loss': ['easy',],
    'coord_wise_tau': [False,],
    'use_y_for_ode': [True,],
    'use_rnn': [True, False],
    'input_sig': [True, False],
    'level': [2,],
    'masked': [False],
    'plot': [True],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'plot_same_yaxis': [True],
    'other_model': ['NJmodel'],
    'saved_models_path': [NJmodel_models_path],
}
param_list_NJmodel = get_parameter_array(param_dict=param_dict_NJmodel)

param_dict_NJmodel1 = {
    'epochs': [200],
    'batch_size': [200],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [50,],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [_nn],
    'readout_nn': [None, _nn],
    'enc_nn': [_nn],
    'func_appl_X': [[]],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'dataset': [
        "BlackScholes",
    ],
    'dataset_id': [None],
    'which_loss': ['easy',],
    'coord_wise_tau': [False,],
    'use_y_for_ode': [True,],
    'use_rnn': [True, False],
    'input_sig': [True, False],
    'level': [2,],
    'masked': [False],
    'plot': [True],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'plot_same_yaxis': [True],
    'saved_models_path': [NJmodel_models_path],
}
param_list_NJmodel += get_parameter_array(param_dict=param_dict_NJmodel1)


param_dict_NJmodel2 = {
    'epochs': [200],
    'batch_size': [200],
    'save_every': [1],
    'learning_rate': [0.001],
    'test_size': [0.2],
    'seed': [398],
    'hidden_size': [None,],
    'bias': [True],
    'dropout_rate': [0.1],
    'ode_nn': [None],
    'readout_nn': [None],
    'enc_nn': [((2000, 'tanh'),), ((200, 'tanh'), (200, 'tanh'),)],
    'func_appl_X': [[]],
    'solver': ["euler"],
    'weight': [0.5],
    'weight_decay': [1.],
    'dataset': [
        "BlackScholes",
    ],
    'dataset_id': [None],
    'which_loss': ['easy',],
    'coord_wise_tau': [False,],
    'use_y_for_ode': [True,],
    'use_rnn': [True],
    'input_sig': [True],
    'level': [2,],
    'masked': [False],
    'plot': [True],
    'evaluate': [True],
    'paths_to_plot': [(0,1,2,3,4,)],
    'plot_same_yaxis': [True],
    'other_model': ['NJmodel'],
    'saved_models_path': [NJmodel_models_path],
}
param_list_NJmodel += get_parameter_array(param_dict=param_dict_NJmodel2)

overview_dict_NJmodel = dict(
    ids_from=1, ids_to=len(param_list_NJmodel),
    path=NJmodel_models_path,
    params_extract_desc=('data_dict', 'dataset', 'other_model',
                         'readout_nn', 'ode_nn',
                         'learning_rate',
                         'hidden_size', 'batch_size', 'which_loss',
                         'use_rnn', 'input_sig', 'level', ),
    val_test_params_extract=(
        ("max", "epoch", "epoch", "epochs_trained"),
        ("min", "evaluation_mean_diff",
         "evaluation_mean_diff", "evaluation_mean_diff_min"),
        ("min", "eval_loss", "eval_loss", "eval_loss_min"),
    ),
    sortby=["evaluation_mean_diff_min"],
)

plot_paths_NJmodel_dict = {
    'model_ids': [11, 18],
    'saved_models_path': NJmodel_models_path,
    'which': 'best', 'paths_to_plot': [0,1,2,3,4],
    'save_extras': {'bbox_inches': 'tight', 'pad_inches': 0.01},}



if __name__ == '__main__':
    pass
