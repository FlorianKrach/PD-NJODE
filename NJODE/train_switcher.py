"""
author: Florian Krach
"""

import train
import climate_train
import physionet_train
import LOB_train
import retrain_LOB_classifier
import synthetic_datasets


def train_switcher(**params):
    """
    function to call the correct train function depending on the dataset. s.t.
    parallel training easily works altough different fuctions need to be called
    :param params: all params needed by the train function, as passed by
            parallel_training
    :return: function call to the correct train function
    """
    if 'dataset' not in params:
        if 'data_dict' not in params:
            raise KeyError('the "dataset" needs to be specified')
        else:
            data_dict = params["data_dict"]
            if isinstance(data_dict, str):
                from configs import config
                data_dict = eval("config."+data_dict)
            params["dataset"] = data_dict["model_name"]
    if params['dataset'] in list(synthetic_datasets.DATASETS) or \
            'combined' in params['dataset'] or 'FBM[' in params['dataset']:
        return train.train(**params)
    elif params['dataset'] in ['climate', 'Climate']:
        return climate_train.train(**params)
    elif params['dataset'] in ['physionet', 'Physionet']:
        return physionet_train.train(**params)
    elif params['dataset'] in ['LOB',]:
        return LOB_train.train(**params)
    elif params['dataset'] in ['retrain_LOB',]:
        return retrain_LOB_classifier.train(**params)
    else:
        raise ValueError('the specified "dataset" is not supported')

