"""
author: Florian Krach

this files contains the utilities that are shared by all config files
"""

import os
import pandas as pd
from sklearn.model_selection import ParameterGrid

# ==============================================================================
# Global variables
data_path = '../data/'
training_data_path = '{}training_data/'.format(data_path)


# ==============================================================================
# GLOBAL CLASSES
class SendBotMessage:
    def __init__(self):
        pass

    @staticmethod
    def send_notification(text, *args, **kwargs):
        print(text)


# ==============================================================================
# FUNCTIONS
def get_parameter_array(param_dict):
    """
    helper function to get a list of parameter-list with all combinations of
    parameters specified in a parameter-dict

    :param param_dict: dict with parameters
    :return: 2d-array with the parameter combinations
    """
    param_combs_dict_list = list(ParameterGrid(param_dict))
    return param_combs_dict_list


def get_dataset_overview(training_data_path=training_data_path):
    data_overview = '{}dataset_overview.csv'.format(
        training_data_path)
    makedirs(training_data_path)
    if not os.path.exists(data_overview):
        df_overview = pd.DataFrame(
            data=None, columns=['name', 'id', 'description'])
    else:
        df_overview = pd.read_csv(data_overview, index_col=0)
    return df_overview, data_overview


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


