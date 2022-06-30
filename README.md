# Path-Dependent Neural Jump ODEs

This repository is the official implementation of the paper
[Optimal Estimation of Generic Dynamics by Path-Dependent Neural Jump ODEs](https://arxiv.org/abs/2206.14284) which is the second part of the paper
[Neural Jump Ordinary Differential Equations: Consistent Continuous-Time Prediction and Filtering](https://openreview.net/forum?id=JFKR3WqwyXR).

The code is based on the [code of the first paper](https://github.com/HerreraKrachTeichmann/NJODE), 
but was developed further such that it is more user-friendly. 
All experiments from the first part can be run with this repo as well (see 
[Instructions for NJODE](#instructions-for-running-experiments-of-neural-jump-ordinary-differential-equations)).


## Requirements

This code was executed using Python 3.7.

To install requirements, download this Repo and cd into it. Then (e.g. in a new 
conda environment):

```sh
pip install -r requirements.txt
```

## Dataset Generation
go to the source directory:
```sh
cd NJODE
```

generate Poisson Point Process dataset:
```sh
python data_utils.py --dataset_name=PoissonPointProcess --dataset_params=poisson_pp_dict
```

generate fractional Brownian Motion (FBM) datasets:
```shell
python data_utils.py --dataset_name=FBM --dataset_params=FBM_1_dict
python data_utils.py --dataset_name=FBM --dataset_params=FBM_2_dict
python data_utils.py --dataset_name=FBM --dataset_params=FBM_3_dict
python data_utils.py --dataset_name=FBM --dataset_params=FBM_4_dict
```

generate 2-dim correlated BM dataset:
```sh
python data_utils.py --dataset_name=BM2DCorr --dataset_params=BM_2D_dict
```


generate BM and its Square dataset:
```sh
python data_utils.py --dataset_name=BMandVar --dataset_params=BM_and_Var_dict
python data_utils.py --dataset_name=BM --dataset_params=BM_dict
```


generate BS with dependent observation intensity dataset:
```sh
python data_utils.py --dataset_name=BlackScholes --dataset_params=BS_dep_intensity_dict
```


**REMARK**: the physionet and climate datasets are automatically downloaded when 
training a model on them for the first time (=> do not train multiple models at 
parallel at first time, since otherwise each will start the download and 
overwrite each others output. after download is complete, parallel training can 
be used normally!)


## Training & Testing
go to the source directory:
```sh
cd NJODE
```

(parallel) training of models:
```sh
python run.py --params=... --NB_JOBS=... --NB_CPUS=... --first_id=...
```

List of all flags:

- **params**: name of the params list (defined in config.py) to use for parallel run
- **NB_JOBS**: nb of parallel jobs to run with joblib
- **first_id**: First id of the given list / to start training of
- **get_overview**: name of the dict (defined in config.py) defining input for extras.get_training_overview
- **USE_GPU**: whether to use GPU for training
- **ANOMALY_DETECTION**: whether to run in torch debug mode
- **SEND**: whether to send results via telegram
- **NB_CPUS**: nb of CPUs used by each training
- **model_ids**: List of model ids to run
- **DEBUG**: whether to run parallel in debug mode
- **saved_models_path**: path where the models are saved
- **overwrite_params**: name of dict (defined in config.py) to use for overwriting params
- **plot_paths**: name of the dict (in config.py) defining input for extras.plot_paths_from_checkpoint
- **climate_crossval**: name of the dict (in config.py) defining input for extras.get_cross_validation
- **plot_conv_study**: name of the dict (in config.py) defining input for extras.plot_convergence_study


train model on Poisson Point Process:
```sh
python run.py --params=param_list_poissonpp1 --NB_JOBS=1 --NB_CPUS=1 --first_id=1
python run.py --plot_paths=plot_paths_ppp_dict
```

train model on FBM:
```sh
python run.py --params=param_list_FBM1 --NB_JOBS=1 --NB_CPUS=1 --first_id=1  --get_overview=overview_dict_FBM1
python run.py --plot_paths=plot_paths_FBM_dict
```

train model on BM2DCorr:
```sh
python run.py --params=param_list_BM2D_1 --NB_JOBS=1 --NB_CPUS=1 --first_id=1  --get_overview=overview_dict_BM2D_1
python run.py --plot_paths=plot_paths_BMV2D_dict
```

train model on BMandVar:
```sh
python run.py --params=param_list_BMandVar_1 --NB_JOBS=2 --NB_CPUS=1 --first_id=1
python run.py --plot_paths=plot_paths_BMVar_dict
```

train model on dependent observation intensity datasets:
```sh
python run.py --params=param_list_DepIntensity_1 --NB_JOBS=2 --NB_CPUS=1 --first_id=1
python run.py --plot_paths=plot_paths_DepIntensity_dict
```


train model on PhysioNet datasets:
```sh
python run.py --params=param_list_physio --NB_JOBS=8 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_physio --crossval=crossval_dict_physio
```

train model on Climate datasets:
```sh
python run.py --params=param_list_climate --NB_JOBS=8 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_climate --crossval=crossval_dict_climate
```





--------------------------------------------------------------------------------
## Usage, License & Citation

This code can be used in accordance with the [LICENSE](LICENSE).

If you find this code useful or include parts of it in your own work, 
please cite our papers:  

[Optimal Estimation of Generic Dynamics by Path-Dependent Neural Jump ODEs](https://arxiv.org/abs/2206.14284)

```
@article{PDNJODE
  url = {https://arxiv.org/abs/2206.14284},
  author = {Krach, Florian and Nübel, Marc and Teichmann, Josef},
  title = {Optimal Estimation of Generic Dynamics by Path-Dependent Neural Jump ODEs},
  publisher = {arXiv},
  year = {2022},
}
```

[Neural Jump Ordinary Differential Equations: Consistent Continuous-Time Prediction and Filtering](https://openreview.net/forum?id=JFKR3WqwyXR)

```
@inproceedings{
herrera2021neural,
title={Neural Jump Ordinary Differential Equations: Consistent Continuous-Time Prediction and Filtering},
author={Calypso Herrera and Florian Krach and Josef Teichmann},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=JFKR3WqwyXR}
}
```


## Acknowledgements and References
This code is based on the code-repo of the first paper [Neural Jump Ordinary Differential Equations: Consistent Continuous-Time Prediction and Filtering](https://openreview.net/forum?id=JFKR3WqwyXR):
https://github.com/HerreraKrachTeichmann/NJODE  
Parts of this code are based on and/or copied from the code of:
https://github.com/edebrouwer/gru_ode_bayes, of the paper
[GRU-ODE-Bayes: Continuous modeling of sporadically-observed time series](https://arxiv.org/abs/1905.12374)
and the code of: https://github.com/YuliaRubanova/latent_ode, of the paper
[Latent ODEs for Irregularly-Sampled Time Series](https://arxiv.org/abs/1907.03907).

The GIFs of the training progress were generated with imageio:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3674137.svg)](https://doi.org/10.5281/zenodo.3674137)


--------------------------------------------------------------------------------
# Instructions for Running Experiments of Neural Jump Ordinary Differential Equations

The original repository of the paper [Neural Jump Ordinary Differential Equations: Consistent Continuous-Time Prediction and Filtering](https://openreview.net/forum?id=JFKR3WqwyXR) 
is available [here](https://github.com/HerreraKrachTeichmann/NJODE).
In the current repository, the code was developed further such that it is more user-friendly.
Below are the instructions to run the experiments of the first paper.


## Dataset Generation
go to the source directory:
```sh
cd NJODE
```

generate the standard BlackScholes, Heston and OrnsteinUhlenbeck datasets:
```sh
python data_utils.py --dataset_name=BlackScholes --dataset_params=hyperparam_default
python data_utils.py --dataset_name=Heston --dataset_params=hyperparam_default
python data_utils.py --dataset_name=OrnsteinUhlenbeck --dataset_params=hyperparam_default
```

generate dataset of Heston model without Feller condition:
```sh
python data_utils.py --dataset_name=HestonWOFeller --dataset_params=HestonWOFeller_dict1
python data_utils.py --dataset_name=HestonWOFeller --dataset_params=HestonWOFeller_dict2
```

generate combined dataset OU+BS:
```sh
python data_utils.py --dataset_name=combined_OrnsteinUhlenbeck_BlackScholes --dataset_params=combined_OU_BS_dataset_dicts
```

generate sine-drift BS dataset OU+BS:
```sh
python data_utils.py --dataset_name=sine_BlackScholes --dataset_params=sine_BS_dataset_dict1
python data_utils.py --dataset_name=sine_BlackScholes --dataset_params=sine_BS_dataset_dict2
```


## Training & Testing
go to the source directory:
```sh
cd NJODE
```

train models on the 3 standard datasets (Black-Scholes, Heston, Ornstein-Uhlenbeck):
```sh
python run.py --params=params_list1 --NB_JOBS=3 --NB_CPUS=1 --first_id=1
```

run the convergence study on the 3 standard datasets:
```shell
# Black-Scholes
python run.py --params=params_list_convstud_BS --NB_JOBS=32 --NB_CPUS=1 --first_id=1 --plot_conv_study=plot_conv_stud_BS_dict1
python run.py --plot_conv_study=plot_conv_stud_BS_dict2

# Heston
python run.py --params=params_list_convstud_Heston --NB_JOBS=32 --NB_CPUS=1 --first_id=1 --plot_conv_study=plot_conv_stud_heston_dict1
python run.py --plot_conv_study=plot_conv_stud_heston_dict2

# Ornstein-Uhlenbeck
python run.py --params=params_list_convstud_OU --NB_JOBS=32 --NB_CPUS=1 --first_id=1 --plot_conv_study=plot_conv_stud_OU_dict1
python run.py --plot_conv_study=plot_conv_stud_OU_dict2
```

train model on HestonWOFeller dataset:
```sh
python run.py --params=params_list_HestonWOFeller --NB_JOBS=2 --NB_CPUS=1 --first_id=1 
```

train model on Combined OU+BS dataset (regime switch)
```sh
python run.py --params=params_list_combined_OU_BS --NB_JOBS=1 --NB_CPUS=1 --first_id=1 
```

train model on sine-drift BS dataset (explicit time dependence)
```sh
python run.py --params=params_list_sine_BS --NB_JOBS=2 --NB_CPUS=1 --first_id=1 
```

train the GRU-ODE-Bayes model on the 3 standard datasets:
```sh
python run.py --params=params_list_GRUODEBayes --NB_JOBS=32 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_GRUODEBayes
```

train model on climate dataset of [GRU-ODE-Bayes](https://arxiv.org/abs/1905.12374):
```sh
python run.py --params=params_list_NJODE1_climate --NB_JOBS=32 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_NJODE1_climate --crossval=crossval_dict_NJODE1_climate
```

train model on Physionet Dataset of [Latent ODEs for Irregularly-Sampled Time Series](https://arxiv.org/abs/1907.03907):
```sh
python run.py --params=params_list_NJODE1_physionet --NB_JOBS=32 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_NJODE1_physionet --crossval=crossval_dict_NJODE1_physionet
```




