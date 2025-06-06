# Neural Jump ODEs

[![DOI](https://zenodo.org/badge/507857738.svg)](https://zenodo.org/badge/latestdoi/507857738)

This repository is the official implementation of the papers 
- [Optimal Estimation of Generic Dynamics by Path-Dependent Neural Jump ODEs](https://arxiv.org/abs/2206.14284)
- [Extending Path-Dependent NJ-ODEs to Noisy Observations and a Dependent Observation Framework](https://openreview.net/forum?id=0T2OTVCCC1) 
- [Learning Chaotic Systems and Long-Term Predictions with Neural Jump ODEs](https://arxiv.org/abs/2407.18808)
- [Nonparametric Filtering, Estimation and Classification using Neural Jump ODEs](https://arxiv.org/abs/2412.03271)

which are part of the series of works on Neural Jump ODEs that started with
[Neural Jump Ordinary Differential Equations: Consistent Continuous-Time Prediction and Filtering](https://openreview.net/forum?id=JFKR3WqwyXR).

The code is based on the [code of the first paper](https://github.com/HerreraKrachTeichmann/NJODE), 
but was developed further such that it is more user-friendly. 
All experiments from the first paper can be run with this repo as well (see 
[Instructions for Neural Jump Ordinary Differential Equations: Consistent Continuous-Time Prediction and Filtering](#instructions-for-running-experiments-of-neural-jump-ordinary-differential-equations)).

The experiments from the follow-up papers can be run with:
- [Instructions for Optimal Estimation of Generic Dynamics by Path-Dependent Neural Jump ODEs](#instructions-for-running-experiments-of-optimal-estimation-of-generic-dynamics-by-path-dependent-neural-jump-odes)
- [Instructions for Extending Path-Dependent NJ-ODEs to Noisy Observations and a Dependent Observation Framework](#instructions-for-running-experiments-of-extending-path-dependent-nj-odes-to-noisy-observations-and-a-dependent-observation-framework)
- [Instructions for Learning Chaotic Systems and Long-Term Predictions with Neural Jump ODEs](#instructions-for-running-experiments-of-learning-chaotic-systems-and-long-term-predictions-with-neural-jump-odes)
- [Instructions for Nonparametric Filtering, Estimation and Classification using Neural Jump ODEs](#instructions-for-running-experiments-of-nonparametric-filtering-estimation-and-classification-using-neural-jump-odes)

A notebook studying Explainability in the context of NJODEs is available at [Explainability of NJODEs](https://gist.github.com/FlorianKrach/7a610cd88d9739b2f8bbda8455a558b4).


## Requirements

This code was executed using Python 3.7.

To install requirements, download this Repo and cd into it.

Then create a new environment and install all dependencies and this repo.
With [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html):
 ```sh
conda create --name njode python=3.7
conda activate njode
pip install -r requirements.txt
 ```

To use the Telegram-Bot see installation instructions [here](https://github.com/FlorianKrach/Telegram-Bot-Install).
The code will run without the Telegram-Bot, but you will not receive notification and results via Telegram when the training is finished (useful when running on a server).


--------------------------------------------------------------------------------
## Usage, License & Citation

This code can be used in accordance with the [LICENSE](LICENSE).

If you find this code useful or include parts of it in your own work, 
please cite our papers:  

- [Optimal Estimation of Generic Dynamics by Path-Dependent Neural Jump ODEs](https://arxiv.org/abs/2206.14284)
    ```
    @article{PDNJODE
      url = {https://arxiv.org/abs/2206.14284},
      author = {Krach, Florian and Nübel, Marc and Teichmann, Josef},
      title = {Optimal Estimation of Generic Dynamics by Path-Dependent Neural Jump ODEs},
      publisher = {arXiv},
      year = {2022},
    }
    ```


- [Extending Path-Dependent NJ-ODEs to Noisy Observations and a Dependent Observation Framework](https://openreview.net/forum?id=0T2OTVCCC1)
    ```
    @article{
        andersson2024extending,
        title={Extending Path-Dependent {NJ}-{ODE}s to Noisy Observations and a Dependent Observation Framework},
        author={William Andersson and Jakob Heiss and Florian Krach and Josef Teichmann},
        journal={Transactions on Machine Learning Research},
        issn={2835-8856},
        year={2024},
        url={https://openreview.net/forum?id=0T2OTVCCC1},
        note={}
    }
    ```
  
- [Learning Chaotic Systems and Long-Term Predictions with Neural Jump ODEs](https://arxiv.org/abs/2407.18808)
    ```
    @misc{krach2024learningchaoticsystemslongterm,
      title={Learning Chaotic Systems and Long-Term Predictions with Neural Jump ODEs}, 
      author={Florian Krach and Josef Teichmann},
      year={2024},
      eprint={2407.18808},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2407.18808},
    }
    ```

- [Nonparametric Filtering, Estimation and Classification using Neural Jump ODEs](https://arxiv.org/abs/2412.03271)
  ```
  @misc{heiss2024nonparametricfilteringestimationclassification,
      title={Nonparametric Filtering, Estimation and Classification using Neural Jump ODEs}, 
      author={Jakob Heiss and Florian Krach and Thorsten Schmidt and Félix B. Tambe-Ndonfack},
      year={2024},
      eprint={2412.03271},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2412.03271}, 
  }
  ```


- [Neural Jump Ordinary Differential Equations: Consistent Continuous-Time Prediction and Filtering](https://openreview.net/forum?id=JFKR3WqwyXR)
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
[Latent ODEs for Irregularly-Sampled Time Series](https://arxiv.org/abs/1907.03907)
and the code of: https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/, of the paper
[DeepLOB: Deep convolutional neural networks for limit order books](https://arxiv.org/abs/1808.03668).

The [High Frequency Crypto Limit Order Book Data](https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data)
was made public in Kaggle.
The other bitcoin LOB dataset was gratefully provided by Covario, but is not publicly available.

The GIFs of the training progress were generated with imageio:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3674137.svg)](https://doi.org/10.5281/zenodo.3674137)



--------------------------------------------------------------------------------
# Instructions for Running Experiments of Optimal Estimation of Generic Dynamics by Path-Dependent Neural Jump ODEs

The configs for the experiment are in the main config file [config.py](NJODE/configs/config.py) as well as in [config_LOB.py](NJODE/configs/config_LOB.py), [config_NJmodel.py](NJODE/configs/config_NJmodel.py) and [config_randomizedNJODE.py](NJODE/configs/config_randomizedNJODE.py).

## Dataset Generation
go to the source directory:
```sh
cd NJODE
```


### Synthetic Datasets
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

generate double-pendulum (deterministic chaotic system) dataset:
```sh
python data_utils.py --dataset_name=DoublePendulum --dataset_params=DP_dict1
python data_utils.py --dataset_name=DoublePendulum --dataset_params=DP_dict2
python data_utils.py --dataset_name=DoublePendulum --dataset_params=DP_dict3
python data_utils.py --dataset_name=DoublePendulum --dataset_params=DP_dict3_test --seed=3
```

generate BM Filtering Problem dataset:
```sh
python data_utils.py --dataset_name=BMFiltering --dataset_params=BM_Filter_dict
python data_utils.py --dataset_name=BMFiltering --dataset_params=BM_Filter_dict_1
python data_utils.py --dataset_name=BMFiltering --dataset_params=BM_Filter_dict_testdata
```

generate BM TimeLag dataset:
```sh
python data_utils.py --dataset_name=BMwithTimeLag --dataset_params=BM_TimeLag_dict_1 --seed=3
python data_utils.py --dataset_name=BMwithTimeLag --dataset_params=BM_TimeLag_dict_2 --seed=3
python data_utils.py --dataset_name=BMwithTimeLag --dataset_params=BM_TimeLag_dict_testdata
```

generate BlackScholes dataset:
```sh
python data_utils.py --dataset_name=BlackScholes --dataset_params=BlackScholes_dict
```


### Real World Datasets
generate Limit Order Book (LOB) dataset with raw data from Covario
(**ATTENTION**: the needed raw data was provided by Covario but is not publicly
available, hence this data generation will raise an error, unless you provide a
working link to raw data):
```sh
python data_utils.py --dataset_name=LOB --dataset_params=LOB_dict1
python data_utils.py --dataset_name=LOB --dataset_params=LOB_dict1_2
python data_utils.py --dataset_name=LOB --dataset_params=LOB_dict2
python data_utils.py --dataset_name=LOB --dataset_params=LOB_dict3
python data_utils.py --dataset_name=LOB --dataset_params=LOB_dict3_2
```

generate Limit Order Book (LOB) dataset with raw data from [Kaggle data](https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data):
```sh
python data_utils.py --dataset_name=LOB --dataset_params=LOB_dict_K_1
python data_utils.py --dataset_name=LOB --dataset_params=LOB_dict_K_1_2
python data_utils.py --dataset_name=LOB --dataset_params=LOB_dict_K_2
python data_utils.py --dataset_name=LOB --dataset_params=LOB_dict_K_3
python data_utils.py --dataset_name=LOB --dataset_params=LOB_dict_K_3_2
python data_utils.py --dataset_name=LOB --dataset_params=LOB_dict_K_4
python data_utils.py --dataset_name=LOB --dataset_params=LOB_dict_K_4_2
python data_utils.py --dataset_name=LOB --dataset_params=LOB_dict_K_5
python data_utils.py --dataset_name=LOB --dataset_params=LOB_dict_K_6
python data_utils.py --dataset_name=LOB --dataset_params=LOB_dict_K_6_2
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


### Synthetic Datasets

*See [config.py](NJODE/configs/config.py):*

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
python run.py --plot_paths=plot_paths_BM2D_dict
```

train model on BMandVar:
```sh
python run.py --params=param_list_BMandVar_1 --NB_JOBS=2 --NB_CPUS=1 --first_id=1
python run.py --plot_paths=plot_paths_BMVar_dict
```

train model on dependent observation intensity datasets:
```sh
python run.py --params=param_list_DepIntensity_1 --NB_JOBS=24 --NB_CPUS=1 --first_id=1
python run.py --plot_paths=plot_paths_DepIntensity_dict
```

train PD-NJODE on DoublePendulum:
```shell
python run.py --params=param_list_DP --NB_JOBS=40 --NB_CPUS=2 --first_id=1 --get_overview=overview_dict_DP
python run.py --plot_paths=plot_paths_DP_dict
python run.py --params=param_list_DP --NB_JOBS=40 --first_id=1 --saved_models_path="../data/saved_models_DoublePendulum/" --overwrite_params="{'test_data_dict': 'DP_dict3_test'}"
python run.py --plot_paths=plot_paths_DP_dict
```

train model on BM Filtering Problem dataset:
```sh
python run.py --params=param_list_BMFilter_1 --NB_JOBS=8 --NB_CPUS=2 --first_id=1 --get_overview=overview_dict_BMFilter_1
python run.py --plot_paths=plot_paths_BMFilter_dict
```

train model on BM with TimeLag dataset:
```sh
python run.py --params=param_list_BMTimeLag --NB_JOBS=8 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_BMTimeLag
python run.py --plot_paths=plot_paths_BMTimeLag_dict
```

train NJmodel on BS (*see [config_NJmodel.py](NJODE/configs/config_NJmodel.py)*):
```shell
python run.py --params=param_list_NJmodel --NB_JOBS=8 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_NJmodel
python run.py --plot_paths=plot_paths_NJmodel_dict
```



### Real World Datasets

*See [config.py](NJODE/configs/config.py):*

train model on PhysioNet datasets:
```sh
python run.py --params=param_list_physio --NB_JOBS=8 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_physio --crossval=crossval_dict_physio
```

train model on Climate datasets:
```sh
python run.py --params=param_list_climate --NB_JOBS=8 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_climate --crossval=crossval_dict_climate
```

*See [config_LOB.py](NJODE/configs/config_LOB.py):*

train model on Limit Order Book datasets:
```sh
python run.py --params=param_list_LOB --NB_JOBS=22 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_LOB
python run.py --plot_paths=plot_paths_LOB_dict
```

retrain classifier on LOB dataset:
```shell
python run.py --params=param_list_retrainLOB --NB_JOBS=42 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_retrainLOB
```

train model on Limit Order Book Kaggle datasets:
```sh
python run.py --params=param_list_LOB_K --NB_JOBS=22 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_LOB_K
python run.py --params=param_list_LOB_n --NB_JOBS=22 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_LOB_n
python run.py --plot_paths=plot_paths_LOB_K_dict
```

retrain classifier on LOB Kaggle dataset:
```shell
python run.py --params=param_list_retrainLOB_K --NB_JOBS=42 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_retrainLOB_K
python run.py --params=param_list_retrainLOB_n --NB_JOBS=42 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_retrainLOB_n
```

training of the DeepLOB model from the paper [DeepLOB: Deep Convolutional Neural Networks for Limit Order Books](https://arxiv.org/abs/1808.03668) as comparison
```shell
# training of the DeepLOB model
python ../DeepLOB/train_DeepLOB.py
```

fitting of LinReg models as baselines on LOB data:
```shell
python LOB_linreg.py
```




--------------------------------------------------------------------------------
# Experimental: Randomized NJODE
In this section we provide instructions for running the experiments of the 
randomized NJODE model. In particular, this is the NJODE model, where only the
(last layer of) the readout map is trained, while the neural-ODE and the 
encoder/jump network are kept fixed. There are two versions of the randomized
NJODE model: one where the readout map is trained via SGD and one where it
is trained via the closed-form solution of OLS (i.e. fitting a linear regression
model).

See [config_randomizedNJODE.py](NJODE/configs/config_randomizedNJODE.py):

train randomizedNJODE on BM:
```shell
python run.py --params=param_list_randNJODE_1 --NB_JOBS=1 --NB_CPUS=3 --first_id=1
python run.py --params=param_list_randNJODE_2 --NB_JOBS=1 --NB_CPUS=3 --first_id=3
```

train randomizedNJODE on BS:
```shell
python run.py --params=param_list_randNJODE_BS --NB_JOBS=1 --NB_CPUS=4 --first_id=1 --get_overview=overview_dict_randNJODE_BS
```

train randomizedNJODE on DoublePendulum:
```shell
python run.py --params=param_list_randNJODE_3 --NB_JOBS=1 --NB_CPUS=3 --first_id=100
```




--------------------------------------------------------------------------------
# Experimental: learning quantiles with NJODE

Parameters defined in [config_Quantiles.py](NJODE/configs/config_Quantiles.py).

get datasets:
```shell
python data_utils.py --dataset_name=BMandQuantiles --dataset_params=BM_Quantiles --seed=0
python data_utils.py --dataset_name=BMandQuantiles --dataset_params=BM_Quantiles_test --seed=1
```

train PD-NJODE:
```shell
python run.py --params=param_list_BMQ --NB_JOBS=16 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_BMQ
python run.py --plot_paths=plot_paths_BMQ_dict
```


--------------------------------------------------------------------------------
# Direct Variance Prediction with NJODE

Instead of predicting the variance via the first 2 moments, which can lead to numerical instabilities (and therefore implausible negative values or non positive-semi definite covariance matrices),
we can also directly predict the marginal variance or the covariance matrix using the NJODE model.
In particular, we define the model with two output parts $(Y,W)$, where $Y$ is the original output and $W$ is the variance/covariance output.
Then we can train $Y$ with the standard loss function to approximate the conditional expectation of the process $X$ and train $V=W^2$ (in the marginal variance case) or $V=W^\top W$ (in the covariance matrix case) to approximate the process $(X-Y)^2$.
By using the square $V$ of $W$ to approximate $(X-Y)^2$, we have a hard-coded way to avoid numerical instabilities (negative values of the variance or non positive-semi definite covariance matrices), 
since the network output $W$ corresponds to the standard deviation, which is squared to get the variance (in particular, $W$ can have negative entries, which are simply interpreted as positive).
By the theoretical results, the model output $W$ or $V$, respectively, learns to approximate the conditional expectation of $(X-Y)^2$ arbitrarily well, which coincides with the conditional variance of $X$.  
See the [NJODE website](https://floriankrach.github.io/njode/index.html) for more details.

Parameters defined in [config_uncertainty.py](NJODE/configs/config_uncertainty.py).


train NJODE on Physionet dataset with variance prediction:
```shell
python run.py --params=param_list_physio_U --NB_JOBS=25 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_physio_U
python run.py --plot_paths=eval_physio_U_dict
```

get datasets:
```shell
python data_utils.py --dataset_name=BM --dataset_params=BM_dict_U --seed=0
```

train NJODE on BM dataset with variance prediction:
```shell
python run.py --params=param_list_BMandVar_U --NB_JOBS=25 --NB_CPUS=1 --first_id=1
```





--------------------------------------------------------------------------------
# Instructions for Running Experiments of Extending Path-Dependent NJ-ODEs to Noisy Observations and a Dependent Observation Framework

The code for the experiments of the paper [Extending Path-Dependent NJ-ODEs to Noisy Observations and a Dependent Observation Framework](https://openreview.net/forum?id=0T2OTVCCC1).

The configs for these experiments are in [config_NJODE3.py](NJODE/configs/config_NJODE3.py).

## Dataset Generation
go to the source directory:
```sh
cd NJODE
```

generate the BlackScholes datset with dependent observations
```sh
python data_utils.py --dataset_name=BlackScholes --dataset_params=BS_dep_obs_dict
```

generate the Brownian motion datset with noisy observations
```sh
python data_utils.py --dataset_name=BMNoisyObs --dataset_params=BM_NoisyObs_dict
```


## Training & Testing
go to the source directory:
```sh
cd NJODE
```

train model on BlackScholes dataset with dependent observations
```sh
python run.py --params=param_list_DepObs_1 --NB_JOBS=24 --NB_CPUS=1 --first_id=1
python run.py --plot_paths=plot_paths_DepObs_dict
```

train model on BM dataset with noisy observations
```sh
python run.py --params=param_list_BM_NoisyObs --NB_JOBS=24 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_BM_NoisyObs
python run.py --plot_paths=plot_paths_BM_NoisyObs_dict
```

train model for noisy observations on PhysioNet datasets:
```sh
python run.py --params=param_list_physio_N3 --NB_JOBS=12 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_physio_N3 --plot_loss_comparison=plot_loss_comparison_physio_N3
```

train model for noisy observations on Climate datasets:
```sh
python run.py --params=param_list_climate_N3 --NB_JOBS=40 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_climate_N3 --crossval=crossval_dict_climate_N3
```



--------------------------------------------------------------------------------
# Instructions for Running Experiments of Learning Chaotic Systems and Long-Term Predictions with Neural Jump ODEs

The code for the experiments of the paper [Learning Chaotic Systems and Long-Term Predictions with Neural Jump ODEs](https://arxiv.org/abs/2407.18808).

The configs for these experiments are in [config_ODE.py](NJODE/configs/config_ODE.py).


generate datasets:
```shell
python data_utils.py --dataset_name=DoublePendulum --dataset_params=DP_dict3_2_test --seed=3
python data_utils.py --dataset_name=DoublePendulum --dataset_params=DP_dict4
python data_utils.py --dataset_name=DoublePendulum --dataset_params=DP_dict4_test --seed=3

python data_utils.py --dataset_name=BlackScholes --dataset_params=BS_LT_dict
python data_utils.py --dataset_name=BlackScholes --dataset_params=BS_LT_dict1
python data_utils.py --dataset_name=BlackScholes --dataset_params=BS_LT_dict_test --seed=3
python data_utils.py --dataset_name=BlackScholes --dataset_params=BS_LT_dict2
python data_utils.py --dataset_name=BlackScholes --dataset_params=BS_LT_dict2_test --seed=3
```

train PD-NJODE on DoublePendulum:
```shell
python run.py --params=param_list_ODE_DP --NB_JOBS=14 --NB_CPUS=2 --first_id=1 --get_overview=overview_dict_ODE_DP
python run.py --plot_paths=plot_paths_ODE_DP_dict
```

train model on BlackScholes dataset for long-term forecasting:
```shell
python run.py --params=param_list_BS_LT --NB_JOBS=4 --NB_CPUS=2 --first_id=1 --get_overview=overview_dict_BS_LT
python run.py --plot_paths=plot_paths_BS_LT_dict
```

train on physionet dataset for long-term forecasting:
```shell
python run.py --params=param_list_physioLT --NB_JOBS=8 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_physioLT # --crossval=crossval_dict_physio
```


--------------------------------------------------------------------------------
# Instructions for Running Experiments of Nonparametric Filtering, Estimation and Classification using Neural Jump ODEs

The code for the experiments of the paper [Nonparametric Filtering, Estimation and Classification using Neural Jump ODEs](https://arxiv.org/abs/2412.03271).

The configs for these experiments are in [config_ParamFilter.py](NJODE/configs/config_ParamFilter.py).


generate datasets:
```shell
python data_utils.py --dataset_name=BlackScholesUncertainParams --dataset_params=PF_BSUP_dict1 --seed=0
python data_utils.py --dataset_name=BlackScholesUncertainParams --dataset_params=PF_BSUP_dict1_test --seed=1

python data_utils.py --dataset_name=BlackScholesUncertainParams --dataset_params=PF_BSUP_dict2 --seed=0
python data_utils.py --dataset_name=BlackScholesUncertainParams --dataset_params=PF_BSUP_dict2_test --seed=1

for i in 1 2 
do
  for j in {1..6}
  do
    python data_utils.py --dataset_name=BlackScholesUncertainParams --dataset_params=PF_BSUP_dict_CS${i}_${j} --seed=0
    python data_utils.py --dataset_name=BlackScholesUncertainParams --dataset_params=PF_BSUP_dict_CS${i}_test_${j} --seed=1
  done
done

python data_utils.py --dataset_name=BMwithUncertainDrift --dataset_params=PF_BMwUD_dict1 --seed=0
python data_utils.py --dataset_name=BMwithUncertainDrift --dataset_params=PF_BMwUD_dict1_test --seed=1

python data_utils.py --dataset_name=BMFiltering --dataset_params=IO_BM_Filter_dict_1 --seed=0
python data_utils.py --dataset_name=BMFiltering --dataset_params=IO_BM_Filter_dict_1_test --seed=1

python data_utils.py --dataset_name=BlackScholes --dataset_params=IO_BS_dict --seed=0
python data_utils.py --dataset_name=BlackScholes --dataset_params=IO_BS_dict_test --seed=1

python data_utils.py --dataset_name=CIRUncertainParams --dataset_params=PF_CIR_dict1 --seed=0
python data_utils.py --dataset_name=CIRUncertainParams --dataset_params=PF_CIR_dict1_test --seed=1
python data_utils.py --dataset_name=CIRUncertainParams --dataset_params=PF_CIR_dict2 --seed=0
python data_utils.py --dataset_name=CIRUncertainParams --dataset_params=PF_CIR_dict2_test --seed=1
python data_utils.py --dataset_name=CIRUncertainParams --dataset_params=PF_CIR_dict3 --seed=0
python data_utils.py --dataset_name=CIRUncertainParams --dataset_params=PF_CIR_dict3_test --seed=1
python data_utils.py --dataset_name=CIRUncertainParams --dataset_params=PF_CIR_dict4 --seed=0
python data_utils.py --dataset_name=CIRUncertainParams --dataset_params=PF_CIR_dict4_test --seed=1

python data_utils.py --dataset_name=BMClassification --dataset_params=IO_BMClass_dict --seed=0
python data_utils.py --dataset_name=BMClassification --dataset_params=IO_BMClass_dict_test --seed=1
```

train NJODE:
```shell
python run.py --params=param_list_PF_BSUP --NB_JOBS=32 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_PF_BSUP

python run.py --params=param_list_PF_BSUP2 --NB_JOBS=32 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_PF_BSUP2
python run.py --plot_paths=plot_paths_PF_BSUP_dict2
python run.py --plot_paths=plot_paths_PF_BSUP_dict2_1

python run.py --params=param_list_PF_BSUP_CS --NB_JOBS=32 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_PF_BSUP_CS
python run.py --plot_paths=plot_paths_PF_BSUP_CS_dict

python run.py --params=param_list_PF_BMwUD --NB_JOBS=32 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_PF_BMwUD
python run.py --plot_paths=plot_paths_PF_BMwUD_dict

python run.py --params=param_list_IO_BMFilter --NB_JOBS=32 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_IO_BMFilter
python run.py --plot_paths=plot_paths_IO_BMFilter_dict

python run.py --params=param_list_IO_BS_LJ --NB_JOBS=32 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_IO_BS_LJ
python run.py --plot_losses=plot_loss_IO_BS_LJ

python run.py --params=param_list_PF_CIR --NB_JOBS=32 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_PF_CIR
python run.py --plot_paths=plot_paths_PF_CIR_dict

python run.py --params=param_list_IO_BMClass --NB_JOBS=16 --NB_CPUS=1 --first_id=1 --get_overview=overview_dict_IO_BMClass
python run.py --plot_paths=plot_paths_IO_BMClass_dict
```







--------------------------------------------------------------------------------
# Instructions for Running Experiments of Neural Jump Ordinary Differential Equations

The original repository of the paper [Neural Jump Ordinary Differential Equations: Consistent Continuous-Time Prediction and Filtering](https://openreview.net/forum?id=JFKR3WqwyXR) 
is available [here](https://github.com/HerreraKrachTeichmann/NJODE).
In the current repository, the code was developed further such that it is more user-friendly.
Below are the instructions to run the experiments of the first paper.

The configs for these experiments are in [config_NJODE1.py](NJODE/configs/config_NJODE1.py).

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

------------
# Changes and potantial backward incompatibilities

- Changed naming of `eval_loss` -> `val_loss`, `eval_time` -> `val_time` and `which_eval_loss` -> `which_val_loss`. 
  This might cause problems when loading old models, where the metric_id-x.csv files use the old column naming.
  To fix this, one can simply rename the columns in the csv files to the new naming.
  Moreover, when loading old models where `which_eval_loss` was specified, one has to change this to `which_val_loss` in the saved model_overview.csv file. This can be done with overwriting params option.


