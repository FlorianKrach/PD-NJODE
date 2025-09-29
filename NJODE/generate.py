"""
author: Florian Krach
"""

import copy
import numpy as np
import os
import pandas as pd
import json
import socket
import matplotlib
import tqdm
from joblib import Parallel, delayed
from absl import app
from absl import flags
import torch
from sklearn.model_selection import train_test_split
import scipy.stats as stats

from configs import config
import extras
import data_utils
import models

try:
    from telegram_notifications import send_bot_message as SBM
except Exception:
    from configs.config import SendBotMessage as SBM

# =====================================================================================================================
# FLAGS
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "params", None,
    "name of the params dict (in config.py) to run generation")
flags.DEFINE_bool("DEBUG", False, "whether to run parallel in debug mode")
flags.DEFINE_bool("USE_GPU", False, "whether to use GPU for training")
flags.DEFINE_integer("GPU_NUM", 0, "which GPU to use for training")
flags.DEFINE_bool("ANOMALY_DETECTION", False,
                  "whether to run in torch debug mode")
flags.DEFINE_integer("N_DATASET_WORKERS", 0,
                     "number of processes that generate batches in parallel")

# check whether running on computer or server
if 'ada-' not in socket.gethostname():
    SERVER = False
    flags.DEFINE_integer("NB_JOBS", 1,
                         "nb of parallel jobs to run  with joblib")
    flags.DEFINE_integer("NB_CPUS", 1, "nb of CPUs used by each training")
    flags.DEFINE_bool("SEND", False, "whether to send with telegram bot")
else:
    SERVER = True
    flags.DEFINE_integer("NB_JOBS", 24,
                         "nb of parallel jobs to run  with joblib")
    flags.DEFINE_integer("NB_CPUS", 2, "nb of CPUs used by each training")
    flags.DEFINE_bool("SEND", True, "whether to send with telegram bot")
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
print(socket.gethostname())
print('SERVER={}'.format(SERVER))


# =====================================================================================================================
# Functions
def get_model(
    hidden_size=10, bias=True, dropout_rate=0.1,
    ode_nn=None, readout_nn=None, enc_nn=None, use_rnn=False,
    solver="euler", weight=0.5, weight_decay=1.,
    dataset_metadata=None,
    **options):
    """
    get the model from the parameters
    """

    input_coords = None
    output_coords = None
    signature_coords = None
    if "input_coords" in options:
        input_coords = options["input_coords"]
    elif "input_coords" in dataset_metadata:
        input_coords = dataset_metadata["input_coords"]
    if "output_coords" in options:
        output_coords = options["output_coords"]
    elif "output_coords" in dataset_metadata:
        output_coords = dataset_metadata["output_coords"]
    if "signature_coords" in options:
        signature_coords = options["signature_coords"]
    elif "signature_coords" in dataset_metadata:
        signature_coords = dataset_metadata["signature_coords"]
    if input_coords is None:
        input_size = dataset_metadata['dimension']
        input_coords = np.arange(input_size)
    else:
        input_size = len(input_coords)
    if output_coords is None:
        output_size = dataset_metadata['dimension']
        output_coords = np.arange(output_size)
    else:
        output_size = len(output_coords)
    if signature_coords is None:
        signature_coords = input_coords
    if 'func_appl_X' in options:
        functions = options['func_appl_X']
        if isinstance(functions, list) and len(functions) > 0:
            raise ValueError(
                "generative model not compatible with function application")

    compute_variance = None
    var_weight, which_var_loss = 1., None
    if 'var_weight' in options:
        var_weight = options['var_weight']
    if 'which_var_loss' in options:
        which_var_loss = options['which_var_loss']
    var_size = 0
    if 'compute_variance' in options:
        compute_variance = options['compute_variance']
        if compute_variance in ['covariance', 'covar']:
            compute_variance = "covariance"
            var_size = output_size ** 2
        elif compute_variance in ['vola', 'volatility']:
            compute_variance = "volatility"
            var_size = output_size ** 2
        elif compute_variance not in [None, False]:
            compute_variance = 'variance'
            var_size = output_size
        else:
            compute_variance = None
            var_size = 0
        output_size += var_size

    params_dict = {
        'input_size': input_size, 'hidden_size': hidden_size,
        'output_size': output_size, 'bias': bias, 'ode_nn': ode_nn,
        'readout_nn': readout_nn, 'enc_nn': enc_nn,
        'use_rnn': use_rnn, 'dropout_rate': dropout_rate,
        'solver': solver, 'weight': weight,
        'weight_decay': weight_decay, 'options': options,
        'input_coords': input_coords, 'output_coords': output_coords,
        'signature_coords': signature_coords,
        'compute_variance': compute_variance, 'var_size': var_size}

    return models.NJODE(**params_dict)


def generate_paths(
    model_drift_path, model_drift_id, model_drift_params,
    model_diffusion_path=None, model_diffusion_id=None,
    model_diffusion_params=None,
    model_drift_inputs=("X_inc", "X"),
    model_diffusion_inputs=("Z", "X"),
    joint_model=True, load_best=True,
    start_X=None, init_times=None, init_X=None, start_M=None, init_M=None,
    training_data_dict=None, T=None, delta_t=None,
    steps_ahead_prediction=0, nb_samples_gen=None,
    instantaneous_coeff_estimation=True,
    drift_predicts_inc=True, return_coeffs=False,
    save_gen_paths_path=None,
    save_paths=False, plot_paths=False, plot_real_paths=True,
    max_paths_to_plot=1000,
    gen_seed=None, estimate_GBM_params=False, estimate_OU_params=False,
    plot_which_coeff_paths=(0,1,2,3,4), mu_function=None, sigma_function=None,
    plot_dist_at_t_indices=(-1,), plot_dist_num_bins=50,
    send=False, use_gpu=False, gpu_num=0,
):
    """

    Args:
        model_drift_path: str, path to the drift model
        model_drift_id: int, id of the drift model
        model_drift_params: dict, parameters of the drift model
        model_diffusion_path: None or str, path to the diffusion model
        model_diffusion_id: None or int, id of the diffusion model
        model_diffusion_params: None or dict, parameters of the diffusion model
        joint_model: bool, if True the drift model also provides the diffusion
            predictions via the
        model_drift_inputs: list of str out of {"X_inc", "X", "Z", "QV"},
            specifying all the inputs that the drift model uses (including input
            and output coordinates. output coordinates are not used in the
            generative method, but must be supplied as dummy variables)
        model_diffusion_inputs: list of str, similar to model_drift_inputs
        load_best: bool, whether to load the best model or the last
        start_X: np.array of dim (innerdim,), the starting point for the paths
            IMPORTANT: this must correspond to "X" only, not including "X_inc"
                "Z" or "QV" (even if the training data does so). the necessary
                additional features will be computed automatically. same holds
                for init_X (and start_M, init_M).
        init_times: None or np.array of dim (timesteps,), the initial times of
            observations of the process that form the starting sequence, not
            including 0
        init_X: None or np.array of dim (timesteps, innerdim), the initial
            values of observations of the process that form the starting
            sequence
        start_M: None or np.array of dim (innerdim,)
        init_M: None or np.array of dim (timesteps, innerdim),
            ATTENTION: if provided, the last observation of the starting
                sequence must be fully observed. if the last observation is the
                start_X, then this must be fully observed. (can be generalised,
                but not implemented!)
        training_data_dict: str, used to load dataset_metadata and the training
            paths for plotting and evaluation comparing to real paths.
        T: None or float, the end time for generation, overrides value loaded
            from dataset
        delta_t: None or float, the timestep for generation, overrides
            value loaded from dataset
        steps_ahead_prediction: int, the number of steps the model predicts
            ahead for computing the coefficients
        nb_samples_gen: int, number of samples to generate
        instantaneous_coeff_estimation: bool, if True the models predict the
            instantaneous coefficient estimates, otherwise, the coefficients
            are computed by dividing by delta_t*steps_ahead_prediction
        drift_predicts_inc: bool, if True the drift predictions will be the
            increment of the process (hence, the increment does not need to be
            computed). this is set to true if instantaneous_coeff_estimation is
            True.
        return_coeffs: bool, if True, the predicted coefficients are also
            returned
        send: bool, whether to send telegram notifications
        use_gpu: bool, whether to use the GPU
        gpu_num: int, which GPU to use
        save_gen_paths_path: None or str, if str, the filepath where to save
            the data
        save_paths: bool, whether to save the generated paths
        plot_paths: bool, whether to plot the generated paths
        plot_real_paths: bool, whether to plot real paths for comparison
            (only if plot_paths is True)
        max_paths_to_plot: int, maximum number of paths to plot
        gen_seed: None or int, random seed for generating the paths
        estimate_GBM_params: bool, if True, the parameters of a GBM are
            estimated from the generated paths and printed
        estimate_OU_params: bool, if True, the parameters of an OU process are
            estimated from the generated paths and printed
        plot_which_coeff_paths: None or list of int, if not None, the indices of
            the batch samples for which the paths of the coefficients
            should be plotted
        mu_function: None or function or str, if not None, the function to
            compute the true drift coefficients from data
        sigma_function: None or function or str, if not None, the function to
            compute the true diffusion coefficients from data
        plot_dist_at_t_indices: None or list of int, if not None, the indices of
            the time steps at which the distribution of the generated paths
            should be plotted
        plot_dist_num_bins: int, number of bins to use for plotting the
            distribution of the paths

    Returns:
        paths: np.array of dim (batch_size, innerdim, timesteps)
        all_times: np.array of dim (timesteps,), the times of the generated
            paths including the initial times and 0
        gen_times: np.array of dim (timesteps-len(init_times)-1,), the times
            of the generated paths excluding the initial times and 0
        mus_gen: np.array of dim ,
            the predicted drift coefficients at each time step (only if

    """
    assert (training_data_dict is not None)
    assert (joint_model or (model_diffusion_id is not None and
                            model_diffusion_path is not None))
    batch_size = nb_samples_gen
    if instantaneous_coeff_estimation:
        drift_predicts_inc = True
    if gen_seed is not None:
        torch.manual_seed(gen_seed)
        np.random.seed(gen_seed)

    # get the device for torch
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(gpu_num))
        torch.cuda.set_device(gpu_num)
    else:
        device = torch.device("cpu")

    dataset, dataset_id = data_utils._get_dataset_name_id_from_dict(
        data_dict=training_data_dict)
    dataset_id = int(dataset_id)
    dataset_metadata = data_utils.load_metadata(
        stock_model_name=dataset, time_id=dataset_id)
    if T is None:
        T = dataset_metadata['maturity']
    if delta_t is None:
        delta_t = dataset_metadata['dt']

    # get drift model
    model_drift_path = '{}id-{}/'.format(model_drift_path, model_drift_id)
    if load_best:
        model_drift_path = '{}best_checkpoint/'.format(model_drift_path)
    else:
        model_drift_path = '{}last_checkpoint/'.format(model_drift_path)
    model_drift = get_model(
        dataset_metadata=dataset_metadata, **model_drift_params)
    model_drift.to(device)
    dummy_optimizer = torch.optim.Adam(
        model_drift.parameters(), lr=0., weight_decay=0.)
    models.get_ckpt_model(
        model_drift_path, model_drift, dummy_optimizer, device)
    print("loaded drift model from {}, id={} - in epoch={}".format(
        model_drift_path, model_drift_id, model_drift.epoch))

    # if needed, get diffusion model
    model_diffusion = None
    if not joint_model:
        model_diffusion_path = '{}id-{}/'.format(
            model_diffusion_path, model_diffusion_id)
        if load_best:
            model_diffusion_path = '{}best_checkpoint/'.format(
                model_diffusion_path)
        else:
            model_diffusion_path = '{}last_checkpoint/'.format(
                model_diffusion_path)
        model_diffusion = get_model(
            dataset_metadata=dataset_metadata, **model_diffusion_params)
        model_diffusion.to(device)
        dummy_optimizer = torch.optim.Adam(
            model_diffusion.parameters(), lr=0., weight_decay=0.)
        models.get_ckpt_model(
            model_diffusion_path, model_diffusion, dummy_optimizer, device)
        print("loaded diffusion model from {}, id={} - in epoch={}".format(
            model_diffusion_path, model_diffusion_id,
            model_diffusion.epoch))

    # prepare paths array
    data_dim = start_X.shape[0]
    last_init_time = 0.
    if init_times is None:
        init_times = []
    if isinstance(init_times, list):
        init_times = np.array(init_times)
    if len(init_times) > 0:
        last_init_time = init_times[-1]
    gen_times = [last_init_time+delta_t*i for i in range(
        1, int(np.ceil((T-last_init_time)/delta_t+1)))]
    all_times = np.array([0]+init_times.tolist()+gen_times)
    paths = np.zeros((1,len(all_times),data_dim))
    paths[0,0,:] = start_X
    if len(init_times) > 0:
        paths[0, 1:len(init_times)+1, :] = init_X
    paths = paths.repeat(batch_size, axis=0)

    # prepare start inputs for models
    start_X_drift = build_startX(start_X, model_drift_inputs)
    init_X_drift = None
    start_M_drift = None
    init_M_drift = None
    if init_X is not None:
        init_times_shift = np.roll(init_times, 1)
        init_times_shift[0] = 0
        Delta_t = init_times - init_times_shift
        Delta_t = Delta_t.reshape(-1, 1).repeat(data_dim, axis=1)
        init_X_drift = build_input(
            X=init_X, last_X=paths[0, :len(init_times)], Delta_t=Delta_t,
            divide_by_t=instantaneous_coeff_estimation,
            inputs_list=model_drift_inputs)
        if start_M is not None:
            start_M_drift = np.ones_like(start_X_drift)
            start_M_drift[-data_dim:] = start_M
        if init_M is not None:
            init_M_shift = np.roll(init_M, 1, axis=0)
            init_M_shift[0] = start_M
            init_M_drift = build_input(
                init_M, init_M_shift, Delta_t=Delta_t,
                divide_by_t=False, inputs_list=model_drift_inputs, mask=True)
    start_X_diffusion = None
    init_X_diffusion = None
    start_M_diffusion = None
    init_M_diffusion = None
    if not joint_model:
        start_X_diffusion = build_startX(start_X, model_diffusion_inputs)
        if init_X is not None:
            init_X_diffusion = build_input(
                X=init_X, last_X=paths[0, :len(init_times)], Delta_t=Delta_t,
                divide_by_t=instantaneous_coeff_estimation,
                inputs_list=model_diffusion_inputs)
            if start_M is not None:
                start_M_diffusion = np.ones_like(start_X_diffusion)
                start_M_diffusion[-data_dim:] = start_M
            if init_M is not None:
                init_M_shift = np.roll(init_M, 1, axis=0)
                init_M_shift[0] = start_M
                init_M_diffusion = build_input(
                    init_M, init_M_shift, Delta_t=Delta_t,
                    divide_by_t=False,
                    inputs_list=model_diffusion_inputs, mask=True)

    # generate paths
    print("generating paths...")
    res = efficient_gen(
        data_dim=data_dim,
        start_X_drift=start_X_drift, init_times=init_times, init_X_drift=init_X_drift,
        start_M_drift=start_M_drift, init_M_drift=init_M_drift,
        start_X_diffusion=start_X_diffusion, init_X_diffusion=init_X_diffusion,
        start_M_diffusion=start_M_diffusion, init_M_diffusion=init_M_diffusion,
        model_drift=model_drift, model_drift_inputs=model_drift_inputs,
        model_diffusion=model_diffusion, model_diffusion_inputs=model_diffusion_inputs,
        joint_model=joint_model, delta_t=delta_t, T=T,
        steps_ahead_prediction=steps_ahead_prediction, batch_size=batch_size,
        instantaneous_coeff_estimation=instantaneous_coeff_estimation,
        drift_predicts_inc=drift_predicts_inc,
        paths=paths, last_init_time=last_init_time,
        gen_times=gen_times, all_times=all_times,
        return_coeffs=return_coeffs)

    if save_gen_paths_path is not None:
        outpath = os.path.join(
            save_gen_paths_path,
            "gen_paths_drift-{}_diffusion-{}_bs-{}".format(
                model_drift_id,
                model_diffusion_id if model_diffusion_id is not None else "joint",
                batch_size))
        config.makedirs(outpath)
        outpath_plots = os.path.join(outpath, "plots/")
        config.makedirs(outpath_plots)
    else:
        return res

    if save_paths:
        print("saving generated paths...")
        names = ["paths", "all_times", "gen_times", "mus_gen", "sigmas_gen"]
        res_dict = {}
        for i, r in enumerate(res):
            res_dict[names[i]] = r
        outpath_paths = os.path.join(outpath, "paths.npz")
        np.savez_compressed(outpath_paths, **res_dict)

    x_real = None
    if plot_real_paths:
        train_idx, val_idx = train_test_split(
            np.arange(dataset_metadata["nb_paths"]),
            test_size=model_drift_params["test_size"],
            random_state=model_drift_params["seed"])
        data_train = data_utils.IrregularDataset(
            model_name=dataset, time_id=dataset_id, idx=train_idx)
        # shape: (nb_paths, timesteps, innerdim)
        x_real = data_train.stock_paths.transpose(0, 2, 1)
        x_real = x_real[:, :, -data_dim:]
        #print(x_real[0].flatten().tolist())

    files_to_send = []
    x_fake = res[0]
    if plot_paths:
        print("plotting paths...")
        for dim in range(data_dim):
            files_to_send += plot_gen_paths(
                x_fake=x_fake, x_real=x_real, times=res[1],
                file_path=outpath_plots, dim=dim,
                paths_to_plot=max_paths_to_plot)

    if estimate_GBM_params:
        print("estimating GBM parameters...")
        mus_fake, sigmas_fake, nb_excluded = [], [], []
        mus_real, sigmas_real = [], []
        for dim in range(data_dim):
            which_neg_vals = np.any(x_fake[:, :, dim] <= 0, axis=1)
            nb_neg_vals = np.sum(which_neg_vals)
            nb_excluded.append(nb_neg_vals)
            _x_fake = x_fake[~which_neg_vals]
            mu, sigma = estimate_params_GBM(_x_fake[:, :, dim], dt=delta_t)
            print("estimated GBM params for fake data dim {}: "
                  "mu={}, sigma={}, amount paths with values <=0 (excluded for "
                  "parame estimation): {}".format(dim, mu, sigma, nb_neg_vals))
            mus_fake.append(mu)
            sigmas_fake.append(sigma)
            if x_real is not None:
                mu, sigma = estimate_params_GBM(
                    x_real[:, :, dim], dt=delta_t)
                print("estimated GBM params for real data dim {}: "
                      "mu={}, sigma={}".format(dim, mu, sigma))
                mus_real.append(mu)
                sigmas_real.append(sigma)
        cols = ["dim", "mu_fake", "sigma_fake", "nb_excluded_paths"]
        dat = np.stack(
            [np.array(list(range(data_dim))), np.array(mus_fake),
             np.array(sigmas_fake), np.array(nb_excluded)], axis=1)
        if x_real is not None:
            cols = cols + ["mu_real", "sigma_real"]
            dat = np.concatenate(
                [dat, np.stack(
                    [np.array(mus_real), np.array(sigmas_real)], axis=1)],
                axis=1)
        df = pd.DataFrame(dat, columns=cols)
        file_est_GBM = os.path.join(outpath, "estimated_GBM_params.csv")
        df.to_csv(file_est_GBM)
        files_to_send.append(file_est_GBM)

    if estimate_OU_params:
        print("estimating OU parameters...")
        kappas_fake, sigmas_fake, thetas_fake = [], [], []
        kappas_real, sigmas_real, thetas_real = [], [], []
        for dim in range(data_dim):
            kappa, theta, sig = estimate_params_OU(x_fake, dt=delta_t, dim=dim)
            print("estimated OU params for fake data dim {}: "
                  "kappa={}, theta={}, sigma={}".format(
                dim, kappa, theta, sig))
            kappas_fake.append(kappa)
            thetas_fake.append(theta)
            sigmas_fake.append(sig)
            if x_real is not None:
                kappa, theta, sig = estimate_params_OU(x_real, dt=delta_t, dim=dim)
                print("estimated OU params for real data dim {}: "
                      "kappa={}, theta={}, sigma={}".format(
                    dim, kappa, theta, sig))
                kappas_real.append(kappa)
                thetas_real.append(theta)
                sigmas_real.append(sig)
        cols = ["dim", "kappa_fake", "theta_fake", "sigma_fake"]
        dat = np.stack(
            [np.array(list(range(data_dim))), np.array(kappas_fake),
             np.array(thetas_fake), np.array(sigmas_fake)], axis=1)
        if x_real is not None:
            cols = cols + ["kappa_real", "theta_real", "sigma_real"]
            dat = np.concatenate(
                [dat, np.stack(
                    [np.array(kappas_real), np.array(thetas_real),
                     np.array(sigmas_real)], axis=1)], axis=1)
        df = pd.DataFrame(dat, columns=cols)
        file_est_OU = os.path.join(outpath, "estimated_OU_params.csv")
        df.to_csv(file_est_OU)
        files_to_send.append(file_est_OU)

    if return_coeffs and plot_coeff_paths is not None:
        print("plotting coefficient paths...")
        mus_gen = res[3]
        sigmas_gen = res[4]
        mus_true, sigmas_true = None, None
        if mu_function is not None:
            if isinstance(mu_function, str):
                mu_function = eval(mu_function)
            mus_true = mu_function(
                x_fake[:, len(init_times):-1], all_times[len(init_times):-1])
        if sigma_function is not None:
            if isinstance(sigma_function, str):
                sigma_function = eval(sigma_function)
            sigmas_true = sigma_function(
                x_fake[:, len(init_times):-1], all_times[len(init_times):-1])
        files_to_send += plot_coeff_paths(
            mus_gen=mus_gen, sigmas_gen=sigmas_gen,
            mus_true=mus_true, sigmas_true=sigmas_true,
            times=all_times[len(init_times):-1],
            file_path=outpath_plots,
            paths_to_plot=plot_which_coeff_paths)

    if plot_dist_at_t_indices is not None:
        print("plotting distribution of paths...")
        if plot_dist_num_bins is None:
            plot_dist_num_bins = 50
        files_to_send += plot_distribution_at_t(
            x_fake=x_fake, t_indices=plot_dist_at_t_indices,
            num_bins=plot_dist_num_bins,
            x_real=x_real, file_path=outpath_plots,
            times=all_times,)

    if send:
        SBM.send_notification(
            text="finished path generation for drift model {}{}{} and "
                 "diffusion model {}{}{} - in epoch={}{}".format(
                model_drift_path, model_drift_id,
                " (joint model)" if joint_model else "",
                model_diffusion_path if model_diffusion_path is not None else "--",
                model_diffusion_id if model_diffusion_id is not None else "--",
                "" if joint_model else " (not joint model)",
                model_drift.epoch, "" if joint_model else " and {}".format(
                    model_diffusion.epoch),),
            files=files_to_send,
            chat_id=config.CHAT_ID)

    return res


def efficient_gen(
    data_dim,
    start_X_drift, init_times, init_X_drift, start_M_drift, init_M_drift,
    start_X_diffusion, init_X_diffusion, start_M_diffusion, init_M_diffusion,
    model_drift, model_drift_inputs, model_diffusion, model_diffusion_inputs,
    joint_model, delta_t, T, steps_ahead_prediction, batch_size,
    instantaneous_coeff_estimation, drift_predicts_inc,
    paths, last_init_time, gen_times, all_times, return_coeffs=False
):
    model_drift.restart_generation()
    if not joint_model:
        model_diffusion.restart_generation()
    current_t = last_init_time
    current_X = torch.tensor(paths[:, len(init_times), :], dtype=torch.float32)
    next_X_drift = None
    next_X_diffusion = None
    next_obs_time = None
    mus_gen = np.zeros((len(gen_times), batch_size, data_dim))
    sigmas_gen = np.zeros((len(gen_times), batch_size, data_dim ** 2))

    for i, step in tqdm.tqdm(enumerate(gen_times), total=len(gen_times)):
        next_T = current_t + delta_t * steps_ahead_prediction
        pred_drift, pred_drift_var = model_drift.generative_step(
            delta_t=delta_t, next_T=next_T, batch_size=batch_size,
            init_times=init_times,
            start_X=start_X_drift, init_X=init_X_drift,
            start_M=start_M_drift, init_M=init_M_drift,
            next_X=next_X_drift, next_obs_time=next_obs_time,
            paths=paths[:, :len(init_times)+i+1, :])
        if joint_model:
            pred_diffusion = pred_drift_var
        else:
            pred_diffusion, _ = model_diffusion.generative_step(
                delta_t=delta_t, next_T=next_T, batch_size=batch_size,
                init_times=init_times,
                start_X=start_X_diffusion, init_X=init_X_diffusion,
                start_M=start_M_diffusion, init_M=init_M_diffusion,
                next_X=next_X_diffusion, next_obs_time=next_obs_time,
                paths=paths[:, :len(init_times)+i+1, :])
        if instantaneous_coeff_estimation:
            mu = pred_drift
            sigma = pred_diffusion
        else:
            if not drift_predicts_inc:
                pred_drift = pred_drift - current_X
            mu = pred_drift / (delta_t * steps_ahead_prediction)
            sigma = pred_diffusion / np.sqrt(delta_t * steps_ahead_prediction)

        gen_X = generate_next_value(
            X_t=current_X, mu_t=mu,
            sigma_t=sigma.view(batch_size, data_dim, data_dim),
            delta_t=delta_t)
        current_t += delta_t
        paths[:, len(init_times) + i + 1, :] = gen_X.numpy()
        mus_gen[i, :, :] = mu.numpy()
        sigmas_gen[i, :, :] = sigma.numpy()
        next_X_drift = build_input(
            X=gen_X.numpy(), last_X=current_X.numpy(), Delta_t=delta_t,
            divide_by_t=instantaneous_coeff_estimation,
            inputs_list=model_drift_inputs)
        if not joint_model:
            next_X_diffusion = build_input(
                X=gen_X.numpy(), last_X=current_X.numpy(), Delta_t=delta_t,
                divide_by_t=instantaneous_coeff_estimation,
                inputs_list=model_diffusion_inputs)
        next_obs_time = step
        current_X = gen_X
    mus_gen = np.transpose(mus_gen, (1, 0, 2))
    sigmas_gen = np.transpose(sigmas_gen, (1, 0, 2))

    # paths shape: (batch_size, timesteps, innerdim)
    # mus_gen shape: (batch_size, len(gen_times), innerdim)
    # sigmas_gen shape: (batch_size, len(gen_times), innerdim**2)
    res = [paths, all_times, gen_times]
    if return_coeffs:
        res = res + [mus_gen, sigmas_gen]
    return res


def build_input(X, last_X, Delta_t, divide_by_t, inputs_list, mask=False):
    if mask:
        X_inc = X * last_X
    else:
        X_inc = X - last_X
    Z = np.expand_dims(X_inc, axis=2)
    QV = np.expand_dims(X_inc, axis=2)
    if divide_by_t:
        X_inc = X_inc / Delta_t
        Z = Z / np.sqrt(Delta_t)
    Z = np.matmul(Z, Z.transpose(0, 2, 1)).reshape(Z.shape[0], -1)
    QV = np.matmul(QV, QV.transpose(0, 2, 1)).reshape(QV.shape[0], -1)
    Z_plus = np.zeros_like(Z)
    input = np.empty((X.shape[0], 0))
    if "X" in inputs_list:
        input = np.concatenate((X, input), axis=1)
    if "X_inc" in inputs_list:
        input = np.concatenate([X_inc, input], axis=1)
    if "Z" in inputs_list:
        input = np.concatenate([Z_plus, Z, input], axis=1)
    if "QV" in inputs_list:
        input = np.concatenate([Z_plus, QV, input], axis=1)
    return input


def build_startX(X, inputs_list):
    d = X.shape[0]
    X_inc = np.zeros((d,))
    Z = np.zeros((d**2,))
    input = np.empty((0,))
    if "X" in inputs_list:
        input = np.concatenate((X, input), axis=0)
    if "X_inc" in inputs_list:
        input = np.concatenate([X_inc, input], axis=0)
    if "Z" in inputs_list:
        input = np.concatenate([Z, Z, input], axis=0)
    if "QV" in inputs_list:
        input = np.concatenate([Z, Z, input], axis=0)
    return input


def generate_next_value(X_t, mu_t, sigma_t, delta_t):
    """
    Generate the next value in the time series using the Euler-Maruyama scheme.

    :param X_t: current value tensor of shape (batch_size, d)
    :param mu_t: drift coefficient tensor of shape (batch_size, d)
    :param sigma_t: diffusion coefficient tensor of shape (batch_size, d, d)
    :param delta_t: time difference float
    :return: next value tensor of shape (batch_size, d)
    """
    delta_Wt = torch.randn_like(X_t) * np.sqrt(delta_t)
    delta_Wt_sigma_t = torch.bmm(sigma_t, delta_Wt.unsqueeze(
       2)).squeeze(2)
    X_t_next = X_t + mu_t * delta_t + delta_Wt_sigma_t

    return X_t_next


def plot_gen_paths(
    x_fake, x_real=None, times=None,
    titles=["Real", "Generated"], file_path=None,
    dim=0, paths_to_plot=100,
):
    plot_size = min(paths_to_plot, x_fake.shape[0],
                    x_real.shape[0] if x_real is not None else paths_to_plot)
    if times is None:
        times = np.arange(x_fake.shape[1])
    dims = x_fake.shape[2]

    ncols = 1
    ax_fake = 0
    if x_real is not None:
        ncols = 2
        ax_fake = 1
    fig, ax = plt.subplots(
        1, ncols, figsize=[6*ncols, 4], sharex=True, sharey=True)
    if ncols == 1:
        ax = [ax]

    if x_real is not None:
        ax[0].plot(
            times, x_real[:plot_size, :, dim].T, alpha=0.3, marker="o",
            linewidth=1, markersize=1,)
    ax[ax_fake].plot(
        times, x_fake[:plot_size, :, dim].T, alpha=0.3, marker="o", linewidth=1,
        markersize=1,)

    if titles:
        if ncols == 1:
            ax[0].set_title(titles[1])
        else:
            ax[0].set_title(titles[0])
            ax[1].set_title(titles[1])

    for i in range(ncols):
        ax[i].set_xlabel("$t$")
    ax[0].set_ylabel("X[{}]".format(dim) if dims > 1 else "X")
    plt.tight_layout()

    files = []
    if file_path is not None:
        file = os.path.join(file_path, 'paths_dim-{}.pdf'.format(dim))
        fig.savefig(file)
        files.append(file)
    plt.close()

    return files


def estimate_params_GBM(X, dt=0.01):
  rets = np.log(X[:,1:]/X[:,:-1])
  r=rets.ravel()
  m=np.mean(r)
  s=np.std(r)
  sigma=s/np.sqrt(dt)
  mu=m/dt+0.5*sigma**2
  return mu, sigma


def estimate_params_OU(paths, dt=0.01, dim=0):
    """
    estimate coefficients of the Ornstein-Uhlenbeeck model, which is defined by
    the SDE:
        dX = -kappa*(X-theta)*dt + sig*dW
    -> mean = theta, speed = kappa, volatility = sig

    Args:
        paths: np.array of shape (batch_size, timesteps, innerdim)
        dt:

    Returns: kappa, theta, sig
    """
    XX = np.concatenate([paths[i, :-1, dim] for i in range(len(paths))])
    YY = np.concatenate([paths[i, 1:, dim] for i in range(len(paths))])
    beta, alpha, _, _, _ = stats.linregress(XX, YY)  # OLS
    kappa_ols = -np.log(beta) / dt
    theta_ols = alpha / (1 - beta)
    res = YY - beta * XX - alpha  # residuals
    std_resid = np.std(res, ddof=2)
    sig_ols = std_resid * np.sqrt(2 * kappa_ols / (1 - beta ** 2))

    return kappa_ols, theta_ols, sig_ols



def plot_coeff_paths(
    mus_gen, sigmas_gen, mus_true=None, sigmas_true=None,
    times=None, file_path=None, paths_to_plot=(0,1,2,3,4),
):
    if times is None:
        times = np.arange(mus_gen.shape[1])
    dims = mus_gen.shape[2]

    if file_path is not None:
        filename = os.path.join(file_path, 'coeff_paths-{}_dim-{}.pdf')

    # square sigmas
    sigmas_gen = sigmas_gen.reshape(
        sigmas_gen.shape[0], sigmas_gen.shape[1], dims, dims)
    sigmas_gen = np.matmul(sigmas_gen, sigmas_gen.transpose((0,1,3,2)))
    if sigmas_true is not None:
        sigmas_true = sigmas_true.reshape(
            sigmas_true.shape[0], sigmas_true.shape[1], dims, dims)
        sigmas_true = np.matmul(
            sigmas_true, sigmas_true.transpose((0,1,3,2)))

    files = []
    for p in paths_to_plot:
        for d in range(dims):
            fig, ax = plt.subplots(
                1, 2, figsize=[12, 4], sharex=True)
            ax[0].plot(
                times, mus_gen[p, :, d], alpha=0.7, marker="o",
                linewidth=1, markersize=3, label="estimated")
            if mus_true is not None:
                ax[0].plot(
                    times, mus_true[p, :, d], alpha=0.7, marker="o",
                    linewidth=1, markersize=3, label="true")
            ax[1].plot(
                times, sigmas_gen[p, :, d, d], alpha=0.7, marker="o",
                linewidth=1, markersize=3, label="estimated")
            if sigmas_true is not None:
                ax[1].plot(
                    times, sigmas_true[p, :, d, d], alpha=0.7,
                    marker="o", linewidth=1, markersize=3, label="true")
            ax[0].set_title(
                "Drift coefficient" + "$\\mu$[{}]".format(
                    d) if dims > 1 else "$\\mu$")
            ax[1].set_title(
                "Diffusion coefficient" + "$\\sigma^2$[{},{}]".format(
                    d, d) if dims > 1 else "$\\sigma^2$")
            ax[0].set_xlabel("$t$")
            ax[1].set_xlabel("$t$")
            ax[0].legend()
            plt.tight_layout()
            if file_path is not None:
                fig.savefig(filename.format(p, d))
                files.append(filename.format(p, d))
            plt.close()
    return files


def plot_distribution_at_t(
    x_fake, t_indices, num_bins=50, x_real=None, file_path=None,
    times=None
):
    if times is None:
        times = np.arange(x_fake.shape[1])
    data_dim = x_fake.shape[2]
    if file_path is not None:
        filename = os.path.join(file_path, 'distribution_at_t-{}_dim-{}.pdf')

    files = []
    for d in range(data_dim):
        for t_idx in t_indices:
            fig, ax = plt.subplots(1, 1, figsize=[6, 4])
            ax.hist(
                x_fake[:, t_idx, d], bins=num_bins, density=True,
                alpha=0.5, label="generated")
            if x_real is not None:
                ax.hist(
                    x_real[:, t_idx, d], bins=num_bins, density=True,
                    alpha=0.5, label="real")
            ax.set_title(
                "Distribution at t={:.2f}".format(times[t_idx]) +
                (" of X[{}]".format(d) if data_dim > 1 else " of X"))
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend()
            plt.tight_layout()
            if file_path is not None:
                fn = filename.format(times[t_idx], d)
                fig.savefig(fn)
                files.append(fn)
            plt.close()
    return files


def main(arg):
    """
    function to run generation of paths
    """
    del arg
    params = None
    if FLAGS.params:
        params = eval("config."+FLAGS.params)
    # set number of CPUs
    torch.set_num_threads(FLAGS.NB_CPUS)
    if params is not None:
        generate_paths(
            **params, send=FLAGS.SEND, use_gpu=FLAGS.USE_GPU, gpu_num=FLAGS.GPU_NUM)


if __name__ == '__main__':
    app.run(main)


