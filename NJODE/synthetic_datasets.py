"""
author: Florian Krach & Calypso Herrera

code to generate synthetic data from stock-model SDEs
"""

# ==============================================================================
import numpy as np
import scipy
import torch
import scipy.stats as stats
import matplotlib.pyplot as plt
import copy
from fbm import fbm, fgn  # import fractional brownian motion package
from scipy.special import iv

from loss_functions import LOSS_FUN_DICT

import matplotlib.animation as animation


# ==============================================================================
# CLASSES
class StockModel:
    """
    mother class for all stock models defining the variables and methods shared
    amongst all of them, some need to be defined individually
    """

    def __init__(self, drift, volatility, S0, nb_paths, nb_steps,
                 maturity, sine_coeff, **kwargs):
        self.drift = drift
        self.volatility = volatility
        self.S0 = S0
        self.nb_paths = nb_paths
        self.nb_steps = nb_steps
        self.maturity = maturity
        self.dt = maturity / nb_steps
        self.dimensions = np.size(S0)
        if sine_coeff is None:
            self.periodic_coeff = lambda t: 1
        else:
            self.periodic_coeff = lambda t: (1 + np.sin(sine_coeff * t))
        self.loss = None
        self.path_t = None
        self.path_y = None
        self.path_var_y = None

        self.return_var_implemented = False
        self.loss_comp_for_pow2_implemented = False

    def generate_paths(self, **options):
        """
        generate random paths according to the model hyperparams
        :return: stock paths as np.array, dim: [nb_paths, data_dim, nb_steps]
        """
        raise ValueError("not implemented yet")

    def next_cond_exp(self, *args, **kwargs):
        """
        compute the next point of the conditional expectation starting from
        given point for given time_delta
        :return: cond. exp. at next time_point (= current_time + time_delta)
        """
        raise ValueError("not implemented yet")

    def compute_cond_exp(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, return_path=True, get_loss=False,
                         weight=0.5, store_and_use_stored=True,
                         start_time=None, which_loss="easy",
                         return_var=False,
                         **kwargs):
        """
        compute conditional expectation similar to computing the prediction in
        the model.NJODE.forward
        ATTENTION: Works correctly only for non-masked data!
        :param times: see model.NJODE.forward
        :param time_ptr: see model.NJODE.forward
        :param X: see model.NJODE.forward, as np.array
        :param obs_idx: see model.NJODE.forward, as np.array
        :param delta_t: see model.NJODE.forward, as np.array
        :param T: see model.NJODE.forward
        :param start_X: see model.NJODE.forward, as np.array
        :param n_obs_ot: see model.NJODE.forward, as np.array
        :param return_path: see model.NJODE.forward
        :param get_loss: see model.NJODE.forward
        :param weight: see model.NJODE.forward
        :param store_and_use_stored: bool, whether the loss, and cond exp path
            should be stored and reused when calling the function again
        :param start_time: None or float, if float, this is first time point
        :param return_var: bool, whether to return the variance additionally to
                the process; only if self.return_var_implemented is True
        :param kwargs: unused, to allow for additional unused inputs
        :return: float (loss), if wanted paths of t and y (np.arrays)
        """
        if return_var and not self.return_var_implemented:
            return_var = False

        if return_path and store_and_use_stored:
            res = [self.loss, self.path_t, self.path_y]
            if return_var:
                res.append(self.path_var_y)
            if get_loss:
                if self.path_t is not None and self.loss is not None:
                    return res
            else:
                if self.path_t is not None:
                    return res
        elif store_and_use_stored:
            if get_loss:
                if self.loss is not None:
                    return self.loss

        y = start_X
        if return_var:
            var_y = start_X*0
        batch_size = start_X.shape[0]
        current_time = 0.0
        if start_time:
            current_time = start_time

        loss = 0

        if return_path:
            if start_time:
                path_t = []
                path_y = []
                if return_var:
                    path_var_y = []
            else:
                path_t = [0.]
                path_y = [y]
                if return_var:
                    path_var_y = [var_y]

        for i, obs_time in enumerate(times):
            # the following is needed for the combined stock model datasets
            if obs_time > T + 1e-10*delta_t:
                break
            if obs_time <= current_time:
                continue
            # Calculate conditional expectation stepwise
            while current_time < (obs_time - 1e-10*delta_t):
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time
                y = self.next_cond_exp(y, delta_t_, current_time)
                if return_var:
                    var_y = self.next_cond_exp(
                        var_y, delta_t_, current_time, variance=True)
                current_time = current_time + delta_t_

                # Storing the conditional expectation
                if return_path:
                    path_t.append(current_time)
                    path_y.append(y)
                    if return_var:
                        path_var_y.append(var_y)

            # Reached an observation - set new interval
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]

            # Update h. Also updating loss, tau and last_X
            Y_bj = y
            temp = copy.deepcopy(y)
            temp[i_obs] = X_obs
            y = temp
            Y = y
            if return_var:
                temp_var = copy.deepcopy(var_y)
                temp_var[i_obs] = X_obs*0
                var_y = temp_var

            if get_loss:
                loss = loss + compute_loss(
                    X_obs=X_obs, Y_obs=Y[i_obs], Y_obs_bj=Y_bj[i_obs],
                    n_obs_ot=n_obs_ot[i_obs],
                    batch_size=batch_size, weight=weight, which_loss=which_loss)
            if return_path:
                path_t.append(obs_time)
                path_y.append(y)
                if return_var:
                    path_var_y.append(var_y)

        # after every observation has been processed, propagating until T
        while current_time < T - 1e-10 * delta_t:
            if current_time < T - delta_t:
                delta_t_ = delta_t
            else:
                delta_t_ = T - current_time
            y = self.next_cond_exp(y, delta_t_, current_time)
            if return_var:
                var_y = self.next_cond_exp(
                    var_y, delta_t_, current_time, variance=True)
            current_time = current_time + delta_t_

            # Storing the predictions.
            if return_path:
                path_t.append(current_time)
                path_y.append(y)
                if return_var:
                    path_var_y.append(var_y)

        if get_loss and store_and_use_stored:
            self.loss = loss
        if return_path and store_and_use_stored:
            self.path_t = np.array(path_t)
            self.path_y = np.array(path_y)
            if return_var:
                self.path_var_y = np.array(path_var_y)

        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            res = [loss, np.array(path_t), np.array(path_y)]
            if return_var:
                res.append(np.array(path_var_y))
            return res
        else:
            return loss

    def get_optimal_loss(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, weight=0.5, M=None, mult=None,
                         store_and_use_stored=True, return_var=False,
                         which_loss="easy"):
        if mult is not None and mult > 1:
            bs, dim = start_X.shape
            _dim = round(dim / mult)
            X = X[:, :_dim]
            start_X = start_X[:, :_dim]
            if M is not None:
                M = M[:, :_dim]

        # depending on whether method can return the variance, the result has
        #   len 3 or 4
        res = self.compute_cond_exp(
            times, time_ptr, X, obs_idx, delta_t, T, start_X, n_obs_ot,
            return_path=True, get_loss=True, weight=weight, M=M,
            store_and_use_stored=store_and_use_stored, return_var=return_var,
            which_loss=which_loss)
        loss = res[0]
        return loss


class Heston(StockModel):
    """
    the Heston model, see: https://en.wikipedia.org/wiki/Heston_model
    a basic stochastic volatility stock price model
    """

    def __init__(self, drift, volatility, mean, speed, correlation, nb_paths,
                 nb_steps, S0, maturity, sine_coeff=None, **kwargs):
        super(Heston, self).__init__(
            drift=drift, volatility=volatility, nb_paths=nb_paths,
            nb_steps=nb_steps,
            S0=S0, maturity=maturity,
            sine_coeff=sine_coeff
        )
        self.mean = mean
        self.speed = speed
        self.correlation = correlation

    def next_cond_exp(self, y, delta_t, current_t):
        return y * np.exp(self.drift * self.periodic_coeff(current_t) * delta_t)

    def generate_paths(self, start_X=None):
        # Diffusion of the spot: dS = mu*S*dt + sqrt(v)*S*dW
        spot_drift = lambda x, t: self.drift * self.periodic_coeff(t) * x
        spot_diffusion = lambda x, v, t: np.sqrt(v) * x

        # Diffusion of the variance: dv = -k(v-vinf)*dt + sqrt(v)*v*dW
        var_drift = lambda v, t: - self.speed * (v - self.mean)
        var_diffusion = lambda v, t: self.volatility * np.sqrt(v)

        spot_paths = np.empty(
            (self.nb_paths, self.dimensions, self.nb_steps + 1))
        var_paths = np.empty(
            (self.nb_paths, self.dimensions, self.nb_steps + 1))

        dt = self.dt
        if start_X is not None:
            spot_paths[:, :, 0] = start_X
        for i in range(self.nb_paths):
            if start_X is None:
                spot_paths[i, :, 0] = self.S0
            var_paths[i, :, 0] = self.mean
            for k in range(1, self.nb_steps + 1):
                normal_numbers_1 = np.random.normal(0, 1, self.dimensions)
                normal_numbers_2 = np.random.normal(0, 1, self.dimensions)
                dW = normal_numbers_1 * np.sqrt(dt)
                dZ = (self.correlation * normal_numbers_1 + np.sqrt(
                    1 - self.correlation ** 2) * normal_numbers_2) * np.sqrt(dt)

                var_paths[i, :, k] = (
                        var_paths[i, :, k - 1]
                        + var_drift(var_paths[i, :, k - 1], (k) * dt) * dt
                        + var_diffusion(var_paths[i, :, k - 1], (k) * dt) * dZ)

                spot_paths[i, :, k] = (
                        spot_paths[i, :, k - 1]
                        + spot_drift(spot_paths[i, :, k - 1], (k - 1) * dt) * dt
                        + spot_diffusion(spot_paths[i, :, k - 1],
                                                var_paths[i, :, k],
                                                (k) * dt) * dW)
        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


class HestonWOFeller(StockModel):
    """
    the Heston model, see: https://en.wikipedia.org/wiki/Heston_model
    a basic stochastic volatility stock price model, that can be used
    even if Feller condition is not satisfied
    Feller condition: 2*speed*mean > volatility**2
    """

    def __init__(self, drift, volatility, mean, speed, correlation, nb_paths,
                 nb_steps, S0, maturity, scheme='euler', return_vol=False,
                 v0=None, sine_coeff=None, **kwargs):
        super(HestonWOFeller, self).__init__(
            drift=drift, volatility=volatility, nb_paths=nb_paths,
            nb_steps=nb_steps,
            S0=S0, maturity=maturity,
            sine_coeff=sine_coeff
        )
        self.mean = mean
        self.speed = speed
        self.correlation = correlation

        self.scheme = scheme
        self.retur_vol = return_vol
        if v0 is None:
            self.v0 = self.mean
        else:
            self.v0 = v0

    def next_cond_exp(self, y, delta_t, current_t):
        if self.retur_vol:
            s, v = np.split(y, indices_or_sections=2, axis=1)
            s = s * np.exp(self.drift * self.periodic_coeff(current_t) * delta_t)
            exp_delta = np.exp(-self.speed * delta_t)
            v = v * exp_delta + self.mean * (1 - exp_delta)
            y = np.concatenate([s, v], axis=1)
            return y
        else:
            return y * np.exp(self.drift * self.periodic_coeff(current_t) * delta_t)

    def generate_paths(self, start_X=None):
        if self.scheme == 'euler':
            # Diffusion of the spot: dS = mu*S*dt + sqrt(v)*S*dW
            log_spot_drift = lambda v, t: \
                (self.drift * self.periodic_coeff(t) - 0.5 * np.maximum(v, 0))
            log_spot_diffusion = lambda v: np.sqrt(np.maximum(v, 0))

            # Diffusion of the variance: dv = -k(v-vinf)*dt + sqrt(v)*v*dW
            var_drift = lambda v: - self.speed * (np.maximum(v, 0) - self.mean)
            var_diffusion = lambda v: self.volatility * np.sqrt(np.maximum(v, 0))

            spot_paths = np.empty(
                (self.nb_paths, self.dimensions, self.nb_steps + 1))
            var_paths = np.empty(
                (self.nb_paths, self.dimensions, self.nb_steps + 1))

            dt = self.dt
            if start_X is not None:
                spot_paths[:, :, 0] = start_X
            for i in range(self.nb_paths):
                if start_X is None:
                    spot_paths[i, :, 0] = self.S0
                var_paths[i, :, 0] = self.v0
                for k in range(1, self.nb_steps + 1):
                    normal_numbers_1 = np.random.normal(0, 1, self.dimensions)
                    normal_numbers_2 = np.random.normal(0, 1, self.dimensions)
                    dW = normal_numbers_1 * np.sqrt(dt)
                    dZ = (self.correlation * normal_numbers_1 + np.sqrt(
                        1 - self.correlation ** 2) * normal_numbers_2) * np.sqrt(dt)

                    spot_paths[i, :, k] = np.exp(
                        np.log(spot_paths[i, :, k - 1])
                        + log_spot_drift(
                            var_paths[i, :, k - 1], (k - 1) * dt) * dt
                        + log_spot_diffusion(var_paths[i, :, k - 1]) * dW
                    )
                    var_paths[i, :, k] = (
                            var_paths[i, :, k - 1]
                            + var_drift(var_paths[i, :, k - 1]) * dt
                            + var_diffusion(var_paths[i, :, k - 1]) * dZ
                    )
            if self.retur_vol:
                spot_paths = np.concatenate([spot_paths, var_paths], axis=1)
            # stock_path dimension: [nb_paths, dimension, time_steps]
            return spot_paths, dt

        else:
            raise ValueError('unknown sampling scheme')


class BlackScholes(StockModel):
    """
    standard Black-Scholes model, see:
    https://en.wikipedia.org/wiki/Black–Scholes_model
    https://en.wikipedia.org/wiki/Geometric_Brownian_motion
    """

    def __init__(self, drift, volatility, nb_paths, nb_steps, S0,  # initialize relevant parameters
                 maturity, sine_coeff=None, **kwargs):
        super(BlackScholes, self).__init__(
            drift=drift, volatility=volatility, nb_paths=nb_paths,
            nb_steps=nb_steps, S0=S0, maturity=maturity,
            sine_coeff=sine_coeff
        )

    def next_cond_exp(self, y, delta_t, current_t):
        return y * np.exp(self.drift * self.periodic_coeff(current_t) * delta_t)

    def generate_paths(self, start_X=None):
        drift = lambda x, t: self.drift * self.periodic_coeff(t) * x
        diffusion = lambda x, t: self.volatility * x
        spot_paths = np.empty(
            (self.nb_paths, self.dimensions, self.nb_steps + 1))
        dt = self.dt
        if start_X is not None:
            spot_paths[:, :, 0] = start_X
        for i in range(self.nb_paths):
            if start_X is None:
                spot_paths[i, :, 0] = self.S0
            for k in range(1, self.nb_steps + 1):
                random_numbers = np.random.normal(0, 1, self.dimensions)
                dW = random_numbers * np.sqrt(dt)
                spot_paths[i, :, k] = (
                        spot_paths[i, :, k - 1]
                        + drift(spot_paths[i, :, k - 1], (k - 1) * dt) * dt
                        + diffusion(spot_paths[i, :, k - 1], (k) * dt) * dW)
        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


DISTRIBUTIONS = {
    'normal': np.random.normal,
    'uniform': np.random.uniform,
    'gamma': np.random.gamma,
    'exponential': np.random.exponential,
    'lognormal': np.random.lognormal,
    'poisson': np.random.poisson,
    'beta': np.random.beta,
    'chisquare': np.random.chisquare,
    'dirichlet': np.random.dirichlet,
    'f': np.random.f,
    'geometric': np.random.geometric,
    'gumbel': np.random.gumbel,
    'laplace': np.random.laplace,
    'logistic': np.random.logistic,
    'multinomial': np.random.multinomial,
    'multivariate_normal': np.random.multivariate_normal,
    'noncentral_chisquare': np.random.noncentral_chisquare,
    'noncentral_f': np.random.noncentral_f,
    'pareto': np.random.pareto,
    'rayleigh': np.random.rayleigh,
    'triangular': np.random.triangular,
    'vonmises': np.random.vonmises,
    'wald': np.random.wald,
    'weibull': np.random.weibull,
    'zipf': np.random.zipf,
    'fixed': lambda size, value: np.array([value]).repeat(
        np.prod(size), axis=0).reshape(size),
}

class BlackScholesUncertainParams(StockModel):
    """
    standard Black-Scholes model, see:
    https://en.wikipedia.org/wiki/Black–Scholes_model
    https://en.wikipedia.org/wiki/Geometric_Brownian_motion
    """

    def __init__(self, drift_dist, volatility_dist, nb_paths, nb_steps, S0,
                 maturity, num_particles=1000, **kwargs):
        self.drift_dist = lambda size: DISTRIBUTIONS[drift_dist["dist"]](
            size=size, **drift_dist["params"])
        self.vola_dist = lambda size: DISTRIBUTIONS[volatility_dist["dist"]](
            size=size, **volatility_dist["params"])
        self.drift_dist_name = drift_dist["dist"]
        self.volatility_dist_name = volatility_dist["dist"]
        self.drift_mean = np.mean(self.drift_dist(100000))
        self.volatility_mean = np.mean(self.vola_dist(100000))

        self.num_particles = num_particles

        super(BlackScholesUncertainParams, self).__init__(
            drift=None, volatility=None, nb_paths=nb_paths,
            nb_steps=nb_steps, S0=S0, maturity=maturity,
            sine_coeff=None,
        )

    def next_cond_exp(self, y, delta_t, current_t):
        return y

    def get_mu(self, jj, **kwargs):
        """
        approximate the conditional expectation of the parameters of the
        Black-Scholes model using the observed data via an extension (for
        irregular time steps) of:
        https://thescipub.com/pdf/jmssp.2010.421.424.pdf
        """
        logrets = np.log(np.array(self.observed_X[jj][1:]) /
                         np.array(self.observed_X[jj][:-1]))
        m = np.mean(logrets/np.array(self.observed_t_inc[jj]))
        sqrt_t_inc = np.sqrt(np.array(self.observed_t_inc[jj]))
        sig2 = np.mean((logrets/sqrt_t_inc - m*sqrt_t_inc) ** 2)
        mu = m + sig2/2
        if self.drift_dist_name == "fixed":
            mu = self.drift_dist(1)[0]
        if self.volatility_dist_name == "fixed":
            sig2 = self.vola_dist(1)[0]**2
            mu = m + sig2/2
        return np.array([mu, np.sqrt(sig2)])

    def compute_fin_estimator(
            self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
            n_obs_ot, return_path=True, get_loss=False,
            weight=0.5, store_and_use_stored=True,
            start_time=None, M=None, which_loss="easy", **kwargs):
        """
        Compute conditional expectation
        :param times: np.array, of observation times
        :param time_ptr: list, start indices of X and obs_idx for a given
                observation time, first element is 0, this pointer tells how
                many (and which) of the observations of X along the batch-dim
                belong to the current time, and obs_idx then tells to which of
                the batch elements they belong. In particular, not each batch-
                element has to jump at the same time, and only those elements
                which jump at the current time should be updated with a jump
        :param X: np.array, data tensor
        :param obs_idx: list, index of the batch elements where jumps occur at
                current time
        :param delta_t: float, time step for Euler
        :param T: float, the final time
        :param start_X: np.array, the starting point of X
        :param n_obs_ot: np.array, the number of observations over the
                entire time interval for each element of the batch
        :param return_path: bool, whether to return the path of h
        :param get_loss: bool, whether to compute the loss, otherwise 0 returned
        :param until_T: bool, whether to continue until T (for eval) or only
                until last observation (for training)
        :param M: None or np.array, if not None: the mask for the data, same
                size as X, with 0 or 1 entries
        :return: float (loss), if wanted paths of t and y (np.arrays)
        """
        if return_path and store_and_use_stored:
            if get_loss:
                if self.path_t is not None and self.loss is not None:
                    return self.loss, self.path_t, self.path_y
            else:
                if self.path_t is not None:
                    return self.loss, self.path_t, self.path_y
        elif store_and_use_stored:
            if get_loss:
                if self.loss is not None:
                    return self.loss

        bs = start_X.shape[0]
        self.observed_X = [[self.S0] for x in range(bs)]
        self.observed_t = [[] for x in range(bs)]
        self.observed_t_inc = [[] for x in range(bs)]

        y = np.array([[self.drift_mean, self.volatility_mean]]).repeat(
            bs, axis=0)
        loss = 0
        batch_size = bs
        current_time = 0.0
        if start_time:
            current_time = start_time
        if return_path:
            if start_time:
                path_t = []
                path_y = []
            else:
                path_t = [0.]
                path_y = [y]

        for i, obs_time in enumerate(times):
            # Calculate conditional expectation stepwise
            while current_time < (obs_time - 1e-10 * delta_t):
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time
                y = self.next_cond_exp(y, delta_t_, current_time)
                current_time = current_time + delta_t_

                # Storing the conditional expectation
                if return_path:
                    path_t.append(current_time)
                    path_y.append(y)

            # Reached an observation - set new interval
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]

            # Update h. Also updating loss, tau and last_X
            Y_bj = y
            temp = copy.copy(y)
            for j, jj in enumerate(i_obs):
                self.observed_X[jj].append(X_obs[j, 0])
                self.observed_t[jj].append(obs_time)
                last = self.observed_t[jj][-2] if len(
                    self.observed_t[jj]) > 1 else 0
                self.observed_t_inc[jj].append(obs_time - last)
                temp[jj] = self.get_mu(jj=jj)
            y = temp
            Y = y

            if get_loss:
                loss = loss + compute_loss(
                    X_obs=X_obs[:, 1:], Y_obs=Y[i_obs], Y_obs_bj=Y_bj[i_obs],
                    n_obs_ot=n_obs_ot[i_obs], batch_size=batch_size,
                    weight=weight, M_obs=None, which_loss=which_loss)
            if return_path:
                path_t.append(obs_time)
                path_y.append(y)

        # after every observation has been processed, propagating until T
        while current_time < T - 1e-10 * delta_t:
            if current_time < T - delta_t:
                delta_t_ = delta_t
            else:
                delta_t_ = T - current_time
            y = self.next_cond_exp(y, delta_t_)
            current_time = current_time + delta_t_

            # Storing the predictions.
            if return_path:
                path_t.append(current_time)
                path_y.append(y)

        if get_loss and store_and_use_stored:
            self.loss = loss
        if return_path and store_and_use_stored:
            self.path_t = np.array(path_t)
            self.path_y = np.array(path_y)

        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            return loss, np.array(path_t), np.array(path_y)
        else:
            return loss

    @staticmethod
    def transition_density(x_curr, x_prev, mu, sigma, delta_t):
        # Log transformation
        log_mean = (mu - 0.5 * sigma ** 2) * delta_t
        log_var = sigma ** 2 * delta_t

        # Log-normal density
        density = (1 / (x_curr * sigma * np.sqrt(2 * np.pi * delta_t)) *
                   np.exp(- (np.log(x_curr / x_prev) - log_mean) ** 2 / (
                               2 * log_var)))

        return density

    def update(self, jj):
        # Update the prediction
        x_prev = self.observed_X[jj][-2]
        x_curr = self.observed_X[jj][-1]
        delta_t = self.observed_t_inc[jj][-1]

        # Update weights
        for j in range(self.num_particles):
            self.weights[jj][j] *= self.transition_density(
                x_curr, x_prev, self.mu_particles[jj][j],
                self.sigma_particles[jj][j], delta_t)

        # Normalize weights
        self.weights[jj] /= np.sum(self.weights[jj])

        # # Resample particles based on weights
        # indices = np.random.choice(
        #     range(self.num_particles), size=self.num_particles,
        #     p=self.weights[jj])
        # mu_particles = self.mu_particles[jj][indices]
        # sigma_particles = self.sigma_particles[jj][indices]
        # weights = np.ones(self.num_particles) / self.num_particles

        # Compute conditional expectations
        hat_mu_t = np.average(
            self.mu_particles[jj], weights=self.weights[jj])
        hat_sigma_t = np.average(
            self.sigma_particles[jj], weights=self.weights[jj])

        return [hat_mu_t, hat_sigma_t]

    def _compute_cond_exp(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, return_path=True, get_loss=False,
                         weight=0.5, store_and_use_stored=True,
                         start_time=None, M=None, which_loss="easy",
                         **kwargs):
        """
        Compute conditional expectation
        :param times: np.array, of observation times
        :param time_ptr: list, start indices of X and obs_idx for a given
                observation time, first element is 0, this pointer tells how
                many (and which) of the observations of X along the batch-dim
                belong to the current time, and obs_idx then tells to which of
                the batch elements they belong. In particular, not each batch-
                element has to jump at the same time, and only those elements
                which jump at the current time should be updated with a jump
        :param X: np.array, data tensor
        :param obs_idx: list, index of the batch elements where jumps occur at
                current time
        :param delta_t: float, time step for Euler
        :param T: float, the final time
        :param start_X: np.array, the starting point of X
        :param n_obs_ot: np.array, the number of observations over the
                entire time interval for each element of the batch
        :param return_path: bool, whether to return the path of h
        :param get_loss: bool, whether to compute the loss, otherwise 0 returned
        :param until_T: bool, whether to continue until T (for eval) or only
                until last observation (for training)
        :param M: None or np.array, if not None: the mask for the data, same
                size as X, with 0 or 1 entries
        :return: float (loss), if wanted paths of t and y (np.arrays)
        """
        if return_path and store_and_use_stored:
            if get_loss:
                if self.path_t is not None and self.loss is not None:
                    return self.loss, self.path_t, self.path_y
            else:
                if self.path_t is not None:
                    return self.loss, self.path_t, self.path_y
        elif store_and_use_stored:
            if get_loss:
                if self.loss is not None:
                    return self.loss

        bs = start_X.shape[0]
        self.observed_X = [[self.S0] for x in range(bs)]
        self.observed_t = [[] for x in range(bs)]
        self.observed_t_inc = [[] for x in range(bs)]
        self.mu_particles = [self.drift_dist(self.num_particles) for x in range(bs)]
        self.sigma_particles = [self.vola_dist(self.num_particles) for x in range(bs)]
        self.weights = [np.ones(self.num_particles) / self.num_particles for x in range(bs)]

        y = np.array([[self.drift_mean, self.volatility_mean]]).repeat(
            bs, axis=0)
        loss = 0
        batch_size = bs
        current_time = 0.0
        if start_time:
            current_time = start_time
        if return_path:
            if start_time:
                path_t = []
                path_y = []
            else:
                path_t = [0.]
                path_y = [y]

        for i, obs_time in enumerate(times):
            # Calculate conditional expectation stepwise
            while current_time < (obs_time - 1e-10 * delta_t):
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time
                y = self.next_cond_exp(y, delta_t_, current_time)
                current_time = current_time + delta_t_

                # Storing the conditional expectation
                if return_path:
                    path_t.append(current_time)
                    path_y.append(y)

            # Reached an observation - set new interval
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]

            # Update h. Also updating loss, tau and last_X
            Y_bj = y
            temp = copy.deepcopy(y)
            for j, jj in enumerate(i_obs):
                self.observed_X[jj].append(X_obs[j, 0])
                self.observed_t[jj].append(obs_time)
                last = self.observed_t[jj][-2] if len(
                    self.observed_t[jj]) > 1 else 0
                self.observed_t_inc[jj].append(obs_time - last)
                temp[jj] = self.update(jj)
            y = temp
            Y = y

            if get_loss:
                loss = loss + compute_loss(
                    X_obs=X_obs[:, 1:], Y_obs=Y[i_obs], Y_obs_bj=Y_bj[i_obs],
                    n_obs_ot=n_obs_ot[i_obs], batch_size=batch_size,
                    weight=weight, M_obs=None, which_loss=which_loss)
            if return_path:
                path_t.append(obs_time)
                path_y.append(y)

        # after every observation has been processed, propagating until T
        while current_time < T - 1e-10 * delta_t:
            if current_time < T - delta_t:
                delta_t_ = delta_t
            else:
                delta_t_ = T - current_time
            y = self.next_cond_exp(y, delta_t_)
            current_time = current_time + delta_t_

            # Storing the predictions.
            if return_path:
                path_t.append(current_time)
                path_y.append(y)

        if get_loss and store_and_use_stored:
            self.loss = loss
        if return_path and store_and_use_stored:
            self.path_t = np.array(path_t)
            self.path_y = np.array(path_y)

        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            return loss, np.array(path_t), np.array(path_y)
        else:
            return loss

    def compute_cond_exp(
            self, times, time_ptr, X, obs_idx, delta_t, T,
            start_X, n_obs_ot, return_path=True, get_loss=False,
            weight=0.5, store_and_use_stored=True,
            start_time=None, M=None, which_loss="easy",
            ref_model=None, **kwargs):
        if ref_model in ["fin_estimator", 'financial estimator']:
            return self.compute_fin_estimator(
                times, time_ptr, X, obs_idx, delta_t, T, start_X,
                n_obs_ot, return_path=return_path, get_loss=get_loss,
                weight=weight, store_and_use_stored=store_and_use_stored,
                start_time=start_time, M=M, which_loss=which_loss, **kwargs)
        else:
            return self._compute_cond_exp(
                times, time_ptr, X, obs_idx, delta_t, T, start_X,
                n_obs_ot, return_path=return_path, get_loss=get_loss,
                weight=weight, store_and_use_stored=store_and_use_stored,
                start_time=start_time, M=M, which_loss=which_loss, **kwargs)

    def generate_paths(self, start_X=None):
        drift = self.drift_dist((self.nb_paths, self.dimensions))
        volatility = self.vola_dist((self.nb_paths, self.dimensions))
        spot_paths = np.empty(
            (self.nb_paths, self.dimensions*3, self.nb_steps + 1))
        spot_paths[:, self.dimensions:2*self.dimensions, :] = drift.reshape(
            (self.nb_paths, self.dimensions, 1)).repeat(self.nb_steps+1, axis=2)
        spot_paths[:, 2*self.dimensions:, :] = volatility.reshape(
            (self.nb_paths, self.dimensions, 1)).repeat(self.nb_steps+1, axis=2)
        dt = self.dt
        if start_X is not None:
            spot_paths[:, :, 0] = start_X
        else:
            spot_paths[:, :self.dimensions, 0] = np.array(self.S0).reshape(
                (1, -1)).repeat(self.nb_paths, axis=0)
        for k in range(1, self.nb_steps + 1):
            random_numbers = np.random.normal(
                0, 1, (self.nb_paths, self.dimensions))
            dW = random_numbers * np.sqrt(dt)
            spot_paths[:, :self.dimensions, k] = (
                    spot_paths[:, :self.dimensions, k - 1] +
                    drift * spot_paths[:, :self.dimensions, k - 1] * dt +
                    volatility * spot_paths[:, :self.dimensions, k - 1] * dW)
        # stock_path shape: [nb_paths, 3*dimension, time_steps]
        return spot_paths, dt


class CIRUncertainParams(StockModel):
    """
    Cox-Ingersoll-Ross (CIR) model, see:
    https://en.wikipedia.org/wiki/Cox–Ingersoll–Ross_model
    """

    def __init__(self, a_dist, b_dist, volatility_dist, nb_paths, nb_steps, S0,
                 maturity, num_particles=1000, sin_coeff=None, **kwargs):
        self.a_dist = lambda size: DISTRIBUTIONS[a_dist["dist"]](
            size=size, **a_dist["params"])
        self.b_dist = lambda size: DISTRIBUTIONS[b_dist["dist"]](
            size=size, **b_dist["params"])
        self.vola_dist = lambda size: DISTRIBUTIONS[volatility_dist["dist"]](
            size=size, **volatility_dist["params"])
        self.a_dist_name = a_dist["dist"]
        self.b_dist_name = b_dist["dist"]
        self.volatility_dist_name = volatility_dist["dist"]
        self.a_mean = np.mean(self.a_dist(100000))
        self.b_mean = np.mean(self.b_dist(100000))
        self.volatility_mean = np.mean(self.vola_dist(100000))

        if sin_coeff is None:
            sin_coeff = 0
        self.sin_coeff = sin_coeff
        self.num_particles = num_particles

        super(CIRUncertainParams, self).__init__(
            drift=None, volatility=None, nb_paths=nb_paths,
            nb_steps=nb_steps, S0=S0, maturity=maturity,
            sine_coeff=None,
        )

    def next_cond_exp(self, y, delta_t, current_t):
        return y

    @staticmethod
    def transition_density(x_curr, x_prev, a, b, sigma, delta_t):
        # CIR conditional density
        c = 2 * a / ((1 - np.exp(-a * delta_t)) * sigma ** 2)
        q = 2 * a * b / (sigma ** 2) - 1
        u = c * x_prev * np.exp(-a * delta_t)
        v = c * x_curr
        density = c * np.exp(-u - v) * (v / u) ** (q / 2) * \
            iv(q, 2 * np.sqrt(u * v))

        if np.isnan(density) or np.isinf(density):
            density = 0.
            # print("density is nan or inf")
            # print("c: ", c, "q: ", q, "u: ", u, "v: ", v)

        return density

    def update(self, jj, last_t):
        # Update the prediction
        x_prev = self.observed_X[jj][-2]
        x_curr = self.observed_X[jj][-1]
        delta_t = self.observed_t_inc[jj][-1]

        seasonality = 1
        if self.sin_coeff is not None:
            seasonality = 1 + 0.5 * np.sin(self.sin_coeff * last_t)

        # Update weights
        for j in range(self.num_particles):
            self.weights[jj][j] *= self.transition_density(
                x_curr, x_prev, self.a_particles[jj][j],
                self.b_particles[jj][j]*seasonality,
                self.sigma_particles[jj][j], delta_t)

        # Normalize weights
        if np.sum(self.weights[jj]) > 0:
            self.weights[jj] /= np.sum(self.weights[jj])
        else:
            self.weights[jj] = np.ones(self.num_particles) / self.num_particles

        # # Resample particles based on weights
        # indices = np.random.choice(
        #     range(self.num_particles), size=self.num_particles,
        #     p=self.weights[jj])
        # mu_particles = self.mu_particles[jj][indices]
        # sigma_particles = self.sigma_particles[jj][indices]
        # weights = np.ones(self.num_particles) / self.num_particles

        # Compute conditional expectations
        hat_a_t = np.average(
            self.a_particles[jj], weights=self.weights[jj])
        hat_b_t = np.average(
            self.b_particles[jj], weights=self.weights[jj])
        hat_sigma_t = np.average(
            self.sigma_particles[jj], weights=self.weights[jj])

        return [hat_a_t, hat_b_t, hat_sigma_t]

    def _compute_cond_exp(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, return_path=True, get_loss=False,
                         weight=0.5, store_and_use_stored=True,
                         start_time=None, M=None, which_loss="easy",
                         **kwargs):
        """
        Compute conditional expectation
        :param times: np.array, of observation times
        :param time_ptr: list, start indices of X and obs_idx for a given
                observation time, first element is 0, this pointer tells how
                many (and which) of the observations of X along the batch-dim
                belong to the current time, and obs_idx then tells to which of
                the batch elements they belong. In particular, not each batch-
                element has to jump at the same time, and only those elements
                which jump at the current time should be updated with a jump
        :param X: np.array, data tensor
        :param obs_idx: list, index of the batch elements where jumps occur at
                current time
        :param delta_t: float, time step for Euler
        :param T: float, the final time
        :param start_X: np.array, the starting point of X
        :param n_obs_ot: np.array, the number of observations over the
                entire time interval for each element of the batch
        :param return_path: bool, whether to return the path of h
        :param get_loss: bool, whether to compute the loss, otherwise 0 returned
        :param until_T: bool, whether to continue until T (for eval) or only
                until last observation (for training)
        :param M: None or np.array, if not None: the mask for the data, same
                size as X, with 0 or 1 entries
        :return: float (loss), if wanted paths of t and y (np.arrays)
        """
        if return_path and store_and_use_stored:
            if get_loss:
                if self.path_t is not None and self.loss is not None:
                    return self.loss, self.path_t, self.path_y
            else:
                if self.path_t is not None:
                    return self.loss, self.path_t, self.path_y
        elif store_and_use_stored:
            if get_loss:
                if self.loss is not None:
                    return self.loss

        bs = start_X.shape[0]
        self.observed_X = [[self.S0] for x in range(bs)]
        self.observed_t = [[] for x in range(bs)]
        self.observed_t_inc = [[] for x in range(bs)]
        self.a_particles = [self.a_dist(self.num_particles) for x in range(bs)]
        self.b_particles = [self.b_dist(self.num_particles) for x in range(bs)]
        self.sigma_particles = [self.vola_dist(self.num_particles) for x in range(bs)]
        self.weights = [np.ones(self.num_particles) / self.num_particles for x in range(bs)]

        y = np.array([[self.a_mean, self.b_mean, self.volatility_mean]]).repeat(
            bs, axis=0)
        loss = 0
        batch_size = bs
        current_time = 0.0
        if start_time:
            current_time = start_time
        if return_path:
            if start_time:
                path_t = []
                path_y = []
            else:
                path_t = [0.]
                path_y = [y]

        seasonality = np.ones((bs, 3))
        if self.sin_coeff is not None:
            seasonality[:, 1] = 1 + 0.5 * np.sin(
                self.sin_coeff * current_time)

        for i, obs_time in enumerate(times):
            # Calculate conditional expectation stepwise
            while current_time < (obs_time - 1e-10 * delta_t):
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time
                y = self.next_cond_exp(y, delta_t_, current_time)
                current_time = current_time + delta_t_

                # Storing the conditional expectation
                if return_path:
                    if self.sin_coeff is not None:
                        seasonality[:, 1] = 1 + 0.5 * np.sin(
                            self.sin_coeff * current_time)
                    path_t.append(current_time)
                    path_y.append(y*seasonality)

            # Reached an observation - set new interval
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]

            # Update h. Also updating loss, tau and last_X
            Y_bj = y
            temp = copy.deepcopy(y)
            for j, jj in enumerate(i_obs):
                self.observed_X[jj].append(X_obs[j, 0])
                self.observed_t[jj].append(obs_time)
                last = self.observed_t[jj][-2] if len(
                    self.observed_t[jj]) > 1 else 0
                self.observed_t_inc[jj].append(obs_time - last)
                temp[jj] = self.update(jj, last)
            y = temp
            Y = y

            if self.sin_coeff is not None:
                seasonality[:, 1] = 1 + 0.5 * np.sin(
                    self.sin_coeff * current_time)

            if get_loss:
                loss = loss + compute_loss(
                    X_obs=X_obs[:, 1:], Y_obs=Y[i_obs]*seasonality[i_obs],
                    Y_obs_bj=Y_bj[i_obs]*seasonality[i_obs],
                    n_obs_ot=n_obs_ot[i_obs], batch_size=batch_size,
                    weight=weight, M_obs=None, which_loss=which_loss)
            if return_path:
                path_t.append(obs_time)
                path_y.append(y*seasonality)

        # after every observation has been processed, propagating until T
        while current_time < T - 1e-10 * delta_t:
            if current_time < T - delta_t:
                delta_t_ = delta_t
            else:
                delta_t_ = T - current_time
            y = self.next_cond_exp(y, delta_t_, current_time)
            current_time = current_time + delta_t_

            # Storing the predictions.
            if return_path:
                if self.sin_coeff is not None:
                    seasonality[:, 1] = 1 + 0.5 * np.sin(
                        self.sin_coeff * current_time)
                path_t.append(current_time)
                path_y.append(y*seasonality)

        if get_loss and store_and_use_stored:
            self.loss = loss
        if return_path and store_and_use_stored:
            self.path_t = np.array(path_t)
            self.path_y = np.array(path_y)

        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            return loss, np.array(path_t), np.array(path_y)
        else:
            return loss

    def compute_cond_exp(
            self, times, time_ptr, X, obs_idx, delta_t, T,
            start_X, n_obs_ot, return_path=True, get_loss=False,
            weight=0.5, store_and_use_stored=True,
            start_time=None, M=None, which_loss="easy",
            ref_model=None, **kwargs):

        return self._compute_cond_exp(
            times, time_ptr, X, obs_idx, delta_t, T, start_X,
            n_obs_ot, return_path=return_path, get_loss=get_loss,
            weight=weight, store_and_use_stored=store_and_use_stored,
            start_time=start_time, M=M, which_loss=which_loss, **kwargs)

    def generate_paths(self, start_X=None):
        a = self.a_dist((self.nb_paths, self.dimensions))
        b = self.b_dist((self.nb_paths, self.dimensions))
        volatility = self.vola_dist((self.nb_paths, self.dimensions))
        seasonality = 1
        if self.sin_coeff is not None:
            seasonality = 1 + 0.5 * np.sin(self.sin_coeff * np.linspace(
                0, self.maturity, self.nb_steps + 1)).reshape(
                1, 1, -1).repeat(self.nb_paths, axis=0).repeat(
                self.dimensions, axis=1)
        spot_paths = np.empty(
            (self.nb_paths, self.dimensions*4, self.nb_steps + 1))
        spot_paths[:, self.dimensions:2*self.dimensions, :] = a.reshape(
            (self.nb_paths, self.dimensions, 1)).repeat(self.nb_steps+1, axis=2)
        spot_paths[:, 2*self.dimensions:3*self.dimensions, :] = b.reshape(
            (self.nb_paths, self.dimensions, 1)).repeat(
            self.nb_steps+1, axis=2) * seasonality
        spot_paths[:, 3*self.dimensions:, :] = volatility.reshape(
            (self.nb_paths, self.dimensions, 1)).repeat(self.nb_steps+1, axis=2)
        dt = self.dt
        if start_X is not None:
            spot_paths[:, :, 0] = start_X
        else:
            spot_paths[:, :self.dimensions, 0] = np.array(self.S0).reshape(
                (1, -1)).repeat(self.nb_paths, axis=0)
        for k in range(1, self.nb_steps + 1):
            random_numbers = np.random.normal(
                0, 1, (self.nb_paths, self.dimensions))
            dW = random_numbers * np.sqrt(dt)
            spot_paths[:, :self.dimensions, k] = np.maximum(0,
                    spot_paths[:, :self.dimensions, k - 1] +
                    a * (b - spot_paths[:, :self.dimensions, k - 1]) * dt +
                    volatility * np.sqrt(spot_paths[:, :self.dimensions, k-1])
                    * dW)
        # stock_path shape: [nb_paths, 4*dimension, time_steps]
        return spot_paths, dt


class PoissonPointProcess(StockModel):
    """
    standard Poisson Point Process model, see:
    https://en.wikipedia.org/wiki/Poisson_point_process
    """

    def __init__(self, poisson_lambda, nb_paths, nb_steps, S0,
                 maturity, sine_coeff=None, **kwargs):
        super().__init__(
            drift=None, volatility=None, nb_paths=nb_paths,
            nb_steps=nb_steps, S0=S0, maturity=maturity,
            sine_coeff=sine_coeff
        )
        self.poisson_lambda = poisson_lambda

    def next_cond_exp(self, y, delta_t, current_t):
        return y + self.poisson_lambda*delta_t

    def generate_paths(self, start_X=None):
        spot_paths = np.empty(
            (self.nb_paths, self.dimensions, self.nb_steps + 1))
        dt = self.dt
        if start_X is not None:
            spot_paths[:, :, 0] = start_X
        else:
            spot_paths[:, :, 0] = self.S0
        # generate arrival times
        exp_rvs = np.random.exponential(
            scale=1/self.poisson_lambda,
            size=(self.nb_paths, self.dimensions,
                  int(self.poisson_lambda*self.maturity)+1))
        while np.any(np.sum(exp_rvs, axis=2) <= self.maturity):
            exp_rvs_app = np.random.exponential(
                scale=1/self.poisson_lambda,
                size=(self.nb_paths, self.dimensions,
                      int(self.poisson_lambda*self.maturity)+1))
            exp_rvs = np.concatenate([exp_rvs, exp_rvs_app], axis=2)
        exp_rvs = np.cumsum(exp_rvs, axis=2)

        for k in range(1, self.nb_steps + 1):
            spot_paths[:, :, k] = np.argmin(exp_rvs <= k*dt, axis=2) + \
                                  spot_paths[:, :, 0]

        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


class BM(StockModel):
    """
    Brownian Motion
    """

    def __init__(
            self, nb_paths, nb_steps, maturity, dimension, drift=0, **kwargs):
        super().__init__(
            drift=drift, volatility=None, nb_paths=nb_paths,
            nb_steps=nb_steps, S0=0., maturity=maturity,
            sine_coeff=None,)
        assert dimension == 1
        self.return_var_implemented = True

    def next_cond_exp(self, y, delta_t, current_t, variance=False):
        if variance:
            next_y = y + delta_t
        else:
            next_y = y + self.drift * delta_t
        return next_y

    def generate_paths(self, start_X=None):
        spot_paths = np.zeros(
            (self.nb_paths, 1, self.nb_steps + 1))
        if start_X is not None:
            spot_paths[:, :, 0] = start_X
        dt = self.dt

        random_numbers = np.random.normal(
            0, 1, (self.nb_paths, 1, self.nb_steps)) * np.sqrt(dt)
        W = np.cumsum(random_numbers, axis=2)

        spot_paths[:, 0, 1:] = (
                spot_paths[:, :, 0] + W[:, 0, :] +
                self.drift * np.arange(1, self.nb_steps + 1) * dt)

        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


class BMandQuantiles(BM):
    """
    Brownian Motion. The conditional expectation returns the quantiles.
    """

    def __init__(self, nb_paths, nb_steps, maturity, dimension, **kwargs):
        super().__init__(nb_paths, nb_steps, maturity, dimension, **kwargs)
        assert dimension == 1
        self.quantiles = None

    def set_quantiles(self, quantiles):
        self.quantiles = quantiles

    def next_quantiles(self, y, var):
        quantiles = stats.norm.ppf(
            np.array(self.quantiles).reshape(1, 1, -1).repeat(
                y.shape[0], axis=0).repeat(y.shape[1], axis=1),
            loc=y[:,:,np.newaxis].repeat(len(self.quantiles), axis=2),
            scale=np.sqrt(var[:,:,np.newaxis]).repeat(
                len(self.quantiles), axis=2),)
        return quantiles

    def compute_cond_exp(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, return_path=True, get_loss=False,
                         weight=0.5, store_and_use_stored=True,
                         start_time=None,
                         **kwargs):
        if return_path and store_and_use_stored:
            if get_loss:
                if self.path_t is not None and self.loss is not None:
                    return self.loss, self.path_t, self.path_y
            else:
                if self.path_t is not None:
                    return self.loss, self.path_t, self.path_y
        elif store_and_use_stored:
            if get_loss:
                if self.loss is not None:
                    return self.loss

        LOSS = LOSS_FUN_DICT["quantile"](self.quantiles)

        y = start_X
        var = np.zeros_like(y)
        quantiles = np.expand_dims(np.zeros_like(y),axis=2).repeat(
            len(self.quantiles), axis=2)
        batch_size = start_X.shape[0]
        current_time = 0.0
        if start_time:
            current_time = start_time

        loss = 0

        if return_path:
            if start_time:
                path_t = []
                path_y = []
            else:
                path_t = [0.]
                path_y = [quantiles]

        for i, obs_time in enumerate(times):
            # Calculate conditional expectation stepwise
            while current_time < (obs_time - 1e-10*delta_t):
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time
                var += delta_t_
                quantiles = self.next_quantiles(y, var)
                current_time = current_time + delta_t_

                # Storing the conditional expectation
                if return_path:
                    path_t.append(current_time)
                    path_y.append(quantiles)

            # Reached an observation - set new interval
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]

            # Update h. Also updating loss, tau and last_X
            y[i_obs] = X_obs
            var[i_obs] = 0.
            quantiles_bj = quantiles
            quantiles = copy.deepcopy(quantiles_bj)
            quantiles[i_obs] = X_obs[:, :, np.newaxis].repeat(
                len(self.quantiles), axis=2)

            if get_loss:
                loss = loss + LOSS(
                    X_obs=torch.from_numpy(X_obs),
                    Y_obs=torch.from_numpy(quantiles[i_obs]),
                    Y_obs_bj=torch.from_numpy(quantiles_bj[i_obs]),
                    n_obs_ot=torch.from_numpy(n_obs_ot[i_obs]),
                    batch_size=batch_size, weight=weight).detach().cpu().numpy()
            if return_path:
                path_t.append(obs_time)
                path_y.append(quantiles)

        # after every observation has been processed, propagating until T
        while current_time < T - 1e-10 * delta_t:
            if current_time < T - delta_t:
                delta_t_ = delta_t
            else:
                delta_t_ = T - current_time
            var += delta_t_
            quantiles = self.next_quantiles(y, var)
            current_time = current_time + delta_t_

            # Storing the predictions.
            if return_path:
                path_t.append(current_time)
                path_y.append(quantiles)

        if get_loss and store_and_use_stored:
            self.loss = loss
        if return_path and store_and_use_stored:
            self.path_t = np.array(path_t)
            self.path_y = np.array(path_y)

        if return_path:
            # path dimension:
            #   [time_steps, batch_size, output_size, quantiles]
            return loss, np.array(path_t), np.array(path_y)
        else:
            return loss

    def generate_paths(self, start_X=None):
        spot_paths = np.zeros(
            (self.nb_paths, 1, self.nb_steps + 1))
        dt = self.dt

        random_numbers = np.random.normal(
            0, 1, (self.nb_paths, 1, self.nb_steps)) * np.sqrt(dt)
        W = np.cumsum(random_numbers, axis=2)

        spot_paths[:, 0, 1:] = W[:, 0, :]

        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


class BMandVar(StockModel):
    """
    Brownian Motion and its square
    """

    def __init__(self, nb_paths, nb_steps, maturity, dimension, **kwargs):
        super().__init__(
            drift=None, volatility=None, nb_paths=nb_paths,
            nb_steps=nb_steps, S0=np.array([0,0]), maturity=maturity,
            sine_coeff=None,)
        assert dimension == 2

    def next_cond_exp(self, y, delta_t, current_t):
        next_y = y
        next_y[:, 1] += delta_t
        return next_y

    def generate_paths(self, start_X=None):
        spot_paths = np.zeros(
            (self.nb_paths, 2, self.nb_steps + 1))
        dt = self.dt

        random_numbers = np.random.normal(
            0, 1, (self.nb_paths, 1, self.nb_steps)) * np.sqrt(dt)
        W = np.cumsum(random_numbers, axis=2)

        spot_paths[:, 0, 1:] = W[:, 0, :]
        spot_paths[:, 1, 1:] = W[:, 0, :]**2

        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


class BM2DCorr(StockModel):
    """
    2-dim Brownian Motion with correlation and correct cond. expectation
    for incomplete observations
    https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    """

    def __init__(self, nb_paths, nb_steps, maturity, alpha_sq, dimension,
                 **kwargs):
        super(BM2DCorr, self).__init__(
            drift=None, volatility=None, nb_paths=nb_paths,
            nb_steps=nb_steps, S0=np.array([0,0]), maturity=maturity,
            sine_coeff=None
        )
        self.alpha_sq = alpha_sq
        self.beta_sq = 1 - alpha_sq
        assert 0 < alpha_sq < 1, "alpha_sq needs to be in (0,1)"
        assert dimension == 2, "dimension has to be set to 2 for this dataset"
        self.path_t = None
        self.loss = None

    def next_cond_exp(self, y, delta_t, current_t):
        return y

    def get_mu(self, jj, which_coord_obs):
        sig = np.diag(self.observed_t_all_inc[jj]*3)
        M0 = np.tri(N=len(self.observed_0[jj]), k=0)[
            np.array(self.observed_0[jj])==1]
        M1 = np.tri(N=len(self.observed_1[jj]), k=0)[
            np.array(self.observed_1[jj])==1]
        r1, c1 = M0.shape
        r2, c2 = M1.shape
        M = np.zeros((r1+r2, c1*3))
        M[:r1, :c1] = np.sqrt(self.alpha_sq)*M0
        M[:r1, c1:c1*2] = np.sqrt(self.beta_sq)*M0
        M[r1:, :c1] = np.sqrt(self.alpha_sq)*M1
        M[r1:, c1*2:c1*3] = np.sqrt(self.beta_sq)*M1
        sig_bar_22_inv = np.linalg.inv(
            np.dot(np.dot(M, sig), np.transpose(M)))
        m = np.zeros((1,c1*3))
        m[0, :c1] = np.sqrt(self.alpha_sq)
        if which_coord_obs == 0:
            m[0, c1*2:c1*3] = np.sqrt(self.beta_sq)
        else:
            m[0, c1:c1*2] = np.sqrt(self.beta_sq)
        sig_bar_12 = np.dot(np.dot(m, sig), np.transpose(M))
        obs_arr = np.array(
            self.observed_X0[jj]+self.observed_X1[jj])
        mu = np.dot(np.dot(sig_bar_12, sig_bar_22_inv), obs_arr)
        return mu

    def compute_cond_exp(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, return_path=True, get_loss=False,
                         weight=0.5, store_and_use_stored=True,
                         start_time=None, M=None, which_loss="easy",
                         **kwargs):
        """
        Compute conditional expectation
        :param times: np.array, of observation times
        :param time_ptr: list, start indices of X and obs_idx for a given
                observation time, first element is 0, this pointer tells how
                many (and which) of the observations of X along the batch-dim
                belong to the current time, and obs_idx then tells to which of
                the batch elements they belong. In particular, not each batch-
                element has to jump at the same time, and only those elements
                which jump at the current time should be updated with a jump
        :param X: np.array, data tensor
        :param obs_idx: list, index of the batch elements where jumps occur at
                current time
        :param delta_t: float, time step for Euler
        :param T: float, the final time
        :param start_X: np.array, the starting point of X
        :param n_obs_ot: np.array, the number of observations over the
                entire time interval for each element of the batch
        :param return_path: bool, whether to return the path of h
        :param get_loss: bool, whether to compute the loss, otherwise 0 returned
        :param until_T: bool, whether to continue until T (for eval) or only
                until last observation (for training)
        :param M: None or np.array, if not None: the mask for the data, same
                size as X, with 0 or 1 entries
        :return: float (loss), if wanted paths of t and y (np.arrays)
        """
        if return_path and store_and_use_stored:
            if get_loss:
                if self.path_t is not None and self.loss is not None:
                    return self.loss, self.path_t, self.path_y
            else:
                if self.path_t is not None:
                    return self.loss, self.path_t, self.path_y
        elif store_and_use_stored:
            if get_loss:
                if self.loss is not None:
                    return self.loss

        bs = start_X.shape[0]
        self.observed_0 = [[] for x in range(bs)]
        self.observed_1 = [[] for x in range(bs)]
        self.observed_X0 = [[] for x in range(bs)]
        self.observed_X1 = [[] for x in range(bs)]
        self.observed_t_all = [[] for x in range(bs)]
        self.observed_t_all_inc = [[] for x in range(bs)]

        y = start_X
        batch_size = bs
        current_time = 0.0
        if start_time:
            current_time = start_time

        loss = 0

        if return_path:
            if start_time:
                path_t = []
                path_y = []
            else:
                path_t = [0.]
                path_y = [y]

        for i, obs_time in enumerate(times):
            # Calculate conditional expectation stepwise
            while current_time < (obs_time - 1e-10 * delta_t):
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time
                y = self.next_cond_exp(y, delta_t_, current_time)
                current_time = current_time + delta_t_

                # Storing the conditional expectation
                if return_path:
                    path_t.append(current_time)
                    path_y.append(y)

            # Reached an observation - set new interval
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]
            M_obs = M[start:end]

            # Update h. Also updating loss, tau and last_X
            Y_bj = y
            temp = copy.copy(y)
            for j, jj in enumerate(i_obs):
                if M_obs[j, 0] == 1:
                    self.observed_X0[jj].append(X_obs[j, 0])
                    self.observed_0[jj].append(1)
                else:
                    self.observed_0[jj].append(0)
                if M_obs[j, 1] == 1:
                    self.observed_X1[jj].append(X_obs[j, 1])
                    self.observed_1[jj].append(1)
                else:
                    self.observed_1[jj].append(0)
                assert M_obs[j, :].sum() > 0
                self.observed_t_all[jj].append(obs_time)
                last = [0.] + self.observed_t_all[jj]
                self.observed_t_all_inc[jj].append(obs_time - last[-2])

                if M_obs[j, 0] == 1 and M_obs[j, 1] == 1:
                    temp[jj, 0] = X_obs[j, 0]
                    temp[jj, 1] = X_obs[j, 1]
                else:

                    if M_obs[j, 0] == 1:
                        temp[jj, 0] = X_obs[j, 0]
                        temp[jj, 1] = self.get_mu(jj=jj, which_coord_obs=0)
                    else:
                        temp[jj, 1] = X_obs[j, 1]
                        temp[jj, 0] = self.get_mu(jj=jj, which_coord_obs=1)
            y = temp
            Y = y

            if get_loss:
                loss = loss + compute_loss(
                    X_obs=X_obs, Y_obs=Y[i_obs], Y_obs_bj=Y_bj[i_obs],
                    n_obs_ot=n_obs_ot[i_obs], batch_size=batch_size,
                    weight=weight, M_obs=M_obs, which_loss=which_loss)
            if return_path:
                path_t.append(obs_time)
                path_y.append(y)

        # after every observation has been processed, propagating until T
        while current_time < T - 1e-10 * delta_t:
            if current_time < T - delta_t:
                delta_t_ = delta_t
            else:
                delta_t_ = T - current_time
            y = self.next_cond_exp(y, delta_t_)
            current_time = current_time + delta_t_

            # Storing the predictions.
            if return_path:
                path_t.append(current_time)
                path_y.append(y)

        if get_loss and store_and_use_stored:
            self.loss = loss
        if return_path and store_and_use_stored:
            self.path_t = np.array(path_t)
            self.path_y = np.array(path_y)

        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            return loss, np.array(path_t), np.array(path_y)
        else:
            return loss

    def generate_paths(self):
        spot_paths = np.zeros(
            (self.nb_paths, 2, self.nb_steps + 1))
        dt = self.dt

        random_numbers = np.random.normal(
            0, 1, (self.nb_paths, 3, self.nb_steps)) * np.sqrt(dt)
        W = np.cumsum(random_numbers, axis=2)


        spot_paths[:, 0, 1:] = W[:, 0, :] * np.sqrt(self.alpha_sq) + \
                               W[:, 1, :] * np.sqrt(self.beta_sq)
        spot_paths[:, 1, 1:] = W[:, 0, :] * np.sqrt(self.alpha_sq) + \
                               W[:, 2, :] * np.sqrt(self.beta_sq)

        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


class BMFiltering(BM2DCorr):
    """
    A Brownian Motion filtering example. The Signal process is a BM $X$ and the
    observation process is given by
        $ Y = \\alpha X + W, $
    where $W$ is also a BM independent of $X$ and $\\alpha \\in \\R$.
    $ Z = (Y, X) $, i.e. the first coordinate is $Y$ and is always observed.
    expectation for incomplete observations
    https://en.wikipedia.org/wiki/Multivariate_normal_distribution

    optionally, in the IO_version, Y is the input and X the output coordinate
    """

    def __init__(self, nb_paths, nb_steps, maturity, alpha, dimension,
                 IO_version=False, **kwargs):
        super(BM2DCorr, self).__init__(
            drift=None, volatility=None, nb_paths=nb_paths,
            nb_steps=nb_steps, S0=np.array([0,0]), maturity=maturity,
            sine_coeff=None
        )
        self.alpha = alpha
        assert dimension == 2, "dimension has to be set to 2 for this dataset"
        self.path_t = None
        self.loss = None
        self.IO_version = IO_version
        if self.IO_version:
            self.output_coords = [1]
        else:
            self.output_coords = [0, 1]

    def get_mu(self, jj, which_coord_obs):
        if self.IO_version:
            N = len(self.observed_t_all[jj])
            sig_11 = np.zeros((N,N))
            for i in range(N):
                for j in range(N):
                    sig_11[i,j] = ((self.alpha**2+1) *
                                   self.observed_t_all[jj][min(i,j)])
            sig_21 = self.alpha * np.array(
                self.observed_t_all[jj]).reshape((1,-1))
            sig_22 = self.observed_t_all[jj][-1]
            sig_11_inv = np.linalg.inv(sig_11)
            mu = np.dot(sig_21, np.dot(
                sig_11_inv, np.array(self.observed_X0[jj])))
            return mu

        sig = np.diag(self.observed_t_all_inc[jj]*2)
        M0 = np.tri(N=len(self.observed_0[jj]), k=0)[
            np.array(self.observed_0[jj])==1]
        M1 = np.tri(N=len(self.observed_1[jj]), k=0)[
            np.array(self.observed_1[jj])==1]
        r1, c1 = M0.shape
        r2, c2 = M1.shape
        M = np.zeros((r1+r2, c1*2))
        M[:r1, :c1] = self.alpha*M0
        M[:r1, c1:c1*2] = M0
        M[r1:, :c1] = M1
        sig_bar_22_inv = np.linalg.inv(
            np.dot(np.dot(M, sig), np.transpose(M)))
        m = np.zeros((1,c1*2))
        m[0, :c1] = 1
        sig_bar_12 = np.dot(np.dot(m, sig), np.transpose(M))
        obs_arr = np.array(
            self.observed_X0[jj]+self.observed_X1[jj])
        mu = np.dot(np.dot(sig_bar_12, sig_bar_22_inv), obs_arr)
        return mu

    def compute_cond_exp(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, return_path=True, get_loss=False,
                         weight=0.5, store_and_use_stored=True,
                         start_time=None, M=None, which_loss="easy",
                         **kwargs):
        """
        Compute conditional expectation
        :param times: np.array, of observation times
        :param time_ptr: list, start indices of X and obs_idx for a given
                observation time, first element is 0, this pointer tells how
                many (and which) of the observations of X along the batch-dim
                belong to the current time, and obs_idx then tells to which of
                the batch elements they belong. In particular, not each batch-
                element has to jump at the same time, and only those elements
                which jump at the current time should be updated with a jump
        :param X: np.array, data tensor
        :param obs_idx: list, index of the batch elements where jumps occur at
                current time
        :param delta_t: float, time step for Euler
        :param T: float, the final time
        :param start_X: np.array, the starting point of X
        :param n_obs_ot: np.array, the number of observations over the
                entire time interval for each element of the batch
        :param return_path: bool, whether to return the path of h
        :param get_loss: bool, whether to compute the loss, otherwise 0 returned
        :param until_T: bool, whether to continue until T (for eval) or only
                until last observation (for training)
        :param M: None or np.array, if not None: the mask for the data, same
                size as X, with 0 or 1 entries
        :return: float (loss), if wanted paths of t and y (np.arrays)
        """
        if return_path and store_and_use_stored:
            if get_loss:
                if self.path_t is not None and self.loss is not None:
                    return self.loss, self.path_t, self.path_y
            else:
                if self.path_t is not None:
                    return self.loss, self.path_t, self.path_y
        elif store_and_use_stored:
            if get_loss:
                if self.loss is not None:
                    return self.loss

        # in case of the IO version
        if M is None:
            M = np.ones_like(X)

        bs = start_X.shape[0]
        self.observed_0 = [[] for x in range(bs)]
        self.observed_1 = [[] for x in range(bs)]
        self.observed_X0 = [[] for x in range(bs)]
        self.observed_X1 = [[] for x in range(bs)]
        self.observed_t_all = [[] for x in range(bs)]
        self.observed_t_all_inc = [[] for x in range(bs)]

        y = start_X
        batch_size = bs
        current_time = 0.0
        if start_time:
            current_time = start_time

        loss = 0

        if return_path:
            if start_time:
                path_t = []
                path_y = []
            else:
                path_t = [0.]
                path_y = [y]

        for i, obs_time in enumerate(times):
            # Calculate conditional expectation stepwise
            while current_time < (obs_time - 1e-10 * delta_t):
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time
                y = self.next_cond_exp(y, delta_t_, current_time)
                current_time = current_time + delta_t_

                # Storing the conditional expectation
                if return_path:
                    path_t.append(current_time)
                    path_y.append(y)

            # Reached an observation - set new interval
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]
            M_obs = M[start:end]

            # Update h. Also updating loss, tau and last_X
            Y_bj = y
            temp = copy.deepcopy(y)
            for j, jj in enumerate(i_obs):
                if M_obs[j, 0] == 1:
                    self.observed_X0[jj].append(X_obs[j, 0])
                    self.observed_0[jj].append(1)
                else:
                    self.observed_0[jj].append(0)
                if M_obs[j, 1] == 1 and not self.IO_version:
                    self.observed_X1[jj].append(X_obs[j, 1])
                    self.observed_1[jj].append(1)
                else:
                    self.observed_1[jj].append(0)
                assert M_obs[j, :].sum() > 0
                self.observed_t_all[jj].append(obs_time)
                last = [0.] + self.observed_t_all[jj]
                self.observed_t_all_inc[jj].append(obs_time - last[-2])

                if (M_obs[j, 0] == 1 and M_obs[j, 1] == 1
                        and not self.IO_version):
                    temp[jj, 0] = X_obs[j, 0]
                    temp[jj, 1] = X_obs[j, 1]
                else:
                    assert M_obs[j, 0] == 1
                    temp[jj, 0] = X_obs[j, 0]
                    temp[jj, 1] = self.get_mu(jj=jj, which_coord_obs=0)
            y = temp
            Y = y

            if get_loss:
                loss = loss + compute_loss(
                    X_obs=X_obs[:, self.output_coords],
                    Y_obs=Y[i_obs][:, self.output_coords],
                    Y_obs_bj=Y_bj[i_obs][:, self.output_coords],
                    n_obs_ot=n_obs_ot[i_obs], batch_size=batch_size,
                    weight=weight, M_obs=M_obs[:, self.output_coords],
                    which_loss=which_loss)
            if return_path:
                path_t.append(obs_time)
                path_y.append(y)

        # after every observation has been processed, propagating until T
        while current_time < T - 1e-10 * delta_t:
            if current_time < T - delta_t:
                delta_t_ = delta_t
            else:
                delta_t_ = T - current_time
            y = self.next_cond_exp(y, delta_t_)
            current_time = current_time + delta_t_

            # Storing the predictions.
            if return_path:
                path_t.append(current_time)
                path_y.append(y)

        if get_loss and store_and_use_stored:
            self.loss = loss
        if return_path and store_and_use_stored:
            self.path_t = np.array(path_t)
            self.path_y = np.array(path_y)[:, :, self.output_coords]

        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            return (loss, np.array(path_t),
                    np.array(path_y)[:, :, self.output_coords])
        else:
            return loss

    def generate_paths(self):
        spot_paths = np.zeros(
            (self.nb_paths, 2, self.nb_steps + 1))
        dt = self.dt

        random_numbers = np.random.normal(
            0, 1, (self.nb_paths, 2, self.nb_steps)) * np.sqrt(dt)
        W = np.cumsum(random_numbers, axis=2)

        spot_paths[:, 0, 1:] = W[:, 0, :] * self.alpha + W[:, 1, :]
        spot_paths[:, 1, 1:] = W[:, 0, :]

        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


class BMClassification(BM2DCorr):
    """
    A Brownian Motion filtering example. The Signal process is a BM $X$ and the
    observation process is given by
        $ Y = \\alpha X + W, $
    where $W$ is also a BM independent of $X$ and $\\alpha \\in \\R$.
    $ Z = (Y, X) $, i.e. the first coordinate is $Y$ and is always observed.
    expectation for incomplete observations
    https://en.wikipedia.org/wiki/Multivariate_normal_distribution

    optionally, in the IO_version, Y is the input and X the output coordinate
    """

    def __init__(self, nb_paths, nb_steps, maturity, alpha, dimension,
                 **kwargs):
        super(BM2DCorr, self).__init__(
            drift=None, volatility=None, nb_paths=nb_paths,
            nb_steps=nb_steps, S0=np.array([0,0]), maturity=maturity,
            sine_coeff=None
        )
        self.alpha = alpha
        assert dimension == 2, "dimension has to be set to 2 for this dataset"
        self.path_t = None
        self.loss = None

    def curr_cond_exp(self, y, delta_t, current_t, last_obs, last_obs_time):
        return 1 - scipy.stats.norm.cdf(
            (self.alpha - last_obs)/np.sqrt(current_t+delta_t-last_obs_time))

    def compute_cond_exp(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, return_path=True, get_loss=False,
                         weight=0.5, store_and_use_stored=True,
                         start_time=None, M=None, which_loss="easy",
                         **kwargs):
        """
        Compute conditional expectation
        :param times: np.array, of observation times
        :param time_ptr: list, start indices of X and obs_idx for a given
                observation time, first element is 0, this pointer tells how
                many (and which) of the observations of X along the batch-dim
                belong to the current time, and obs_idx then tells to which of
                the batch elements they belong. In particular, not each batch-
                element has to jump at the same time, and only those elements
                which jump at the current time should be updated with a jump
        :param X: np.array, data tensor
        :param obs_idx: list, index of the batch elements where jumps occur at
                current time
        :param delta_t: float, time step for Euler
        :param T: float, the final time
        :param start_X: np.array, the starting point of X
        :param n_obs_ot: np.array, the number of observations over the
                entire time interval for each element of the batch
        :param return_path: bool, whether to return the path of h
        :param get_loss: bool, whether to compute the loss, otherwise 0 returned
        :param until_T: bool, whether to continue until T (for eval) or only
                until last observation (for training)
        :param M: None or np.array, if not None: the mask for the data, same
                size as X, with 0 or 1 entries
        :return: float (loss), if wanted paths of t and y (np.arrays)
        """
        if return_path and store_and_use_stored:
            if get_loss:
                if self.path_t is not None and self.loss is not None:
                    return self.loss, self.path_t, self.path_y
            else:
                if self.path_t is not None:
                    return self.loss, self.path_t, self.path_y
        elif store_and_use_stored:
            if get_loss:
                if self.loss is not None:
                    return self.loss

        # in case of the IO version
        if M is None:
            M = np.ones_like(X)

        bs = start_X.shape[0]
        last_obs = start_X[:, 0]
        last_obs_time = np.zeros(bs)

        y = (start_X[:, 0] > self.alpha)*1.
        y[start_X[:, 0] == self.alpha] = 0.5
        batch_size = bs
        current_time = 0.0
        if start_time:
            current_time = start_time

        loss = 0

        if return_path:
            if start_time:
                path_t = []
                path_y = []
            else:
                path_t = [0.]
                path_y = [y]

        for i, obs_time in enumerate(times):
            # Calculate conditional expectation stepwise
            while current_time < (obs_time - 1e-10 * delta_t):
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time
                y = self.curr_cond_exp(
                    y, delta_t_, current_time, last_obs=last_obs,
                    last_obs_time=last_obs_time)
                current_time = current_time + delta_t_

                # Storing the conditional expectation
                if return_path:
                    path_t.append(current_time)
                    path_y.append(y)

            # Reached an observation - set new interval
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]
            M_obs = M[start:end]

            # Update h. Also updating loss, tau and last_X
            Y_bj = y
            temp = copy.deepcopy(y)
            last_obs[i_obs] = X_obs[:, 0]
            last_obs_time[i_obs] = obs_time
            temp[i_obs] = (X_obs[:, 0] >= self.alpha)*1.
            y = temp
            Y = y

            if get_loss:
                loss = loss + compute_loss(
                    X_obs=X_obs[:, 1:2],
                    Y_obs=Y[i_obs].reshape(-1, 1),
                    Y_obs_bj=Y_bj[i_obs].reshape(-1, 1),
                    n_obs_ot=n_obs_ot[i_obs], batch_size=batch_size,
                    weight=weight, M_obs=M_obs[:, 1:2],
                    which_loss=which_loss)
            if return_path:
                path_t.append(obs_time)
                path_y.append(y)

        # after every observation has been processed, propagating until T
        while current_time < T - 1e-10 * delta_t:
            if current_time < T - delta_t:
                delta_t_ = delta_t
            else:
                delta_t_ = T - current_time
            y = self.curr_cond_exp(
                y, delta_t_, current_time, last_obs, last_obs_time)
            current_time = current_time + delta_t_

            # Storing the predictions.
            if return_path:
                path_t.append(current_time)
                path_y.append(y)

        if get_loss and store_and_use_stored:
            self.loss = loss
        if return_path and store_and_use_stored:
            self.path_t = np.array(path_t)
            self.path_y = np.expand_dims(np.array(path_y), axis=2)

        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            return (loss, np.array(path_t),
                    np.expand_dims(np.array(path_y), axis=2))
        else:
            return loss

    def generate_paths(self):
        spot_paths = np.zeros(
            (self.nb_paths, 2, self.nb_steps + 1))
        dt = self.dt

        random_numbers = np.random.normal(
            0, 1, (self.nb_paths, self.nb_steps)) * np.sqrt(dt)
        W = np.cumsum(random_numbers, axis=1)

        spot_paths[:, 0, 1:] = W[:, :]
        spot_paths[:, 1, :] = (spot_paths[:, 0, :] >= self.alpha)*1.

        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


class BMwithTimeLag(BM2DCorr):
    """
    A Brownian Motion and its time-lagged version. In particular, the first
    coordinate is a BM $X$ and the second coordinate is $Y_t = X_{t - \alpha}$
    for some $\alpha > 0$.

    For computing conditional expectations, the first coordinate is assumed to
    be always observed (otherwise not working correctly). However, this is
    therefore only needed on the test set (where cond. exp. are computed) but
    not on the training set (where they are not computed).
    """

    def __init__(self, nb_paths, nb_steps, maturity, alpha_in_dt_steps,
                 dimension, **kwargs):
        """
        Args:
            nb_paths:
            nb_steps:
            maturity:
            alpha_in_dt_steps: int, this defines via
                alpha = dt * alpha_in_dt_steps
            dimension:
            **kwargs:
        """
        super(BM2DCorr, self).__init__(
            drift=None, volatility=None, nb_paths=nb_paths,
            nb_steps=nb_steps, S0=np.array([0,0]), maturity=maturity,
            sine_coeff=None
        )
        self.alpha_in_dt_steps = alpha_in_dt_steps
        self.dt = self.maturity / self.nb_steps
        self.alpha = self.dt * self.alpha_in_dt_steps
        assert dimension == 2, "dimension has to be set to 2 for this dataset"
        self.path_t = None
        self.loss = None

    def next_cond_exp(self, y, delta_t, current_t):
        t = current_t + delta_t
        s = t - self.alpha
        next_y = copy.deepcopy(y)
        for j in range(y.shape[0]):
            next_y[j, 1] = self.get_next_y1(j, y[j, 1], t, s)
        return next_y

    def get_next_y1(self, j, y1, t, s):
        if len(self.observed_t_0[j]) == 0:
            return 0.
        last_obs_time = np.max(self.observed_t_0[j])
        if t <= self.alpha:
            next_y1 = 0.
        elif t >= last_obs_time + self.alpha:
            next_y1 = y1
        else:
            # in this case: s < last_obs_time
            i = np.argmax((np.array(self.observed_t_0[j]) - s) > 0)
            t_obs_after = self.observed_t_0[j][i]
            X0_obs_after = self.observed_X0[j][i]
            if i==0:
                t_obs_before = 0.
                X0_obs_before = 0.
            else:
                t_obs_before = self.observed_t_0[j][i-1]
                X0_obs_before = self.observed_X0[j][i-1]
            if len(self.observed_t_1[j])>0 and \
                    self.observed_t_1[j][-1] - self.alpha > t_obs_before:
                t_obs_before = self.observed_t_1[j][-1] - self.alpha
                X0_obs_before = self.observed_X1[j][-1]
            w = (s - t_obs_before)/(t_obs_after - t_obs_before)
            next_y1 = (1-w)*X0_obs_before + w*X0_obs_after
        return next_y1

    def compute_cond_exp(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, return_path=True, get_loss=False,
                         weight=0.5, store_and_use_stored=True,
                         start_time=None, M=None, which_loss="easy",
                         **kwargs):
        """
        Compute conditional expectation
        :param times: np.array, of observation times
        :param time_ptr: list, start indices of X and obs_idx for a given
                observation time, first element is 0, this pointer tells how
                many (and which) of the observations of X along the batch-dim
                belong to the current time, and obs_idx then tells to which of
                the batch elements they belong. In particular, not each batch-
                element has to jump at the same time, and only those elements
                which jump at the current time should be updated with a jump
        :param X: np.array, data tensor
        :param obs_idx: list, index of the batch elements where jumps occur at
                current time
        :param delta_t: float, time step for Euler
        :param T: float, the final time
        :param start_X: np.array, the starting point of X
        :param n_obs_ot: np.array, the number of observations over the
                entire time interval for each element of the batch
        :param return_path: bool, whether to return the path of h
        :param get_loss: bool, whether to compute the loss, otherwise 0 returned
        :param until_T: bool, whether to continue until T (for eval) or only
                until last observation (for training)
        :param M: None or np.array, if not None: the mask for the data, same
                size as X, with 0 or 1 entries
        :return: float (loss), if wanted paths of t and y (np.arrays)
        """
        if return_path and store_and_use_stored:
            if get_loss:
                if self.path_t is not None and self.loss is not None:
                    return self.loss, self.path_t, self.path_y
            else:
                if self.path_t is not None:
                    return self.loss, self.path_t, self.path_y
        elif store_and_use_stored:
            if get_loss:
                if self.loss is not None:
                    return self.loss

        bs = start_X.shape[0]
        self.observed_0 = [[] for x in range(bs)]
        self.observed_1 = [[] for x in range(bs)]
        self.observed_X0 = [[] for x in range(bs)]
        self.observed_X1 = [[] for x in range(bs)]
        self.observed_t_0 = [[] for x in range(bs)]
        self.observed_t_1 = [[] for x in range(bs)]

        y = start_X
        batch_size = bs
        current_time = 0.0
        if start_time:
            current_time = start_time

        loss = 0

        if return_path:
            if start_time:
                path_t = []
                path_y = []
            else:
                path_t = [0.]
                path_y = [y]

        for i, obs_time in enumerate(times):
            # Calculate conditional expectation stepwise
            while current_time < (obs_time - 1e-10 * delta_t):
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time
                y = self.next_cond_exp(y, delta_t_, current_time)
                current_time = current_time + delta_t_

                # Storing the conditional expectation
                if return_path:
                    path_t.append(current_time)
                    path_y.append(y)

            # Reached an observation - set new interval
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]
            M_obs = M[start:end]

            # Update h. Also updating loss, tau and last_X
            Y_bj = y
            temp = copy.copy(y)
            for j, jj in enumerate(i_obs):
                if M_obs[j, 0] == 1:
                    self.observed_X0[jj].append(X_obs[j, 0])
                    self.observed_0[jj].append(1)
                    self.observed_t_0[jj].append(obs_time)
                else:
                    self.observed_0[jj].append(0)
                if M_obs[j, 1] == 1:
                    self.observed_X1[jj].append(X_obs[j, 1])
                    self.observed_1[jj].append(1)
                    self.observed_t_1[jj].append(obs_time)
                else:
                    self.observed_1[jj].append(0)
                assert M_obs[j, :].sum() > 0

                if M_obs[j, 0] == 1 and M_obs[j, 1] == 1:
                    temp[jj, 0] = X_obs[j, 0]
                    temp[jj, 1] = X_obs[j, 1]
                elif M_obs[j, 0] == 1:
                    temp[jj, 0] = X_obs[j, 0]
                    temp[jj, 1] = self.get_next_y1(
                        jj, y[jj, 1], obs_time, obs_time-self.alpha)
                else:
                    temp[jj, 1] = X_obs[j, 1]

            y = temp
            Y = y

            if get_loss:
                loss = loss + compute_loss(
                    X_obs=X_obs, Y_obs=Y[i_obs], Y_obs_bj=Y_bj[i_obs],
                    n_obs_ot=n_obs_ot[i_obs], batch_size=batch_size,
                    weight=weight, M_obs=M_obs, which_loss=which_loss)
            if return_path:
                path_t.append(obs_time)
                path_y.append(y)

        # after every observation has been processed, propagating until T
        while current_time < T - 1e-10 * delta_t:
            if current_time < T - delta_t:
                delta_t_ = delta_t
            else:
                delta_t_ = T - current_time
            y = self.next_cond_exp(y, delta_t_)
            current_time = current_time + delta_t_

            # Storing the predictions.
            if return_path:
                path_t.append(current_time)
                path_y.append(y)

        if get_loss and store_and_use_stored:
            self.loss = loss
        if return_path and store_and_use_stored:
            self.path_t = np.array(path_t)
            self.path_y = np.array(path_y)

        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            return loss, np.array(path_t), np.array(path_y)
        else:
            return loss

    def generate_paths(self):
        spot_paths = np.zeros(
            (self.nb_paths, 2, self.nb_steps + 1))
        dt = self.dt

        random_numbers = np.random.normal(
            0, 1, (self.nb_paths, 1, self.nb_steps)) * np.sqrt(dt)
        W = np.cumsum(random_numbers, axis=2)

        spot_paths[:, 0, 1:] = W[:, 0, :]
        spot_paths[:, 1, 1+self.alpha_in_dt_steps:] = \
            W[:, 0, :-self.alpha_in_dt_steps]

        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


class BMNoisyObs(BM):
    """
    A Brownian Motion with noisy observations
    """

    def __init__(self, nb_paths, nb_steps, maturity, obs_noise, dimension,
                 **kwargs):
        super().__init__(
            nb_paths=nb_paths, nb_steps=nb_steps, maturity=maturity,
            dimension=dimension,)
        assert dimension == 1, "dimension has to be set to 1 for this dataset"
        self.path_t = None
        self.loss = None
        self.noise_sig = obs_noise["scale"]

    def compute_cond_exp(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, return_path=True, get_loss=False,
                         weight=0.5, store_and_use_stored=True,
                         start_time=None, M=None, which_loss="easy",
                         **kwargs):
        """
        Compute conditional expectation
        :param times: np.array, of observation times
        :param time_ptr: list, start indices of X and obs_idx for a given
                observation time, first element is 0, this pointer tells how
                many (and which) of the observations of X along the batch-dim
                belong to the current time, and obs_idx then tells to which of
                the batch elements they belong. In particular, not each batch-
                element has to jump at the same time, and only those elements
                which jump at the current time should be updated with a jump
        :param X: np.array, data tensor
        :param obs_idx: list, index of the batch elements where jumps occur at
                current time
        :param delta_t: float, time step for Euler
        :param T: float, the final time
        :param start_X: np.array, the starting point of X
        :param n_obs_ot: np.array, the number of observations over the
                entire time interval for each element of the batch
        :param return_path: bool, whether to return the path of h
        :param get_loss: bool, whether to compute the loss, otherwise 0 returned
        :param until_T: bool, whether to continue until T (for eval) or only
                until last observation (for training)
        :param M: None or np.array, if not None: the mask for the data, same
                size as X, with 0 or 1 entries
        :return: float (loss), if wanted paths of t and y (np.arrays)
        """
        if return_path and store_and_use_stored:
            if get_loss:
                if self.path_t is not None and self.loss is not None:
                    return self.loss, self.path_t, self.path_y
            else:
                if self.path_t is not None:
                    return self.loss, self.path_t, self.path_y
        elif store_and_use_stored:
            if get_loss:
                if self.loss is not None:
                    return self.loss

        bs = start_X.shape[0]
        self.observations = [[] for x in range(bs)]
        self.observed_t_all = [[] for x in range(bs)]

        y = start_X
        batch_size = bs
        current_time = 0.0
        if start_time:
            current_time = start_time

        loss = 0

        if return_path:
            if start_time:
                path_t = []
                path_y = []
            else:
                path_t = [0.]
                path_y = [y]

        for i, obs_time in enumerate(times):
            # Calculate conditional expectation stepwise
            while current_time < (obs_time - 1e-10 * delta_t):
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time
                y = self.next_cond_exp(y, delta_t_, current_time)
                current_time = current_time + delta_t_

                # Storing the conditional expectation
                if return_path:
                    path_t.append(current_time)
                    path_y.append(y)

            # Reached an observation - set new interval
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]

            # Update h. Also updating loss, tau and last_X
            Y_bj = y
            temp = copy.copy(y)
            for j, jj in enumerate(i_obs):
                self.observations[jj].append(X_obs[j])
                self.observed_t_all[jj].append(obs_time)
                size = len(self.observed_t_all[jj])
                Sig = np.zeros((size, size))
                for k in range(size):
                    tmin = self.observed_t_all[jj][k]
                    Sig[k, k+1:] = tmin
                    Sig[k+1:, k] = tmin
                    Sig[k, k] = tmin + self.noise_sig**2
                sigvec = np.array(self.observed_t_all[jj])
                Sig_inv = np.linalg.inv(Sig)
                obs = np.array(self.observations[jj])
                mu = np.dot(np.dot(sigvec, Sig_inv), obs)
                temp[jj] = mu
            y = temp
            Y = y

            if get_loss:
                loss = loss + compute_loss(
                    X_obs=X_obs, Y_obs=Y[i_obs], Y_obs_bj=Y_bj[i_obs],
                    n_obs_ot=n_obs_ot[i_obs], batch_size=batch_size,
                    weight=weight, M_obs=None, which_loss=which_loss)
            if return_path:
                path_t.append(obs_time)
                path_y.append(y)

        # after every observation has been processed, propagating until T
        while current_time < T - 1e-10 * delta_t:
            if current_time < T - delta_t:
                delta_t_ = delta_t
            else:
                delta_t_ = T - current_time
            y = self.next_cond_exp(y, delta_t_)
            current_time = current_time + delta_t_

            # Storing the predictions.
            if return_path:
                path_t.append(current_time)
                path_y.append(y)

        if get_loss and store_and_use_stored:
            self.loss = loss
        if return_path and store_and_use_stored:
            self.path_t = np.array(path_t)
            self.path_y = np.array(path_y)

        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            return loss, np.array(path_t), np.array(path_y)
        else:
            return loss


class BMwithUncertainDrift(BM2DCorr):
    """
    A Brownian Motion with uncertain drift and known volatility:
        dX_t = mu dt + sigma dW_t  <=> X_t = X_0 + mu t + sigma W_t
        mu ~ N(mu_mean, mu_std)
    where the goal is to estimate the drift mu from observations of X.
    """

    def __init__(self, nb_paths, nb_steps, maturity, volatility,
                 drift_mean, drift_std, S0,
                 **kwargs):
        super(BM2DCorr, self).__init__(
            drift=None, volatility=volatility, nb_paths=nb_paths,
            nb_steps=nb_steps, S0=np.array([S0]), maturity=maturity,
            sine_coeff=None)
        self.drift_mean = drift_mean
        self.drift_std = drift_std
        self.path_t = None
        self.path_var_y = None
        self.loss = None
        self.observed_X = None
        self.observed_t = None
        self.observed_t_inc = None
        self.loss_comp_for_pow2_implemented = True

    def get_mu(self, jj, **kwargs):
        sig = np.diag(self.observed_t_inc[jj]+[self.drift_std**2])
        M0 = np.tri(N=len(self.observed_t_inc[jj]), k=0)
        r1, c1 = M0.shape
        M = np.zeros((r1+1, c1+1))
        M[:r1, :c1] = self.volatility*M0
        M[:r1, c1] = self.observed_t[jj]
        M[r1, c1] = 1.

        sig_bar = M @ sig @ M.T
        sig_bar_11_inv = np.linalg.inv(sig_bar[:r1, :c1])
        sig_bar_21 = sig_bar[r1:, :c1]
        sig_bar_22 = sig_bar[r1, c1]
        sig_bar_12 = sig_bar[:r1, c1:]
        mu = self.drift_mean + sig_bar_21 @ sig_bar_11_inv @ (
            np.array(self.observed_X[jj]) -
            self.drift_mean*np.array(self.observed_t[jj]))
        Sig = sig_bar_22 - sig_bar_21 @ sig_bar_11_inv @ sig_bar_12
        return mu, Sig

    def compute_cond_exp(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, return_path=True, get_loss=False,
                         weight=0.5, store_and_use_stored=True,
                         start_time=None, M=None, return_var=False,
                         which_loss="easy",
                         **kwargs):
        """
        Compute conditional expectation
        :param times: np.array, of observation times
        :param time_ptr: list, start indices of X and obs_idx for a given
                observation time, first element is 0, this pointer tells how
                many (and which) of the observations of X along the batch-dim
                belong to the current time, and obs_idx then tells to which of
                the batch elements they belong. In particular, not each batch-
                element has to jump at the same time, and only those elements
                which jump at the current time should be updated with a jump
        :param X: np.array, data tensor
        :param obs_idx: list, index of the batch elements where jumps occur at
                current time
        :param delta_t: float, time step for Euler
        :param T: float, the final time
        :param start_X: np.array, the starting point of X
        :param n_obs_ot: np.array, the number of observations over the
                entire time interval for each element of the batch
        :param return_path: bool, whether to return the path of h
        :param get_loss: bool, whether to compute the loss, otherwise 0 returned
        :param until_T: bool, whether to continue until T (for eval) or only
                until last observation (for training)
        :param M: None or np.array, if not None: the mask for the data, same
                size as X, with 0 or 1 entries
        :param return_var: bool, whether to return the variance of the
                conditional expectation
        :return: float (loss), if wanted paths of t and y (np.arrays)
        """
        if return_path and store_and_use_stored:
            res = [self.loss, self.path_t, self.path_y]
            if return_var:
                res.append(self.path_var_y)
            if get_loss:
                if self.path_t is not None and self.loss is not None:
                    return res
            else:
                if self.path_t is not None:
                    return res
        elif store_and_use_stored:
            if get_loss:
                if self.loss is not None:
                    return self.loss

        bs = start_X.shape[0]
        self.observed_X = [[] for x in range(bs)]
        self.observed_t = [[] for x in range(bs)]
        self.observed_t_inc = [[] for x in range(bs)]

        y = np.ones((bs,1))*self.drift_mean
        var_y = np.ones((bs,1))*self.drift_std**2
        loss = 0
        batch_size = bs
        current_time = 0.0
        if start_time:
            current_time = start_time
        if return_path:
            if start_time:
                path_t = []
                path_y = []
                path_var_y = []
            else:
                path_t = [0.]
                path_y = [y]
                path_var_y = [var_y]

        for i, obs_time in enumerate(times):
            # Calculate conditional expectation stepwise
            while current_time < (obs_time - 1e-10 * delta_t):
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time
                y = self.next_cond_exp(y, delta_t_, current_time)
                var_y = self.next_cond_exp(var_y, delta_t_, current_time)
                current_time = current_time + delta_t_

                # Storing the conditional expectation
                if return_path:
                    path_t.append(current_time)
                    path_y.append(y)
                    path_var_y.append(var_y)

            # Reached an observation - set new interval
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]

            # Update h. Also updating loss, tau and last_X
            Y_bj = y
            Var_y_bj = var_y
            temp = copy.copy(y)
            temp_var = copy.copy(var_y)
            for j, jj in enumerate(i_obs):
                self.observed_X[jj].append(X_obs[j, 0])
                self.observed_t[jj].append(obs_time)
                last = self.observed_t[jj][-2] if len(
                    self.observed_t[jj]) > 1 else 0
                self.observed_t_inc[jj].append(obs_time - last)
                mu, Sig = self.get_mu(jj=jj)
                temp[jj] = mu
                temp_var[jj] = Sig
            y = temp
            var_y = temp_var
            Y = y

            if get_loss:
                if return_var:
                    Y_bj = np.concatenate([Y_bj, Y_bj**2+Var_y_bj], axis=1)
                    Y = np.concatenate([Y, Y**2+var_y], axis=1)
                    X_obs = X_obs[:, [1,3]]
                else:
                    X_obs = X_obs[:, [1]]
                # print("X_obs", X_obs.shape)
                # print("Y", Y[i_obs].shape)
                # print("Y_bj", Y_bj[i_obs].shape)
                # print("X_obs", X_obs[0])
                # print("Y", Y[i_obs][0])
                # print("Y_bj", Y_bj[i_obs][0])
                loss = loss + compute_loss(
                    X_obs=X_obs, Y_obs=Y[i_obs], Y_obs_bj=Y_bj[i_obs],
                    n_obs_ot=n_obs_ot[i_obs], batch_size=batch_size,
                    weight=weight, M_obs=None, which_loss=which_loss)
            if return_path:
                path_t.append(obs_time)
                path_y.append(y)
                path_var_y.append(var_y)

        # after every observation has been processed, propagating until T
        while current_time < T - 1e-10 * delta_t:
            if current_time < T - delta_t:
                delta_t_ = delta_t
            else:
                delta_t_ = T - current_time
            y = self.next_cond_exp(y, delta_t_)
            var_y = self.next_cond_exp(var_y, delta_t_)
            current_time = current_time + delta_t_

            # Storing the predictions.
            if return_path:
                path_t.append(current_time)
                path_y.append(y)
                path_var_y.append(var_y)

        if get_loss and store_and_use_stored:
            self.loss = loss
        if return_path and store_and_use_stored:
            self.path_t = np.array(path_t)
            self.path_y = np.array(path_y)
            self.path_var_y = np.array(path_var_y)

        if return_path:
            if return_var:
                return (loss, np.array(path_t), np.array(path_y),
                        np.array(path_var_y))
            # path dimension: [time_steps, batch_size, output_size]
            return loss, np.array(path_t), np.array(path_y)
        return loss

    def generate_paths(self):
        spot_paths = np.zeros(
            (self.nb_paths, self.dimensions * 2, self.nb_steps + 1))
        dt = self.dt
        mu = np.random.normal(
            self.drift_mean, self.drift_std, (self.nb_paths, self.dimensions))
        spot_paths[:, self.dimensions:, :] = mu.reshape(
            (self.nb_paths, self.dimensions, 1)).repeat(self.nb_steps+1, axis=2)

        random_numbers = np.random.normal(
            0, 1, (self.nb_paths, self.dimensions, self.nb_steps)) * np.sqrt(dt)
        W = np.cumsum(random_numbers, axis=2)
        drift = np.cumsum(spot_paths[:, self.dimensions:, 1:] * dt, axis=2)

        spot_paths[:, :self.dimensions, 1:] = drift + self.volatility*W

        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


class FracBM(StockModel):
    """
    Implementing FBM via FBM package
    """

    def __init__(self, nb_paths, nb_steps, S0, maturity, hurst,
                 method="daviesharte", **kwargs):
        """Instantiate the FBM"""
        super().__init__(
            drift=None, volatility=None, S0=S0, nb_paths=nb_paths,
            nb_steps=nb_steps, maturity=maturity, sine_coeff=None,)
        self.nb_paths = nb_paths
        self.nb_steps = nb_steps
        self.S0 = S0
        self.maturity = maturity
        self.hurst = hurst
        self.method = method
        self.dimensions = np.size(S0)
        self.loss = None
        self.path_t = None
        self.path_y = None

    def r_H(self, t, s):
        return 0.5 * (t**(2*self.hurst) + s**(2*self.hurst) -
                      np.abs(t-s)**(2*self.hurst))

    def get_cov_mat(self, times):
        m = np.array(times).reshape((-1,1)).repeat(len(times), axis=1)
        return self.r_H(m, np.transpose(m))

    def next_cond_exp(self, y, delta_t, current_t):
        t = current_t+delta_t
        next_y = np.zeros_like(y)
        for ii in range(y.shape[0]):
            if self.obs_cov_mat_inv[ii] is not None:
                r = self.r_H(np.array(self.observed_t[ii]), t)
                next_y[ii] = np.dot(r, np.matmul(
                    self.obs_cov_mat_inv[ii], np.array(self.observed_X[ii])))
        return next_y

    def compute_cond_exp(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, return_path=True, get_loss=False,
                         weight=0.5, store_and_use_stored=True,
                         start_time=None, which_loss="easy",
                         **kwargs):
        """
        Compute conditional expectation
        :param times: np.array, of observation times
        :param time_ptr: list, start indices of X and obs_idx for a given
                observation time, first element is 0, this pointer tells how
                many (and which) of the observations of X along the batch-dim
                belong to the current time, and obs_idx then tells to which of
                the batch elements they belong. In particular, not each batch-
                element has to jump at the same time, and only those elements
                which jump at the current time should be updated with a jump
        :param X: np.array, data tensor
        :param obs_idx: list, index of the batch elements where jumps occur at
                current time
        :param delta_t: float, time step for Euler
        :param T: float, the final time
        :param start_X: np.array, the starting point of X
        :param n_obs_ot: np.array, the number of observations over the
                entire time interval for each element of the batch
        :param return_path: bool, whether to return the path of h
        :param get_loss: bool, whether to compute the loss, otherwise 0 returned
        :param until_T: bool, whether to continue until T (for eval) or only
                until last observation (for training)
        :param M: None or torch.tensor, if not None: the mask for the data, same
                size as X, with 0 or 1 entries
        :return: float (loss), if wanted paths of t and y (np.arrays)
        """
        if return_path and store_and_use_stored:
            if get_loss:
                if self.path_t is not None and self.loss is not None:
                    return self.loss, self.path_t, self.path_y
            else:
                if self.path_t is not None:
                    return self.loss, self.path_t, self.path_y
        elif store_and_use_stored:
            if get_loss:
                if self.loss is not None:
                    return self.loss

        assert self.dimensions == 1, "cond. exp. computation of FBM only for 1d"
        assert self.S0 == 0, "cond. exp. computation of FBM only for S0=0"

        bs = start_X.shape[0]
        self.observed_t = [[] for x in range(bs)]
        self.observed_X = [[] for x in range(bs)]
        self.obs_cov_mat = [None for x in range(bs)]
        self.obs_cov_mat_inv = [None for x in range(bs)]

        y = start_X
        batch_size = bs
        current_time = 0.0
        if start_time:
            current_time = start_time

        loss = 0

        if return_path:
            if start_time:
                path_t = []
                path_y = []
            else:
                path_t = [0.]
                path_y = [y]

        for i, obs_time in enumerate(times):
            # Calculate conditional expectation stepwise
            while current_time < (obs_time - 1e-10 * delta_t):
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time
                y = self.next_cond_exp(y, delta_t_, current_time)
                current_time = current_time + delta_t_

                # Storing the conditional expectation
                if return_path:
                    path_t.append(current_time)
                    path_y.append(y)

            # Reached an observation - set new interval
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]

            # add to observed
            for j, ii in enumerate(i_obs):
                self.observed_t[ii].append(obs_time)
                self.observed_X[ii].append(X_obs[j, 0])
                self.obs_cov_mat[ii] = self.get_cov_mat(self.observed_t[ii])
                self.obs_cov_mat_inv[ii] = np.linalg.inv(self.obs_cov_mat[ii])

            # Update h. Also updating loss, tau and last_X
            Y_bj = y
            temp = copy.copy(y)
            temp[i_obs] = X_obs
            y = temp
            Y = y

            if get_loss:
                loss = loss + compute_loss(
                    X_obs=X_obs, Y_obs=Y[i_obs], Y_obs_bj=Y_bj[i_obs],
                    n_obs_ot=n_obs_ot[i_obs], batch_size=batch_size,
                    weight=weight, which_loss=which_loss)
            if return_path:
                path_t.append(obs_time)
                path_y.append(y)

        # after every observation has been processed, propagating until T
        while current_time < T - 1e-10 * delta_t:
            if current_time < T - delta_t:
                delta_t_ = delta_t
            else:
                delta_t_ = T - current_time
            y = self.next_cond_exp(y, delta_t_)
            current_time = current_time + delta_t_

            # Storing the predictions.
            if return_path:
                path_t.append(current_time)
                path_y.append(y)

        if get_loss and store_and_use_stored:
            self.loss = loss
        if return_path and store_and_use_stored:
            self.path_t = np.array(path_t)
            self.path_y = np.array(path_y)

        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            return loss, np.array(path_t), np.array(path_y)
        else:
            return loss

    def generate_paths(self, start_X=None):
        spot_paths = np.empty(
            (self.nb_paths, self.dimensions, self.nb_steps + 1))
        dt = self.dt
        if start_X is not None:
            spot_paths[:, :, 0] = start_X
        else:
            spot_paths[:, :, 0] = self.S0
        for i in range(self.nb_paths):
            for j in range(self.dimensions):
                fgn_sample = fgn(n=self.nb_steps, hurst=self.hurst,
                             length=self.maturity, method=self.method)
                spot_paths[i, j, 1:] = np.cumsum(fgn_sample)+spot_paths[i, j, 0]
        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


class SP500(StockModel):
    """
    Data from SP500: https://www.kaggle.com/camnugent/sandp500
    """
    def __init__(self, **kwargs):
        self.nb_paths = 470

    def generate_paths(self, start_X=None):
        spot_paths = np.genfromtxt("SP500data.csv", delimiter=",")
        spot_paths = np.expand_dims(spot_paths, axis=1)
        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, 1

class OrnsteinUhlenbeck(StockModel):
    """
    Ornstein-Uhlenbeeck stock model, see:
    https://en.wikipedia.org/wiki/Ornstein–Uhlenbeck_process
    """

    def __init__(self, volatility, nb_paths, nb_steps, S0,
                 mean, speed, maturity, sine_coeff=None, **kwargs):
        super(OrnsteinUhlenbeck, self).__init__(
            volatility=volatility, nb_paths=nb_paths, drift=None,
            nb_steps=nb_steps, S0=S0, maturity=maturity,
            sine_coeff=sine_coeff
        )
        self.mean = mean
        self.speed = speed

    def next_cond_exp(self, y, delta_t, current_t):
        exp_delta = np.exp(-self.speed * self.periodic_coeff(current_t) * delta_t)
        return y * exp_delta + self.mean * (1 - exp_delta)

    def generate_paths(self, start_X=None):
        # Diffusion of the variance: dv = -k(v-vinf)*dt + vol*dW
        drift = lambda x, t: - self.speed * self.periodic_coeff(t) * (x - self.mean)
        diffusion = lambda x, t: self.volatility

        spot_paths = np.empty(
            (self.nb_paths, self.dimensions, self.nb_steps + 1))
        dt = self.dt
        if start_X is not None:
            spot_paths[:, :, 0] = start_X
        for i in range(self.nb_paths):
            if start_X is None:
                spot_paths[i, :, 0] = self.S0
            for k in range(1, self.nb_steps + 1):
                random_numbers = np.random.normal(0, 1, self.dimensions)
                dW = random_numbers * np.sqrt(dt)
                spot_paths[i, :, k] = (
                        spot_paths[i, :, k - 1]
                        + drift(spot_paths[i, :, k - 1], (k - 1) * dt) * dt
                        + diffusion(spot_paths[i, :, k - 1], (k) * dt) * dW)
        # stock_path dimension: [nb_paths, dimension, time_steps]
        return spot_paths, dt


class Combined(StockModel):
    def __init__(self, stock_model_names, hyperparam_dicts, **kwargs):
        self.stock_model_names = stock_model_names
        self.hyperparam_dicts = hyperparam_dicts

    def compute_cond_exp(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                         n_obs_ot, return_path=True, get_loss=False,
                         weight=0.5, which_loss="easy", **kwargs):
        # get first stockmodel
        stockmodel = DATASETS[self.stock_model_names[0]](
            **self.hyperparam_dicts[0])
        T = self.hyperparam_dicts[0]['maturity']
        loss, path_t, path_y = stockmodel.compute_cond_exp(
            times, time_ptr, X, obs_idx, delta_t,
            T, start_X,
            n_obs_ot, return_path=True, get_loss=get_loss,
            weight=weight, store_and_use_stored=False, which_loss=which_loss)
        for i in range(1, len(self.stock_model_names)):
            start_X = path_y[-1, :, :]
            start_time = path_t[-1]
            T += self.hyperparam_dicts[i]['maturity']
            stockmodel = DATASETS[self.stock_model_names[i]](
                **self.hyperparam_dicts[i])
            _loss, _path_t, _path_y = stockmodel.compute_cond_exp(
                times, time_ptr, X, obs_idx, delta_t,
                T, start_X,
                n_obs_ot, return_path=True, get_loss=get_loss,
                weight=weight, start_time=start_time,
                store_and_use_stored=False, which_loss=which_loss)
            loss += _loss
            path_t = np.concatenate([path_t, _path_t])
            path_y = np.concatenate([path_y, _path_y], axis=0)

        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            return loss, np.array(path_t), np.array(path_y)
        else:
            return loss


class DoublePendulum:
    """
    see formulas at https://math24.net/double-pendulum.html
    Runge-Kutta-4 is used to solve the ODE numerically
    """
    def __init__(self, start_alpha, length, mass_ratio, step_size=0.01):
        self.start_alpha = start_alpha
        self.reset()
        self.l = length
        self.m1 = 1
        self.mu = mass_ratio
        self.step_size = step_size
        self.g = 9.81

    def reset(self):
        self.Z = np.array(
            [self.start_alpha,
             self.start_alpha,
             np.zeros_like(self.start_alpha),
             np.zeros_like(self.start_alpha)])

    def set(self, Z):
        self.Z = Z

    def diff_equ(self, Z):
        return np.array([self.f1(Z), self.f2(Z), self.f3(Z), self.f4(Z)])

    @staticmethod
    def a1(Z):
        return Z[0]

    @staticmethod
    def a2(Z):
        return Z[1]

    @staticmethod
    def p1(Z):
        return Z[2]

    @staticmethod
    def p2(Z):
        return Z[3]

    def d(self, Z):
        return self.m1*(self.l**2)*(1+self.mu*np.sin(self.a1(Z)-self.a2(Z))**2)

    def f1(self, Z):
        n = self.p1(Z)-self.p2(Z)*np.cos(self.a1(Z)-self.a2(Z))
        d = self.d(Z)
        return n/d

    def f2(self, Z):
        n = self.p2(Z)*(1+self.mu)-self.p1(Z)*self.mu*np.cos(self.a1(Z)-self.a2(Z))
        d = self.d(Z)
        return n/d

    def A1(self, Z):
        n = self.p1(Z)*self.p2(Z)*np.sin(self.a1(Z)-self.a2(Z))
        d = self.d(Z)
        return n/d

    def A2(self, Z):
        d = 2*self.d(Z)*(1+self.mu*np.sin(self.a1(Z)-self.a2(Z))**2)
        n1 = self.p1(Z)**2*self.mu-\
             2*self.p1(Z)*self.p2(Z)*self.mu*np.cos(self.a1(Z)-self.a2(Z))+\
             self.p2(Z)**2*(1+self.mu)
        n2 = np.sin(2*(self.a1(Z)-self.a2(Z)))
        return n1*n2/d

    def f3(self, Z):
        n1 = -self.m1*(1+self.mu)*self.g*self.l*np.sin(self.a1(Z))
        return n1-self.A1(Z)+self.A2(Z)

    def f4(self, Z):
        n1 = -self.m1*self.mu*self.g*self.l*np.sin(self.a2(Z))
        return n1+self.A1(Z)-self.A2(Z)

    def update(self):
        """
        uses RK4 method to update the state vektor Z
        """
        Y1 = self.step_size*self.diff_equ(self.Z)
        Y2 = self.step_size*self.diff_equ(self.Z+Y1/2)
        Y3 = self.step_size*self.diff_equ(self.Z+Y2/2)
        Y4 = self.step_size*self.diff_equ(self.Z+Y3)
        Znew = self.Z + (Y1+2*Y2+2*Y3+Y4)/6
        self.Z = Znew
        return Znew


def plot_pendulum(nb_steps, start_alpha=0., length=1., mass_ratio=1.,
                  step_size=0.01, interval=10):
    l = length
    fig, ax = plt.subplots()
    p0, = ax.plot([0], [0], marker="o")
    x1 = l*np.cos(start_alpha-np.pi/2)
    y1 = l*np.sin(start_alpha-np.pi/2)
    x2 = x1 + l*np.cos(start_alpha-np.pi/2)
    y2 = y1 + l*np.sin(start_alpha-np.pi/2)
    p1, = ax.plot([x1], [y1], marker="o")
    p2, = ax.plot([x2], [y2], marker="o")
    l1, = ax.plot([0, x1], [0, y1], marker="", lw="1")
    l2, = ax.plot([x1, x2], [y1, y2], marker="", lw="1")

    def init():
        ax.set_ylim(-3, 3)
        ax.set_xlim(-3, 3)

    def data_gen():
        s = 0
        DP = DoublePendulum(start_alpha=start_alpha, length=length,
                            mass_ratio=mass_ratio, step_size=step_size)
        while s < nb_steps:
            s += 1
            yield DP.update()

    def run(data):
        Z = data
        x1 = l*np.cos(Z[0]-np.pi/2)
        y1 = l*np.sin(Z[0]-np.pi/2)
        x2 = x1 + l*np.cos(Z[1]-np.pi/2)
        y2 = y1 + l*np.sin(Z[1]-np.pi/2)
        p1.set_data([x1], [y1])
        p2.set_data([x2], [y2])
        l1.set_data([0, x1], [0, y1])
        l2.set_data([x1, x2], [y1, y2])

    ani = animation.FuncAnimation(
        fig, run, data_gen, blit=False, interval=interval,
        repeat=False, init_func=init)

    plt.show()


class DoublePendulumDataset(StockModel):
    def __init__(
            self, start_alpha_mean, start_alpha_std, length, mass_ratio,
            nb_paths, maturity=1., sampling_step_size=0.01,
            sampling_nb_steps=1000, use_every_n=10, **kwargs):
        """
        Args:
            start_alpha_mean: mean for normal distribution of which starting
                angle is sampled
            start_alpha_std: std for normal distribution of which starting
                angle is sampled
            length: length of pendulum components
            mass_ratio: ratio m2/m1 for mass of first (m1) and second (m2)
                pendulum
            nb_paths: int, number of paths to sample
            nb_steps: int, number of steps
            step_size:
            use_every_n:
        """
        self.path_t = None
        self.path_y = None

        self.start_alpha_mean = start_alpha_mean
        self.start_alpha_std = start_alpha_std
        self.nb_paths = nb_paths
        self.maturity = maturity
        self.sampling_step_size = maturity/sampling_nb_steps
        self.sampling_nb_steps = sampling_nb_steps
        self.use_every_n = use_every_n
        self.nb_steps = self.sampling_nb_steps/use_every_n
        self.dt = maturity/self.nb_steps
        self.dimensions = 4

        self.length = length
        self.mass_ratio = mass_ratio

    def generate_paths(self, start_X=None):
        start_alphas = np.random.normal(
            self.start_alpha_mean, self.start_alpha_std, self.nb_paths)

        paths = np.empty(
            (self.nb_paths, self.dimensions, self.sampling_nb_steps+1))
        DP = DoublePendulum(
            start_alpha=start_alphas, length=self.length,
            mass_ratio=self.mass_ratio, step_size=self.sampling_step_size)
        paths[:, :, 0] = np.transpose(DP.Z)
        for j in range(self.sampling_nb_steps):
            paths[:, :, j+1] = np.transpose(DP.update())

        # stock_path dimension: [nb_paths, dimension, time_steps]
        return paths[:, :, ::self.use_every_n], self.dt

    def next_cond_exp(self, y, delta_t_, current_time):
        DP = DoublePendulum(
            start_alpha=0., length=self.length, mass_ratio=self.mass_ratio,
            step_size=delta_t_*self.sampling_step_size*
                      self.sampling_nb_steps/self.maturity)
        DP.set(Z=np.transpose(y))
        next_y = np.transpose(DP.update())
        return next_y








# ==============================================================================
# this is needed for computing the loss with the true conditional expectation
def compute_loss(X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
                 weight=0.5, M_obs=None, which_loss="easy"):
    """
    compute the loss of the true conditional expectation, as in
    model.compute_loss
    """
    loss_fun = LOSS_FUN_DICT[which_loss]
    if M_obs is not None:
        M_obs = torch.from_numpy(M_obs)
    loss = loss_fun(
        X_obs=torch.from_numpy(X_obs),
        Y_obs=torch.from_numpy(Y_obs),
        Y_obs_bj=torch.from_numpy(Y_obs_bj),
        n_obs_ot=torch.from_numpy(n_obs_ot),
        batch_size=batch_size, eps=eps, weight=weight,
        M_obs=M_obs).detach().cpu().numpy()
    return loss


# ==============================================================================
# dict for the supported stock models to get them from their name
DATASETS = {
    "BlackScholes": BlackScholes,
    "Heston": Heston,
    "OrnsteinUhlenbeck": OrnsteinUhlenbeck,
    "HestonWOFeller": HestonWOFeller,
    "combined": Combined,
    "sine_BlackScholes": BlackScholes,
    "sine_Heston": Heston,
    "sine_OrnsteinUhlenbeck": OrnsteinUhlenbeck,
    "PoissonPointProcess": PoissonPointProcess,
    "FBM": FracBM,
    "BM2DCorr": BM2DCorr,
    "BMandVar": BMandVar,
    "BM": BM,
    "SP500": SP500,
    "DoublePendulum": DoublePendulumDataset,
    "BMFiltering": BMFiltering,
    "BMwithTimeLag": BMwithTimeLag,
    "BMNoisyObs": BMNoisyObs,
    "BlackScholesUncertainParams": BlackScholesUncertainParams,
    "BMwithUncertainDrift": BMwithUncertainDrift,
    "BMandQuantiles": BMandQuantiles,
    "CIRUncertainParams": CIRUncertainParams,
    "BMClassification": BMClassification,
}
# ==============================================================================



if __name__ == '__main__':

    # ------------------------------
    # sm = FracBM(1, 100, 0, 1, 0.05)
    # paths = sm.generate_paths()
    # t = np.linspace(0, 1, 101)
    # rand_ind = np.random.random(len(t))
    # rand_ind = rand_ind > 0.9
    # rand_ind = rand_ind.tolist()
    # x = paths[0,0,rand_ind]
    # obs = t[rand_ind]
    # plt.plot(t, paths[0,0])
    # plt.plot(obs, x)
    # plt.show()
    # sm.observed_t = [obs[:7]]
    # sm.observed_X = [x[:7]]
    # sm.obs_cov_mat = [sm.get_cov_mat(sm.observed_t[0])]
    # sm.obs_cov_mat_inv = [np.linalg.inv(sm.obs_cov_mat[0])]
    # sm.next_cond_exp(x[6:7], 0, obs[6])
    # sm.next_cond_exp(x[6:7], 0.1, obs[6])
    # sm.next_cond_exp(x[6:7], 1e-16, obs[6])

    # ------------------------------------
    # plot_pendulum(1000, start_alpha=np.pi-0.0013, length=1., mass_ratio=1.,
    #               step_size=0.01, interval=1)
    # plot_pendulum(100, start_alpha=np.pi-0.2, length=1., mass_ratio=1.,
    #               step_size=0.1, interval=100)
    plot_pendulum(100, start_alpha=np.pi-0.05, length=1., mass_ratio=1.,
                  step_size=0.025, interval=100)

    pass


