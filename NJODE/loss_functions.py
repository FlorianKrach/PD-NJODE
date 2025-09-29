"""
author: Florian Krach

implementation of the loss functions used in the NJODE model
"""



# ==============================================================================
import torch
import numpy as np
import warnings

# ==============================================================================
def compute_var_loss(
        X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
        weight=0.5, M_obs=None,
        compute_variance=None, var_weight=1.,
        Y_var_bj=None, Y_var=None, dim_to=None, type=1, **kwargs):
    if dim_to is None:
        dim_to = Y_obs.shape[1]
    if compute_variance == "variance":
        Y_var_bj_v = Y_var_bj[:, :dim_to] ** 2
        Y_var_v = Y_var[:, :dim_to] ** 2
        true_var_bj = (X_obs - Y_obs_bj.detach()) ** 2
        true_var = (X_obs - Y_obs.detach()) ** 2
        M_obs_var = M_obs
        sum_dim = 1
    elif compute_variance == "covariance":
        d2 = Y_var.shape[1]
        d = int(np.sqrt(d2))
        Y_var_bj_v = Y_var_bj.view(-1, d, d)
        Y_var_bj_v = Y_var_bj_v[:, :dim_to, :dim_to]
        Y_var_bj_v = torch.matmul(Y_var_bj_v, Y_var_bj_v.transpose(1, 2))
        Y_var_v = Y_var.view(-1, d, d)
        Y_var_v = Y_var_v[:, :dim_to, :dim_to]
        Y_var_v = torch.matmul(Y_var_v, Y_var_v.transpose(1, 2))
        # the following is a row vector for each batch sample
        true_var_bj = (X_obs - Y_obs_bj.detach()).unsqueeze(2)
        true_var = (X_obs - Y_obs.detach()).unsqueeze(2)
        true_var_bj = torch.matmul(true_var_bj, true_var_bj.transpose(1, 2))
        true_var = torch.matmul(true_var, true_var.transpose(1, 2))
        if M_obs == 1.:
            M_obs_var = 1.
        else:
            M_obs_var = M_obs.unsqueeze(2)
            M_obs_var = torch.matmul(M_obs_var, M_obs_var.transpose(1, 2))
        sum_dim = (1, 2)
    else:
        raise ValueError("compute_variance must be either 'variance' or "
                         "'covariance'")
    # print("compute_variance: ", compute_variance)
    # print("M_obs_var: ", M_obs_var.shape)
    # print("true_var_bj: ", true_var_bj.shape)
    # print("true_var: ", true_var.shape)
    # print("Y_var_bj_v: ", Y_var_bj_v.shape)
    # print("Y_var_v: ", Y_var_v.shape)
    # print("sum_dim: ", sum_dim)
    if type == 1:
        inner_var = (2 * weight * torch.sqrt(
            torch.sum(M_obs_var * (true_var - Y_var_v)**2, dim=sum_dim) + eps) +
                     2 * (1 - weight) * torch.sqrt(
                    torch.sum(M_obs_var * (true_var_bj - Y_var_bj_v) ** 2,
                              dim=sum_dim) + eps)) ** 2
    elif type == 2:
        inner_var = 2 * weight * torch.sum(
            M_obs_var * (true_var - Y_var_v) ** 2, dim=sum_dim) + \
                    2 * (1 - weight) * torch.sum(
            M_obs_var * (true_var_bj - Y_var_bj_v) ** 2, dim=sum_dim)
    elif type == 3:
        inner_var = torch.sum(M_obs_var * (true_var_bj - Y_var_bj_v) ** 2,
                              dim=sum_dim)
        # TODO: it could also make sense to use the second part of the var loss
        #   additionally here, since this does not (wrongly) force the model to
        #   set the var to 0. However, it should be checked theoretically,
        #   whether the conditional variance of observations O given past obs of
        #   O is the same as the conditional variance of X given past obs of O.
        #   Not checked yet! Use with caution!
        warnings.warn("compute_variance not theoretically checked yet "
                      "-> use with caution!", UserWarning)
    else:
        raise ValueError("type must be in {1, 2, 3}")

    outer = var_weight * torch.sum(inner_var / n_obs_ot)

    return outer


def compute_loss(
        X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
        weight=0.5, M_obs=None,
        compute_variance=None, var_weight=1.,
        Y_var_bj=None, Y_var=None, dim_to=None, which_var_loss=None,
        **kwargs):
    """
    loss function from the paper
    :param X_obs: torch.tensor, the true X values at the observations
    :param Y_obs: torch.tensor, the predicted values at the observation
    :param Y_obs_bj: torch.tensor, the predicted values before the jump at the
            observation
    :param n_obs_ot: torch.tensor, the number of observations over the entire
            time-line for each element of the batch
    :param batch_size: int or float
    :param eps: float, a small constant which is added before taking torch.sqrt
            s.t. the sqrt never becomes zero (which would yield NaNs for the
            gradient)
    :param weight: float in [0,1], weighting of the two parts of the loss
            function,
                0.5: standard loss as described in paper
                (0.5, 1): more weight to be correct after the jump, can be
                theoretically justified similar to standard loss
                1: only the value after the jump is trained
    :param M_obs: None or torch.tensor, if not None: same size as X_obs with
            0  and 1 entries, telling which coordinates were observed
    :param compute_variance: None or str in {"variance", "covariance"}, if not
            None: compute the marginal variance or covariance matrix of X
    :param var_weight: float, weight of the variance term in the loss function
    :param Y_var_bj: None or torch.tensor, the predicted variance output before
            the jump. Note: the variance output corresponds to the square root
            of the variance or Cholesky decomposition of the covariance matrix.
    :param Y_var: None or torch.tensor, the predicted variance output after the
            jump. Note: the variance output corresponds to the square root of
            the variance or Cholesky decomposition of the covariance matrix.
    :param dim_to: None or int, if not None: the dimension to which the
            covariance matrix should be reduced. This is only used if
            compute_variance is not None. Y_obs_bj and Y_obs are already passed
            in reduces size.
    :param which_var_loss: None or int, if not None: the type of variance loss
            function to be used. This is only used if compute_variance is not
            None. 1: standard loss, 2: sum of the two squared terms,
            3: only the second term (like noisy obs loss). default: None; this
            uses predefined type based on the chosen loss function.
    :return: torch.tensor (with the loss, reduced to 1 dim)
    """
    if M_obs is None:
        M_obs = 1.

    inner = (2 * weight * torch.sqrt(
        torch.sum(M_obs * (X_obs - Y_obs) ** 2, dim=1) + eps) +
             2 * (1 - weight) * torch.sqrt(
                torch.sum(M_obs * (Y_obs_bj - Y_obs)**2, dim=1) + eps))**2
    outer = torch.sum(inner / n_obs_ot)

    # compute the variance loss term if wanted
    if compute_variance is not None:
        var_loss_type = 1
        if which_var_loss is not None:
            var_loss_type = which_var_loss
        outer += compute_var_loss(
            X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
            weight=weight, M_obs=M_obs,
            compute_variance=compute_variance, var_weight=var_weight,
            Y_var_bj=Y_var_bj, Y_var=Y_var, dim_to=dim_to, type=var_loss_type)

    return outer / batch_size


def compute_loss_2(
        X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
        weight=0.5, M_obs=None,
        compute_variance=None, var_weight=1.,
        Y_var_bj=None, Y_var=None, dim_to=None, which_var_loss=None,
        **kwargs):
    """
    similar to compute_loss, but using X_obs also in second part of loss
    instead of Y_obs -> should make the learning easier
    """
    if M_obs is None:
        M_obs = 1.

    inner = (2*weight * torch.sqrt(
        torch.sum(M_obs * (X_obs - Y_obs) ** 2, dim=1) + eps) +
             2*(1 - weight) * torch.sqrt(
                torch.sum(M_obs * (Y_obs_bj - X_obs) ** 2, dim=1)
                + eps)) ** 2
    outer = torch.sum(inner / n_obs_ot)

    # compute the variance loss term if wanted
    if compute_variance is not None:
        var_loss_type = 1
        if which_var_loss is not None:
            var_loss_type = which_var_loss
        outer += compute_var_loss(
            X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
            weight=weight, M_obs=M_obs,
            compute_variance=compute_variance, var_weight=var_weight,
            Y_var_bj=Y_var_bj, Y_var=Y_var, dim_to=dim_to, type=var_loss_type)

    return outer / batch_size


def compute_loss_2_1(
        X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
        weight=0.5, M_obs=None,
        compute_variance=None, var_weight=1.,
        Y_var_bj=None, Y_var=None, dim_to=None, which_var_loss=None,
        **kwargs):
    """
    similar to compute_loss, but using X_obs also in second part of loss
    instead of Y_obs and squaring the two terms directly instead of squaring the
    sum of the two terms
    """
    if M_obs is None:
        M_obs = 1.

    inner = 2*weight * torch.sum(M_obs * (X_obs - Y_obs) ** 2, dim=1) +\
            2*(1-weight) * torch.sum(M_obs * (Y_obs_bj - X_obs) ** 2, dim=1)

    outer = torch.sum(inner / n_obs_ot)

    # compute the variance loss term if wanted
    if compute_variance is not None:
        var_loss_type = 2
        if which_var_loss is not None:
            var_loss_type = which_var_loss
        outer += compute_var_loss(
            X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
            weight=weight, M_obs=M_obs,
            compute_variance=compute_variance, var_weight=var_weight,
            Y_var_bj=Y_var_bj, Y_var=Y_var, dim_to=dim_to, type=var_loss_type)

    return outer / batch_size


def compute_loss_noisy_obs(
        X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
        weight=0.5, M_obs=None,
        compute_variance=None, var_weight=1.,
        Y_var_bj=None, Y_var=None, dim_to=None, which_var_loss=None,
        **kwargs):
    """
    similar to compute_loss, but only using the 2nd term of the original loss
    function, which enables training with noisy observations
    """
    if M_obs is None:
        M_obs = 1.

    inner = torch.sum(M_obs * (Y_obs_bj - X_obs) ** 2, dim=1)

    outer = torch.sum(inner / n_obs_ot)

    # compute the variance loss term if wanted
    if compute_variance is not None:
        var_loss_type = 3
        if which_var_loss is not None:
            var_loss_type = which_var_loss
        outer += compute_var_loss(
            X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
            weight=weight, M_obs=M_obs,
            compute_variance=compute_variance, var_weight=var_weight,
            Y_var_bj=Y_var_bj, Y_var=Y_var, dim_to=dim_to, type=var_loss_type)

    return outer / batch_size


def compute_loss_3(
        X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
        weight=0.5, M_obs=None,
        compute_variance=None, var_weight=1.,
        Y_var_bj=None, Y_var=None, dim_to=None, which_var_loss=None,
        **kwargs):
    """
    loss function with 1-norm instead of 2-norm for the two terms
    intuitively this should lead to the conditional median instead of mean (not
    proven)

    :param X_obs: torch.tensor, the true X values at the observations
    :param Y_obs: torch.tensor, the predicted values at the observation
    :param Y_obs_bj: torch.tensor, the predicted values before the jump at the
            observation
    :param n_obs_ot: torch.tensor, the number of observations over the entire
            time-line for each element of the batch
    :param batch_size: int or float
    :param eps: float, a small constant which is added before taking torch.sqrt
            s.t. the sqrt never becomes zero (which would yield NaNs for the
            gradient)
    :param weight: float in [0,1], weighting of the two parts of the loss
            function,
                0.5: standard loss as described in paper
                (0.5, 1): more weight to be correct after the jump, can be
                theoretically justified similar to standard loss
                1: only the value after the jump is trained
    :param M_obs: None or torch.tensor, if not None: same size as X_obs with
            0  and 1 entries, telling which coordinates were observed
    :return: torch.tensor (with the loss, reduced to 1 dim)
    """
    if M_obs is None:
        M_obs = 1.

    inner = (2*weight * torch.sum(M_obs * (torch.abs(X_obs - Y_obs)), dim=1) +
             2*(1 - weight) * torch.sum(M_obs * (torch.abs(Y_obs_bj - Y_obs)), dim=1))
    outer = torch.sum(inner / n_obs_ot)

    if compute_variance is not None:
        warnings.warn("compute_variance not implemented for the loss function "
                      "'compute_loss_3'", UserWarning)

    return outer / batch_size


def compute_jump_loss(
        X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
        weight=0.5, M_obs=None, compute_variance=None, **kwargs):
    """
    loss function with 1-norm instead of 2-norm for the two terms
    intuitively this should lead to the conditional median instead of mean (not
    proven)

    :param X_obs: torch.tensor, the true X values at the observations
    :param Y_obs: torch.tensor, the predicted values at the observation
    :param Y_obs_bj: torch.tensor, the predicted values before the jump at the
            observation
    :param n_obs_ot: torch.tensor, the number of observations over the entire
            time-line for each element of the batch
    :param batch_size: int or float
    :param eps: float, a small constant which is added before taking torch.sqrt
            s.t. the sqrt never becomes zero (which would yield NaNs for the
            gradient)
    :param weight: float in [0,1], weighting of the two parts of the loss
            function,
                0.5: standard loss as described in paper
                (0.5, 1): more weight to be correct after the jump, can be
                theoretically justified similar to standard loss
                1: only the value after the jump is trained
    :param M_obs: None or torch.tensor, if not None: same size as X_obs with
            0  and 1 entries, telling which coordinates were observed
    :param kwargs: additional arguments, not used; to be compatible with the
            other loss functions which allow for variance computations
    :return: torch.tensor (with the loss, reduced to 1 dim)
    """
    if M_obs is None:
        M_obs = 1.

    inner = torch.sum(M_obs * (X_obs - Y_obs) ** 2, dim=1)
    outer = torch.sum(inner / n_obs_ot)

    if compute_variance is not None:
        warnings.warn("compute_variance not implemented for the loss function "
                      "'compute_jump_loss'", UserWarning)

    return outer / batch_size


def compute_quantile_loss(quantiles):
    """
    function to create a quantile loss function
    :param quantiles: list of float, the quantiles at which the loss should be
            computed
    :return: function, the loss function
    """
    def loss(X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
             weight=0.5, M_obs=None, compute_variance=None, **kwargs):
        """
        loss function with quantile loss
        :param X_obs: torch.tensor, the true X values at the observations
        :param Y_obs: torch.tensor, the predicted values at the observation
        :param Y_obs_bj: torch.tensor, the predicted values before the jump at
                the observation
        :param n_obs_ot: torch.tensor, the number of observations over the
                entire time-line for each element of the batch
        :param batch_size: int or float
        :param eps: float, a small constant which is added before taking
                torch.sqrt s.t. the sqrt never becomes zero (which would yield
                NaNs for the gradient)
        :param weight: float in [0,1], weighting of the two parts of the loss
                function,
                    0.5: standard loss as described in paper
                    (0.5, 1): more weight to be correct after the jump, can be
                    theoretically justified similar to standard loss
                    1: only the value after the jump is trained
        :param M_obs: None or torch.tensor, if not None: same size as X_obs with
                0  and 1 entries, telling which coordinates were observed
        :param kwargs: additional arguments, not used
        :return: torch.tensor (with the loss, reduced to 1 dim)
        """
        nbq = len(quantiles)
        if M_obs is None:
            M_obs = 1.

        _X_obs = X_obs.unsqueeze(2).repeat(1, 1, nbq)
        quants = torch.tensor(quantiles).to(X_obs.device)
        quants = quants.reshape(1,1,-1).repeat(X_obs.shape[0],X_obs.shape[1],1)

        inner = torch.sum(M_obs * torch.max(
            (_X_obs - Y_obs_bj)*quants, (Y_obs_bj - _X_obs)*(1-quants)), dim=2)
        outer = torch.sum(inner, dim=1) / n_obs_ot

        if compute_variance is not None:
            warnings.warn(
                "compute_variance not implemented for the loss function "
                "'compute_loss_3'", UserWarning)

        return torch.sum(outer) / batch_size

    return loss


def compute_quantile_jump_loss(quantiles):
    """
    function to create a quantile loss function
    :param quantiles: list of float, the quantiles at which the loss should be
            computed
    :return: function, the loss function
    """
    def loss(X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
             weight=0.5, M_obs=None, compute_variance=None, **kwargs):
        """
        loss function with quantile loss
        :param X_obs: torch.tensor, the true X values at the observations
        :param Y_obs: torch.tensor, the predicted values at the observation
        :param Y_obs_bj: torch.tensor, the predicted values before the jump at
                the observation
        :param n_obs_ot: torch.tensor, the number of observations over the
                entire time-line for each element of the batch
        :param batch_size: int or float
        :param eps: float, a small constant which is added before taking
                torch.sqrt s.t. the sqrt never becomes zero (which would yield
                NaNs for the gradient)
        :param weight: float in [0,1], weighting of the two parts of the loss
                function,
                    0.5: standard loss as described in paper
                    (0.5, 1): more weight to be correct after the jump, can be
                    theoretically justified similar to standard loss
                    1: only the value after the jump is trained
        :param M_obs: None or torch.tensor, if not None: same size as X_obs with
                0  and 1 entries, telling which coordinates were observed
        :param kwargs: additional arguments, not used
        :return: torch.tensor (with the loss, reduced to 1 dim)
        """
        nbq = len(quantiles)
        if M_obs is None:
            _M_obs = 1.
        else:
            _M_obs = M_obs.unsqueeze(2).repeat(1, 1, nbq)
        _X_obs = X_obs.unsqueeze(2).repeat(1, 1, nbq)
        quants = torch.tensor(quantiles).to(X_obs.device)
        quants = quants.reshape(1,1,-1).repeat(X_obs.shape[0],X_obs.shape[1],1)

        inner = 2 * weight * torch.sum(_M_obs * torch.max(
            (_X_obs - Y_obs_bj)*quants, (Y_obs_bj - _X_obs)*(1-quants)), dim=2)
        inner += 2 * (1-weight) * torch.sum(
            _M_obs * (_X_obs - Y_obs) ** 2, dim=2)
        outer = torch.sum(inner, dim=1) / n_obs_ot

        if compute_variance is not None:
            warnings.warn(
                "compute_variance not implemented for the loss function "
                "'compute_loss_3'", UserWarning)

        return torch.sum(outer) / batch_size

    return loss


def compute_loss_vola(
        X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
        weight=0.5, M_obs=None,
        compute_variance=None, var_weight=1.,
        Y_var_bj=None, Y_var=None, dim_to=None, which_var_loss=None, version=1,
        **kwargs):
    """
    loss function for learning the vola estimate of an Ito process for usage in
    a generative model.
    in particular, X_obs is assumed to be the left-limit of the Z process

    X_obs = Z_minus
    Z_minus = (X_t - X_tau)^2

    -> the loss trains the model to replicate Z_minus before the jump (with
    Y_obs_bj) and to jump to 0 (with Y_obs), which is the value of Z_plus at an
    observation.

    the loss supports 3 versions:
    - 0: original (easy) loss function style (joint squaring of two loss terms)
    - 1: IO loss function style (individual squaring of two loss terms)
    - 2: noise-adapted loss function style (no direct penalization after jump).
        this is used by compute_loss_vola_lim (see description there).
    default: version 1
    """
    if M_obs is None:
        M_obs = 1.

    d2 = Y_obs_bj.shape[1]
    d = int(np.sqrt(d2))
    Y_obs_bj_mat = Y_obs_bj.view(-1, d, d)
    Y_obs_bj_mat1 = torch.matmul(Y_obs_bj_mat, Y_obs_bj_mat.transpose(1, 2))
    Y_obs_bj_mod = Y_obs_bj_mat1.view(-1, d2)

    d2 = Y_obs.shape[1]
    d = int(np.sqrt(d2))
    Y_obs_mat = Y_obs.view(-1, d, d)
    Y_obs_mat1 = torch.matmul(Y_obs_mat, Y_obs_mat.transpose(1, 2))
    Y_obs_mod = Y_obs_mat1.view(-1, d2)

    if version == 0:  # style: original (easy) loss function (joint squaring)
      inner = (2*weight * torch.sqrt(
        torch.sum(M_obs * (Y_obs_bj_mod - X_obs) ** 2, dim=1) + eps) +
              2*(1 - weight) * torch.sqrt(
                torch.sum(M_obs * (Y_obs_mod) ** 2, dim=1)
                + eps)) ** 2
    elif version == 1:  # style: IO loss function (individual squaring)
        inner = (2*weight*torch.sum(M_obs * (Y_obs_bj_mod - X_obs) ** 2, dim=1)+
              2*(1 - weight) * torch.sum(M_obs * (Y_obs_mod) ** 2, dim=1))
    elif version == 2:  # style: noise-adapted loss function
        inner = torch.sum(M_obs * (Y_obs_bj_mod - X_obs) ** 2, dim=1)
    else:
        raise ValueError("version must be in {0, 1, 2}")

    outer = torch.sum(inner / n_obs_ot)

    return outer / batch_size


def compute_loss_vola_lim(
        X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
        weight=0.5, M_obs=None,
        compute_variance=None, var_weight=1.,
        Y_var_bj=None, Y_var=None, dim_to=None, which_var_loss=None,
        **kwargs):
    """
    loss function for learning the vola estimate of an Ito process for usage in
    a generative model.
    in particular, X_obs is assumed to be the left-limit of the Z process devided
    by the time difference since last observation.

    X_obs = Z_minus/tdiff
    Z_minus = (X_t - X_tau)^2
    tdiff = t - tau

    -> the loss trains the model to replicate Z_minus before the jump (with
    Y_obs_bj) and does not penalize the jump directly. with suitable training
    data, this forces the model to jump to the right-limit of Z_minus/tdiff for
    tdiff->0, which is the instantaneous vola coefficient (if necessary
    information available, otherwise its best estimate).

    BIAS CORRECTION:
    the loss also supports the case where a bias-corrected version of Z_minus is
    used instead of Z_minus, i.e., Z_minus is replaced by
    Z_minus_biascorrected = (X_t - X_tau - E[X_t - X_tau | X_tau])^2 =
        = (X_t - E[X_t | X_tau])^2.
    in this case, the theoretical right-limit is still the instantaneous vola
    coefficient. since the values of Z_minus_biascorrected have a smaller
    magnitude, this can improve the accuracy of the model's estimates.
    """

    return compute_loss_vola(
        X_obs=X_obs, Y_obs=Y_obs, Y_obs_bj=Y_obs_bj, n_obs_ot=n_obs_ot,
        batch_size=batch_size, eps=eps,
        weight=weight, M_obs=M_obs,
        compute_variance=compute_variance, var_weight=var_weight,
        Y_var_bj=Y_var_bj, Y_var=Y_var, dim_to=dim_to,
        which_var_loss=which_var_loss, version=2)


def compute_loss_drift_lim(
        X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
        weight=0.5, M_obs=None,
        compute_variance=None, var_weight=1.,
        Y_var_bj=None, Y_var=None, dim_to=None, which_var_loss=None,
        **kwargs):
    """
    loss function for learning the drift estimate of an Ito process for usage in
    a generative model.
    in particular, X_obs needs to be the increment process, i.e., the current
    (left-limit of the) increment since the last observation, divided by the
    time difference since last observation.

    X_obs = (X_t - X_tau)/tdiff
    tdiff = t - tau

    -> the loss trains the model to replicate X_obs before the jump (with
    Y_obs_bj) and does not penalize the jump directly. with suitable training
    data, this forces the model to jump to the right-limit of X_obs for
    tdiff->0, which is the instantaneous drift coefficient (if necessary
    information available, otherwise its best estimate).
    """

    return compute_loss_noisy_obs(
        X_obs=X_obs, Y_obs=Y_obs, Y_obs_bj=Y_obs_bj, n_obs_ot=n_obs_ot,
        batch_size=batch_size, eps=eps,
        weight=weight, M_obs=M_obs,
        compute_variance=compute_variance, var_weight=var_weight,
        Y_var_bj=Y_var_bj, Y_var=Y_var, dim_to=dim_to,
        which_var_loss=which_var_loss,)


def compute_loss_gen_coeffs(
        X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
        weight=0.5, M_obs=None,
        compute_variance=None, var_weight=1., tdiff=None,
        Y_var_bj=None, Y_var=None, dim_to=None, which_var_loss=None,
        **kwargs):
    """
    compute the loss for retrieving the coefficients of an Ito process for usage
    in a generative model.
    in particular, X_obs needs to be the increment process, i.e., the current
    (left-limit of the) increment since the last observation, divided by the
    time difference since last observation.

    X_obs = (X_t - X_tau)/tdiff
    tdiff = t - tau

    the loss has two components:

    1) | X_obs - Y_obs_bj |^2
      -> training Y_obs to approximate the drift
    2) | tdiff * (X_obs - Y_obs_bj.detach())^2 - Y_var_bj |^2
      -> trainineg Y_var to approximate the volatility; detach Y_obs s.t. it
         acts like a constant, i.e., like the value of the conditional
         expactation of X_obs (if Y_obs is trained well enough), but doesn't
         trigger backprop through it. we have to multiply by tdiff, because
         (after squaring) X_obs and Y_obs include a division by tdiff^2, while
         the vola estimate is only divided by tdiff.
         subtracting Y_obs_bj.detach() acts like a bias correction through the
         model's estimates itself (compare with compute_loss_vola_lim -> BIAS
         CORRECTION for more info).

    training both components with the noisy obs type loss, the model is forced
    (indirectly) to jump to the correct instantaneous drift and volatility
    values (if necessary information available, otherwise their best estimates),
    since those are the right-limits of the respective increment processes.
    (also see compute_loss_vola_lim).
    """
    if M_obs is None:
        M_obs = 1.
        M_obs_vol = 1.
    else:
        M_obs_vol = torch.matmul(M_obs.unsqueeze(2), M_obs.unsqueeze(1))
        M_obs_vol = M_obs_vol.view(-1, M_obs.shape[1] ** 2)

    tdiff = torch.from_numpy(tdiff).to(X_obs.device).reshape(-1, 1).repeat(
        1, X_obs.shape[1])

    d2 = Y_var_bj.shape[1]
    d = int(np.sqrt(d2))
    Y_var_bj_mat = Y_var_bj.view(-1, d, d)
    Y_var_bj_mat1 = torch.matmul(Y_var_bj_mat, Y_var_bj_mat.transpose(1, 2))
    Y_var_bj_mod = Y_var_bj_mat1.view(-1, d2)

    vol_target1 = torch.sqrt(tdiff) * (X_obs - Y_obs_bj.detach())
    vol_target = torch.matmul(vol_target1.unsqueeze(2),vol_target1.unsqueeze(1))
    vol_target_mod = vol_target.view(-1, d2)

    # part 1: drift
    inner1 = torch.sum(M_obs * (Y_obs_bj - X_obs) ** 2, dim=1)
    # part 2: volatility
    inner2 = var_weight * torch.sum(
        M_obs_vol * (Y_var_bj_mod - vol_target_mod) ** 2, dim=1)
    inner = inner1 + inner2
    outer = torch.sum(inner / n_obs_ot)

    return outer / batch_size




LOSS_FUN_DICT = {
    # dictionary of used loss functions.
    # Reminder inputs: (X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
    #   weight=0.5, M_obs=None)
    'standard': compute_loss,
    'easy': compute_loss_2,
    'very_easy': compute_loss_2_1,
    'IO': compute_loss_2_1,
    'noisy_obs': compute_loss_noisy_obs,
    'abs': compute_loss_3,
    'jump': compute_jump_loss,
    'vola': compute_loss_vola,
    'vola_lim': compute_loss_vola_lim,
    'drift_lim': compute_loss_drift_lim,
    'gen_coeffs': compute_loss_gen_coeffs,

    # Quantile loss functions, based on the quantile regression loss.
    #   They are called with the quantiles and return the corresponding
    #   loss function.
    #   All loss functions which have 'quantile' in their name, are treated as
    #   quantile loss functions (i.e., called with quantiles first; quantile
    #   need to be provided).
    'quantile': compute_quantile_loss,
    'quantile_jump': compute_quantile_jump_loss,
}




if __name__ == '__main__':
    pass
