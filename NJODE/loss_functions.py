"""
author: Florian Krach

implementation of the loss functions used in the NJODE model
"""



# ==============================================================================
import torch


# ==============================================================================
def compute_loss(X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
                 weight=0.5, M_obs=None):
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
    :return: torch.tensor (with the loss, reduced to 1 dim)
    """
    if M_obs is None:
        M_obs = 1.

    inner = (2 * weight * torch.sqrt(
        torch.sum(M_obs * (X_obs - Y_obs) ** 2, dim=1) + eps) +
             2 * (1 - weight) * torch.sqrt(
                torch.sum(M_obs * (Y_obs_bj - Y_obs)**2, dim=1) + eps))**2
    outer = torch.sum(inner / n_obs_ot)
    return outer / batch_size


def compute_loss_2(X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
                   weight=0.5, M_obs=None):
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
    return outer / batch_size


def compute_loss_2_1(X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
                   weight=0.5, M_obs=None):
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
    return outer / batch_size


def compute_loss_noisy_obs(X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size,
                           eps=1e-10, weight=0.5, M_obs=None):
    """
    similar to compute_loss, but only using the 2nd term of the original loss
    function, which enables training with noisy observations
    """
    if M_obs is None:
        M_obs = 1.

    inner = torch.sum(M_obs * (Y_obs_bj - X_obs) ** 2, dim=1)

    outer = torch.sum(inner / n_obs_ot)
    return outer / batch_size


def compute_loss_3(X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
                   weight=0.5, M_obs=None):
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
    return outer / batch_size


def compute_jump_loss(
        X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
        weight=0.5, M_obs=None):
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

    inner = torch.sum(M_obs * (X_obs - Y_obs) ** 2, dim=1)
    outer = torch.sum(inner / n_obs_ot)
    return outer / batch_size


def compute_quantile_loss(quantiles):
    """
    function to create a quantile loss function
    :param quantiles: list of float, the quantiles at which the loss should be
            computed
    :return: function, the loss function
    """
    def loss(X_obs, Y_obs, Y_obs_bj, n_obs_ot, batch_size, eps=1e-10,
             weight=0.5, M_obs=None):
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
             weight=0.5, M_obs=None):
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

        return torch.sum(outer) / batch_size

    return loss




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
