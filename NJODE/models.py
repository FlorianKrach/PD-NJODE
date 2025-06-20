"""
authors: Florian Krach & Marc Nuebel & Calypso Herrera

implementation of the model for NJ-ODE
"""

# =====================================================================================================================
import torch
import numpy as np
import os
import iisignature as sig
import sklearn
import scipy.linalg

from loss_functions import LOSS_FUN_DICT


# =====================================================================================================================
def init_weights(m, bias=0.0):  # initialize weights for model for linear NN
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(bias)


def save_checkpoint(model, optimizer, path, epoch, retrain_epoch=0):
    """
    save a trained torch model and the used optimizer at the given path, s.t.
    training can be resumed at the exact same point
    :param model: a torch model, e.g. instance of NJODE
    :param optimizer: a torch optimizer
    :param path: str, the path where to save the model
    :param epoch: int, the current epoch
    """
    if not os.path.exists(path):
        os.makedirs(path)
    filename = os.path.join(path, 'checkpt.tar')
    torch.save({'epoch': epoch,
                'weight': model.weight,
                'retrain_epoch': retrain_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               filename)


def get_ckpt_model(ckpt_path, model, optimizer, device):
    """
    load a saved torch model and its optimizer, inplace
    :param ckpt_path: str, path where the model is saved
    :param model: torch model instance, of which the weights etc. should be
            reloaded
    :param optimizer: torch optimizer, which should be loaded
    :param device: the device to which the model should be loaded
    """
    ckpt_path = os.path.join(ckpt_path, 'checkpt.tar')
    if not os.path.exists(ckpt_path):
        raise Exception("Checkpoint " + ckpt_path + " does not exist.")
    # Load checkpoint.
    checkpt = torch.load(ckpt_path)
    state_dict = checkpt['model_state_dict']
    optimizer.load_state_dict(checkpt['optimizer_state_dict'])
    model.load_state_dict(state_dict)
    model.epoch = checkpt['epoch']
    model.weight = checkpt['weight']
    if 'retrain_epoch' in checkpt:
        model.retrain_epoch = checkpt['retrain_epoch']
    if isinstance(model.readout_map, LinReg):
        model.readout_map.fitted = True
    model.to(device)



nonlinears = {  # dictionary of used non-linear activation functions. Reminder inputs
    'tanh': torch.nn.Tanh,
    'relu': torch.nn.ReLU,
    'prelu': torch.nn.PReLU,
    'softmax': torch.nn.Softmax,
}


def get_ffnn(input_size, output_size, nn_desc, dropout_rate, bias):
    """
    function to get a feed-forward neural network with the given description
    :param input_size: int, input dimension
    :param output_size: int, output dimension
    :param nn_desc: list of lists or None, each inner list defines one hidden
            layer and has 2 elements: 1. int, the hidden dim, 2. str, the
            activation function that should be applied (see dict nonlinears for
            possible options)
    :param dropout_rate: float,
    :param bias: bool, whether a bias is used in the layers
    :return: torch.nn.Sequential, the NN function
    """
    last_activation = None
    if isinstance(nn_desc, (tuple, list)) and len(nn_desc) > 0 \
            and isinstance(nn_desc[-1], str):
        last_activation = nn_desc[-1]
        nn_desc = nn_desc[:-1]  # remove last activation function from desc
    if nn_desc is not None and len(nn_desc) == 0:
        return torch.nn.Identity()
    if nn_desc is None or nn_desc[0] == "linear":  # if no NN desc given, or only linear
        layers = [torch.nn.Linear(in_features=input_size, out_features=output_size, bias=bias)]  # take linear NN if
        # not specified otherwise
    else:
        layers = [torch.nn.Linear(in_features=input_size, out_features=nn_desc[0][0], bias=bias)]  # first linear
        # layer to specified dimension
        if len(nn_desc) > 1:
            for i in range(len(nn_desc) - 1):  # add multiple layers if multiple were given as input
                layers.append(nonlinears[nn_desc[i][1]]())  # add layer with specified activation function
                layers.append(torch.nn.Dropout(p=dropout_rate))  # add dropout layer
                layers.append(
                    torch.nn.Linear(nn_desc[i][0], nn_desc[i + 1][0],  # add linear layer between specified dimensions
                                    bias=bias))
        layers.append(nonlinears[nn_desc[-1][1]]())  # last specified activation function
        layers.append(torch.nn.Dropout(p=dropout_rate))  # add another dropout layer
        layers.append(torch.nn.Linear(in_features=nn_desc[-1][0], out_features=output_size, bias=bias))  # linear
        # output layer
    if last_activation is not None:  # if a last activation function was specified, add it
        layers.append(nonlinears[last_activation]())
    return torch.nn.Sequential(*layers)  # return the constructed NN


# =====================================================================================================================
class ODEFunc(torch.nn.Module):
    """
    implementing continuous update between observatios, f_{\theta} in paper
    """

    def __init__(self, input_size, hidden_size, ode_nn, dropout_rate=0.0,
                 bias=True, input_current_t=False, input_sig=False,
                 sig_depth=3, coord_wise_tau=False, input_scaling_func="tanh",
                 use_current_y_for_ode=False, input_var_t_helper=False):
        super().__init__()  # initialize class with given parameters
        self.input_current_t = input_current_t
        self.input_sig = input_sig
        self.sig_depth = sig_depth
        self.use_current_y_for_ode = use_current_y_for_ode
        self.input_var_t_helper = input_var_t_helper
        if input_scaling_func in ["id", "identity"]:
            self.sc_fun = torch.nn.Identity()
            print("neuralODE use input scaling with identity (no scaling)")
        else:
            self.sc_fun = torch.tanh
            print("neuralODE use input scaling with tanh")

        # create feed-forward NN, f(H,X,tau,t-tau)
        if coord_wise_tau:
            add = 2*input_size
        else:
            add = 2
        if input_current_t:
            if coord_wise_tau:
                add += input_size
            else:
                add += 1
        if input_var_t_helper:
            if coord_wise_tau:
                add += input_size
            else:
                add += 1
        if input_sig:
            add += sig_depth
        if use_current_y_for_ode:
            add += input_size
        self.f = get_ffnn(  # get a feedforward NN with the given specifications
            input_size=input_size + hidden_size + add, output_size=hidden_size,
            nn_desc=ode_nn, dropout_rate=dropout_rate, bias=bias
        )

    def forward(self, x, h, tau, tdiff, signature=None, current_y=None,
                delta_t=None):
        # dimension should be (batch, input_size) for x, (batch, hidden) for h, 
        #    (batch, 1) for times

        input_f = torch.cat([self.sc_fun(x), self.sc_fun(h), tau, tdiff], dim=1)

        if self.input_current_t:
            input_f = torch.cat([input_f, tau+tdiff], dim=1)
        if self.input_var_t_helper:
            input_f = torch.cat([input_f, 1/torch.sqrt(tdiff+delta_t)], dim=1)
        if self.input_sig:
            input_f = torch.cat([input_f, signature], dim=1)
        if self.use_current_y_for_ode:
            input_f = torch.cat([input_f, self.sc_fun(current_y)], dim=1)

        df = self.f(input_f)
        return df


class GRUCell(torch.nn.Module):
    """
    Implements discrete update based on the received observations, \rho_{\theta}
    in paper
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.gru_d = torch.nn.GRUCell(input_size=input_size, hidden_size=hidden_size, bias=bias)

        self.input_size = input_size

    def forward(self, h, X_obs, i_obs):
        temp = h.clone()
        temp[i_obs] = self.gru_d(X_obs, h[i_obs])
        h = temp
        return h


class LinReg(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinReg, self).__init__()
        self.input_dim = input_size
        self.output_dim = output_size
        self.weights = torch.nn.parameter.Parameter(
            torch.empty((self.input_dim+1, self.output_dim)),
            requires_grad=False)
        self.fitted = False

    def forward(self, input):
        if not self.fitted:
            raise ValueError("LinReg has to be fitted first")
        x = torch.cat([torch.ones((input.shape[0],1)), input], dim=1)
        return torch.matmul(x, self.weights)

    def fit(self, X, y):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        X = np.concatenate([np.ones((X.shape[0],1)), X], axis=1)
        assert y.shape[1] == self.output_dim
        for i in range(self.output_dim):
            yi = y[:, i]
            whichnan = np.isnan(yi)
            p, _, _, _ = scipy.linalg.lstsq(X[~whichnan], yi[~whichnan])
            self.weights.data[:, i] = torch.Tensor(p.squeeze())
        self.fitted = True


class FFNN(torch.nn.Module):
    """
    Implements feed-forward neural networks with tanh applied to inputs and the
    option to use a residual NN version
    """

    def __init__(self, input_size, output_size, nn_desc, dropout_rate=0.0,
                 bias=True, residual=False, masked=False, recurrent=False,
                 input_sig=False, sig_depth=3, clamp=None, input_t=False,
                 t_size=None, nb_outputs=None,
                 **kwargs):
        super().__init__()

        self.use_lstm = False
        if nn_desc is not None and isinstance(nn_desc[0][0], str) \
                and nn_desc[0][0].lower() == "lstm":
            self.use_lstm = True
            print("USE LSTM")
        in_size = input_size
        if masked:
            in_size = 2 * input_size
        if recurrent and not self.use_lstm:
            in_size += output_size
        if input_sig:
            in_size += sig_depth
        self.masked = masked
        self.recurrent = recurrent
        self.output_size = output_size
        self.nb_outputs = nb_outputs
        if self.nb_outputs is None:
            self.nb_outputs = 1
        self.input_sig = input_sig
        self.sig_depth = sig_depth
        self.input_t = input_t
        if self.input_t:
            in_size += t_size
        self.clamp = clamp
        self.lstm = None
        if self.use_lstm:
            self.lstm = torch.nn.LSTMCell(
                input_size=in_size, hidden_size=nn_desc[0][1], bias=bias)
            self.c_h = None
            in_size = nn_desc[0][1]*2
            assert in_size == output_size, \
                "when using an LSTM, the hidden_size has to be 2* " \
                "the LSTM output size"
            nn_desc = nn_desc[1:]
        self.ffnn = get_ffnn(
            input_size=in_size, output_size=self.output_size*self.nb_outputs,
            nn_desc=nn_desc, dropout_rate=dropout_rate, bias=bias)

        if residual:
            print('use residual network: input_size={}, output_size={}'.format(
                input_size, output_size))
            if input_size <= output_size:
                self.case = 1
            if input_size > output_size:
                self.case = 2
        else:
            self.case = 0

    def forward(self, nn_input, mask=None, sig=None, h=None, t=None):
        identity = None
        if self.case == 1:
            identity = torch.zeros((nn_input.shape[0], self.output_size)).to(
                self.device)
            identity[:, 0:nn_input.shape[1]] = nn_input
        elif self.case == 2:
            identity = nn_input[:, 0:self.output_size]

        if self.recurrent or self.use_lstm:
            assert h is not None
            # x = torch.tanh(nn_input)
            x = nn_input
        else:
            x = torch.tanh(nn_input)  # maybe not helpful
        if self.recurrent and not self.use_lstm:
            x = torch.cat((x, h), dim=1)
        if self.input_t:
            x = torch.cat((x, t), dim=1)
        if self.masked:
            assert mask is not None
            x = torch.cat((x, mask), dim=1)
        if self.input_sig:
            assert sig is not None
            x = torch.cat((x, sig), dim=1)
        if self.use_lstm:
            h_, c_ = torch.chunk(h, chunks=2, dim=1)
            h_, c_ = self.lstm(x.float(), (h_, c_))
            x = torch.concat((h_, c_), dim=1)
        out = self.ffnn(x.float())

        if self.nb_outputs > 1:
            out = out.reshape(-1, self.output_size, self.nb_outputs)
            if self.case > 0:
                identity = identity.reshape(-1, self.output_size, 1).repeat(
                    1, 1, self.nb_outputs)

        if self.case == 0:
            pass
        else:
            out = identity + out

        if self.clamp is not None:
            out = torch.clamp(out, min=-self.clamp, max=self.clamp)

        return out

    @property
    def device(self):
        device = next(self.parameters()).device
        return device



class NJODE(torch.nn.Module):
    """
    NJ-ODE model
    """
    def __init__(  # initialize the class by naming relevant features
            self, input_size, hidden_size, output_size,
            ode_nn, readout_nn, enc_nn, use_rnn,
            bias=True, dropout_rate=0, solver="euler",
            weight=0.5, weight_decay=1.,
            input_coords=None, output_coords=None,
            signature_coords=None, compute_variance=None, var_size=0,
            **options
    ):
        """
        init the model
        :param input_size: int
        :param hidden_size: int, size of latent variable process
        :param output_size: int
        :param ode_nn: list of list, defining the NN f, see get_ffnn
        :param readout_nn: list of list, defining the NN g, see get_ffnn
        :param enc_nn: list of list, defining the NN e, see get_ffnn
        :param use_rnn: bool, whether to use the RNN for 'jumps'
        :param bias: bool, whether to use a bias for the NNs
        :param dropout_rate: float
        :param solver: str, specifying the ODE solver, suppoorted: {'euler'}
        :param weight: float in [0.5, 1], the initial weight used in the loss
        :param weight_decay: float in [0,1], the decay applied to the weight of
                the loss function after each epoch, decaying towards 0.5
                    1: no decay, weight stays the same
                    0: immediate decay to 0.5 after 1st epoch
                    (0,1): exponential decay towards 0.5
        :param level: level for signature transform
        :param input_coords: list of int, the coordinates of the input
        :param output_coords: list of int, the coordinates of the output
        :param signature_coords: list of int, the coordinates of the signature
        :param compute_variance: None or one of {"variance", "covariance"},
                whether to compute the (marginal) variance or covariance matrix
        :param var_size: int, the size of the model variance estimate; this is
                already included in the output_size, but the variance
                coordinates are not included in the output_coords
        :param options: kwargs, used:
                - "classifier_nn"
                - "options" with arg a dict passed from train.train
                    used kwords: 'which_loss', 'residual_enc_dec',
                    'residual_enc', 'residual_dec',
                    'masked', 'input_current_t', 'input_sig', 'level',
                    'use_y_for_ode', 'enc_input_t', 'use_current_y_for_ode',
                    'use_observation_as_input', 'coord_wise_tau', 'clamp',
                    'ode_input_scaling_func', 'use_sig_for_classifier',
                    'classifier_loss_weight'
        """
        super().__init__()  # super refers to base class, init initializes

        self.epoch = 1
        self.retrain_epoch = 0
        self.weight = weight
        self.weight_decay = weight_decay
        self.use_rnn = use_rnn  # use RNN for jumps
        self.input_coords = input_coords
        self.output_coords = output_coords
        self.signature_coords = signature_coords
        self.compute_variance = compute_variance
        self.var_size = var_size
        self.var_weight = 1.
        self.which_var_loss = None


        # get options from the options of train input
        options1 = options['options']
        if 'which_loss' in options1:
            self.which_loss = options1['which_loss']
        else:
            self.which_loss = 'standard'  # otherwise take the standard loss
        assert self.which_loss in LOSS_FUN_DICT
        print('using loss: {}'.format(self.which_loss))
        self.loss_quantiles = None
        self.nb_quantiles = None
        if "quantile" in self.which_loss:
            self.loss_quantiles = options1['loss_quantiles']
            self.nb_quantiles = len(self.loss_quantiles)
            print("using quantile loss with quantiles:", self.loss_quantiles)
        if "var_weight" in options1:
            self.var_weight = options1['var_weight']
            print("using variance loss weight:", self.var_weight)
        if "which_var_loss" in options1:
            self.which_var_loss = options1['which_var_loss']
            print("using variance loss:", self.which_var_loss)

        self.residual_enc = True
        self.residual_dec = True
        # for backward compatibility, set residual_enc to False as default
        #   if RNN is used. (before, it was not possible to use residual
        #   connections with RNNs)
        if self.use_rnn:
            self.residual_enc = False
        if 'residual_enc_dec' in options1:
            residual_enc_dec = options1['residual_enc_dec']
            self.residual_enc = residual_enc_dec
            self.residual_dec = residual_enc_dec
        if 'residual_enc' in options1:
            self.residual_enc = options1['residual_enc']
        if 'residual_dec' in options1:
            self.residual_dec = options1['residual_dec']

        self.input_current_t = False
        if 'input_current_t' in options1:
            self.input_current_t = options1['input_current_t']
        self.input_var_t_helper = False
        if 'input_var_t_helper' in options1:
            self.input_var_t_helper = options1['input_var_t_helper']
        self.input_sig = False
        if 'input_sig' in options1:
            self.input_sig = options1['input_sig']
        self.level = 2
        if 'level' in options1:
            self.level = options1['level']
        self.sig_depth = sig.siglength(len(self.signature_coords)+1, self.level)
        self.masked = False
        if 'masked' in options1:
            self.masked = options1['masked']
        self.use_y_for_ode = False
        if 'use_y_for_ode' in options1:
            self.use_y_for_ode = options1['use_y_for_ode']
        self.use_current_y_for_ode = False
        if 'use_current_y_for_ode' in options1:
            self.use_current_y_for_ode = options1['use_current_y_for_ode']
        if self.nb_quantiles is not None and self.nb_quantiles > 1:
            assert self.use_current_y_for_ode is False, \
                "Quantile loss not implemented for use_current_y_for_ode"
            assert self.use_y_for_ode is False, \
                "Quantile loss not implemented for use_y_for_ode"
        self.coord_wise_tau = False
        if 'coord_wise_tau' in options1 and self.masked:
            self.coord_wise_tau = options1['coord_wise_tau']
        self.enc_input_t = False
        if 'enc_input_t' in options1:
            self.enc_input_t = options1['enc_input_t']
        self.clamp = None
        if 'clamp' in options1:
            self.clamp = options1['clamp']
        self.ode_input_scaling_func = "tanh"
        if 'ode_input_scaling_func' in options1:
            self.ode_input_scaling_func = options1['ode_input_scaling_func']
        classifier_dict = None
        if 'classifier_dict' in options:
            classifier_dict = options["classifier_dict"]
        self.use_sig_for_classifier = False
        if 'use_sig_for_classifier' in options1:
            self.use_sig_for_classifier = options1['use_sig_for_classifier']
        self.class_loss_weight = 1.
        self.loss_weight = 1.
        if 'classifier_loss_weight' in options1:
            class_loss_weight = options1['classifier_loss_weight']
            if class_loss_weight == np.infty:
                self.class_loss_weight = 1.
                self.loss_weight = 0.
            else:
                self.class_loss_weight = class_loss_weight
        t_size = 2
        if self.coord_wise_tau:
            t_size = 2*input_size
        use_observation_as_input = None
        if 'use_observation_as_input' in options1:
            use_observation_as_input = options1['use_observation_as_input']
        if use_observation_as_input is None:
            self.use_observation_as_input = lambda x: True
        elif isinstance(use_observation_as_input, bool):
            self.use_observation_as_input = \
                lambda x: use_observation_as_input
        elif isinstance(use_observation_as_input, float):
            self.use_observation_as_input = \
                lambda x: np.random.random() < use_observation_as_input
        elif isinstance(use_observation_as_input, str):
            self.use_observation_as_input = \
                eval(use_observation_as_input)
        val_use_observation_as_input = None
        if 'val_use_observation_as_input' in options1:
            val_use_observation_as_input = \
                options1['val_use_observation_as_input']
        if val_use_observation_as_input is None:
            self.val_use_observation_as_input = self.use_observation_as_input
        elif isinstance(val_use_observation_as_input, bool):
            self.val_use_observation_as_input = \
                lambda x: val_use_observation_as_input
        elif isinstance(val_use_observation_as_input, float):
            self.val_use_observation_as_input = \
                lambda x: np.random.random() < val_use_observation_as_input
        elif isinstance(val_use_observation_as_input, str):
            self.val_use_observation_as_input = \
                eval(val_use_observation_as_input)

        self.ode_f = ODEFunc(
            input_size=input_size, hidden_size=hidden_size, ode_nn=ode_nn,
            dropout_rate=dropout_rate, bias=bias,
            input_current_t=self.input_current_t, input_sig=self.input_sig,
            sig_depth=self.sig_depth, coord_wise_tau=self.coord_wise_tau,
            input_scaling_func=self.ode_input_scaling_func,
            use_current_y_for_ode=self.use_current_y_for_ode,
            input_var_t_helper=self.input_var_t_helper)
        self.encoder_map = FFNN(
            input_size=input_size, output_size=hidden_size, nn_desc=enc_nn,
            dropout_rate=dropout_rate, bias=bias, recurrent=self.use_rnn,
            masked=self.masked, residual=self.residual_enc,
            input_sig=self.input_sig, sig_depth=self.sig_depth,
            input_t=self.enc_input_t, t_size=t_size)
        self.readout_map = FFNN(
            input_size=hidden_size, output_size=output_size, nn_desc=readout_nn,
            dropout_rate=dropout_rate, bias=bias,
            residual=self.residual_dec, clamp=self.clamp,
            nb_outputs=self.nb_quantiles)
        self.get_classifier(classifier_dict=classifier_dict)

        self.solver = solver
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.apply(init_weights)

    @property
    def device(self):
        device = next(self.parameters()).device
        return device

    def get_classifier(self, classifier_dict):
        self.classifier = None
        self.SM = None
        self.CEL = None
        if classifier_dict is not None:
            if self.use_sig_for_classifier:
                classifier_dict['input_size'] += self.sig_depth
            self.classifier = FFNN(**classifier_dict)
            self.SM = torch.nn.Softmax(dim=1)
            self.CEL = torch.nn.CrossEntropyLoss()

    def weight_decay_step(self):
        inc = (self.weight - 0.5)
        self.weight = 0.5 + inc * self.weight_decay
        return self.weight

    def ode_step(self, h, delta_t, current_time, last_X, tau, signature=None,
                 current_y=None):
        """Executes a single ODE step"""
        if not self.input_sig:
            signature = None
        if self.solver == "euler":
            h = h + delta_t * self.ode_f(
                x=last_X, h=h, tau=tau, tdiff=current_time - tau,
                signature=signature, current_y=current_y, delta_t=delta_t)
        else:
            raise ValueError("Unknown solver '{}'.".format(self.solver))

        current_time += delta_t
        return h, current_time

    def recreate_data(self, times, time_ptr, X, obs_idx, start_X):
        """
        recreates matrix of all observations
        first dim: which data path
        second dim: which time
        """
        # shape: [batch_size, time_steps+1, dimension]
        data = np.empty(shape=(start_X.shape[0], 1+len(times), start_X.shape[1]))
        data[:] = np.nan
        data[:,0,:] = start_X.detach().cpu().numpy()

        X = X.detach().cpu().numpy()
        for j, time in enumerate(times):
            start = time_ptr[j]
            end = time_ptr[j + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]
            data[i_obs, j+1, :] = X_obs
        times_new = np.concatenate(([0], times), axis=0)

        return times_new, data

    def get_signature(self, times, time_ptr, X, obs_idx, start_X):
        """
        Input: See forward
        Returns: signature of paths as nested list
        """
        # reconstructing the data, shape: [batch_size, time_steps+1, dim]
        times_new, data = self.recreate_data(
            times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx,
            start_X=start_X)

        # list of list of lists, shape: [batch_size, obs_dates[j], sig_length]
        signature = []
        for j in range(data.shape[0]):  # iterate over batch
            data_j = data[j, :, :]
            observed_j = []
            for i in range(data_j.shape[0]):
                # if the current batch-sample has an observation at the current
                #   time, add it to the list of observations
                if not np.all(np.isnan(data_j[i])):
                    observed_j += [i]
            data_j = data_j[observed_j, :]

            # replace no observations with last observation
            for i in range(1, data_j.shape[0]):
                # # OLD VERSION (SLOW)
                # for k in range(data_j.shape[1]):
                #     if np.isnan(data_j[i, k]):
                #         data_j[i, k] = data_j[i-1, k]
                ks = np.isnan(data_j[i, :])
                data_j[i, ks] = data_j[i-1, ks]

            times_j = times_new[observed_j].reshape(-1, 1)
            # add times to data for signature call
            path_j = np.concatenate((times_j, data_j), axis=1)
            # the following computes the signatures of all partial paths, from
            #   start to each point of the path
            signature.append(sig.sig(path_j, self.level, 2))

        return signature

    def forward(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                n_obs_ot, return_path=False, get_loss=True, until_T=False,
                M=None, start_M=None, which_loss=None, dim_to=None,
                predict_labels=None, return_classifier_out=False,
                return_at_last_obs=False, compute_variance_loss=True):
        """
        the forward run of this module class, used when calling the module
        instance without a method
        :param times: np.array, of observation times
        :param time_ptr: list, start indices of X and obs_idx for a given
                observation time, first element is 0, this pointer tells how
                many (and which) of the observations of X along the batch-dim
                belong to the current time, and obs_idx then tells to which of
                the batch elements they belong. In particular, not each batch-
                element has to jump at the same time, and only those elements
                which jump at the current time should be updated with a jump
        :param X: torch.tensor, data tensor
        :param obs_idx: list, index of the batch elements where jumps occur at
                current time
        :param delta_t: float, time step for Euler
        :param T: float, the final time
        :param start_X: torch.tensor, the starting point of X
        :param n_obs_ot: torch.tensor, the number of observations over the
                entire time interval for each element of the batch
        :param return_path: bool, whether to return the path of h
        :param get_loss: bool, whether to compute the loss, otherwise 0 returned
        :param until_T: bool, whether to continue until T (for eval) or only
                until last observation (for training)
        :param M: None or torch.tensor, if not None: the mask for the data, same
                size as X, with 0 or 1 entries
        :param start_M: None or torch.tensor, if not None: the mask for start_X,
                same size as start_X
        :param which_loss: see train.train, to overwrite which loss for eval
        :param dim_to: None or int, if given not all coordinates along the
                data-dimension axis are used but only up to dim_to. this can be
                used if func_appl_X is used in train, but the loss etc. should
                only be computed for the original coordinates (without those
                resulting from the function applications)
        :param predict_labels: None or torch.tensor with the true labels to
                predict
        :param return_classifier_out: bool, whether to return the output of the
                classifier
        :param return_at_last_obs: bool, whether to return the hidden state at
                the last observation time or at the final time
        :param epoch: int, the current epoch
        :param compute_variance_loss: bool, whether to compute the variance
                loss (in case self.compute_variance is not None). default: True

        :return: torch.tensor (hidden state at final time), torch.tensor (loss),
                    if wanted the paths of t (np.array) and h, y (torch.tensors)
        """
        if which_loss is None:
            which_loss = self.which_loss
        if 'quantile' in which_loss:
            LOSS = LOSS_FUN_DICT[which_loss](self.loss_quantiles)
        else:
            LOSS = LOSS_FUN_DICT[which_loss]

        last_X = start_X
        batch_size = start_X.size()[0]
        data_dim = start_X.size()[1]
        if dim_to is None:
            dim_to = len(self.output_coords)
        out_coords = self.output_coords[:dim_to]

        impute = False
        if (len(self.input_coords) == len(self.output_coords) and
                np.all(self.input_coords == self.output_coords) and
                self.loss_quantiles is None):
            impute = True
        if not impute and self.use_y_for_ode:
            raise ValueError(
                "use_y_for_ode can only be used when imputation is possible, "
                "i.e., when input and output coordinates are the same")


        if self.coord_wise_tau:
            tau = torch.tensor([[0.0]]).repeat(batch_size, self.input_size).to(
                self.device)
        else:
            tau = torch.tensor([[0.0]]).repeat(batch_size, 1).to(
                self.device)
        current_time = 0.0
        loss = torch.tensor(0.).to(self.device)
        c_sig = None

        if (self.input_sig or self.use_sig_for_classifier):
            if X.shape[0] == 0:  # if no data, set signature to 0
                pass
            elif self.masked:
                Mdc = M.clone()
                Mdc[Mdc==0] = np.nan
                X_obs_impute = X * Mdc
                signature = self.get_signature(
                    times=times, time_ptr=time_ptr,
                    X=X_obs_impute[:, self.signature_coords],
                    obs_idx=obs_idx, start_X=start_X[:, self.signature_coords])
            else:
                signature = self.get_signature(
                    times=times, time_ptr=time_ptr,
                    X=X[:, self.signature_coords],
                    obs_idx=obs_idx, start_X=start_X[:, self.signature_coords])

            # in beginning, no path was observed => set sig to 0
            current_sig = np.zeros((batch_size, self.sig_depth))
            current_sig_nb = np.zeros(batch_size).astype(int)
            c_sig = torch.from_numpy(current_sig).float().to(self.device)

        if self.masked:
            if start_M is None:
                start_M = torch.ones_like(start_X)
                start_M = start_M[:, self.input_coords]
        else:
            start_M = None

        h = self.encoder_map(
            start_X[:, self.input_coords], mask=start_M,
            sig=c_sig,
            h=torch.zeros((batch_size, self.hidden_size)).to(self.device),
            t=torch.cat((tau, current_time - tau), dim=1).to(self.device))
        # if self.encoder_map.use_lstm:
        #     self.c_ = torch.chunk(h.clone(), chunks=2, dim=1)[1]

        if return_path:
            path_t = [0]
            path_h = [h]
            y = self.readout_map(h)
            if self.var_size > 0:
                path_y = [y[:, :-self.var_size]]
                path_var = [y[:, -self.var_size:]]
            else:
                path_y = [y]
                path_var = None
        h_at_last_obs = h.clone()
        sig_at_last_obs = c_sig

        assert len(times) + 1 == len(time_ptr)

        for i, obs_time in enumerate(times):
            # Propagation of the ODE until next observation
            while current_time < (obs_time - 1e-10 * delta_t):  # 0.0001 delta_t used for numerical consistency.
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time
                if self.solver == 'euler':
                    h, current_time = self.ode_step(
                        h, delta_t_, current_time,
                        last_X=last_X[:, self.input_coords], tau=tau,
                        signature=c_sig, current_y=self.readout_map(h))
                    current_time_nb = int(round(current_time / delta_t))
                else:
                    raise NotImplementedError

                # Storing the predictions.
                if return_path:
                    path_t.append(current_time)
                    path_h.append(h)
                    y = self.readout_map(h)
                    if self.var_size > 0:
                        path_y.append(y[:, :-self.var_size])
                        path_var.append(y[:, -self.var_size:])
                    else:
                        path_y.append(y)

            # Reached an observation - only update those elements of the batch, 
            #    for which an observation is made
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]
            if self.masked:
                if isinstance(M, np.ndarray):
                    M_obs = torch.from_numpy(M[start:end]).to(self.device)
                else:
                    M_obs = M[start:end]
                M_obs_in = M_obs[:, self.input_coords]
                M_obs_out = M_obs[:, out_coords]
                M_obs_sig = M_obs[:, self.signature_coords]
            else:
                M_obs = None
                M_obs_in = None
                M_obs_out = None
                M_obs_sig = None

            # decide whether to use observation as input
            if self.training:  # check whether model is in training or eval mode
                use_as_input = self.use_observation_as_input(self.epoch)
            else:
                use_as_input = self.val_use_observation_as_input(self.epoch)

            # update signature
            if self.input_sig or self.use_sig_for_classifier:
                for ij, j in enumerate(i_obs):
                    # the signature is updated only if one of the sig-coords is
                    #   observed -> hence, it can happen that even though the
                    #   j-th batch sample is observed, the signature is not
                    #   updated because none of the sig-coords is observed
                    if M_obs_sig is None or M_obs_sig[ij].sum() > 0:
                        current_sig[j, :] = signature[j][current_sig_nb[j]]
                        current_sig_nb[j] += 1
                if use_as_input:
                    # TODO: this is not fully correct, since if we didn't
                    #   use some intermediate observations, the signature still
                    #   has their information when using the signature up to
                    #   some later observation. However, this just means that
                    #   during training, the model conditions on a (slightly)
                    #   different sigma-algebra (if the signature is used), but
                    #   for inference the model should still work correctly.
                    #   Especially, if we are interested in predicting
                    #   \hat{X}_{t,s}
                    c_sig = torch.from_numpy(current_sig).float().to(
                        self.device)

            # Using RNNCell to update h. Also updating loss, tau and last_X
            Y_bj = self.readout_map(h)
            if use_as_input:
                X_obs_impute = X_obs
                temp = h.clone()
                if self.masked:
                    if impute:
                        # self imputation only possible if input and output are
                        #    the same and no quantile loss is used
                        X_obs_impute = X_obs * M_obs + (torch.ones_like(
                            M_obs.long()) - M_obs) * Y_bj[i_obs.long(), :data_dim]
                    else:
                        # otherwise set all masked entries to last value of X
                        X_obs_impute = X_obs * M_obs + (1-M_obs) * last_X
                c_sig_iobs = None
                if self.input_sig:
                    c_sig_iobs = c_sig[i_obs]
                temp[i_obs.long()] = self.encoder_map(
                    X_obs_impute[:, self.input_coords],
                    mask=M_obs_in, sig=c_sig_iobs, h=h[i_obs],
                    t=torch.cat((tau[i_obs], current_time - tau[i_obs]), dim=1))
                h = temp
                Y = self.readout_map(h)

                # update h and sig at last observation
                h_at_last_obs[i_obs.long()] = h[i_obs.long()].clone()
                sig_at_last_obs = c_sig
            else:
                Y = Y_bj

            if get_loss:
                Y_var_bj = None
                Y_var = None
                if self.var_size > 0:
                    Y_var_bj = Y_bj[i_obs.long(), -self.var_size:]
                    Y_var = Y[i_obs.long(), -self.var_size:]

                # INFO: X_obs has input and output coordinates, out_coords only
                #   has the output coordinates until dim_to; Y_obs has only the
                #   output coordinates (+ the var coords appended in the end),
                #   so taking them until dim_to (which is at max the size of the
                #   output_coords) corresponds to the out_coords
                if compute_variance_loss:
                    compute_variance = self.compute_variance
                else:
                    compute_variance = None
                loss = loss + LOSS(
                    X_obs=X_obs[:, out_coords], Y_obs=Y[i_obs.long(), :dim_to],
                    Y_obs_bj=Y_bj[i_obs.long(), :dim_to],
                    n_obs_ot=n_obs_ot[i_obs.long()], batch_size=batch_size,
                    weight=self.weight, M_obs=M_obs_out,
                    compute_variance=compute_variance,
                    var_weight=self.var_weight,
                    Y_var_bj=Y_var_bj, Y_var=Y_var, dim_to=dim_to,
                    which_var_loss=self.which_var_loss)

            # make update of last_X and tau, that is not inplace 
            #    (otherwise problems in autograd)
            if use_as_input:
                temp_X = last_X.clone()
                temp_tau = tau.clone()
                if self.use_y_for_ode:
                    temp_X[i_obs.long()] = Y[i_obs.long(), :data_dim]
                else:
                    temp_X[i_obs.long()] = X_obs_impute
                if self.coord_wise_tau:
                    _M = torch.zeros_like(temp_tau)
                    _M[i_obs] = M_obs[:, self.input_coords]
                    temp_tau[_M==1] = obs_time.astype(np.float64)
                else:
                    temp_tau[i_obs.long()] = obs_time.astype(np.float64)
                last_X = temp_X
                tau = temp_tau

            if return_path:
                path_t.append(obs_time)
                path_h.append(h)
                if self.var_size > 0:
                    path_y.append(Y[:, :-self.var_size])
                    path_var.append(Y[:, -self.var_size:])
                else:
                    path_y.append(Y)

        # after last observation has been processed, apply classifier if wanted
        cl_out = None
        if self.classifier is not None and predict_labels is not None:
            cl_loss = torch.tensor(0.)
            cl_input = h_at_last_obs
            if self.use_sig_for_classifier:
                cl_input = torch.cat([cl_input, sig_at_last_obs], dim=1)
            cl_out = self.classifier(cl_input)
            cl_loss = cl_loss + self.CEL(
                input=cl_out, target=predict_labels[:, 0])
            loss = [self.loss_weight*loss + self.class_loss_weight*cl_loss,
                    loss, cl_loss]

        # after every observation has been processed, propagating until T
        if until_T:
            while current_time < T - 1e-10 * delta_t:
                if current_time < T - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = T - current_time
                if self.solver == 'euler':
                    h, current_time = self.ode_step(
                        h, delta_t_, current_time, last_X=last_X, tau=tau,
                        signature=c_sig, current_y=self.readout_map(h))
                else:
                    raise NotImplementedError

                # Storing the predictions.
                if return_path:
                    path_t.append(current_time)
                    path_h.append(h)
                    y = self.readout_map(h)
                    if self.var_size > 0:
                        path_y.append(y[:, :-self.var_size])
                        path_var.append(y[:, -self.var_size:])
                    else:
                        path_y.append(y)

        if return_at_last_obs:
            return h_at_last_obs, sig_at_last_obs
        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            var_path = None
            if self.var_size > 0:
                var_path = torch.stack(path_var)
            if return_classifier_out:
                return h, loss, np.array(path_t), torch.stack(path_h), \
                       torch.stack(path_y)[:, :, :dim_to], var_path, cl_out
            return h, loss, np.array(path_t), torch.stack(path_h), \
                   torch.stack(path_y)[:, :, :dim_to], var_path
        else:
            if return_classifier_out and self.classifier is not None:
                return h, loss, cl_out
            return h, loss

    def evaluate(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                 n_obs_ot, stockmodel, cond_exp_fun_kwargs=None,
                 diff_fun=lambda x, y: np.nanmean((x - y) ** 2),
                 return_paths=False, M=None, true_paths=None, start_M=None,
                 true_mask=None, mult=None, use_stored_cond_exp=False,):
        """
        evaluate the model at its current training state against the true
        conditional expectation
        :param times: see forward
        :param time_ptr: see forward
        :param X: see forward
        :param obs_idx: see forward
        :param delta_t: see forward
        :param T: see forward
        :param start_X: see forward
        :param n_obs_ot: see forward
        :param stockmodel: stock_model.StockModel instance, used to compute true
                cond. exp.
        :param cond_exp_fun_kwargs: dict, the kwargs for the cond. exp. function
                currently not used
        :param diff_fun: function, to compute difference between optimal and
                predicted cond. exp
        :param return_paths: bool, whether to return also the paths
        :param M: see forward
        :param start_M: see forward
        :param true_paths: np.array, shape [batch_size, dimension, time_steps+1]
        :param true_mask: as true_paths, with mask entries
        :param mult: None or int, if given not all coordinates along the
                data-dimension axis are used but only up to dim/mult. this can be
                used if func_appl_X is used in train, but the loss etc. should
                only be computed for the original coordinates (without those
                resulting from the function applications)
        :param use_stored_cond_exp: bool, whether to recompute the cond. exp.

        :return: eval-loss, if wanted paths t, y for true and pred
        """
        self.eval()

        dim = start_X.shape[1]
        dim_to = dim
        output_dim_to = len(self.output_coords)
        if mult is not None and mult > 1:
            dim_to = round(dim/mult)
            output_dim_to = round(len(self.output_coords)/mult)

        _, _, path_t, path_h, path_y, path_var = self.forward(
            times, time_ptr, X, obs_idx, delta_t, T, start_X, None,
            return_path=True, get_loss=False, until_T=True, M=M,
            start_M=start_M, dim_to=output_dim_to)

        if true_paths is None:
            if M is not None:
                M = M.detach().cpu().numpy()[:, :dim_to]
            if X.shape[0] > 0:  # if no data (eg. bc. obs_perc=0, not possible)
                X = X.detach().cpu().numpy()[:, :dim_to]
            _, true_path_t, true_path_y = stockmodel.compute_cond_exp(
                times, time_ptr, X,
                obs_idx.detach().cpu().numpy(),
                delta_t, T, start_X.detach().cpu().numpy()[:, :dim_to],
                n_obs_ot.detach().cpu().numpy(),
                return_path=True, get_loss=False, M=M,
                store_and_use_stored=use_stored_cond_exp)
        else:
            true_t = np.linspace(0, T, true_paths.shape[2])
            which_t_ind = []
            for t in path_t:
                which_t_ind.append(np.argmin(np.abs(true_t - t)))
            # INFO: first get the correct output coordinate, then the correct
            #   time index; afterwards transpose to [time, batch_size, dim]
            true_path_y = true_paths[:, self.output_coords[:output_dim_to], :][
                :, :, which_t_ind]
            true_path_y = np.transpose(true_path_y, axes=(2, 0, 1))
            true_path_t = true_t[which_t_ind]

        if path_y.detach().cpu().numpy().shape == true_path_y.shape:
            eval_metric = diff_fun(path_y.detach().cpu().numpy(), true_path_y)
        else:
            print(path_y.detach().cpu().numpy().shape)
            print(true_path_y.shape)
            raise ValueError("Shapes do not match!")
        if return_paths:
            return eval_metric, path_t, true_path_t, path_y, true_path_y
        else:
            return eval_metric

    def evaluate_LOB(
            self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
            n_obs_ot, return_paths=False, predict_times=None,
            true_predict_vals=None, true_predict_labels=None, true_samples=None,
            normalizing_mean=0., normalizing_std=1., eval_predict_steps=None,
            thresholds=None, predict_labels=None,
            coord_to_compare=(0,), class_report=False):
        """
        evaluate the model at its current training state for the LOB dataset

        :param times: see forward
        :param time_ptr: see forward
        :param X: see forward
        :param obs_idx: see forward
        :param delta_t: see forward
        :param T: see forward
        :param start_X: see forward
        :param n_obs_ot: see forward
        :param return_paths: bool, whether to return also the paths
        :param predict_times: np.array with the times at which each sample
                should be predicted
        :param true_predict_vals: np.array with the true values that should be
                predicted at the predict_times
        :param true_predict_labels: np.array, the correct labels at
                predict_times
        :param true_samples: np.array, the true samples, needed to compute
                predicted labels
        :param normalizing_mean: float, the mean with which the price data was
                normalized
        :param normalizing_std: float, the std with which the price data was
                normalized
        :param eval_predict_steps: list of int, the amount of steps ahead at
                which to predict
        :param thresholds: list of float, the labelling thresholds for each
                entry of eval_predict_steps
        :param predict_labels: as true_predict_labels, but as torch.tensor with
                classes in {0,1,2}
        :param coord_to_compare: list or None, the coordinates on which the
                output and input are compared, applied to the inner dimension of
                the time series, e.g. use [0] to compare on the midprice only
        :param class_report: bool, whether to print the classification report
        :return: eval-loss, if wanted paths t, y for true and pred
        """
        self.eval()

        bs = start_X.shape[0]
        dim = start_X.shape[1]
        if coord_to_compare is None:
            coord_to_compare = np.arange(dim)

        _, _, path_t, path_h, path_y, path_var, cl_out = self.forward(
            times, time_ptr, X, obs_idx, delta_t, T, start_X, n_obs_ot,
            return_path=True, get_loss=False, until_T=True, M=None,
            start_M=None, dim_to=None, predict_labels=predict_labels,
            return_classifier_out=True)

        path_y = path_y.detach().cpu().numpy()
        predicted_vals = np.zeros_like(true_predict_vals)
        for i in range(bs):
            t = predict_times[i][0]
            t_ind = np.argmin(np.abs(path_t - t))
            predicted_vals[i, :, 0] = path_y[t_ind, i, :]

        eval_metric = np.nanmean(
            (predicted_vals[:, coord_to_compare, 0] -
             true_predict_vals[:, coord_to_compare, 0])**2,
            axis=(0,1))

        ref_eval_metric = np.nanmean(
            (true_samples[:, coord_to_compare, -1] -
             true_predict_vals[:, coord_to_compare, 0])**2,
            axis=(0,1))

        f1_scores = None
        predicted_labels = None
        if true_samples is not None and true_predict_labels is not None:
            predicted_labels = np.zeros(bs)
            if cl_out is not None:
                class_probs = self.SM(cl_out).detach().cpu().numpy()
                classes = np.argmax(class_probs, axis=1) - 1
                f1_scores = sklearn.metrics.f1_score(
                    true_predict_labels[:, 0], classes,
                    average="weighted")
                predicted_labels = classes
            else:
                # TODO: this computes the labels incorrectly, since the shift by
                #  X_0 is missing -> results should not be trusted, better to
                #  use classifier
                m_minus = np.mean(
                    true_samples[:, 0, -eval_predict_steps[0]:] *
                    normalizing_std + normalizing_mean, axis=1)
                m_plus = predicted_vals[:, 0, 0]*normalizing_std + \
                         normalizing_mean
                pctc = (m_plus - m_minus) / m_minus
                predicted_labels[pctc > thresholds[0]] = 1
                predicted_labels[pctc < -thresholds[0]] = -1
                f1_scores = sklearn.metrics.f1_score(
                    true_predict_labels[:, 0], predicted_labels,
                    average="weighted")
            if class_report:
                print("eval-mse: {:.5f}".format(eval_metric))
                print("f1-score: {:.5f}".format(f1_scores))
                print("classification report \n",
                      sklearn.metrics.classification_report(
                          true_predict_labels[:, 0], predicted_labels,))

        if return_paths:
            return eval_metric, ref_eval_metric, f1_scores, path_t, path_y, \
                   predicted_vals[:, :, 0], predicted_labels
        else:
            return eval_metric, f1_scores

    def get_pred(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                 n_obs_ot, M=None, start_M=None, which_loss=None):
        """
        get predicted path
        :param times: see forward
        :param time_ptr: see forward
        :param X: see forward
        :param obs_idx: see forward
        :param delta_t: see forward
        :param T: see forward
        :param start_X: see forward
        :param M: see forward
        :param start_M: see forward
        :return: dict, with prediction y and times t
        """
        self.eval()
        h, loss, path_t, path_h, path_y, path_var = self.forward(
            times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx,
            delta_t=delta_t, T=T, start_X=start_X, n_obs_ot=n_obs_ot,
            return_path=True, get_loss=True, until_T=True, M=M,
            start_M=start_M, which_loss=which_loss)
        return {'pred': path_y, 'pred_t': path_t, 'loss': loss,
                'pred_var': path_var}

    def forward_classifier(self, x, y):
        # after last observation has been processed, apply classifier if wanted
        cl_out = None
        cl_loss = None
        if self.classifier is not None:
            cl_out = self.classifier(x)
            cl_loss = self.CEL(input=cl_out, target=y)
        return cl_loss, cl_out



class randomizedNJODE(torch.nn.Module):
    """
    NJ-ODE model
    """
    def __init__(
            self, input_size, hidden_size, output_size,
            ode_nn, readout_nn, enc_nn, use_rnn,
            bias=True, dropout_rate=0, solver="euler",
            weight=0.5, weight_decay=1.,
            **options
    ):
        """
        init the model
        :param input_size: int
        :param hidden_size: int, size of latent variable process
        :param output_size: int
        :param ode_nn: list of list, defining the NN f, see get_ffnn
        :param readout_nn: list of list, defining the NN g, see get_ffnn
        :param enc_nn: list of list, defining the NN e, see get_ffnn
        :param use_rnn: bool, whether to use the RNN for 'jumps'
        :param bias: bool, whether to use a bias for the NNs
        :param dropout_rate: float
        :param solver: str, specifying the ODE solver, suppoorted: {'euler'}
        :param weight: float in [0.5, 1], the initial weight used in the loss
        :param weight_decay: float in [0,1], the decay applied to the weight of
                the loss function after each epoch, decaying towards 0.5
                    1: no decay, weight stays the same
                    0: immediate decay to 0.5 after 1st epoch
                    (0,1): exponential decay towards 0.5
        :param level: level for signature transform
        :param options: kwargs, used:
                - "options" with arg a dict passed
                    from train.train (kwords: 'which_loss', 'residual_enc_dec',
                    'masked', 'input_current_t', 'input_sig', 'level',
                    'use_y_for_ode' are used)
        """
        super().__init__()  # super refers to base class, init initializes

        self.epoch = 1
        self.retrain_epoch = 0
        self.weight = weight
        self.weight_decay = weight_decay
        self.use_rnn = use_rnn  # use RNN for jumps

        # get options from the options of train input
        options1 = options['options']
        if 'which_loss' in options1:
            self.which_loss = options1['which_loss']  # change loss if specified in options
        else:
            self.which_loss = 'standard'  # otherwise take the standard loss
        assert self.which_loss in LOSS_FUN_DICT
        print('using loss: {}'.format(self.which_loss))

        self.residual_enc_dec = True
        if self.use_rnn:
            self.residual_enc_dec = False
        if 'residual_enc_dec' in options1:
            self.residual_enc_dec = options1['residual_enc_dec']
        if 'residual_enc' in options1:
            self.residual_enc_dec = options1['residual_enc']
        self.input_current_t = False
        if 'input_current_t' in options1:
            self.input_current_t = options1['input_current_t']
        self.input_sig = False
        if 'input_sig' in options1:
            self.input_sig = options1['input_sig']
        self.level = 2
        if 'level' in options1:
            self.level = options1['level']
        self.sig_depth = sig.siglength(input_size+1, self.level)
        self.masked = False
        if 'masked' in options1:
            self.masked = options1['masked']
        self.use_y_for_ode = True
        if 'use_y_for_ode' in options1:
            self.use_y_for_ode = options1['use_y_for_ode']
        self.coord_wise_tau = False
        if 'coord_wise_tau' in options1 and self.masked:
            self.coord_wise_tau = options1['coord_wise_tau']
        classifier_dict = None
        if 'classifier_dict' in options:
            classifier_dict = options["classifier_dict"]
        self.use_sig_for_classifier = False
        if 'use_sig_for_classifier' in options1:
            self.use_sig_for_classifier = options1['use_sig_for_classifier']
        self.class_loss_weight = 1.
        self.loss_weight = 1.
        if 'classifier_loss_weight' in options1:
            class_loss_weight = options1['classifier_loss_weight']
            if class_loss_weight == np.infty:
                self.class_loss_weight = 1.
                self.loss_weight = 0.
            else:
                self.class_loss_weight = class_loss_weight

        self.ode_f = ODEFunc(
            input_size=input_size, hidden_size=hidden_size, ode_nn=ode_nn,
            dropout_rate=dropout_rate, bias=bias,
            input_current_t=self.input_current_t, input_sig=self.input_sig,
            sig_depth=self.sig_depth, coord_wise_tau=self.coord_wise_tau)
        self.encoder_map = FFNN(
            input_size=input_size, output_size=hidden_size, nn_desc=enc_nn,
            dropout_rate=dropout_rate, bias=bias, recurrent=self.use_rnn,
            masked=self.masked, residual=self.residual_enc_dec,
            input_sig=self.input_sig, sig_depth=self.sig_depth)
        self.readout_map = LinReg(hidden_size, output_size)
        self.get_classifier(classifier_dict=classifier_dict)

        self.solver = solver
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.apply(init_weights)

    @property
    def device(self):
        device = next(self.parameters()).device
        return device

    def get_classifier(self, classifier_dict):
        self.classifier = None
        self.SM = None
        self.CEL = None
        if classifier_dict is not None:
            if self.use_sig_for_classifier:
                classifier_dict['input_size'] += self.sig_depth
            self.classifier = FFNN(**classifier_dict)
            self.SM = torch.nn.Softmax(dim=1)
            self.CEL = torch.nn.CrossEntropyLoss()

    def weight_decay_step(self):
        inc = (self.weight - 0.5)
        self.weight = 0.5 + inc * self.weight_decay
        return self.weight

    def ode_step(self, h, delta_t, current_time, last_X, tau, signature=None):
        """Executes a single ODE step"""
        if not self.input_sig:
            signature = None
        if self.solver == "euler":
            h = h + delta_t * self.ode_f(
                x=last_X, h=h, tau=tau, tdiff=current_time - tau,
                signature=signature)
        else:
            raise ValueError("Unknown solver '{}'.".format(self.solver))

        current_time += delta_t
        return h, current_time

    def recreate_data(self, times, time_ptr, X, obs_idx, start_X):
        """
        recreates matrix of all observations
        first dim: which data path
        second dim: which time
        """
        # shape: [batch_size, time_steps+1, dimension]
        data = np.empty(shape=(start_X.shape[0], 1+len(times), start_X.shape[1]))
        data[:] = np.nan
        data[:,0,:] = start_X.detach().cpu().numpy()

        X = X.detach().cpu().numpy()
        for j, time in enumerate(times):
            start = time_ptr[j]
            end = time_ptr[j + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]
            data[i_obs, j+1, :] = X_obs
        times_new = np.concatenate(([0], times), axis=0)

        return times_new, data

    def get_signature(self, times, time_ptr, X, obs_idx, start_X):
        """
        Input: See forward
        Returns: signature of paths as nested list
        """
        # reconstructing the data, shape: [batch_size, time_steps+1, dim]
        times_new, data = self.recreate_data(
            times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx,
            start_X=start_X)

        # list of list of list, shape: [batch_size, obs_dates[j], sig_length]
        signature = []
        for j in range(data.shape[0]):  # iterate over batch
            data_j = data[j, :, :]
            observed_j = []
            for i in range(data_j.shape[0]):
                # if the current batch-sample has an observation at the current
                #   time, add it to the list of observations
                if not np.all(np.isnan(data_j[i])):
                    observed_j += [i]
            data_j = data_j[observed_j, :]

            # replace no observations with last observation
            for i in range(1, data_j.shape[0]):
                # # OLD VERSION (SLOW)
                # for k in range(data_j.shape[1]):
                #     if np.isnan(data_j[i, k]):
                #         data_j[i, k] = data_j[i-1, k]
                ks = np.isnan(data_j[i, :])
                data_j[i, ks] = data_j[i-1, ks]

            times_j = times_new[observed_j].reshape(-1, 1)
            # add times to data for signature call
            path_j = np.concatenate((times_j, data_j), axis=1)
            # the following computes the signatures of all partial paths, from
            #   start to each point of the path
            signature.append(sig.sig(path_j, self.level, 2))

        return signature

    def get_Xy_reg(
            self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
            n_obs_ot, return_path=False, until_T=False,
            M=None, start_M=None, which_loss=None, dim_to=None,
            predict_labels=None, return_classifier_out=False,
            return_at_last_obs=False):
        """
        the forward run of this module class, used when calling the module
        instance without a method
        :param times: np.array, of observation times
        :param time_ptr: list, start indices of X and obs_idx for a given
                observation time, first element is 0, this pointer tells how
                many (and which) of the observations of X along the batch-dim
                belong to the current time, and obs_idx then tells to which of
                the batch elements they belong. In particular, not each batch-
                element has to jump at the same time, and only those elements
                which jump at the current time should be updated with a jump
        :param X: torch.tensor, data tensor
        :param obs_idx: list, index of the batch elements where jumps occur at
                current time
        :param delta_t: float, time step for Euler
        :param T: float, the final time
        :param start_X: torch.tensor, the starting point of X
        :param n_obs_ot: torch.tensor, the number of observations over the
                entire time interval for each element of the batch
        :param return_path: bool, whether to return the path of h
        :param get_loss: bool, whether to compute the loss, otherwise 0 returned
        :param until_T: bool, whether to continue until T (for eval) or only
                until last observation (for training)
        :param M: None or torch.tensor, if not None: the mask for the data, same
                size as X, with 0 or 1 entries
        :param start_M: None or torch.tensor, if not None: the mask for start_X,
                same size as start_X
        :param which_loss: see train.train, to overwrite which loss for eval
        :param dim_to: None or int, if given not all coordinates along the
                data-dimension axis are used but only up to dim_to. this can be
                used if func_appl_X is used in train, but the loss etc. should
                only be computed for the original coordinates (without those
                resulting from the function applications)
        :param predict_labels: None or torch.tensor with the true labels to
                predict
        :param return_classifier_out: bool, whether to return the output of the
                classifier
        :return: torch.tensor (hidden state at final time), torch.tensor (loss),
                    if wanted the paths of t (np.array) and h, y (torch.tensors)
        """

        linreg_X = []
        linreg_y = []

        if which_loss is None:
            which_loss = self.which_loss

        last_X = start_X
        batch_size = start_X.size()[0]
        data_dim = start_X.size()[1]
        if dim_to is None:
            dim_to = data_dim
        if self.coord_wise_tau:
            tau = torch.tensor([[0.0]]).repeat(batch_size, data_dim)
        else:
            tau = torch.tensor([[0.0]]).repeat(batch_size, 1)
        current_time = 0.0
        loss = 0
        c_sig = None

        if self.input_sig:
            if self.masked:
                Mdc = M.clone()
                Mdc[Mdc==0] = np.nan
                X_obs_impute = X * Mdc
                signature = self.get_signature(
                    times=times, time_ptr=time_ptr, X=X_obs_impute,
                    obs_idx=obs_idx, start_X=start_X)
            else:
                signature = self.get_signature(
                    times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx,
                    start_X=start_X)

            # in beginning, no path was observed => set sig to 0
            current_sig = np.zeros((batch_size, self.sig_depth))
            current_sig_nb = np.zeros(batch_size).astype(int)
            c_sig = torch.from_numpy(current_sig).float()

        if self.masked:
            if start_M is None:
                start_M = torch.ones_like(start_X)
        else:
            start_M = None

        h = self.encoder_map(
            start_X, mask=start_M, sig=c_sig,
            h=torch.zeros((batch_size, self.hidden_size)))

        if return_path:
            path_t = [0]
            path_h = [h]
        h_at_last_obs = h.clone()
        sig_at_last_obs = c_sig

        assert len(times) + 1 == len(time_ptr)

        for i, obs_time in enumerate(times):
            # Propagation of the ODE until next observation
            while current_time < (obs_time - 1e-10 * delta_t):  # 0.0001 delta_t used for numerical consistency.
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time
                if self.solver == 'euler':
                    h, current_time = self.ode_step(
                        h, delta_t_, current_time, last_X=last_X, tau=tau,
                        signature=c_sig)
                    current_time_nb = int(round(current_time / delta_t))
                else:
                    raise NotImplementedError

                # Storing the predictions.
                if return_path:
                    path_t.append(current_time)
                    path_h.append(h)

            # Reached an observation - only update those elements of the batch,
            #    for which an observation is made
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]
            if self.masked:
                if isinstance(M, np.ndarray):
                    M_obs = torch.from_numpy(M[start:end])
                else:
                    M_obs = M[start:end]
            else:
                M_obs = None

            # update signature
            if self.input_sig:
                for j in i_obs:
                    current_sig[j, :] = signature[j][current_sig_nb[j]]
                current_sig_nb[i_obs] += 1
                c_sig = torch.from_numpy(current_sig).float()

            # Using RNNCell to update h. Also updating loss, tau and last_X
            h_bj = h.clone()
            X_obs_impute = X_obs
            temp = h.clone()
            if self.masked:
                # TODO: imputation does not work with OLS -> therefore set to 0
                #  for non-observed coordinates
                X_obs_impute = X_obs * M_obs
            c_sig_iobs = None
            if self.input_sig:
                c_sig_iobs = c_sig[i_obs]
            temp[i_obs.long()] = self.encoder_map(
                X_obs_impute, mask=M_obs, sig=c_sig_iobs, h=h[i_obs])
            h = temp
            h_aj = h.clone()

            # update h and sig at last observation
            h_at_last_obs[i_obs.long()] = h[i_obs.long()].clone()
            sig_at_last_obs = c_sig

            for ii, o in enumerate(i_obs.long()):
                linreg_X.append(h_bj[o].detach().cpu().numpy())
                linreg_X.append(h_aj[o].detach().cpu().numpy())
                target = X_obs[ii, :dim_to].detach().cpu().numpy()
                if self.masked:
                    target[M_obs[ii, :dim_to].detach().cpu().numpy()==0] = np.nan
                linreg_y.append(target)
                linreg_y.append(target)

            # make update of last_X and tau, that is not inplace
            #    (otherwise problems in autograd)
            temp_X = last_X.clone()
            temp_tau = tau.clone()
            temp_X[i_obs.long()] = X_obs_impute
            if self.coord_wise_tau:
                _M = torch.zeros_like(temp_tau)
                _M[i_obs] = M_obs
                temp_tau[_M==1] = obs_time.astype(np.float64)
            else:
                temp_tau[i_obs.long()] = obs_time.astype(np.float64)
            last_X = temp_X
            tau = temp_tau

            if return_path:
                path_t.append(obs_time)
                path_h.append(h)

        # after last observation has been processed, apply classifier if wanted
        cl_out = None
        if self.classifier is not None and predict_labels is not None:
            cl_loss = torch.tensor(0.)
            cl_input = h_at_last_obs
            if self.use_sig_for_classifier:
                cl_input = torch.cat([cl_input, sig_at_last_obs], dim=1)
            cl_out = self.classifier(cl_input)
            cl_loss = cl_loss + self.CEL(
                input=self.SM(cl_out), target=predict_labels[:, 0])
            loss = [self.loss_weight*loss + self.class_loss_weight*cl_loss,
                    loss, cl_loss]

        # after every observation has been processed, propagating until T
        if until_T:
            if self.input_sig:
                c_sig = torch.from_numpy(current_sig).float()
            while current_time < T - 1e-10 * delta_t:
                if current_time < T - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = T - current_time
                if self.solver == 'euler':
                    h, current_time = self.ode_step(
                        h, delta_t_, current_time, last_X=last_X, tau=tau,
                        signature=c_sig)
                else:
                    raise NotImplementedError

                # Storing the predictions.
                if return_path:
                    path_t.append(current_time)
                    path_h.append(h)

        if return_at_last_obs:
            return h_at_last_obs, sig_at_last_obs
        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            if return_classifier_out:
                return linreg_X, linreg_y, \
                       np.array(path_t), torch.stack(path_h), cl_out
            return linreg_X, linreg_y, \
                   np.array(path_t), torch.stack(path_h)
        else:
            if return_classifier_out and self.classifier is not None:
                return linreg_X, linreg_y, cl_out
            return linreg_X, linreg_y

    def forward(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                n_obs_ot, return_path=False, get_loss=True, until_T=False,
                M=None, start_M=None, which_loss=None, dim_to=None,
                predict_labels=None, return_classifier_out=False,
                return_at_last_obs=False):
        """
        the forward run of this module class, used when calling the module
        instance without a method
        :param times: np.array, of observation times
        :param time_ptr: list, start indices of X and obs_idx for a given
                observation time, first element is 0, this pointer tells how
                many (and which) of the observations of X along the batch-dim
                belong to the current time, and obs_idx then tells to which of
                the batch elements they belong. In particular, not each batch-
                element has to jump at the same time, and only those elements
                which jump at the current time should be updated with a jump
        :param X: torch.tensor, data tensor
        :param obs_idx: list, index of the batch elements where jumps occur at
                current time
        :param delta_t: float, time step for Euler
        :param T: float, the final time
        :param start_X: torch.tensor, the starting point of X
        :param n_obs_ot: torch.tensor, the number of observations over the
                entire time interval for each element of the batch
        :param return_path: bool, whether to return the path of h
        :param get_loss: bool, whether to compute the loss, otherwise 0 returned
        :param until_T: bool, whether to continue until T (for eval) or only
                until last observation (for training)
        :param M: None or torch.tensor, if not None: the mask for the data, same
                size as X, with 0 or 1 entries
        :param start_M: None or torch.tensor, if not None: the mask for start_X,
                same size as start_X
        :param which_loss: see train.train, to overwrite which loss for eval
        :param dim_to: None or int, if given not all coordinates along the
                data-dimension axis are used but only up to dim_to. this can be
                used if func_appl_X is used in train, but the loss etc. should
                only be computed for the original coordinates (without those
                resulting from the function applications)
        :param predict_labels: None or torch.tensor with the true labels to
                predict
        :param return_classifier_out: bool, whether to return the output of the
                classifier
        :return: torch.tensor (hidden state at final time), torch.tensor (loss),
                    if wanted the paths of t (np.array) and h, y (torch.tensors)
        """
        if which_loss is None:
            which_loss = self.which_loss

        last_X = start_X
        batch_size = start_X.size()[0]
        data_dim = start_X.size()[1]
        if dim_to is None:
            dim_to = data_dim
        if self.coord_wise_tau:
            tau = torch.tensor([[0.0]]).repeat(batch_size, data_dim).to(
                self.device)
        else:
            tau = torch.tensor([[0.0]]).repeat(batch_size, 1).to(self.device)
        current_time = 0.0
        loss = 0
        c_sig = None

        if self.input_sig:
            if self.masked:
                Mdc = M.clone()
                Mdc[Mdc==0] = np.nan
                X_obs_impute = X * Mdc
                signature = self.get_signature(
                    times=times, time_ptr=time_ptr, X=X_obs_impute,
                    obs_idx=obs_idx, start_X=start_X)
            else:
                signature = self.get_signature(
                    times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx,
                    start_X=start_X)

            # in beginning, no path was observed => set sig to 0
            current_sig = np.zeros((batch_size, self.sig_depth))
            current_sig_nb = np.zeros(batch_size).astype(int)
            c_sig = torch.from_numpy(current_sig).float().to(self.device)

        if self.masked:
            if start_M is None:
                start_M = torch.ones_like(start_X)
        else:
            start_M = None

        h = self.encoder_map(
            start_X, mask=start_M, sig=c_sig,
            h=torch.zeros((batch_size, self.hidden_size)).to(self.device))

        if return_path:
            path_t = [0]
            path_h = [h]
            path_y = [self.readout_map(h)]
        h_at_last_obs = h.clone()
        sig_at_last_obs = c_sig

        assert len(times) + 1 == len(time_ptr)

        for i, obs_time in enumerate(times):
            # Propagation of the ODE until next observation
            while current_time < (obs_time - 1e-10 * delta_t):  # 0.0001 delta_t used for numerical consistency.
                if current_time < obs_time - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = obs_time - current_time
                if self.solver == 'euler':
                    h, current_time = self.ode_step(
                        h, delta_t_, current_time, last_X=last_X, tau=tau,
                        signature=c_sig)
                    current_time_nb = int(round(current_time / delta_t))
                else:
                    raise NotImplementedError

                # Storing the predictions.
                if return_path:
                    path_t.append(current_time)
                    path_h.append(h)
                    path_y.append(self.readout_map(h))

            # Reached an observation - only update those elements of the batch,
            #    for which an observation is made
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]
            i_obs = obs_idx[start:end]
            if self.masked:
                if isinstance(M, np.ndarray):
                    M_obs = torch.from_numpy(M[start:end]).to(self.device)
                else:
                    M_obs = M[start:end]
            else:
                M_obs = None

            # update signature
            if self.input_sig:
                for j in i_obs:
                    current_sig[j, :] = signature[j][current_sig_nb[j]]
                current_sig_nb[i_obs] += 1
                c_sig = torch.from_numpy(current_sig).float().to(self.device)

            # Using RNNCell to update h. Also updating loss, tau and last_X
            Y_bj = self.readout_map(h)
            X_obs_impute = X_obs
            temp = h.clone()
            if self.masked:
                X_obs_impute = X_obs * M_obs
            c_sig_iobs = None
            if self.input_sig:
                c_sig_iobs = c_sig[i_obs]
            temp[i_obs.long()] = self.encoder_map(
                X_obs_impute, mask=M_obs, sig=c_sig_iobs, h=h[i_obs])
            h = temp
            Y = self.readout_map(h)

            # update h and sig at last observation
            h_at_last_obs[i_obs.long()] = h[i_obs.long()].clone()
            sig_at_last_obs = c_sig

            if get_loss:
                loss = loss + LOSS_FUN_DICT[which_loss](
                    X_obs=X_obs[:, :dim_to], Y_obs=Y[i_obs.long(), :dim_to],
                    Y_obs_bj=Y_bj[i_obs.long(), :dim_to],
                    n_obs_ot=n_obs_ot[i_obs.long()], batch_size=batch_size,
                    weight=self.weight, M_obs=M_obs)

            # make update of last_X and tau, that is not inplace
            #    (otherwise problems in autograd)
            temp_X = last_X.clone()
            temp_tau = tau.clone()
            temp_X[i_obs.long()] = X_obs_impute
            if self.coord_wise_tau:
                _M = torch.zeros_like(temp_tau)
                _M[i_obs] = M_obs
                temp_tau[_M==1] = obs_time.astype(np.float64)
            else:
                temp_tau[i_obs.long()] = obs_time.astype(np.float64)
            last_X = temp_X
            tau = temp_tau

            if return_path:
                path_t.append(obs_time)
                path_h.append(h)
                path_y.append(Y)

        # after last observation has been processed, apply classifier if wanted
        cl_out = None
        if self.classifier is not None and predict_labels is not None:
            cl_loss = torch.tensor(0.)
            cl_input = h_at_last_obs
            if self.use_sig_for_classifier:
                cl_input = torch.cat([cl_input, sig_at_last_obs], dim=1)
            cl_out = self.classifier(cl_input)
            cl_loss = cl_loss + self.CEL(
                input=self.SM(cl_out), target=predict_labels[:, 0])
            loss = [self.loss_weight*loss + self.class_loss_weight*cl_loss,
                    loss, cl_loss]

        # after every observation has been processed, propagating until T
        if until_T:
            if self.input_sig:
                c_sig = torch.from_numpy(current_sig).float().to(self.device)
            while current_time < T - 1e-10 * delta_t:
                if current_time < T - delta_t:
                    delta_t_ = delta_t
                else:
                    delta_t_ = T - current_time
                if self.solver == 'euler':
                    h, current_time = self.ode_step(
                        h, delta_t_, current_time, last_X=last_X, tau=tau,
                        signature=c_sig)
                else:
                    raise NotImplementedError

                # Storing the predictions.
                if return_path:
                    path_t.append(current_time)
                    path_h.append(h)
                    path_y.append(self.readout_map(h))

        if return_at_last_obs:
            return h_at_last_obs, sig_at_last_obs
        if return_path:
            # path dimension: [time_steps, batch_size, output_size]
            if return_classifier_out:
                return h, loss, np.array(path_t), torch.stack(path_h), \
                       torch.stack(path_y)[:, :, :dim_to], cl_out
            return h, loss, np.array(path_t), torch.stack(path_h), \
                   torch.stack(path_y)[:, :, :dim_to]
        else:
            if return_classifier_out and self.classifier is not None:
                return h, loss, cl_out
            return h, loss

    def evaluate(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                 n_obs_ot, stockmodel, cond_exp_fun_kwargs=None,
                 diff_fun=lambda x, y: np.nanmean((x - y) ** 2),
                 return_paths=False, M=None, true_paths=None, start_M=None,
                 true_mask=None, mult=None):
        """
        evaluate the model at its current training state against the true
        conditional expectation
        :param times: see forward
        :param time_ptr: see forward
        :param X: see forward
        :param obs_idx: see forward
        :param delta_t: see forward
        :param T: see forward
        :param start_X: see forward
        :param n_obs_ot: see forward
        :param stockmodel: stock_model.StockModel instance, used to compute true
                cond. exp.
        :param cond_exp_fun_kwargs: dict, the kwargs for the cond. exp. function
                currently not used
        :param diff_fun: function, to compute difference between optimal and
                predicted cond. exp
        :param return_paths: bool, whether to return also the paths
        :param M: see forward
        :param start_M: see forward
        :param true_paths: np.array, shape [batch_size, dimension, time_steps+1]
        :param true_mask: as true_paths, with mask entries
        :return: eval-loss, if wanted paths t, y for true and pred
        """
        self.eval()

        dim = start_X.shape[1]
        dim_to = dim
        if mult is not None and mult > 1:
            dim_to = round(dim/mult)

        _, _, path_t, path_h, path_y = self.forward(
            times, time_ptr, X, obs_idx, delta_t, T, start_X, None,
            return_path=True, get_loss=False, until_T=True, M=M,
            start_M=start_M, dim_to=dim_to)

        if true_paths is None:
            if M is not None:
                M = M.detach().cpu().numpy()[:, :dim_to]
            _, true_path_t, true_path_y = stockmodel.compute_cond_exp(
                times, time_ptr, X.detach().cpu().numpy()[:, :dim_to],
                obs_idx.detach().cpu().numpy(),
                delta_t, T, start_X.detach().cpu().numpy()[:, :dim_to],
                n_obs_ot.detach().cpu().numpy(),
                return_path=True, get_loss=False, M=M, )
        else:
            true_t = np.linspace(0, T, true_paths.shape[2])
            which_t_ind = []
            for t in path_t:
                which_t_ind.append(np.argmin(np.abs(true_t - t)))
            true_path_y = true_paths[:, :dim_to, which_t_ind]
            true_path_y = np.transpose(true_path_y, axes=(2, 0, 1))
            true_path_t = true_t[which_t_ind]

        if path_y.detach().cpu().numpy().shape == true_path_y.shape:
            eval_metric = diff_fun(path_y.detach().cpu().numpy(), true_path_y)
        else:
            print(path_y.detach().cpu().numpy().shape)
            print(true_path_y.shape)
            raise ValueError("Shapes do not match!")
        if return_paths:
            return eval_metric, path_t, true_path_t, path_y, true_path_y
        else:
            return eval_metric

    def evaluate_LOB(
            self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
            n_obs_ot, return_paths=False, predict_times=None,
            true_predict_vals=None, true_predict_labels=None, true_samples=None,
            normalizing_mean=0., normalizing_std=1., eval_predict_steps=None,
            thresholds=None, predict_labels=None,
            coord_to_compare=(0,), class_report=False):
        """
        evaluate the model at its current training state for the LOB dataset

        :param times: see forward
        :param time_ptr: see forward
        :param X: see forward
        :param obs_idx: see forward
        :param delta_t: see forward
        :param T: see forward
        :param start_X: see forward
        :param n_obs_ot: see forward
        :param return_paths: bool, whether to return also the paths
        :param predict_times: np.array with the times at which each sample
                should be predicted
        :param true_predict_vals: np.array with the true values that should be
                predicted at the predict_times
        :param true_predict_labels: np.array, the correct labels at
                predict_times
        :param true_samples: np.array, the true samples, needed to compute
                predicted labels
        :param normalizing_mean: float, the mean with which the price data was
                normalized
        :param normalizing_std: float, the std with which the price data was
                normalized
        :param eval_predict_steps: list of int, the amount of steps ahead at
                which to predict
        :param thresholds: list of float, the labelling thresholds for each
                entry of eval_predict_steps
        :param predict_labels: as true_predict_labels, but as torch.tensor with
                classes in {0,1,2}
        :param coord_to_compare: list or None, the coordinates on which the
                output and input are compared, applied to the inner dimension of
                the time series, e.g. use [0] to compare on the midprice only
        :param class_report: bool, whether to print the classification report
        :return: eval-loss, if wanted paths t, y for true and pred
        """
        self.eval()

        bs = start_X.shape[0]
        dim = start_X.shape[1]
        if coord_to_compare is None:
            coord_to_compare = np.arange(dim)

        _, _, path_t, path_h, path_y, cl_out = self.forward(
            times, time_ptr, X, obs_idx, delta_t, T, start_X, n_obs_ot,
            return_path=True, get_loss=False, until_T=True, M=None,
            start_M=None, dim_to=None, predict_labels=predict_labels,
            return_classifier_out=True)

        path_y = path_y.detach().cpu().numpy()
        predicted_vals = np.zeros_like(true_predict_vals)
        for i in range(bs):
            t = predict_times[i][0]
            t_ind = np.argmin(np.abs(path_t - t))
            predicted_vals[i, :, 0] = path_y[t_ind, i, :]

        eval_metric = np.nanmean(
            (predicted_vals[:, coord_to_compare, 0] -
             true_predict_vals[:, coord_to_compare, 0])**2,
            axis=(0,1))

        ref_eval_metric = np.nanmean(
            (true_samples[:, coord_to_compare, -1] -
             true_predict_vals[:, coord_to_compare, 0])**2,
            axis=(0,1))

        f1_scores = None
        predicted_labels = None
        if true_samples is not None and true_predict_labels is not None:
            predicted_labels = np.zeros(bs)
            if cl_out is not None:
                class_probs = self.SM(cl_out).detach().cpu().numpy()
                classes = np.argmax(class_probs, axis=1) - 1
                f1_scores = sklearn.metrics.f1_score(
                    true_predict_labels[:, 0], classes,
                    average="weighted")
                predicted_labels = classes
            else:
                # TODO: this computes the labels incorrectly, since the shift by
                #  X_0 is missing -> results should not be trusted, better to
                #  use classifier
                m_minus = np.mean(
                    true_samples[:, 0, -eval_predict_steps[0]:] *
                    normalizing_std + normalizing_mean, axis=1)
                m_plus = predicted_vals[:, 0, 0]*normalizing_std + \
                         normalizing_mean
                pctc = (m_plus - m_minus) / m_minus
                predicted_labels[pctc > thresholds[0]] = 1
                predicted_labels[pctc < -thresholds[0]] = -1
                f1_scores = sklearn.metrics.f1_score(
                    true_predict_labels[:, 0], predicted_labels,
                    average="weighted")
            if class_report:
                print("eval-mse: {:.5f}".format(eval_metric))
                print("f1-score: {:.5f}".format(f1_scores))
                print("classification report \n",
                      sklearn.metrics.classification_report(
                          true_predict_labels[:, 0], predicted_labels,))

        if return_paths:
            return eval_metric, ref_eval_metric, f1_scores, path_t, path_y, \
                   predicted_vals[:, :, 0], predicted_labels
        else:
            return eval_metric, f1_scores

    def get_pred(self, times, time_ptr, X, obs_idx, delta_t, T, start_X,
                 n_obs_ot, M=None, start_M=None, which_loss=None, dim_to=None):
        """
        get predicted path
        :param times: see forward
        :param time_ptr: see forward
        :param X: see forward
        :param obs_idx: see forward
        :param delta_t: see forward
        :param T: see forward
        :param start_X: see forward
        :param M: see forward
        :param start_M: see forward
        :return: dict, with prediction y and times t
        """
        self.eval()
        h, loss, path_t, path_h, path_y = self.forward(
            times=times, time_ptr=time_ptr, X=X, obs_idx=obs_idx,
            delta_t=delta_t, T=T, start_X=start_X, n_obs_ot=n_obs_ot,
            return_path=True, get_loss=True, until_T=True, M=M,
            start_M=start_M, which_loss=which_loss, dim_to=dim_to)
        return {'pred': path_y, 'pred_t': path_t, 'loss': loss}

    def forward_classifier(self, x, y):
        # after last observation has been processed, apply classifier if wanted
        cl_out = None
        cl_loss = None
        if self.classifier is not None:
            cl_out = self.classifier(x)
            cl_loss = self.CEL(input=self.SM(cl_out), target=y)
        return cl_loss, cl_out


class NJmodel(NJODE):
    """
    Neural Jump model without an ODE, i.e. directly learning the Doob-Dynkin
    function
    """
    def __init__(
            self, input_size, hidden_size, output_size,
            ode_nn, readout_nn, enc_nn, use_rnn,
            bias=True, dropout_rate=0, solver="euler",
            weight=0.5, weight_decay=1.,
            **options
    ):
        """
        init the model
        :param input_size: int
        :param hidden_size: int, size of latent variable process
        :param output_size: int
        :param ode_nn: list of list, defining the NN f, see get_ffnn
        :param readout_nn: list of list, defining the NN g, see get_ffnn
        :param enc_nn: list of list, defining the NN e, see get_ffnn
        :param use_rnn: bool, whether to use the RNN for 'jumps'
        :param bias: bool, whether to use a bias for the NNs
        :param dropout_rate: float
        :param solver: str, specifying the ODE solver, suppoorted: {'euler'}
        :param weight: float in [0.5, 1], the initial weight used in the loss
        :param weight_decay: float in [0,1], the decay applied to the weight of
                the loss function after each epoch, decaying towards 0.5
                    1: no decay, weight stays the same
                    0: immediate decay to 0.5 after 1st epoch
                    (0,1): exponential decay towards 0.5
        :param level: level for signature transform
        :param options: kwargs, used:
                - "classifier_nn"
                - "options" with arg a dict passed
                    from train.train (kwords: 'which_loss', 'residual_enc_dec',
                    'residual_enc'
                    'masked', 'input_current_t', 'input_sig', 'level',
                    'use_y_for_ode', 'enc_input_t' are used)
        """
        super().__init__(
            input_size, hidden_size, output_size,
            ode_nn, readout_nn, enc_nn, use_rnn,
            bias, dropout_rate, solver,
            weight, weight_decay, **options)

        self.enc_input_t = True

        t_size = 2
        if self.coord_wise_tau:
            t_size = 2*input_size

        self.ode_f = None
        self.encoder_map = FFNN(
            input_size=input_size, output_size=output_size, nn_desc=enc_nn,
            dropout_rate=dropout_rate, bias=bias, recurrent=self.use_rnn,
            masked=self.masked, residual=self.residual_enc,
            input_sig=self.input_sig, sig_depth=self.sig_depth,
            input_t=self.enc_input_t, t_size=t_size)
        self.readout_map = torch.nn.Identity()

        self.solver = solver
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.apply(init_weights)

    def ode_step(self, h, delta_t, current_time, last_X, tau, signature=None):
        """Executes a single ODE step"""
        if not self.input_sig:
            signature = None
        current_time += delta_t
        next_h = self.encoder_map(
            last_X, mask=torch.zeros_like(last_X), sig=signature, h=h,
            t=torch.cat((tau, current_time - tau), dim=1))

        return next_h, current_time


