"""
author: Florian Krach & Calypso Herrera

This is mainly a copy of:
# Latent ODEs for Irregularly-Sampled Time Series
# Authors: Yulia Rubanova and Ricky Chen
with the licence:

# MIT License
#
# Copyright (c) 2019 Yulia Rubanova
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

Some minor changes were made to adapt to our settings.
Added:
- variable_time_collate_fn1
"""
###########################


import os
import matplotlib
import sys, socket
import numpy as np
import tarfile
import torch
from torchvision.datasets.utils import download_url

sys.path.append("../../")
import baselines.latent_ODE.utils_LODE as utils

if 'ada-' not in socket.gethostname():
    SERVER = False
else:
    SERVER = True
if SERVER:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Adapted from: https://github.com/rtqichen/time-series-datasets

# get minimum and maximum for each feature across the whole dataset
def get_data_min_max(records, device):

    data_min, data_max = None, None
    inf = torch.Tensor([float("Inf")])[0].to(device)

    for b, (record_id, tt, vals, mask, labels, noise) in enumerate(records):
        n_features = vals.size(-1)

        batch_min = []
        batch_max = []
        for i in range(n_features):
            non_missing_vals = vals[:, i][mask[:, i] == 1]
            if len(non_missing_vals) == 0:
                batch_min.append(inf)
                batch_max.append(-inf)
            else:
                batch_min.append(torch.min(non_missing_vals))
                batch_max.append(torch.max(non_missing_vals))

        batch_min = torch.stack(batch_min)
        batch_max = torch.stack(batch_max)

        if (data_min is None) and (data_max is None):
            data_min = batch_min
            data_max = batch_max
        else:
            data_min = torch.min(data_min, batch_min)
            data_max = torch.max(data_max, batch_max)

    return data_min, data_max


class PhysioNet(object):
    urls = [
        'https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download',
        'https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download',
    ]

    outcome_urls = [
        'https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt']

    params = [
        # TODO (FK): to be as described  in paper, delete the following line
        'Age', 'Gender', 'Height', 'ICUType',
        'Weight', 'Albumin', 'ALP', 'ALT',
        'AST', 'Bilirubin', 'BUN',
        'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
        'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg',
        'MAP', 'MechVent', 'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'PaCO2',
        'PaO2', 'pH', 'Platelets', 'RespRate',
        'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC'
    ]

    params_dict = {k: i for i, k in enumerate(params)}

    labels = ["SAPS-I", "SOFA", "Length_of_stay", "Survival",
              "In-hospital_death"]
    labels_dict = {k: i for i, k in enumerate(labels)}

    def __init__(self, root, train=True, download=False,
                 quantization=0.1, n_samples=None, device=torch.device("cpu"),
                 obs_noise=None):

        self.root = root
        self.train = train
        self.reduce = "average"
        self.quantization = quantization

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        if device == torch.device("cpu"):
            self.data = torch.load(
                os.path.join(self.processed_folder, data_file),
                map_location='cpu')
            self.labels = torch.load(
                os.path.join(self.processed_folder, self.label_file),
                map_location='cpu')
        else:
            self.data = torch.load(
                os.path.join(self.processed_folder, data_file))
            self.labels = torch.load(
                os.path.join(self.processed_folder, self.label_file))

        if n_samples is not None:
            self.data = self.data[:n_samples]
            self.labels = self.labels[:n_samples]

        # FK: added
        if obs_noise is None:
            obs_noise = {'std_factor': 0., 'seed': 0}
        np.random.seed(obs_noise['seed'])
        if "stds" in obs_noise and obs_noise["stds"] is not None:
            stds = obs_noise["stds"]
        else:
            stds = []
            for i, p in enumerate(self.params):
                dat = []
                for j, (record_id, tt, vals, mask, labels) in enumerate(
                        self.data):
                    dat.append(vals[:, i][mask[:, i] == 1])
                dat = np.concatenate(dat)
                stds.append(np.std(dat))

        data_new = []
        for j, (record_id, tt, vals, mask, labels) in enumerate(
                self.data):
            noise = torch.zeros_like(vals)
            for i, p in enumerate(self.params):
                noise[:, i][mask[:, i] == 1] += torch.tensor(
                    np.random.normal(
                        0, obs_noise['std_factor'] * stds[i],
                        len(vals[:, i][mask[:, i] == 1])),
                    dtype=torch.float32)
            data_new.append((record_id, tt, vals, mask, labels, noise))
        self.data = data_new
        self.stds = stds
        self.obs_noise = obs_noise


    def download(self):
        if self._check_exists():
            return

        self.device = torch.device("cpu")

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # Download outcome data
        for url in self.outcome_urls:
            filename = url.rpartition('/')[2]
            download_url(url, self.raw_folder, filename, None)

            txtfile = os.path.join(self.raw_folder, filename)
            with open(txtfile) as f:
                lines = f.readlines()
                outcomes = {}
                for l in lines[1:]:
                    l = l.rstrip().split(',')
                    record_id, labels = l[0], np.array(l[1:]).astype(float)
                    outcomes[record_id] = torch.Tensor(labels).to(self.device)

                torch.save(
                    labels,
                    os.path.join(self.processed_folder,
                                 filename.split('.')[0] + '.pt')
                )

        for url in self.urls:
            filename = url.rpartition('/')[2]
            download_url(url, self.raw_folder, filename, None)
            tar = tarfile.open(os.path.join(self.raw_folder, filename), "r:gz")
            tar.extractall(self.raw_folder)
            tar.close()

            print('Processing {}...'.format(filename))

            dirname = os.path.join(self.raw_folder, filename.split('.')[0])
            patients = []
            total = 0
            for txtfile in os.listdir(dirname):
                record_id = txtfile.split('.')[0]
                with open(os.path.join(dirname, txtfile)) as f:
                    lines = f.readlines()
                    prev_time = 0
                    tt = [0.]
                    vals = [torch.zeros(len(self.params)).to(self.device)]
                    mask = [torch.zeros(len(self.params)).to(self.device)]
                    nobs = [torch.zeros(len(self.params))]
                    for l in lines[1:]:
                        total += 1
                        time, param, val = l.split(',')
                        # Time in hours
                        time = float(time.split(':')[0]) + float(
                            time.split(':')[1]) / 60.
                        # round up the time stamps (up to 6 min by default)
                        # used for speed -- we actually don't need to quantize it in Latent ODE
                        time = round(
                            time / self.quantization) * self.quantization

                        if time != prev_time:
                            tt.append(time)
                            vals.append(
                                torch.zeros(len(self.params)).to(self.device))
                            mask.append(
                                torch.zeros(len(self.params)).to(self.device))
                            nobs.append(
                                torch.zeros(len(self.params)).to(self.device))
                            prev_time = time

                        if param in self.params_dict:
                            # vals[-1][self.params_dict[param]] = float(val)
                            n_observations = nobs[-1][self.params_dict[param]]
                            if self.reduce == 'average' and n_observations > 0:
                                prev_val = vals[-1][self.params_dict[param]]
                                new_val = (prev_val * n_observations + float(
                                    val)) / (n_observations + 1)
                                vals[-1][self.params_dict[param]] = new_val
                            else:
                                vals[-1][self.params_dict[param]] = float(val)
                            mask[-1][self.params_dict[param]] = 1
                            nobs[-1][self.params_dict[param]] += 1
                        else:
                            assert param == 'RecordID', 'Read unexpected param {}'.format(
                                param)
                tt = torch.tensor(tt).to(self.device)
                vals = torch.stack(vals)
                mask = torch.stack(mask)

                labels = None
                if record_id in outcomes:
                    # Only training set has labels
                    labels = outcomes[record_id]
                    # Out of 5 label types provided for Physionet, take only the last one -- mortality
                    labels = labels[4]

                patients.append((record_id, tt, vals, mask, labels))

            torch.save(
                patients,
                os.path.join(self.processed_folder,
                             filename.split('.')[0] + "_" + str(
                                 self.quantization) + '.pt')
            )

        print('Done!')

    def _check_exists(self):
        for url in self.urls:
            filename = url.rpartition('/')[2]

            if not os.path.exists(
                    os.path.join(self.processed_folder,
                                 filename.split('.')[0] + "_" + str(
                                     self.quantization) + '.pt')
            ):
                return False
        return True

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def training_file(self):
        return 'set-a_{}.pt'.format(self.quantization)

    @property
    def test_file(self):
        return 'set-b_{}.pt'.format(self.quantization)

    @property
    def label_file(self):
        return 'Outcomes-a.pt'

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_label(self, record_id):
        return self.labels[record_id]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(
            'train' if self.train is True else 'test')
        fmt_str += '    Root Location: {}\n'.format(self.root)
        fmt_str += '    Quantization: {}\n'.format(self.quantization)
        fmt_str += '    Reduce: {}\n'.format(self.reduce)
        return fmt_str

    def visualize(self, timesteps, data, mask, plot_name):
        width = 15
        height = 15

        non_zero_attributes = (torch.sum(mask, 0) > 2).numpy()
        non_zero_idx = [i for i in range(len(non_zero_attributes)) if
                        non_zero_attributes[i] == 1.]
        n_non_zero = sum(non_zero_attributes)

        mask = mask[:, non_zero_idx]
        data = data[:, non_zero_idx]

        params_non_zero = [self.params[i] for i in non_zero_idx]
        params_dict = {k: i for i, k in enumerate(params_non_zero)}

        n_col = 3
        n_row = n_non_zero // n_col + (n_non_zero % n_col > 0)
        fig, ax_list = plt.subplots(n_row, n_col, figsize=(width, height),
                                    facecolor='white')

        # for i in range(len(self.params)):
        for i in range(n_non_zero):
            param = params_non_zero[i]
            param_id = params_dict[param]

            tp_mask = mask[:, param_id].long()

            tp_cur_param = timesteps[tp_mask == 1.]
            data_cur_param = data[tp_mask == 1., param_id]

            ax_list[i // n_col, i % n_col].plot(tp_cur_param.numpy(),
                                                data_cur_param.numpy(),
                                                marker='o')
            ax_list[i // n_col, i % n_col].set_title(param)

        fig.tight_layout()
        fig.savefig(plot_name)
        plt.close(fig)


def variable_time_collate_fn(batch, args, device=torch.device("cpu"),
                             data_type="train",
                             data_min=None, data_max=None):
    """
	Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
		- record_id is a patient id
		- tt is a 1-dimensional tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
		- labels is a list of labels for the current patient, if labels are available. Otherwise None.
	Returns:
		combined_tt: The union of all time observations.
		combined_vals: (M, T, D) tensor containing the observed values.
		combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
	"""
    D = batch[0][2].shape[1]
    combined_tt, inverse_indices = torch.unique(
        torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
    combined_tt = combined_tt.to(device)

    offset = 0
    combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)

    combined_labels = None
    N_labels = 1

    combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(
        float('nan'))
    combined_labels = combined_labels.to(device=device)

    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        tt = tt.to(device)
        vals = vals.to(device)
        mask = mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        indices = inverse_indices[offset:offset + len(tt)]
        offset += len(tt)

        combined_vals[b, indices] = vals
        combined_mask[b, indices] = mask

        if labels is not None:
            combined_labels[b] = labels

    combined_vals, _, _ = utils.normalize_masked_data(combined_vals,
                                                      combined_mask,
                                                      att_min=data_min,
                                                      att_max=data_max)

    if torch.max(combined_tt) != 0.:
        combined_tt = combined_tt / torch.max(combined_tt)

    data_dict = {
        "data": combined_vals,
        "time_steps": combined_tt,
        "mask": combined_mask,
        "labels": combined_labels}

    data_dict = utils.split_and_subsample_batch(data_dict, args,
                                                data_type=data_type)
    return data_dict


def variable_time_collate_fn1(batch, args, device=torch.device("cpu"),
                              data_type="train",
                              data_min=None, data_max=None,):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
		- record_id is a patient id
		- tt is a 1-dimensional tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
		- labels is a list of labels for the current patient, if labels are available. Otherwise None.
	Returns:
		combined_tt: The union of all time observations.
		combined_vals: (M, T, D) tensor containing the observed values.
		combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
		all processed to be used by NJODE
	"""
    D = batch[0][2].shape[1]
    combined_tt, inverse_indices = torch.unique(
        torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
    combined_tt = combined_tt.to(device)

    offset = 0
    combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_noise = torch.zeros([len(batch), len(combined_tt), D]).to(device)

    combined_labels = None
    N_labels = 1

    combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(
        float('nan'))
    combined_labels = combined_labels.to(device=device)

    for b, (record_id, tt, vals, mask, labels, noise) in enumerate(batch):
        tt = tt.to(device)
        vals = vals.to(device)
        mask = mask.to(device)
        noise = noise.to(device)
        if labels is not None:
            labels = labels.to(device)

        indices = inverse_indices[offset:offset + len(tt)]
        offset += len(tt)

        combined_vals[b, indices] = vals
        combined_mask[b, indices] = mask
        combined_noise[b, indices] = noise

        if labels is not None:
            combined_labels[b] = labels

    combined_vals, _, _ = utils.normalize_masked_data(combined_vals,
                                                      combined_mask,
                                                      att_min=data_min,
                                                      att_max=data_max)
    combined_noise, _, _ = utils.normalize_masked_data(
        combined_noise,
        combined_mask,
        att_min=0.,  # FK: since the data is already shifted, we
                     #   don't want to shift the noise, but only scale it
        att_max=data_max)

    combined_tt = combined_tt / 48.  # FK: 48 is the max amount of time

    combined_mask = combined_mask.detach().numpy()
    combined_vals = combined_vals.detach().numpy()
    combined_noise = combined_noise.detach().numpy()
    times = combined_tt.detach().numpy()
    times_val = None
    vals_val = None
    mask_val = None
    noise_val = None
    if data_type == "test":
        n_times_obs = len(times) // 2
        times_val = times[n_times_obs:]
        vals_val = combined_vals[:, n_times_obs:, :]
        mask_val = combined_mask[:, n_times_obs:, :]
        noise_val = combined_noise[:, n_times_obs:, :]
        times = times[:n_times_obs]
        combined_mask = combined_mask[:, :n_times_obs, :]
        combined_vals = combined_vals[:, :n_times_obs, :]
        combined_noise = combined_noise[:, :n_times_obs, :]
    X = []
    M = []
    time_ptr = [0]
    obs_idx = []
    counter = 0
    for t_ind, t in enumerate(times):
        for i in range(len(batch)):
            if combined_mask[i, t_ind, :].sum() > 0:
                counter += 1
                X.append(combined_vals[i, t_ind, :]+combined_noise[i, t_ind, :])
                M.append(combined_mask[i, t_ind, :])
                obs_idx.append(i)
        time_ptr.append(counter)

    # if eval_input_prob is given, append the eval data by randomly chosen data
    #   points in the left-out time interval (second half)
    if data_type == "test" and args.eval_input_prob:
        np.random.seed(args.eval_input_seed)
        times = list(times)
        for t_ind, t in enumerate(times_val):
            first = True
            for i in range(len(batch)):
                if mask_val[i, t_ind, :].sum() > 0:
                    if np.random.rand() < args.eval_input_prob:
                        counter += 1
                        X.append(vals_val[i, t_ind, :]+noise_val[i, t_ind, :])
                        M.append(mask_val[i, t_ind, :])
                        obs_idx.append(i)
                        if first:
                            times.append(t)
                            first = False
            time_ptr.append(counter)
        times = np.array(times)

    res = {
        "times": times,
        "batch_size": len(batch),
        'time_ptr': np.array(time_ptr),
        'obs_idx': torch.tensor(obs_idx, dtype=torch.long),
        'X': torch.tensor(np.array(X), dtype=torch.float32),
        'M': torch.tensor(np.array(M), dtype=torch.float32),
        "times_val": times_val,
        "vals_val": vals_val,
        "mask_val": mask_val,
        "combined_vals": combined_vals,
        "combined_mask": combined_mask,
        "combined_noise": combined_noise,
    }
    # path dimension (of vals_val and mask_val):
    # 	[batch_size, time_steps, dimension]
    return res


if __name__ == '__main__':
    torch.manual_seed(1991)

    dataset = PhysioNet('../data/training_data/physionet', train=False,
                        download=True)




# FK: the variable_time_collate_fn() needs args passed to it (hence doesn't
# 	work here), compare with parse_datasets_LODE.py
# dataloader = DataLoader(dataset, batch_size=10, shuffle=True,
# 						collate_fn=variable_time_collate_fn)
# print(dataloader.__iter__().next())
