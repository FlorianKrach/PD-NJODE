"""
author: Florian Krach

This code is based on the DeepLOB code provided in
https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/
"""
# load packages
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import sklearn
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.utils import data
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

import sys
import socket
import matplotlib.colors

sys.path.append("../")
sys.path.append("../NJODE/")
import configs.config as config
import NJODE.data_utils as data_utils

try:
    from telegram_notifications import send_bot_message as SBM
except Exception:
    from config import SendBotMessage as SBM


# =====================================================================================================================
# check whether running on computer or server
if 'ada-' not in socket.gethostname():
    SERVER = False
    N_CPUS = 1
    SEND = False
else:
    SERVER = True
    N_CPUS = 1
    SEND = True
    matplotlib.use('Agg')
print(socket.gethostname())
print('SERVER={}'.format(SERVER))


# ==============================================================================
# Global variables
CHAT_ID = config.CHAT_ID
ERROR_CHAT_ID = config.ERROR_CHAT_ID

data_path = config.data_path
saved_models_path = config.saved_models_path
flagfile = config.flagfile

METR_COLUMNS = ['epoch', 'train_time', 'eval_time', 'train_loss',
                'eval_loss', 'mse_eval_loss', 'classification_eval_loss']

ANOMALY_DETECTION = False
N_DATASET_WORKERS = 0
USE_GPU = False
# ==============================================================================

def prepare_x(data):
    df1 = data[:40, :].T
    return np.array(df1)

def get_label(data):
    lob = data[-5:, :].T
    return lob

def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX, dataY

def torch_data(x, y):
    x = torch.from_numpy(x)
    x = torch.unsqueeze(x, 1)
    y = torch.from_numpy(y)
    y = F.one_hot(y, num_classes=3)
    return x, y


class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, data, k, num_classes, T):
        """Initialization"""
        self.k = k
        self.num_classes = num_classes
        self.T = T

        x = prepare_x(data)
        y = get_label(data)
        x, y = data_classification(x, y, self.T)
        y = y[:,self.k] - 1
        self.length = len(x)

        x = torch.from_numpy(x)
        self.x = torch.unsqueeze(x, 1)
        self.y = torch.from_numpy(y)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index]


class deeplob(nn.Module):
    def __init__(self, y_len, device):
        super().__init__()
        self.y_len = y_len
        self.device = device

        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=0.01),
            #             nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # lstm layers
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, self.y_len)

    def forward(self, x):
        # h0: (number of hidden layers, batch size, hidden size)
        h0 = torch.zeros(1, x.size(0), 64).to(self.device)
        c0 = torch.zeros(1, x.size(0), 64).to(self.device)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)

        #         x = torch.transpose(x, 1, 2)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))

        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc1(x)
        forecast_y = torch.softmax(x, dim=1)

        return forecast_y


def batch_gd(
        model, criterion, optimizer, train_loader, test_loader, test_loader2,
        epochs, device):

    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    eval_f1_scores = np.zeros(epochs)
    best_test_loss = np.inf
    best_test_epoch = 0

    for it in tqdm(range(epochs)):
        model.train()
        t0 = datetime.now()
        train_loss = []
        for b in train_loader:
            inputs = b["samples"]
            targets = b["labels"][:, 0]
            # move data to GPU
            inputs, targets = inputs.to(device, dtype=torch.float), \
                              targets.to(device, dtype=torch.int64)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)

        model.eval()
        test_loss = []
        f1_scores = []
        for b in test_loader:
            inputs = b["samples"]
            targets = b["labels"][:, 0]
            true_labels = b["true_labels"][:, 0]
            inputs, targets = inputs.to(device, dtype=torch.float), \
                              targets.to(device, dtype=torch.int64)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())

            class_probs = outputs.detach().numpy()
            classes = np.argmax(class_probs, axis=1) - 1
            f1_scores.append(sklearn.metrics.f1_score(
                true_labels, classes,
                average="weighted"))
        test_loss = np.mean(test_loss)

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss

        do_eval = False
        if test_loss < best_test_loss:
            # torch.save(model, './best_val_model_pytorch')
            best_test_loss = test_loss
            best_test_epoch = it
            print('model saved')

        # eval model on entire eval set
        model.eval()
        for b in test_loader2:
            inputs = b["samples"]
            targets = b["labels"][:, 0]
            true_labels = b["true_labels"][:, 0]
            inputs, targets = inputs.to(device, dtype=torch.float), \
                              targets.to(device, dtype=torch.int64)
            outputs = model(inputs)
            class_probs = outputs.detach().numpy()
            classes = np.argmax(class_probs, axis=1) - 1
            print("classification report \n",
                  sklearn.metrics.classification_report(
                      true_labels, classes,))
            f1_score = sklearn.metrics.f1_score(
                true_labels, classes, average="weighted")
            eval_f1_scores[it] = f1_score

        dt = datetime.now() - t0
        print(f'Epoch {it}/{epochs}, Train Loss: {train_loss:.4f}, \
          Validation Loss: {test_loss:.4f}, Duration: {dt}, \
          Best Val Epoch: {best_test_epoch}')

    return train_losses, test_losses, eval_f1_scores


def train(
        anomaly_detection=None, n_dataset_workers=None, use_gpu=None,
        nb_cpus=None, send=None, model_id=1, epochs=50,
        batch_size=64, dataset_id=10, data_dict=None, test_size=0.2, seed=398,
        lr=0.0001):

    global ANOMALY_DETECTION, USE_GPU, SEND, N_CPUS, N_DATASET_WORKERS
    if anomaly_detection is not None:
        ANOMALY_DETECTION = anomaly_detection
    if use_gpu is not None:
        USE_GPU = use_gpu
    if send is not None:
        SEND = send
    if nb_cpus is not None:
        N_CPUS = nb_cpus
    if n_dataset_workers is not None:
        N_DATASET_WORKERS = n_dataset_workers

    if ANOMALY_DETECTION:
        # allow backward pass to print the traceback of the forward operation
        #   if it fails, "nan" backward computation produces error
        torch.autograd.set_detect_anomaly(True)
        torch.manual_seed(0)
        np.random.seed(0)
        # set seed and deterministic to make reproducible
        cudnn.deterministic = True

    # set number of CPUs
    torch.set_num_threads(N_CPUS)

    # get the device for torch
    if USE_GPU and torch.cuda.is_available():
        gpu_num = 0
        device = torch.device("cuda:{}".format(gpu_num))
        torch.cuda.set_device(gpu_num)
        print('\nusing GPU')
    else:
        device = torch.device("cpu")
        print('\nusing CPU')

    # load dataset-metadata
    if data_dict is not None:
        dataset, dataset_id = data_utils._get_dataset_name_id_from_dict(
            data_dict=data_dict)
        dataset_id = int(dataset_id)
    else:
        dataset = "LOB"
        dataset_id = int(data_utils._get_time_id(
            stock_model_name=dataset, time_id=dataset_id))
    dataset_metadata = data_utils.load_metadata(
        stock_model_name=dataset, time_id=dataset_id)
    use_volume = dataset_metadata['use_volume']
    input_size = dataset_metadata['LOB_level']*2*(1+use_volume) + 1

    # load raw data
    train_idx, val_idx = train_test_split(
        np.arange(dataset_metadata["nb_paths"]), test_size=test_size,
        random_state=seed, shuffle=False)
    data_train = data_utils.LOBDataset(time_id=dataset_id, idx=train_idx)
    data_val = data_utils.LOBDataset(time_id=dataset_id, idx=val_idx)

    # get data-loader
    collate_fn = data_utils.LOBCollateFnGen2()
    train_loader = DataLoader(
        dataset=data_train, collate_fn=collate_fn,
        shuffle=True, batch_size=batch_size, num_workers=N_DATASET_WORKERS)
    val_loader = DataLoader(
        dataset=data_val, collate_fn=collate_fn,
        shuffle=False, batch_size=batch_size, num_workers=N_DATASET_WORKERS)
    val_loader2 = DataLoader(
        dataset=data_val, collate_fn=collate_fn,
        shuffle=False, batch_size=len(data_val), num_workers=N_DATASET_WORKERS)
    print(data_train.samples.shape, data_train.eval_labels.shape)

    # get the model
    model = deeplob(y_len=3, device=device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses, eval_f1_scores = batch_gd(
        model, criterion, optimizer, train_loader, val_loader, val_loader2,
        epochs=epochs, device=device)

    path = config.data_path + "saved_models_DeepLOB/"
    config.makedirs(path)
    model_metric_file = "{}metric_id-{}.csv".format(path, model_id)
    df = pd.DataFrame(
        data={"train_loss": train_losses, "eval_loss": val_losses,
              "eval_f1_scores": eval_f1_scores})
    df.to_csv(model_metric_file)

    if send:
        files_to_send = [model_metric_file]
        caption = "DeepLOB - id={}".format(model_id)
        SBM.send_notification(
            text='finished DeepLOB training: id={}'.format(model_id),
            chat_id=config.CHAT_ID,
            files=files_to_send,
            text_for_files=caption)


if __name__ == '__main__':

    # train(
    #     anomaly_detection=False, n_dataset_workers=0, use_gpu=False,
    #     nb_cpus=1, send=True, model_id=1, epochs=50,
    #     batch_size=64, data_dict="LOB_dict1", test_size=0.2, seed=398,
    #     lr=0.0001)
    # train(
    #     anomaly_detection=False, n_dataset_workers=0, use_gpu=False,
    #     nb_cpus=1, send=True, model_id=2, epochs=50,
    #     batch_size=64, data_dict="LOB_dict_K_1", test_size=0.2, seed=398,
    #     lr=0.0001)
    # train(
    #     anomaly_detection=False, n_dataset_workers=0, use_gpu=False,
    #     nb_cpus=1, send=True, model_id=3, epochs=50,
    #     batch_size=64, data_dict="LOB_dict_K_4", test_size=0.2, seed=398,
    #     lr=0.0001)

    train(
        anomaly_detection=False, n_dataset_workers=0, use_gpu=False,
        nb_cpus=1, send=True, model_id=11, epochs=50,
        batch_size=64, data_dict="LOB_dict1_2", test_size=0.2, seed=398,
        lr=0.0001)
    train(
        anomaly_detection=False, n_dataset_workers=0, use_gpu=False,
        nb_cpus=1, send=True, model_id=12, epochs=50,
        batch_size=64, data_dict="LOB_dict_K_1_2", test_size=0.2, seed=398,
        lr=0.0001)
    train(
        anomaly_detection=False, n_dataset_workers=0, use_gpu=False,
        nb_cpus=1, send=True, model_id=13, epochs=50,
        batch_size=64, data_dict="LOB_dict_K_4_2", test_size=0.2, seed=398,
        lr=0.0001)


    pass
