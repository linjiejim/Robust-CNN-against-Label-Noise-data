import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_dataset(dataset_path):

    dataset = np.load(dataset_path)
    img_size = dataset['Xtr'][0].shape[-2]

    n_train = len(dataset['Xtr'])
    n_test = len(dataset['Xts'])
    X_train, y_train_raw = dataset['Xtr'], dataset['Str']
    X_test, y_test_raw = dataset['Xts'], dataset['Yts']

    X_train = X_train.reshape([n_train, img_size, img_size, -1])
    X_train = np.rollaxis(X_train, 3, 1) / 127.5 - 1.0
    y_train_raw = y_train_raw.reshape([n_train, ])
    y_train_onehot = np.zeros((n_train, len(set(y_train_raw))))
    y_train_onehot[np.arange(n_train), y_train_raw] = 1

    X_test = X_test.reshape([n_test, img_size, img_size, -1])
    X_test = np.rollaxis(X_test, 3, 1) / 127.5 - 1.0
    y_test_raw = y_test_raw.reshape([n_test, ])

    return X_train, y_train_raw, y_train_onehot, X_test, y_test_raw

class SymmetricCrossEntropyLoss(nn.Module):
    def __init__(self, alpha, beta):
        super(SymmetricCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, outputs, targets):
        outputs = F.softmax(outputs)
        targets = targets.float()

        outputs_1 = outputs
        targets_1 = targets

        outputs_2 = outputs
        targets_2 = targets

        outputs_1 = torch.clamp(outputs_1, 1e-7, 1.0)
        targets_2 = torch.clamp(targets_2, 1e-4, 1.0)

        return self.alpha * torch.mean(-torch.sum(targets_1 * torch.log(outputs_1), dim=-1)) + self.beta * torch.mean(-torch.sum(outputs_2 * torch.log(targets_2), dim=-1))

class CrossEntropy(nn.Module):

    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, outputs, targets):
        outputs = F.softmax(outputs)
        outputs = torch.div(outputs, torch.sum(outputs, dim=-1, keepdim=True))
        outputs = torch.clamp(outputs, 1e-7, 1.0 - 1e-7)
        targets = targets.float()

        return torch.mean(-torch.sum(targets * torch.log(outputs), dim=-1))

class CrossEntropyMPE(nn.Module):
    def __init__(self, T):
        super(CrossEntropyMPE, self).__init__()
        self.T = torch.from_numpy(T).cuda().float()

    def forward(self, outputs, targets):
        outputs = F.softmax(outputs)
        outputs = torch.div(outputs, torch.sum(outputs, dim=-1, keepdim=True))
        outputs = torch.clamp(outputs, 1e-7, 1.0 - 1e-7)
        targets = targets.float()

        return torch.mean(-torch.sum(targets * torch.log(torch.mm(outputs, self.T).float()), dim=-1))

class BootstrapLoss(nn.Module):
    def __init__(self, beta):
        super(BootstrapLoss, self).__init__()
        self.beta = beta

    def forward(self, outputs, targets):
        outputs = F.softmax(outputs)
        outputs = torch.div(outputs, torch.sum(outputs, dim=-1, keepdim=True))
        outputs = torch.clamp(outputs, 1e-7, 1.0 - 1e-7)
        targets = targets.float()

        return torch.mean(-torch.sum((self.beta * targets + (1 - self.beta) * outputs) * torch.log(outputs), dim=-1))

def augment(X_train, y_train, y_train_onehot):
    flip_X = np.flip(X_train, 3)
    X_train = np.concatenate([X_train, flip_X])
    y_train = np.concatenate([y_train, y_train])
    y_train_onehot = np.concatenate([y_train_onehot, y_train_onehot])
    return X_train, y_train, y_train_onehot

def get_loss(loss_name, opt):
    if loss_name == "ce":
        return CrossEntropy()
    elif loss_name == 'symmetric':
        return SymmetricCrossEntropyLoss(alpha=0.01, beta=1.0)
    elif loss_name == "mpe":
        return CrossEntropyMPE(opt.trans_matrix)
    elif loss_name == 'bootstrap':
        return BootstrapLoss(beta=0.5)
    else:
        raise NotImplementedError

def log_to_file(content, file_path):
    with open(file_path, 'a') as file:
        file.writelines(content)
        file.writelines("\r\n")

def compute_T(model, inputs, n_class):

    '''Compute the transiton matrix for the nn model.
    Input:
        model:      nn.Model
        inputs:     tensor[n_sample, n_channel, (img_shape)]
        n_class:    the number of prediction results
    Output:
        T: transition matrix
    '''

    # Transition matrix
    T = torch.zeros((n_class, n_class))

    # compute the prediction of the entired dataset
    with torch.no_grad():
        outputs = model(torch.Tensor(np.array(inputs)).float().cuda())
        outputs = F.softmax(outputs)
        print(outputs.shape)
        prediction = torch.argmax(outputs, axis=1)

    # group data by predicted labels
    for i in range(n_class):
        class_idx = (prediction == i)

        # the probalility P(Y^=i | X=Xi)
        Xi_outputs = outputs[class_idx]

        # average the probalility
        Xi_outputs_mean = torch.mean(Xi_outputs, dim=0)

        # store results to the i(th) column
        T[i, :] = Xi_outputs_mean

    return T.detach().numpy()

def compute_T2(model, X_clean, n_class):
    '''Compute the transiton matrix for the nn model.
    Input:
        model:      nn.Model
        inputs:     tensor[n_sample, n_channel, (img_shape)]
        n_class:    the number of prediction results
    Output:
        T: transition matrix
    '''
    # Transition matrix

    T = torch.zeros((n_class, n_class))

    # compute the prediction of the entired dataset
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        outputs = model(torch.Tensor(np.array(X_clean)).float().to(device))
        outputs = F.softmax(outputs)

    # print(prediction)
    # group data by predicted labels
    index = int(len(outputs) * 0.1)
    for i in range(n_class):
        xi_max_idx = outputs[:,i].argsort(descending=True)[index]

        # print(xi.shape)
        xi_max = outputs[xi_max_idx]

        T[i,:] = xi_max

    return T.detach().numpy()