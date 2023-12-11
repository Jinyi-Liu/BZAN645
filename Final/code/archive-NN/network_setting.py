import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as torch_optim
import torch.nn as nn
import torch.nn.functional as F
from functions import quadratic_weighted_kappa


class PetFinderData(Dataset):
    def __init__(self, X, Y, emb_cols):
        X = X.copy()
        self.X1 = torch.tensor(
            X.loc[:, emb_cols].copy().values
        ).long()  # categorical columns
        self.X2 = torch.tensor(
            X.drop(columns=emb_cols).copy().values
        ).float()  # numerical columns
        self.y = torch.tensor(Y.values).to(torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


class PetFinderModel(nn.Module):
    def __init__(self, embedding_sizes, n_cont):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(categories, size) for categories, size in embedding_sizes]
        )
        n_emb = sum(
            e.embedding_dim for e in self.embeddings
        )  # length of all embeddings combined
        self.n_emb, self.n_cont = n_emb, n_cont
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 128)
        self.lin4 = nn.Linear(128, 32)
        self.lin5 = nn.Linear(32, 1)
        # self.lin4 = nn.Sequential(nn.Linear(30, 1), nn.Softmax(dim=1))
        self.bn1 = nn.ReLU()
        self.bn2 = nn.ReLU()
        self.bn3 = nn.ReLU()
        self.bn4 = nn.ReLU()
        self.output = nn.ReLU()
        self.emb_drop = nn.Dropout(0.2)
        self.drops = nn.Dropout(0.1)

    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x2 = self.bn1(x_cont)

        x = torch.cat([x, x2], 1)
        x = self.lin1(x)
        x = self.bn2(x)
        x = self.drops(x)
        x = self.lin2(x)
        x = self.bn3(x)
        x = self.drops(x)
        x = self.lin3(x)
        x = self.bn4(x)
        x = self.drops(x)
        x = self.lin4(x)
        x = self.output(x)
        x = self.lin5(x)
        # map x to [0,4]
        x = torch.sigmoid(x) * 4
        return x


def get_optimizer(model, lr=0.0001, wd=0.0):
    optim = torch_optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return optim


def train_model(model, optim, train_dl):
    model.train()
    total = 0
    sum_loss = 0
    correct = []
    for x1, x2, y in train_dl:
        batch = y.shape[0]
        # print(batch)
        output = model(x1, x2)
        # Use cross entropy loss for classification
        loss = F.mse_loss(output, y.view(-1, 1))
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += batch
        sum_loss += batch * (loss.item())

        pred = torch.round(output).cpu().detach().numpy().reshape(-1).astype(int)
        weighted_kappa = quadratic_weighted_kappa(
            y.view(-1, 1).cpu().detach().numpy().reshape(-1).astype(int), pred
        )
        correct.append(weighted_kappa)

    return sum_loss / total, np.mean(correct)


def val_loss(model, valid_dl):
    model.eval()
    total = 0
    sum_loss = 0
    correct = []
    for x1, x2, y in valid_dl:
        current_batch_size = y.shape[0]
        output = model(x1, x2)

        loss = F.mse_loss(output, y.view(-1, 1))
        sum_loss += current_batch_size * (loss.item())
        total += current_batch_size
        pred = torch.round(output)
        # convert to numpy array
        pred = pred.cpu().detach().numpy().reshape(-1).astype(int)
        weighted_kappa = quadratic_weighted_kappa(
            y.view(-1, 1).cpu().detach().numpy().reshape(-1).astype(int), pred
        )
        correct.append(weighted_kappa)
        # correct += (pred == y.view(-1, 1)).sum().item()

    # print("valid loss %.3f and kappa %.3f" % (sum_loss / total, np.mean(correct)))
    return sum_loss / total, np.mean(correct)


def train_loop(model, epochs, lr=0.01, wd=0.01, train_dl=None, valid_dl=None):
    optim = get_optimizer(model, lr=lr, wd=wd)
    history = []
    for i in range(epochs):
        train_loss, train_kappa = train_model(model, optim, train_dl)
        valua_loss, val_kappa = val_loss(model, valid_dl)
        if i % 50 == 0:
            print(
                "episode: %d\ntraining loss: %.3f, kappa: %.3f"
                % (i, train_loss, train_kappa)
            )
            print("validation loss: %.3f, kappa: %.3f" % (valua_loss, val_kappa))
        history.append([train_loss, train_kappa, valua_loss, val_kappa])
    return history
