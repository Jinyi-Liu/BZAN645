import pandas as pd

path = "/app/code"
# path='.'
# Load data from csv
data_df = pd.read_csv(path + "/data/train/train.csv")
# data_df.columns
cols_to_drop = ["Name", "RescuerID", "VideoAmt", "Description", "PetID", "PhotoAmt"]
data_df.drop(cols_to_drop, axis=1, inplace=True)
data_df["Type"] -= 1
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as torch_optim
import torch.nn as nn
import torch.nn.functional as F

# Embedding the categorical variables using nn.Embedding
cat_cols = ["Breed1", "Breed2", "Color1", "Color2", "Color3", "State"]

from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for cat_col in cat_cols:
    label_encoders[cat_col] = LabelEncoder()
    data_df[cat_col] = label_encoders[cat_col].fit_transform(data_df[cat_col])

emb_c = {n: len(col.unique()) for n, col in data_df.items() if n in cat_cols}
emb_cols = emb_c.keys()  # names of columns chosen for embedding
emb_szs = [
    (c, min(10, (c + 1) // 2)) for _, c in emb_c.items()
]  # embedding sizes for the chosen columns


class PetFinderData(Dataset):
    def __init__(self, X, Y, emb_cols):
        X = X.copy()
        self.X1 = X[emb_cols].astype(np.int64).values
        self.X2 = X.drop(columns=emb_cols).astype(np.float32).values
        self.y = Y.values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]

    # Split data into train and validation


train_df = data_df.iloc[: len(data_df) * 4 // 5, :]
valid_df = data_df.iloc[len(data_df) * 4 // 5 :, :]
train_df.shape, valid_df.shape

X_train = train_df.drop(columns="Type")
y_train = train_df["Type"]
X_valid = valid_df.drop(columns="Type")
y_valid = valid_df["Type"]

train_ds = PetFinderData(X_train, y_train, emb_cols)
valid_ds = PetFinderData(X_valid, y_valid, emb_cols)

n_cont = len(X_train.columns) - len(emb_cols)  # number of continuous columns


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
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 200)
        self.lin2 = nn.Linear(200, 70)
        self.lin3 = nn.Sequential(nn.Linear(70, 1), nn.Sigmoid())
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(200)
        self.bn3 = nn.BatchNorm1d(70)
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)

    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x2 = self.bn1(x_cont)
        x = torch.cat([x, x2], 1)
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = self.bn2(x)
        x = F.relu(self.lin2(x))
        x = self.drops(x)
        x = self.bn3(x)
        x = self.lin3(x)
        return x


model = PetFinderModel(emb_szs, n_cont)
device = get_default_device()
to_device(model, device)


def get_optimizer(model, lr=0.0001, wd=0.0):
    optim = torch_optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    return optim


def train_model(model, optim, train_dl):
    model.train()
    total = 0
    sum_loss = 0
    for x1, x2, y in train_dl:
        batch = y.shape[0]
        output = model(x1, x2)
        loss = F.cross_entropy(output, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += batch
        sum_loss += batch * (loss.item())
    return sum_loss / total


def val_loss(model, valid_dl):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    for x1, x2, y in valid_dl:
        current_batch_size = y.shape[0]
        out = model(x1, x2)
        loss = F.cross_entropy(out, y)
        sum_loss += current_batch_size * (loss.item())
        total += current_batch_size
        pred = torch.max(out, 1)[1]
        correct += (pred == y).float().sum().item()
    print("valid loss %.3f and accuracy %.3f" % (sum_loss / total, correct / total))
    return sum_loss / total, correct / total


def train_loop(model, epochs, lr=0.01, wd=0.01):
    optim = get_optimizer(model, lr=lr, wd=wd)
    for i in range(epochs):
        loss = train_model(model, optim, train_dl)
        print("training loss: ", loss)
        val_loss(model, valid_dl)


# Get data into device
batch_size = 64
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)

train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)
train_loop(model, epochs=10, lr=0.05, wd=0.001)
