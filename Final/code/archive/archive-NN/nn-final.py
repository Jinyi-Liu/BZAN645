import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as torch_optim
import torch.nn as nn
import torch.nn.functional as F
from functions import quadratic_weighted_kappa

path = "/app/Final/code"
# path = "."
# This is the dataset processed from the midterm
train_size = 14993
data_df = pd.read_csv(path + "/data/data_df_proc.csv")[:train_size]
data_df.head()

cols_to_drop = ["Name", "RescuerID", "VideoAmt", "Description", "PetID", "PhotoAmt"]
to_drop_columns = [
    "PetID",
    "Name",
    "RescuerID",
    "Description",
    "BreedName_full",
    "Breed1Name",
    "Breed2Name",
]
data_df.drop(cols_to_drop + to_drop_columns, axis=1, inplace=True)
# Fill missing values with mean
data_df.fillna(data_df.mean(), inplace=True)

# Embedding the categorical variables using nn.Embedding
cat_cols = [
    "Breed1",
    "Breed2",
    "Gender",
    "Color1",
    "Color2",
    "Color3",
    "State",
    "Breed_full",
    "Color_full",
    "hard_interaction",
]
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for cat_col in cat_cols:
    label_encoders[cat_col] = LabelEncoder()
    data_df[cat_col] = label_encoders[cat_col].fit_transform(data_df[cat_col])

# Normalize the continuous variables
# cont_cols = data_df.columns.difference(cat_cols + ["AdoptionSpeed"])
# data_df[cont_cols] = data_df[cont_cols].apply(
#     lambda x: (x - x.mean()) / x.std(), axis=0
# )

emb_c = {n: len(col.unique()) for n, col in data_df.items() if n in cat_cols}
emb_cols = emb_c.keys()  # names of columns chosen for embedding
emb_szs = [
    (c, min(20, (c + 1) // 2)) for _, c in emb_c.items()
]  # embedding sizes for the chosen columns

# Split data into train and validation by AdoptionSpeed and stratify
from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(
    data_df, test_size=0.2, random_state=42, stratify=data_df["AdoptionSpeed"]
)

X_train = train_df.drop(columns="AdoptionSpeed")
y_train = train_df["AdoptionSpeed"]
X_valid = valid_df.drop(columns="AdoptionSpeed")
y_valid = valid_df["AdoptionSpeed"]

n_cont = len(X_train.columns) - len(emb_cols)  # number of continuous columns


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
        x2 = x_cont

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


model = PetFinderModel(emb_szs, n_cont)
device = get_default_device()
to_device(model, device)

train_ds = PetFinderData(X_train, y_train, emb_cols)
valid_ds = PetFinderData(X_valid, y_valid, emb_cols)

# Get data into device
batch_size = 512
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)

train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)


# Train model
epochs = 500
history = train_loop(
    model, epochs=epochs, lr=0.00005, wd=0.0001, train_dl=train_dl, valid_dl=valid_dl
)
# Save model
torch.save(model.state_dict(), "./model-stratify.pt")
# Save history
history = np.array(history)
np.save("./history.npy", history)

import matplotlib.pyplot as plt

# range(epochs)
plt.plot(range(epochs), history[:, 0], label="train_loss")
plt.plot(range(epochs), history[:, 2], label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.legend()
# plt.show()
plt.savefig(path + "/figure/loss-statify.png")

plt.clf()
plt.plot(range(epochs), history[:, 1], label="train_kappa")
plt.plot(range(epochs), history[:, 3], label="val_kappa")
plt.xlabel("Epochs")
plt.ylabel("Kappa")
plt.title("Quadratic Weighted Kappa vs Epochs")
plt.legend()
# plt.show()
plt.savefig(path + "/figure/kappa-stratify.png")
