import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold
import os
import random

def setup_seed(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Autoencoder(nn.Module):
    def __init__(self, input_size, h_s):
        super(Autoencoder, self).__init__()
        self.e1 = nn.Linear(input_size, 128)
        self.ebn1 = nn.BatchNorm1d(128)
        self.t1 = nn.ReLU()
        self.e2 = nn.Linear(128, 128)
        self.ebn2 = nn.BatchNorm1d(128)
        self.t2 = nn.ReLU()
        self.e3 = nn.Linear(128, 64)
        self.t3 = nn.ReLU()
        self.e4 = nn.Linear(64, h_s)
        self.t4 = nn.ReLU()

        self.d1 = nn.Linear(h_s, 64)
        self.a1 = nn.ReLU()
        self.d2 = nn.Linear(64, 128)
        self.a2 = nn.ReLU()
        self.d3 = nn.Linear(128, 128)
        self.a3 = nn.ReLU()
        self.d4 = nn.Linear(128, input_size)

        self.l1 = nn.Linear(h_s, 64)
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(64, 128)
        self.r2 = nn.ReLU()
        self.l3 = nn.Linear(128, 1)

    def forward(self, x1):
        e = self.e1(x1)
        e = self.ebn1(e)
        e = self.t1(e)
        e = self.e2(e)
        e = self.ebn2(e)
        e = self.t2(e)
        e1 = self.e3(e)
        e = self.t3(e1)
        e = self.e4(e)
        encoded = self.t4(e)
        d = self.d1(encoded)
        d = self.a1(d)
        d = self.d2(d)
        d = self.a2(d)
        d = self.d3(d)
        d = self.a3(d)
        decoded = self.d4(d)

        r = self.l1(encoded)
        r = self.r1(r)
        r = self.l2(r)
        r = self.r2(r)
        regression = self.l3(r)

        return encoded, decoded, regression


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using ", device)
init = pd.read_csv("./data/MXene/Mxene_train.CSV")
test_init = pd.read_csv("./data/MXene/MXene_test_band_e.CSV")
data = pd.read_csv("./data/MXene/MXene_att.csv", index_col=0)
Projected_target = "band_e"

h_s = 10
a = 0.05

data = data.values
values = init[Projected_target].values
values_test = test_init[Projected_target].values
ID = init["ID"].values
test_ID = test_init["ID"].values
ID_values = {}
for i in range(len(values)):
    ID_values[ID[i]] = values[i]
ID_values_test = {}
for i in range(len(values_test)):
    ID_values_test[test_ID[i]] = values_test[i]

k = 10 # len(values)
kf = KFold(n_splits=k, shuffle=True)
setup_seed()
test_loss_best = 100
test_best_MAE = 0
test_best_MSE = 0

val_loss = []
test_loss = []

val_data = []
test_data = []

print("Training......")
index = 0
for train_index, val_index in kf.split(ID):
    index += 1
    print("======== {} / {} ======".format(index, k))
    i = np.array(ID)[val_index]
    init_x_train = []
    init_y_train = []
    init_x_test = []
    init_y_test = []
    init_x_val = []
    init_y_val = []
    train_mask = []
    ID_train = []
    ID_test = []
    ID_val = []
    init_all_train = []
    label = []
    setup_seed()
    for j in data:
        if j[0] in i:
            init_x_val.append(j[3:])
            init_y_val.append([ID_values[j[0]]])
            ID_val.append([j[0]])
            train_mask.append(False)
            init_all_train.append(j[3:])
            label.append(ID_values[j[0]])
        elif (j[0] in ID) and (j[0] not in i):
            init_x_train.append(j[3:])
            init_y_train.append([ID_values[j[0]]])
            ID_train.append([j[0]])
            train_mask.append(True)
            init_all_train.append(j[3:])
            label.append(ID_values[j[0]])
        elif j[0] in test_ID:
            init_x_test.append(j[3:])
            init_y_test.append([ID_values_test[j[0]]])
            ID_test.append([j[0]])
            train_mask.append(False)
            init_all_train.append(j[3:])
            label.append(np.nan)
        else:
            train_mask.append(False)
            init_all_train.append(j[3:])

    init_all_train = np.array(init_all_train)
    init_x_val = np.array(init_x_val)
    init_x_test = np.array(init_x_test)

    scaler = StandardScaler()
    scaler.fit(init_all_train)

    init_all_train = scaler.transform(init_all_train)
    init_x_val = scaler.transform(init_x_val)
    init_x_test = scaler.transform(init_x_test)

    input_tensor = torch.tensor(init_all_train, dtype=torch.float32).to(device)
    init_x_val = torch.tensor(init_x_val, dtype=torch.float32).to(device)
    init_y_train = torch.tensor(init_y_train, dtype=torch.float32).to(device)
    init_y_val = torch.tensor(init_y_val, dtype=torch.float32).to(device)
    init_x_test = torch.tensor(init_x_test, dtype=torch.float32).to(device)
    init_y_test = torch.tensor(init_y_test, dtype=torch.float32).to(device)

    input_size = input_tensor.shape[1]

    autoencoder = Autoencoder(input_size, h_s).to(device)
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.0001, weight_decay=0.0001)
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()

    num_epochs = 2000
    best_loss_rg_mae = 100
    best_loss_rg_mse = 100
    best_data = None
    best_model = None
    for epoch in tqdm(range(num_epochs)):
        autoencoder.train()
        encoded, decoded, regression = autoencoder(input_tensor)
        autoencoder_loss = criterion_l1(decoded, input_tensor)
        regression_loss = criterion_l1(regression[train_mask], init_y_train)
        a = a
        total_loss = (1 - a) * autoencoder_loss + a * regression_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            autoencoder.eval()
            _, _, regression_m = autoencoder(init_x_val)
            regression_loss_m = criterion_l1(regression_m, init_y_val)
            regression_loss_m_mse = criterion_mse(regression_m, init_y_val)

            if regression_loss_m <= best_loss_rg_mae:
                best_loss_rg_mae = regression_loss_m
                best_loss_rg_mse = regression_loss_m_mse
                best_data = regression_m
                best_model = autoencoder
    for v in range(len(best_data)):
        val_data.append([best_data.cpu().numpy()[v][0], init_y_val.cpu().numpy()[v][0]])


    with torch.no_grad():
        _, _, regression_t = best_model(init_x_test)
        regression_loss_t = criterion_l1(regression_t, init_y_test)
        regression_loss_t_mse = criterion_mse(regression_t, init_y_test)
    if regression_loss_t < test_loss_best:
        test_data = []
        test_loss_best = regression_loss_t
        test_best_MAE = regression_loss_t
        test_best_MSE = regression_loss_t_mse
        for v in range(len(regression_t)):
            test_data.append([regression_t.cpu().numpy()[v][0], init_y_test.cpu().numpy()[v][0]])
    val_loss.append([best_loss_rg_mae.cpu().numpy(), best_loss_rg_mse.cpu().numpy()])
    test_loss.append([regression_loss_t.cpu().numpy(), regression_loss_t_mse.cpu().numpy()])
print("Mean [MAE, MSE] of validation set for {}-fold cross-validation: ".format(k), np.mean(val_loss, axis=0))
print("Mean [MAE, MSE] of test set for {}-fold cross-validation: ".format(k), np.mean(test_loss, axis=0))
print("Best [MAE, MSE] of test set for {}-fold cross-validation: ".format(k), [float(test_best_MAE), float(test_best_MSE)])


fig, ax = plt.subplots(1, 2, figsize=(4, 2))
val_data = np.array(val_data)
test_data = np.array(test_data)

ax[0].scatter(val_data[:,0], val_data[:,1], s=11, c='#2a3371')
ax[1].scatter(test_data[:,0], test_data[:,1], s=11, c='#2a3371')

ax[0].plot([np.min(val_data[:, 1])-1, np.max(val_data[:, 1])+1],
           [np.min(val_data[:, 1])-1, np.max(val_data[:, 1])+1], color='#da82a3')
ax[1].plot([np.min(test_data[:, 1])-1, np.max(test_data[:, 1])+1],
           [np.min(test_data[:, 1])-1, np.max(test_data[:, 1])+1], color='#da82a3')

ax[0].set_xlabel('Predicted Values', fontsize=9)
ax[0].set_ylabel('True Values', fontsize=9)
ax[1].set_xlabel('Predicted Values', fontsize=9)

ax[0].set_title('val set', fontsize=9)
ax[1].set_title('test set', fontsize=9)
plt.savefig("./data/MXene/ARAnet_test_{}_{}_fold.png".format(Projected_target, k))
plt.show()