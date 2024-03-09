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
data = pd.read_csv("./data/result/knn_0.csv")
print(data)

init = []
for j in [5]:
    dd = pd.read_csv("./data/result/knn_traindata_{}.csv".format(j)).values.reshape([1, -1])[0]
    for na in dd:
        init.append(na)
print(init)
init = np.unique(np.array(init))
print(init)
print(len(init))
print(len(init)/len(data))

Projected_target = "n"

h_s = 2
a = 0.35
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
for train_index, val_index in kf.split(init):
    train_name = np.array(init)[train_index]
    val_name = np.array(init)[val_index]
    index += 1
    print("======== {} / {} ======".format(index, k))
    init_x_train = []
    init_y_train = []
    init_x_test = []
    init_y_test = []
    init_x_val = []
    init_y_val = []
    init_all_train = []
    train_mask = []
    setup_seed()
    for j in data.values:
        if j[0] in train_name:
            init_x_train.append(j[3:])
            init_y_train.append([j[1]/100.0])
            init_all_train.append(j[3:])
            train_mask.append(True)
        else:
            if j[0] in val_name:
                init_x_val.append(j[3:])
                init_y_val.append([j[1]/100.0])
                init_all_train.append(j[3:])
                train_mask.append(False)
            else:
                init_x_test.append(j[3:])
                init_y_test.append([j[1]/100.0])
                init_all_train.append(j[3:])
                train_mask.append(False)
    print("Train set size： ", len(init_x_train), "  ", len(init_x_train) / len(init_all_train))
    print("Validation Set Size： ", len(init_x_val), "  ", len(init_x_val) / len(init_all_train))
    print("Test set size： ", len(init_x_test), "  ", len(init_x_test) / len(init_all_train))

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
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001, weight_decay=0.0001)
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    # criterion_mse = nn.L1Loss()
    # criterion_l1 = nn.MSELoss()

    num_epochs = 500
    best_loss_rg_mae = 10000
    best_loss_rg_mse = 10000
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
        autoencoder.eval()
        _, _, regression_t = autoencoder(init_x_test)
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
print("="*10, "h_s:{}, a:{}".format(h_s, a), "="*10)
print("Best [MAE] of test set for {}-fold cross-validation: ".format(k),
      [float(test_best_MAE)])

fig, ax = plt.subplots(1, 2)
val_data = np.array(val_data)
test_data = np.array(test_data)
ax[0].scatter(val_data[:, 0], val_data[:, 1], s=11, c='#2a3371')
ax[1].scatter(test_data[:, 0], test_data[:, 1], s=11, c='#2a3371')

ax[0].plot([np.min(val_data[:, 1]) - 1, np.max(val_data[:, 1]) + 1],
           [np.min(val_data[:, 1]) - 1, np.max(val_data[:, 1]) + 1], color='#da82a3')
ax[1].plot([np.min(test_data[:, 1]) - 1, np.max(test_data[:, 1]) + 1],
           [np.min(test_data[:, 1]) - 1, np.max(test_data[:, 1]) + 1], color='#da82a3')

ax[0].set_xlabel('Predicted Values', fontsize=9)
ax[0].set_ylabel('True Values', fontsize=9)
ax[1].set_xlabel('Predicted Values', fontsize=9)

ax[0].set_title('val set', fontsize=9)
ax[1].set_title('test set', fontsize=9)
# ax[1].set_xlim(0, 1500)
# ax[1].set_ylim(0, 1500)
# ax[0].set_xlim(0, 1500)
# ax[0].set_ylim(0, 1500)
plt.savefig("./data/FAVAL_ARAnet_test_{}_{}_fold_1.png".format("matbench_steels", k))
plt.show()