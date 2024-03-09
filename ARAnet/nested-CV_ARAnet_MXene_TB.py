import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
# plt.style.use(['science', 'no-latex'])
# plt.rcParams['font.family'] = "Arial"
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from sklearn.manifold import TSNE
import numpy as np
import random
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
init = pd.read_csv("./data/MXene/Mxene_nested.CSV")
data = pd.read_csv("./data/MXene/MXene_att.csv", index_col=0)
Projected_target = "T"
print(device)
# data = data.drop(columns=["workfunction"])
data = data.values
# random.shuffle(data)
values = init[Projected_target].values
ID = init["ID"].values
print(values)
ID_values = {}
for i in range(len(values)):
    ID_values[ID[i]] = values[i]


h_s = 15
a = 0.05

k_loss_mae = []
k_loss_mse = []
data_X = []
data_Y = []
val_loss = []

k = 10

kf = KFold(n_splits=k, shuffle=True)
setup_seed()

for in_index, test_index in kf.split(ID):
    test_data = np.array(ID)[test_index]
    in_data = np.array(ID)[in_index]
    k = 9
    kf = KFold(n_splits=k, shuffle=True)

    for train_index, val_index in kf.split(in_data):

        train_data = np.array(in_data)[train_index]
        val_data = np.array(in_data)[val_index]
        setup_seed()
        init_x_train = []
        init_y_train = []
        init_x_test = []
        init_y_test = []
        init_x_val = []
        init_y_val = []
        train_mask = []
        init_all_train = []

        for j in data:
            if j[0] in val_data:
                init_x_val.append(j[3:])
                init_y_val.append([ID_values[j[0]]])
                train_mask.append(False)
                init_all_train.append(j[3:])
            elif j[0] in train_data:
                init_x_train.append(j[3:])
                init_y_train.append([ID_values[j[0]]])
                train_mask.append(True)
                init_all_train.append(j[3:])
            elif j[0] in test_data:
                init_x_test.append(j[3:])
                init_y_test.append([ID_values[j[0]]])
                train_mask.append(False)
                init_all_train.append(j[3:])
            else:
                train_mask.append(False)
                init_all_train.append(j[3:])


        scaler = StandardScaler()
        scaler.fit(init_all_train)
        init_all_train = scaler.transform(init_all_train)
        init_x_val = scaler.transform(init_x_val)
        init_x_test = scaler.transform(init_x_test)

        input_tensor = torch.tensor(init_all_train, dtype=torch.float32).to(device)
        init_x_val = torch.tensor(init_x_val, dtype=torch.float32).to(device)
        init_x_test = torch.tensor(init_x_test, dtype=torch.float32).to(device)
        init_y_train = torch.tensor(init_y_train, dtype=torch.float32).to(device)
        init_y_val = torch.tensor(init_y_val, dtype=torch.float32).to(device)
        init_y_test = torch.tensor(init_y_test, dtype=torch.float32).to(device)

        input_size = input_tensor.shape[1]
        autoencoder = Autoencoder(input_size, h_s).to(device)
        optimizer = optim.Adam(autoencoder.parameters(), lr=0.001, weight_decay=0.001)
        criterion_l1 = nn.L1Loss()
        criterion_mse = nn.MSELoss()

        num_epochs = 1000

        best_val_loss = 100
        best_val_model = None

        for epoch in tqdm(range(num_epochs)):
            autoencoder.train()
            encoded, decoded, regression = autoencoder(input_tensor)
            autoencoder_loss = criterion_l1(decoded, input_tensor)
            regression_loss = criterion_l1(regression[train_mask], init_y_train)
            a = a  # 0.2
            total_loss = (1 - a) * autoencoder_loss + a * regression_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            with torch.no_grad():
                autoencoder.eval()
                encoded_a, decoded_a, _ = autoencoder(input_tensor)
                _, _, regression_m = autoencoder(init_x_val)
                regression_loss_m = criterion_l1(regression_m, init_y_val)
                regression_loss_m_mse = criterion_mse(regression_m, init_y_val)
                # best_val_model = autoencoder
                if regression_loss_m < best_val_loss:
                    best_val_loss = regression_loss_m
                    best_val_model = autoencoder
        with torch.no_grad():
            best_val_model.eval()
            # encoded_a, decoded_a, _ = best_val_model(input_tensor)
            _, _, regression_m = best_val_model(init_x_test)
            regression_loss_m = criterion_l1(regression_m, init_y_test)
            regression_loss_m_mse = criterion_mse(regression_m, init_y_test)
            k_loss_mae.append(regression_loss_m.cpu().numpy())
            k_loss_mse.append(regression_loss_m_mse.cpu().numpy())
            for v in range(len(regression_m)):
                data_X.append(init_y_test.cpu().numpy()[v])
                data_Y.append(regression_m.cpu().numpy()[v])

    # print(k_loss_mae, k_loss_mse)
print("{} ---------- {}".format(h_s, a))
print("model num: ", len(k_loss_mae), len(k_loss_mse))
print("Mean [MAE, MSE] of validation set for nested-CV {}-fold cross-validation: ".format(k), "[", np.mean(np.array(k_loss_mae)), np.mean(np.array(k_loss_mse)), "]")

fig, ax = plt.subplots(1, 1, figsize=(4, 4))

ax.scatter(data_X, data_Y, s=11, c='#2a3371')

ax.plot([np.min(data_X)-0.2, np.max(data_Y)+0.2],
           [np.min(data_X)-0.2, np.max(data_Y)+0.2], color='#da82a3')

ax.set_ylabel('Predicted Values', fontsize=9)
ax.set_xlabel('True Values', fontsize=9)
plt.savefig("./data/MXene/nested-CV_ARAnet_{}_{}_fold.png".format("TB", k))
plt.show()