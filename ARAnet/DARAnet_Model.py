import numpy as np
import json
from pymatgen.core import Structure, Element
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from pymatgen.io.vasp import Vasprun
from pymatgen.electronic_structure.dos import OrbitalType
import torch
from torch import nn
import random
import os


def data_read():
    print("data reading......\n")
    c2db_name = []
    json_name = []
    dos_data = []

    folder_path = './data/MXene/DOS_data/band_dos'
    file_names = os.listdir(folder_path)
    x_linear = np.linspace(-3, 3, num=2000)

    vasp = Vasprun("./data/MXene/DOS_data/vasprun.xml")
    cdos = vasp.complete_dos
    C_dos = cdos.get_element_spd_dos("C")
    F_dos = cdos.get_element_spd_dos("F")
    Diamane_energies = np.array(C_dos[OrbitalType(0)].energies) + 4.5562


    Diamane_C_s = np.interp(x_linear, Diamane_energies, list(C_dos[OrbitalType(0)].densities.values())[0])
    Diamane_C_p = np.interp(x_linear, Diamane_energies, list(C_dos[OrbitalType(1)].densities.values())[0])
    Diamane_C_d = np.interp(x_linear, Diamane_energies, list(C_dos[OrbitalType(2)].densities.values())[0])
    Diamane_F_s = np.interp(x_linear, Diamane_energies, list(F_dos[OrbitalType(0)].densities.values())[0])
    Diamane_F_p = np.interp(x_linear, Diamane_energies, list(F_dos[OrbitalType(1)].densities.values())[0])
    Diamane_F_d = np.interp(x_linear, Diamane_energies, list(F_dos[OrbitalType(2)].densities.values())[0])
    Diamane_T = Diamane_C_s + Diamane_C_p + Diamane_C_d + Diamane_F_s + Diamane_F_p + Diamane_F_d

    for file_name in tqdm(file_names[0:-1]):
        with open("./data/MXene/DOS_data/band_dos/"+file_name) as file:
            data = json.load(file)
        data_x = np.array(data["data_x"]) - data["efermi"]
        data_A_s = np.zeros(len(x_linear))
        data_B_s = np.zeros(len(x_linear))
        data_C_s = np.zeros(len(x_linear))
        data_A_p = np.zeros(len(x_linear))
        data_B_p = np.zeros(len(x_linear))
        data_C_p = np.zeros(len(x_linear))
        data_A_d = np.zeros(len(x_linear))
        data_B_d = np.zeros(len(x_linear))
        data_C_d = np.zeros(len(x_linear))
        data_T = np.zeros(len(x_linear))
        for s in data["symbols"]:
            if s not in ["H", "C", "N", "O", "F"]:
                for y in ["s", "p", "d"]:
                    try:
                        y_linear = np.interp(x_linear, data_x, data[s][y]["data_y"])
                    except:
                        continue
                    if y == "s":
                        data_A_s = data_A_s + y_linear
                        data_T = data_T + y_linear
                    elif y == "p":
                        data_A_p = data_A_p + y_linear
                        data_T = data_T + y_linear
                    elif y == "d":
                        data_A_d = data_A_d + y_linear
                        data_T = data_T + y_linear
            else:
                for y in ["s", "p", "d"]:
                    try:
                        y_linear = np.interp(x_linear, data_x, data[s][y]["data_y"])
                    except:
                        continue
                    if y == "s":
                        data_B_s = data_B_s + y_linear
                        data_T = data_T + y_linear
                    elif y == "p":
                        data_B_p = data_B_p + y_linear
                        data_T = data_T + y_linear
                    elif y == "d":
                        data_B_d = data_B_d + y_linear
                        data_T = data_T + y_linear
        structure = Structure.from_file("./data/MXene/DOS_data/poscar/{}.vasp".format(file_name[:-5]))
        min_e = 100
        interface_atom = Element("H")
        for i in range(len(structure.cart_coords)):
            if structure.cart_coords[i][2] <= min_e:
                min_e = structure.cart_coords[i][2]
                interface_atom = structure.species[i]
        for y in ["s", "p", "d"]:
            try:
                y_linear = np.interp(x_linear, data_x, data[str(interface_atom)][y]["data_y"])
            except:
                continue
            if y == "s":
                data_C_s = data_C_s + y_linear
                data_T = data_T + y_linear
            elif y == "p":
                data_C_p = data_C_p + y_linear
                data_T = data_T + y_linear
            elif y == "d":
                data_C_d = data_C_d + y_linear
                data_T = data_T + y_linear




        result_list = [data_T, Diamane_T,
                       data_A_s, data_A_p, data_A_d,
                       data_B_s, data_B_p, data_B_d,
                       data_C_s, data_C_p, data_C_d,
                       Diamane_C_s, Diamane_C_p, Diamane_C_d,
                       Diamane_F_s, Diamane_F_p, Diamane_F_d]

        result_array = np.array(result_list)
        scal = StandardScaler()
        result_array = np.array(scal.fit_transform(result_array.T)).T
        result_array = np.nan_to_num(result_array, nan=0.0)
        dos_data.append(result_array)
        c2db_name.append(data["c2db_id"])
        json_name.append(data["json_id"])

    dos_data = np.array(dos_data)
    print("data readed!\n")

    return dos_data, c2db_name, json_name

class Autoencoder(nn.Module):
    def __init__(self, h_s):
        super(Autoencoder, self).__init__()
        self.h_s = h_s

        self.conv1 = nn.Conv1d(17, 128, kernel_size=2, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.r1 = nn.ReLU()
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.r2 = nn.ReLU()
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, stride=2, padding=1)
        self.r3 = nn.ReLU()
        self.conv4 = nn.Conv1d(64, 32, kernel_size=3, stride=2, padding=1)
        self.r4 = nn.ReLU()
        self.conv5 = nn.Conv1d(32, 1, kernel_size=3, stride=1, padding=1)
        self.r5 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.drop1 = nn.Dropout(0.2)
        self.linear1 = nn.Linear(126*1, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.r6 = nn.ReLU()
        self.linear2 = nn.Linear(128, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.r7 = nn.ReLU()
        self.linear3 = nn.Linear(128, 64)
        self.r8 = nn.ReLU()
        self.linear4 = nn.Linear(64, h_s)
        self.r9 = nn.ReLU()


        self.delinear1 = nn.Linear(h_s, 64)
        self.der1 = nn.ReLU()
        self.delinear2 = nn.Linear(64, 128)
        self.der2 = nn.ReLU()
        self.delinear3 = nn.Linear(128, 128)
        self.der3 = nn.ReLU()
        self.delinear4 = nn.Linear(128, 126*1)
        self.der4 = nn.ReLU()
        self.deconv1 = nn.ConvTranspose1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.der5 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.der6 = nn.ReLU()
        self.deconv3 = nn.ConvTranspose1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.der7 = nn.ReLU()
        self.deconv4 = nn.ConvTranspose1d(128, 128, kernel_size=3, stride=2, padding=1)
        self.der8 = nn.ReLU()
        self.deconv5 = nn.ConvTranspose1d(128, 17, kernel_size=2, stride=2, padding=1)
        self.der9 = nn.Sigmoid()

        self.l1 = nn.Linear(h_s, 64)
        self.lr1 = nn.ReLU()
        self.l2 = nn.Linear(64, 128)
        self.lr2 = nn.ReLU()
        self.l3 = nn.Linear(128, 1)


    def forward(self, input_dos):
        x = self.conv1(input_dos)
        x = self.r1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.r2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.r3(x)
        x = self.conv4(x)
        x = self.r4(x)
        x = self.conv5(x)
        x = self.r5(x)
        original_shape = x.shape
        x1 = self.flatten(x)
        x = self.drop1(x1)
        x = self.linear1(x)
        x = self.r6(x)
        # x = self.bn3(x)
        x = self.linear2(x)
        x = self.r7(x)
        # x = self.bn4(x)
        x = self.linear3(x)
        x = self.r8(x)
        x = self.linear4(x)
        encoded = self.r9(x)

        x = self.delinear1(encoded)
        x = self.der1(x)
        x = self.delinear2(x)
        x = self.der2(x)
        x = self.delinear3(x)
        x = self.der3(x)
        x = self.delinear4(x)
        x = self.der4(x)
        x = x.reshape(original_shape)
        x = self.deconv1(x)
        x = self.der5(x)
        x = self.deconv2(x)
        x = self.der6(x)
        x = self.deconv3(x)
        x = self.der7(x)
        x = self.deconv4(x)
        x = self.der8(x)
        decoded = self.deconv5(x)
        # decoded = self.der9(decoded)

        x = self.l1(encoded)
        x = self.lr1(x)
        x = self.l2(x)
        x = self.lr2(x)
        regression = self.l3(x)
        return encoded, decoded, regression, x1
