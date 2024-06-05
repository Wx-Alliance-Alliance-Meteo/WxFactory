import matplotlib.pyplot as plt
import numpy as np
import subprocess
import netCDF4 as nc
import networkx as nx
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch_geometric.data as data
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv
# import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


def tuple_to_index(i, j, k):
    index = x1_max * x2_max * i + x1_max * k + j
    return index


class GCN(nn.Module):
    def __init__(self, num_features, hidden_dim, out_features, slope):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.conv5 = GCNConv(hidden_dim, hidden_dim)
        self.conv6 = GCNConv(hidden_dim, out_features)
        self.slope = slope

    def forward(self, x, edge_index):
        # First round of message passing
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, self.slope)

        # Second round of message passing
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x, self.slope)

        # Third round of message passing
        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x, self.slope)

        # Fourth round of message passing
        x = self.conv4(x, edge_index)
        x = F.leaky_relu(x, self.slope)

        # Fifth round of message passing
        x = self.conv5(x, edge_index)
        x = F.leaky_relu(x, self.slope)

        # Final message passing
        x = self.conv6(x, edge_index)

        return x

command = "cp ../../../Explore_ai/WxFactory/results/out_case6_base.nc nn_out_case6_base_trainUV_correct.nc"
output = subprocess.run(command, shell=True, capture_output=True, text=True)

file = nc.Dataset('nn_out_case6_base_trainUV_correct.nc', 'r+')

# Print the file format and dimensions
print("File format:", file.file_format)
print("Dimensions:", file.dimensions)

# Print the nf_height names
print("Variables:", file.variables.keys())

# Access nf (node features) and attributes
H = file.variables['h']
#HU1 = file.variables['hu1']
#HU2 = file.variables['hu2']

U = file.variables['U']
V = file.variables['V']

# Load the model
loaded_model = GCN(18, 256, 3, 0.01)
loaded_model.load_state_dict(torch.load('NNmodel.pth'))

node_features_testset = torch.load(
    "node_features_dataset_case6_base_DIM500.6.30.30.pt")
edges_testset = torch.load("edges_dataset_case6_base_DIM500.6.30.30.pt")

# node_features_testset = node_features_testset[:, :, :5]

#Train on H, U and V rather than H, hu1 and hu2
node_features_testset = torch.cat((node_features_testset[:, :, :1],node_features_testset[:, :, 3:]), dim=2)

radius_earth = 6371220.0
node_features_testset = node_features_testset.contiguous()

MEAN_1 = torch.mean(node_features_testset, dim=1).contiguous()
STD_1 = torch.std(node_features_testset, dim=1).contiguous()

shape = node_features_testset.shape

for x in range(0, 5): #shape[0]):
    print(x)
    for y in range(0, shape[1]):
        for z in range(0, shape[2]):
            node_features_testset[x, y, z] = (node_features_testset[x, y, z] - MEAN_1[x, z]) /(STD_1[x, z])

#`shape = node_features_testset.shape
#`H_mean = torch.mean(node_features_testset[1, :, 0])
#`H_std = torch.std(node_features_testset[1, :, 0])
#`HU1_mean = torch.mean(node_features_testset[1, :, 1])
#`HU1_std = torch.std(node_features_testset[1, :, 1])
#`HU2_mean = torch.mean(node_features_testset[1, :, 2])
#`HU2_std = torch.std(node_features_testset[1, :, 2])
#`U_mean = torch.mean(node_features_testset[1, :, 3])
#`U_std = torch.std(node_features_testset[1, :, 3])
#`V_mean = torch.mean(node_features_testset[1, :, 4])
#`V_std = torch.std(node_features_testset[1, :, 4])
#`for x in range(0, shape[0]):
#`    for y in range(0, shape[1]):
#`        node_features_testset[x, y, 0] = (
#`            node_features_testset[x, y, 0] - H_mean)/H_std
#`        node_features_testset[x, y, 1] = (
#`            node_features_testset[x, y, 1] - HU1_mean)/HU1_std
#`        node_features_testset[x, y, 2] = (
#`            node_features_testset[x, y, 2] - HU2_mean)/HU2_std
#`        node_features_testset[x, y, 3] = (
#`            node_features_testset[x, y, 3] - U_mean)/U_std
#`        node_features_testset[x, y, 4] = (
#`            node_features_testset[x, y, 4] - V_mean)/V_std
#`        node_features_testset[x, y, 11] /= (radius_earth * radius_earth)
#`        node_features_testset[x, y, 7] *= (radius_earth * radius_earth)
#`        node_features_testset[x, y, 8] *= (radius_earth * radius_earth)
#`        node_features_testset[x, y, 9] *= (radius_earth * radius_earth)
#`        node_features_testset[x, y, 10] *= (radius_earth * radius_earth)

print("Input Test node features shape: ", node_features_testset.shape)
print("Input Test Edges shape: ", edges_testset.shape)

Test_list = []

#for i in range(1, 200):
#    # for j in range(0, 10):
#    #    print(node_features_testset[i, j, :])
#    Test_list.append(data.Data(
#        x=node_features_testset[i, :, :], edge_index=edges_testset, y=node_features_testset[i+1, :, :3]))

# Put the model in evaluation mode
loaded_model.eval()

# Example usage of the loaded model
t = 1
panels = 6
x1_max = 30
x2_max = 30

#0 has rubbish
geometry_tensor = node_features_testset[1, :, -15:]
next_input = node_features_testset[1, :, :]

for i in range(0, panels):
    for j in range(0, x1_max):
        for k in range(0, x2_max):
            H[t, i, j, k] = next_input[tuple_to_index(
                i, j, k), 0] * STD_1[1, 0] + MEAN_1[1, 0]
            U[t, i, j, k] = next_input[tuple_to_index(
                i, j, k), 1]* STD_1[1, 1] + MEAN_1[1, 1]
            V[t, i, j, k] = next_input[tuple_to_index(
                i, j, k), 2]* STD_1[1, 2] + MEAN_1[1, 2]
            # U[t, i, j, k] = next_input[tuple_to_index(i, j, k), 3]
            # V[t, i, j, k] = next_input[tuple_to_index(i, j, k), 4]
t = t + 1
# t = 0
print(next_input.shape)
with torch.no_grad():
    for i in range(0, 200):
        print("t = ", t, "/200")
        outputs = loaded_model(next_input, edges_testset)
        next_input = torch.cat((outputs, geometry_tensor), dim=1)
        print(next_input.shape)
        for i in range(0, panels):
            for j in range(0, x1_max):
                for k in range(0, x2_max):
                    H[t, i, j, k] = next_input[tuple_to_index(
                        i, j, k), 0] * STD_1[1, 0] + MEAN_1[1, 0]
                    U[t, i, j, k] = next_input[tuple_to_index(
                        i, j, k), 1]* STD_1[1, 1] + MEAN_1[1, 1]
                    V[t, i, j, k] = next_input[tuple_to_index(
                        i, j, k), 2]* STD_1[1, 2] + MEAN_1[1, 2]
                    # U[t, i, j, k] = outputs[tuple_to_index(i, j, k), 3]
                    # V[t, i, j, k] = outputs[tuple_to_index(i, j, k), 4]
        t = t + 1

file.variables['h'] = H
#file.variables['hu1'] = HU1
#file.variables['hu2'] = HU2
file.variables['U'] = U
file.variables['V'] = V

file.close()

print("=======NN test output saved========")
