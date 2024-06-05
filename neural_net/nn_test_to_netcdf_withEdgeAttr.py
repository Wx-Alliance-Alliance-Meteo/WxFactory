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

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.layer_norm(self.fc2(x))
        return x


class GNNLayer(MessagePassing):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim):
        super(GNNLayer, self).__init__(aggr='add')  # "Add" aggregation.
        self.node_mlp = MLP(node_input_dim, hidden_dim, hidden_dim)
        self.edge_mlp = MLP(edge_input_dim, hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.edge_mlp(edge_attr)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.node_mlp(aggr_out)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class Processor(nn.Module):
    def __init__(self, hidden_dim, num_layers=9):
        super(Processor, self).__init__()
        self.layers = nn.ModuleList(
            [GNNLayer(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)])

    def forward(self, x, edge_index, edge_attr):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr) + x  # Residual connection
        return x


class Decoder(nn.Module):
    def __init__(self, hidden_dim, node_output_dim):
        super(Decoder, self).__init__()
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, node_output_dim)
        # Account for the skip connection
        self.fc = nn.Linear(8 + node_output_dim, node_output_dim)

    def forward(self, x, edge_index, edge_attr, initial_state):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        # Concatenate along the feature dimension
        x = torch.cat([x, initial_state], dim=-1)
        return self.fc(x)


class GNNModel(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, node_output_dim):
        super(GNNModel, self).__init__()
        self.encoder_node = Encoder(node_input_dim, hidden_dim)
        self.encoder_edge = Encoder(edge_input_dim, hidden_dim)
        self.processor = Processor(hidden_dim)
        self.decoder = Decoder(hidden_dim, node_output_dim)

    def forward(self, x, edge_index, edge_attr, initial_state):
        encoded_x = self.encoder_node(x, edge_index)
        encoded_edge_attr = self.encoder_edge(edge_attr, edge_index)

        processed = self.processor(encoded_x, edge_index, encoded_edge_attr)
        predicted_change = self.decoder(
            processed, edge_index, encoded_edge_attr, initial_state)
        new_state = initial_state[:, :3] + predicted_change
        return new_state


# Example usage:
node_input_dim = 8
edge_input_dim = 9
hidden_dim = 256
node_output_dim = 3
command = "cp ../../../Explore_ai/WxFactory/results/out_case6_base.nc nn_out_case6_base_trainUV_correct_2.nc"
output = subprocess.run(command, shell=True, capture_output=True, text=True)

file = nc.Dataset('nn_out_case6_base_trainUV_correct_2.nc', 'r+')

# Print the file format and dimensions
print("File format:", file.file_format)
print("Dimensions:", file.dimensions)

# Print the nf_height names
print("Variables:", file.variables.keys())

# Access nf (node features) and attributes
H = file.variables['h']
# HU1 = file.variables['hu1']
# HU2 = file.variables['hu2']
U = file.variables['U']
V = file.variables['V']

# Load the model
# loaded_model = GCN(18, 256, 3, 0.01)
loaded_model = GNNModel(node_input_dim, edge_input_dim,
                        hidden_dim, node_output_dim)

loaded_model.load_state_dict(torch.load('NNmodel.pth'))

node_features_testset = torch.load(
    "node_features_dataset_case6_base_DIM502.6.30.30.pt")
edges_testset = torch.load("edges_dataset_case6_base_DIM502.6.30.30.pt")

node_features_testset = node_features_testset.contiguous()

# Train on H, U and V rather than H, hu1 and hu2

# INPUT : [0] H, [1] Hu1, [2] Hu2, [3] U, [4] V, [5] lats, [6] lons, [7] sinlat, [8] coslat, [9] sinlon, [10] coslon, [11] det(g)

# OUTPUT : [0] H, [1] U, [2] V, [3] det(g), [4] coslat, [5] sinlon, [6] coslon, [7] sinlat

print(node_features_testset.shape)
# Removes Hu1, and Hu2
node_features_testset = torch.cat(
    (node_features_testset[:, :, :1], node_features_testset[:, :, 3:]), dim=2)
# Removes lats, and lons
node_features_testset = torch.cat(
    (node_features_testset[:, :, :2], node_features_testset[:, :, 4:]), dim=2)
# Swaps det(g) with sinlat
temp = node_features_testset[:, :, 3].clone()
node_features_testset[:, :, 3] = node_features_testset[:, :, 7]
node_features_testset[:, :, 7] = temp
print(node_features_testset.shape)

print("======DEBUG INFO=====")
print(torch.mean(node_features_testset[10, :, 0]))
print(torch.std(node_features_testset[10, :, 0]))
print(torch.mean(node_features_testset[10, :, 1]))
print(torch.std(node_features_testset[10, :, 1]))
print(torch.mean(node_features_testset[10, :, 2]))
print(torch.std(node_features_testset[10, :, 2]))
print(torch.mean(node_features_testset[10, :, 3]))
print(torch.std(node_features_testset[10, :, 3]))

MEAN_1 = torch.mean(node_features_testset, dim=1).contiguous()
STD_1 = torch.std(node_features_testset, dim=1).contiguous()

shape = node_features_testset.shape
for x in range(0, 2):
    print("++ Normalizing Input data for H, U, V, det(g) for t = ", x)
    for y in range(0, shape[1]):
        for z in range(0, 4):
            node_features_testset[x, y, z] = (
                node_features_testset[x, y, z] - MEAN_1[x, z]) / (STD_1[x, z])

print("======DEBUG INFO=====")
print(torch.mean(node_features_testset[10, :, 0]))
print(torch.std(node_features_testset[10, :, 0]))
print(torch.mean(node_features_testset[10, :, 1]))
print(torch.std(node_features_testset[10, :, 1]))
print(torch.mean(node_features_testset[10, :, 2]))
print(torch.std(node_features_testset[10, :, 2]))
print(torch.mean(node_features_testset[10, :, 3]))

print("Input Test node features shape: ", node_features_testset.shape)
print("Input Test Edges shape: ", edges_testset.shape)

Test_list = []

# for i in range(1, 200):
#    # for j in range(0, 10):
#    #    print(node_features_testset[i, j, :])
#    Test_list.append(data.Data(
#        x=node_features_testset[i, :, :], edge_index=edges_testset, y=node_features_testset[i+1, :, :3]))

# Put the model in evaluation mode
loaded_model.eval()

edge_attr_tensor = None

# Iterate over each first sample in the node_features_dataset
for i in range(1, 2):
    # Get the node features for the current sample
    node_features = node_features_testset[i]

    # Get the node indices for all edges
    node1_indices = edges_testset[0]
    node2_indices = edges_testset[1]

    # Gather the features for node1 and node2 for all edges
    node1_features = node_features[node1_indices]
    node2_features = node_features[node2_indices]
    print("++ Creating Edge Attribute tensor for t = ", i)

    # Concatenate the features along the last dimension and add the great circle distance
    great_circle_distance = torch.arccos(node1_features[:, 7]*node2_features[:, 7] + node1_features[:, 4]*node2_features[:, 4]*(
        node1_features[:, 5]*node2_features[:, 5] + node1_features[:, 6]*node2_features[:, 6])).view(21600, 1)
    print("++ Great circle dist mean: ", torch.mean(great_circle_distance))
    print("++ Great circle dist std: ", torch.std(great_circle_distance))
    edge_attr_tensor = torch.cat(
        (node1_features[:, -4:], great_circle_distance, node2_features[:, -4:]), dim=1)

# Example usage of the loaded model
t = 1
panels = 6
x1_max = 30
x2_max = 30

# 0 has rubbish
geometry_tensor = node_features_testset[1, :, -5:]
next_input = node_features_testset[1, :, :]

print("Edge attr shape: ", edge_attr_tensor.shape)

for i in range(0, panels):
    for j in range(0, x1_max):
        for k in range(0, x2_max):
            H[t, i, j, k] = next_input[tuple_to_index(
                i, j, k), 0] * STD_1[1, 0] + MEAN_1[1, 0]
            U[t, i, j, k] = next_input[tuple_to_index(
                i, j, k), 1] * STD_1[1, 1] + MEAN_1[1, 1]
            V[t, i, j, k] = next_input[tuple_to_index(
                i, j, k), 2] * STD_1[1, 2] + MEAN_1[1, 2]
            # U[t, i, j, k] = next_input[tuple_to_index(i, j, k), 3]
            # V[t, i, j, k] = next_input[tuple_to_index(i, j, k), 4]
t = t + 1

print(next_input.shape)
with torch.no_grad():
    for i in range(0, 100):
        print("t = ", t, "/200")
        outputs = loaded_model(next_input, edges_testset,
                               edge_attr_tensor, next_input)
        next_input = torch.cat((outputs, geometry_tensor), dim=1)
        #next_input = outputs
        print(next_input.shape)
        for i in range(0, panels):
            for j in range(0, x1_max):
                for k in range(0, x2_max):
                    H[t, i, j, k] = next_input[tuple_to_index(
                        i, j, k), 0] * STD_1[1, 0] + MEAN_1[1, 0]
                    U[t, i, j, k] = next_input[tuple_to_index(
                        i, j, k), 1] * STD_1[1, 1] + MEAN_1[1, 1]
                    V[t, i, j, k] = next_input[tuple_to_index(
                        i, j, k), 2] * STD_1[1, 2] + MEAN_1[1, 2]
        #MEAN_1[1, 0] = torch.mean(next_input[:, 0])
        #STD_1[1, 0] = torch.std(next_input[:, 0])
        #MEAN_1[1, 1] = torch.mean(next_input[:, 1])
        #STD_1[1, 1] = torch.std(next_input[:, 1])
        #MEAN_1[1, 2] = torch.mean(next_input[:, 2])
        #STD_1[1, 2] = torch.std(next_input[:, 2])

        t = t + 1

file.variables['h'] = H
# file.variables['hu1'] = HU1
# file.variables['hu2'] = HU2
file.variables['U'] = U
file.variables['V'] = V

file.close()

#print("=======NN test output saved========")
