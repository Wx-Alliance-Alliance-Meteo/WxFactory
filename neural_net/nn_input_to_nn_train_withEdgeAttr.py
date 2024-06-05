import matplotlib.pyplot as plt

import numpy as np

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
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

# node_features_dataset = torch.load(
#    "node_features_dataset_case6random_DIM600.6.30.30.pt")
# edges_dataset = torch.load("edges_dataset_case6random_DIM600.6.30.30.pt")
node_features_dataset = torch.load(
    "node_features_dataset_case6_base_DIM502.6.30.30.pt")
edges_dataset = torch.load("edges_dataset_case6_base_DIM502.6.30.30.pt")

radius_earth = 6371220.0

node_features_dataset = node_features_dataset.contiguous()

# Train on H, U and V rather than H, hu1 and hu2

# INPUT : [0] H, [1] Hu1, [2] Hu2, [3] U, [4] V, [5] lats, [6] lons, [7] sinlat, [8] coslat, [9] sinlon, [10] coslon, [11] det(g)

# OUTPUT : [0] H, [1] U, [2] V, [3] det(g), [4] coslat, [5] sinlon, [6] coslon, [7] sinlat

print(node_features_dataset.shape)
# Removes Hu1, and Hu2
node_features_dataset = torch.cat(
    (node_features_dataset[:, :, :1], node_features_dataset[:, :, 3:]), dim=2)
# Removes lats, and lons
node_features_dataset = torch.cat(
    (node_features_dataset[:, :, :2], node_features_dataset[:, :, 4:]), dim=2)
# Swaps det(g) with sinlat
temp = node_features_dataset[:, :, 3].clone()
node_features_dataset[:, :, 3] = node_features_dataset[:, :, 7]
node_features_dataset[:, :, 7] = temp
print(node_features_dataset.shape)

print("======DEBUG INFO=====")
print(torch.mean(node_features_dataset[10, :, 0]))
print(torch.std(node_features_dataset[10, :, 0]))
print(torch.mean(node_features_dataset[10, :, 1]))
print(torch.std(node_features_dataset[10, :, 1]))
print(torch.mean(node_features_dataset[10, :, 2]))
print(torch.std(node_features_dataset[10, :, 2]))
print(torch.mean(node_features_dataset[10, :, 3]))
print(torch.std(node_features_dataset[10, :, 3]))

MEAN_1 = torch.mean(node_features_dataset, dim=1).contiguous()
STD_1 = torch.std(node_features_dataset, dim=1).contiguous()

shape = node_features_dataset.shape
for x in range(0, shape[0]):
    print("++ Normalizing Input data for H, U, V, det(g) for t = ", x)
    for y in range(0, shape[1]):
        for z in range(0, 4):
            node_features_dataset[x, y, z] = (
                node_features_dataset[x, y, z] - MEAN_1[x, z]) / (STD_1[x, z])

print("======DEBUG INFO=====")
print(torch.mean(node_features_dataset[10, :, 0]))
print(torch.std(node_features_dataset[10, :, 0]))
print(torch.mean(node_features_dataset[10, :, 1]))
print(torch.std(node_features_dataset[10, :, 1]))
print(torch.mean(node_features_dataset[10, :, 2]))
print(torch.std(node_features_dataset[10, :, 2]))
print(torch.mean(node_features_dataset[10, :, 3]))
print(torch.std(node_features_dataset[10, :, 3]))

print("Normalized Node feature dataset shape:", np.shape(node_features_dataset))

# Save Normalized Node features to files
torch.save(node_features_dataset,
           f'normalized_node_features_dataset_case6_base_DIM200.6.30.30.pt')

num_samples = node_features_dataset.shape[0]
num_edges = edges_dataset.shape[1]
node_feature_dim = node_features_dataset.shape[2]
edge_attr_dim = 9
edge_attr_tensor = torch.zeros((num_samples, num_edges, edge_attr_dim))

# Iterate over each first 2 samples in the node_features_dataset
for i in range(1, 4):
    # Get the node features for the current sample
    node_features = node_features_dataset[i]

    # Get the node indices for all edges
    node1_indices = edges_dataset[0]
    node2_indices = edges_dataset[1]

    # Gather the features for node1 and node2 for all edges
    node1_features = node_features[node1_indices]
    node2_features = node_features[node2_indices]
    print("++ Creating Edge Attribute tensor for t = ", i)

    # Concatenate the features along the last dimension and add the great circle distance
    great_circle_distance = torch.arccos(node1_features[:, 7]*node2_features[:, 7] + node1_features[:, 4]*node2_features[:, 4]*(
        node1_features[:, 5]*node2_features[:, 5] + node1_features[:, 6]*node2_features[:, 6])).view(num_edges, 1)
    print("++ Great circle dist mean: ", torch.mean(great_circle_distance))
    print("++ Great circle dist std: ", torch.std(great_circle_distance))
    edge_attr_tensor[i] = torch.cat(
        (node1_features[:, -4:], great_circle_distance, node2_features[:, -4:]), dim=1)

print(edge_attr_tensor.shape)

Data_list = []

for i in range(1, shape[0] - 1):
    Data_list.append(data.Data(
        x=node_features_dataset[i, :, :], edge_index=edges_dataset, edge_attr=edge_attr_tensor[2, :, :], y=node_features_dataset[i+1, :, :3]))


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

# class Encoder(nn.Module):
#    def __init__(self, node_input_dim, edge_input_dim, hidden_dim):
#        super(Encoder, self).__init__()
#        self.gnn = GNNLayer(node_input_dim, edge_input_dim, hidden_dim)
#
#    def forward(self, x, edge_index, edge_attr):
#        return self.gnn(x, edge_index, edge_attr)


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

model = GNNModel(node_input_dim, edge_input_dim, hidden_dim, node_output_dim)

# Print the model architecture
print(model)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

labels = None
outputs = None
loss = None

# Training loop
for epoch in range(30):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    i = 0
    for batch in Data_list:
        outputs = model(batch.x, batch.edge_index, batch.edge_attr, batch.x)
        labels = batch.y
        # print(outputs.shape)
        # print(labels.shape)
        # for i in range(0, 10):
        #    print(outputs[:10,:])
        #    print(labels[:10,:])

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # labels_last = labels
        # outputs_last = outputs
    print(f"Epoch {epoch+1}/10, Loss: {loss.item()}")
    threshold = 0.5

    # Calculate accuracy based on the threshold
    accuracy = torch.mean(
        (torch.abs(labels[:, 0] - outputs[:, 0])).float())

    # print("Mean Squared Error:", loss.item())
    print("Accuracy h:", accuracy.item(),
          "Mean Value: ", torch.mean(labels[:, 0]))
    accuracy = torch.mean(
        (torch.abs(labels[:, 1] - outputs[:, 1])).float())

    # print("Mean Squared Error:", loss.item())
    print("Accuracy hu1:", accuracy.item(),
          "Mean Value:", torch.mean(labels[:, 1]))
    accuracy = torch.mean(
        (torch.abs(labels[:, 2] - outputs[:, 2])).float())

    # print("Mean Squared Error:", loss.item())
    print("Accuracy hu2:", accuracy.item(),
          "Mean Value:", torch.mean(labels[:, 2]))
    # accuracy = torch.mean(
    #    (torch.abs(labels[:, 3] - outputs[:, 3])).float())

    # print("Mean Squared Error:", loss.item())
    # print("Accuracy U:", accuracy.item(),
    #      "Mean Value:", torch.mean(labels[:, 3]))
    # accuracy = torch.mean(
    #    (torch.abs(labels[:, 4] - outputs[:, 4])).float())

    # print("Mean Squared Error:", loss.item())
    # print("Accuracy V:", accuracy.item(),
    #      "Mean Value:", torch.mean(labels[:, 4]))

print("Training finished!")

# Save the model
torch.save(model.state_dict(), 'NNmodel.pth')
