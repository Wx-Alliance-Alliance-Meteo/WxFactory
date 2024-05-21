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
# import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

#node_features_dataset = torch.load(
#    "node_features_dataset_case6random_DIM600.6.30.30.pt")
#edges_dataset = torch.load("edges_dataset_case6random_DIM600.6.30.30.pt")
node_features_dataset = torch.load(
    "node_features_dataset_case6_base_DIM200.6.30.30.pt")
edges_dataset = torch.load("edges_dataset_case6_base_DIM200.6.30.30.pt")

# node_features_dataset = node_features_dataset[:, :, :5]

radius_earth = 6371220.0
print(torch.mean(node_features_dataset[10, :, 0]))
print(torch.std(node_features_dataset[10, :, 0]))
print(torch.mean(node_features_dataset[10, :, 1]))
print(torch.std(node_features_dataset[10, :, 1]))
print(torch.mean(node_features_dataset[10, :, 2]))
print(torch.std(node_features_dataset[10, :, 2]))

node_features_dataset = node_features_dataset.contiguous()

MEAN_1 = torch.mean(node_features_dataset, dim=1).contiguous()
STD_1 = torch.std(node_features_dataset, dim=1).contiguous()

shape = node_features_dataset.shape
for x in range(0, shape[0]):
    print(x)
    for y in range(0, shape[1]):
        for z in range(0, shape[2]):
            node_features_dataset[x, y, z] = (node_features_dataset[x, y, z] - MEAN_1[x, z]) /(STD_1[x, z])
print(torch.mean(node_features_dataset[10, :, 0]))
print(torch.std(node_features_dataset[10, :, 0]))
print(torch.mean(node_features_dataset[10, :, 1]))
print(torch.std(node_features_dataset[10, :, 1]))
print(torch.mean(node_features_dataset[10, :, 2]))
print(torch.std(node_features_dataset[10, :, 2]))
#MEAN_2 = torch.mean(node_features_dataset, dim=1).contiguous()
#STD_2 = torch.std(node_features_dataset, dim=1).contiguous()
#
#shape = node_features_dataset.shape
#for x in range(0, shape[0]):
#    for y in range(0, shape[1]):
#        for z in range(0, shape[2]):
#            node_features_dataset[x, y, z] = (node_features_dataset[x, y, z] - MEAN_2[x, z]) /(STD_2[x, z])
#print(torch.mean(node_features_dataset[1, :, 0]))
#print(torch.std(node_features_dataset[1, :, 0]))
#print(torch.mean(node_features_dataset[1, :, 1]))
#print(torch.std(node_features_dataset[1, :, 1]))
#print(torch.mean(node_features_dataset[1, :, 2]))
#print(torch.std(node_features_dataset[1, :, 2]))
#MEAN_3 = torch.mean(node_features_dataset, dim=1).contiguous()
#STD_3 = torch.std(node_features_dataset, dim=1).contiguous()
#
#shape = node_features_dataset.shape
#for x in range(0, shape[0]):
#    for y in range(0, shape[1]):
#        for z in range(0, shape[2]):
#            node_features_dataset[x, y, z] = (node_features_dataset[x, y, z] - MEAN_3[x, z]) /(STD_3[x, z])
#
#        #node_features_dataset[x, y, 11] /= (radius_earth * radius_earth)
#        #node_features_dataset[x, y, 11] /= (radius_earth * radius_earth)
#        #node_features_dataset[x, y, 7] *= (radius_earth * radius_earth)
#        #node_features_dataset[x, y, 8] *= (radius_earth * radius_earth)
#        #node_features_dataset[x, y, 9] *= (radius_earth * radius_earth)
#        #node_features_dataset[x, y, 10] *= (radius_earth * radius_earth)
#
#print(torch.mean(node_features_dataset[1, :, 0]))
#print(torch.std(node_features_dataset[1, :, 0]))
#print(torch.mean(node_features_dataset[1, :, 1]))
#print(torch.std(node_features_dataset[1, :, 1]))
#print(torch.mean(node_features_dataset[1, :, 2]))
#print(torch.std(node_features_dataset[1, :, 2]))

Data_list = []

for i in range(1, shape[0] - 1):
    # for j in range(0, 10):
    #    print(node_features_dataset[i, j, :])
    Data_list.append(data.Data(
        x=node_features_dataset[i, :, :], edge_index=edges_dataset, y=node_features_dataset[i+1, :, :3]))

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

class GCNConvolution_layer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out = out + self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


# Instantiate the GCN model
model = GCN(20, 256, 3, 0.001)

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
        outputs = model(batch.x, batch.edge_index)
        labels = batch.y
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
    #accuracy = torch.mean(
    #    (torch.abs(labels[:, 3] - outputs[:, 3])).float())

    ## print("Mean Squared Error:", loss.item())
    #print("Accuracy U:", accuracy.item(),
    #      "Mean Value:", torch.mean(labels[:, 3]))
    #accuracy = torch.mean(
    #    (torch.abs(labels[:, 4] - outputs[:, 4])).float())

    ## print("Mean Squared Error:", loss.item())
    #print("Accuracy V:", accuracy.item(),
    #      "Mean Value:", torch.mean(labels[:, 4]))

print("Training finished!")

# Save the model
torch.save(model.state_dict(), 'NNmodel.pth')
