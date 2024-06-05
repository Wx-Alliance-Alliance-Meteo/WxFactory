import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.lines as mLines
import numpy as np

import networkx as nx
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv

# Open the NetCDF file
file = nc.Dataset(
    '../../../Explore_ai/WxFactory/results/out_case6_base.nc', 'r')

# Print the file format and dimensions
print("File format:", file.file_format)
print("Dimensions:", file.dimensions)

# Print the nf_height names
print("Variables:", file.variables.keys())

# Access nf (node features) and attributes
nf_height = file.variables['h']
nf_hu1 = file.variables['hu1']
nf_hu2 = file.variables['hu2']

nf_U = file.variables['U']
nf_V = file.variables['V']

nf_lats = file.variables['lats']
nf_lons = file.variables['lons']

nf_h_contra_11 = np.sin(np.radians(nf_lats)) #file.variables['h_contra_11']
nf_h_contra_12 = np.cos(np.radians(nf_lats)) #file.variables['h_contra_12']
nf_h_contra_21 = np.sin(np.radians(nf_lons)) #file.variables['h_contra_21']
nf_h_contra_22 = np.cos(np.radians(nf_lons)) #file.variables['h_contra_22']

nf_sqrt_g = file.variables['sqrt_g']

#nf_christie_1_01 = file.variables['christoffel_1_01']
#nf_christie_1_02 = file.variables['christoffel_1_02']
#nf_christie_1_11 = file.variables['christoffel_1_11']
#nf_christie_1_12 = file.variables['christoffel_1_12']
#nf_christie_2_01 = file.variables['christoffel_2_01']
#nf_christie_2_02 = file.variables['christoffel_2_02']
#nf_christie_2_12 = file.variables['christoffel_2_12']
#nf_christie_2_22 = file.variables['christoffel_2_22']

print(np.max(nf_height), " ", np.min(nf_height), " ", np.mean(nf_height))
print(np.max(nf_hu1), " ", np.min(nf_hu1), " ", np.mean(nf_hu1))
print(np.max(nf_hu2), " ", np.min(nf_hu2), " ", np.mean(nf_hu2))
print(np.max(nf_U), " ", np.min(nf_U), " ", np.mean(nf_U))
print(np.max(nf_V), " ", np.min(nf_V), " ", np.mean(nf_V))
print(np.max(nf_lats), " ", np.min(nf_lats), " ", np.mean(nf_lats))
print(np.max(nf_lons), " ", np.min(nf_lons), " ", np.mean(nf_lons))
print(np.max(nf_h_contra_11), " ", np.min(
    nf_h_contra_11), " ", np.mean(nf_h_contra_11))
print(np.max(nf_h_contra_12), " ", np.min(
    nf_h_contra_12), " ", np.mean(nf_h_contra_12))
print(np.max(nf_h_contra_21), " ", np.min(
    nf_h_contra_21), " ", np.mean(nf_h_contra_21))
print(np.max(nf_h_contra_22), " ", np.min(
    nf_h_contra_22), " ", np.mean(nf_h_contra_22))
print(np.max(nf_sqrt_g), " ", np.min(nf_sqrt_g), " ", np.mean(nf_sqrt_g))
#print(np.max(nf_christie_1_01), " ", np.min(
#    nf_christie_1_01), " ", np.mean(nf_christie_1_01))
#print(np.max(nf_christie_1_02), " ", np.min(
#    nf_christie_1_02), " ", np.mean(nf_christie_1_02))
#print(np.max(nf_christie_1_11), " ", np.min(
#    nf_christie_1_11), " ", np.mean(nf_christie_1_11))
#print(np.max(nf_christie_1_12), " ", np.min(
#    nf_christie_1_12), " ", np.mean(nf_christie_1_12))
#print(np.max(nf_christie_2_01), " ", np.min(
#    nf_christie_2_01), " ", np.mean(nf_christie_2_01))
#print(np.max(nf_christie_2_02), " ", np.min(
#    nf_christie_2_02), " ", np.mean(nf_christie_2_02))
#print(np.max(nf_christie_2_12), " ", np.min(
#    nf_christie_2_12), " ", np.mean(nf_christie_2_12))
#print(np.max(nf_christie_2_22), " ", np.min(
#    nf_christie_2_22), " ", np.mean(nf_christie_2_22))

adjacency_list = {}
adjacency_list_2 = {}

my_north_panel = -1
my_south_panel = -1
my_west_panel = -1
my_east_panel = -1

# Iterate over each grid point
panels = 6
x1_max = file.dimensions['Ydim'].size
x2_max = file.dimensions['Xdim'].size

edges_dataset = torch.zeros((2, 4*x1_max*x2_max*panels))
node_features_dataset = torch.zeros((1, x1_max*x2_max*panels, 12))


def draw_graph(adjacency):
    """Draw the given adjacency array as an unwrapped cube."""

    fig, ax = plt.subplots()
    plt.plot([0], [0])

    def add_line(p1, p2, *args, **kwargs):
        ax.add_line(mLines.Line2D([p1[0], p2[0]], [
                    p1[1], p2[1]], *args, **kwargs))

    coords = {}
    coords[0] = [0, 0]

    offsets = [
        [0, 0],
        [x1_max, 0],
        [2*x1_max, 0],
        [-x1_max, 0],
        [0, x2_max],
        [0, -x2_max]
    ]

    for panel in range(6):
        for x1 in range(x1_max):
            for x2 in range(x2_max):
                id = tuple_to_index(panel, x1, x2)
                coords[id] = [offsets[panel][0] + x1, offsets[panel][1] + x2]

    for point, neighbors in adjacency.items():
        for neighbor in neighbors:
            add_line(coords[point], coords[neighbor],
                     color='red' if point < neighbor else 'green')

    plt.tight_layout()
    plt.savefig('graph.png')


def tuple_to_index(i, j, k):
    index = x1_max * x2_max * i + x1_max * k + j
    return index


for t in range(0, nf_height.shape[0]):
#for t in range(0, 200):
    #print(f"Conversion: NetCDF file to GNN input data t = {t}/500")
    print(f"Conversion: NetCDF file to GNN input data t = {t}/{nf_height.shape[0]}")
    for i in range(panels):

        if i == 0:
            my_north_panel = 4
            my_south_panel = 5
            my_west_panel = 3
            my_east_panel = 1
        elif i == 1:
            my_north_panel = 4
            my_south_panel = 5
            my_west_panel = 0
            my_east_panel = 2
        elif i == 2:
            my_north_panel = 4
            my_south_panel = 5
            my_west_panel = 1
            my_east_panel = 3
        elif i == 3:
            my_north_panel = 4
            my_south_panel = 5
            my_west_panel = 2
            my_east_panel = 0
        elif i == 4:
            my_north_panel = 2
            my_south_panel = 0
            my_west_panel = 3
            my_east_panel = 1
        elif i == 5:
            my_north_panel = 0
            my_south_panel = 2
            my_west_panel = 3
            my_east_panel = 1
        # panel 0
        if i == 0:
            for j in range(x1_max):
                for k in range(x2_max):
                    neighbors = []
                    # Add west neighbor
                    if j > 0:
                        neighbors.append(tuple_to_index(i, j-1, k))
                    if j == 0:
                        neighbors.append(tuple_to_index(
                            my_west_panel, x1_max - 1, k))
                    # Add east neighbor
                    if j < x1_max - 1:
                        neighbors.append(tuple_to_index(i, j+1, k))
                    if j == x1_max - 1:
                        neighbors.append(tuple_to_index(my_east_panel, 0, k))
                    # Add south neighbor
                    if k > 0:
                        neighbors.append(tuple_to_index(i, j, k-1))
                    if k == 0:
                        neighbors.append(tuple_to_index(
                            my_south_panel, j, x2_max - 1))
                    # Add north neighbor
                    if k < x2_max - 1:
                        neighbors.append(tuple_to_index(i, j, k+1))
                    if k == x2_max - 1:
                        neighbors.append(tuple_to_index(my_north_panel, j, 0))
                    # Store neighbors for current grid point
                    adjacency_list[tuple_to_index(i, j, k)] = neighbors
                    arr = (np.array([nf_height[t, i, j, k], nf_hu1[t, i, j, k], nf_hu2[t, i, j, k], nf_U[t, i, j, k], nf_V[t, i, j, k], nf_lats[i, j, k], nf_lons[i, j, k], nf_h_contra_11[i, j, k], nf_h_contra_12[i, j, k], nf_h_contra_21[i, j, k], nf_h_contra_22[i,
                           j, k], nf_sqrt_g[i, j, k]]))#, nf_christie_1_01[i, j, k], nf_christie_1_02[i, j, k], nf_christie_1_11[i, j, k], nf_christie_1_12[i, j, k], nf_christie_2_01[i, j, k], nf_christie_2_02[i, j, k], nf_christie_2_12[i, j, k], nf_christie_2_22[i, j, k]]))
                    adjacency_list_2[tuple_to_index(i, j, k)] = arr
        # panel 1
        elif i == 1:
            for j in range(x1_max):
                for k in range(x2_max):
                    neighbors = []
                    # Add west neighbor
                    if j > 0:
                        neighbors.append(tuple_to_index(i, j-1, k))
                    if j == 0:
                        neighbors.append(tuple_to_index(
                            my_west_panel, x1_max - 1, k))
                    # Add east neighbor
                    if j < x1_max - 1:
                        neighbors.append(tuple_to_index(i, j+1, k))
                    if j == x1_max - 1:
                        neighbors.append(tuple_to_index(my_east_panel, 0, k))
                    # Add south neighbor
                    if k > 0:
                        neighbors.append(tuple_to_index(i, j, k-1))
                    if k == 0:
                        neighbors.append(tuple_to_index(
                            my_south_panel, x1_max - 1, x2_max - 1 - j))
                    # Add north neighbor
                    if k < x2_max - 1:
                        neighbors.append(tuple_to_index(i, j, k+1))
                    if k == x2_max - 1:
                        neighbors.append(tuple_to_index(
                            my_north_panel, x1_max - 1, j))
                    # Store neighbors for current grid point
                    adjacency_list[tuple_to_index(i, j, k)] = neighbors
                    arr = (np.array([nf_height[t, i, j, k], nf_hu1[t, i, j, k], nf_hu2[t, i, j, k], nf_U[t, i, j, k], nf_V[t, i, j, k], nf_lats[i, j, k], nf_lons[i, j, k], nf_h_contra_11[i, j, k], nf_h_contra_12[i, j, k], nf_h_contra_21[i, j, k], nf_h_contra_22[i,
                           j, k], nf_sqrt_g[i, j, k]]))#, nf_christie_1_01[i, j, k], nf_christie_1_02[i, j, k], nf_christie_1_11[i, j, k], nf_christie_1_12[i, j, k], nf_christie_2_01[i, j, k], nf_christie_2_02[i, j, k], nf_christie_2_12[i, j, k], nf_christie_2_22[i, j, k]]))
                    adjacency_list_2[tuple_to_index(i, j, k)] = arr
        # panel 2
        elif i == 2:
            for j in range(x1_max):
                for k in range(x2_max):
                    neighbors = []
                    # Add west neighbor
                    if j > 0:
                        neighbors.append(tuple_to_index(i, j-1, k))
                    if j == 0:
                        neighbors.append(tuple_to_index(
                            my_west_panel, x1_max - 1, k))
                    # Add east neighbor
                    if j < x1_max - 1:
                        neighbors.append(tuple_to_index(i, j+1, k))
                    if j == x1_max - 1:
                        neighbors.append(tuple_to_index(my_east_panel, 0, k))
                    # Add south neighbor
                    if k > 0:
                        neighbors.append(tuple_to_index(i, j, k-1))
                    if k == 0:
                        neighbors.append(tuple_to_index(
                            my_south_panel, x1_max - j - 1, 0))
                    # Add north neighbor
                    if k < x2_max - 1:
                        neighbors.append(tuple_to_index(i, j, k+1))
                    if k == x2_max - 1:
                        neighbors.append(tuple_to_index(
                            my_north_panel, x1_max - 1 - j, x2_max - 1))
                    # Store neighbors for current grid point
                    adjacency_list[tuple_to_index(i, j, k)] = neighbors
                    arr = (np.array([nf_height[t, i, j, k], nf_hu1[t, i, j, k], nf_hu2[t, i, j, k], nf_U[t, i, j, k], nf_V[t, i, j, k], nf_lats[i, j, k], nf_lons[i, j, k], nf_h_contra_11[i, j, k], nf_h_contra_12[i, j, k], nf_h_contra_21[i, j, k], nf_h_contra_22[i,
                           j, k], nf_sqrt_g[i, j, k]]))#, nf_christie_1_01[i, j, k], nf_christie_1_02[i, j, k], nf_christie_1_11[i, j, k], nf_christie_1_12[i, j, k], nf_christie_2_01[i, j, k], nf_christie_2_02[i, j, k], nf_christie_2_12[i, j, k], nf_christie_2_22[i, j, k]]))
                    adjacency_list_2[tuple_to_index(i, j, k)] = arr
        # panel 3
        elif i == 3:
            for j in range(x1_max):
                for k in range(x2_max):
                    neighbors = []
                    # Add west neighbor
                    if j > 0:
                        neighbors.append(tuple_to_index(i, j-1, k))
                    if j == 0:
                        neighbors.append(tuple_to_index(
                            my_west_panel, x1_max - 1, k))
                    # Add east neighbor
                    if j < x1_max - 1:
                        neighbors.append(tuple_to_index(i, j+1, k))
                    if j == x1_max - 1:
                        neighbors.append(tuple_to_index(my_east_panel, 0, k))
                    # Add south neighbor
                    if k > 0:
                        neighbors.append(tuple_to_index(i, j, k-1))
                    if k == 0:
                        neighbors.append(tuple_to_index(my_south_panel, 0, j))
                    # Add north neighbor
                    if k < x2_max - 1:
                        neighbors.append(tuple_to_index(i, j, k+1))
                    if k == x2_max - 1:
                        neighbors.append(tuple_to_index(
                            my_north_panel, 0, x2_max - 1 - j))
                    # Store neighbors for current grid point
                    adjacency_list[tuple_to_index(i, j, k)] = neighbors
                    arr = (np.array([nf_height[t, i, j, k], nf_hu1[t, i, j, k], nf_hu2[t, i, j, k], nf_U[t, i, j, k], nf_V[t, i, j, k], nf_lats[i, j, k], nf_lons[i, j, k], nf_h_contra_11[i, j, k], nf_h_contra_12[i, j, k], nf_h_contra_21[i, j, k], nf_h_contra_22[i,
                           j, k], nf_sqrt_g[i, j, k]]))#, nf_christie_1_01[i, j, k], nf_christie_1_02[i, j, k], nf_christie_1_11[i, j, k], nf_christie_1_12[i, j, k], nf_christie_2_01[i, j, k], nf_christie_2_02[i, j, k], nf_christie_2_12[i, j, k], nf_christie_2_22[i, j, k]]))
                    adjacency_list_2[tuple_to_index(i, j, k)] = arr
        # panel 4
        elif i == 4:
            for j in range(x1_max):
                for k in range(x2_max):
                    neighbors = []
                    # Add west neighbor
                    if j > 0:
                        neighbors.append(tuple_to_index(i, j-1, k))
                    if j == 0:
                        neighbors.append(tuple_to_index(
                            my_west_panel, x1_max - 1 - k, x2_max - 1))
                    # Add east neighbor
                    if j < x1_max - 1:
                        neighbors.append(tuple_to_index(i, j+1, k))
                    if j == x1_max - 1:
                        neighbors.append(tuple_to_index(
                            my_east_panel, k, x2_max - 1))
                    # Add south neighbor
                    if k > 0:
                        neighbors.append(tuple_to_index(i, j, k-1))
                    if k == 0:
                        neighbors.append(tuple_to_index(
                            my_south_panel, j, x2_max - 1))
                    # Add north neighbor
                    if k < x2_max - 1:
                        neighbors.append(tuple_to_index(i, j, k+1))
                    if k == x2_max - 1:
                        neighbors.append(tuple_to_index(
                            my_north_panel, x1_max - 1 - j, x2_max - 1))
                    # Store neighbors for current grid point
                    adjacency_list[tuple_to_index(i, j, k)] = neighbors
                    arr = (np.array([nf_height[t, i, j, k], nf_hu1[t, i, j, k], nf_hu2[t, i, j, k], nf_U[t, i, j, k], nf_V[t, i, j, k], nf_lats[i, j, k], nf_lons[i, j, k], nf_h_contra_11[i, j, k], nf_h_contra_12[i, j, k], nf_h_contra_21[i, j, k], nf_h_contra_22[i,
                           j, k], nf_sqrt_g[i, j, k]]))#, nf_christie_1_01[i, j, k], nf_christie_1_02[i, j, k], nf_christie_1_11[i, j, k], nf_christie_1_12[i, j, k], nf_christie_2_01[i, j, k], nf_christie_2_02[i, j, k], nf_christie_2_12[i, j, k], nf_christie_2_22[i, j, k]]))
                    adjacency_list_2[tuple_to_index(i, j, k)] = arr
        # panel 5
        elif i == 5:
            for j in range(x1_max):
                for k in range(x2_max):
                    neighbors = []
                    # Add west neighbor
                    if j > 0:
                        neighbors.append(tuple_to_index(i, j-1, k))
                    if j == 0:
                        neighbors.append(tuple_to_index(my_west_panel, k, 0))
                    # Add east neighbor
                    if j < x1_max - 1:
                        neighbors.append(tuple_to_index(i, j+1, k))
                    if j == x1_max - 1:
                        neighbors.append(tuple_to_index(
                            my_east_panel, x1_max - 1 - k, 0))
                    # Add south neighbor
                    if k > 0:
                        neighbors.append(tuple_to_index(i, j, k-1))
                    if k == 0:
                        neighbors.append(tuple_to_index(
                            my_south_panel, x1_max - 1 - j, 0))
                    # Add north neighbor
                    if k < x2_max - 1:
                        neighbors.append(tuple_to_index(i, j, k+1))
                    if k == x2_max - 1:
                        neighbors.append(tuple_to_index(my_north_panel, j, 0))
                    # Store neighbors for current grid point
                    adjacency_list[tuple_to_index(i, j, k)] = neighbors
                    arr = (np.array([nf_height[t, i, j, k], nf_hu1[t, i, j, k], nf_hu2[t, i, j, k], nf_U[t, i, j, k], nf_V[t, i, j, k], nf_lats[i, j, k], nf_lons[i, j, k], nf_h_contra_11[i, j, k], nf_h_contra_12[i, j, k], nf_h_contra_21[i, j, k], nf_h_contra_22[i,
                           j, k], nf_sqrt_g[i, j, k]]))#, nf_christie_1_01[i, j, k], nf_christie_1_02[i, j, k], nf_christie_1_11[i, j, k], nf_christie_1_12[i, j, k], nf_christie_2_01[i, j, k], nf_christie_2_02[i, j, k], nf_christie_2_12[i, j, k], nf_christie_2_22[i, j, k]]))
                    adjacency_list_2[tuple_to_index(i, j, k)] = arr

    #draw_graph(adjacency_list)

    # Convert adjacency list to edge indices

    edges = []
    node_features = []

    # print('In Row Major:')
    # print(' ')
    for node, neighbors in sorted(adjacency_list_2.items()):
        # print(node, ":", neighbors)
        node_features.append(neighbors)

    for node, neighbors in sorted(adjacency_list.items()):
        # print(node, ":", neighbors)
        for neighbor in neighbors:
            edges.append([node, neighbor])

    # Convert edge list to PyG-compatible format
    edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if t == 1:
        edges_dataset = edges
    node_features = np.array(node_features)
    node_features = torch.tensor(node_features, dtype=torch.float).contiguous()
    node_features_dataset = torch.cat(
        (node_features_dataset, node_features.unsqueeze(0)), dim=0)
    # print(node_features)

# Create the Data object
node_features_dataset = torch.tensor(node_features_dataset)
print("Node feature dataset shape:", np.shape(node_features_dataset))
print("Edges dataset shape: ", np.shape(edges_dataset))

# Save the data to files
#torch.save(node_features_dataset,
#           f'node_features_dataset_case6_base_DIM500.6.30.30.pt')
torch.save(node_features_dataset, f'node_features_dataset_case6_base_DIM{nf_height.shape[0]}.6.30.30.pt')
#torch.save(edges_dataset, f'edges_dataset_case6_base_DIM500.6.30.30.pt')
torch.save(edges_dataset, f'edges_dataset_case6_base_DIM{nf_height.shape[0]}.6.30.30.pt')

file.close()

print("====GNN Input Dataset saved====")
