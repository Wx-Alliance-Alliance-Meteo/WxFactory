import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
import einops
import cartopy.crs as ccrs

from torch_geometric.utils import to_networkx
from torch_geometric.data import Batch

#Based on GraphCast
from MyGenCast_Source.graph_from_latlon.graph_builder import GraphBuilder
#Based on GenCast Denoiser
from MyGenCast_Source.encoder import Encoder
#Based on GenCast
from MyGenCast_Source.processor import Processor
#Based on GenCast Denoiser
from MyGenCast_Source.decoder import Decoder


# Create grid latitudes and longitudes
grid_lats = np.linspace(-90, 90, 180)
grid_lons = np.linspace(0, 360, 360)

grid_features = torch.load("../data_points_gencast256.2.180.360.3.pt")
print("Input Grid Features shape: ", grid_features.shape)
data_point = grid_features[0, 0, :, :, :]

# Build the graph
graphs = GraphBuilder(grid_lat=grid_lats, grid_lon=grid_lons, splits=4, num_hops=1)

print("Graph Grid to Mesh: ", graphs.g2m_graph)
print("Graph mesh to grid: ", graphs.m2g_graph)
print("Graph on Mesh: ", graphs.mesh_graph)

########## Convert the mesh graph to NetworkX graph and visualize

g = to_networkx(graphs.g2m_graph)
#nx.draw(g, with_labels=True, node_size=50, node_color='skyblue', font_size=8, font_weight='bold')
#plt.show()

########## Define DIMS

output_features_dim = 3
input_features_dim = 3
hidden_dims = [256,256]

########## Initialize Encoder-Processor-Decoder

encodes = Encoder(
            grid_dim=input_features_dim + graphs.grid_nodes_dim,
            mesh_dim=graphs.mesh_nodes_dim,
            edge_dim=graphs.g2m_edges_dim,
            hidden_dims=hidden_dims,
            activation_layer=torch.nn.SiLU,
            use_layer_norm=True,)

processes = Processor(
            input_dim=hidden_dims[0],
            edge_dim=4,
            num_blocks=8,
            hidden_dim_processor_edge=hidden_dims[0],
            hidden_layers_processor_node=2,
            hidden_dim_processor_node=hidden_dims[0],
            hidden_layers_processor_edge=2,
            mlp_norm_type="LayerNorm")

decodes = Decoder(
            edges_dim=graphs.m2g_edges_dim,
            output_dim=output_features_dim,
            hidden_dims=hidden_dims,
            activation_layer=torch.nn.SiLU,
            use_layer_norm=True,)

print(encodes)
print(sum(p.numel() for p in encodes.parameters() if p.requires_grad))
print(processes)
print(sum(p.numel() for p in processes.parameters() if p.requires_grad))
print(decodes)
print(sum(p.numel() for p in decodes.parameters() if p.requires_grad))

######### RUN ENCODER

def run_encoder(grid_features):
    # build big graph with batch_size disconnected copies of the graph, with features [(b n) f].
    batch_size = grid_features.shape[0]
    g2m_batched = Batch.from_data_list([graphs.g2m_graph] * batch_size)

    # load features.
    grid_features = einops.rearrange(grid_features, "b n f -> (b n) f")
    grid_features = torch.tensor(grid_features)
    input_grid_nodes = torch.cat([grid_features, g2m_batched["grid_nodes"].x], dim=-1).type(
        torch.float32
    )
    input_mesh_nodes = g2m_batched["mesh_nodes"].x
    input_edge_attr = g2m_batched["grid_nodes", "to", "mesh_nodes"].edge_attr
    edge_index = g2m_batched["grid_nodes", "to", "mesh_nodes"].edge_index

    # run the encoder.
    print(input_grid_nodes.shape)
    latent_grid_nodes, latent_mesh_nodes, latent_edge_emb = encodes(
        input_grid_nodes=input_grid_nodes,
        input_mesh_nodes=input_mesh_nodes,
        input_edge_attr=input_edge_attr,
        edge_index=edge_index,
    )

    # restore nodes dimension: [b, n, f]
    latent_grid_nodes = einops.rearrange(latent_grid_nodes, "(b n) f -> b n f", b=batch_size)
    latent_mesh_nodes = einops.rearrange(latent_mesh_nodes, "(b n) f -> b n f", b=batch_size)
    latent_edge_emb = einops.rearrange(latent_edge_emb, "(b n) f -> b n f", b=batch_size)

    assert not torch.isnan(latent_grid_nodes).any()
    assert not torch.isnan(latent_mesh_nodes).any()
    return latent_grid_nodes, latent_mesh_nodes, latent_edge_emb

########## RUN PROCESSOR

def run_processor(latent_mesh_nodes):
    # build big graph with batch_size disconnected copies of the graph, with features [(b n) f].
    batch_size = latent_mesh_nodes.shape[0]
    num_nodes = latent_mesh_nodes.shape[1]
    mesh_batched = Batch.from_data_list([graphs.mesh_graph] * batch_size)

    # load features.
    latent_mesh_nodes = einops.rearrange(latent_mesh_nodes, "b n f -> (b n) f")
    input_edge_attr = mesh_batched.edge_attr #if self.use_edges_features else None
    edge_index = mesh_batched.edge_index

    print(latent_mesh_nodes.shape)
    print(input_edge_attr.shape)
    # run the processor.
    latent_mesh_nodes = processes.forward(
        x=latent_mesh_nodes,
        edge_attr=input_edge_attr,
        edge_index=edge_index,
    )

    # restore nodes dimension: [b, n, f]
    latent_mesh_nodes = einops.rearrange(latent_mesh_nodes, "(b n) f -> b n f", b=batch_size)

    assert not torch.isnan(latent_mesh_nodes).any()
    return latent_mesh_nodes

########## RUN DECODER

def run_decoder(latent_mesh_nodes, latent_grid_nodes):
    # build big graph with batch_size disconnected copies of the graph, with features [(b n) f].
    batch_size = latent_mesh_nodes.shape[0]
    m2g_batched = Batch.from_data_list([graphs.m2g_graph] * batch_size)

    # load features.
    input_mesh_nodes = einops.rearrange(latent_mesh_nodes, "b n f -> (b n) f")
    input_grid_nodes = einops.rearrange(latent_grid_nodes, "b n f -> (b n) f")
    input_edge_attr = m2g_batched["mesh_nodes", "to", "grid_nodes"].edge_attr
    edge_index = m2g_batched["mesh_nodes", "to", "grid_nodes"].edge_index

    # run the decoder.
    output_grid_nodes = decodes(
        input_mesh_nodes=input_mesh_nodes,
        input_grid_nodes=input_grid_nodes,
        input_edge_attr=input_edge_attr,
        edge_index=edge_index,
    )

    # restore nodes dimension: [b, n, f]
    output_grid_nodes = einops.rearrange(output_grid_nodes, "(b n) f -> b n f", b=batch_size)

    assert not torch.isnan(output_grid_nodes).any()
    return output_grid_nodes

########### Get the Output from Encoder-Processor-Decoder

latent_grid_nodes, latent_mesh_nodes, latent_edge_emb = run_encoder(data_point.reshape(1, 180*360, 3))
print("Encoder Output grid nodes shape: ", latent_grid_nodes.shape)
print("Encoder Output mesh nodes shape: ", latent_mesh_nodes.shape)
print("Encoder Output edge emb shape: ", latent_edge_emb.shape)

latent_mesh_nodes = run_processor(latent_mesh_nodes)
print("Processor Output mesh nodes shape: ", latent_mesh_nodes.shape)

output_grid_nodes = run_decoder(latent_mesh_nodes, latent_grid_nodes)
print("Decoder Output grid nodes shape: ", output_grid_nodes.shape)

########## Visualize the Input and Output

# Create a figure with two subplots and separate projections
fig, axs = plt.subplots(1, 2, subplot_kw={'projection': ccrs.Orthographic(central_longitude=0.0, central_latitude=25.0)}, figsize=(12, 6))
#fig, axs = plt.subplots(1, 3, figsize=(12, 6))

# Plot the original data
ax1 = axs[0]
ax1 = fig.add_subplot(axs[0])
im1 = ax1.pcolormesh(grid_lons, grid_lats, data_point[:, :, 2], cmap='twilight_shifted', transform=ccrs.PlateCarree(), antialiased=False, vmax=3, vmin=-3)
ax1.set_title('Original Data')
ax1.coastlines()
plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.1)
plt.show()
plt.close()

# Plot the interpolated data
fig, axs = plt.subplots(1, 2, subplot_kw={'projection': ccrs.Orthographic(central_longitude=0.0, central_latitude=25.0)}, figsize=(12, 6))
ax2 = axs[0]
proj2 = ccrs.Orthographic(central_longitude=180.0, central_latitude=0.0)
ax2 = fig.add_subplot(axs[0])
im2 = ax2.pcolormesh(grid_lons, grid_lats, output_grid_nodes.reshape(1, 180, 360, 3).detach().numpy().squeeze(0)[:, :, 2], cmap='twilight_shifted', transform=ccrs.PlateCarree(), antialiased=False, vmax=3, vmin=-3)
ax2.set_title('Output Data')
ax2.coastlines()
plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.1)
plt.show()
plt.close()
plt.figure(figsize=(10, 10))
