import numpy as np
import matplotlib.pyplot as plt
import torch
import einops
import cartopy.crs as ccrs
from torch_geometric.utils import to_networkx
from torch_geometric.data import Batch
from torch.nn import MSELoss
from torch.optim import Adam

# Load the necessary components
from GenCast_Sourcefiles.graph_from_latlon.graph_builder import GraphBuilder
from GenCast_Sourcefiles.encoder import Encoder
from GenCast_Sourcefiles.processor import Processor
from GenCast_Sourcefiles.decoder import Decoder

# Create grid latitudes and longitudes
grid_lats = np.linspace(-90, 90, 180)
grid_lons = np.linspace(0, 360, 360)

# Load the dataset
grid_features = torch.load("../data_points_gencast256.2.180.360.3.pt")
print("Input Grid Features shape: ", grid_features.shape)

# Build the graph
graphs = GraphBuilder(grid_lat=grid_lats, grid_lon=grid_lons, splits=5, num_hops=1)

# Define DIMS
output_features_dim = 3
input_features_dim = 3
hidden_dims = [256, 256]

# Initialize Encoder-Processor-Decoder
encodes = Encoder(
    grid_dim=input_features_dim + graphs.grid_nodes_dim,
    mesh_dim=graphs.mesh_nodes_dim,
    edge_dim=graphs.g2m_edges_dim,
    hidden_dims=hidden_dims,
    activation_layer=torch.nn.SiLU,
    use_layer_norm=True,
)

processes = Processor(
    input_dim=hidden_dims[0],
    edge_dim=4,
    num_blocks=8,
    hidden_dim_processor_edge=hidden_dims[0],
    hidden_layers_processor_node=2,
    hidden_dim_processor_node=hidden_dims[0],
    hidden_layers_processor_edge=2,
    mlp_norm_type="LayerNorm"
)

decodes = Decoder(
    edges_dim=graphs.m2g_edges_dim,
    output_dim=output_features_dim,
    hidden_dims=hidden_dims,
    activation_layer=torch.nn.SiLU,
    use_layer_norm=True,
)

# Initialize loss function and optimizer
criterion = MSELoss()
optimizer = Adam(list(encodes.parameters()) + list(processes.parameters()) + list(decodes.parameters()), lr=1e-3)

# Functions for encoding, processing, and decoding
def run_encoder(grid_features):
    batch_size = grid_features.shape[0]
    g2m_batched = Batch.from_data_list([graphs.g2m_graph] * batch_size)
    grid_features = einops.rearrange(grid_features, "b n f -> (b n) f").type(torch.float32)
    input_grid_nodes = torch.cat([grid_features, g2m_batched["grid_nodes"].x], dim=-1)
    input_mesh_nodes = g2m_batched["mesh_nodes"].x
    input_edge_attr = g2m_batched["grid_nodes", "to", "mesh_nodes"].edge_attr
    edge_index = g2m_batched["grid_nodes", "to", "mesh_nodes"].edge_index

    latent_grid_nodes, latent_mesh_nodes, latent_edge_emb = encodes(
        input_grid_nodes=input_grid_nodes,
        input_mesh_nodes=input_mesh_nodes,
        input_edge_attr=input_edge_attr,
        edge_index=edge_index,
    )

    latent_grid_nodes = einops.rearrange(latent_grid_nodes, "(b n) f -> b n f", b=batch_size)
    latent_mesh_nodes = einops.rearrange(latent_mesh_nodes, "(b n) f -> b n f", b=batch_size)
    latent_edge_emb = einops.rearrange(latent_edge_emb, "(b n) f -> b n f", b=batch_size)

    return latent_grid_nodes, latent_mesh_nodes, latent_edge_emb

def run_processor(latent_mesh_nodes):
    batch_size = latent_mesh_nodes.shape[0]
    mesh_batched = Batch.from_data_list([graphs.mesh_graph] * batch_size)
    latent_mesh_nodes = einops.rearrange(latent_mesh_nodes, "b n f -> (b n) f")
    input_edge_attr = mesh_batched.edge_attr
    edge_index = mesh_batched.edge_index

    latent_mesh_nodes = processes.forward(
        x=latent_mesh_nodes,
        edge_attr=input_edge_attr,
        edge_index=edge_index,
    )

    latent_mesh_nodes = einops.rearrange(latent_mesh_nodes, "(b n) f -> b n f", b=batch_size)

    return latent_mesh_nodes

def run_decoder(latent_mesh_nodes, latent_grid_nodes):
    batch_size = latent_mesh_nodes.shape[0]
    m2g_batched = Batch.from_data_list([graphs.m2g_graph] * batch_size)
    input_mesh_nodes = einops.rearrange(latent_mesh_nodes, "b n f -> (b n) f")
    input_grid_nodes = einops.rearrange(latent_grid_nodes, "b n f -> (b n) f")
    input_edge_attr = m2g_batched["mesh_nodes", "to", "grid_nodes"].edge_attr
    edge_index = m2g_batched["mesh_nodes", "to", "grid_nodes"].edge_index

    output_grid_nodes = decodes(
        input_mesh_nodes=input_mesh_nodes,
        input_grid_nodes=input_grid_nodes,
        input_edge_attr=input_edge_attr,
        edge_index=edge_index,
    )

    output_grid_nodes = einops.rearrange(output_grid_nodes, "(b n) f -> b n f", b=batch_size)

    return output_grid_nodes

def save_visualization(epoch, inp, tar, output_grid_nodes):
    inp = inp.reshape(1, 180, 360, 3).squeeze(0).detach().numpy()
    tar = tar.reshape(1, 180, 360, 3).squeeze(0).numpy()
    output_grid_nodes = output_grid_nodes.reshape(1, 180, 360, 3).squeeze(0).detach().numpy()

    fig, axs = plt.subplots(1, 3, subplot_kw={'projection': ccrs.Orthographic(central_longitude=0.0, central_latitude=25.0)}, figsize=(18, 6))

    # Plot the original data
    ax1 = axs[0]
    im1 = ax1.pcolormesh(grid_lons, grid_lats, inp[:, :, 2], cmap='twilight_shifted', transform=ccrs.PlateCarree(), antialiased=False, vmax=3, vmin=-3)
    ax1.set_title('Original Data')
    ax1.coastlines()
    plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.1)

    # Plot the target data
    ax2 = axs[1]
    im2 = ax2.pcolormesh(grid_lons, grid_lats, tar[:, :, 2], cmap='twilight_shifted', transform=ccrs.PlateCarree(), antialiased=False, vmax=3, vmin=-3)
    ax2.set_title('Target Data')
    ax2.coastlines()
    plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.1)

    # Plot the output data
    ax3 = axs[2]
    im3 = ax3.pcolormesh(grid_lons, grid_lats, output_grid_nodes[:, :, 2], cmap='twilight_shifted', transform=ccrs.PlateCarree(), antialiased=False, vmax=3, vmin=-3)
    ax3.set_title('Output Data')
    ax3.coastlines()
    plt.colorbar(im3, ax=ax3, orientation='horizontal', pad=0.1)

    plt.tight_layout()
    plt.savefig(f"visualization_epoch_{epoch+1}.png")
    plt.close()

# Training loop
num_epochs = 10  # Adjust the number of epochs as needed
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for i in range(256):
        inp = torch.tensor(grid_features[i, 0], dtype=torch.float32).reshape(1, 180*360, 3)
        tar = torch.tensor(grid_features[i, 1], dtype=torch.float32).reshape(1, 180*360, 3)

        optimizer.zero_grad()

        latent_grid_nodes, latent_mesh_nodes, latent_edge_emb = run_encoder(inp)
        latent_mesh_nodes = run_processor(latent_mesh_nodes)
        output_grid_nodes = run_decoder(latent_mesh_nodes, latent_grid_nodes)

        loss = criterion(output_grid_nodes, tar)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Data Point [{i+1}/256], Loss: {loss.item()}")

    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss / 256}")

    # Save visualization after every epoch
    save_visualization(epoch, inp, tar, output_grid_nodes)

# Save the model if needed
torch.save(encodes.state_dict(), 'encoder.pth')
torch.save(processes.state_dict(), 'processor.pth')
torch.save(decodes.state_dict(), 'decoder.pth')

