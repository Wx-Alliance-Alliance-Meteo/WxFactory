===================================================================
Instructions to RUN Graph Neural Network on Cubed Sphere Output
===================================================================

- Example run case:

--Train on Case 2 data and test on Case 2 data:
=============

- Step 1:

-- Store the output of Classical solver in "out_case2_random.nc"

netcdf_to_nn_input.py:

-- Line 17: Add the path to "out_case2_random.nc".

-- run python3 netcdf_to_nn_input.py

the code will output in these 2 files:
"edges_dataset_case2_base_DIM200.6.30.30.pt"
"node_features_dataset_case2_base_DIM200.6.30.30.pt"

200.6.30.30 --> timesteps.panels.xDim.yDim
==============

- Step 2:

nn_input_to_nn_train.py:

-- Line 23, 24: Add the path to node_features_dataset and edges_dataset .pt files.

-- Line 38 - 50: Normalization of input data using mean and std for each node feature.

-- Next we define the GNNModel class with GNNEncoder - GNNProcessor - GNNDecoder.

-- Line 234: Instantiate the GNNModel.

-- Line 249 - 306: Training Loop for the model, with each epoch running on full dataset.

-- Line 309: Save the model to file "NNmodel.pth".
==============

-Step 3:

nn_test_to_netcdf.py:

-- Line 23: Loads the "NNModel.pth" to initialize the weights and biases of the system.

-- Line 24: Renormalizes t = 1 timestep data to be fed into the model testing.

-- Line 150 - 200: Autoregressive testing for t = 200 timesteps.

-- Line 220: Outputs the GNN output in "nn_out_case6_base.nc".
==============

-Step 4:

bash panoply.sh:

-- Open "nn_out_case6_base.nc" and Visualize the results :)
==============
