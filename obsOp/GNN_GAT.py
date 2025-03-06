#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch_geometric


# # Model 

# In[ ]:


class GNN_GAT(torch.nn.Module):
    def __init__(self, list_predictors, list_targets, activation, weight_initializer, conv_filters, batch_normalization, heads=1):
        super(GNN_GAT, self).__init__()
        self.list_predictors = list_predictors
        self.list_targets = list_targets
        self.activation = activation
        self.weight_initializer = weight_initializer
        self.conv_filters = conv_filters
        self.batch_normalization = batch_normalization
        self.n_conv_layers = len(self.conv_filters)
        self.heads = heads

        self.conv_1 = torch_geometric.nn.GATConv(len(self.list_predictors), self.conv_filters[0] // heads, heads = heads)
        self.conv_2 = torch_geometric.nn.GATConv(self.conv_filters[0], self.conv_filters[1] // heads, heads = heads)
        if self.n_conv_layers == 3:
            self.conv_3 = torch_geometric.nn.GATConv(self.conv_filters[1], self.conv_filters[2] // heads, heads = heads)

        self.batch_norm_conv_1 = torch.nn.BatchNorm1d(self.conv_filters[0])
        self.batch_norm_conv_2 = torch.nn.BatchNorm1d(self.conv_filters[1])
        if self.n_conv_layers == 3:
            self.batch_norm_conv_3 = torch.nn.BatchNorm1d(self.conv_filters[2])

        self.pool = torch_geometric.nn.global_mean_pool

        self.dense_1 = torch.nn.Linear(self.conv_filters[-1], self.conv_filters[-1] // 2)
        self.dense_2 = torch.nn.Linear(self.conv_filters[-1] // 2, self.conv_filters[-1] // 4)
        self.batch_norm_dense_1 = torch.nn.BatchNorm1d(conv_filters[-1] // 2)
        self.batch_norm_dense_2 = torch.nn.BatchNorm1d(conv_filters[-1] // 4)

        self.fc = torch.nn.Linear(self.conv_filters[-1] // 4, len(self.list_targets))

        self._initialize_weights()

    def _initialize_weights(self):
        if self.weight_initializer is not None:
            for m in self.modules():
                if isinstance(m, torch.nn.Linear):
                    self.weight_initializer(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif isinstance(m, torch_geometric.nn.GATConv):
                    self.weight_initializer(m.lin.weight)
                    if m.lin.bias is not None:
                        torch.nn.init.zeros_(m.lin.bias)

    def convolutional_block(self, x, edge_index, conv, batch_norm):
        x = conv(x, edge_index)  # Convolution
        if self.batch_normalization:
            x = batch_norm(x)  # Batch normalization
        x = self.activation(x)  # Apply activation
        return x

    def dense_block(self, x, dense, batch_norm):
        x = dense(x)
        if self.batch_normalization:
            x = batch_norm(x)
        x = self.activation(x)
        return x

    def forward(self, node_features, edge_index, batch):
        # Convolutional blocks
        x = self.convolutional_block(node_features, edge_index, self.conv_1, self.batch_norm_conv_1)
        x = self.convolutional_block(x, edge_index, self.conv_2, self.batch_norm_conv_2)
        if self.n_conv_layers == 3:
            x = self.convolutional_block(x, edge_index, self.conv_3, self.batch_norm_conv_3)

        # Pool the node embeddings
        x = self.pool(x, batch)

        # Dense layers
        x = self.dense_block(x, self.dense_1, self.batch_norm_dense_1)
        x = self.dense_block(x, self.dense_2, self.batch_norm_dense_2)

        # Output layer
        output = self.fc(x)
        return output

