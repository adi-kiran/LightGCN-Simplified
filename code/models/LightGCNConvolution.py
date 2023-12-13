from torch_geometric.nn.conv import MessagePassing
import torch_scatter

class LightGCNConv(MessagePassing):
    def __init__(self, **kwargs):
        super(LightGCNConv, self).__init__(node_dim=0, **kwargs)

    def forward(self, x, edge_index, size=None):
        return self.propagate(edge_index=edge_index, x=(x[0], x[1]), size=size)

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index, dim_size=None):
        return torch_scatter.scatter(src=inputs, index=index, dim=0, dim_size=dim_size, reduce='mean')