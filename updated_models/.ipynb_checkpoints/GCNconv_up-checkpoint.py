from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter, ReLU, Tanh, Sigmoid

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm


class GCNconv(MessagePassing): #ex ARMAConv
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 act: Optional[str] = "relu",
                 num_layers: int = 1,
                 shared_weights: bool = False,
                 dropout: float = 0.,
                 bias: bool = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.shared_weights = shared_weights
        self.dropout = dropout
        activation_map = {
            "relu": ReLU,
            "sigmoid": Sigmoid,
            "tanh": Tanh
        }
        if act not in activation_map:
            raise ValueError(f"Funzione di attivazione non valida: {act}. Scegli tra: {list(activation_map.keys())}")
        self.act = activation_map[act]()

        T, F_in, F_out = num_layers, in_channels, out_channels
        T = 1 if self.shared_weights else T

        self.weight = Parameter(torch.empty(max(1, T - 1), F_out, F_out))
        if in_channels > 0:
            self.init_weight = Parameter(torch.empty(F_in, F_out))
            
        if bias:
            self.bias = Parameter(torch.empty(T, 24, F_out))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        if not isinstance(self.init_weight, torch.nn.UninitializedParameter):
            glorot(self.init_weight)
            #glorot(self.root_weight)
        zeros(self.bias)


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim),
                add_self_loops=False, flow=self.flow, dtype=x.dtype)

        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim),
                add_self_loops=False, flow=self.flow, dtype=x.dtype)

        x = x.unsqueeze(-3)
        out = x
        for t in range(self.num_layers):
            if t == 0:
                out = out @ self.init_weight
            else:
                out = out @ self.weight[0 if self.shared_weights else t - 1]

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            out = self.propagate(edge_index, x=out, edge_weight=edge_weight)

            #root = F.dropout(x, p=self.dropout, training=self.training)
            #root = root @ self.root_weight[0 if self.shared_weights else t]
            #out = out + root

            if self.bias is not None:
                out = out + self.bias[0 if self.shared_weights else t]
    
            out = self.act(out)
        # print('-------------')
        # print(out.mean(dim=-3).shape)
        return out.mean(dim=-3)


    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    # @torch.no_grad()
    # def initialize_parameters(self, module, input):
    #     if isinstance(self.init_weight, nn.parameter.UninitializedParameter):
    #         F_in, F_out = input[0].size(-1), self.out_channels
    #         T, K = self.weight.size(0) + 1, self.weight.size(1)
    #         self.init_weight.materialize((K, F_in, F_out))
    #         self.root_weight.materialize((T, K, F_in, F_out))
    #         glorot(self.init_weight)
    #         glorot(self.root_weight)

    #     module._hook.remove()
    #     delattr(module, '_hook')

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, '
                f'num_layers={self.num_layers})')