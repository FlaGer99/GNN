from typing import Callable, Optional

import torch
from torch import Tensor
from torch.nn import Parameter, ReLU, Tanh, Sigmoid

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.typing import OptTensor
from torch_geometric.utils import get_laplacian

class ChebConv_w(MessagePassing):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_stacks: int = 1,
        num_layers: int = 1,
        normalization: Optional[str] = 'sym',
        act: Optional[Callable] = ReLU(),
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert num_stacks > 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.num_stacks = num_stacks
        self.num_layers = num_layers
        activation_map = {
            "relu": ReLU,
            "sigmoid": Sigmoid,
            "tanh": Tanh
        }
        if act not in activation_map:
            raise ValueError(f"Funzione di attivazione non valida: {act}. Scegli tra: {list(activation_map.keys())}")
        self.act = activation_map[act]()

        K, T, F_in, F_out = num_stacks, num_layers, in_channels, out_channels

        self.weight = Parameter(torch.empty(max(1, T - 1), F_out, F_out))
        if in_channels > 0:
            self.init_weight = Parameter(torch.empty(F_in, F_out))

        if K > 1:
            self.lins = torch.nn.ModuleList([Linear(in_channels, out_channels,
                                                    bias=False, weight_initializer='glorot') for _ in range(K)])
        if K > 1:
            self.lin_o = Linear(out_channels, in_channels, bias = False, weight_initializer='glorot')
        
        # if K > 1:
        #     self.lins = torch.nn.ModuleList()
        #     self.lins.append(Linear(in_channels, out_channels, bias=False, weight_initializer='glorot'))
        #     self.lins.extend([Linear(out_channels, out_channels, bias=False, weight_initializer='glorot') for _ in range(K - 1)])


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
        if self.num_stacks > 1:
            for lin in self.lins:
                lin.reset_parameters()
        if self.num_stacks > 1:
            self.lin_o.reset_parameters()
        zeros(self.bias)


    def __norm__(
        self,
        edge_index: Tensor,
        num_nodes: Optional[int],
        edge_weight: OptTensor,
        normalization: Optional[str],
        lambda_max: OptTensor = None,
        dtype: Optional[int] = None,
        batch: OptTensor = None,
    ):
        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)
        assert edge_weight is not None

        if lambda_max is None:
            lambda_max = 2.0 * edge_weight.max()
        elif not isinstance(lambda_max, Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=dtype,
                                      device=edge_index.device)
        assert lambda_max is not None

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        loop_mask = edge_index[0] == edge_index[1]
        edge_weight[loop_mask] -= 1

        return edge_index, edge_weight

    def weighted_propagation(
        self,
        x: Tensor,
        edge_index: Tensor,
        norm: OptTensor = None,
    ) -> Tensor:

        out = x
        for t in range(self.num_layers):
            #print('-weighted-')
            if t == 0:
                out = out @ self.init_weight
            else:
                out = out @ self.weight[t - 1]
                #print(self.weight[t-1])
            #print('------')
            #print(out.shape)
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            out = self.propagate(edge_index, x=out, norm=norm)
            #print('++++++')
            if self.bias is not None:
                out = out + self.bias[t]

            if self.act is not None:
                out = self.act(out)

        if self.num_stacks > 1:
            out = self.lin_o(out)
        #print('******')
        #print(out.shape)
        return out
        
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
        lambda_max: OptTensor = None,
    ) -> Tensor:

        edge_index, norm = self.__norm__(
            edge_index,
            x.size(self.node_dim),
            edge_weight,
            self.normalization,
            lambda_max,
            dtype=x.dtype,
            batch=batch,
        )

        if self.num_stacks > 1:
            Tx_0 = x
            Tx_1 = x  # Dummy.
            out = self.lins[0](Tx_0)
        elif self.num_stacks == 1:
            out = self.weighted_propagation(edge_index=edge_index, x=x, norm=norm)
        #print(x.shape)
        #print(out.shape)

        # propagate_type: (x: Tensor, norm: Tensor)
        if self.num_stacks > 1:
            if len(self.lins) > 1:
                Tx_1 = self.weighted_propagation(edge_index=edge_index, x=x, norm=norm)
                out = out + self.lins[1](Tx_1)
    
            for lin in self.lins[2:]:
                Tx_2 = self.weighted_propagation(edge_index=edge_index, x=Tx_1, norm=norm)
                Tx_2 = 2. * Tx_2 - Tx_0
                out = out + lin.forward(Tx_2)
                Tx_0, Tx_1 = Tx_1, Tx_2

        #print('+-+-+-+-++-+-+-+-+-+-+-+-+-+-+-+-+-+-')
        #print(out.shape)
        return out


    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}, '
                f'normalization={self.normalization})')