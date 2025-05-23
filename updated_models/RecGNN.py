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


class RecG(MessagePassing): #ex ARMAConv
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 act: Optional[str] = "relu",#Optional[Callable] = ReLU(),
                 num_stacks: int = 1,
                 toll: float = 0.1,
                 bias: bool = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_stacks = num_stacks
        self.toll = toll
        activation_map = {
            "relu": ReLU,
            "sigmoid": Sigmoid,
            "tanh": Tanh
        }
        if act not in activation_map:
            raise ValueError(f"Funzione di attivazione non valida: {act}. Scegli tra: {list(activation_map.keys())}")
        self.act = activation_map[act]()

        F_in, F_out = in_channels, out_channels

        self.weight = Parameter(torch.empty(F_out, F_out))
        if in_channels > 0:
            self.init_weight = Parameter(torch.empty(F_in, F_out))

        if bias:
            self.bias = Parameter(torch.empty(24, F_out))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        
        #
        self.list_t = []
        self.list_max = []


    def reset_parameters(self):
        super().reset_parameters()
        torch.nn.init.xavier_uniform_(self.weight)
        self.weight = torch.nn.Parameter(self.weight/4)
        if not isinstance(self.init_weight, torch.nn.UninitializedParameter):
            torch.nn.init.xavier_uniform_(self.init_weight)
            self.init_weight = torch.nn.Parameter(self.init_weight/4)
        zeros(self.bias)


    def forward(self, x: Tensor,
                edge_index: Adj,
                edge_weight: OptTensor = None,
                lambda_max: OptTensor = None) -> Tensor:

        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim),
                add_self_loops=False, flow=self.flow, dtype=x.dtype)

        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim),
                add_self_loops=False, flow=self.flow, dtype=x.dtype)

        out = x
        out = out @ self.init_weight
        out = self.propagate(edge_index, x=out, edge_weight=edge_weight)
        if self.bias is not None:
            out = out + self.bias  
        out = self.act(out)

        t = 1
        #print(t)
        out_i = F.pad(x, (0, self.out_channels-self.in_channels))
        #print('-----------------ciclo------------------')
        #print('----entra?---- '+str(abs(out_i - out).sum()))
        max_abs_diff = torch.tensor(0)
        while torch.abs(out-out_i).sum() > self.toll:
            #max_abs_diff = torch.max(torch.abs(out-out_i).sum(), max_abs_diff)
            try:
                out_i = out
                out = out @ self.weight
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                out = self.propagate(edge_index, x=out, edge_weight=edge_weight)
    
                if self.bias is not None:
                    out = out + self.bias

                if self.act is not None:
                    out = self.act(out)
                t = t+1
                #print(t)
                #print('----continua?---- '+str(abs(out_i - out).sum()))
                if t >100:
                    #print(t)
                    raise ValueError("Condizione di uscita raggiunta!")
            except ValueError as e:
                #print(out_i)
                #print(out)
                #print('ciclo finito '+str(abs(out-out_i).sum()))
                out_i = out
                #print(f"Ciclo interrotto con errore: {e}")
        #self.list_max.append(max_abs_diff)
        self.list_t.append(t)
        return out


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
                f'{self.out_channels}, num_stacks={self.num_stacks}, '
                f'act={self.act})')