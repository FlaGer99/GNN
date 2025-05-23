import torch
from torch.nn import Parameter
from typing import Callable, Optional
from updated_models.Cheb import ChebConv_w
from torch_geometric.nn.inits import glorot, zeros


class GConvLSTM_W_(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_stacks: int = 1,
        num_layers: int = 1,
        normalization: str = "sym",
        bias: bool = True,
        act: Optional[str] = "relu",
    ):
        super(GConvLSTM_W_, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_stacks = num_stacks
        self.num_layers = num_layers
        self.normalization = normalization
        self.bias = bias
        self.act = act
        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_input_gate_parameters_and_layers(self):

        self.conv_x_i = ChebConv_w(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            num_stacks = self.num_stacks,
            num_layers = self.num_layers,
            act = self.act,
        )

        self.conv_h_i = ChebConv_w(
            in_channels = self.out_channels,
            out_channels = self.out_channels,
            num_stacks = self.num_stacks,
            num_layers = self.num_layers,
            act = self.act,
        )

        #self.w_c_i = Parameter(torch.Tensor(1, self.out_channels))
        self.b_i = Parameter(torch.Tensor(1, self.out_channels))

    def _create_forget_gate_parameters_and_layers(self):

        self.conv_x_f = ChebConv_w(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            num_stacks = self.num_stacks,
            num_layers = self.num_layers,
            act = self.act,
        )

        self.conv_h_f = ChebConv_w(
            in_channels = self.out_channels,
            out_channels = self.out_channels,
            num_stacks = self.num_stacks,
            num_layers = self.num_layers,
            act = self.act,
        )

        #self.w_c_f = Parameter(torch.Tensor(1, self.out_channels))
        self.b_f = Parameter(torch.Tensor(1, self.out_channels))

    def _create_cell_state_parameters_and_layers(self):

        self.conv_x_c = ChebConv_w(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            num_stacks = self.num_stacks,
            num_layers = self.num_layers,
            act = self.act,
        )

        self.conv_h_c = ChebConv_w(
            in_channels = self.out_channels,
            out_channels = self.out_channels,
            num_stacks = self.num_stacks,
            num_layers = self.num_layers,
            act = self.act,
        )

        self.b_c = Parameter(torch.Tensor(1, self.out_channels))

    def _create_output_gate_parameters_and_layers(self):

        self.conv_x_o = ChebConv_w(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            num_stacks = self.num_stacks,
            num_layers = self.num_layers,
            act = self.act,
        )

        self.conv_h_o = ChebConv_w(
            in_channels = self.out_channels,
            out_channels = self.out_channels,
            num_stacks = self.num_stacks,
            num_layers = self.num_layers,
            act = self.act,
        )

        #self.w_c_o = Parameter(torch.Tensor(1, self.out_channels))
        self.b_o = Parameter(torch.Tensor(1, self.out_channels))

    def _create_parameters_and_layers(self):
        self._create_input_gate_parameters_and_layers()
        self._create_forget_gate_parameters_and_layers()
        self._create_cell_state_parameters_and_layers()
        self._create_output_gate_parameters_and_layers()

    def _set_parameters(self):
        #glorot(self.w_c_i)
        #glorot(self.w_c_f)
        #glorot(self.w_c_o)
        zeros(self.b_i)
        zeros(self.b_f)
        zeros(self.b_c)
        zeros(self.b_o)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _set_cell_state(self, X, C):
        if C is None:
            C = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return C

    def _calculate_input_gate(self, X, edge_index, edge_weight, H, lambda_max):
        I = self.conv_x_i(X, edge_index, edge_weight, lambda_max=lambda_max)
        I = I + self.conv_h_i(H, edge_index, edge_weight, lambda_max=lambda_max)
        #I = I + (self.w_c_i * C)
        I = I + self.b_i
        I = torch.sigmoid(I)
        return I

    def _calculate_forget_gate(self, X, edge_index, edge_weight, H, lambda_max):
        F = self.conv_x_f(X, edge_index, edge_weight, lambda_max=lambda_max)
        F = F + self.conv_h_f(H, edge_index, edge_weight, lambda_max=lambda_max)
        #F = F + (self.w_c_f * C)
        F = F + self.b_f
        F = torch.sigmoid(F)
        return F

    def _calculate_cell_state(self, X, edge_index, edge_weight, H, C, I, F, lambda_max):
        T = self.conv_x_c(X, edge_index, edge_weight, lambda_max=lambda_max)
        T = T + self.conv_h_c(H, edge_index, edge_weight, lambda_max=lambda_max)
        T = T + self.b_c
        T = torch.tanh(T)
        C = F * C + I * T
        return C

    def _calculate_output_gate(self, X, edge_index, edge_weight, H, lambda_max):
        O = self.conv_x_o(X, edge_index, edge_weight, lambda_max=lambda_max)
        O = O + self.conv_h_o(H, edge_index, edge_weight, lambda_max=lambda_max)
        #O = O + (self.w_c_o * C)
        O = O + self.b_o
        O = torch.sigmoid(O)
        return O

    def _calculate_hidden_state(self, O, C):
        H = O * torch.tanh(C)
        return H

    
    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        C: torch.FloatTensor = None,
        lambda_max: torch.Tensor = None,
    ) -> torch.FloatTensor:
        
        H = self._set_hidden_state(X, H)
        C = self._set_cell_state(X, C)
        I = self._calculate_input_gate(X, edge_index, edge_weight, H, lambda_max)
        F = self._calculate_forget_gate(X, edge_index, edge_weight, H, lambda_max)
        C = self._calculate_cell_state(X, edge_index, edge_weight, H, C, I, F, lambda_max)
        O = self._calculate_output_gate(X, edge_index, edge_weight, H, lambda_max)
        H = self._calculate_hidden_state(O, C)
        return H, C