import torch
from torch.nn import Parameter
from updated_models.Cheb import ChebConv_w
from torch_geometric.nn.inits import glorot, zeros
from typing import Callable, Optional


class GConvGRU_W(torch.nn.Module):

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
        super(GConvGRU_W, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_stacks = num_stacks
        self.num_layers = num_layers
        self.normalization = normalization
        self.bias = bias
        self.act = act
        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_update_gate_parameters_and_layers(self):

        self.conv_x_z = ChebConv_w(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            num_stacks = self.num_stacks,
            num_layers = self.num_layers,
            act = self.act,
        )

        self.conv_q_z = ChebConv_w(
            in_channels = self.out_channels,
            out_channels = self.out_channels,
            num_stacks = self.num_stacks,
            num_layers = self.num_layers,
            act = self.act,
        )

        self.w_q_z = Parameter(torch.empty(self.out_channels, self.out_channels))
        self.w_x_z = Parameter(torch.empty(self.out_channels, self.out_channels))
        self.b_z = Parameter(torch.empty(24, self.out_channels))

    def _create_reset_gate_parameters_and_layers(self):

        self.conv_x_r = ChebConv_w(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            num_stacks = self.num_stacks,
            num_layers = self.num_layers,
            act = self.act,
        )

        self.conv_q_r = ChebConv_w(
            in_channels = self.out_channels,
            out_channels = self.out_channels,
            num_stacks = self.num_stacks,
            num_layers = self.num_layers,
            act = self.act,
        )

        self.w_q_r = Parameter(torch.empty(self.out_channels, self.out_channels))
        self.w_x_r = Parameter(torch.empty(self.out_channels, self.out_channels))
        self.b_r = Parameter(torch.empty(24, self.out_channels))

    def _create_candidate_state_parameters_and_layers(self):

        self.conv_x_h = ChebConv_w(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            num_stacks = self.num_stacks,
            num_layers = self.num_layers,
            act = self.act,
        )

        self.conv_q_h = ChebConv_w(
            in_channels = self.out_channels,
            out_channels = self.out_channels,
            num_stacks = self.num_stacks,
            num_layers = self.num_layers,
            act = self.act,
        )

        self.w_q_h = Parameter(torch.empty(self.out_channels, self.out_channels))
        self.w_x_h = Parameter(torch.empty(self.out_channels, self.out_channels))
        self.b_h = Parameter(torch.empty(24, self.out_channels))

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_parameters(self):
        glorot(self.w_x_z)
        glorot(self.w_x_r)
        glorot(self.w_x_h)
        glorot(self.w_q_r)
        glorot(self.w_q_z)
        glorot(self.w_q_h)
        zeros(self.b_r)
        zeros(self.b_z)
        zeros(self.b_h)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H, lambda_max):
        Z = self.conv_x_z(X, edge_index, edge_weight, lambda_max=lambda_max) @ self.w_x_z.T
        Z = Z + self.conv_q_z(H, edge_index, edge_weight, lambda_max=lambda_max) @ self.w_q_z.T
        Z = torch.sigmoid(Z+self.b_z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H, lambda_max):
        R = self.conv_x_r(X, edge_index, edge_weight, lambda_max=lambda_max) @ self.w_x_r.T
        R = R + self.conv_q_r(H, edge_index, edge_weight, lambda_max=lambda_max) @ self.w_q_r.T
        R = torch.sigmoid(R+self.b_r)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R, lambda_max):
        H_tilde = self.conv_x_h(X, edge_index, edge_weight, lambda_max=lambda_max) @ self.w_x_h.T
        H_tilde = H_tilde + self.conv_q_h(H * R, edge_index, edge_weight, lambda_max=lambda_max) @ self.w_q_h.T
        H_tilde = torch.tanh(H_tilde+self.b_h)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H_tilde + (1 - Z) * H
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
        lambda_max: torch.Tensor = None,
    ) -> torch.FloatTensor:
        
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H, lambda_max)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H, lambda_max)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R, lambda_max)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H