import torch
from torch.nn import Parameter
from updated_models.RecGNN import RecG
from updated_models.RecGNN_up import RecG_up
from torch_geometric.nn.inits import glorot, zeros
from typing import Optional


class RecGRU_W_up(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        #out_channels: int,
        num_stacks: int = 1,
        toll: float = 0.1,
        normalization: str = "sym",
        bias: bool = True,
        act: Optional[str] = "relu",
    ):
        super(RecGRU_W_up, self).__init__()

        self.in_channels = in_channels
        self.num_stacks = num_stacks
        self.toll = toll
        self.normalization = normalization
        self.bias = bias
        self.act = act
        
        self._create_parameters_and_layers()
        self._set_parameters()

    def _create_update_gate_parameters_and_layers(self):

        self.conv_x_z = RecG_up(
            in_channels = self.in_channels,
            num_stacks = self.num_stacks,
            toll = self.toll,
            act = self.act,
        )

        self.conv_h_z = RecG_up(
            in_channels = self.in_channels,
            num_stacks = self.num_stacks,
            toll = self.toll,
            act = self.act,
        )
        self.w_q_z = Parameter(torch.empty(self.in_channels, self.in_channels))
        self.w_x_z = Parameter(torch.empty(self.in_channels, self.in_channels))
        self.b_z = Parameter(torch.empty(24, self.in_channels))

    def _create_reset_gate_parameters_and_layers(self):

        self.conv_x_r = RecG_up(
            in_channels = self.in_channels,
            num_stacks = self.num_stacks,
            toll = self.toll,
            act = self.act,
        )

        self.conv_h_r = RecG_up(
            in_channels = self.in_channels,
            num_stacks = self.num_stacks,
            toll = self.toll,
            act = self.act,
        )
        self.w_q_r = Parameter(torch.empty(self.in_channels, self.in_channels))
        self.w_x_r = Parameter(torch.empty(self.in_channels, self.in_channels))
        self.b_r = Parameter(torch.empty(24, self.in_channels))

    def _create_candidate_state_parameters_and_layers(self):

        self.conv_x_h = RecG_up(
            in_channels = self.in_channels,
            num_stacks = self.num_stacks,
            toll = self.toll,
            act = self.act,
        )

        self.conv_h_h = RecG_up(
            in_channels = self.in_channels,
            num_stacks = self.num_stacks,
            toll = self.toll,
            act = self.act,
        )
        self.w_q_h = Parameter(torch.empty(self.in_channels, self.in_channels))
        self.w_x_h = Parameter(torch.empty(self.in_channels, self.in_channels))
        self.b_h = Parameter(torch.empty(24, self.in_channels))

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_parameters(self):
        glorot(self.w_x_r)
        glorot(self.w_x_z)
        glorot(self.w_x_h)
        glorot(self.w_q_r)
        glorot(self.w_q_z)
        glorot(self.w_q_h)
        zeros(self.b_r)
        zeros(self.b_z)
        zeros(self.b_h)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.in_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H, lambda_max):
        Z = self.conv_x_z(X, edge_index, edge_weight, lambda_max=lambda_max) @ self.w_x_z.T
        Z = Z + self.conv_h_z(H, edge_index, edge_weight, lambda_max=lambda_max) @ self.w_q_z.T
        Z = torch.sigmoid(Z+self.b_z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H, lambda_max):
        R = self.conv_x_r(X, edge_index, edge_weight, lambda_max=lambda_max) @ self.w_x_r.T
        R = R + self.conv_h_r(H, edge_index, edge_weight, lambda_max=lambda_max) @ self.w_q_r.T
        R = torch.sigmoid(R+self.b_r)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R, lambda_max):
        H_tilde = self.conv_x_h(X, edge_index, edge_weight, lambda_max=lambda_max) @ self.w_x_h.T
        H_tilde = H_tilde + self.conv_h_h(H * R, edge_index, edge_weight, lambda_max=lambda_max) @ self.w_q_h.T
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