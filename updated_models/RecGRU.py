import torch
from updated_models.RecGNN import RecG
from updated_models.RecGNN_up import RecG_up
from torch_geometric.nn.inits import glorot, zeros
from typing import Optional


class RecGRU_W(torch.nn.Module):

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
        super(RecGRU_W, self).__init__()

        self.in_channels = in_channels
        #self.out_channels = out_channels
        self.num_stacks = num_stacks
        self.toll = toll
        self.normalization = normalization
        self.bias = bias
        self.act = act
        
        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):

        self.conv_x_z = RecG_up(
            in_channels = self.in_channels,
            #out_channels = self.out_channels,
            num_stacks = self.num_stacks,
            toll = self.toll,
            act = self.act,
        )

        self.conv_h_z = RecG_up(
            in_channels = self.in_channels,#out
            #out_channels = self.out_channels,
            num_stacks = self.num_stacks,
            toll = self.toll,
            act = self.act,
        )

    def _create_reset_gate_parameters_and_layers(self):

        self.conv_x_r = RecG_up(
            in_channels = self.in_channels,
            #out_channels = self.out_channels,
            num_stacks = self.num_stacks,
            toll = self.toll,
            act = self.act,
        )

        self.conv_h_r = RecG_up(
            in_channels = self.in_channels,#out
            #out_channels = self.out_channels,
            num_stacks = self.num_stacks,
            toll = self.toll,
            act = self.act,
        )

    def _create_candidate_state_parameters_and_layers(self):

        self.conv_x_h = RecG_up(
            in_channels = self.in_channels,
            #out_channels = self.out_channels,
            num_stacks = self.num_stacks,
            toll = self.toll,
            act = self.act,
        )

        self.conv_h_h = RecG_up(
            in_channels = self.in_channels,#out
            #out_channels = self.out_channels,
            num_stacks = self.num_stacks,
            toll = self.toll,
            act = self.act,
        )

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.in_channels).to(X.device)             #out
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H, lambda_max):
        Z = self.conv_x_z(X, edge_index, edge_weight, lambda_max=lambda_max)
        Z = Z + self.conv_h_z(H, edge_index, edge_weight, lambda_max=lambda_max)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H, lambda_max):
        R = self.conv_x_r(X, edge_index, edge_weight, lambda_max=lambda_max)
        R = R + self.conv_h_r(H, edge_index, edge_weight, lambda_max=lambda_max)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R, lambda_max):
        H_tilde = self.conv_x_h(X, edge_index, edge_weight, lambda_max=lambda_max)
        H_tilde = H_tilde + self.conv_h_h(H * R, edge_index, edge_weight, lambda_max=lambda_max)
        H_tilde = torch.tanh(H_tilde)
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