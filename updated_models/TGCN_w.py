import torch
from torch.nn import Tanh, ReLU
from typing import Callable, Optional
from updated_models.GCNconv_up import GCNconv
import torch.nn.init as init

class TGCN_W(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act: Optional[str] = "relu",
        num_stacks: int = 1,
        num_layers: int = 1,
    ):
        super(TGCN_W, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_stacks = num_stacks
        self.num_layers = num_layers
        self.act = act

        self._create_parameters_and_layers()
        #print(self.conv_h.__dict__)
        self._set_parameters()
    

    def _create_update_gate_parameters_and_layers(self):

        self.conv_z = GCNconv(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            num_stacks = self.num_stacks,
            num_layers = self.num_layers,
            act = self.act,
        )

        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):

        self.conv_r = GCNconv(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            num_stacks = self.num_stacks,
            num_layers = self.num_layers,
            act = self.act,
        )

        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):

        self.conv_h = GCNconv(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            num_stacks = self.num_stacks,
            num_layers = self.num_layers,
            act = self.act,
        )

        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_parameters(self):
        init.xavier_uniform_(self.linear_z.weight)
        init.xavier_uniform_(self.linear_r.weight)
        init.xavier_uniform_(self.linear_h.weight)
        init.zeros_(self.linear_z.bias)
        init.zeros_(self.linear_r.bias)
        init.zeros_(self.linear_h.bias)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = torch.cat([self.conv_z(X, edge_index, edge_weight), H], axis=1)
        Z = self.linear_z(Z)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = torch.cat([self.conv_r(X, edge_index, edge_weight), H], axis=1)
        R = self.linear_r(R)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = torch.cat([self.conv_h(X, edge_index, edge_weight), H * R], axis=1)
        H_tilde = self.linear_h(H_tilde)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H_tilde + (1 - Z) * H   #########################
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H