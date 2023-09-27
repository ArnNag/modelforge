import torch.nn as nn
from typing import Dict, Callable

from .models import BaseNNP
from .utils import (
    EnergyReadout,
    GaussianRBF,
    scatter_softmax
)
import torch


class Sake(BaseNNP):
    def __init__(
            self,
            n_atom_basis: int,
            n_interactions: int,
            n_filters: int = 0,
            n_rbf: int = 50,
            cutoff: float = 5.0,
            update: bool = True

    ) -> None:
        """
        Initialize the Sake class.


        Parameters
        ----------
        n_atom_basis : int
            Number of atom basis, defines the dimensionality of the output features.
        n_interactions : int
            Number of interaction blocks in the architecture.
        n_filters : int, optional
            Number of filters, defines the dimensionality of the intermediate features.
            Default is 0.
        cutoff : float, optional
            Cutoff value for the pairlist. Default is 5.0.
        """
        from .models import PairList

        super().__init__()

        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.calculate_distances_and_pairlist = PairList(cutoff)

        self.readout = EnergyReadout(n_atom_basis)

        self.interaction = SakeInteractionBlock(
            n_atom_basis=self.n_atom_basis,
            out_features=n_filters,
            hidden_features=n_filters,
            n_rbf=n_rbf,
            update=update,
            cutoff=cutoff
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """
        Compute atomic representations/embeddings.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.
        """
        # get tensors from input dictionary
        Z = inputs["Z"]
        mask = Z == -1
        pairlist = self.calculate_distances_and_pairlist(mask, inputs["R"])

        q = nn.functional.one_hot(Z, self.n_atom_basis)
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)

        idx_i, idx_j = pairlist["atom_index12"]

        for i in range(self.n_interactions):
            q = self.interaction(
                q,
                mu,
                pairlist["r_ij"],
                pairlist["d_ij"],
                idx_i,
                idx_j
            )

        q = q.squeeze(1)

        return self.readout(q)


class SakeInteractionBlock(nn.Module):
    def __init__(
            self,
            n_atom_basis: int,
            out_features: int,
            hidden_features: int,
            n_rbf: int = 50,
            activation: nn.Module = torch.nn.SiLU(),
            n_heads: int = 4,
            update: bool = True,
            use_semantic_attention: bool = True,
            use_euclidean_attention: bool = True,
            use_spatial_attention: bool = True,
            cutoff: Callable = None
    ):
        """
        Initialize the Sake interaction block.

        Parameters
        ----------
        n_atom_basis : int
            Number of atom basis, defines the dimensionality of the output features.

        """
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.activation = activation
        self.n_heads = n_heads
        self.update = update
        self.use_semantic_attention = use_semantic_attention
        self.use_euclidean_attention = use_euclidean_attention
        self.use_spatial_attention = use_spatial_attention
        self.cutoff = cutoff
        self.edge_model = ContinuousFilterConvolutionWithConcatenation(self.hidden_features, self.out_features, n_rbf)
        self.n_coefficients = self.n_heads * self.hidden_features

        self.node_mlp = nn.Sequential(
            nn.Linear(self.hidden_features, self.hidden_features),
            self.activation,
            nn.Linear(self.hidden_features, self.out_features),
            self.activation
        )

        self.semantic_attention_mlp = nn.Sequential(
            nn.Linear(self.hidden_features, self.n_heads),
            nn.CELU(alpha=2.0)
        )

        self.post_norm_mlp = nn.Sequential(
            nn.Linear(self.n_coefficients, self.hidden_features),
            self.activation,
            nn.Linear(self.hidden_features, self.hidden_features),
            self.activation
        )

        self.x_mixing = nn.Sequential(
            nn.Linear(self.n_coefficients, self.n_coefficients, bias=False),
            nn.Tanh()
        )

        log_gamma = -torch.log(torch.linspace(1.0, 5.0, self.n_heads))
        if self.use_semantic_attention and self.use_euclidean_attention:
            self.log_gamma = nn.Parameter(log_gamma)
        else:
            self.log_gamma = nn.Parameter(torch.ones(self.n_heads))

    def spatial_attention(self, q_ij_mtx, r_ij, d_ij, idx_j, n_atoms):
        # q_ij_mtx shape: (n_pairs,  n_coefficients)
        # coefficients shape: (n_pairs, n_coefficients)
        coefficients = self.x_mixing(q_ij_mtx)

        # d_ij shape: (n_pairs, 3)
        r_ij = r_ij / (d_ij + 1e-5)

        # p: pair axis; x: position axis, c: coefficient axis
        combinations = torch.einsum("px,pc->pcx", r_ij, coefficients)
        out_shape = (n_atoms, self.n_coefficients, 3)
        combinations_sum = torch.zeros(out_shape).scatter_reduce(0,
                                                                 idx_j,
                                                                 combinations,
                                                                 "mean",
                                                                 include_self=False
                                                                 )
        combinations_norm = (combinations_sum ** 2).sum(-1)
        q_combinations = self.post_norm_mlp(combinations_norm)
        return q_combinations

    def aggregate(self, q_ij_mtx, idx_j, n_atoms):
        out_shape = (n_atoms, self.n_coefficients)
        return torch.zeros(out_shape).scatter_add(0, idx_j, q_ij_mtx)

    def node_model(self, q, q_ij, q_combinations):
        out = torch.cat([q, q_ij, q_combinations], dim=-1)
        out = self.node_mlp(out)
        out = q + out
        return out

    def semantic_attention(self, q_ij_mtx, idx_j, n_atoms):
        # att shape: (n_pairs, n_heads)
        att = self.semantic_attention_mlp(q_ij_mtx)
        semantic_attention = scatter_softmax(att, idx_j, dim_size=n_atoms)
        return semantic_attention

    def combined_attention(self, d_ij, q_ij_mtx, idx_j, n_atoms):
        # semantic_attention shape: (n_pairs, n_heads)
        semantic_attention = self.semantic_attention(q_ij_mtx, idx_j, n_atoms)
        if self.cutoff is not None:
            euclidean_attention = self.cutoff(d_ij)
        else:
            euclidean_attention = 1.0

        # combined_attention shape: (n_pairs, n_heads)
        combined_attention = euclidean_attention * semantic_attention
        # combined_attention_agg shape: (n_nodes, n_heads)
        out_shape = (n_atoms, self.n_heads)
        combined_attention_agg = torch.zeros(out_shape).scatter_add(0, idx_j, combined_attention)
        combined_attention = combined_attention / combined_attention_agg[idx_j]

        return combined_attention

    def forward(
            self,
            q: torch.Tensor,
            mu: torch.Tensor,
            r_ij: torch.Tensor,
            d_ij: torch.Tensor,
            idx_i: torch.Tensor,
            idx_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the interaction block.

        Parameters
        ----------
        q: scalar input values
        mu: vector input values
        r_ij: torch.Tensor, shape [n_pairs, 3]
            Displacement vectors
        d_ij: torch.Tensor, shape [n_pairs]
            Distances
        idx_i : torch.Tensor, shape [n_pairs]
            Indices for the first atom in each pair.
        idx_j : torch.Tensor, shape [n_pairs]
            Indices for the second atom in each pair.

        Returns
        -------
        torch.Tensor, shape [batch_size, n_atoms, n_atom_basis]
            Updated feature tensor after interaction block.
        """
        n_atoms = q.shape[0]
        # qi_cat_qj shape: (n_pairs, hidden_features * 2 [concatenated sender and receiver]) 
        qi_cat_qj = torch.cat([q[idx_i], q[idx_j]], -1)

        # q_ij_mtx shape: (n_pairs, hidden_features)
        q_ij_mtx = self.edge_model(qi_cat_qj)
        # combined_attention shape: (n_pairs, n_heads)
        combined_attention = self.combined_attention(d_ij, q_ij_mtx, idx_j, n_atoms)
        # p: pair axis; f: hidden feature axis; h: head axis
        q_ij_att = torch.einsum("pf,ph->pfh", q_ij_mtx, combined_attention)
        # q_ij_att shape before reshape: (n_pairs, hidden_features, n_heads)
        q_ij_att = torch.reshape(q_ij_att, q_ij_att.shape[:-2] + (-1,))
        # q_ij_att shape after reshape: (n_pairs, n_coefficients)
        q_combinations = self.spatial_attention(q_ij_att, r_ij, d_ij, idx_j, n_atoms)

        if not self.use_spatial_attention:
            q_combinations = torch.zeros_like(q_combinations)

        q_ij = self.aggregate(q_ij_att, idx_j, n_atoms)
        q = self.node_model(q, q_ij, q_combinations)

        return q


class ContinuousFilterConvolutionWithConcatenation(nn.Module):

    def __init__(self, in_features, out_features, n_rbf, activation=torch.nn.SiLU()):
        super().__init__()
        self.kernel = GaussianRBF(n_rbf=n_rbf, cutoff=5.0, trainable=True)
        self.mlp_in = nn.Linear(n_rbf, n_rbf)
        self.mlp_out = nn.Sequential(
            nn.Linear(n_rbf, out_features),
            activation,
            nn.Linear(out_features, out_features),
        )

    def forward(self, q, mu):
        q0 = q
        q = self.mlp_in(q)
        _mu = self.kernel(mu) * q

        q = self.mlp_out(
            torch.cat(
                [q0, _mu, mu],
                dim=-1
            )
        )

        return q
