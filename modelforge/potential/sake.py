import torch.nn as nn
from loguru import logger
from typing import Dict, Tuple, List, Callable

from .models import BaseNNP
from .utils import (
    EnergyReadout,
    GaussianRBF,
    ShiftedSoftplus,
    cosine_cutoff,
    scatter_softmax
)
import torch


class Sake(BaseNNP):
    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        n_filters: int = 0,
        cutoff: float = 5.0,
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

        self.calculate_distances_and_pairlist = PairList(cutoff)

        self.representation = SakeRepresentation(
            n_atom_basis, n_filters, n_interactions
        )
        self.readout = EnergyReadout(n_atom_basis)
        self.embedding = nn.Embedding(100, n_atom_basis, padding_idx=-1)

    def calculate_energy(
        self, inputs: Dict[str, torch.Tensor], cached_pairlist: bool = False
    ) -> torch.Tensor:
        """
        Calculate the energy for a given input batch.

        Parameters
        ----------
        inputs : dict, contains
            - 'Z': torch.Tensor, shape [batch_size, n_atoms]
                Atomic numbers for each atom in each molecule in the batch.
            - 'R': torch.Tensor, shape [batch_size, n_atoms, 3]
                Coordinates for each atom in each molecule in the batch.
        cached_pairlist : bool, optional
            Whether to use a cached pairlist. Default is False. NOTE: is this really needed?
        Returns
        -------
        torch.Tensor, shape [batch_size]
            Calculated energies for each molecule in the batch.

        """
        # compute atom and pair features (see Fig1 in 10.1063/1.5019779)
        # initializing x^{l}_{0} as x^l)0 = aZ_i
        Z = inputs["Z"]
        x = self.embedding(Z)
        mask = Z == -1
        pairlist = self.calculate_distances_and_pairlist(mask, inputs["R"])

        x = self.representation(x, pairlist)
        # pool average over atoms
        return self.readout(x)


def sequential_block(in_features: int, out_features: int):
    """
    Create a sequential block for the neural network.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.

    Returns
    -------
    nn.Sequential
        Sequential layer block.
    """
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        ShiftedSoftplus(),
        nn.Linear(out_features, out_features),
    )


class SakeInteractionBlock(nn.Module):
    def __init__(
        self,
        n_rbf: int = 50,
        n_atom_basis: int,
        out_features: int,
        hidden_features: int,
        activation: Callable = jax.nn.silu, 
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
        n_filters : int
            Number of filters, defines the dimensionality of the intermediate features.

        """
        super().__init__()
        self.n_rbf = n_rbf
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
        self.edge_model = ContinuousFilterConvolutionWithConcatenation(self.hidden_features) #TODO: implement ContinuousFilterConvolutionWithConcatenation
        self.n_coefficients = self.n_heads * self.hidden_features

        self.node_mlp = nn.Sequential(
            [
                nn.Linear(float('nan'), self.hidden_features),
                self.activation,
                nn.Linear(self.hidden_features, self.out_features),
                self.activation,
            ]
        )

        self.semantic_attention_mlp = nn.Sequential(
            [
                nn.Dense(self.n_heads),
                nn.CELU(alpha=2.0),
            ],
        )

        self.post_norm_mlp = nn.Sequential(
            [
                nn.Linear(float('nan'), self.hidden_features),
                self.activation,
                nn.Linear(self,hidden_features, self.hidden_features),
                self.activation,
            ]
        )

        self.x_mixing = nn.Sequential(
            [
                nn.Linear(self.n_coefficients, self.n_coefficients, bias=False), #TODO: check the input dimension
                nn.Tanh()
            ]
        )

        log_gamma = -torch.log(torch.linspace(1.0, 5.0, self.n_heads))
        if self.use_semantic_attention and self.use_euclidean_attention:
            self.log_gamma = nn.Parameter(log_gamma)
        else:
            self.log_gamma = nn.Parameter(torch.ones(self.n_heads))

    def spatial_attention(self, h_e_mtx, x_minus_xt, x_minus_xt_norm, idx_j, n_atoms):
        # h_e_mtx shape: (n_pairs,  hidden_features * n_heads)
        # coefficients shape: (n_pairs, hidden_features * n_heads)
        coefficients = self.x_mixing(h_e_mtx)

        # x_minus_xt shape: (n_pairs, 3)
        x_minus_xt = x_minus_xt / (x_minus_xt_norm + 1e-5)

        # p: pair axis; x: position axis, c: coefficient axis
        combinations = torch.einsum("px,pc->pcx", x_minus_xt, coefficients)
        out_shape = (n_atoms, self.n_coefficients, 3)
        combinations_sum = torch.zeros(out_shape).scatter_reduce(0,
            idx_j,
            combinations,
            "mean",
            include_self=False
        )
        combinations_norm = (combinations_sum ** 2).sum(-1)
        h_combinations = self.post_norm_mlp(combinations_norm)
        return h_combinations

    def aggregate(self, h_e_mtx, idx_j, n_atoms):
        out_shape = (n_atoms, self.n_coefficients)
        return torch.zeros(out_shape).scatter_sum(0, idx_j, h_e_mtx)

    def node_model(self, h, h_e, h_combinations):
        out = torch.cat([h, h_e, h_combinations], dim=-1)
        out = self.node_mlp(out)
        out = h + out
        return out

    def semantic_attention(self, h_e_mtx, idx_j, n_atoms):
        # att shape: (n_pairs, n_heads)
        att = self.semantic_attention_mlp(h_e_mtx)
        semantic_attention = scatter_softmax(att, idx_j, dim_size=n_atoms)
        return semantic_attention

    def combined_attention(self, x_minus_xt_norm, h_e_mtx, idx_j, n_atoms):
        # semantic_attention shape: (n_pairs, n_heads)
        semantic_attention = self.semantic_attention(h_e_mtx, idx_j, n_atoms)
        if self.cutoff is not None:
            euclidean_attention = self.cutoff(x_minus_xt_norm)
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
        h: torch.Tensor,
        x: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the interaction block.

        Parameters
        ----------
        x : torch.Tensor, shape [batch_size, n_atoms, n_atom_basis]
            Input feature tensor for atoms.
            Radial basis functions for pairs of atoms.
        idx_i : torch.Tensor, shape [n_pairs]
            Indices for the first atom in each pair.
        idx_j : torch.Tensor, shape [n_pairs]
            Indices for the second atom in each pair.

        Returns
        -------
        torch.Tensor, shape [batch_size, n_atoms, n_atom_basis]
            Updated feature tensor after interaction block.
        """
        n_atoms = h.shape[0]
        # x_minus_xt shape: (n_pairs, 3)
        x_minus_xt = get_x_minus_xt_sparse(x, idx_i, idx_j) #TODO: implement get_x_minus_xt_sparse
        # x_minus_xt norm shape: (n_pairs, 1)
        x_minus_xt_norm = get_x_minus_xt_norm(x_minus_xt=x_minus_xt) #TODO: implement get_x_minus_xt_norm
        # h_cat_ht shape: (n_pairs, hidden_features * 2 [concatenated sender and receiver]) 
        h_cat_ht = get_h_cat_ht_sparse(h, idx_i, idx_j) #TODO: implement get_h_cat_ht_sparse

        if he is not None:
            h_cat_ht = torch.cat([h_cat_ht, he], -1)

        # h_e_mtx shape: (n_pairs, hidden_features)
        h_e_mtx = self.edge_model(h_cat_ht, x_minus_xt_norm)
        # combined_attention shape: (n_pairs, n_heads)
        combined_attention = self.combined_attention(x_minus_xt_norm, h_e_mtx, idx_j, n_atoms)
        # p: pair axis; f: hidden feature axis; h: head axis
        h_e_att = torch.einsum("pf,ph->pfh", h_e_mtx, combined_attention) 
        h_e_att = torch.reshape(h_e_att, h_e_att.shape[:-2] + (-1, ))
        # h_e_att shape after reshape: (n_pairs,  hidden_features * n_heads)
        h_combinations = self.spatial_attention(h_e_att, x_minus_xt, x_minus_xt_norm, idx_j, n_atoms)

        if not self.use_spatial_attention:
            h_combinations = torch.zeros_like(h_combinations)
            delta_v = torch.zeros_like(delta_v)

        h_e = self.aggregate(h_e_att, idx_j, n_atoms)
        h = self.node_model(h, h_e, h_combinations)

        return h


class SakeRepresentation(nn.Module):
    def __init__(
        self,
        n_atom_basis: int,
        n_filters: int,
        n_interactions: int,
    ):
        """
        Initialize the Sake representation layer.

        Parameters
        ----------
        n_atom_basis : int
            Number of atom basis.
        n_filters : int
            Number of filters.
        n_interactions : int
            Number of interaction layers.
        """
        super().__init__()

        self.interactions = nn.ModuleList(
            [
                SakeInteractionBlock(n_atom_basis, n_filters)
                for _ in range(n_interactions)
            ]
        )
        self.cutoff = 5.0
        self.radial_basis = GaussianRBF(n_rbf=20, cutoff=self.cutoff)

    def _distance_to_radial_basis(
        self, d_ij: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert distances to radial basis functions.

        Parameters
        ----------
        d_ij : torch.Tensor, shape [n_pairs]
            Pairwise distances between atoms.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - Radial basis functions, shape [n_pairs, n_rbf]
            - cutoff values, shape [n_pairs]
        """
        f_ij = self.radial_basis(d_ij)
        rcut_ij = cosine_cutoff(d_ij, self.cutoff)
        return f_ij, rcut_ij

    def forward(
        self, x: torch.Tensor, pairlist: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass for the representation layer.

        Parameters
        ----------
        x : torch.Tensor, shape [batch_size, n_atoms, n_atom_basis]
            Input feature tensor for atoms.
        pairlist: Dict[str, torch.Tensor]
            Pairlist dictionary containing the following keys:
            - 'atom_index12': torch.Tensor, shape [n_pairs, 2]
                Atom indices for pairs of atoms
            - 'd_ij': torch.Tensor, shape [n_pairs]
                Pairwise distances between atoms.
        Returns
        -------
        torch.Tensor, shape [batch_size, n_atoms, n_atom_basis]
            Output tensor after forward pass.
        """
        atom_index12 = pairlist["atom_index12"]
        d_ij = pairlist["d_ij"]

        f_ij, rcut_ij = self._distance_to_radial_basis(d_ij)

        idx_i, idx_j = atom_index12[0], atom_index12[1]
        for interaction in self.interactions:
            v = interaction(x, f_ij, idx_i, idx_j, rcut_ij)
            x = x + v

        return x
