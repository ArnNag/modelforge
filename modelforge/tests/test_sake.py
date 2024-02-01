from typing import Dict

import pytest

import torch
import numpy as np

from modelforge.potential.sake import SAKE

from .helper_functions import generate_methane_input, setup_simple_model, SIMPLIFIED_INPUT_DATA


@pytest.mark.parametrize("lightning", [False])
def test_SAKE_init(lightning):
    """Test initialization of the SAKE neural network potential."""

    sake = setup_simple_model(SAKE, lightning=lightning)
    assert sake is not None, "PaiNN model should be initialized."

# def test_Sake_init():
#     """Test initialization of the Sake model."""
#     sake = Sake(128, 6, 2)
#     assert sake is not None, "Sake model should be initialized."


@pytest.mark.parametrize("lightning", [False])
@pytest.mark.parametrize("input_data", SIMPLIFIED_INPUT_DATA)
def test_Sake_forward(lightning, input_data):
    """
    Test the forward pass of the Sake model.
    """
    sake = setup_simple_model(SAKE, lightning=lightning)
    energy = sake(input_data)
    assert energy.shape == (
        2,
        1,
    )  # Assuming energy is calculated per conformer in the batch


def test_calculate_energies_and_forces():
    """
    Test the calculation of energies and forces for a molecule.
    This test will be adapted once we have a trained model.
    """

    sake = SAKE(128, 6, 64)
    methane_inputs = generate_methane_input()
    print(methane_inputs)
    result = sake(methane_inputs)
    forces = -torch.autograd.grad(
        result, methane_inputs["R"], create_graph=True, retain_graph=True
    )[0]

    assert result.shape == (1, 1)  #  only one molecule
    assert forces.shape == (1, 5, 3)  #  only one molecule


def get_input_for_interaction_block(
    nr_atom_basis: int, nr_embeddings: int
) -> Dict[str, torch.Tensor]:
    """
    Prepare inputs for testing the SchNet interaction block.

    Parameters
    ----------
    nr_atom_basis : int
        Number of atom basis.
    nr_embeddings : int
        Number of embeddings.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary containing tensors for the interaction block test.
    """

    import torch.nn as nn

    from modelforge.dataset.qm9 import QM9Dataset
    from modelforge.potential.utils import GaussianRBF, _distance_to_radial_basis

    from .helper_functions import prepare_pairlist_for_single_batch, return_single_batch

    embedding = nn.Embedding(nr_embeddings, nr_atom_basis, padding_idx=0)
    batch = return_single_batch(QM9Dataset, "fit")
    pairlist = prepare_pairlist_for_single_batch(batch)
    radial_basis = GaussianRBF(n_rbf=20, cutoff=5.0)

    atom_index12 = pairlist["atom_index12"]
    d_ij = pairlist["d_ij"]
    f_ij, rcut_ij = _distance_to_radial_basis(d_ij, radial_basis)
    return {
        "x": embedding(batch["Z"]),
        "f_ij": f_ij,
        "idx_i": atom_index12[0],
        "idx_j": atom_index12[1],
        "rcut_ij": rcut_ij,
    }


def test_sake_interaction_layer():
    """
    Test the Sake interaction layer.
    """
    from modelforge.potential.sake import SakeInteractionBlock

    nr_atom_basis = 128
    nr_embeddings = 100
    r = get_input_for_interaction_block(nr_atom_basis, nr_embeddings)
    assert r["x"].shape == (
        64,
        17,
        nr_atom_basis,
    ), "Input shape mismatch for x tensor."
    interaction = SakeInteractionBlock(nr_atom_basis, 4)
    v = interaction(r["x"], r["f_ij"], r["idx_i"], r["idx_j"], r["rcut_ij"])
    assert v.shape == (
        64,
        17,
        nr_atom_basis,
    ), "Output shape mismatch for v tensor."


# def test_schnet_reimplementation_against_original_implementation():
#    import numpy as np
#    np.load('tests/qm9tut/split.npz')['train_idx']
