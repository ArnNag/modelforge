from loguru import logger

from modelforge.potential.sake import Sake

from .helper_functinos import generate_methane_input
import torch


def test_Sake_init():
    sake = Sake(128, 6, 2)
    assert sake is not None


def test_sake_forward():
    model = Sake(128, 3)
    inputs = {
        "Z": torch.tensor([[1, 2], [2, 3]]),
        "R": torch.tensor(
            [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]]
        ),
    }
    energy = model(inputs)
    assert energy.shape == (
        2,
        1,
    )  # Assuming energy is calculated per sample in the batch


def test_calculate_energies_and_forces():
    # this test will be adopted as soon as we have a
    # trained model. Here we want to test the
    # energy and force calculatino on Methane

    sake = Sake(n_atom_basis=31, n_interactions=3, n_filters=127, n_rbf=63)
    methane_inputs = generate_methane_input()
    result = sake(methane_inputs)
    forces = -torch.autograd.grad(
        result, methane_inputs["r"], create_graph=true, retain_graph=true
    )[0]

    assert result.shape == (1, 1)  #  only one molecule
    assert forces.shape == (1, 5, 3)  #  only one molecule
    logger.debug(result)
