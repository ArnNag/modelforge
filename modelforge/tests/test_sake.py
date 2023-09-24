from loguru import logger

from modelforge.potential.sake import Sake

from .helper_functinos import methane_input
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
    energy = model.calculate_energy(inputs)
    assert energy.shape == (
        2,
        1,
    )  # Assuming energy is calculated per sample in the batch


def test_calculate_energies_and_forces():
    # this test will be adopted as soon as we have a
    # trained model. Here we want to test the
    # energy and force calculatino on Methane

    sake = Sake(128, 6, 64)
    methane_inputs = methane_input()
    result = sake.calculate_energy(methane_inputs)
    logger.debug(result)
    assert result.shape[0] == 1  # Assuming only one molecule
