from typing import Optional, Type

import pytest

from modelforge.potential.models import BaseNNP
from modelforge.potential.schnet import SchNET

from .helper_functions import (
    MODELS_TO_TEST,
    DATASETS,
    initialize_dataset,
    setup_simple_model,
)


@pytest.mark.parametrize("model_class", MODELS_TO_TEST)
@pytest.mark.parametrize("dataset", DATASETS)
def test_train_with_lightning(dataset: Type[BaseNNP], model_class: Type[BaseNNP]):
    """
    Test the forward pass for a given model and dataset.

    Parameters
    ----------
    dataset : Type[BaseNNP]
        The dataset class to be used in the test.
    model_class : Type[BaseNNP]
        The model class to be used in the test.
    """

    from lightning import Trainer
    import torch

    model: Optional[BaseNNP] = setup_simple_model(model_class, lightning=True)
    if model is None:
        pytest.fail("Failed to set up the model.")

    # Initialize dataset and data loader
    dataset = initialize_dataset(dataset)
    if dataset is None:
        pytest.fail("Failed to initialize the dataset.")
    # Initialize PyTorch Lightning Trainer
    trainer = Trainer(max_epochs=2)

    # Move model to the appropriate dtype and device
    model = model.to(torch.float32)
    # Run training loop and validate
    trainer.fit(model, dataset.train_dataloader(), dataset.val_dataloader())


def test_pt_lightning():
    # This is an example script that trains the PaiNN model on the .
    from lightning import Trainer
    import torch
    from modelforge.potential.painn import LighningPaiNN

    from modelforge.potential import CosineCutoff, GaussianRBF
    from modelforge.potential.utils import SlicedEmbedding

    from openmm import unit

    max_atomic_number = 100
    nr_atom_basis = 128
    nr_rbf = 20
    nr_interaction_blocks = 4

    cutoff = unit.Quantity(5, unit.angstrom)
    embedding = SlicedEmbedding(max_atomic_number, nr_atom_basis, sliced_dim=0)
    assert embedding.embedding_dim == nr_atom_basis
    rbf = GaussianRBF(n_rbf=nr_rbf, cutoff=cutoff)

    cutoff = CosineCutoff(cutoff=cutoff)

    model = LighningPaiNN(
        embedding=embedding,
        nr_interaction_blocks=nr_interaction_blocks,
        radial_basis=rbf,
        cutoff=cutoff,
    )

    from modelforge.dataset.qm9 import QM9Dataset
    from modelforge.dataset.dataset import TorchDataModule

    data = QM9Dataset(for_unit_testing=True)
    dataset = TorchDataModule(data, batch_size=64)
    dataset.prepare_data()
    dataset.setup()
    from lightning.pytorch.callbacks.early_stopping import EarlyStopping

    trainer = Trainer(
        max_epochs=500,
        num_nodes=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=3, min_delta=0.001)
        ],
    )

    # Move model to the appropriate dtype and device
    model = model.to(torch.float32)
    # Run training loop and validate
    trainer.fit(model, dataset.train_dataloader(), dataset.val_dataloader())
