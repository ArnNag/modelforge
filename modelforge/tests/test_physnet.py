from typing import Optional


from modelforge.tests.helper_functions import setup_potential_for_test


def test_init():

    model = setup_potential_for_test("physnet", "training")
    assert model is not None, "PhysNet model should be initialized."


def test_forward(single_batch_with_batchsize):
    import torch

    model = setup_potential_for_test("physnet", "training")
    print(model)
    batch = batch = single_batch_with_batchsize(batch_size=64, dataset_name="QM9")

    yhat = model(batch.nnp_input.to(dtype=torch.float32))


def test_compare_representation():
    # This test compares the RBF calculation of the original PhysNet
    # implemntation against the SAKE/PhysNet implementation in modelforge
    # # NOTE: input in PhysNet is expected in angstrom, in contrast to modelforge which expects input in nanomter

    import numpy as np
    import torch
    from openff.units import unit

    # set up test parameters
    number_of_radial_basis_functions = K = 20
    _max_distance = unit.Quantity(5, unit.angstrom)
    #############################
    # RBF comparision
    #############################
    # Initialize the rbf class
    from modelforge.potential.utils import PhysNetRadialBasisFunction

    rbf = PhysNetRadialBasisFunction(
        number_of_radial_basis_functions,
        max_distance=_max_distance.to(unit.nanometer).m,
    )

    # compare the rbf output
    #################
    # PhysNet implementation
    from .precalculated_values import provide_reference_for_test_physnet_test_rbf

    reference_rbf = provide_reference_for_test_physnet_test_rbf()
    D = np.array([[1.0394776], [3.375541]], dtype=np.float32) / 10

    calculated_rbf = rbf(torch.tensor(D))
    assert np.allclose(np.flip(reference_rbf.squeeze(), axis=1), calculated_rbf.numpy())
