from modelforge.tests.helper_functions import setup_potential_for_test
import pytest


def test_init():
    """Test initialization of the TensorNet model."""

    # load default parameters
    # read default parameters
    model = setup_potential_for_test(
        use="inference",
        potential_seed=42,
        potential_name="tensornet",
        simulation_environmen="JAX",
    )
    assert model is not None, "TensorNet model should be initialized."


@pytest.mark.parametrize("simulation_environment", ["PyTorch", "JAX"])
def test_forward_with_inference_model(
    simulation_environment, single_batch_with_batchsize
):

    batch = single_batch_with_batchsize(batch_size=32, dataset_name="QM9")

    # load default parameters
    model = setup_potential_for_test(
        use="inference",
        potential_seed=42,
        potential_name="tensornet",
        simulation_environmen=simulation_environment,
        use_training_mode_neighborlist=True,
    )

    model(batch.nnp_input_tuple)


def test_input():
    import torch
    from loguru import logger as log
    from modelforge.tests.precalculated_values import (
        prepare_values_for_test_tensornet_input,
    )

    # setup model
    model = setup_potential_for_test(
        use="inference",
        potential_seed=42,
        potential_name="tensornet",
        simulation_environmen="PyTorch",
        use_training_mode_neighborlist=True,
    )

    from importlib import resources
    from modelforge.tests import data

    # load reference data
    reference_data = resources.files(data) / "tensornet_input.pt"
    reference_batch = resources.files(data) / "mf_input.pkl"
    import pickle

    mf_input = pickle.load(open(reference_batch, "rb"))

    # calculate pairlist
    pairlist_output = model.neighborlist.forward(mf_input)

    # compare to torchmd-net pairlist
    if reference_data:
        log.warning('Using reference data for "test_input"')
        edge_index, edge_weight, edge_vec = torch.load(reference_data)
    else:
        log.warning('Calculating reference data from  "test_input"')
        edge_index, edge_weight, edge_vec = prepare_values_for_test_tensornet_input(
            mf_input,
            seed=0,
        )

    # reshape and compare
    pair_indices = pairlist_output.pair_indices.t()
    edge_index = edge_index.t()
    for _, pair_index in enumerate(pair_indices):
        idx = ((edge_index == pair_index).sum(axis=1) == 2).nonzero()[0][
            0
        ]  # select [True, True]
        assert torch.allclose(pairlist_output.d_ij[_][0], edge_weight[idx])
        assert torch.allclose(pairlist_output.r_ij[_], -edge_vec[idx])


def test_compare_radial_symmetry_features():
    # Compare the TensorNet radial symmetry function to the output of the
    # modelforge radial symmetry function TODO: only 'expnorm' from TensorNet
    # implemented
    import torch
    from openff.units import unit

    from modelforge.potential.utils import CosineAttenuationFunction
    from modelforge.potential.utils import TensorNetRadialBasisFunction
    from modelforge.tests.precalculated_values import (
        prepare_values_for_test_tensornet_compare_radial_symmetry_features,
    )

    seed = 0
    torch.manual_seed(seed)
    from importlib import resources
    from modelforge.tests import data

    reference_data = resources.files(data) / "tensornet_radial_symmetry_features.pt"

    # generate a random list of distances, all < 5
    d_ij = unit.Quantity(
        torch.tensor([[2.4813], [3.8411], [0.4424], [0.6602], [1.5371]]), unit.angstrom
    )

    # TensorNet constants
    maximum_interaction_radius = unit.Quantity(5.1, unit.angstrom)
    minimum_interaction_radius = unit.Quantity(0.0, unit.angstrom)
    number_of_per_atom_features = 8
    alpha = (
        (maximum_interaction_radius - minimum_interaction_radius)
        / unit.Quantity(5.0, unit.angstrom)
    ).m / 10

    rsf = TensorNetRadialBasisFunction(
        number_of_radial_basis_functions=number_of_per_atom_features,
        max_distance=maximum_interaction_radius.to(unit.nanometer).m,
        min_distance=minimum_interaction_radius.to(unit.nanometer).m,
        alpha=alpha,
    )
    mf_r = rsf(d_ij.to(unit.nanometer).m)  # torch.Size([5, 8])
    cutoff_module = CosineAttenuationFunction(
        maximum_interaction_radius.to(unit.nanometer).m
    )

    rcut_ij = cutoff_module(d_ij.to(unit.nanometer).m)  # torch.Size([5, 1])
    mf_r = (mf_r * rcut_ij).unsqueeze(1)

    from importlib import resources
    from modelforge.tests import data

    reference_data = resources.files(data) / "tensornet_radial_symmetry_features.pt"

    if reference_data:
        tn_r = torch.load(reference_data)
    else:
        tn_r = prepare_values_for_test_tensornet_compare_radial_symmetry_features(
            d_ij,
            minimum_interaction_radius,
            maximum_interaction_radius,
            number_of_per_atom_features,
            trainable=False,
            seed=seed,
        )

    assert torch.allclose(mf_r, tn_r, atol=1e-4)


def test_representation(single_batch_with_batchsize):
    import torch
    from openff.units import unit
    from torch import nn

    from modelforge.potential.tensornet import TensorNetRepresentation

    from importlib import resources
    from modelforge.tests import data

    reference_data = resources.files(data) / "tensornet_representation.pt"

    number_of_per_atom_features = 8
    num_rbf = 16
    act_class = nn.SiLU
    cutoff_lower = 0.0
    cutoff_upper = 5.1
    trainable_rbf = False
    maximum_atomic_number = 128

    import pickle

    reference_batch = resources.files(data) / "mf_input.pkl"
    nnp_input = pickle.load(open(reference_batch, "rb"))
    # -------------------------------#
    # -------------------------------#
    # Test that we can add the reference energy correctly
    # get methane input
    model = setup_potential_for_test(
        use="inference",
        potential_seed=42,
        potential_name="tensornet",
        simulation_environmen="PyTorch",
        use_training_mode_neighborlist=True,
    )
    pairlist_output = model.neighborlist.forward(nnp_input)

    ################ modelforge TensorNet ################
    tensornet_representation_module = TensorNetRepresentation(
        number_of_per_atom_features,
        num_rbf,
        act_class(),
        unit.Quantity(cutoff_upper, unit.angstrom).to(unit.nanometer).m,
        unit.Quantity(cutoff_lower, unit.angstrom).to(unit.nanometer).m,
        trainable_rbf,
        maximum_atomic_number,
    )
    mf_X, _ = tensornet_representation_module(nnp_input, pairlist_output)
    ################ modelforge TensorNet ################

    ################ torchmd-net TensorNet ################
    if reference_data:
        tn_X = torch.load(reference_data)
    else:
        tn_X = prepare_values_for_test_tensornet_representation(
            nnp_input,
            number_of_per_atom_features,
            num_rbf,
            act_class,
            cutoff_lower,
            cutoff_upper,
            trainable_rbf,
            highest_atomic_number,
            seed=0,
        )
    ################ torchmd-net TensorNet ################

    assert mf_X.shape == tn_X.shape
    assert torch.allclose(mf_X, tn_X)


def test_interaction():
    import torch
    from openff.units import unit
    from torch import nn

    from modelforge.dataset.dataset import DataModule
    from modelforge.dataset.utils import FirstComeFirstServeSplittingStrategy
    from modelforge.potential.tensornet import TensorNet
    from modelforge.potential.tensornet import TensorNetInteraction
    from modelforge.tests.precalculated_values import (
        prepare_values_for_test_tensornet_interaction,
    )
    from modelforge.tests.test_models import load_configs_into_pydantic_models

    seed = 0
    torch.manual_seed(seed)
    from importlib import resources

    from modelforge.tests import data

    reference_data = resources.files(data) / "tensornet_interaction.pt"

    # reference_data = "modelforge/tests/data/tensornet_interaction.pt"
    # reference_data = None

    number_of_per_atom_features = 8
    num_rbf = 16
    act_class = nn.SiLU
    cutoff_lower = 0.0
    cutoff_upper = 5.1

    # Set up a dataset
    # prepare reference value
    dataset = DataModule(
        name="QM9",
        batch_size=1,
        version_select="nc_1000_v0",
        splitting_strategy=FirstComeFirstServeSplittingStrategy(),
        remove_self_energies=True,
        regression_ase=False,
    )
    dataset.prepare_data()
    dataset.setup()
    # -------------------------------#
    # -------------------------------#
    # Test that we can add the reference energy correctly
    # get methane input
    mf_input = next(iter(dataset.train_dataloader())).nnp_input
    # modelforge TensorNet
    torch.manual_seed(seed)
    # load default parameters
    config = load_configs_into_pydantic_models(f"tensornet", "qm9")
    # initialize model
    tensornet = TensorNet(
        **config["potential"].model_dump()["core_parameter"],
        postprocessing_parameter=config["potential"].model_dump()[
            "postprocessing_parameter"
        ],
    )
    tensornet.compute_interacting_pairs._input_checks(mf_input)
    pairlist_output = tensornet.compute_interacting_pairs.forward(mf_input)

    ################ modelforge TensorNet ################
    tensornet_representation_module = tensornet.core_module.representation_module
    nnp_input = tensornet.core_module._model_specific_input_preparation(
        mf_input, pairlist_output
    )
    X, _ = tensornet_representation_module(nnp_input)

    radial_feature_vector = tensornet_representation_module.radial_symmetry_function(
        nnp_input.d_ij
    )
    rcut_ij = tensornet_representation_module.cutoff_module(nnp_input.d_ij)
    radial_feature_vector = (radial_feature_vector * rcut_ij).unsqueeze(1)

    atomic_charges = torch.zeros_like(nnp_input.atomic_numbers)

    # interaction
    torch.manual_seed(seed)
    interaction_module = TensorNetInteraction(
        number_of_per_atom_features,
        num_rbf,
        act_class(),
        cutoff_upper * unit.angstrom,
        "O(3)",
    )
    mf_X = interaction_module(
        X,
        nnp_input.pair_indices,
        nnp_input.d_ij.squeeze(-1),
        radial_feature_vector.squeeze(1),
        atomic_charges,
    )
    ################ modelforge TensorNet ################

    ################ TensorNet ################
    if reference_data:
        tn_X = torch.load(reference_data)
    else:
        tn_X = prepare_values_for_test_tensornet_interaction(
            X,
            nnp_input,
            radial_feature_vector,
            atomic_charges,
            number_of_per_atom_features,
            num_rbf,
            act_class,
            cutoff_lower,
            cutoff_upper,
            seed,
        )
    ################ TensorNet ################

    assert mf_X.shape == tn_X.shape
    assert torch.allclose(mf_X, tn_X)


if __name__ == "__main__":
    import torch

    torch.manual_seed(0)

    # test_forward()

    # test_input()

    # test_compare_radial_symmetry_features()

    # test_representation()

    test_interaction()
