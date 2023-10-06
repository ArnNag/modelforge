from modelforge.dataset.dataset import TorchDataset
import torch
from torch.utils.data import DataLoader, Sampler
from typing import Union, Iterable, Optional, List, Dict
from collections.abc import Generator



def get_batch_loader(
        dataset: TorchDataset, 
        cutoff: float,
        max_batch_edges: int,
        sampler: Optional[Union[Sampler, Iterable]] = None
    ) -> DataLoader:
    """Return a DataLoader object for the given dataset with molecules in the order specified by sampler. First, computes the number of edges within a Euclidean cutoff and groups the molecules into batches such that the total number of edges in the batch is less than max_batch_edges. Each iteration of the returned DataLoader yields a dictionary with the concatenated Z, R, and E tensors for the batch, as well a list of the indices of the molecules in the dataset and a list of the number of atoms in each molecule. """
    one_molecule_loader = DataLoader(dataset, sampler=sampler)
    batch_sampler = get_batch_sampler(one_molecule_loader, cutoff, max_batch_edges)

    def collate_molecules(molecule_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        #TODO: violates abstraction barrier. Define a class for TorchDataset.__get__item__() output with a collate method instead of using a dictionary?
        """Concatenate the Z, R, and E tensors from a list of molecules into a single tensor each, and return a new dictionary with the concatenated tensors."""
        Z_list = []
        R_list = []
        E_list = []
        idx_list = []
        num_atoms_per_molecule = []
        for molecule in molecule_list:
            Z_list.append(molecule['Z'])
            R_list.append(molecule['R'])
            E_list.append(molecule['E'])
            idx_list.append(molecule['idx'])
            num_atoms_per_molecule.append(len(molecule['Z']))
        Z_cat = torch.cat(Z_list)
        R_cat = torch.cat(R_list)
        E_cat = torch.cat(E_list)
        return {"Z": Z_cat, "R": R_cat, "E": E_cat, "idx": idx_list, "n_atoms": num_atoms_per_molecule}

    return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_molecules)
    

def get_batch_sampler(
    one_molecule_loader: DataLoader,
    cutoff: float,
    max_batch_edges: int
    ) -> Generator[List[int], None, None]:
    """Return a generator for a DataLoader that yields one molecule at a time. The generator yields a list of indices of molecules in the DataLoader such that the total number of edges in the batch is less than max_batch_edges."""

    def get_num_edges_within_cutoff(molecule, cutoff):
        """Return the number of edges within the cutoff for the given molecule."""
        #TODO: reimplement modelforge.potential.utils.neighbor_pairs_nopbc. Should not have mask in function signature.
        return len(molecule['Z']) ** 2


    added_idxs = []
    added_edges = 0
    for molecule in one_molecule_loader:
        num_molecule_edges = get_num_edges_within_cutoff(molecule, cutoff)
        if added_edges + num_molecule_edges > max_batch_edges:
            yield added_idxs
            added_idxs = []
            added_edges = 0
        added_idxs.append(molecule['idx'].item())
        added_edges += num_molecule_edges

