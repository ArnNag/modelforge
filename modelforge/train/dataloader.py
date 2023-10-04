from modelforge.dataset.dataset import TorchDataset
import torch
from torch.utils.data import DataLoader, Sampler
from typing import Union, Iterable
from collections.abc import Iterator



def get_batch_loader(
        dataset: TorchDataset, 
        sampler: Union[Sampler, Iterable], 
        cutoff: float,
        max_batch_edges: int) -> DataLoader:
    """Return a DataLoader object for the given dataset with molecules in the order specified by sampler. First, computes the number of edges within a Euclidean cutoff and groups the molecules into batches such that the total number of edges in the batch is less than max_batch_edges. Each iteration of the returned DataLoader yields a dictionary with the concatenated Z, R, and E tensors for the batch, as well a list of the indices of the molecules in the dataset and a list of the number of atoms in each molecule. """
    one_molecule_loader = DataLoader(dataset, sampler=sampler)
    batch_sampler = BatchSampler(one_molecule_loader, cutoff, max_batch_edges)

    def collate_molecules(molecule_list):
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
    

class BatchSampler(Iterator):
    def __init__(self,
            one_molecule_loader: DataLoader,
            cutoff: float,
            max_batch_edges: int):
        """Return a BatchSampler object for a DataLoader that yields one molecule at a time. The BatchSampler yields a list of indices of molecules in the DataLoader such that the total number of edges in the batch is less than max_batch_edges."""
        super().__init__()
        self.one_molecule_iter = iter(one_molecule_loader)
        self.cutoff = cutoff
        self.max_batch_edges = max_batch_edges

    def get_num_edges_within_cutoff(self, molecule):
        """Return the number of edges within the cutoff for the given molecule."""
        #TODO: reimplement modelforge.poential.utils.neighbor_pairs_nopbc. Should not have mask in function signature.
        return len(molecule['X']) ** 2


    def __next__(self):
        added_idxs = []
        added_edges = 0
        while True:
            molecule = next(self.one_molecule_iter)
            num_molecule_edges = self.get_num_edges_within_cutoff(molecule)
            added_idxs.append(molecule['idx'])
            if added_edges + num_molecule_edges > self.max_batch_edges:
                yield added_idxs
                added_idxs = [molecule['idx']]
                added_edges = num_molecule_edges
            else:
                added_idxs.append(molecule['idx'])
                added_edges += num_molecule_edges

    def __len__(self):
        return 0 #TODO: implement
