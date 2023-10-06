from modelforge import dataset
from modelforge.train.dataloader import get_batch_loader

def test_qm9_batch_loader():
    qm9_data = dataset.QM9Dataset() 
    factory = dataset.dataset.DatasetFactory() 
    torch_dataset = factory.create_dataset(qm9_data) 
    batch_loader = get_batch_loader(torch_dataset, 10, 50)

    total_molecules = 0
    num_batches = 0
    for batch in batch_loader:
        num_batches += 1
        total_molecules += len(batch['n_atoms'])

    print("num_batches:", num_batches)
    print("total_molecules:", total_molecules)
    print("len(torch_dataset):", len(torch_dataset))
    assert total_molecules == len(torch_dataset)


test_qm9_batch_loader()

