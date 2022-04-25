from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader

from maderapp.data import MaderappDataset
from maderapp.data_inference import MaderappDatasetInference

metadata = pd.read_csv("metadata.csv")
BATCH_SIZE = 4


def batch_size_test():
    ds = MaderappDataset("training-img", metadata)
    dl = DataLoader(ds, BATCH_SIZE)
    dataset = next(iter(dl))
    assert len(dataset) == 2
    assert len(dataset[0]) == BATCH_SIZE


def inference_dataloader_test():
    metadata = [str(path) for path in list(Path("validation").glob("*.jpg"))]
    ds = MaderappDatasetInference(metadata)
    dl = DataLoader(ds, BATCH_SIZE)
    dataset = next(iter(dl))
    assert len(dataset) == 2
    assert len(dataset[0]) == BATCH_SIZE