from maderapp.data.data import MaderappDataset
from maderapp.data.data_inference import MaderappDatasetInference
from maderapp.data.data_patches import MaderappPatchesDataset

datasets = {
    "basic": MaderappDataset,
    "patches": MaderappPatchesDataset,
    "inference": MaderappDatasetInference
}