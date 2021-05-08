# maderapp
Macroscopic tree recognition model training


# Creating a data loader. 
Creating a generator in pytorch is really simple, just need to create a class which inherits from torch.utils.data.Dataset and define three methods as seen below.

```python
from torch.utils.data import Dataset

class DataLoader(Dataset)

    def __init__(self, data_path):
        # Initializa variables such as data_path
        pass
    
    def __len__(self):
        # Return number of items in the dataset
        pass
    
    def __getitem__(self, idx):
        # Reads features and labels
        # Applys transformations
        # Return features, labels from idx
        pass

```


# Setting up Azure environment
The initial azure training environment was setup by folllowing this steps: 
1. https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-sdk-setup-local
2. https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-hello-world
3. https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-sdk-train
4. https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-bring-data
