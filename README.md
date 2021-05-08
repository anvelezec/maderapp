# maderapp
Macroscopic tree image recognition model training and delivering to production

Nowadays there are plenty of models implementations really to use them, this does not mean we should stop leaning the basics of ML-DL modelation. These implementations save users time allowing prototyping cycle iterations faster and also gives extra time to pay attention to data cleaning and its quality, ethics and delivering models to production faster. As Andrew Ng says, switching [from model centric to data centric approach](https://www.youtube.com/watch?v=06-AZXmwHjo) machine learning engineers could leverage model results significatively.

# Data collection
We collect images from the Peru Amazonia Servicio Nacional Forestal y de Fauna Silvestre (SERFOR) control points central Peru Amazonia. For the images collection we followed [Filho P et al., (2014)](https://web.inf.ufpr.br/vri/databases/forest-species-database-macroscopic/) guide. Instead of a camera we used a smartphone with > 15 MP camera.


# Creating a data loader. 
To feed the images to the model it's important to create a generator so we can pass batches of information without consuming the total amount of a RAM (GPU) machine. Creating a generator in pytorch is simple and intuitive, just need to create a class which inherits from torch.utils.data.Dataset and define three methods as seen below.

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
