# training/01-create-workspace.py
from azureml.core import Workspace

ws = Workspace.create(name='maderapp_ml', # provide a name for your workspace
                      subscription_id='6a2ed9cc-77d9-4e51-8f52-cfc60b06428e', # provide your subscription ID
                      resource_group='maderapp', # provide a resource group name
                      create_resource_group=True,
                      location='eastus2') # For example: 'westeurope' or 'eastus2' or 'westus2' or 'southeastasia'.

# write out the workspace details to a configuration file: .azureml/config.json
ws.write_config(path='.azureml')