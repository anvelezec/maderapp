import torch

model1 = torch.jit.load("mobile_models/efficientNet.pt")
model2 = torch.jit.load("mobile_models/trace_efficientNet.pt")


x = torch.rand((1, 3, 224, 224))

from datetime import datetime
now = datetime.now()
y = model1(x)
print(datetime.now() - now)

now = datetime.now()
y = model2(x)
print(datetime.now() - now)