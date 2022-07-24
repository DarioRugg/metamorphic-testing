from math import log2

import numpy as np
import torch
from torch import nn

np.random.seed(123)

x = torch.tensor(np.random.rand(3,2))
x_hat = torch.tensor(np.random.rand(3,2))

print(x_hat.shape[0])
loss = []

for i in range(x_hat.shape[0]):
    loss.append(-torch.sum(x[i, :]*torch.log2(x_hat[i, :])))

# loss = cross_entropy(x, x_hat)
loss *= 50

print(x)
print(x_hat)
print(loss)

