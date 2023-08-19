# Goal is to make my autograd library this simple.
import torch

from torch.optim import SGD 
from torch.nn.functional import mse_loss
from torch.nn.parameter import Parameter 

# make linear data
true_weight = 2.0 
true_bias = -3.0
x = torch.linspace(-10,10, 100)
y = true_weight * x + true_bias

# shuffle
perm = torch.randperm(100)
x = x[perm]
y = y[perm]

# training loop
w = Parameter(torch.tensor([0.0]))
b = Parameter(torch.tensor([0.0]))
optimizer = SGD([w, b], lr=1e-2)

epochs = 10
for epoch_i in range(epochs):
    for i in range(100): 
        optimizer.zero_grad()
        out = w * x[i] + b 
        loss = mse_loss(out, y[i])
        loss.backward() 
        optimizer.step()
    print('epoch {} loss {}'.format(epoch_i, loss.item()))

print('w {} b {}'.format(w.item(), b.item()))
