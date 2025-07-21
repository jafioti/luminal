import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Define MLP class
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(dims[-1], output_dim))  # Output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# 2. Create the MLP model
input_dim = 5       # Input features
hidden_dims = [10]
output_dim = 5       # Output classes
mlp = MLP(input_dim, hidden_dims, output_dim)

# 3. Generate random input data
batch_size = 4
x = torch.randn(batch_size, input_dim)

# 4. Inference using random weights
with torch.no_grad():
    output = mlp(x)

print("Input:")
print(x)
print("\nOutput:")
print(output)
