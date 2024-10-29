import torch
from torch import nn
from thop import profile

# Define a simple PyTorch module to perform the matrix multiplication
class MatMulModule(nn.Module):
    def forward(self, a, b):
        return a @ b.transpose(-1, -2)

# Create input data
a = torch.randn(10, 256, 512)
b = torch.randn(10, 256, 512)

# Instantiate the module
model = MatMulModule()

# Use thop to calculate the FLOPs for this operation
inputs = (a, b)
total_flops, total_params = profile(model, inputs=inputs, verbose=False)

print(f"Total FLOPs: {total_flops} FLOPs")
print(f"Total Params: {total_params} parameters")