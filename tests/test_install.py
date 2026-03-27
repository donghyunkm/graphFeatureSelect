#!/usr/bin/env python3
"""Test script to verify PyTorch and PyTorch Geometric installation with GPU support."""

import torch
import torch_geometric

print("=" * 80)
print("PyTorch Installation Test")
print("=" * 80)

# Check PyTorch version
print(f"\n1. PyTorch Version: {torch.__version__}")

# Check CUDA availability
print(f"2. CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"3. CUDA Version: {torch.version.cuda}")
    print(f"4. Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"   - Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"   - Compute Capability: {props.major}.{props.minor}")

    # Test GPU tensor operations
    print("\n5. Testing GPU tensor operations...")
    device = torch.device("cuda:0")
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = torch.matmul(x, y)
    print(f"   ✓ Matrix multiplication on GPU successful")
    print(f"   Result tensor shape: {z.shape}, device: {z.device}")
else:
    print("   ⚠ CUDA is not available. Using CPU only.")

print("\n" + "=" * 80)
print("PyTorch Geometric Installation Test")
print("=" * 80)

# Check PyG version
print(f"\n6. PyTorch Geometric Version: {torch_geometric.__version__}")

# Test PyG operations
print("\n7. Testing PyG operations...")
from torch_geometric.data import Data

# Create a simple graph
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.randn(3, 16)  # 3 nodes with 16 features each

if torch.cuda.is_available():
    edge_index = edge_index.to("cuda:0")
    x = x.to("cuda:0")

data = Data(x=x, edge_index=edge_index)
print(f"   ✓ PyG Data object created successfully")
print(f"   Number of nodes: {data.num_nodes}")
print(f"   Number of edges: {data.num_edges}")
print(f"   Node features shape: {data.x.shape}")
if torch.cuda.is_available():
    print(f"   Device: {data.x.device}")

# Test a simple GNN layer
print("\n8. Testing GNN layer...")
from torch_geometric.nn import GCNConv

conv = GCNConv(16, 32)
if torch.cuda.is_available():
    conv = conv.to("cuda:0")

out = conv(data.x, data.edge_index)
print(f"   ✓ GCNConv layer forward pass successful")
print(f"   Output shape: {out.shape}")
if torch.cuda.is_available():
    print(f"   Output device: {out.device}")

print("\n" + "=" * 80)
print("✓ All tests passed successfully!")
print("=" * 80)
