import torch

from maderapp.model.timber_clasification_patches import (
    ActPool,
    ResidualBlock,
    TimberPatchesNet,
)


def test_residual_block():
    x = torch.randn(2, 64, 64, 64)
    block = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3)
    y = block(x)
    assert y.shape == x.shape


def test_act_pool():
    input_tensor = torch.randn(2, 64, 64, 64)
    act_bloc = ActPool()
    output_tensor = act_bloc(input_tensor)
    assert input_tensor.shape[-1] / 2 == output_tensor.shape[-1]


def test_patch_model():
    input_tensor = torch.randn(2, 3, 64, 64)
    num_classes = 20
    tp = TimberPatchesNet(num_classes=num_classes)
    output_tensor = tp(input_tensor)
    assert output_tensor.shape[-1] == num_classes
