import torch
from torch.nn import Module
from torch import Tensor


SYMETRIES = [
    lambda t: torch.rot90(t, k=0, dims=(-2, -1)),
    lambda t: torch.rot90(t, k=1, dims=(-2, -1)),
    lambda t: torch.rot90(t, k=2, dims=(-2, -1)),
    lambda t: torch.rot90(t, k=3, dims=(-2, -1)),
    lambda t: torch.flip(t, dims=(-1,)),
    lambda t: torch.flip(t, dims=(-2,)),
    lambda t: torch.flip(torch.rot90(t, k=-1, dims=(-2, -1)), dims=(-1,)),
    lambda t: torch.flip(torch.rot90(t, k=1, dims=(-2, -1)), dims=(-1,)),
]

I_SYMETRIES = [
    lambda t: torch.rot90(t, k=0, dims=(-2, -1)),
    lambda t: torch.rot90(t, k=-1, dims=(-2, -1)),
    lambda t: torch.rot90(t, k=-2, dims=(-2, -1)),
    lambda t: torch.rot90(t, k=-3, dims=(-2, -1)),
    lambda t: torch.flip(t, dims=(-1,)),
    lambda t: torch.flip(t, dims=(-2,)),
    lambda t: torch.flip(torch.rot90(t, k=-1, dims=(-2, -1)), dims=(-1,)),
    lambda t: torch.flip(torch.rot90(t, k=1, dims=(-2, -1)), dims=(-1,)),
]


class Symetry_Inveriant_Conv2D(Module):
    def __init__(self, conv_layer: Module, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv_layer = conv_layer

    def forward(self, x: Tensor) -> Tensor:
        # [n, C, H, W]
        input_shape = x.shape

        # Compute all symetries
        # [n, 8, C, H, W]
        x = torch.stack([f(x) for f in SYMETRIES], -4)

        # Linearize along the batch dimension
        # [n*8, C, H, W]
        x = torch.reshape(x, (-1, *x.shape[-3:]))

        # Apply the conv_layer
        # [n*8, c, h, w]
        x = self.conv_layer(x)

        # Nest back to original first dimensions shape
        # [n, 8, c, h, w]
        nested_shape = (*input_shape[:-3], 8, *x.shape[-3:])
        x = torch.reshape(x, nested_shape)

        # Inverse the symetries
        # [n, 8, c, h, w]
        x = torch.stack(
            [f(t[:, 0]) for t, f in zip(torch.split(x, 1, dim=-4), I_SYMETRIES)], -4
        )

        # Agregate
        # [n, c, h, w]
        x = torch.amax(x, dim=-4)

        return x
