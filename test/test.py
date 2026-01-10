import torch
from torch import nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.Transformer(
            d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6
        )

    def forward(self, x):
        return self.n1(xtgt=x, xsrc=x)

if __name__ == "__main__":
    model = SimpleModel()
    x = torch.randn(5, 10)
    output = model(x)
    print(output)