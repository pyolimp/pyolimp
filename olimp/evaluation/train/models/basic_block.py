from torch import nn, Tensor


class BasicMLPBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dims: tuple[int, ...] = (32, 16),
        out_features: int = 1,
        dropout: float = 0.1,
        apply_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim: int = in_features

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, out_features))

        self.net: nn.Sequential = nn.Sequential(*layers)
        self.apply_sigmoid = apply_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        if self.apply_sigmoid:
            x = self.sigmoid(x)
        return x
