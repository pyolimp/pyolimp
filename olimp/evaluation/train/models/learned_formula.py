import torch
from torch import nn, Tensor
from olimp.evaluation.train.models.basic_block import BasicMLPBlock


class LearnedMetricFromFormula(nn.Module):
    def __init__(
        self,
        formula: str,
        learned_vars: list[str],
        predictor_input_vars: list[str],
        in_features: int = 2,
        hidden_dims: tuple[int, ...] = (64, 32),
        apply_sigmoid: bool = True,
    ):
        super().__init__()
        self.formula = compile(formula, "<string>", "eval")
        self.learned_vars = learned_vars
        self.predictor_input_vars = predictor_input_vars

        self.predictors = nn.ModuleDict(
            {
                var: BasicMLPBlock(
                    in_features=in_features,
                    hidden_dims=hidden_dims,
                    out_features=1,
                    apply_sigmoid=apply_sigmoid,
                )
                for var in learned_vars
            }
        )

    def forward(self, input_vars: dict[str, Tensor]) -> Tensor:
        context = {
            name: self.predictors[name](
                torch.stack(
                    [input_vars[k] for k in self.predictor_input_vars], dim=1
                ).float()
            ).squeeze(1)
            for name in self.learned_vars
        }

        for k, v in input_vars.items():
            context[k] = (
                v
                if isinstance(v, Tensor)
                else torch.tensor(v, dtype=torch.float32)
            )

        result = eval(self.formula, {"__builtins__": __builtins__}, context)

        if not isinstance(result, Tensor):
            raise TypeError("Formula result must be a Tensor")

        return result
