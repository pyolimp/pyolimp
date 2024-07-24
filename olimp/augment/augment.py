from __future__ import annotations
from typing import Any, Callable, Sequence, Iterator
from random import Random
from .transformation import (
    create as create_transformation,
    Transformation,
    Datum,
)


def prepare(
    operations: Sequence[dict[str, Any]]
) -> Iterator[tuple[float, Transformation]]:
    for operation in operations:
        assert isinstance(operation, dict), operation
        probability: float = operation.get("probability", 1.0)
        if probability <= 0.0:
            continue
        inst = create_transformation(
            operation["name"], operation.get("args", {})
        )
        yield probability, inst


def create_augmentation(
    operations: Sequence[dict[str, Any]]
) -> Callable[[Datum, Random], Datum]:
    transformations = list(prepare(operations))

    def augment(datum: Datum, random: Random) -> Datum:
        for probability, transformation in transformations:
            if probability >= 1.0 or probability >= random.random():
                datum = transformation(datum, random)
        return datum

    return augment


def main():
    augment = create_augmentation(
        [
            {
                "name": "scale",
                "args": {
                    "distribution": {
                        "name": "gauss",
                        "a": 0.5,
                        "b": 1.5,
                        "sigma": 1.0,
                    }
                },
            },
            {"name": "rasterize"},
            # {"name": "noising", "args": {"mean": 10, "std": 0.1}},
        ]
    )
    import torch

    random = Random(93218344)

    image_in = torch.nn.functional.pad(torch.ones((20, 30)), (10, 10, 10, 10))[
        None, None, ...
    ].repeat(3, 1, 1, 1)
    datum_out = augment(Datum.from_tensor(image_in), random)
    from minimg.view.view_client import connect

    c = connect()
    assert datum_out.image is not None
    c.log("OUT", datum_out.image.numpy()[:, 0].transpose(1, 2, 0).copy())


if __name__ == "__main__":
    main()
