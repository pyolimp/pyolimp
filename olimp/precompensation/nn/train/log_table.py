from __future__ import annotations


class LogTable:
    from rich.box import DOUBLE_EDGE as _box

    def __init__(self, columns: list[tuple[str, int]]) -> None:
        self._names, self._widths = zip(*columns)

    def header(self) -> str:
        box = self._box
        line: list[str] = [
            box.mid_left,
            box.mid_vertical.join(
                [
                    f" {name:^{width-2}s} "
                    for name, width in zip(self._names, self._widths)
                ]
            ),
            box.mid_right,
        ]
        return "\n".join(
            [
                box.get_top(self._widths),
                "".join(line),
                box.get_row(self._widths),
            ]
        )

    def row(self, columns: list[float | int]) -> str:
        box = self._box
        return "".join(
            [
                box.mid_left,
                box.mid_vertical.join(
                    [
                        f" {value:>{width-2}{'.6f' if isinstance(value, float) else 'd'}} "
                        for value, width in zip(
                            columns, self._widths, strict=True
                        )
                    ]
                ),
                box.mid_right,
            ]
        )

    def end(self) -> str:
        return self._box.get_bottom(self._widths)
