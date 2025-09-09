from __future__ import annotations

from olimp.simulate.color_blindness_distortion import ColorBlindnessDistortion
from olimp.simulate._demo_distortion import demo


if __name__ == "__main__":

    def demo_simulate():
        yield ColorBlindnessDistortion.from_type("protan")(), "protan"
        yield ColorBlindnessDistortion.from_type("deutan")(), "deutan"
        yield ColorBlindnessDistortion.from_type("tritan")(), "tritan"

    demo("ColorBlindnessDistortion", demo_simulate)
