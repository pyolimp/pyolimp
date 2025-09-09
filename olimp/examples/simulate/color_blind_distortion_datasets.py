from __future__ import annotations

from olimp.precompensation.nn.dataset.cvd_angle import ColorBlindnessDataset
from olimp.simulate._demo_distortion import demo


if __name__ == "__main__":

    dataset = ColorBlindnessDataset(
        angle_deg={"name": "uniform", "a": 33.0, "b": 360.0}, seed=11, size=365
    )

    def demo_simulate():
        funcs = []

        for i in range(3):
            angle = dataset[i].item()
            funcs.append(
                (
                    lambda image, distortion=dataset._distortions[
                        i
                    ]: distortion()(image),
                    f"{angle:.1f}°",
                )
            )

        return funcs

    demo("ColorBlindnessDistortion (3 angles)", demo_simulate)
