import torch
from torch import Tensor
from . import PrecompensationKERUNC
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _demo():
    from ...._demo import demo
    from typing import Callable

    def demo_kerunc(
        image: Tensor, psf: Tensor, progress: Callable[[float], None]
    ) -> Tensor:
        # Log input shapes
        print(f"Image shape: {image.shape}")
        print(f"PSF shape: {psf.shape}")

        # Model initialization
        start_time = time.time()
        model = (
            PrecompensationKERUNC()
        )  # .from_path(path="hf://RVI/KERUNC.pt")
        init_time = time.time() - start_time
        print(f"Model initialization time: {init_time:.4f} seconds")

        with torch.inference_mode():
            # Preprocessing
            start_time = time.time()
            inputs = model.preprocess(image, psf.to(torch.float32))
            preprocess_time = time.time() - start_time
            print(f"Preprocessing time: {preprocess_time:.4f} seconds")
            progress(0.1)

            # Model inference
            start_time = time.time()
            (precompensation,) = model(inputs)
            inference_time = time.time() - start_time
            print(f"Model inference time: {inference_time:.4f} seconds")
            progress(1.0)

            return precompensation

    print("Starting KERUNC demo")
    demo("KERUNC", demo_kerunc, mono=True, num_output_channels=3)
    print("Demo completed")


if __name__ == "__main__":
    _demo()
