import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from torchvision.transforms.v2 import Resize

from .chromaticity_difference import srgb2lab


def _log_gabor(rows: int, cols: int, omega0: float, sigmaF: float) -> Tensor:
    u1 = torch.arange(cols).float() - (cols // 2 + 1)
    u2 = torch.arange(rows).float() - (rows // 2 + 1)
    u1 = u1 / (cols - (cols % 2))
    u2 = u2 / (rows - (rows % 2))

    u1, u2 = torch.meshgrid(u1, u2)

    mask = torch.ones(rows, cols)
    mask[(u1**2 + u2**2) > 0.25] = 0

    u1 = u1 * mask
    u2 = u2 * mask

    u1 = torch.fft.fftshift(u1)
    u2 = torch.fft.fftshift(u2)

    radius = torch.sqrt(u1**2 + u2**2)
    radius[0, 0] = 1  # Avoid division by zero

    # Compute the Log-Gabor filter
    LG = torch.exp((-((torch.log(radius / omega0)) ** 2)) / (2 * (sigmaF**2)))
    LG[0, 0] = 0  # Set the DC component to zero

    return LG


class VSI(Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _SDSP(
        image: Tensor,
        sigmaF: float,
        omega0: float,
        sigmaD: float,
        sigmaC: float,
    ):
        """
        Calc saliency map
        """

        dsImage = Resize((256, 256))(image)

        lab = srgb2lab(dsImage.permute(1, 2, 0))
        LChannel = lab[..., 0]
        AChannel = lab[..., 1]
        BChannel = lab[..., 2]

        LFFT = torch.fft.fft2(LChannel)
        AFFT = torch.fft.fft2(AChannel)
        BFFT = torch.fft.fft2(BChannel)

        LG = _log_gabor(256, 256, omega0, sigmaF)

        FinalLResult = torch.fft.ifft2(LFFT * LG)
        FinalAResult = torch.fft.ifft2(AFFT * LG)
        FinalBResult = torch.fft.ifft2(BFFT * LG)

        # Compute the saliency map
        SFMap = torch.sqrt(FinalLResult**2 + FinalAResult**2 + FinalBResult**2)

        # Central bias map
        rows, cols = SFMap.shape
        coordinateMtx = torch.zeros((2, rows, cols))
        coordinateMtx[0, :, :] = torch.arange(0, rows).reshape(-1, 1)
        coordinateMtx[1, :, :] = torch.arange(0, cols)

        centerY = rows / 2
        centerX = cols / 2
        centerMtx = torch.zeros((2, rows, cols))
        centerMtx[0, :, :] = centerY
        centerMtx[1, :, :] = centerX
        SDMap = torch.exp(
            -torch.sum((coordinateMtx - centerMtx) ** 2, axis=0) / sigmaD**2
        )

        # Warm colors bias
        maxA = torch.max(AChannel)
        minA = torch.min(AChannel)
        normalizedA = (AChannel - minA) / (maxA - minA)

        maxB = torch.max(BChannel)
        minB = torch.min(BChannel)
        normalizedB = (BChannel - minB) / (maxB - minB)

        labDistSquare = normalizedA**2 + normalizedB**2
        SCMap = 1 - torch.exp(-labDistSquare / (sigmaC**2))

        # Final visual saliency map
        VSMap = SFMap * SDMap * SCMap

        VSMap = Resize((image.shape[1], image.shape[2]))(VSMap[None])

        return VSMap

    @classmethod
    def _prepare_img_info(
        cls,
        image: Tensor,
        sigmaF: float,
        omega0: float,
        sigmaD: float,
        sigmaC: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        saliencyMap = cls._SDSP(image, sigmaF, omega0, sigmaD, sigmaC)

        L = (
            0.06 * image[0, :, :]
            + 0.63 * image[1, :, :]
            + 0.27 * image[2, :, :]
        )
        M = (
            0.30 * image[0, :, :]
            + 0.04 * image[1, :, :]
            - 0.35 * image[2, :, :]
        )
        N = (
            0.34 * image[0, :, :]
            - 0.60 * image[1, :, :]
            + 0.17 * image[2, :, :]
        )

        rows, cols, _ = image.shape

        # Downsample the image
        minDimension = min(rows, cols)
        stride = max(1, round(minDimension / 256))

        def average_pooling(image: Tensor, kernel_size: int) -> Tensor:
            return F.avg_pool2d(
                image.unsqueeze(0), kernel_size, stride=kernel_size
            ).squeeze(0)

        M = average_pooling(M, stride)
        N = average_pooling(N, stride)
        L = average_pooling(L, stride)

        saliencyMap = average_pooling(saliencyMap.real[0], stride)

        # Calculate the gradient map
        dx = (
            torch.tensor(
                [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32
            )
            / 16
        )
        dy = (
            torch.tensor(
                [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32
            )
            / 16
        )

        def compute_gradient(image: Tensor) -> Tensor:
            Ix = F.conv2d(image[None], dx[None, None], padding="same")
            Iy = F.conv2d(image[None], dy[None, None], padding="same")
            return torch.hypot(Ix, Iy)

        gradientMap = compute_gradient(L)[0]

        return saliencyMap, gradientMap, M, N

    def forward(self, img1: Tensor, img2: Tensor) -> float:
        # Constants
        constForVS = 1.27
        constForGM = 386
        constForChrom = 130
        alpha = 0.40
        lambda_ = 0.020
        sigmaF = 1.34
        omega0 = 0.0210
        sigmaD = 145
        sigmaC = 0.001

        saliencyMap1, gradientMap1, M1, N1 = self._prepare_img_info(
            img1, sigmaF, omega0, sigmaD, sigmaC
        )
        saliencyMap2, gradientMap2, M2, N2 = self._prepare_img_info(
            img2, sigmaF, omega0, sigmaD, sigmaC
        )

        # Calculate the VSI
        VSSimMatrix = (2 * saliencyMap1 * saliencyMap2 + constForVS) / (
            saliencyMap1**2 + saliencyMap2**2 + constForVS
        )
        gradientSimMatrix = (2 * gradientMap1 * gradientMap2 + constForGM) / (
            gradientMap1**2 + gradientMap2**2 + constForGM
        )

        weight = torch.max(saliencyMap1, saliencyMap2)

        ISimMatrix = (2 * M1 * M2 + constForChrom) / (
            M1**2 + M2**2 + constForChrom
        )
        QSimMatrix = (2 * N1 * N2 + constForChrom) / (
            N1**2 + N2**2 + constForChrom
        )

        SimMatrixC = (
            (gradientSimMatrix**alpha)
            * VSSimMatrix
            * (ISimMatrix * QSimMatrix) ** lambda_
            * weight
        )
        sim = torch.sum(SimMatrixC) / torch.sum(weight)
        return sim
