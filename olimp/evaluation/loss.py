from piq import MultiScaleSSIMLoss


def ms_ssim(pred, target):
    _ms_ssim = MultiScaleSSIMLoss()
    loss = _ms_ssim(pred, target)
    return loss