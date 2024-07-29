from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian(window_size, sigma):
    """Generate Gaussian Distribution.
    """
    gauss = torch.tensor(
        [exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    """Kernel Inflating. `k => C x 1 x K x K`
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def get_ssim(img1, img2, L=255, window_size=11, K1=0.01, K2=0.03):
    """Return the SSIM score between `img1` and `img2`.
    Args:
        img1 (tensor): Holding the first RGB image batch.
        img2 (tensor): Holding the second RGB image batch.
        L       (int): The dynamic range of the images.
        window_size (int): Size of blur kernel to use (will be reduced for small images).
        k1 (float): Constant used to maintain stability in the SSIM calculation (0.01 in the original paper).
        k2 (float): Constant used to maintain stability in the SSIM calculation (0.03 in the original paper).
    Returns:
        SSIM score between `img1` and `img2`.
    """
    (_, C, H, W) = img1.shape

    real_size = min(window_size, H, W)
    window = create_window(real_size, channel=C)

    # get mean values through group conv operation
    mu1 = F.conv2d(img1, window.to(img1.device), groups=C)
    mu2 = F.conv2d(img2, window.to(img2.device), groups=C)

    mu1_square = mu1.pow(2)
    mu2_square = mu2.pow(2)

    # `Var(X) = E[X^2] -(E[X])^2`
    sigma1_square = F.conv2d(img1 * img1, window.to(img1.device), groups=C) - mu1_square
    sigma2_square = F.conv2d(img2 * img2, window.to(img2.device), groups=C) - mu2_square

    # `Cov(X,Y) = E[XY] - E[X]E[Y]`
    sigma12 = F.conv2d(img1 * img2, window.to(img2.device), groups=C) - mu1 * mu2

    C1 = (K1 * L)**2
    C2 = (K2 * L)**2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_square + sigma2_square + C2

    # contrast & structure
    cs = torch.mean(v1 / v2)
    # ssim mean value
    ssim = ((2 * mu1 * mu2 + C1) * v1) / ((mu1_square + mu2_square + C1) * v2)
    ssim = torch.clamp(ssim, min=1e-8, max=1)
    ssim = torch.mean(ssim)

    return ssim, cs


def get_msssim(img1, img2, normalize=False):
    """Return the MS-SSIM score between `img1` and `img2`.
    """
    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=img1.device)
    levels = weights.shape[0]
    msssim = []
    mscs = []

    for _ in range(levels):
        ssim, cs = get_ssim(img1, img2)
        msssim.append(ssim)
        mscs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    msssim = torch.stack(msssim)
    mscs = torch.stack(mscs)

    # Normalize (to avoid NaNs during training unstable models,
    # not compliant with original definition)
    if normalize:
        msssim = (msssim + 1) / 2
        mscs = (mscs + 1) / 2

    pow1 = mscs**weights
    pow2 = msssim**weights
    output = torch.prod(pow1[:-1] * pow2[-1])

    return output


class SSIM(nn.Module):
    """SSIM loss function.
    """
    def __init__(self):
        super(SSIM, self).__init__()

    def forward(self, img1, img2):
        return get_ssim(img1, img2)


class MSSSIM(nn.Module):
    """MS-SSIM loss function.
    """
    def __init__(self):
        super(MSSSIM, self).__init__()

    def forward(self, img1, img2):
        return 1 - get_msssim(img1, img2)
