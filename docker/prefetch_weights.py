"""Download the torchvision and torchmetrics weights used by the Track A metrics.

The evaluation runs without network access, so the VGG16 weights for the perceptual
distance and the Inception weights for FID, KID, and Inception Score are part of the image.
"""

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision.models import VGG16_Weights, vgg16

vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
FrechetInceptionDistance(feature=2048)
KernelInceptionDistance(feature=2048, subset_size=2)
InceptionScore()
print("Prefetched VGG16 and Inception weights.")
