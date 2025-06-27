"""Data transforms and augmentations for shot type classification."""

from typing import List, Tuple
from fastai.vision.all import *


def get_extra_transforms(base_size: int = 375) -> List:
    """Get additional transforms as used in the original implementation.
    
    Note: Many of these transforms from fastai v1 are not directly available
    in fastai v2, so we'll use the standard aug_transforms instead.
    
    Args:
        base_size: Base size for transform calculations
        
    Returns:
        List of transforms (placeholder for v2 compatibility)
    """
    # These were the original v1 transforms that are no longer available:
    # cutout, jitter, skew, squish, tilt, perspective_warp, crop_pad, rgb_randomize
    
    # In fastai v2, these are handled by aug_transforms()
    return []


def get_transforms(
    do_flip: bool = True,
    flip_vert: bool = False,
    max_zoom: float = 1.0,
    max_lighting: float = 0.4,
    max_warp: float = 0.3,
    p_affine: float = 0.85,
    p_lighting: float = 0.85,
    image_size: Tuple[int, int] = (375, 666)
) -> List:
    """Get data transforms compatible with fastai v2.
    
    Args:
        do_flip: Whether to apply horizontal flipping
        flip_vert: Whether to apply vertical flipping  
        max_zoom: Maximum zoom factor
        max_lighting: Maximum lighting change
        max_warp: Maximum warping
        p_affine: Probability of affine transforms
        p_lighting: Probability of lighting transforms
        image_size: Target image size (height, width)
        
    Returns:
        List of transforms for fastai v2
    """
    return [
        *aug_transforms(
            do_flip=do_flip,
            flip_vert=flip_vert,
            max_zoom=max_zoom,
            max_lighting=max_lighting,
            max_warp=max_warp,
            p_affine=p_affine,
            p_lighting=p_lighting
        ),
        Normalize.from_stats(*imagenet_stats)
    ]


def create_data_transforms(image_size: Tuple[int, int] = (375, 666)) -> dict:
    """Create item and batch transforms for DataBlock.
    
    Args:
        image_size: Target image size (height, width)
        
    Returns:
        Dictionary with item_tfms and batch_tfms
    """
    return {
        'item_tfms': Resize(image_size, method=ResizeMethod.Squish),
        'batch_tfms': get_transforms(image_size=image_size)
    } 