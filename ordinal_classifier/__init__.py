"""
Ordinal Classifier
"""

__version__ = "2.0.0"
__author__ = "Rahul Somani"
__email__ = "rahulsomani95@gmail.com"

from .core import ShotTypeClassifier
from .transforms import get_transforms, get_extra_transforms
from .evaluation import ModelEvaluator
from .ordinal import OrdinalShotTypeClassifier, OrdinalLoss, LabelSmoothingOrdinalLoss

__all__ = [
    "ShotTypeClassifier", 
    "get_transforms", 
    "get_extra_transforms", 
    "ModelEvaluator",
    "OrdinalShotTypeClassifier",
    "OrdinalLoss",
    "LabelSmoothingOrdinalLoss"
] 