"""Ordinal regression approach for shot type classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.vision.all import *
from fastai.callback.core import Callback
from fastai.learner import Learner
from pathlib import Path
from typing import Union, Tuple, Optional, List
import numpy as np

from .core import ShotTypeClassifier


class OrdinalLoss(nn.Module):
    """
    Ordinal regression loss that penalizes distant errors more than adjacent ones.
    Uses a combination of cross-entropy loss and distance-weighted penalty.
    """
    
    def __init__(self, class_to_ordinal_map: dict, lambda_ord: float = 1.0):
        """
        Args:
            class_to_ordinal_map: Mapping from class names to ordinal positions
            lambda_ord: Weight for ordinal penalty term
        """
        super().__init__()
        self.class_to_ordinal_map = class_to_ordinal_map
        self.lambda_ord = lambda_ord
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard cross-entropy loss
        ce_loss = self.ce_loss(predictions, targets)
        
        # Ordinal penalty: punish predictions based on distance from true class
        batch_size = predictions.size(0)
        ordinal_penalty = 0.0
        
        # Get class names from targets (assuming vocab is available)
        for i in range(batch_size):
            true_class_idx = targets[i].item()
            pred_probs = F.softmax(predictions[i], dim=0)
            
            # Calculate expected distance penalty between predicted and true ordinal positions
            for j in range(predictions.size(1)):
                # Calculate ordinal distance between classes j and true_class_idx
                ordinal_distance = abs(j - true_class_idx)  # Simple index-based distance
                ordinal_penalty += pred_probs[j] * (ordinal_distance ** 2)
        
        ordinal_penalty = ordinal_penalty / batch_size
        
        return ce_loss + self.lambda_ord * ordinal_penalty


class LabelSmoothingOrdinalLoss(nn.Module):
    """
    Label smoothing approach for ordinal classification.
    Gives higher probabilities to adjacent classes.
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        """
        Args:
            num_classes: Number of ordinal classes
            smoothing: Amount of smoothing (0.0 = no smoothing, 1.0 = uniform)
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size = predictions.size(0)
        log_probs = F.log_softmax(predictions, dim=1)
        
        # Create smooth labels
        smooth_labels = torch.zeros_like(log_probs)
        
        for i in range(batch_size):
            true_class = targets[i].item()
            
            # Base probability for true class
            smooth_labels[i, true_class] = 1.0 - self.smoothing
            
            # Distribute smoothing to adjacent classes with decreasing weights
            remaining_prob = self.smoothing
            for distance in range(1, self.num_classes):
                weight = max(0, 1.0 - distance * 0.3)  # Decrease by 30% per distance unit
                
                for direction in [-1, 1]:
                    adj_class = true_class + direction * distance
                    if 0 <= adj_class < self.num_classes:
                        prob = remaining_prob * weight / 2  # Split between left and right
                        smooth_labels[i, adj_class] = prob
                        remaining_prob -= prob
                        
                if remaining_prob <= 0:
                    break
        
        return -(smooth_labels * log_probs).sum(dim=1).mean()


class OrdinalAccuracy(AccumMetric):
    """
    Ordinal accuracy that gives partial credit for close predictions.
    """
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        super().__init__(self._ordinal_accuracy_func)
        
    def _ordinal_accuracy_func(self, inp, targ):
        """Calculate ordinal accuracy with partial credit."""
        pred_classes = inp.argmax(dim=1)
        distances = torch.abs(pred_classes - targ).float()
        max_distance = self.num_classes - 1
        
        # Give full credit for exact match, partial credit based on distance
        ordinal_acc = torch.clamp(1.0 - distances / max_distance, min=0.0)
        return ordinal_acc


class OrdinalMeanAbsoluteError(AccumMetric):
    """
    Mean Absolute Error for ordinal predictions.
    """
    
    def __init__(self):
        super().__init__(self._mae_func)
        
    def _mae_func(self, inp, targ):
        """Calculate mean absolute error."""
        pred_classes = inp.argmax(dim=1)
        mae = torch.abs(pred_classes - targ).float()
        return mae


class OrdinalShotTypeClassifier(ShotTypeClassifier):
    """
    Shot type classifier using ordinal regression approach.
    """
    
    # Define ordinal mapping: closer numbers = more similar shot types
    ORDINAL_MAPPING = {
        '0-extreme-close-up': 0,    # Closest
        '1-close-up': 1,
        '2-medium-close-up': 2,
        '3-medium': 3,
        '4-long': 4,
        '5-extreme-wide': 5,        # Farthest
        '6-none': 6                 # Special case
    }
    
    REVERSE_ORDINAL_MAPPING = {v: k for k, v in ORDINAL_MAPPING.items()}
    
    def __init__(self, model_path: Optional[Union[str, Path]] = None, loss_type: str = 'ordinal'):
        """
        Initialize ordinal classifier.
        
        Args:
            model_path: Path to pre-trained model
            loss_type: 'ordinal', 'label_smoothing', or 'standard'
        """
        super().__init__(model_path)
        self.loss_type = loss_type
        
    def create_ordinal_dataloaders(
        self,
        data_path: Union[str, Path],
        train_folder: str = "train",
        valid_folder: str = "valid",
        image_size: Tuple[int, int] = (375, 666),
        batch_size: int = 16,
        num_workers: int = 0
    ) -> DataLoaders:
        """
        Create dataloaders with ordinal label encoding.
        """
        data_path = Path(data_path)
        
        def ordinal_label_func(x):
            """Convert folder name to ordinal category name"""
            folder_name = x.parent.name
            # Return the folder name itself, let fastai handle the encoding
            return folder_name
        
        # Create DataBlock with ordinal labels
        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            splitter=GrandparentSplitter(train_name=train_folder, valid_name=valid_folder),
            get_y=ordinal_label_func,
            item_tfms=Resize(image_size, method=ResizeMethod.Squish),
            batch_tfms=[
                *aug_transforms(
                    do_flip=True,
                    flip_vert=False,
                    max_zoom=1.0,
                    max_lighting=0.4,
                    max_warp=0.3,
                    p_affine=0.85,
                    p_lighting=0.85
                ),
                Normalize.from_stats(*imagenet_stats)
            ]
        )
        
        self.dls = dblock.dataloaders(data_path, bs=batch_size, num_workers=num_workers)
        return self.dls
    
    def create_ordinal_learner(
        self,
        arch=resnet50,
        pretrained: bool = True,
        lambda_ord: float = 1.0,
        smoothing: float = 0.1
    ) -> Learner:
        """
        Create learner with ordinal loss function.
        """
        if self.dls is None:
            raise ValueError("Must create dataloaders first")
            
        num_classes = len(self.ORDINAL_MAPPING)
        
        # Choose loss function
        if self.loss_type == 'ordinal':
            loss_func = OrdinalLoss(self.ORDINAL_MAPPING, lambda_ord)
        elif self.loss_type == 'label_smoothing':
            loss_func = LabelSmoothingOrdinalLoss(num_classes, smoothing)
        else:
            loss_func = CrossEntropyLossFlat()
            
        # Metrics
        metrics = [
            accuracy,
            OrdinalAccuracy(num_classes),
            OrdinalMeanAbsoluteError()
        ]
        
        self.learn = vision_learner(
            self.dls,
            arch,
            metrics=metrics,
            pretrained=pretrained,
            loss_func=loss_func
        )
        
        return self.learn
    
    def train_ordinal(
        self,
        data_path: Union[str, Path],
        epochs: int = 5,
        learning_rate: float = 1e-3,
        fine_tune_epochs: int = 3,
        image_size: Tuple[int, int] = (375, 666),
        batch_size: int = 16,
        save_model: bool = True,
        model_name: str = "ordinal-shot-classifier",
        loss_type: str = 'ordinal',
        lambda_ord: float = 1.0
    ) -> None:
        """
        Train ordinal regression model.
        """
        self.loss_type = loss_type
        
        # Create ordinal dataloaders
        self.create_ordinal_dataloaders(
            data_path,
            image_size=image_size,
            batch_size=batch_size
        )
        
        # Create ordinal learner
        self.create_ordinal_learner(
            lambda_ord=lambda_ord
        )
        
        print(f"Training with {loss_type} loss...")
        print(f"Ordinal mapping: {self.ORDINAL_MAPPING}")
        
        # Find learning rate
        self.learn.lr_find()
        
        # Train
        self.learn.fit_one_cycle(epochs, learning_rate)
        
        # Fine-tune
        self.learn.unfreeze()
        self.learn.fit_one_cycle(fine_tune_epochs, learning_rate/10)
        
        # Save model
        if save_model:
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            self.learn.export(model_dir / f"{model_name}.pkl")
            print(f"Model saved to {model_dir / f'{model_name}.pkl'}")
    
    def predict_ordinal(
        self,
        image_path: Union[str, Path],
        return_probs: bool = False
    ) -> Union[str, Tuple[str, torch.Tensor, int]]:
        """
        Predict with ordinal model.
        """
        if self.learn is None:
            raise ValueError("Must load or create a model first")
            
        img = PILImage.create(image_path)
        pred_class, pred_idx, probs = self.learn.predict(img)
        
        # The model returns the class name directly, not an ordinal index
        class_name = str(pred_class)
        
        if return_probs:
            return class_name, probs, int(pred_idx)
        return class_name
    
    def predict_single(
        self, 
        image_path: Union[str, Path],
        return_probs: bool = False
    ) -> Union[str, Tuple[str, torch.Tensor]]:
        """
        Override parent method to use ordinal prediction.
        """
        return self.predict_ordinal(image_path, return_probs)


def compare_models(
    data_path: Union[str, Path],
    test_path: Union[str, Path] = None,
    epochs: int = 3
) -> dict:
    """
    Compare standard vs ordinal classification approaches.
    """
    results = {}
    
    # Train standard classifier
    print("="*50)
    print("TRAINING STANDARD CLASSIFIER")
    print("="*50)
    
    standard_classifier = ShotTypeClassifier()
    standard_classifier.train(
        data_path=data_path,
        epochs=epochs,
        model_name="standard-comparison"
    )
    
    # Train ordinal classifiers
    for loss_type in ['ordinal', 'label_smoothing']:
        print("="*50)
        print(f"TRAINING ORDINAL CLASSIFIER ({loss_type.upper()})")
        print("="*50)
        
        ordinal_classifier = OrdinalShotTypeClassifier(loss_type=loss_type)
        ordinal_classifier.train_ordinal(
            data_path=data_path,
            epochs=epochs,
            model_name=f"ordinal-{loss_type}-comparison",
            loss_type=loss_type
        )
        
        results[f'ordinal_{loss_type}'] = ordinal_classifier
    
    results['standard'] = standard_classifier
    
    return results 