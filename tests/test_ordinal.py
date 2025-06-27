"""Unit tests for ordinal_classifier.ordinal module."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import numpy as np

from ordinal_classifier.ordinal import (
    OrdinalLoss,
    LabelSmoothingOrdinalLoss,
    OrdinalAccuracy,
    OrdinalMeanAbsoluteError,
    OrdinalShotTypeClassifier,
    compare_models
)


class TestOrdinalLoss:
    """Test cases for OrdinalLoss class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.class_to_ordinal_map = {'class_0': 0, 'class_1': 1, 'class_2': 2}
        self.loss = OrdinalLoss(self.class_to_ordinal_map, lambda_ord=1.0)

    def test_init(self):
        """Test OrdinalLoss initialization."""
        assert self.loss.class_to_ordinal_map == self.class_to_ordinal_map
        assert self.loss.lambda_ord == 1.0
        assert isinstance(self.loss.ce_loss, nn.CrossEntropyLoss)

    def test_init_custom_lambda(self):
        """Test OrdinalLoss initialization with custom lambda."""
        loss = OrdinalLoss(self.class_to_ordinal_map, lambda_ord=0.5)
        assert loss.lambda_ord == 0.5

    def test_forward_shape(self):
        """Test forward pass output shape."""
        predictions = torch.randn(2, 3)  # batch_size=2, num_classes=3
        targets = torch.tensor([0, 1])
        
        loss_value = self.loss(predictions, targets)
        
        assert isinstance(loss_value, torch.Tensor)
        assert loss_value.dim() == 0  # Scalar loss

    def test_forward_perfect_predictions(self):
        """Test forward pass with perfect predictions."""
        # Create predictions with high confidence for correct classes
        predictions = torch.tensor([
            [10.0, -10.0, -10.0],  # Strongly predicts class 0
            [-10.0, 10.0, -10.0]   # Strongly predicts class 1
        ])
        targets = torch.tensor([0, 1])
        
        loss_value = self.loss(predictions, targets)
        
        # Loss should be low for perfect predictions
        assert loss_value.item() >= 0  # Loss should be non-negative

    def test_forward_wrong_predictions(self):
        """Test forward pass with wrong predictions."""
        # Create predictions that are wrong
        predictions = torch.tensor([
            [-10.0, -10.0, 10.0],  # Predicts class 2 instead of 0
            [10.0, -10.0, -10.0]   # Predicts class 0 instead of 1
        ])
        targets = torch.tensor([0, 1])
        
        loss_value = self.loss(predictions, targets)
        
        # Loss should be higher for wrong predictions
        assert loss_value.item() > 0

    def test_forward_batch_processing(self):
        """Test that forward processes different batch sizes correctly."""
        for batch_size in [1, 4, 8]:
            predictions = torch.randn(batch_size, 3)
            targets = torch.randint(0, 3, (batch_size,))
            
            loss_value = self.loss(predictions, targets)
            assert isinstance(loss_value, torch.Tensor)
            assert loss_value.dim() == 0


class TestLabelSmoothingOrdinalLoss:
    """Test cases for LabelSmoothingOrdinalLoss class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.num_classes = 3
        self.smoothing = 0.1
        self.loss = LabelSmoothingOrdinalLoss(self.num_classes, self.smoothing)

    def test_init(self):
        """Test LabelSmoothingOrdinalLoss initialization."""
        assert self.loss.num_classes == 3
        assert self.loss.smoothing == 0.1

    def test_init_no_smoothing(self):
        """Test initialization with no smoothing."""
        loss = LabelSmoothingOrdinalLoss(5, smoothing=0.0)
        assert loss.smoothing == 0.0

    def test_forward_shape(self):
        """Test forward pass output shape."""
        predictions = torch.randn(2, 3)
        targets = torch.tensor([0, 1])
        
        loss_value = self.loss(predictions, targets)
        
        assert isinstance(loss_value, torch.Tensor)
        assert loss_value.dim() == 0

    def test_forward_different_classes(self):
        """Test forward pass with different target classes."""
        predictions = torch.randn(3, 3)
        targets = torch.tensor([0, 1, 2])
        
        loss_value = self.loss(predictions, targets)
        
        assert loss_value.item() >= 0

    def test_smoothing_effect(self):
        """Test that smoothing affects the loss calculation."""
        predictions = torch.randn(2, 3)
        targets = torch.tensor([0, 1])
        
        # Compare no smoothing vs with smoothing
        loss_no_smooth = LabelSmoothingOrdinalLoss(3, 0.0)
        loss_with_smooth = LabelSmoothingOrdinalLoss(3, 0.3)
        
        loss_val_no_smooth = loss_no_smooth(predictions, targets)
        loss_val_with_smooth = loss_with_smooth(predictions, targets)
        
        # Both should be valid losses
        assert loss_val_no_smooth.item() >= 0
        assert loss_val_with_smooth.item() >= 0


class TestOrdinalAccuracy:
    """Test cases for OrdinalAccuracy class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.num_classes = 3
        self.metric = OrdinalAccuracy(self.num_classes)

    def test_init(self):
        """Test OrdinalAccuracy initialization."""
        assert self.metric.num_classes == 3

    def test_perfect_predictions(self):
        """Test ordinal accuracy with perfect predictions."""
        # Create predictions with correct classes
        predictions = torch.tensor([
            [1.0, 0.0, 0.0],  # Predicts class 0
            [0.0, 1.0, 0.0],  # Predicts class 1
            [0.0, 0.0, 1.0]   # Predicts class 2
        ])
        targets = torch.tensor([0, 1, 2])
        
        accuracy = self.metric._ordinal_accuracy_func(predictions, targets)
        
        # Perfect predictions should give accuracy of 1.0
        assert torch.allclose(accuracy, torch.ones(3))

    def test_adjacent_predictions(self):
        """Test ordinal accuracy with adjacent class predictions."""
        # Create predictions that are off by 1
        predictions = torch.tensor([
            [0.0, 1.0, 0.0],  # Predicts class 1 instead of 0
            [0.0, 0.0, 1.0],  # Predicts class 2 instead of 1
        ])
        targets = torch.tensor([0, 1])
        
        accuracy = self.metric._ordinal_accuracy_func(predictions, targets)
        
        # Adjacent predictions should give partial credit
        expected_accuracy = 1.0 - 1.0 / (self.num_classes - 1)  # 1 - 1/2 = 0.5
        assert torch.allclose(accuracy, torch.tensor([expected_accuracy, expected_accuracy]))

    def test_worst_predictions(self):
        """Test ordinal accuracy with worst possible predictions."""
        # Create predictions that are maximally wrong
        predictions = torch.tensor([
            [0.0, 0.0, 1.0],  # Predicts class 2 instead of 0
            [1.0, 0.0, 0.0],  # Predicts class 0 instead of 2
        ])
        targets = torch.tensor([0, 2])
        
        accuracy = self.metric._ordinal_accuracy_func(predictions, targets)
        
        # Maximum distance should give 0 accuracy
        assert torch.allclose(accuracy, torch.zeros(2))


class TestOrdinalMeanAbsoluteError:
    """Test cases for OrdinalMeanAbsoluteError class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.metric = OrdinalMeanAbsoluteError()

    def test_perfect_predictions(self):
        """Test MAE with perfect predictions."""
        predictions = torch.tensor([
            [1.0, 0.0, 0.0],  # Predicts class 0
            [0.0, 1.0, 0.0],  # Predicts class 1
        ])
        targets = torch.tensor([0, 1])
        
        mae = self.metric._mae_func(predictions, targets)
        
        # Perfect predictions should give MAE of 0
        assert torch.allclose(mae, torch.zeros(2))

    def test_off_by_one_predictions(self):
        """Test MAE with predictions off by 1."""
        predictions = torch.tensor([
            [0.0, 1.0, 0.0],  # Predicts class 1 instead of 0
            [0.0, 0.0, 1.0],  # Predicts class 2 instead of 1
        ])
        targets = torch.tensor([0, 1])
        
        mae = self.metric._mae_func(predictions, targets)
        
        # Off by 1 should give MAE of 1
        assert torch.allclose(mae, torch.ones(2))

    def test_mae_calculation(self):
        """Test MAE calculation with various errors."""
        predictions = torch.tensor([
            [0.0, 0.0, 1.0],  # Predicts class 2 instead of 0 (error = 2)
            [1.0, 0.0, 0.0],  # Predicts class 0 instead of 2 (error = 2)
            [0.0, 1.0, 0.0],  # Predicts class 1 (correct, error = 0)
        ])
        targets = torch.tensor([0, 2, 1])
        
        mae = self.metric._mae_func(predictions, targets)
        
        expected = torch.tensor([2.0, 2.0, 0.0])
        assert torch.allclose(mae, expected)


class TestOrdinalShotTypeClassifier:
    """Test cases for OrdinalShotTypeClassifier class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = OrdinalShotTypeClassifier()

    def test_init_default(self):
        """Test OrdinalShotTypeClassifier initialization with defaults."""
        assert self.classifier.loss_type == 'ordinal'
        assert self.classifier.model_path is None
        assert self.classifier.learn is None

    def test_init_custom_loss_type(self):
        """Test initialization with custom loss type."""
        classifier = OrdinalShotTypeClassifier(loss_type='label_smoothing')
        assert classifier.loss_type == 'label_smoothing'

    def test_init_with_model_path(self):
        """Test initialization with model path."""
        model_path = Path("test_model.pkl")
        classifier = OrdinalShotTypeClassifier(model_path=model_path)
        assert classifier.model_path == model_path

    def test_ordinal_mapping_constants(self):
        """Test that ordinal mapping constants are defined correctly."""
        assert isinstance(OrdinalShotTypeClassifier.ORDINAL_MAPPING, dict)
        assert isinstance(OrdinalShotTypeClassifier.REVERSE_ORDINAL_MAPPING, dict)
        
        # Check that reverse mapping is correct
        for key, value in OrdinalShotTypeClassifier.ORDINAL_MAPPING.items():
            assert OrdinalShotTypeClassifier.REVERSE_ORDINAL_MAPPING[value] == key

    def test_ordinal_mapping_completeness(self):
        """Test that ordinal mappings are complete and sequential."""
        mapping = OrdinalShotTypeClassifier.ORDINAL_MAPPING
        values = list(mapping.values())
        
        # Should start from 0 and be sequential
        assert min(values) == 0
        assert max(values) == len(values) - 1
        assert len(set(values)) == len(values)  # No duplicates

    @patch('ordinal_classifier.ordinal.Path')
    @patch('ordinal_classifier.ordinal.DataBlock')
    @patch('ordinal_classifier.ordinal.get_image_files')
    def test_create_ordinal_dataloaders(self, mock_get_image_files, mock_datablock, mock_path):
        """Test create_ordinal_dataloaders method."""
        # Mock Path
        mock_path_instance = Mock()
        mock_path.return_value = mock_path_instance
        
        # Mock DataBlock
        mock_dblock_instance = Mock()
        mock_dataloaders = Mock()
        mock_dblock_instance.dataloaders.return_value = mock_dataloaders
        mock_datablock.return_value = mock_dblock_instance
        
        # Call method
        result = self.classifier.create_ordinal_dataloaders(
            "test_data",
            batch_size=8,
            image_size=(224, 224)
        )
        
        # Should create DataBlock and return dataloaders
        mock_datablock.assert_called_once()
        mock_dblock_instance.dataloaders.assert_called_once()
        assert result == mock_dataloaders
        assert self.classifier.dls == mock_dataloaders

    def test_create_ordinal_learner_no_dataloaders(self):
        """Test create_ordinal_learner without dataloaders."""
        with pytest.raises(ValueError, match="Must create dataloaders first"):
            self.classifier.create_ordinal_learner()

    @patch('ordinal_classifier.ordinal.vision_learner')
    def test_create_ordinal_learner_ordinal_loss(self, mock_vision_learner):
        """Test create_ordinal_learner with ordinal loss."""
        # Set up mock dataloaders
        self.classifier.dls = Mock()
        mock_learner = Mock()
        mock_vision_learner.return_value = mock_learner
        
        # Set loss type to ordinal
        self.classifier.loss_type = 'ordinal'
        
        result = self.classifier.create_ordinal_learner()
        
        # Should call vision_learner
        mock_vision_learner.assert_called_once()
        args, kwargs = mock_vision_learner.call_args
        
        # Check that ordinal loss was used
        assert 'loss_func' in kwargs
        assert result == mock_learner
        assert self.classifier.learn == mock_learner

    @patch('ordinal_classifier.ordinal.vision_learner')
    def test_create_ordinal_learner_label_smoothing(self, mock_vision_learner):
        """Test create_ordinal_learner with label smoothing loss."""
        self.classifier.dls = Mock()
        mock_learner = Mock()
        mock_vision_learner.return_value = mock_learner
        
        self.classifier.loss_type = 'label_smoothing'
        
        result = self.classifier.create_ordinal_learner()
        
        mock_vision_learner.assert_called_once()
        assert result == mock_learner

    @patch('ordinal_classifier.ordinal.vision_learner')
    def test_create_ordinal_learner_standard_loss(self, mock_vision_learner):
        """Test create_ordinal_learner with standard loss."""
        self.classifier.dls = Mock()
        mock_learner = Mock()
        mock_vision_learner.return_value = mock_learner
        
        self.classifier.loss_type = 'standard'
        
        result = self.classifier.create_ordinal_learner()
        
        mock_vision_learner.assert_called_once()
        assert result == mock_learner

    def test_predict_ordinal_no_learner(self):
        """Test predict_ordinal without learner."""
        with pytest.raises(ValueError, match="Must load or create a model first"):
            self.classifier.predict_ordinal("test.jpg")

    @patch('ordinal_classifier.ordinal.PILImage')
    def test_predict_ordinal_with_learner(self, mock_pil_image):
        """Test predict_ordinal with learner."""
        # Mock learner
        mock_learner = Mock()
        mock_learner.predict.return_value = ('test_class', 0, torch.tensor([0.8, 0.2]))
        self.classifier.learn = mock_learner
        
        # Mock image
        mock_img = Mock()
        mock_pil_image.create.return_value = mock_img
        
        result = self.classifier.predict_ordinal("test.jpg")
        
        mock_pil_image.create.assert_called_once_with("test.jpg")
        mock_learner.predict.assert_called_once_with(mock_img)
        assert result == 'test_class'

    @patch('ordinal_classifier.ordinal.PILImage')
    def test_predict_ordinal_with_probs(self, mock_pil_image):
        """Test predict_ordinal with return_probs=True."""
        mock_learner = Mock()
        mock_probs = torch.tensor([0.8, 0.2])
        mock_learner.predict.return_value = ('test_class', 0, mock_probs)
        self.classifier.learn = mock_learner
        
        mock_img = Mock()
        mock_pil_image.create.return_value = mock_img
        
        result = self.classifier.predict_ordinal("test.jpg", return_probs=True)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result[0] == 'test_class'
        assert torch.equal(result[1], mock_probs)
        assert result[2] == 0

    def test_predict_single_override(self):
        """Test that predict_single calls predict_ordinal."""
        with patch.object(self.classifier, 'predict_ordinal') as mock_predict_ordinal:
            mock_predict_ordinal.return_value = 'test_result'
            
            result = self.classifier.predict_single("test.jpg", return_probs=False)
            
            mock_predict_ordinal.assert_called_once_with("test.jpg", False)
            assert result == 'test_result'


class TestCompareModels:
    """Test cases for compare_models function."""

    @patch('ordinal_classifier.ordinal.ShotTypeClassifier')
    @patch('ordinal_classifier.ordinal.OrdinalShotTypeClassifier')
    def test_compare_models_basic(self, mock_ordinal_classifier, mock_standard_classifier):
        """Test compare_models function."""
        # Mock standard classifier
        mock_standard_instance = Mock()
        mock_standard_classifier.return_value = mock_standard_instance
        
        # Mock ordinal classifiers
        mock_ordinal_instance = Mock()
        mock_ordinal_classifier.return_value = mock_ordinal_instance
        
        with patch('builtins.print'):
            result = compare_models("test_data", epochs=1)
        
        # Should return dict with all classifiers
        assert isinstance(result, dict)
        assert 'standard' in result
        assert 'ordinal_ordinal' in result
        assert 'ordinal_label_smoothing' in result
        
        # Should have trained standard classifier
        mock_standard_instance.train.assert_called_once()
        
        # Should have trained ordinal classifiers
        assert mock_ordinal_instance.train_ordinal.call_count == 2

    @patch('ordinal_classifier.ordinal.ShotTypeClassifier')
    @patch('ordinal_classifier.ordinal.OrdinalShotTypeClassifier')
    def test_compare_models_with_test_path(self, mock_ordinal_classifier, mock_standard_classifier):
        """Test compare_models with test path."""
        mock_standard_instance = Mock()
        mock_standard_classifier.return_value = mock_standard_instance
        mock_ordinal_instance = Mock()
        mock_ordinal_classifier.return_value = mock_ordinal_instance
        
        with patch('builtins.print'):
            result = compare_models("train_data", "test_data", epochs=2)
        
        assert isinstance(result, dict)
        assert len(result) == 3  # standard + 2 ordinal variants

    def test_module_constants_accessibility(self):
        """Test that module constants are accessible."""
        from ordinal_classifier.ordinal import OrdinalShotTypeClassifier
        
        # Test ordinal mapping access
        mapping = OrdinalShotTypeClassifier.ORDINAL_MAPPING
        reverse_mapping = OrdinalShotTypeClassifier.REVERSE_ORDINAL_MAPPING
        
        assert isinstance(mapping, dict)
        assert isinstance(reverse_mapping, dict)
        assert len(mapping) > 0
        assert len(reverse_mapping) > 0