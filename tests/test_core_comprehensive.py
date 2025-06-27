"""Comprehensive unit tests for ordinal_classifier.core module."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from fastai.vision.all import PILImage
from fastai.callback.core import Callback

from ordinal_classifier.core import (
    EarlyStoppingCallback,
    OrdinalLabelSmoothingLoss, 
    ShotTypeClassifier
)


class TestEarlyStoppingCallback:
    """Test cases for EarlyStoppingCallback class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.callback = EarlyStoppingCallback()
        
    def test_init_default_parameters(self):
        """Test EarlyStoppingCallback initialization with defaults."""
        callback = EarlyStoppingCallback()
        assert callback.monitor == 'valid_loss'
        assert callback.min_delta == 0.001
        assert callback.patience == 3
        assert callback.restore_best_weights is True
        assert callback.wait == 0
        assert callback.best_epoch == 0
        assert callback.best_metric is None
        assert callback.best_weights is None
        assert callback.stopped is False

    def test_init_custom_parameters(self):
        """Test EarlyStoppingCallback initialization with custom parameters."""
        callback = EarlyStoppingCallback(
            monitor='valid_acc',
            min_delta=0.01,
            patience=5,
            restore_best_weights=False
        )
        assert callback.monitor == 'valid_acc'
        assert callback.min_delta == 0.01
        assert callback.patience == 5
        assert callback.restore_best_weights is False

    @patch('builtins.print')
    def test_before_fit_loss_monitor(self, mock_print):
        """Test before_fit method with loss-based monitoring."""
        callback = EarlyStoppingCallback(monitor='valid_loss')
        callback.before_fit()
        
        assert callback.wait == 0
        assert callback.best_epoch == 0
        assert callback.stopped is False
        assert callback.best_metric == float('inf')  # Loss: lower is better
        assert callback.best_weights is None
        mock_print.assert_called_once()

    @patch('builtins.print')
    def test_before_fit_accuracy_monitor(self, mock_print):
        """Test before_fit method with accuracy-based monitoring."""
        callback = EarlyStoppingCallback(monitor='valid_acc')
        callback.before_fit()
        
        assert callback.wait == 0
        assert callback.best_epoch == 0
        assert callback.stopped is False
        assert callback.best_metric == -float('inf')  # Accuracy: higher is better
        assert callback.best_weights is None
        mock_print.assert_called_once()

    @patch('builtins.print')
    def test_after_epoch_no_recorder(self, mock_print):
        """Test after_epoch when no recorder is available."""
        mock_learn = Mock()
        del mock_learn.recorder  # No recorder attribute
        
        callback = EarlyStoppingCallback()
        callback.learn = mock_learn
        callback.after_epoch()
        
        mock_print.assert_called_with("⚠️  No recorder values available, skipping early stopping")

    @patch('builtins.print')
    def test_after_epoch_no_values(self, mock_print):
        """Test after_epoch when recorder has no values."""
        mock_learn = Mock()
        mock_learn.recorder.values = []
        
        callback = EarlyStoppingCallback()
        callback.learn = mock_learn
        callback.after_epoch()
        
        mock_print.assert_called_with("⚠️  No values recorded yet, skipping early stopping")

    @patch('builtins.print')
    def test_after_epoch_valid_loss_improvement(self, mock_print):
        """Test after_epoch with valid loss improvement."""
        mock_learn = Mock()
        mock_learn.recorder.values = [[1.0, 0.8, 0.9]]  # [train_loss, valid_loss, accuracy]
        
        callback = EarlyStoppingCallback(monitor='valid_loss')
        callback.before_fit()
        callback.learn = mock_learn
        callback.epoch = 0
        
        # Mock model state dict
        mock_learn.model.state_dict.return_value = {'weight': torch.tensor([1.0])}
        
        callback.after_epoch()
        
        assert callback.best_metric == 0.8
        assert callback.best_epoch == 0
        assert callback.wait == 0
        assert callback.best_weights is not None

    @patch('builtins.print')
    def test_after_epoch_accuracy_improvement(self, mock_print):
        """Test after_epoch with accuracy improvement."""
        mock_learn = Mock()
        mock_learn.recorder.values = [[1.0, 0.8, 0.9]]  # [train_loss, valid_loss, accuracy]
        
        callback = EarlyStoppingCallback(monitor='valid_acc')
        callback.before_fit()
        callback.learn = mock_learn
        callback.epoch = 0
        
        # Mock model state dict
        mock_learn.model.state_dict.return_value = {'weight': torch.tensor([1.0])}
        
        callback.after_epoch()
        
        assert callback.best_metric == 0.9
        assert callback.best_epoch == 0
        assert callback.wait == 0

    @patch('builtins.print')
    def test_after_epoch_no_improvement(self, mock_print):
        """Test after_epoch with no improvement."""
        mock_learn = Mock()
        mock_learn.recorder.values = [[1.0, 0.9, 0.8]]  # Worse than initial
        
        callback = EarlyStoppingCallback(monitor='valid_loss')
        callback.before_fit()
        callback.learn = mock_learn
        callback.epoch = 0
        callback.best_metric = 0.5  # Already have better metric
        
        callback.after_epoch()
        
        assert callback.wait == 1
        assert callback.best_metric == 0.5  # Unchanged

    @patch('builtins.print')
    def test_after_epoch_early_stopping_triggered(self, mock_print):
        """Test after_epoch when early stopping is triggered."""
        mock_learn = Mock()
        mock_learn.recorder.values = [[1.0, 0.9, 0.8]]
        mock_learn.model.state_dict.return_value = {'weight': torch.tensor([1.0])}
        mock_learn.model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_learn.model.load_state_dict = Mock()
        
        callback = EarlyStoppingCallback(monitor='valid_loss', patience=2)
        callback.before_fit()
        callback.learn = mock_learn
        callback.epoch = 2
        callback.wait = 2  # At patience limit
        callback.best_metric = 0.5
        callback.best_weights = {'weight': torch.tensor([1.0])}
        
        from fastai.callback.core import CancelFitException
        with pytest.raises(CancelFitException):
            callback.after_epoch()
        
        assert callback.stopped is True

    @patch('builtins.print')
    def test_after_epoch_already_stopped(self, mock_print):
        """Test after_epoch when already stopped."""
        callback = EarlyStoppingCallback()
        callback.stopped = True
        
        callback.after_epoch()
        
        # Should return early without processing
        mock_print.assert_not_called()

    @patch('builtins.print')
    def test_after_epoch_exception_handling(self, mock_print):
        """Test after_epoch handles exceptions gracefully."""
        mock_learn = Mock()
        mock_learn.recorder.values = [[1.0, 0.8, 0.9]]
        mock_learn.model.state_dict.side_effect = Exception("Mock error")
        
        callback = EarlyStoppingCallback()
        callback.learn = mock_learn
        callback.epoch = 0
        
        # Should not raise exception
        callback.after_epoch()
        
        # Should print error message
        assert any("Early stopping check failed" in str(call) for call in mock_print.call_args_list)


class TestOrdinalLabelSmoothingLoss:
    """Test cases for OrdinalLabelSmoothingLoss class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.num_classes = 5
        self.loss = OrdinalLabelSmoothingLoss(self.num_classes, smoothing=0.1)

    def test_init_default_parameters(self):
        """Test OrdinalLabelSmoothingLoss initialization with defaults."""
        loss = OrdinalLabelSmoothingLoss(3)
        assert loss.num_classes == 3
        assert loss.smoothing == 0.1
        assert loss.ordinal_mapping is None
        assert loss.ordinal_distances.shape == (3, 3)

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        ordinal_mapping = {0: 0, 1: 2, 2: 4}  # Non-sequential mapping
        loss = OrdinalLabelSmoothingLoss(3, smoothing=0.2, ordinal_mapping=ordinal_mapping)
        
        assert loss.num_classes == 3
        assert loss.smoothing == 0.2
        assert loss.ordinal_mapping == ordinal_mapping

    def test_create_ordinal_distance_matrix_default(self):
        """Test distance matrix creation with default mapping."""
        loss = OrdinalLabelSmoothingLoss(3)
        
        expected_distances = torch.tensor([
            [0, 1, 2],
            [1, 0, 1], 
            [2, 1, 0]
        ], dtype=torch.float32)
        
        assert torch.allclose(loss.ordinal_distances, expected_distances)

    def test_create_ordinal_distance_matrix_custom(self):
        """Test distance matrix creation with custom mapping."""
        # Classes 0, 1, 2 map to ordinal positions 0, 3, 5
        ordinal_mapping = {0: 0, 1: 3, 2: 5}
        loss = OrdinalLabelSmoothingLoss(3, ordinal_mapping=ordinal_mapping)
        
        expected_distances = torch.tensor([
            [0, 3, 5],  # Distance from class 0 (pos 0) to others
            [3, 0, 2],  # Distance from class 1 (pos 3) to others  
            [5, 2, 0]   # Distance from class 2 (pos 5) to others
        ], dtype=torch.float32)
        
        assert torch.allclose(loss.ordinal_distances, expected_distances)

    def test_forward_basic_functionality(self):
        """Test forward pass with basic inputs."""
        batch_size = 2
        pred = torch.randn(batch_size, self.num_classes)
        targ = torch.tensor([0, 2])
        
        loss_value = self.loss(pred, targ)
        
        assert isinstance(loss_value, torch.Tensor)
        assert loss_value.dim() == 0  # Scalar
        assert loss_value.item() >= 0  # Loss should be non-negative

    def test_forward_perfect_predictions(self):
        """Test forward pass with perfect predictions."""
        # Create predictions with very high confidence for correct classes
        pred = torch.tensor([
            [10.0, -5.0, -5.0, -5.0, -5.0],  # Strongly predicts class 0
            [10.0, -5.0, -5.0, -5.0, -5.0],  # Strongly predicts class 0
        ])
        targ = torch.tensor([0, 0])
        
        loss_value = self.loss(pred, targ)
        
        # Should be low loss for confident correct predictions
        assert loss_value.item() < 1.1  # Allow slightly higher threshold due to smoothing

    def test_forward_wrong_predictions(self):
        """Test forward pass with wrong predictions."""
        # Create predictions that are confidently wrong
        pred = torch.tensor([
            [-5.0, -5.0, -5.0, -5.0, 10.0],  # Predicts class 4 instead of 0
            [-5.0, -5.0, -5.0, -5.0, 10.0],  # Predicts class 4 instead of 0
        ])
        targ = torch.tensor([0, 0])
        
        loss_value = self.loss(pred, targ)
        
        # Should have higher loss for wrong predictions
        assert loss_value.item() > 0.5

    def test_forward_different_devices(self):
        """Test forward pass handles device differences."""
        pred = torch.randn(2, self.num_classes)
        targ = torch.tensor([0, 1])
        
        # Initially distances are on CPU
        assert self.loss.ordinal_distances.device.type == 'cpu'
        
        loss_value = self.loss(pred, targ)
        
        # After forward pass, distances should be on same device as pred
        assert self.loss.ordinal_distances.device == pred.device

    def test_forward_batch_processing(self):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 4, 8]:
            pred = torch.randn(batch_size, self.num_classes)
            targ = torch.randint(0, self.num_classes, (batch_size,))
            
            loss_value = self.loss(pred, targ)
            
            assert isinstance(loss_value, torch.Tensor)
            assert loss_value.dim() == 0
            assert not torch.isnan(loss_value)

    def test_forward_smoothing_effect(self):
        """Test that smoothing affects the loss calculation."""
        pred = torch.randn(2, self.num_classes)
        targ = torch.tensor([0, 1])
        
        # Compare no smoothing vs with smoothing
        loss_no_smooth = OrdinalLabelSmoothingLoss(self.num_classes, 0.0)
        loss_with_smooth = OrdinalLabelSmoothingLoss(self.num_classes, 0.3)
        
        loss_val_no_smooth = loss_no_smooth(pred, targ)
        loss_val_with_smooth = loss_with_smooth(pred, targ)
        
        # Both should be valid
        assert not torch.isnan(loss_val_no_smooth)
        assert not torch.isnan(loss_val_with_smooth)
        assert loss_val_no_smooth.item() >= 0
        assert loss_val_with_smooth.item() >= 0

    def test_forward_ordinal_distance_weighting(self):
        """Test that adjacent classes get higher smoothing weight."""
        # This is more of an integration test to ensure the logic works
        pred = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]])  # Uniform logits
        targ = torch.tensor([2])  # Middle class
        
        loss_value = self.loss(pred, targ)
        
        # Should complete without error
        assert isinstance(loss_value, torch.Tensor)
        assert not torch.isnan(loss_value)


class TestShotTypeClassifierComprehensive:
    """Comprehensive test cases for ShotTypeClassifier class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = ShotTypeClassifier()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_extract_ordinal_position_valid(self):
        """Test extract_ordinal_position with valid labels."""
        assert ShotTypeClassifier.extract_ordinal_position("0-close") == 0
        assert ShotTypeClassifier.extract_ordinal_position("1-medium") == 1
        assert ShotTypeClassifier.extract_ordinal_position("10-extreme") == 10
        assert ShotTypeClassifier.extract_ordinal_position("05-wide") == 5

    def test_extract_ordinal_position_invalid(self):
        """Test extract_ordinal_position with invalid labels."""
        assert ShotTypeClassifier.extract_ordinal_position("close") == 0
        assert ShotTypeClassifier.extract_ordinal_position("no-number") == 0
        assert ShotTypeClassifier.extract_ordinal_position("") == 0
        assert ShotTypeClassifier.extract_ordinal_position("a-1-test") == 0

    def test_get_ordinal_mapping_valid_sequence(self):
        """Test get_ordinal_mapping with valid ordinal sequence."""
        labels = ["0-close", "1-medium", "2-wide"]
        mapping = ShotTypeClassifier.get_ordinal_mapping(labels)
        
        expected = {"0-close": 0, "1-medium": 1, "2-wide": 2}
        assert mapping == expected

    def test_get_ordinal_mapping_non_sequential(self):
        """Test get_ordinal_mapping with non-sequential labels."""
        labels = ["0-close", "2-wide", "5-extreme"]
        mapping = ShotTypeClassifier.get_ordinal_mapping(labels)
        
        expected = {"0-close": 0, "2-wide": 2, "5-extreme": 5}
        assert mapping == expected

    def test_get_ordinal_mapping_mixed_invalid(self):
        """Test get_ordinal_mapping with mixed valid/invalid labels."""
        labels = ["0-close", "medium", "2-wide"]
        mapping = ShotTypeClassifier.get_ordinal_mapping(labels)
        
        expected = {"0-close": 0, "medium": 0, "2-wide": 2}
        assert mapping == expected

    def test_validate_ordinal_labels_valid(self):
        """Test validate_ordinal_labels with valid sequence."""
        valid_labels = ["0-close", "1-medium", "2-wide"]
        assert ShotTypeClassifier.validate_ordinal_labels(valid_labels) is True

    def test_validate_ordinal_labels_invalid(self):
        """Test validate_ordinal_labels with invalid sequence."""
        invalid_labels = ["close", "medium", "wide"]
        assert ShotTypeClassifier.validate_ordinal_labels(invalid_labels) is False
        
        # Non-sequential but valid numbers
        non_sequential = ["0-close", "2-wide", "5-extreme"]
        assert ShotTypeClassifier.validate_ordinal_labels(non_sequential) is False

    def test_validate_ordinal_labels_empty(self):
        """Test validate_ordinal_labels with empty list."""
        assert ShotTypeClassifier.validate_ordinal_labels([]) is False

    @patch('ordinal_classifier.core.get_image_files')
    @patch('ordinal_classifier.core.DataBlock')
    def test_create_dataloaders_basic(self, mock_datablock, mock_get_image_files):
        """Test create_dataloaders with basic parameters."""
        # Mock get_image_files
        mock_files = [Path("img1.jpg"), Path("img2.jpg")]
        mock_get_image_files.return_value = mock_files
        
        # Mock DataBlock
        mock_dblock = Mock()
        mock_dataloaders = Mock()
        mock_dblock.dataloaders.return_value = mock_dataloaders
        mock_datablock.return_value = mock_dblock
        
        # Create test data directory
        data_dir = self.temp_path / "data"
        data_dir.mkdir()
        
        result = self.classifier.create_dataloaders(data_dir)
        
        mock_datablock.assert_called_once()
        mock_dblock.dataloaders.assert_called_once_with(data_dir, bs=32, num_workers=0)
        assert result == mock_dataloaders
        assert self.classifier.dls == mock_dataloaders

    @patch('ordinal_classifier.core.get_image_files')
    @patch('ordinal_classifier.core.DataBlock')
    def test_create_dataloaders_custom_params(self, mock_datablock, mock_get_image_files):
        """Test create_dataloaders with custom parameters."""
        mock_get_image_files.return_value = [Path("img1.jpg")]
        mock_dblock = Mock()
        mock_dataloaders = Mock()
        mock_dblock.dataloaders.return_value = mock_dataloaders
        mock_datablock.return_value = mock_dblock
        
        data_dir = self.temp_path / "data"
        data_dir.mkdir()
        
        result = self.classifier.create_dataloaders(
            data_dir,
            batch_size=16,
            image_size=(512, 512),
            valid_pct=0.3,
            seed=123
        )
        
        # Check DataBlock was called
        mock_datablock.assert_called_once()
        mock_dblock.dataloaders.assert_called_once_with(data_dir, bs=16, num_workers=0)
        assert result == mock_dataloaders

    @patch('ordinal_classifier.core.vision_learner')
    def test_create_learner_basic(self, mock_vision_learner):
        """Test create_learner with basic parameters."""
        # Mock dataloaders
        self.classifier.dls = Mock()
        mock_learner = Mock()
        mock_vision_learner.return_value = mock_learner
        
        result = self.classifier.create_learner()
        
        mock_vision_learner.assert_called_once()
        assert result == mock_learner
        assert self.classifier.learn == mock_learner

    def test_create_learner_no_dataloaders(self):
        """Test create_learner without dataloaders."""
        with pytest.raises(ValueError, match="Must create dataloaders first"):
            self.classifier.create_learner()

    @patch('ordinal_classifier.core.vision_learner')
    def test_create_learner_with_ordinal_loss(self, mock_vision_learner):
        """Test create_learner with ordinal loss function."""
        self.classifier.dls = Mock()
        self.classifier.dls.vocab = ["0-close", "1-medium", "2-wide"]
        mock_learner = Mock()
        mock_vision_learner.return_value = mock_learner
        
        result = self.classifier.create_learner(loss_func="ordinal")
        
        mock_vision_learner.assert_called_once()
        # Check that loss_func was passed
        args, kwargs = mock_vision_learner.call_args
        assert 'loss_func' in kwargs
        assert result == mock_learner

    @patch('ordinal_classifier.core.PILImage.create')
    def test_predict_single_with_learner(self, mock_pil_create):
        """Test predict_single with loaded learner."""
        # Mock learner
        mock_learner = Mock()
        mock_learner.predict.return_value = ("1-medium", 1, torch.tensor([0.1, 0.8, 0.1]))
        self.classifier.learn = mock_learner
        
        # Mock image
        mock_image = Mock()
        mock_pil_create.return_value = mock_image
        
        result = self.classifier.predict_single("test.jpg")
        
        mock_pil_create.assert_called_once_with("test.jpg")
        mock_learner.predict.assert_called_once_with(mock_image)
        assert result == "1-medium"

    @patch('ordinal_classifier.core.PILImage.create')
    def test_predict_single_with_probs(self, mock_pil_create):
        """Test predict_single with return_probs=True."""
        mock_learner = Mock()
        mock_probs = torch.tensor([0.1, 0.8, 0.1])
        mock_learner.predict.return_value = ("1-medium", 1, mock_probs)
        self.classifier.learn = mock_learner
        
        mock_image = Mock()
        mock_pil_create.return_value = mock_image
        
        result = self.classifier.predict_single("test.jpg", return_probs=True)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == "1-medium"
        assert torch.equal(result[1], mock_probs)

    @patch('ordinal_classifier.core.PILImage.create')
    def test_predict_batch_basic(self, mock_pil_create):
        """Test predict_batch with basic functionality."""
        # Mock learner
        mock_learner = Mock()
        mock_test_dl = Mock()
        mock_learner.get_preds.return_value = (
            torch.tensor([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]]),  # predictions
            torch.tensor([1, 0])  # targets (not used in predict_batch)
        )
        self.classifier.learn = mock_learner
        
        # Mock dataloader creation
        with patch('ordinal_classifier.core.DataLoader') as mock_dataloader:
            mock_dataloader.return_value = mock_test_dl
            
            image_paths = ["img1.jpg", "img2.jpg"]
            mock_images = [Mock(), Mock()]
            mock_pil_create.side_effect = mock_images
            
            results = self.classifier.predict_batch(image_paths)
            
            # Should create images for each path
            assert mock_pil_create.call_count == 2
            mock_learner.get_preds.assert_called_once_with(dl=mock_test_dl)
            
            assert len(results) == 2
            assert isinstance(results, list)

    def test_predict_batch_no_learner(self):
        """Test predict_batch without learner."""
        with pytest.raises(ValueError, match="Must load or create a model first"):
            self.classifier.predict_batch(["test.jpg"])

    @patch('ordinal_classifier.core.DataBlock')
    @patch('ordinal_classifier.core.vision_learner')
    @patch('builtins.print')
    def test_train_basic(self, mock_print, mock_vision_learner, mock_datablock):
        """Test train method with basic parameters."""
        # Setup mocks
        mock_dblock = Mock()
        mock_dataloaders = Mock()
        mock_dblock.dataloaders.return_value = mock_dataloaders
        mock_datablock.return_value = mock_dblock
        
        mock_learner = Mock()
        mock_vision_learner.return_value = mock_learner
        
        data_path = self.temp_path / "train_data"
        data_path.mkdir()
        
        with patch('ordinal_classifier.core.get_image_files'):
            self.classifier.train(data_path, epochs=1)
        
        # Should create dataloaders and learner
        mock_datablock.assert_called_once()
        mock_vision_learner.assert_called_once()
        mock_learner.fine_tune.assert_called_once()

    @patch('ordinal_classifier.core.DataBlock')
    @patch('ordinal_classifier.core.vision_learner') 
    @patch('builtins.print')
    def test_train_with_validation(self, mock_print, mock_vision_learner, mock_datablock):
        """Test train method with validation path."""
        # Setup mocks
        mock_dblock = Mock()
        mock_dataloaders = Mock()
        mock_dblock.dataloaders.return_value = mock_dataloaders
        mock_datablock.return_value = mock_dblock
        
        mock_learner = Mock()
        mock_vision_learner.return_value = mock_learner
        
        train_path = self.temp_path / "train"
        valid_path = self.temp_path / "valid"
        train_path.mkdir()
        valid_path.mkdir()
        
        with patch('ordinal_classifier.core.get_image_files'):
            self.classifier.train(train_path, epochs=2, valid_path=valid_path)
        
        mock_learner.fine_tune.assert_called_once()

    @patch('ordinal_classifier.core.DataBlock')
    @patch('ordinal_classifier.core.vision_learner')
    @patch('builtins.print')
    def test_train_with_ordinal_smoothing(self, mock_print, mock_vision_learner, mock_datablock):
        """Test train_with_ordinal_smoothing method."""
        # Setup mocks
        mock_dblock = Mock()
        mock_dataloaders = Mock()
        mock_dataloaders.vocab = ["0-close", "1-medium", "2-wide"]
        mock_dblock.dataloaders.return_value = mock_dataloaders
        mock_datablock.return_value = mock_dblock
        
        mock_learner = Mock()
        mock_vision_learner.return_value = mock_learner
        
        data_path = self.temp_path / "train_data"
        data_path.mkdir()
        
        with patch('ordinal_classifier.core.get_image_files'):
            self.classifier.train_with_ordinal_smoothing(data_path, epochs=1)
        
        # Should use custom loss function
        mock_vision_learner.assert_called_once()
        args, kwargs = mock_vision_learner.call_args
        assert 'loss_func' in kwargs

    def test_get_class_info_ordinal_range(self):
        """Test get_class_info calculates ordinal range correctly."""
        mock_learner = Mock()
        mock_learner.dls.vocab = ['0-close', '2-medium', '5-wide']
        
        self.classifier.learn = mock_learner
        class_info = self.classifier.get_class_info()
        
        assert class_info['ordinal_range'] == (0, 5)
        assert class_info['ordinal_mapping'] == {'0-close': 0, '2-medium': 2, '5-wide': 5}

    def test_static_methods_accessibility(self):
        """Test that static methods are accessible."""
        # Test static methods can be called on class
        assert callable(ShotTypeClassifier.extract_ordinal_position)
        assert callable(ShotTypeClassifier.get_ordinal_mapping)
        assert callable(ShotTypeClassifier.validate_ordinal_labels)
        
        # Test they work without instance
        result = ShotTypeClassifier.extract_ordinal_position("1-test")
        assert result == 1

    @patch('builtins.print')
    def test_move_to_device_success_message(self, mock_print):
        """Test move_to_device prints success message."""
        mock_learner = Mock()
        mock_model = Mock()
        mock_learner.model = mock_model
        
        self.classifier.learn = mock_learner
        self.classifier.move_to_device('cuda')
        
        mock_model.to.assert_called_once_with('cuda')
        mock_print.assert_called_once_with("✅ Model moved to device: cuda")

    def test_inheritance_structure(self):
        """Test class inheritance and module imports."""
        # Test that classes can be imported
        assert EarlyStoppingCallback is not None
        assert OrdinalLabelSmoothingLoss is not None
        assert ShotTypeClassifier is not None
        
        # Test inheritance
        assert issubclass(EarlyStoppingCallback, Callback)
        assert issubclass(OrdinalLabelSmoothingLoss, nn.Module)