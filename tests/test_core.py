"""Unit tests for ordinal_classifier.core module."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import pandas as pd
from fastai.vision.all import PILImage

from ordinal_classifier.core import ShotTypeClassifier


class TestShotTypeClassifier:
    """Test cases for ShotTypeClassifier class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = ShotTypeClassifier()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_default_model_path(self):
        """Test classifier initialization with default model path."""
        classifier = ShotTypeClassifier()
        assert classifier.model_path is None
        assert classifier.learn is None

    def test_init_custom_model_path(self):
        """Test classifier initialization with custom model path."""
        custom_path = Path("custom/model.pkl")
        classifier = ShotTypeClassifier(custom_path)
        assert classifier.model_path == custom_path
        assert classifier.learn is None

    @patch('ordinal_classifier.core.load_learner')
    def test_load_model_success(self, mock_load_learner):
        """Test successful model loading."""
        mock_learner = Mock()
        mock_load_learner.return_value = mock_learner
        
        # Create a dummy model file
        model_file = self.temp_path / "test_model.pkl"
        model_file.touch()
        
        classifier = ShotTypeClassifier(model_file)
        classifier.load_model()
        
        assert classifier.learn == mock_learner
        mock_load_learner.assert_called_once_with(model_file)

    def test_load_model_file_not_found(self):
        """Test model loading when file doesn't exist."""
        non_existent_path = self.temp_path / "non_existent.pkl"
        classifier = ShotTypeClassifier(non_existent_path)
        
        with pytest.raises(FileNotFoundError):
            classifier.load_model()

    def test_get_device_with_learner(self):
        """Test get_device when learner is loaded."""
        mock_learner = Mock()
        mock_model = Mock()
        mock_param = Mock()
        mock_param.device = torch.device('cpu')
        mock_model.parameters.return_value = iter([mock_param])
        mock_learner.model = mock_model
        
        classifier = ShotTypeClassifier()
        classifier.learn = mock_learner
        
        assert classifier.get_device() == 'cpu'

    def test_get_device_without_learner(self):
        """Test get_device when learner is not loaded."""
        classifier = ShotTypeClassifier()
        assert classifier.get_device() == 'cpu'

    def test_move_to_device_with_learner(self):
        """Test moving model to device when learner is loaded."""
        mock_learner = Mock()
        mock_model = Mock()
        mock_learner.model = mock_model
        
        classifier = ShotTypeClassifier()
        classifier.learn = mock_learner
        
        with patch('builtins.print'):
            classifier.move_to_device('cuda')
        mock_model.to.assert_called_once_with('cuda')

    def test_move_to_device_without_learner(self):
        """Test moving model to device when learner is not loaded."""
        classifier = ShotTypeClassifier()
        # Should raise an exception
        with pytest.raises(ValueError, match="Must load a model first"):
            classifier.move_to_device('cuda')

    def test_get_class_info_without_learner(self):
        """Test get_class_info when learner is not loaded."""
        classifier = ShotTypeClassifier()
        
        class_info = classifier.get_class_info()
        
        assert class_info == {}

    def test_get_class_info_with_learner(self):
        """Test get_class_info with loaded learner."""
        mock_learner = Mock()
        mock_learner.dls.vocab = ['0-close', '1-medium', '2-wide']
        
        classifier = ShotTypeClassifier()
        classifier.learn = mock_learner
        
        class_info = classifier.get_class_info()
        
        assert class_info['num_classes'] == 3
        assert class_info['classes'] == ['0-close', '1-medium', '2-wide']
        assert class_info['ordinal_mapping'] == {'0-close': 0, '1-medium': 1, '2-wide': 2}
        assert class_info['is_valid_ordinal_sequence']
        assert class_info['ordinal_range'] == (0, 2)

    def test_get_class_info_invalid_ordinal_sequence(self):
        """Test get_class_info with invalid ordinal sequence."""
        mock_learner = Mock()
        mock_learner.dls.vocab = ['close', 'medium', 'wide']
        
        classifier = ShotTypeClassifier()
        classifier.learn = mock_learner
        
        class_info = classifier.get_class_info()
        
        assert class_info['num_classes'] == 3
        assert class_info['classes'] == ['close', 'medium', 'wide']
        assert class_info['ordinal_mapping'] == {'close': 0, 'medium': 0, 'wide': 0}
        assert not class_info['is_valid_ordinal_sequence']
        assert class_info['ordinal_range'] == (0, 0)

    @patch('ordinal_classifier.core.PILImage.create')
    def test_predict_single_without_learner(self, mock_pil_create):
        """Test predict_single when learner is not loaded."""
        classifier = ShotTypeClassifier()
        
        with pytest.raises(ValueError, match="Must load or create a model first"):
            classifier.predict_single("test.jpg")
