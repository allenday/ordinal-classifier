"""Unit tests for ordinal_classifier.uncertainty module."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import torch
import hashlib

from ordinal_classifier.uncertainty import (
    calculate_md5,
    calculate_entropy,
    calculate_ordinal_uncertainty,
    find_uncertain_images
)


class TestUncertaintyFunctions:
    """Test cases for uncertainty calculation functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_calculate_md5(self):
        """Test MD5 hash calculation."""
        # Create a test file
        test_file = self.temp_path / "test.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)
        
        # Calculate expected MD5
        expected_md5 = hashlib.md5(test_content).hexdigest()
        
        # Test function
        result_md5 = calculate_md5(test_file)
        assert result_md5 == expected_md5

    def test_calculate_entropy_uniform(self):
        """Test entropy calculation for uniform distribution."""
        # Uniform distribution over 4 classes
        probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
        entropy = calculate_entropy(probs)
        
        # Entropy of uniform distribution is log2(n)
        expected_entropy = 2.0  # log2(4)
        assert torch.isclose(entropy, torch.tensor(expected_entropy), atol=1e-6)

    def test_calculate_entropy_certain(self):
        """Test entropy calculation for certain prediction."""
        # Certain prediction (all probability on one class)
        probs = torch.tensor([1.0, 0.0, 0.0, 0.0])
        entropy = calculate_entropy(probs)
        
        # Entropy should be 0 for certain prediction
        assert torch.isclose(entropy, torch.tensor(0.0), atol=1e-6)

    def test_calculate_entropy_batch(self):
        """Test entropy calculation for batch of predictions."""
        # Batch of 2 predictions
        probs = torch.tensor([
            [0.5, 0.5, 0.0, 0.0],  # 1 bit of entropy
            [1.0, 0.0, 0.0, 0.0]   # 0 bits of entropy
        ])
        entropies = calculate_entropy(probs)
        
        assert len(entropies) == 2
        assert torch.isclose(entropies[0], torch.tensor(1.0), atol=1e-6)
        assert torch.isclose(entropies[1], torch.tensor(0.0), atol=1e-6)

    def test_calculate_ordinal_uncertainty_with_mapping(self):
        """Test ordinal uncertainty with valid class mapping."""
        # Mock classifier with ordinal mapping
        mock_classifier = Mock()
        mock_classifier.get_class_info.return_value = {
            'classes': ['0-close', '1-medium', '2-wide'],
            'ordinal_mapping': {'0-close': 0, '1-medium': 1, '2-wide': 2}
        }
        
        # Prediction exactly between classes 0 and 1 (uncertainty = 0.5)
        probs = torch.tensor([0.5, 0.5, 0.0])
        uncertainty = calculate_ordinal_uncertainty(probs, mock_classifier)
        
        # Expected position: 0*0.5 + 1*0.5 = 0.5
        # Uncertainty:  < /dev/null | 0.5 - round(0.5)| = |0.5 - 1| = 0.5
        # Note: round(0.5) = 1 in Python
        assert abs(uncertainty - 0.5) < 1e-6

    def test_calculate_ordinal_uncertainty_certain(self):
        """Test ordinal uncertainty for certain prediction."""
        # Mock classifier
        mock_classifier = Mock()
        mock_classifier.get_class_info.return_value = {
            'classes': ['0-close', '1-medium', '2-wide'],
            'ordinal_mapping': {'0-close': 0, '1-medium': 1, '2-wide': 2}
        }
        
        # Certain prediction (all probability on class 1)
        probs = torch.tensor([0.0, 1.0, 0.0])
        uncertainty = calculate_ordinal_uncertainty(probs, mock_classifier)
        
        # Expected position: 1.0, uncertainty should be 0
        assert abs(uncertainty - 0.0) < 1e-6

    def test_calculate_ordinal_uncertainty_fallback(self):
        """Test ordinal uncertainty fallback when mapping is missing."""
        # Mock classifier without mapping
        mock_classifier = Mock()
        mock_classifier.get_class_info.return_value = {
            'classes': None,
            'ordinal_mapping': None
        }
        
        probs = torch.tensor([0.5, 0.5, 0.0])
        
        with patch('builtins.print') as mock_print:
            uncertainty = calculate_ordinal_uncertainty(probs, mock_classifier)
            mock_print.assert_called_once()
            assert "falling back to simple positional uncertainty" in mock_print.call_args[0][0]

    @patch('ordinal_classifier.uncertainty.PILImage.create')
    @patch('ordinal_classifier.uncertainty.calculate_md5')
    def test_find_uncertain_images_no_images(self, mock_md5, mock_pil_create):
        """Test find_uncertain_images when no images are found."""
        mock_classifier = Mock()
        
        with patch('builtins.print') as mock_print:
            find_uncertain_images(self.temp_path, mock_classifier, 'classification')
            
        # Should print "No images found"
        assert any("No images found" in str(call) for call in mock_print.call_args_list)

    @patch('ordinal_classifier.uncertainty.PILImage.create')
    @patch('ordinal_classifier.uncertainty.calculate_md5')
    def test_find_uncertain_images_classification(self, mock_md5, mock_pil_create):
        """Test find_uncertain_images for classification model."""
        # Create test images
        img1 = self.temp_path / "test1.jpg"
        img2 = self.temp_path / "test2.png"
        img1.touch()
        img2.touch()
        
        # Mock PIL Image creation
        mock_pil_create.return_value = Mock()
        
        # Mock MD5 calculation
        mock_md5.side_effect = ['abc123', 'def456']
        
        # Mock classifier
        mock_classifier = Mock()
        mock_classifier.get_device.return_value = 'cpu'
        mock_classifier.predict_single.side_effect = [
            ('class1', torch.tensor([0.7, 0.3])),  # Low entropy
            ('class2', torch.tensor([0.6, 0.4]))   # Higher entropy
        ]
        
        with patch('builtins.print'):
            find_uncertain_images(self.temp_path, mock_classifier, 'classification')
        
        # Check that files were processed
        assert mock_classifier.predict_single.call_count == 2

    @patch('ordinal_classifier.uncertainty.PILImage.create')
    @patch('ordinal_classifier.uncertainty.calculate_md5')
    def test_find_uncertain_images_ordinal(self, mock_md5, mock_pil_create):
        """Test find_uncertain_images for ordinal model."""
        # Create test image
        img1 = self.temp_path / "test1.jpg"
        img1.touch()
        
        # Mock PIL Image creation
        mock_pil_create.return_value = Mock()
        
        # Mock MD5 calculation
        mock_md5.return_value = 'abc123'
        
        # Mock classifier
        mock_classifier = Mock()
        mock_classifier.get_device.return_value = 'cpu'
        mock_classifier.predict_single.return_value = ('1-medium', torch.tensor([0.2, 0.6, 0.2]))
        mock_classifier.get_class_info.return_value = {
            'classes': ['0-close', '1-medium', '2-wide'],
            'ordinal_mapping': {'0-close': 0, '1-medium': 1, '2-wide': 2}
        }
        
        with patch('builtins.print'):
            find_uncertain_images(self.temp_path, mock_classifier, 'ordinal')
        
        # Check that ordinal uncertainty was calculated
        assert mock_classifier.predict_single.call_count == 1

    @patch('ordinal_classifier.uncertainty.PILImage.create')
    def test_find_uncertain_images_corrupt_file(self, mock_pil_create):
        """Test find_uncertain_images with corrupt image file."""
        # Create test image
        img1 = self.temp_path / "corrupt.jpg"
        img1.touch()
        
        # Mock PIL Image to raise exception for corrupt file
        mock_pil_create.side_effect = Exception("Corrupt image")
        
        mock_classifier = Mock()
        mock_classifier.get_device.return_value = 'cpu'
        
        with patch('builtins.print') as mock_print:
            find_uncertain_images(self.temp_path, mock_classifier, 'classification')
        
        # Should print message about skipping corrupt file
        assert any("Skipping corrupt" in str(call) for call in mock_print.call_args_list)

    @patch('ordinal_classifier.uncertainty.PILImage.create')
    @patch('ordinal_classifier.uncertainty.calculate_md5')
    def test_find_uncertain_images_prediction_error(self, mock_md5, mock_pil_create):
        """Test find_uncertain_images when prediction fails."""
        # Create test image
        img1 = self.temp_path / "test1.jpg"
        img1.touch()
        
        # Mock PIL Image creation
        mock_pil_create.return_value = Mock()
        mock_md5.return_value = 'abc123'
        
        # Mock classifier with prediction error
        mock_classifier = Mock()
        mock_classifier.get_device.return_value = 'cpu'
        mock_classifier.predict_single.side_effect = Exception("Prediction failed")
        
        with patch('builtins.print') as mock_print:
            find_uncertain_images(self.temp_path, mock_classifier, 'classification')
        
        # Should print error message
        assert any("Error processing" in str(call) for call in mock_print.call_args_list)

    @patch('ordinal_classifier.uncertainty.PILImage.create')
    @patch('ordinal_classifier.uncertainty.calculate_md5')
    def test_find_uncertain_images_file_renaming(self, mock_md5, mock_pil_create):
        """Test find_uncertain_images file renaming functionality."""
        # Create test image
        img1 = self.temp_path / "original.jpg"
        img1.touch()
        
        # Mock PIL Image creation
        mock_pil_create.return_value = Mock()
        mock_md5.return_value = 'abc123def'
        
        # Mock classifier
        mock_classifier = Mock()
        mock_classifier.get_device.return_value = 'cpu'
        mock_classifier.predict_single.return_value = ('class1', torch.tensor([0.8, 0.2]))
        
        with patch('builtins.print'):
            find_uncertain_images(self.temp_path, mock_classifier, 'classification')
        
        # Check that the original file no longer exists (was renamed)
        assert not img1.exists()
        
        # Check that a file with the new naming pattern exists
        renamed_files = list(self.temp_path.glob("uncertainty_*_pred-*_conf-*_abc123def.jpg"))
        assert len(renamed_files) == 1
