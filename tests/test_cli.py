"""Unit tests for ordinal_classifier.cli module."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner
import torch

from ordinal_classifier.cli import (
    main,
    select_device,
    ARCHITECTURES
)


class TestCLI:
    """Test cases for CLI functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.runner = CliRunner()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_select_device_auto_mps_available(self):
        """Test auto device selection when MPS is available."""
        with patch('torch.backends.mps.is_available', return_value=True):
            device = select_device('auto')
            assert device == 'mps'

    def test_select_device_auto_cuda_available(self):
        """Test auto device selection when CUDA is available but not MPS."""
        with patch('torch.backends.mps.is_available', return_value=False):
            with patch('torch.cuda.is_available', return_value=True):
                device = select_device('auto')
                assert device == 'cuda'

    def test_select_device_auto_cpu_fallback(self):
        """Test auto device selection fallback to CPU."""
        with patch('torch.backends.mps.is_available', return_value=False):
            with patch('torch.cuda.is_available', return_value=False):
                device = select_device('auto')
                assert device == 'cpu'

    def test_select_device_cuda_not_available(self):
        """Test CUDA selection when not available."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('click.echo') as mock_echo:
                device = select_device('cuda')
                assert device == 'cpu'
                mock_echo.assert_called_once()

    def test_select_device_mps_not_available(self):
        """Test MPS selection when not available."""
        with patch('torch.backends.mps.is_available', return_value=False):
            with patch('click.echo') as mock_echo:
                device = select_device('mps')
                assert device == 'cpu'
                mock_echo.assert_called_once()

    def test_select_device_explicit_cpu(self):
        """Test explicit CPU device selection."""
        device = select_device('cpu')
        assert device == 'cpu'

    def test_main_command_help(self):
        """Test main command help output."""
        result = self.runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'Shot Type Classifier' in result.output or 'Ordinal Classifier' in result.output

    def test_main_command_version(self):
        """Test version command."""
        result = self.runner.invoke(main, ['--version'])
        assert result.exit_code == 0
        assert '2.0.0' in result.output

    def test_info_command_no_model(self):
        """Test info command when no model exists."""
        result = self.runner.invoke(main, ['info'])
        assert result.exit_code == 0
        assert 'No model found' in result.output

    @patch('ordinal_classifier.cli.ShotTypeClassifier')
    def test_info_command_with_model(self, mock_classifier_class):
        """Test info command with existing model."""
        # Create mock model file
        model_file = self.temp_path / "test_model.pkl"
        model_file.touch()
        
        # Mock classifier
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = None
        mock_classifier.get_class_info.return_value = {
            'num_classes': 3,
            'classes': ['0-close', '1-medium', '2-wide'],
            'ordinal_mapping': {'0-close': 0, '1-medium': 1, '2-wide': 2},
            'is_valid_ordinal_sequence': True,
            'ordinal_range': (0, 2)
        }
        mock_classifier_class.return_value = mock_classifier
        
        result = self.runner.invoke(main, ['info', '--model-path', str(model_file)])
        assert result.exit_code == 0
        assert 'Model information' in result.output
        assert '3' in result.output  # Number of classes

    def test_train_command_help(self):
        """Test train command help."""
        result = self.runner.invoke(main, ['train', '--help'])
        assert result.exit_code == 0
        assert 'Train a shot type classifier' in result.output

    def test_predict_command_help(self):
        """Test predict command help."""
        result = self.runner.invoke(main, ['predict', '--help'])
        assert result.exit_code == 0
        assert 'Predict shot types for images' in result.output

    def test_evaluate_command_help(self):
        """Test evaluate command help."""
        result = self.runner.invoke(main, ['evaluate', '--help'])
        assert result.exit_code == 0
        assert 'Evaluate model performance' in result.output

    def test_heatmap_command_help(self):
        """Test heatmap command help."""
        result = self.runner.invoke(main, ['heatmap', '--help'])
        assert result.exit_code == 0
        assert 'Generate activation heatmaps' in result.output

    def test_find_uncertain_command_help(self):
        """Test find-uncertain command help."""
        result = self.runner.invoke(main, ['find-uncertain', '--help'])
        assert result.exit_code == 0
        assert 'Find the most uncertain images' in result.output

    def test_rebalance_command_help(self):
        """Test rebalance command help."""
        result = self.runner.invoke(main, ['rebalance', '--help'])
        assert result.exit_code == 0
        assert 'Rebalance dataset' in result.output

    def test_train_command_missing_data_path(self):
        """Test train command with missing data path."""
        result = self.runner.invoke(main, ['train'])
        assert result.exit_code != 0
        assert 'Missing argument' in result.output

    def test_train_command_invalid_image_size(self):
        """Test train command with invalid image size format."""
        # Create dummy data directory
        data_dir = self.temp_path / "data"
        data_dir.mkdir()
        
        result = self.runner.invoke(main, [
            'train', str(data_dir), 
            '--image-size', 'invalid'
        ])
        assert result.exit_code != 0
        assert 'Image size must be in format' in result.output

    @patch('ordinal_classifier.cli.ShotTypeClassifier')
    def test_predict_command_missing_model(self, mock_classifier_class):
        """Test predict command when model loading fails."""
        # Create test image
        test_img = self.temp_path / "test.jpg"
        test_img.touch()
        
        # Mock classifier to raise exception on load
        mock_classifier = Mock()
        mock_classifier.load_model.side_effect = Exception("Model not found")
        mock_classifier_class.return_value = mock_classifier
        
        result = self.runner.invoke(main, ['predict', str(test_img)])
        assert result.exit_code != 0
        assert 'Failed to load model' in result.output

    @patch('ordinal_classifier.cli.ShotTypeClassifier')
    def test_predict_single_image_success(self, mock_classifier_class):
        """Test successful single image prediction."""
        # Create test image
        test_img = self.temp_path / "test.jpg"
        test_img.touch()
        
        # Mock classifier
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = None
        mock_classifier.move_to_device.return_value = None
        mock_classifier.predict_single.return_value = 'test_class'
        mock_classifier_class.return_value = mock_classifier
        
        with patch('ordinal_classifier.cli.select_device', return_value='cpu'):
            result = self.runner.invoke(main, ['predict', str(test_img)])
        
        assert result.exit_code == 0
        assert 'test_class' in result.output

    @patch('ordinal_classifier.cli.ShotTypeClassifier')
    def test_predict_with_probabilities(self, mock_classifier_class):
        """Test prediction with probability display."""
        # Create test image
        test_img = self.temp_path / "test.jpg"
        test_img.touch()
        
        # Mock classifier
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = None
        mock_classifier.move_to_device.return_value = None
        mock_classifier.predict_single.return_value = ('class1', torch.tensor([0.7, 0.3]))
        
        # Mock learner with vocab
        mock_learn = Mock()
        mock_learn.dls.vocab = ['class1', 'class2']
        mock_classifier.learn = mock_learn
        mock_classifier_class.return_value = mock_classifier
        
        with patch('ordinal_classifier.cli.select_device', return_value='cpu'):
            result = self.runner.invoke(main, [
                'predict', str(test_img), 
                '--show-probabilities'
            ])
        
        assert result.exit_code == 0
        assert 'Probabilities' in result.output
        assert 'class1' in result.output

    @patch('ordinal_classifier.cli.find_uncertain_images')
    @patch('ordinal_classifier.cli.ShotTypeClassifier')
    def test_find_uncertain_command(self, mock_classifier_class, mock_find_uncertain):
        """Test find-uncertain command."""
        # Create test directory
        img_dir = self.temp_path / "images"
        img_dir.mkdir()
        
        # Mock classifier
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = None
        mock_classifier.move_to_device.return_value = None
        mock_classifier_class.return_value = mock_classifier
        
        # Mock find_uncertain_images function
        mock_find_uncertain.return_value = None
        
        with patch('ordinal_classifier.cli.select_device', return_value='cpu'):
            result = self.runner.invoke(main, ['find-uncertain', str(img_dir)])
        
        assert result.exit_code == 0
        mock_find_uncertain.assert_called_once()

    @patch('ordinal_classifier.cli.rebalance_dataset')
    @patch('ordinal_classifier.cli.rename_to_md5')
    def test_rebalance_command(self, mock_rename, mock_rebalance):
        """Test rebalance command."""
        # Create test data directory
        data_dir = self.temp_path / "data"
        data_dir.mkdir()
        
        # Mock functions
        mock_rename.return_value = None
        mock_rebalance.return_value = None
        
        result = self.runner.invoke(main, ['rebalance', str(data_dir)])
        
        assert result.exit_code == 0
        mock_rename.assert_called_once()
        mock_rebalance.assert_called_once()

    @patch('ordinal_classifier.cli.rebalance_dataset')
    def test_rebalance_command_skip_rename(self, mock_rebalance):
        """Test rebalance command with skip-rename option."""
        # Create test data directory
        data_dir = self.temp_path / "data"
        data_dir.mkdir()
        
        # Mock rebalance function
        mock_rebalance.return_value = None
        
        result = self.runner.invoke(main, [
            'rebalance', str(data_dir), '--skip-rename'
        ])
        
        assert result.exit_code == 0
        mock_rebalance.assert_called_once()

    def test_rebalance_command_invalid_ratio(self):
        """Test rebalance command with invalid validation ratio."""
        # Create test data directory
        data_dir = self.temp_path / "data"
        data_dir.mkdir()
        
        result = self.runner.invoke(main, [
            'rebalance', str(data_dir), '--valid-ratio', '1.5'
        ])
        
        assert result.exit_code != 0
        assert 'Validation ratio must be between 0 and 1' in result.output

    def test_heatmap_command_invalid_alpha(self):
        """Test heatmap command with invalid alpha value."""
        # Create test image
        test_img = self.temp_path / "test.jpg"
        test_img.touch()
        
        output_dir = self.temp_path / "output"
        
        result = self.runner.invoke(main, [
            'heatmap', str(test_img), str(output_dir),
            '--alpha', '1.5'
        ])
        
        assert result.exit_code != 0
        assert 'Alpha must be between 0.0 and 1.0' in result.output

    def test_architectures_available(self):
        """Test that all expected architectures are available."""
        expected_archs = [
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            'efficientnet_b0', 'efficientnet_b3', 'efficientnet_b5'
        ]
        
        for arch in expected_archs:
            assert arch in ARCHITECTURES
            assert callable(ARCHITECTURES[arch])

    @patch('ordinal_classifier.cli.ShotTypeClassifier')
    def test_evaluate_command_no_save(self, mock_classifier_class):
        """Test evaluate command with no-save option."""
        # Create test data directory
        data_dir = self.temp_path / "data"
        data_dir.mkdir()
        
        # Mock classifier and evaluator
        mock_classifier = Mock()
        mock_classifier.load_model.return_value = None
        mock_classifier.move_to_device.return_value = None
        mock_classifier_class.return_value = mock_classifier
        
        with patch('ordinal_classifier.cli.ModelEvaluator') as mock_evaluator_class:
            mock_evaluator = Mock()
            mock_evaluator.evaluate_directory.return_value = {'accuracy': 0.9}
            mock_evaluator.print_summary.return_value = None
            mock_evaluator_class.return_value = mock_evaluator
            
            with patch('ordinal_classifier.cli.select_device', return_value='cpu'):
                result = self.runner.invoke(main, [
                    'evaluate', str(data_dir), '--no-save'
                ])
        
        assert result.exit_code == 0

    def test_train_command_validation(self):
        """Test train command parameter validation."""
        # Test with non-existent data path
        result = self.runner.invoke(main, ['train', '/non/existent/path'])
        assert result.exit_code != 0
        assert 'does not exist' in result.output or 'Invalid value' in result.output
