"""Unit tests for ordinal_classifier.transforms module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from ordinal_classifier.transforms import (
    get_extra_transforms,
    get_transforms,
    create_data_transforms
)


class TestTransforms:
    """Test cases for transforms module."""

    def test_get_extra_transforms_default(self):
        """Test get_extra_transforms with default parameters."""
        transforms = get_extra_transforms()
        assert isinstance(transforms, list)
        assert len(transforms) == 0  # Returns empty list in v2

    def test_get_extra_transforms_custom_size(self):
        """Test get_extra_transforms with custom base size."""
        transforms = get_extra_transforms(base_size=512)
        assert isinstance(transforms, list)
        assert len(transforms) == 0  # Returns empty list in v2

    @patch('ordinal_classifier.transforms.aug_transforms')
    @patch('ordinal_classifier.transforms.Normalize')
    def test_get_transforms_default(self, mock_normalize, mock_aug_transforms):
        """Test get_transforms with default parameters."""
        # Mock aug_transforms to return a list
        mock_aug_list = [Mock(), Mock()]
        mock_aug_transforms.return_value = mock_aug_list
        
        # Mock Normalize.from_stats
        mock_normalize_instance = Mock()
        mock_normalize.from_stats.return_value = mock_normalize_instance
        
        transforms = get_transforms()
        
        # Should call aug_transforms with defaults
        mock_aug_transforms.assert_called_once_with(
            do_flip=True,
            flip_vert=False,
            max_zoom=1.0,
            max_lighting=0.4,
            max_warp=0.3,
            p_affine=0.85,
            p_lighting=0.85
        )
        
        # Should call Normalize.from_stats
        mock_normalize.from_stats.assert_called_once()
        
        # Should return list with aug_transforms + normalize
        assert isinstance(transforms, list)
        assert len(transforms) == len(mock_aug_list) + 1

    @patch('ordinal_classifier.transforms.aug_transforms')
    @patch('ordinal_classifier.transforms.Normalize')
    def test_get_transforms_custom_params(self, mock_normalize, mock_aug_transforms):
        """Test get_transforms with custom parameters."""
        mock_aug_list = [Mock()]
        mock_aug_transforms.return_value = mock_aug_list
        mock_normalize.from_stats.return_value = Mock()
        
        transforms = get_transforms(
            do_flip=False,
            flip_vert=True,
            max_zoom=1.5,
            max_lighting=0.2,
            max_warp=0.1,
            p_affine=0.5,
            p_lighting=0.3,
            image_size=(224, 224)
        )
        
        mock_aug_transforms.assert_called_once_with(
            do_flip=False,
            flip_vert=True,
            max_zoom=1.5,
            max_lighting=0.2,
            max_warp=0.1,
            p_affine=0.5,
            p_lighting=0.3
        )
        
        assert isinstance(transforms, list)

    @patch('ordinal_classifier.transforms.Resize')
    @patch('ordinal_classifier.transforms.get_transforms')
    def test_create_data_transforms_default(self, mock_get_transforms, mock_resize):
        """Test create_data_transforms with default parameters."""
        # Mock the resize transform
        mock_resize_instance = Mock()
        mock_resize.return_value = mock_resize_instance
        
        # Mock get_transforms
        mock_batch_tfms = [Mock(), Mock()]
        mock_get_transforms.return_value = mock_batch_tfms
        
        result = create_data_transforms()
        
        # Should create resize transform with default size
        mock_resize.assert_called_once()
        
        # Should call get_transforms with default image size
        mock_get_transforms.assert_called_once_with(image_size=(375, 666))
        
        # Should return dict with item_tfms and batch_tfms
        assert isinstance(result, dict)
        assert 'item_tfms' in result
        assert 'batch_tfms' in result
        assert result['item_tfms'] == mock_resize_instance
        assert result['batch_tfms'] == mock_batch_tfms

    @patch('ordinal_classifier.transforms.Resize')
    @patch('ordinal_classifier.transforms.get_transforms')
    def test_create_data_transforms_custom_size(self, mock_get_transforms, mock_resize):
        """Test create_data_transforms with custom image size."""
        mock_resize_instance = Mock()
        mock_resize.return_value = mock_resize_instance
        mock_batch_tfms = [Mock()]
        mock_get_transforms.return_value = mock_batch_tfms
        
        custom_size = (512, 512)
        result = create_data_transforms(image_size=custom_size)
        
        # Should create resize transform with custom size
        mock_resize.assert_called_once()
        
        # Should call get_transforms with custom image size
        mock_get_transforms.assert_called_once_with(image_size=custom_size)
        
        assert isinstance(result, dict)
        assert 'item_tfms' in result
        assert 'batch_tfms' in result

    def test_get_transforms_parameter_types(self):
        """Test that get_transforms accepts correct parameter types."""
        # Test with various parameter combinations to ensure type compatibility
        with patch('ordinal_classifier.transforms.aug_transforms') as mock_aug:
            with patch('ordinal_classifier.transforms.Normalize') as mock_norm:
                mock_aug.return_value = []
                mock_norm.from_stats.return_value = Mock()
                
                # Test boolean parameters
                transforms = get_transforms(do_flip=True, flip_vert=False)
                assert isinstance(transforms, list)
                
                # Test float parameters
                transforms = get_transforms(max_zoom=1.2, max_lighting=0.5)
                assert isinstance(transforms, list)
                
                # Test tuple parameter
                transforms = get_transforms(image_size=(224, 224))
                assert isinstance(transforms, list)

    def test_module_imports(self):
        """Test that all required imports are available."""
        # Test that the module can import required functions
        from ordinal_classifier.transforms import get_extra_transforms
        from ordinal_classifier.transforms import get_transforms  
        from ordinal_classifier.transforms import create_data_transforms
        
        # Test that functions are callable
        assert callable(get_extra_transforms)
        assert callable(get_transforms)
        assert callable(create_data_transforms)

    @patch('ordinal_classifier.transforms.imagenet_stats', (Mock(), Mock()))
    @patch('ordinal_classifier.transforms.Normalize')
    @patch('ordinal_classifier.transforms.aug_transforms')
    def test_imagenet_stats_usage(self, mock_aug_transforms, mock_normalize):
        """Test that imagenet_stats are used correctly."""
        mock_aug_transforms.return_value = []
        mock_norm_instance = Mock()
        mock_normalize.from_stats.return_value = mock_norm_instance
        
        get_transforms()
        
        # Should be called with unpacked imagenet_stats
        mock_normalize.from_stats.assert_called_once()

    def test_create_data_transforms_return_structure(self):
        """Test the exact structure of create_data_transforms return value."""
        with patch('ordinal_classifier.transforms.Resize') as mock_resize:
            with patch('ordinal_classifier.transforms.get_transforms') as mock_get_transforms:
                mock_resize.return_value = "resize_transform"
                mock_get_transforms.return_value = ["batch_transform1", "batch_transform2"]
                
                result = create_data_transforms()
                
                expected = {
                    'item_tfms': "resize_transform",
                    'batch_tfms': ["batch_transform1", "batch_transform2"]
                }
                
                assert result == expected