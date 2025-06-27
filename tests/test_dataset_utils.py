"""Unit tests for ordinal_classifier.dataset_utils module."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, Mock
import hashlib
from collections import defaultdict

from ordinal_classifier.dataset_utils import (
    get_md5_hash,
    is_image_file,
    rename_to_md5,
    count_files_by_class,
    rebalance_dataset
)


class TestDatasetUtils:
    """Test cases for dataset utility functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_md5_hash(self):
        """Test MD5 hash calculation."""
        # Create a test file
        test_file = self.temp_path / "test.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)
        
        # Calculate expected MD5
        expected_md5 = hashlib.md5(test_content).hexdigest()
        
        # Test function
        result_md5 = get_md5_hash(test_file)
        assert result_md5 == expected_md5

    def test_is_image_file_valid_extensions(self):
        """Test is_image_file with valid image extensions."""
        valid_files = [
            Path("test.jpg"),
            Path("test.jpeg"),
            Path("test.png"),
            Path("test.JPG"),
            Path("test.JPEG"),
            Path("test.PNG")
        ]
        
        for file_path in valid_files:
            assert is_image_file(file_path)

    def test_is_image_file_invalid_extensions(self):
        """Test is_image_file with invalid extensions."""
        invalid_files = [
            Path("test.txt"),
            Path("test.pdf"),
            Path("test.doc"),
            Path("test.mp4"),
            Path("test")  # No extension
        ]
        
        for file_path in invalid_files:
            assert not is_image_file(file_path)

    def test_count_files_by_class_empty_directory(self):
        """Test count_files_by_class with empty directory."""
        counts = count_files_by_class(self.temp_path)
        assert counts == {}

    def test_count_files_by_class_with_files(self):
        """Test count_files_by_class with actual files."""
        # Create directory structure
        train_dir = self.temp_path / "train"
        valid_dir = self.temp_path / "valid"
        
        class1_train = train_dir / "class1"
        class1_valid = valid_dir / "class1"
        class2_train = train_dir / "class2"
        
        class1_train.mkdir(parents=True)
        class1_valid.mkdir(parents=True)
        class2_train.mkdir(parents=True)
        
        # Create image files
        (class1_train / "img1.jpg").touch()
        (class1_train / "img2.png").touch()
        (class1_valid / "img3.jpg").touch()
        (class2_train / "img4.jpeg").touch()
        
        # Create non-image file (should be ignored)
        (class1_train / "readme.txt").touch()
        
        counts = count_files_by_class(self.temp_path)
        
        assert counts["class1"]["train"] == 2
        assert counts["class1"]["valid"] == 1
        assert counts["class1"]["total"] == 3
        assert counts["class2"]["train"] == 1
        assert counts["class2"]["valid"] == 0
        assert counts["class2"]["total"] == 1

    def test_rename_to_md5_no_duplicates(self):
        """Test rename_to_md5 with no duplicate files."""
        # Create directory structure
        train_dir = self.temp_path / "train" / "class1"
        train_dir.mkdir(parents=True)
        
        # Create unique files with different content
        file1 = train_dir / "original1.jpg"
        file2 = train_dir / "original2.png"
        
        file1.write_bytes(b"content1")
        file2.write_bytes(b"content2")
        
        # Calculate expected hashes
        hash1 = hashlib.md5(b"content1").hexdigest()
        hash2 = hashlib.md5(b"content2").hexdigest()
        
        with patch('builtins.print'):
            renamed_count, deleted_count = rename_to_md5(self.temp_path)
        
        assert renamed_count == 2
        assert deleted_count == 0
        
        # Check that files were renamed correctly
        assert (train_dir / f"{hash1}.jpg").exists()
        assert (train_dir / f"{hash2}.jpg").exists()
        assert not file1.exists()
        assert not file2.exists()

    def test_rename_to_md5_with_duplicates(self):
        """Test rename_to_md5 with duplicate files."""
        # Create directory structure
        train_dir = self.temp_path / "train" / "class1"
        valid_dir = self.temp_path / "valid" / "class1"
        train_dir.mkdir(parents=True)
        valid_dir.mkdir(parents=True)
        
        # Create duplicate files with same content
        file1 = train_dir / "duplicate1.jpg"
        file2 = valid_dir / "duplicate2.png"
        
        same_content = b"identical content"
        file1.write_bytes(same_content)
        file2.write_bytes(same_content)
        
        # Calculate expected hash
        expected_hash = hashlib.md5(same_content).hexdigest()
        
        with patch('builtins.print'):
            renamed_count, deleted_count = rename_to_md5(self.temp_path)
        
        assert renamed_count == 1
        assert deleted_count == 1
        
        # Check that only one file remains with correct name
        renamed_files = list(self.temp_path.rglob(f"{expected_hash}.jpg"))
        assert len(renamed_files) == 1

    def test_rename_to_md5_file_processing_error(self):
        """Test rename_to_md5 handling file processing errors."""
        # Create directory structure
        train_dir = self.temp_path / "train" / "class1"
        train_dir.mkdir(parents=True)
        
        file1 = train_dir / "test.jpg"
        file1.write_bytes(b"test content")
        
        # Mock get_md5_hash to raise an exception
        with patch('ordinal_classifier.dataset_utils.get_md5_hash', side_effect=Exception("Read error")):
            with patch('builtins.print') as mock_print:
                renamed_count, deleted_count = rename_to_md5(self.temp_path)
        
        assert renamed_count == 0
        assert deleted_count == 0
        
        # Should print error message
        assert any("Could not process file" in str(call) for call in mock_print.call_args_list)

    def test_rebalance_dataset_basic(self):
        """Test basic dataset rebalancing."""
        # Create directory structure with unbalanced data
        train_dir = self.temp_path / "train" / "class1"
        valid_dir = self.temp_path / "valid" / "class1"
        train_dir.mkdir(parents=True)
        valid_dir.mkdir(parents=True)
        
        # Create files (all in train directory)
        for i in range(10):
            (train_dir / f"img{i}.jpg").touch()
        
        with patch('builtins.print'):
            final_counts = rebalance_dataset(self.temp_path, target_valid_ratio=0.2, seed=42)
        
        # Check final distribution
        assert final_counts["class1"]["train"] == 8  # 80% of 10
        assert final_counts["class1"]["valid"] == 2  # 20% of 10
        assert final_counts["class1"]["total"] == 10

    def test_rebalance_dataset_multiple_classes(self):
        """Test rebalancing with multiple classes."""
        # Create directory structure
        for class_name in ["class1", "class2"]:
            train_dir = self.temp_path / "train" / class_name
            valid_dir = self.temp_path / "valid" / class_name
            train_dir.mkdir(parents=True)
            valid_dir.mkdir(parents=True)
            
            # Create different numbers of files for each class
            num_files = 15 if class_name == "class1" else 25
            for i in range(num_files):
                (train_dir / f"img{i}.jpg").touch()
        
        with patch('builtins.print'):
            final_counts = rebalance_dataset(self.temp_path, target_valid_ratio=0.3, seed=42)
        
        # Check final distribution for class1 (15 files)
        assert final_counts["class1"]["train"] == 11  # 70% of 15, rounded
        assert final_counts["class1"]["valid"] == 4   # 30% of 15, rounded
        
        # Check final distribution for class2 (25 files)
        assert final_counts["class2"]["train"] == 18  # 70% of 25, rounded
        assert final_counts["class2"]["valid"] == 7   # 30% of 25, rounded

    def test_rebalance_dataset_minimum_valid_files(self):
        """Test rebalancing ensures at least 1 validation file per class."""
        # Create directory structure with very few files
        train_dir = self.temp_path / "train" / "small_class"
        train_dir.mkdir(parents=True)
        
        # Create only 2 files
        (train_dir / "img1.jpg").touch()
        (train_dir / "img2.jpg").touch()
        
        with patch('builtins.print'):
            final_counts = rebalance_dataset(self.temp_path, target_valid_ratio=0.1, seed=42)
        
        # Even with 10% ratio, should have at least 1 validation file
        assert final_counts["small_class"]["valid"] >= 1
        assert final_counts["small_class"]["train"] >= 1
        assert final_counts["small_class"]["total"] == 2

    def test_rebalance_dataset_seed_reproducibility(self):
        """Test that rebalancing is reproducible with same seed."""
        # Create directory structure
        train_dir = self.temp_path / "train" / "class1"
        train_dir.mkdir(parents=True)
        
        # Create files
        for i in range(10):
            (train_dir / f"img{i}.jpg").touch()
        
        # Run rebalancing twice with same seed
        with patch('builtins.print'):
            counts1 = rebalance_dataset(self.temp_path, target_valid_ratio=0.2, seed=123)
        
        # Reset files to train directory
        valid_dir = self.temp_path / "valid" / "class1"
        for file in valid_dir.glob("*.jpg"):
            shutil.move(str(file), str(train_dir / file.name))
        
        with patch('builtins.print'):
            counts2 = rebalance_dataset(self.temp_path, target_valid_ratio=0.2, seed=123)
        
        # Results should be identical
        assert counts1 == counts2

    def test_rebalance_dataset_no_files(self):
        """Test rebalancing with empty class directories."""
        # Create empty directory structure
        train_dir = self.temp_path / "train" / "empty_class"
        valid_dir = self.temp_path / "valid" / "empty_class"
        train_dir.mkdir(parents=True)
        valid_dir.mkdir(parents=True)
        
        with patch('builtins.print'):
            final_counts = rebalance_dataset(self.temp_path, target_valid_ratio=0.2, seed=42)
        
        # Should handle empty classes gracefully
        assert "empty_class" not in final_counts or final_counts["empty_class"]["total"] == 0

    def test_rebalance_dataset_custom_valid_ratio(self):
        """Test rebalancing with custom validation ratio."""
        # Create directory structure
        train_dir = self.temp_path / "train" / "class1"
        train_dir.mkdir(parents=True)
        
        # Create 20 files
        for i in range(20):
            (train_dir / f"img{i}.jpg").touch()
        
        with patch('builtins.print'):
            final_counts = rebalance_dataset(self.temp_path, target_valid_ratio=0.4, seed=42)
        
        # Check 60/40 split
        assert final_counts["class1"]["train"] == 12  # 60% of 20
        assert final_counts["class1"]["valid"] == 8   # 40% of 20
        assert final_counts["class1"]["total"] == 20
