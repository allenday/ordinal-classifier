"""Comprehensive unit tests for ordinal_classifier.cli module."""

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


class TestSelectDevice:
    """Test cases for select_device function."""

    @patch('ordinal_classifier.cli.torch.backends.mps.is_available')
    @patch('ordinal_classifier.cli.torch.cuda.is_available')
    def test_select_device_auto_mps_available(self, mock_cuda, mock_mps):
        """Test auto device selection when MPS is available."""
        mock_mps.return_value = True
        mock_cuda.return_value = False
        
        result = select_device('auto')
        assert result == 'mps'

    @patch('ordinal_classifier.cli.torch.backends.mps.is_available')
    @patch('ordinal_classifier.cli.torch.cuda.is_available')
    def test_select_device_auto_cuda_available(self, mock_cuda, mock_mps):
        """Test auto device selection when CUDA is available but MPS is not."""
        mock_mps.return_value = False
        mock_cuda.return_value = True
        
        result = select_device('auto')
        assert result == 'cuda'

    @patch('ordinal_classifier.cli.torch.backends.mps.is_available')
    @patch('ordinal_classifier.cli.torch.cuda.is_available')
    def test_select_device_auto_cpu_fallback(self, mock_cuda, mock_mps):
        """Test auto device selection falls back to CPU."""
        mock_mps.return_value = False
        mock_cuda.return_value = False
        
        result = select_device('auto')
        assert result == 'cpu'

    @patch('ordinal_classifier.cli.torch.cuda.is_available')
    @patch('ordinal_classifier.cli.click.echo')
    def test_select_device_cuda_not_available(self, mock_echo, mock_cuda):
        """Test CUDA selection when not available."""
        mock_cuda.return_value = False
        
        result = select_device('cuda')
        assert result == 'cpu'
        mock_echo.assert_called_once()

    @patch('ordinal_classifier.cli.torch.backends.mps.is_available')
    @patch('ordinal_classifier.cli.click.echo')
    def test_select_device_mps_not_available(self, mock_echo, mock_mps):
        """Test MPS selection when not available."""
        mock_mps.return_value = False
        
        result = select_device('mps')
        assert result == 'cpu'
        mock_echo.assert_called_once()

    def test_select_device_explicit_cpu(self):
        """Test explicit CPU selection."""
        result = select_device('cpu')
        assert result == 'cpu'

    @patch('ordinal_classifier.cli.torch.cuda.is_available')
    def test_select_device_cuda_available(self, mock_cuda):
        """Test CUDA selection when available."""
        mock_cuda.return_value = True
        
        result = select_device('cuda')
        assert result == 'cuda'

    @patch('ordinal_classifier.cli.torch.backends.mps.is_available')
    def test_select_device_mps_available(self, mock_mps):
        """Test MPS selection when available."""
        mock_mps.return_value = True
        
        result = select_device('mps')
        assert result == 'mps'


class TestArchitectures:
    """Test cases for architecture constants."""

    def test_architectures_dict_exists(self):
        """Test that ARCHITECTURES constant exists and is populated."""
        assert isinstance(ARCHITECTURES, dict)
        assert len(ARCHITECTURES) > 0

    def test_architectures_contains_resnets(self):
        """Test that ARCHITECTURES contains ResNet variants."""
        expected_resnets = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        for resnet in expected_resnets:
            assert resnet in ARCHITECTURES

    def test_architectures_contains_efficientnets(self):
        """Test that ARCHITECTURES contains EfficientNet variants."""
        expected_efficientnets = ['efficientnet_b0', 'efficientnet_b3', 'efficientnet_b5']
        for efficientnet in expected_efficientnets:
            assert efficientnet in ARCHITECTURES

    def test_architectures_values_are_callable(self):
        """Test that all architecture values are callable."""
        for arch_name, arch_func in ARCHITECTURES.items():
            assert callable(arch_func), f"Architecture {arch_name} is not callable"


class TestMainCommand:
    """Test cases for main CLI group."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_main_command_help(self):
        """Test main command help."""
        result = self.runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'Ordinal Classifier' in result.output

    def test_main_command_version(self):
        """Test main command version."""
        result = self.runner.invoke(main, ['--version'])
        assert result.exit_code == 0
        assert '2.0.0' in result.output

    def test_main_command_no_args(self):
        """Test main command with no arguments."""
        result = self.runner.invoke(main, [])
        # Main command should either show help or succeed
        assert result.exit_code == 0 or 'Usage:' in result.output


class TestTrainCommand:
    """Test cases for train command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_train_command_help(self):
        """Test train command help."""
        result = self.runner.invoke(main, ['train', '--help'])
        assert result.exit_code == 0
        assert 'data_path' in result.output.lower()

    def test_train_command_missing_data_path(self):
        """Test train command with missing data path."""
        result = self.runner.invoke(main, ['train'])
        assert result.exit_code != 0

    def test_train_command_invalid_data_path(self):
        """Test train command with invalid data path."""
        result = self.runner.invoke(main, ['train', '/nonexistent/path'])
        assert result.exit_code != 0

    def test_train_command_architecture_choices(self):
        """Test that train command accepts valid architectures."""
        # Test with invalid architecture
        result = self.runner.invoke(main, ['train', str(self.temp_path), '--arch', 'invalid_arch'])
        assert result.exit_code != 0

    def test_train_command_scheduler_choices(self):
        """Test that train command accepts valid schedulers."""
        # Test with invalid scheduler
        result = self.runner.invoke(main, ['train', str(self.temp_path), '--scheduler', 'invalid_scheduler'])
        assert result.exit_code != 0

    def test_train_command_device_choices(self):
        """Test that train command accepts valid devices."""
        # Test with invalid device
        result = self.runner.invoke(main, ['train', str(self.temp_path), '--device', 'invalid_device'])
        assert result.exit_code != 0


class TestCommandHelp:
    """Test cases for command help messages."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_predict_command_help(self):
        """Test predict command help."""
        result = self.runner.invoke(main, ['predict', '--help'])
        assert result.exit_code == 0
        assert 'input_path' in result.output.lower()

    def test_evaluate_command_help(self):
        """Test evaluate command help."""
        result = self.runner.invoke(main, ['evaluate', '--help'])
        assert result.exit_code == 0
        assert 'data_path' in result.output.lower()

    def test_heatmap_command_help(self):
        """Test heatmap command help."""
        result = self.runner.invoke(main, ['heatmap', '--help'])
        assert result.exit_code == 0
        assert 'input_path' in result.output.lower()

    def test_info_command_help(self):
        """Test info command help."""
        result = self.runner.invoke(main, ['info', '--help'])
        assert result.exit_code == 0

    def test_find_uncertain_command_help(self):
        """Test find-uncertain command help."""
        result = self.runner.invoke(main, ['find-uncertain', '--help'])
        assert result.exit_code == 0
        assert 'image_dir' in result.output.lower()

    def test_rebalance_command_help(self):
        """Test rebalance command help."""
        result = self.runner.invoke(main, ['rebalance', '--help'])
        assert result.exit_code == 0
        assert 'dataset_dir' in result.output.lower() or 'data_dir' in result.output.lower() or 'data-dir' in result.output.lower()


class TestCommandArguments:
    """Test cases for command argument validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_predict_command_missing_args(self):
        """Test predict command with missing arguments."""
        result = self.runner.invoke(main, ['predict'])
        assert result.exit_code != 0

    def test_evaluate_command_missing_args(self):
        """Test evaluate command with missing arguments."""
        result = self.runner.invoke(main, ['evaluate'])
        assert result.exit_code != 0

    def test_heatmap_command_missing_args(self):
        """Test heatmap command with missing arguments."""
        result = self.runner.invoke(main, ['heatmap'])
        assert result.exit_code != 0

    def test_find_uncertain_command_missing_args(self):
        """Test find-uncertain command with missing arguments."""
        result = self.runner.invoke(main, ['find-uncertain'])
        assert result.exit_code != 0

    def test_rebalance_command_missing_args(self):
        """Test rebalance command with missing arguments."""
        result = self.runner.invoke(main, ['rebalance'])
        assert result.exit_code != 0


class TestModuleImports:
    """Test cases for module imports and constants."""

    def test_import_statements(self):
        """Test that all imports work correctly."""
        # Test main imports
        from ordinal_classifier.cli import main, select_device, ARCHITECTURES
        assert main is not None
        assert select_device is not None
        assert ARCHITECTURES is not None

    def test_click_integration(self):
        """Test Click integration."""
        # Test that main is a Click group
        assert hasattr(main, 'commands')
        assert hasattr(main, 'add_command')

    def test_command_registration(self):
        """Test that all commands are registered."""
        expected_commands = ['train', 'predict', 'evaluate', 'heatmap', 'info', 'find-uncertain', 'rebalance']
        
        for cmd_name in expected_commands:
            assert cmd_name in main.commands, f"Command {cmd_name} not found in main.commands"

    def test_architecture_imports(self):
        """Test that architecture imports are accessible."""
        # Test that we can access architectures through the ARCHITECTURES dict
        for arch_name, arch_func in ARCHITECTURES.items():
            assert arch_func is not None
            assert callable(arch_func)


class TestParameterValidation:
    """Test cases for parameter validation across commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_numeric_parameter_help(self):
        """Test that numeric parameters are documented in help."""
        # Test train command numeric parameters
        result = self.runner.invoke(main, ['train', '--help'])
        assert result.exit_code == 0
        
        # Should mention epochs, batch-size, learning-rate etc.
        help_text = result.output.lower()
        assert 'epochs' in help_text
        assert 'batch' in help_text
        assert 'learning' in help_text

    def test_choice_parameter_help(self):
        """Test that choice parameters show valid options in help."""
        # Test architecture choices
        result = self.runner.invoke(main, ['train', '--help'])
        assert result.exit_code == 0
        
        # Should show architecture choices
        help_text = result.output
        for arch in ['resnet50', 'resnet18']:
            assert arch in help_text

    def test_boolean_flags_help(self):
        """Test boolean flag parameters in help."""
        # Test flags like --no-save, --ordinal, etc.
        result = self.runner.invoke(main, ['train', '--help'])
        assert result.exit_code == 0
        
        help_text = result.output
        assert '--no-save' in help_text or 'no-save' in help_text
        assert '--ordinal' in help_text or 'ordinal' in help_text


class TestDeviceSelection:
    """Additional test cases for device selection edge cases."""

    def test_device_selection_invalid_device(self):
        """Test device selection with invalid device string."""
        # Should return the device as-is for unknown devices
        result = select_device('unknown_device')
        assert result == 'unknown_device'

    def test_device_selection_case_sensitivity(self):
        """Test device selection is case sensitive."""
        result = select_device('CPU')  # Uppercase
        assert result == 'CPU'  # Should return as-is

    @patch('ordinal_classifier.cli.torch.cuda.is_available')
    def test_device_selection_cuda_edge_case(self, mock_cuda):
        """Test CUDA device selection edge cases."""
        mock_cuda.return_value = True
        result = select_device('cuda:0')  # Specific GPU
        assert result == 'cuda:0'


class TestArchitectureIntegration:
    """Test cases for architecture integration."""

    def test_resnet_architectures_callable(self):
        """Test that ResNet architectures are properly callable."""
        resnet_archs = ['resnet18', 'resnet34', 'resnet50']
        for arch_name in resnet_archs:
            arch_func = ARCHITECTURES[arch_name]
            assert callable(arch_func)

    def test_efficientnet_architectures_callable(self):
        """Test that EfficientNet architectures are properly callable."""
        efficientnet_archs = ['efficientnet_b0', 'efficientnet_b3', 'efficientnet_b5']
        for arch_name in efficientnet_archs:
            arch_func = ARCHITECTURES[arch_name]
            assert callable(arch_func)

    def test_architecture_consistency(self):
        """Test that architecture names are consistent."""
        # All architecture names should be lowercase with underscores
        for arch_name in ARCHITECTURES.keys():
            assert arch_name.islower()
            assert ' ' not in arch_name  # No spaces
            # Should contain either 'resnet' or 'efficientnet'
            assert 'resnet' in arch_name or 'efficientnet' in arch_name


class TestCLIConstants:
    """Test cases for CLI constants and module-level code."""

    def test_version_constant(self):
        """Test that version is properly defined."""
        # Check that version is accessible through the CLI
        runner = CliRunner()
        result = runner.invoke(main, ['--version'])
        assert result.exit_code == 0
        assert '2.0.0' in result.output

    def test_import_error_handling(self):
        """Test that imports are handled properly."""
        # Test that the module can be imported without errors
        import ordinal_classifier.cli
        assert ordinal_classifier.cli is not None

    def test_module_attributes(self):
        """Test that module has expected attributes."""
        import ordinal_classifier.cli as cli_module
        
        # Test required attributes exist
        assert hasattr(cli_module, 'main')
        assert hasattr(cli_module, 'select_device')
        assert hasattr(cli_module, 'ARCHITECTURES')

    def test_click_decorators(self):
        """Test that Click decorators are properly applied."""
        # Test that main has click decorators
        assert hasattr(main, 'commands')
        assert hasattr(main, 'params')
        
        # Test that commands have proper click attributes
        for cmd_name, cmd in main.commands.items():
            assert hasattr(cmd, 'params')
            assert hasattr(cmd, 'callback')


class TestErrorHandling:
    """Test cases for error handling in CLI."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_invalid_command(self):
        """Test handling of invalid commands."""
        result = self.runner.invoke(main, ['invalid_command'])
        assert result.exit_code != 0

    def test_help_on_error(self):
        """Test that help is available when commands fail."""
        # Most commands should show usage info on error
        result = self.runner.invoke(main, ['train'])
        assert result.exit_code != 0
        # Should provide some guidance
        assert 'Usage:' in result.output or 'Error:' in result.output or 'Missing' in result.output

    def test_command_completion(self):
        """Test that all expected commands are present."""
        # Get list of commands
        result = self.runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        
        # Check that major commands are listed
        commands_text = result.output
        expected_commands = ['train', 'predict', 'evaluate', 'heatmap']
        for cmd in expected_commands:
            assert cmd in commands_text