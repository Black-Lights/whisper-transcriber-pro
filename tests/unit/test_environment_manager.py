"""
Unit tests for EnvironmentManager
"""

import platform
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from src.environment_manager import EnvironmentManager


class TestEnvironmentManager:
    """Test cases for EnvironmentManager"""

    def test_initialization(self):
        """Test EnvironmentManager initialization"""
        env_manager = EnvironmentManager()

        assert env_manager.app_dir.exists()
        assert env_manager.venv_dir.name == "whisper_env"
        assert env_manager.requirements_file.name == "requirements.txt"
        assert isinstance(env_manager.is_windows, bool)
        assert env_manager.python_exe.is_absolute()
        assert env_manager.pip_exe.is_absolute()

    def test_platform_detection(self):
        """Test platform-specific paths"""
        env_manager = EnvironmentManager()

        if platform.system() == "Windows":
            assert env_manager.is_windows is True
            assert "Scripts" in str(env_manager.python_exe)
            assert env_manager.python_exe.name == "python.exe"
        else:
            assert env_manager.is_windows is False
            assert "bin" in str(env_manager.python_exe)
            assert env_manager.python_exe.name == "python"

    def test_get_activate_script(self):
        """Test activation script path generation"""
        env_manager = EnvironmentManager()
        activate_script = env_manager.get_activate_script()

        assert activate_script.is_absolute()
        if env_manager.is_windows:
            assert activate_script.name == "activate.bat"
        else:
            assert activate_script.name == "activate"

    def test_check_environment_no_venv(self):
        """Test environment check when no virtual environment exists"""
        env_manager = EnvironmentManager()

        with patch.object(env_manager.venv_dir, "exists", return_value=False):
            status = env_manager.check_environment()

        assert status["venv_exists"] is False
        assert status["python_works"] is False
        assert status["whisper_installed"] is False
        assert status["torch_installed"] is False
        assert status["gpu_available"] is False

    @patch("subprocess.run")
    def test_check_environment_with_working_venv(self, mock_run):
        """Test environment check with working virtual environment"""
        env_manager = EnvironmentManager()

        # Mock successful subprocess calls
        mock_run.side_effect = [
            Mock(returncode=0, stdout="Python 3.9.0"),  # python --version
            Mock(returncode=0),  # import whisper
            Mock(returncode=0),  # import torch
            Mock(returncode=0, stdout="True"),  # torch.cuda.is_available()
        ]

        with patch.object(
            env_manager.venv_dir, "exists", return_value=True
        ), patch.object(env_manager.python_exe, "exists", return_value=True):
            status = env_manager.check_environment()

        assert status["venv_exists"] is True
        assert status["python_works"] is True
        assert status["whisper_installed"] is True
        assert status["torch_installed"] is True
        assert status["gpu_available"] is True

    @patch("subprocess.run")
    def test_check_environment_with_broken_packages(self, mock_run):
        """Test environment check with broken package installations"""
        env_manager = EnvironmentManager()

        # Mock mixed success/failure subprocess calls
        mock_run.side_effect = [
            Mock(returncode=0, stdout="Python 3.9.0"),  # python --version works
            Mock(returncode=1),  # import whisper fails
            Mock(returncode=0),  # import torch works
            Mock(returncode=0, stdout="False"),  # torch.cuda.is_available() = False
        ]

        with patch.object(
            env_manager.venv_dir, "exists", return_value=True
        ), patch.object(env_manager.python_exe, "exists", return_value=True):
            status = env_manager.check_environment()

        assert status["venv_exists"] is True
        assert status["python_works"] is True
        assert status["whisper_installed"] is False
        assert status["torch_installed"] is True
        assert status["gpu_available"] is False

    @patch("venv.create")
    @patch("subprocess.run")
    def test_setup_environment_fresh_install(self, mock_run, mock_venv_create):
        """Test fresh environment setup"""
        env_manager = EnvironmentManager()
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")

        with patch.object(
            env_manager.venv_dir, "exists", return_value=False
        ), patch.object(
            env_manager, "detect_gpu_support", return_value=False
        ), patch.object(
            env_manager, "install_cpu_packages"
        ) as mock_install:

            result = env_manager.setup_environment()

        assert result is True
        mock_venv_create.assert_called_once_with(env_manager.venv_dir, with_pip=True)
        mock_install.assert_called_once()

    @patch("venv.create")
    @patch("subprocess.run")
    def test_setup_environment_with_gpu(self, mock_run, mock_venv_create):
        """Test environment setup with GPU support"""
        env_manager = EnvironmentManager()
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")

        with patch.object(
            env_manager.venv_dir, "exists", return_value=False
        ), patch.object(
            env_manager, "detect_gpu_support", return_value=True
        ), patch.object(
            env_manager, "install_gpu_packages"
        ) as mock_install:

            result = env_manager.setup_environment()

        assert result is True
        mock_install.assert_called_once()

    @patch("subprocess.run")
    def test_detect_gpu_support_nvidia(self, mock_run):
        """Test GPU detection with NVIDIA GPU present"""
        env_manager = EnvironmentManager()
        mock_run.return_value = Mock(returncode=0)

        result = env_manager.detect_gpu_support()

        assert result is True
        mock_run.assert_called_once_with(
            ["nvidia-smi"], capture_output=True, timeout=10
        )

    @patch("subprocess.run")
    def test_detect_gpu_support_no_nvidia(self, mock_run):
        """Test GPU detection with no NVIDIA GPU"""
        env_manager = EnvironmentManager()
        mock_run.side_effect = FileNotFoundError()

        result = env_manager.detect_gpu_support()

        assert result is False

    @patch("subprocess.run")
    def test_install_cpu_packages(self, mock_run):
        """Test CPU package installation"""
        env_manager = EnvironmentManager()
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")

        with patch.object(env_manager, "run_pip_command") as mock_pip:
            env_manager.install_cpu_packages()

        # Verify pip was called for each expected package
        expected_packages = [
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "tqdm>=4.65.0",
            "numpy>=1.24.0",
            "ffmpeg-python>=0.2.0",
        ]

        for package in expected_packages:
            mock_pip.assert_any_call(["install", package])

    @patch("subprocess.run")
    def test_install_gpu_packages(self, mock_run):
        """Test GPU package installation"""
        env_manager = EnvironmentManager()
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")

        with patch.object(env_manager, "run_pip_command") as mock_pip:
            env_manager.install_gpu_packages()

        # Verify GPU-specific packages were installed
        calls = mock_pip.call_args_list
        gpu_call_found = False
        for call in calls:
            if "torch>=2.0.0" in call[0][0] and "--index-url" in call[0][0]:
                gpu_call_found = True
                break

        assert gpu_call_found, "GPU-specific PyTorch installation not found"

    @patch("subprocess.run")
    def test_run_pip_command_success(self, mock_run):
        """Test successful pip command execution"""
        env_manager = EnvironmentManager()
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="Success")

        result = env_manager.run_pip_command(["install", "package"])

        assert result.returncode == 0
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_run_pip_command_failure(self, mock_run):
        """Test failed pip command execution"""
        env_manager = EnvironmentManager()
        mock_run.return_value = Mock(returncode=1, stderr="Error", stdout="")

        with pytest.raises(Exception, match="Pip command failed"):
            env_manager.run_pip_command(["install", "nonexistent"])

    @patch("subprocess.run")
    def test_run_pip_command_upgrade_notice(self, mock_run):
        """Test pip command with upgrade notice (not a real error)"""
        env_manager = EnvironmentManager()
        mock_run.return_value = Mock(
            returncode=1, stderr="WARNING: A new release of pip is available", stdout=""
        )

        # This should not raise an exception for upgrade notices
        with patch.object(
            env_manager, "run_pip_command", wraps=env_manager.run_pip_command
        ):
            try:
                env_manager.run_pip_command(["install", "package"])
            except Exception:
                pytest.fail("Pip upgrade notice should not raise exception")

    @patch("subprocess.run")
    def test_run_python_command_success(self, mock_run):
        """Test successful Python command execution"""
        env_manager = EnvironmentManager()
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="Success")

        result = env_manager.run_python_command(["-c", "print('test')"])

        assert result.returncode == 0
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_run_python_command_failure(self, mock_run):
        """Test failed Python command execution"""
        env_manager = EnvironmentManager()
        mock_run.return_value = Mock(returncode=1, stderr="Error", stdout="")

        with pytest.raises(Exception, match="Python command failed"):
            env_manager.run_python_command(["-c", "raise Exception()"])

    @patch("shutil.rmtree")
    def test_remove_environment(self, mock_rmtree):
        """Test virtual environment removal"""
        env_manager = EnvironmentManager()

        with patch.object(env_manager.venv_dir, "exists", return_value=True):
            env_manager.remove_environment()

        mock_rmtree.assert_called_once_with(env_manager.venv_dir)

    def test_get_activation_command_windows(self):
        """Test activation command generation for Windows"""
        env_manager = EnvironmentManager()

        with patch.object(env_manager, "is_windows", True):
            command = env_manager.get_activation_command()

        assert "activate.bat" in command
        assert command.startswith('"') and command.endswith('"')

    def test_get_activation_command_unix(self):
        """Test activation command generation for Unix/Linux"""
        env_manager = EnvironmentManager()

        with patch.object(env_manager, "is_windows", False):
            command = env_manager.get_activation_command()

        assert command.startswith('source "')
        assert "activate" in command

    @patch("subprocess.run")
    @patch("json.loads")
    def test_get_environment_info(self, mock_json_loads, mock_run):
        """Test environment information gathering"""
        env_manager = EnvironmentManager()

        # Mock successful environment check
        mock_run.return_value = Mock(
            returncode=0, stdout='[{"name": "package", "version": "1.0"}]'
        )
        mock_json_loads.return_value = [{"name": "package", "version": "1.0"}]

        with patch.object(env_manager, "check_environment") as mock_check:
            mock_check.return_value = {"python_works": True}

            info = env_manager.get_environment_info()

        assert "venv_path" in info
        assert "python_path" in info
        assert "activation_command" in info
        assert "status" in info
        assert "installed_packages" in info

        assert str(env_manager.venv_dir) == info["venv_path"]
        assert str(env_manager.python_exe) == info["python_path"]

    @patch("subprocess.run")
    def test_get_environment_info_broken_env(self, mock_run):
        """Test environment info when environment is broken"""
        env_manager = EnvironmentManager()

        with patch.object(env_manager, "check_environment") as mock_check:
            mock_check.return_value = {"python_works": False}

            info = env_manager.get_environment_info()

        assert info["installed_packages"] == []

    def test_create_requirements_file(self, temp_dir):
        """Test requirements file creation"""
        env_manager = EnvironmentManager()

        # Temporarily change requirements file location
        original_req_file = env_manager.requirements_file
        env_manager.requirements_file = temp_dir / "test_requirements.txt"

        try:
            env_manager.create_requirements_file()

            assert env_manager.requirements_file.exists()
            content = env_manager.requirements_file.read_text()

            # Check for essential packages
            assert "openai-whisper" in content
            assert "torch" in content
            assert "torchaudio" in content

        finally:
            env_manager.requirements_file = original_req_file

    @patch("subprocess.run")
    def test_setup_environment_progress_callback(self, mock_run):
        """Test setup environment with progress callback"""
        env_manager = EnvironmentManager()
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")

        progress_messages = []

        def progress_callback(message):
            progress_messages.append(message)

        with patch.object(
            env_manager.venv_dir, "exists", return_value=False
        ), patch.object(
            env_manager, "detect_gpu_support", return_value=False
        ), patch.object(
            env_manager, "install_cpu_packages"
        ), patch(
            "venv.create"
        ):

            result = env_manager.setup_environment(progress_callback)

        assert result is True
        assert len(progress_messages) > 0
        assert any("environment" in msg.lower() for msg in progress_messages)

    @patch("subprocess.run")
    def test_setup_environment_with_existing_broken_venv(self, mock_run):
        """Test setup when existing venv is broken"""
        env_manager = EnvironmentManager()
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")

        with patch.object(
            env_manager.venv_dir, "exists", return_value=True
        ), patch.object(env_manager, "check_environment") as mock_check, patch.object(
            env_manager, "remove_environment"
        ) as mock_remove, patch.object(
            env_manager, "detect_gpu_support", return_value=False
        ), patch.object(
            env_manager, "install_cpu_packages"
        ), patch(
            "venv.create"
        ):

            # Mock broken environment
            mock_check.return_value = {"python_works": False}

            result = env_manager.setup_environment()

        assert result is True
        mock_remove.assert_called_once()  # Should remove broken environment

    @patch("subprocess.run")
    def test_pip_upgrade_restriction_handling(self, mock_run):
        """Test handling of pip upgrade restrictions in newer Python versions"""
        env_manager = EnvironmentManager()

        # Simulate pip upgrade restriction error
        mock_run.side_effect = [
            Mock(
                returncode=1, stderr="to modify pip, please run the following command"
            ),
            Mock(returncode=0, stderr="", stdout=""),  # Successful retry
        ]

        with patch.object(
            env_manager, "run_pip_command", wraps=env_manager.run_pip_command
        ):
            # This should handle the pip upgrade restriction gracefully
            try:
                result = env_manager.run_pip_command(["install", "--upgrade", "pip"])
                # The wrapped method should handle this case
            except Exception as e:
                # Verify it's trying to handle the pip restriction
                assert "pip" in str(e).lower()


class TestEnvironmentManagerIntegration:
    """Integration tests for EnvironmentManager with real file operations"""

    def test_path_generation_consistency(self, temp_dir):
        """Test that all path generation is consistent across platforms"""
        # Create a temporary EnvironmentManager with custom app_dir
        env_manager = EnvironmentManager()
        original_app_dir = env_manager.app_dir
        env_manager.app_dir = temp_dir
        env_manager.venv_dir = temp_dir / "test_venv"

        try:
            python_exe = env_manager.get_python_executable()
            pip_exe = env_manager.get_pip_executable()
            activate_script = env_manager.get_activate_script()

            # All paths should be within the venv directory
            assert env_manager.venv_dir in python_exe.parents
            assert env_manager.venv_dir in pip_exe.parents
            assert env_manager.venv_dir in activate_script.parents

            # Paths should be platform-appropriate
            if env_manager.is_windows:
                assert "Scripts" in str(python_exe)
                assert python_exe.suffix == ".exe"
            else:
                assert "bin" in str(python_exe)
                assert python_exe.suffix == ""

        finally:
            env_manager.app_dir = original_app_dir

    def test_requirements_file_format(self, temp_dir):
        """Test that generated requirements file has correct format"""
        env_manager = EnvironmentManager()
        original_req_file = env_manager.requirements_file
        env_manager.requirements_file = temp_dir / "test_requirements.txt"

        try:
            env_manager.create_requirements_file()

            content = env_manager.requirements_file.read_text()
            lines = [line.strip() for line in content.split("\n") if line.strip()]

            # Should have package specifications
            package_lines = [line for line in lines if not line.startswith("#")]
            assert len(package_lines) > 0

            # Should have version specifications
            versioned_packages = [line for line in package_lines if ">=" in line]
            assert len(versioned_packages) > 0

        finally:
            env_manager.requirements_file = original_req_file
