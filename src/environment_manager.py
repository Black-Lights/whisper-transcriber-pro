"""
Environment Manager - Handles virtual environment setup and package installation
"""

import json
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path


class EnvironmentManager:
    def __init__(self):
        self.app_dir = Path(__file__).parent.parent
        self.venv_dir = self.app_dir / "whisper_env"
        self.requirements_file = self.app_dir / "requirements.txt"

        # Platform-specific settings
        self.is_windows = platform.system() == "Windows"
        self.python_exe = self.get_python_executable()
        self.pip_exe = self.get_pip_executable()

    def get_python_executable(self):
        """Get the Python executable path in the virtual environment"""
        if self.is_windows:
            return self.venv_dir / "Scripts" / "python.exe"
        else:
            return self.venv_dir / "bin" / "python"

    def get_pip_executable(self):
        """Get the pip executable path in the virtual environment"""
        if self.is_windows:
            return self.venv_dir / "Scripts" / "pip.exe"
        else:
            return self.venv_dir / "bin" / "pip"

    def get_activate_script(self):
        """Get the activation script path"""
        if self.is_windows:
            return self.venv_dir / "Scripts" / "activate.bat"
        else:
            return self.venv_dir / "bin" / "activate"

    def check_environment(self):
        """Check if virtual environment and packages are properly set up"""
        status = {
            "venv_exists": False,
            "python_works": False,
            "whisper_installed": False,
            "torch_installed": False,
            "gpu_available": False,
        }

        # Check if venv directory exists
        if self.venv_dir.exists():
            status["venv_exists"] = True

            # Check if Python executable works
            if self.python_exe.exists():
                try:
                    result = subprocess.run(
                        [str(self.python_exe), "--version"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if result.returncode == 0:
                        status["python_works"] = True
                except:
                    pass

            # Check if packages are installed
            if status["python_works"]:
                try:
                    # Check whisper
                    result = subprocess.run(
                        [str(self.python_exe), "-c", "import whisper"],
                        capture_output=True,
                        timeout=10,
                    )
                    status["whisper_installed"] = result.returncode == 0

                    # Check torch
                    result = subprocess.run(
                        [str(self.python_exe), "-c", "import torch"],
                        capture_output=True,
                        timeout=10,
                    )
                    status["torch_installed"] = result.returncode == 0

                    # Check GPU availability
                    if status["torch_installed"]:
                        result = subprocess.run(
                            [
                                str(self.python_exe),
                                "-c",
                                "import torch; print(torch.cuda.is_available())",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )
                        if result.returncode == 0:
                            status["gpu_available"] = "True" in result.stdout
                except:
                    pass

        return status

    def create_requirements_file(self):
        """Create requirements.txt file"""
        requirements = [
            "openai-whisper>=20231117",
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "torchvision>=0.15.0",
            "tqdm>=4.65.0",
            "numpy>=1.24.0",
            "ffmpeg-python>=0.2.0",
        ]

        # Add GPU support for CUDA if available
        gpu_requirements = [
            "--index-url https://download.pytorch.org/whl/cu118",
            "torch>=2.0.0+cu118",
            "torchaudio>=2.0.0+cu118",
            "torchvision>=0.15.0+cu118",
        ]

        with open(self.requirements_file, "w") as f:
            f.write("# Whisper Transcriber Requirements\n\n")
            f.write("# CPU-only version (default)\n")
            for req in requirements:
                f.write(f"{req}\n")

            f.write("\n# For GPU support, replace torch packages above with:\n")
            for req in gpu_requirements:
                f.write(f"# {req}\n")

    def setup_environment(self, progress_callback=None):
        """Set up virtual environment and install packages"""
        try:
            if progress_callback:
                progress_callback("Setting up virtual environment...")

            # Remove existing venv if it exists but is broken
            if self.venv_dir.exists():
                status = self.check_environment()
                if not status["python_works"]:
                    print("Removing broken virtual environment...")
                    self.remove_environment()

            # Create virtual environment
            if not self.venv_dir.exists():
                print(f"Creating virtual environment in {self.venv_dir}")
                venv.create(self.venv_dir, with_pip=True)

            if progress_callback:
                progress_callback("Upgrading pip...")

            # Try to upgrade pip with better error handling
            try:
                self.run_pip_command(["install", "--upgrade", "pip"])
            except Exception as e:
                # If pip upgrade fails due to the new pip restrictions, try alternative method
                error_msg = str(e).lower()
                if "to modify pip, please run the following command" in error_msg:
                    if progress_callback:
                        progress_callback("Using alternative pip upgrade method...")
                    try:
                        # Use python -m pip method
                        result = subprocess.run(
                            [
                                str(self.python_exe),
                                "-m",
                                "pip",
                                "install",
                                "--upgrade",
                                "pip",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=300,
                        )
                        if result.returncode != 0:
                            print(f"Warning: Could not upgrade pip: {result.stderr}")
                            if progress_callback:
                                progress_callback(
                                    "Continuing with current pip version..."
                                )
                    except Exception as e2:
                        print(f"Warning: Pip upgrade failed, continuing anyway: {e2}")
                        if progress_callback:
                            progress_callback("Continuing with current pip version...")
                else:
                    raise e

            if progress_callback:
                progress_callback("Detecting GPU support...")

            # Detect GPU and install appropriate packages
            gpu_available = self.detect_gpu_support()

            if progress_callback:
                progress_callback("Installing Whisper and dependencies...")

            # Install compatible setuptools first to avoid pkg_resources issues
            if progress_callback:
                progress_callback("Installing compatible setuptools...")

            try:
                # Install a compatible version of setuptools that doesn't have the pkg_resources deprecation issue
                self.run_pip_command(["install", "setuptools<70.0.0"])
            except Exception as e:
                print(f"Warning: Could not install compatible setuptools: {e}")
                # Continue anyway, as this might not be critical

            # Install packages based on GPU availability
            if gpu_available:
                print("GPU detected, installing CUDA-enabled packages...")
                self.install_gpu_packages(progress_callback)
            else:
                print("No GPU detected, installing CPU-only packages...")
                self.install_cpu_packages(progress_callback)

            if progress_callback:
                progress_callback("Environment setup complete!")

            return True

        except Exception as e:
            print(f"Environment setup failed: {e}")
            if progress_callback:
                progress_callback(f"Setup failed: {e}")
            return False

    def detect_gpu_support(self):
        """Detect if NVIDIA GPU is available"""
        try:
            # Try to detect NVIDIA GPU
            result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=10)
            return result.returncode == 0
        except:
            return False

    def install_cpu_packages(self, progress_callback=None):
        """Install CPU-only packages"""
        packages = [
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "tqdm>=4.65.0",
            "numpy>=1.24.0",
            "ffmpeg-python>=0.2.0",
        ]

        for package in packages:
            if progress_callback:
                progress_callback(f"Installing {package.split('>=')[0]}...")
            self.run_pip_command(["install", package])

        # Install openai-whisper last with specific handling
        if progress_callback:
            progress_callback("Installing openai-whisper...")

        try:
            # Try installing openai-whisper with a specific version that's known to work
            self.run_pip_command(["install", "openai-whisper==20231117"])
        except Exception as e:
            if progress_callback:
                progress_callback("Trying alternative whisper installation...")
            try:
                # Try installing from git if the regular installation fails
                self.run_pip_command(
                    ["install", "git+https://github.com/openai/whisper.git"]
                )
            except Exception as e2:
                print(
                    f"Warning: Both whisper installations failed. You may need to install manually."
                )
                print(f"Error 1: {e}")
                print(f"Error 2: {e2}")
                if progress_callback:
                    progress_callback(
                        "Warning: Whisper installation failed - install manually later"
                    )

    def install_gpu_packages(self, progress_callback=None):
        """Install GPU-enabled packages"""
        # Install PyTorch with CUDA support first
        if progress_callback:
            progress_callback("Installing PyTorch with CUDA support...")

        torch_packages = [
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "torchvision>=0.15.0",
            "--index-url",
            "https://download.pytorch.org/whl/cu118",
        ]

        self.run_pip_command(["install"] + torch_packages)

        # Install other packages
        other_packages = ["tqdm>=4.65.0", "numpy>=1.24.0", "ffmpeg-python>=0.2.0"]

        for package in other_packages:
            if progress_callback:
                progress_callback(f"Installing {package.split('>=')[0]}...")
            self.run_pip_command(["install", package])

        # Install openai-whisper last with specific handling
        if progress_callback:
            progress_callback("Installing openai-whisper...")

        try:
            # Try installing openai-whisper with a specific version that's known to work
            self.run_pip_command(["install", "openai-whisper==20231117"])
        except Exception as e:
            if progress_callback:
                progress_callback("Trying alternative whisper installation...")
            try:
                # Try installing from git if the regular installation fails
                self.run_pip_command(
                    ["install", "git+https://github.com/openai/whisper.git"]
                )
            except Exception as e2:
                print(
                    f"Warning: Both whisper installations failed. You may need to install manually."
                )
                print(f"Error 1: {e}")
                print(f"Error 2: {e2}")
                if progress_callback:
                    progress_callback(
                        "Warning: Whisper installation failed - install manually later"
                    )

    def run_pip_command(self, args):
        """Run pip command in virtual environment with improved error handling"""
        cmd = [str(self.pip_exe)] + args
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        # Handle pip upgrade notices and errors more gracefully
        if result.returncode != 0:
            stderr_lower = result.stderr.lower()

            # Check if it's just a pip upgrade notice (not a real error)
            if (
                "new release of pip is available" in stderr_lower
                and "error:" not in stderr_lower
                and "failed" not in stderr_lower
            ):
                print(f"Note: {result.stderr.strip()}")
                return result

            # Handle specific pip modification error
            if "to modify pip, please run the following command" in stderr_lower:
                print("Attempting to resolve pip upgrade issue...")
                try:
                    # Try using python -m pip instead
                    upgrade_cmd = [
                        str(self.python_exe),
                        "-m",
                        "pip",
                        "install",
                        "--upgrade",
                        "pip",
                    ]
                    upgrade_result = subprocess.run(
                        upgrade_cmd, capture_output=True, text=True, timeout=600
                    )

                    if upgrade_result.returncode == 0:
                        print("Pip upgraded successfully, retrying original command...")
                        # Retry the original command
                        retry_result = subprocess.run(
                            cmd, capture_output=True, text=True, timeout=1200
                        )
                        if retry_result.returncode == 0:
                            return retry_result
                        else:
                            # If retry also fails, raise the original error
                            raise Exception(
                                f"Pip command failed after retry: {retry_result.stderr}"
                            )
                    else:
                        print(
                            f"Warning: Could not upgrade pip: {upgrade_result.stderr}"
                        )
                        # Continue with the original error
                        raise Exception(f"Pip command failed: {result.stderr}")
                except Exception as upgrade_error:
                    print(f"Warning: Pip upgrade attempt failed: {upgrade_error}")
                    raise Exception(f"Pip command failed: {result.stderr}")

            # For any other error, raise it
            raise Exception(f"Pip command failed: {result.stderr}")

        return result

    def run_python_command(self, args):
        """Run Python command in virtual environment"""
        cmd = [str(self.python_exe)] + args
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"Python command failed: {result.stderr}")

        return result

    def remove_environment(self):
        """Remove virtual environment"""
        import shutil

        if self.venv_dir.exists():
            shutil.rmtree(self.venv_dir)

    def get_activation_command(self):
        """Get command to activate virtual environment"""
        if self.is_windows:
            return f'"{self.get_activate_script()}"'
        else:
            return f'source "{self.get_activate_script()}"'

    def get_environment_info(self):
        """Get detailed environment information"""
        info = {
            "venv_path": str(self.venv_dir),
            "python_path": str(self.python_exe),
            "activation_command": self.get_activation_command(),
            "status": self.check_environment(),
        }

        # Get installed packages if environment is working
        if info["status"]["python_works"]:
            try:
                result = self.run_pip_command(["list", "--format=json"])
                info["installed_packages"] = json.loads(result.stdout)
            except:
                info["installed_packages"] = []

        return info
