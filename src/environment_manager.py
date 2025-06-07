"""
Environment Manager - Handles virtual environment setup and package installation
Fixed version with proper timeout and error handling
"""

import json
import os
import platform
import subprocess
import sys
import threading
import time
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

            # Ensure requirements.txt exists
            if not self.requirements_file.exists():
                if progress_callback:
                    progress_callback("Creating requirements.txt...")
                self.create_requirements_file()

            if progress_callback:
                progress_callback("Upgrading pip...")

            # Try to upgrade pip with better error handling
            try:
                self.run_pip_command_basic(["install", "--upgrade", "pip"], timeout=300)
            except Exception as e:
                # If pip upgrade fails, try alternative method
                error_msg = str(e).lower()
                if "to modify pip" in error_msg or "externally-managed" in error_msg:
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
                                progress_callback("Continuing with current pip version...")
                    except Exception as e2:
                        print(f"Warning: Pip upgrade failed, continuing anyway: {e2}")
                        if progress_callback:
                            progress_callback("Continuing with current pip version...")
                else:
                    print(f"Warning: Pip upgrade failed: {e}")

            if progress_callback:
                progress_callback("Detecting GPU support...")

            # Detect GPU and install appropriate packages
            gpu_available = self.detect_gpu_support()

            # Install packages based on GPU availability
            if gpu_available:
                print("GPU detected, installing CUDA-enabled packages...")
                success = self.install_gpu_packages(progress_callback)
            else:
                print("No GPU detected, installing CPU-only packages...")
                success = self.install_cpu_packages(progress_callback)

            if success:
                if progress_callback:
                    progress_callback("Environment setup complete!")
                return True
            else:
                if progress_callback:
                    progress_callback("Some packages failed to install")
                return False

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
        """Install CPU-only packages from requirements.txt"""
        try:
            if progress_callback:
                progress_callback("Installing packages from requirements.txt (CPU version)...")
            
            # Install from requirements.txt with CPU-only PyTorch
            success = self.run_pip_command_with_progress(
                ["install", "-r", str(self.requirements_file), 
                 "--index-url", "https://download.pytorch.org/whl/cpu"],
                progress_callback,
                timeout=1800  # 30 minutes
            )
            
            return success

        except Exception as e:
            print(f"CPU package installation failed: {e}")
            return False

    def install_gpu_packages(self, progress_callback=None):
        """Install GPU-enabled packages from requirements.txt"""
        try:
            if progress_callback:
                progress_callback("Installing packages from requirements.txt (GPU version - this may take 20-30 minutes)...")
            
            # Install from requirements.txt with CUDA PyTorch
            success = self.run_pip_command_with_progress(
                ["install", "-r", str(self.requirements_file),
                 "--index-url", "https://download.pytorch.org/whl/cu118"],
                progress_callback,
                timeout=3600  # 1 hour
            )
            
            if not success:
                print("GPU installation failed, falling back to CPU version...")
                return self.install_cpu_packages(progress_callback)

            return success

        except Exception as e:
            print(f"GPU package installation failed: {e}")
            print("Falling back to CPU installation...")
            return self.install_cpu_packages(progress_callback)

    def create_requirements_file(self):
        """Create requirements.txt file if it doesn't exist"""
        if self.requirements_file.exists():
            return  # File already exists, don't overwrite
            
        requirements_content = """# Whisper Transcriber Pro - Requirements (Stable Production Versions)
# Author: Black-Lights (https://github.com/Black-Lights)
# Project: Whisper Transcriber Pro v1.2.0

# Install with: pip install -r requirements.txt

# Core dependencies (Stable, well-tested versions)
openai-whisper>=v20240930
tqdm>=4.67.1
numpy>=2.0.2
requests>=2.32.3
psutil>=7.0.0

# PyTorch (Stable LTS-like versions)
torch>=2.7.1
torchaudio>=2.7.1

# Audio/Video processing
ffmpeg-python>=0.2.0

# GUI dependencies (included with Python)
# tkinter (built-in)
"""
        
        try:
            with open(self.requirements_file, 'w', encoding='utf-8') as f:
                f.write(requirements_content)
            print(f"Created requirements.txt with stable dependencies")
        except Exception as e:
            print(f"Warning: Could not create requirements.txt: {e}")

    def install_whisper(self, progress_callback=None):
        """Install OpenAI Whisper - this is now handled by requirements.txt"""
        # This method is no longer needed since whisper is in requirements.txt
        # But keeping it for compatibility
        return True

    def run_pip_command_with_progress(self, args, progress_callback=None, timeout=3600):
        """Run pip command with real-time progress monitoring"""
        cmd = [str(self.pip_exe)] + args
        
        try:
            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor progress in separate thread
            progress_thread = threading.Thread(
                target=self._monitor_pip_progress,
                args=(process, progress_callback),
                daemon=True
            )
            progress_thread.start()
            
            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                raise Exception(f"Command timed out after {timeout} seconds")
            
            # Wait for progress thread to finish
            progress_thread.join(timeout=5)
            
            if process.returncode == 0:
                return True
            else:
                # Check if it's a real error or just a warning
                if self._is_real_error(stderr):
                    print(f"Pip command failed: {stderr}")
                    return False
                else:
                    print(f"Pip completed with warnings: {stderr}")
                    return True
                    
        except Exception as e:
            print(f"Pip command failed: {e}")
            return False

    def run_pip_command_basic(self, args, timeout=600):
        """Run basic pip command with simple error handling"""
        cmd = [str(self.pip_exe)] + args
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout
            )
            
            if result.returncode == 0:
                return True
            elif not self._is_real_error(result.stderr):
                # Not a real error, just warnings
                return True
            else:
                print(f"Pip command failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"Pip command timed out after {timeout} seconds")
            return False
        except Exception as e:
            print(f"Pip command failed: {e}")
            return False

    def _monitor_pip_progress(self, process, progress_callback):
        """Monitor pip output for progress information"""
        if not progress_callback:
            return
            
        download_started = False
        last_update = time.time()
        
        while process.poll() is None:
            try:
                # Read output line by line
                line = process.stdout.readline()
                if line:
                    line = line.strip()
                    
                    # Look for download progress
                    if "Downloading" in line and ("MB" in line or "GB" in line):
                        if not download_started:
                            progress_callback("Download started...")
                            download_started = True
                        
                        # Extract progress if available
                        if "/" in line and ("MB" in line or "GB" in line):
                            try:
                                # Extract size info
                                parts = line.split()
                                for i, part in enumerate(parts):
                                    if "/" in part and ("MB" in part or "GB" in part):
                                        progress_callback(f"Downloading: {part}")
                                        break
                            except:
                                pass
                    
                    elif "Installing" in line or "Successfully installed" in line:
                        progress_callback("Installing packages...")
                    
                    # Update every 10 seconds to avoid spam
                    if time.time() - last_update > 10:
                        if download_started:
                            progress_callback("Download in progress... please wait")
                        last_update = time.time()
                        
            except Exception:
                break
                
            time.sleep(1)

    def _is_real_error(self, stderr):
        """Determine if stderr contains a real error or just warnings"""
        if not stderr:
            return False
            
        stderr_lower = stderr.lower()
        
        # These are warnings, not errors
        warning_phrases = [
            "new release of pip is available",
            "you should consider upgrading",
            "deprecation",
            "warning",
            "note:",
        ]
        
        # These are real errors
        error_phrases = [
            "error:",
            "failed",
            "could not",
            "unable to",
            "permission denied",
            "access denied",
            "no module named",
            "syntax error",
            "import error",
        ]
        
        # Check for real errors first
        for phrase in error_phrases:
            if phrase in stderr_lower:
                return True
                
        # If only warnings, not a real error
        for phrase in warning_phrases:
            if phrase in stderr_lower:
                return False
                
        # If we can't determine, assume it's an error
        return True

    def run_python_command(self, args):
        """Run Python command in virtual environment"""
        cmd = [str(self.python_exe)] + args
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            raise Exception(f"Python command failed: {result.stderr}")

        return result

    def remove_environment(self):
        """Remove virtual environment"""
        import shutil

        if self.venv_dir.exists():
            try:
                shutil.rmtree(self.venv_dir)
                print("Removed existing virtual environment")
            except Exception as e:
                print(f"Warning: Could not remove virtual environment: {e}")

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
                result = self.run_pip_command_basic(["list", "--format=json"])
                if result:
                    cmd = [str(self.pip_exe), "list", "--format=json"]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    if result.returncode == 0:
                        info["installed_packages"] = json.loads(result.stdout)
            except:
                info["installed_packages"] = []

        return info