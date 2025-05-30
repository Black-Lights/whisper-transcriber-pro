"""
Environment Manager - Handles virtual environment setup and package installation
Author: Black-Lights (https://github.com/Black-Lights)
Project: Whisper Transcriber Pro
"""

import os
import sys
import subprocess
import venv
from pathlib import Path
import json
import platform

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
            'venv_exists': False,
            'python_works': False,
            'whisper_installed': False,
            'torch_installed': False,
            'gpu_available': False
        }
        
        # Check if venv directory exists
        if self.venv_dir.exists():
            status['venv_exists'] = True
            
            # Check if Python executable works
            if self.python_exe.exists():
                try:
                    result = subprocess.run([str(self.python_exe), "--version"], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        status['python_works'] = True
                except:
                    pass
            
            # Check if packages are installed
            if status['python_works']:
                try:
                    # Check whisper
                    result = subprocess.run([str(self.python_exe), "-c", "import whisper"], 
                                          capture_output=True, timeout=10)
                    status['whisper_installed'] = result.returncode == 0
                    
                    # Check torch
                    result = subprocess.run([str(self.python_exe), "-c", "import torch"], 
                                          capture_output=True, timeout=10)
                    status['torch_installed'] = result.returncode == 0
                    
                    # Check GPU availability
                    if status['torch_installed']:
                        result = subprocess.run([str(self.python_exe), "-c", 
                                               "import torch; print(torch.cuda.is_available())"], 
                                              capture_output=True, text=True, timeout=10)
                        if result.returncode == 0:
                            status['gpu_available'] = "True" in result.stdout
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
            "ffmpeg-python>=0.2.0"
        ]
        
        # Add GPU support for CUDA if available
        gpu_requirements = [
            "--index-url https://download.pytorch.org/whl/cu118",
            "torch>=2.0.0+cu118",
            "torchaudio>=2.0.0+cu118",
            "torchvision>=0.15.0+cu118"
        ]
        
        with open(self.requirements_file, 'w') as f:
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
                progress_callback("🔧 Setting up virtual environment...")
            
            # Remove existing venv if it exists but is broken
            if self.venv_dir.exists():
                status = self.check_environment()
                if not status['python_works']:
                    print("Removing broken virtual environment...")
                    self.remove_environment()
            
            # Create virtual environment
            if not self.venv_dir.exists():
                print(f"Creating virtual environment in {self.venv_dir}")
                venv.create(self.venv_dir, with_pip=True)
            
            if progress_callback:
                progress_callback("📦 Upgrading pip...")
            
            # Upgrade pip
            self.run_pip_command(["install", "--upgrade", "pip"])
            
            if progress_callback:
                progress_callback("🔍 Detecting GPU support...")
            
            # Detect GPU and install appropriate packages
            gpu_available = self.detect_gpu_support()
            
            if progress_callback:
                progress_callback("📥 Installing Whisper and dependencies...")
            
            # Install packages based on GPU availability
            if gpu_available:
                print("GPU detected, installing CUDA-enabled packages...")
                self.install_gpu_packages(progress_callback)
            else:
                print("No GPU detected, installing CPU-only packages...")
                self.install_cpu_packages(progress_callback)
            
            if progress_callback:
                progress_callback("✅ Environment setup complete!")
            
            return True
            
        except Exception as e:
            print(f"Environment setup failed: {e}")
            if progress_callback:
                progress_callback(f"❌ Setup failed: {e}")
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
            "openai-whisper>=20231117",
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "tqdm>=4.65.0",
            "numpy>=1.24.0",
            "ffmpeg-python>=0.2.0"
        ]
        
        for package in packages:
            if progress_callback:
                progress_callback(f"Installing {package.split('>=')[0]}...")
            self.run_pip_command(["install", package])
    
    def install_gpu_packages(self, progress_callback=None):
        """Install GPU-enabled packages"""
        # Install PyTorch with CUDA support first
        if progress_callback:
            progress_callback("Installing PyTorch with CUDA support...")
        
        torch_packages = [
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "torchvision>=0.15.0",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ]
        
        self.run_pip_command(["install"] + torch_packages)
        
        # Install other packages
        other_packages = [
            "openai-whisper>=20231117",
            "tqdm>=4.65.0", 
            "numpy>=1.24.0",
            "ffmpeg-python>=0.2.0"
        ]
        
        for package in other_packages:
            if progress_callback:
                progress_callback(f"Installing {package.split('>=')[0]}...")
            self.run_pip_command(["install", package])
    
    def run_pip_command(self, args):
        """Run pip command in virtual environment"""
        cmd = [str(self.pip_exe)] + args
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
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
            'venv_path': str(self.venv_dir),
            'python_path': str(self.python_exe),
            'activation_command': self.get_activation_command(),
            'status': self.check_environment()
        }
        
        # Get installed packages if environment is working
        if info['status']['python_works']:
            try:
                result = self.run_pip_command(["list", "--format=json"])
                info['installed_packages'] = json.loads(result.stdout)
            except:
                info['installed_packages'] = []
        
        return info