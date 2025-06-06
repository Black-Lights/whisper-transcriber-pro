#!/usr/bin/env python3
"""
Whisper Transcriber Pro - Installation Script (Updated for v1.1.0)
Author: Black-Lights (https://github.com/Black-Lights)
This script sets up the complete application with virtual environment and new dependencies
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import shutil

class WhisperInstaller:
    def __init__(self):
        self.app_name = "Whisper Transcriber Pro v1.1.0"
        self.app_dir = Path(__file__).parent
        self.system = platform.system()
        self.is_windows = self.system == "Windows"
        
        print(f"{self.app_name} Installer")
        print("=" * 50)
    
    def check_requirements(self):
        """Check system requirements"""
        print("Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            print("ERROR: Python 3.8 or higher required")
            return False
        
        print(f"SUCCESS: Python {sys.version.split()[0]} detected")
        
        # Check pip
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], 
                          capture_output=True, check=True)
            print("SUCCESS: pip is available")
        except subprocess.CalledProcessError:
            print("ERROR: pip is not available")
            return False
        
        # Check venv
        try:
            import venv
            print("SUCCESS: venv module available")
        except ImportError:
            print("ERROR: venv module not available")
            return False
        
        # Check for ffmpeg (optional but recommended)
        try:
            subprocess.run(["ffmpeg", "-version"], 
                          capture_output=True, check=True, timeout=10)
            print("SUCCESS: ffmpeg detected")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print("WARNING: ffmpeg not detected (optional - some features may be limited)")
        
        return True
    
    def check_existing_installation(self):
        """Check if there's an existing installation"""
        venv_dir = self.app_dir / "whisper_env"
        main_file = self.app_dir / "main.py"
        
        existing_installation = {
            'venv_exists': venv_dir.exists(),
            'main_exists': main_file.exists(),
            'has_batch_file': (self.app_dir / "Whisper_Transcriber.bat").exists() if self.is_windows else (self.app_dir / "whisper_transcriber.sh").exists()
        }
        
        return existing_installation
    
    def update_existing_installation(self):
        """Update existing installation with new dependencies"""
        print("\nUpdating existing installation...")
        
        venv_dir = self.app_dir / "whisper_env"
        
        # Check if virtual environment exists and is working
        if self.is_windows:
            python_exe = venv_dir / "Scripts" / "python.exe"
            pip_exe = venv_dir / "Scripts" / "pip.exe"
        else:
            python_exe = venv_dir / "bin" / "python"
            pip_exe = venv_dir / "bin" / "pip"
        
        if not python_exe.exists():
            print("ERROR: Virtual environment appears to be corrupted")
            return False
        
        try:
            # Test virtual environment
            result = subprocess.run([str(python_exe), "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print("ERROR: Virtual environment is not working properly")
                return False
            
            print(f"   Virtual environment working: {result.stdout.strip()}")
            
            # Install new dependencies
            print("   Installing new dependencies...")
            
            # Install psutil for process management
            result = subprocess.run([str(pip_exe), "install", "psutil>=5.9.0"], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("   SUCCESS: psutil installed for process management")
            else:
                print(f"   WARNING: Could not install psutil: {result.stderr}")
            
            # Upgrade existing packages to ensure compatibility
            print("   Upgrading existing packages...")
            
            packages_to_upgrade = [
                "openai-whisper>=20231117",
                "torch>=2.0.0",
                "torchaudio>=2.0.0", 
                "numpy>=1.24.0",
                "tqdm>=4.65.0"
            ]
            
            for package in packages_to_upgrade:
                try:
                    subprocess.run([str(pip_exe), "install", "--upgrade", package], 
                                 capture_output=True, text=True, timeout=300)
                    print(f"   Updated: {package.split('>=')[0]}")
                except:
                    print(f"   Warning: Could not update {package.split('>=')[0]}")
            
            print("   SUCCESS: Installation updated successfully")
            return True
            
        except Exception as e:
            print(f"   ERROR: Failed to update installation: {e}")
            return False
    
    def create_directory_structure(self):
        """Create application directory structure"""
        print("\nCreating directory structure...")
        
        directories = [
            "src",
            "logs", 
            "temp",
            "models"
        ]
        
        for directory in directories:
            dir_path = self.app_dir / directory
            dir_path.mkdir(exist_ok=True)
            print(f"   Created: {directory}/")
        
        return True
    
    def create_batch_files(self):
        """Create batch/shell files for easy launching"""
        print("\nCreating launcher scripts...")
        
        if self.is_windows:
            # Create Windows batch file
            batch_content = f'''@echo off
cd /d "{self.app_dir}"
call whisper_env\\Scripts\\activate.bat
python main.py
pause
'''
            batch_file = self.app_dir / "Whisper_Transcriber.bat"
            with open(batch_file, 'w') as f:
                f.write(batch_content)
            print("   Created: Whisper_Transcriber.bat")
            
            # Create setup batch file
            setup_batch = f'''@echo off
cd /d "{self.app_dir}"
python install.py
pause
'''
            setup_file = self.app_dir / "Setup.bat"
            with open(setup_file, 'w') as f:
                f.write(setup_batch)
            print("   Created: Setup.bat")
        
        else:
            # Create shell script for Unix/Linux/Mac
            script_content = f'''#!/bin/bash
cd "{self.app_dir}"
source whisper_env/bin/activate
python main.py
'''
            script_file = self.app_dir / "whisper_transcriber.sh"
            with open(script_file, 'w') as f:
                f.write(script_content)
            script_file.chmod(0o755)  # Make executable
            print("   Created: whisper_transcriber.sh")
        
        return True
    
    def setup_environment(self):
        """Set up the virtual environment"""
        print("\nSetting up virtual environment...")
        
        # Import environment manager
        sys.path.append(str(self.app_dir / "src"))
        try:
            from src.environment_manager import EnvironmentManager
        except ImportError:
            print("   ERROR: Could not import environment_manager")
            print("   Make sure all source files are present in the src/ directory")
            return False
        
        env_manager = EnvironmentManager()
        
        def progress_callback(message):
            print(f"   {message}")
        
        success = env_manager.setup_environment(progress_callback)
        
        if success:
            print("   SUCCESS: Environment setup completed")
        else:
            print("   ERROR: Environment setup failed")
        
        return success
    
    def download_default_model(self):
        """Download the default medium model"""
        print("\nDownloading default model...")
        
        try:
            sys.path.append(str(self.app_dir / "src"))
            from src.model_manager import ModelManager
            
            model_manager = ModelManager()
            
            def progress_callback(message):
                print(f"   {message}")
            
            success = model_manager.download_models(["medium"], progress_callback)
            
            if success:
                print("   SUCCESS: Default model downloaded")
            else:
                print("   WARNING: Model download failed (can download later in app)")
            
            return success
            
        except Exception as e:
            print(f"   WARNING: Model download failed: {e}")
            return False
    
    def run_installation(self):
        """Run the complete installation process"""
        try:
            print(f"Installing {self.app_name}...")
            print(f"Target directory: {self.app_dir}")
            print()
            
            # Check requirements
            if not self.check_requirements():
                print("\nERROR: System requirements not met")
                return False
            
            # Check for existing installation
            existing = self.check_existing_installation()
            
            if existing['venv_exists'] and existing['main_exists']:
                print("\n" + "="*50)
                print("EXISTING INSTALLATION DETECTED")
                print("="*50)
                print("Found existing Whisper Transcriber Pro installation.")
                print("This will update your installation with new features:")
                print("  • Live transcription display")
                print("  • Better process management") 
                print("  • Enhanced audio quality handling")
                print("  • Improved timer system")
                print()
                
                update_choice = input("Update existing installation? (Y/n): ").lower()
                if update_choice != 'n':
                    if self.update_existing_installation():
                        print("\n" + "="*50)
                        print("UPDATE COMPLETED!")
                        print("="*50)
                        print("\nYour installation has been updated with new features!")
                        print("You can now use your existing launcher:")
                        if self.is_windows:
                            print("   • Double-click 'Whisper_Transcriber.bat'")
                        else:
                            print("   • Run './whisper_transcriber.sh'")
                        print("\nNew features available:")
                        print("   • Real-time transcription display")
                        print("   • Better handling of silence/poor audio")
                        print("   • Pause/Resume functionality")
                        print("   • Enhanced progress tracking")
                        return True
                    else:
                        print("\nWARNING: Update failed, proceeding with fresh installation...")
                else:
                    print("Update cancelled by user")
                    return False
            
            # Fresh installation
            print("\n" + "="*50)
            print("FRESH INSTALLATION")
            print("="*50)
            
            # Create structure
            if not self.create_directory_structure():
                print("\nERROR: Failed to create directories")
                return False
            
            # Create scripts
            if not self.create_batch_files():
                print("\nERROR: Failed to create launcher scripts")
                return False
            
            # Ask about environment setup
            print("\n" + "="*50)
            print("INSTALLATION OPTIONS")
            print("="*50)
            
            setup_env = input("Set up virtual environment now? (y/N): ").lower().startswith('y')
            
            if setup_env:
                if not self.setup_environment():
                    print("\nWARNING: Environment setup failed, but you can run it later from the app")
                else:
                    # Ask about model download
                    download_model = input("\nDownload default model (medium, ~769MB)? (y/N): ").lower().startswith('y')
                    if download_model:
                        self.download_model()
            
            print("\n" + "="*50)
            print("INSTALLATION COMPLETED!")
            print("="*50)
            
            print("\nNEXT STEPS:")
            if self.is_windows:
                print("   1. Double-click 'Whisper_Transcriber.bat' to run")
            else:
                print("   1. Run './whisper_transcriber.sh' to start")
            
            if not setup_env:
                print("   2. Click 'Setup Environment' in the app")
                print("   3. Click 'Download Models' to get AI models")
            
            print("   4. Select audio/video file and start transcribing!")
            
            print(f"\nInstallation location: {self.app_dir}")
            print("See README.md for detailed instructions")
            
            return True
            
        except KeyboardInterrupt:
            print("\n\nInstallation cancelled by user")
            return False
        except Exception as e:
            print(f"\nERROR: Installation failed: {e}")
            return False

def main():
    """Main installer entry point"""
    installer = WhisperInstaller()
    
    try:
        success = installer.run_installation()
        
        if success:
            print("\nReady to transcribe with new live features!")
            input("\nPress Enter to exit...")
        else:
            print("\nInstallation incomplete")
            input("\nPress Enter to exit...")
        
    except Exception as e:
        print(f"\nFatal error: {e}")
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()