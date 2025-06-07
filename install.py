#!/usr/bin/env python3
"""
Whisper Transcriber Pro - Fixed Installation Script (v1.2.0)
Author: Black-Lights (https://github.com/Black-Lights)
This script sets up the complete application with proper timeout and error handling
"""

import os
import sys
import subprocess
import platform
import time
from pathlib import Path
import shutil

class WhisperInstaller:
    def __init__(self):
        self.app_name = "Whisper Transcriber Pro v1.2.0"
        self.app_dir = Path(__file__).parent
        self.system = platform.system()
        self.is_windows = self.system == "Windows"

        print(f"{self.app_name} Installer (Fixed Version)")
        print("=" * 60)

    def check_requirements(self):
        """Check system requirements"""
        print("Checking system requirements...")

        # Check Python version
        if sys.version_info < (3, 8):
            print("ERROR: Python 3.8 or higher required")
            return False

        print(f"✓ Python {sys.version.split()[0]} detected")

        # Check pip
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"],
                          capture_output=True, check=True, timeout=10)
            print("✓ pip is available")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print("ERROR: pip is not available")
            return False

        # Check venv
        try:
            import venv
            print("✓ venv module available")
        except ImportError:
            print("ERROR: venv module not available")
            return False

        # Check for ffmpeg (optional but recommended)
        try:
            subprocess.run(["ffmpeg", "-version"],
                          capture_output=True, check=True, timeout=10)
            print("✓ ffmpeg detected")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print("⚠ ffmpeg not detected (optional - some features may be limited)")

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
        """Update existing installation with new dependencies from requirements.txt"""
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

            print(f"   ✓ Virtual environment working: {result.stdout.strip()}")

            # Check if requirements.txt exists
            requirements_file = self.app_dir / "requirements.txt"
            if not requirements_file.exists():
                print("   Creating requirements.txt...")
                self.create_requirements_file()

            # Update from requirements.txt
            print("   Updating packages from requirements.txt (this may take several minutes)...")
            
            result = subprocess.run(
                [str(pip_exe), "install", "--upgrade", "-r", str(requirements_file)],
                capture_output=True, text=True, timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                print("   ✓ Packages updated successfully from requirements.txt")
            else:
                print(f"   ⚠ Some packages could not be updated: {result.stderr[:200]}...")
                # Try individual package updates as fallback
                print("   Trying individual package updates...")
                
                # Install psutil separately as it's critical
                result = subprocess.run([str(pip_exe), "install", "--upgrade", "psutil"],
                                      capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    print("   ✓ psutil updated successfully")

            print("   ✓ Update process completed")
            return True

        except subprocess.TimeoutExpired:
            print("   ⚠ Update timed out - this may happen with large downloads")
            print("   You can continue and finish the update later from the app")
            return True
        except Exception as e:
            print(f"   ERROR: Failed to update installation: {e}")
            return False

    def create_requirements_file(self):
        """Create requirements.txt file if it doesn't exist"""
        requirements_file = self.app_dir / "requirements.txt"
        
        if requirements_file.exists():
            return  # File already exists
            
        requirements_content = """# Whisper Transcriber Pro - Requirements (Stable Production Versions)
# Author: Black-Lights (https://github.com/Black-Lights)
# Project: Whisper Transcriber Pro v1.2.0

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
"""
        
        try:
            with open(requirements_file, 'w', encoding='utf-8') as f:
                f.write(requirements_content)
            print(f"   ✓ Created requirements.txt")
        except Exception as e:
            print(f"   ⚠ Could not create requirements.txt: {e}")

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
            try:
                dir_path.mkdir(exist_ok=True)
                print(f"   ✓ Created: {directory}/")
            except Exception as e:
                print(f"   ERROR: Could not create {directory}/: {e}")
                return False

        return True

    def create_batch_files(self):
        """Create batch/shell files for easy launching"""
        print("\nCreating launcher scripts...")

        try:
            if self.is_windows:
                # Create Windows batch file
                batch_content = f'''@echo off
cd /d "{self.app_dir}"
echo Starting Whisper Transcriber Pro...
call whisper_env\\Scripts\\activate.bat
python main.py
if errorlevel 1 (
    echo.
    echo Error occurred. Press any key to exit.
    pause >nul
) else (
    echo.
    echo Application closed normally.
    pause
)
'''
                batch_file = self.app_dir / "Whisper_Transcriber.bat"
                with open(batch_file, 'w', encoding='utf-8') as f:
                    f.write(batch_content)
                print("   ✓ Created: Whisper_Transcriber.bat")

                # Create setup batch file
                setup_batch = f'''@echo off
cd /d "{self.app_dir}"
echo Running Whisper Transcriber Pro installer...
python install.py
pause
'''
                setup_file = self.app_dir / "Setup.bat"
                with open(setup_file, 'w', encoding='utf-8') as f:
                    f.write(setup_batch)
                print("   ✓ Created: Setup.bat")

            else:
                # Create shell script for Unix/Linux/Mac
                script_content = f'''#!/bin/bash
cd "{self.app_dir}"
echo "Starting Whisper Transcriber Pro..."
source whisper_env/bin/activate
python main.py
read -p "Press Enter to exit..."
'''
                script_file = self.app_dir / "whisper_transcriber.sh"
                with open(script_file, 'w', encoding='utf-8') as f:
                    f.write(script_content)
                script_file.chmod(0o755)  # Make executable
                print("   ✓ Created: whisper_transcriber.sh")

            return True
            
        except Exception as e:
            print(f"   ERROR: Failed to create launcher scripts: {e}")
            return False

    def setup_environment(self):
        """Set up the virtual environment"""
        print("\nSetting up virtual environment...")
        print("⚠ This process may take 20-30 minutes for large downloads")
        print("⚠ Please be patient and do not interrupt the installation")

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
            timestamp = time.strftime("%H:%M:%S")
            print(f"   [{timestamp}] {message}")

        print("\n" + "-" * 50)
        print("ENVIRONMENT SETUP STARTING")
        print("-" * 50)
        
        start_time = time.time()
        success = env_manager.setup_environment(progress_callback)
        end_time = time.time()
        
        print("-" * 50)
        print(f"SETUP COMPLETED IN {int(end_time - start_time)} SECONDS")
        print("-" * 50)

        if success:
            print("   ✓ Environment setup completed successfully!")
            
            # Test the installation
            print("\nTesting installation...")
            try:
                status = env_manager.check_environment()
                if status["torch_installed"]:
                    print("   ✓ PyTorch installed and working")
                if status["whisper_installed"]:
                    print("   ✓ Whisper installed and working")
                if status["gpu_available"]:
                    print("   ✓ GPU acceleration available")
                else:
                    print("   ⚠ Using CPU mode (slower but functional)")
                    
            except Exception as e:
                print(f"   ⚠ Could not test installation: {e}")
                
        else:
            print("   ERROR: Environment setup failed")
            print("   You can try running the installer again or install manually")

        return success

    def download_default_model(self):
        """Download the default medium model"""
        print("\nDownloading default model...")

        try:
            sys.path.append(str(self.app_dir / "src"))
            from src.model_manager import ModelManager

            model_manager = ModelManager()

            def progress_callback(message):
                timestamp = time.strftime("%H:%M:%S")
                print(f"   [{timestamp}] {message}")

            print("   Downloading medium model (~769MB)...")
            success = model_manager.download_models(["medium"], progress_callback)

            if success:
                print("   ✓ Default model downloaded successfully")
            else:
                print("   ⚠ Model download failed (you can download it later in the app)")

            return success

        except Exception as e:
            print(f"   ⚠ Model download failed: {e}")
            print("   You can download models later from within the application")
            return False

    def run_installation(self):
        """Run the complete installation process"""
        try:
            print(f"Installing {self.app_name}...")
            print(f"Target directory: {self.app_dir}")
            print()

            # Check requirements
            if not self.check_requirements():
                print("\n❌ ERROR: System requirements not met")
                return False

            # Check for existing installation
            existing = self.check_existing_installation()

            if existing['venv_exists'] and existing['main_exists']:
                print("\n" + "="*60)
                print("EXISTING INSTALLATION DETECTED")
                print("="*60)
                print("Found existing Whisper Transcriber Pro installation.")
                print("This will update your installation with new features:")
                print("  • Live transcription display")
                print("  • Better process management") 
                print("  • Enhanced audio quality handling")
                print("  • Improved timer system")
                print("  • Fixed timeout and error handling")
                print()

                update_choice = input("Update existing installation? (Y/n): ").lower()
                if update_choice != 'n':
                    if self.update_existing_installation():
                        print("\n" + "="*60)
                        print("✅ UPDATE COMPLETED!")
                        print("="*60)
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
                        print("   • Reliable large file downloads")
                        return True
                    else:
                        print("\n⚠ WARNING: Update failed, proceeding with fresh installation...")
                else:
                    print("Update cancelled by user")
                    return False

            # Fresh installation
            print("\n" + "="*60)
            print("FRESH INSTALLATION")
            print("="*60)

            # Create structure
            if not self.create_directory_structure():
                print("\n❌ ERROR: Failed to create directories")
                return False

            # Create scripts
            if not self.create_batch_files():
                print("\n❌ ERROR: Failed to create launcher scripts")
                return False

            # Ensure requirements.txt exists
            requirements_file = self.app_dir / "requirements.txt"
            if not requirements_file.exists():
                print("\nCreating requirements.txt...")
                self.create_requirements_file()

            # Ask about environment setup
            print("\n" + "="*60)
            print("INSTALLATION OPTIONS")
            print("="*60)
            print("Environment setup includes:")
            print("  • Python virtual environment creation")
            print("  • Package installation from requirements.txt")
            print("  • PyTorch installation (2-3GB download)")
            print("  • OpenAI Whisper installation")
            print("  • Supporting libraries")
            print()
            print("⚠ This process typically takes 20-30 minutes")
            print("⚠ Make sure you have a stable internet connection")
            print()

            setup_env = input("Set up virtual environment now? (y/N): ").lower().startswith('y')

            if setup_env:
                success = self.setup_environment()
                if not success:
                    print("\n⚠ WARNING: Environment setup failed")
                    print("You can:")
                    print("  1. Try running the installer again")
                    print("  2. Use 'Setup Environment' button in the app")
                    print("  3. Check the error messages above for issues")
                    return False
                else:
                    # Ask about model download
                    print("\n" + "="*50)
                    print("MODEL DOWNLOAD")
                    print("="*50)
                    print("The medium model provides the best balance of speed and accuracy.")
                    print("Size: ~769MB | Speed: ~2x real-time on GPU")
                    print()
                    
                    download_model = input("Download default model (medium, ~769MB)? (y/N): ").lower().startswith('y')
                    if download_model:
                        self.download_default_model()

            print("\n" + "="*60)
            print("✅ INSTALLATION COMPLETED!")
            print("="*60)

            print("\n🎉 NEXT STEPS:")
            if self.is_windows:
                print("   1. Double-click 'Whisper_Transcriber.bat' to run the application")
            else:
                print("   1. Run './whisper_transcriber.sh' to start the application")

            if not setup_env:
                print("   2. Click 'Setup Environment' in the app to install dependencies")
                print("   3. Click 'Download Models' to get AI models")
                print("   4. Select audio/video file and start transcribing!")
            else:
                print("   2. Select an audio/video file in the application")
                print("   3. Choose your settings and start transcribing!")

            print(f"\n📁 Installation location: {self.app_dir}")
            print("📖 See README.md for detailed instructions")
            print("🐛 If you encounter issues, check the logs/ directory")

            return True

        except KeyboardInterrupt:
            print("\n\n⚠ Installation cancelled by user")
            print("You can run the installer again later to complete setup")
            return False
        except Exception as e:
            print(f"\n❌ ERROR: Installation failed: {e}")
            print("Please check the error message above and try again")
            return False

    def show_completion_summary(self, setup_env_success):
        """Show a summary of what was installed"""
        print("\n" + "="*60)
        print("INSTALLATION SUMMARY")
        print("="*60)
        
        print("✅ Created application structure")
        print("✅ Created launcher scripts")
        
        if setup_env_success:
            print("✅ Virtual environment configured")
            print("✅ Dependencies installed")
            
            # Check what was actually installed
            try:
                sys.path.append(str(self.app_dir / "src"))
                from src.environment_manager import EnvironmentManager
                env_manager = EnvironmentManager()
                status = env_manager.check_environment()
                
                if status.get("torch_installed"):
                    print("✅ PyTorch installed")
                if status.get("whisper_installed"):
                    print("✅ OpenAI Whisper installed")
                if status.get("gpu_available"):
                    print("✅ GPU acceleration available")
                else:
                    print("⚠ CPU mode (GPU not available or not configured)")
                    
            except Exception:
                print("⚠ Could not verify installation status")
        else:
            print("⚠ Environment setup skipped - run later from app")

def main():
    """Main installer entry point"""
    installer = WhisperInstaller()

    try:
        print("Welcome to Whisper Transcriber Pro!")
        print("This installer will set up everything you need for AI-powered transcription.")
        print()
        
        success = installer.run_installation()

        if success:
            print("\n🎉 Ready to transcribe with professional AI features!")
            print("💡 Tip: Start with the 'medium' model for best results")
            print()
        else:
            print("\n⚠ Installation incomplete")
            print("💡 You can run this installer again to retry")
            print()

        input("Press Enter to exit...")

    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        print("Please report this issue if it persists")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()