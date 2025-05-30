#!/usr/bin/env python3
"""
Whisper Transcriber Pro - Installation Script
This script sets up the complete application with virtual environment
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import zipfile
import shutil

class WhisperInstaller:
    def __init__(self):
        self.app_name = "Whisper Transcriber Pro"
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
    
    def create_desktop_shortcut(self):
        """Create desktop shortcut"""
        print("\nCreating desktop shortcut...")
        
        try:
            if self.is_windows:
                # Windows shortcut creation
                desktop = Path.home() / "Desktop"
                shortcut_path = desktop / "Whisper Transcriber Pro.lnk"
                
                # Try to create shortcut using COM
                try:
                    import win32com.client
                    shell = win32com.client.Dispatch("WScript.Shell")
                    shortcut = shell.CreateShortCut(str(shortcut_path))
                    shortcut.Targetpath = str(self.app_dir / "Whisper_Transcriber.bat")
                    shortcut.WorkingDirectory = str(self.app_dir)
                    shortcut.IconLocation = str(self.app_dir / "Whisper_Transcriber.bat")
                    shortcut.save()
                    print("   SUCCESS: Desktop shortcut created")
                except ImportError:
                    print("   WARNING: Could not create shortcut (win32com not available)")
                    print(f"   MANUAL: Create shortcut to {self.app_dir / 'Whisper_Transcriber.bat'}")
            
            else:
                # Linux desktop file
                desktop_dirs = [
                    Path.home() / "Desktop",
                    Path.home() / ".local" / "share" / "applications"
                ]
                
                desktop_content = f'''[Desktop Entry]
Name=Whisper Transcriber Pro
Comment=AI-powered audio/video transcription
Exec={self.app_dir / "whisper_transcriber.sh"}
Icon=audio-x-generic
Terminal=false
Type=Application
Categories=AudioVideo;Audio;
'''
                
                for desktop_dir in desktop_dirs:
                    if desktop_dir.exists():
                        desktop_file = desktop_dir / "whisper-transcriber.desktop"
                        with open(desktop_file, 'w') as f:
                            f.write(desktop_content)
                        desktop_file.chmod(0o755)
                        print(f"   SUCCESS: Created: {desktop_file}")
                        break
        
        except Exception as e:
            print(f"   WARNING: Could not create desktop shortcut: {e}")
        
        return True
    
    def create_readme(self):
        """Create README file with instructions"""
        print("\nCreating README...")
        
        readme_content = f'''# {self.app_name}

## Quick Start

### Windows:
1. Double-click `Whisper_Transcriber.bat` to run the application
2. If first time, click "Setup Environment" in the app
3. Select your audio/video file and start transcribing!

### Linux/Mac:
1. Run `./whisper_transcriber.sh` from terminal
2. If first time, click "Setup Environment" in the app  
3. Select your audio/video file and start transcribing!

## Features

- GPU acceleration for faster processing
- Multiple output formats (Text, SRT, VTT, Detailed)
- Multiple AI models (tiny to large)
- Multi-language support
- Smart text cleaning and formatting
- Advanced customization options

## Supported Formats

**Audio:** MP3, WAV, FLAC, M4A, AAC, OGG, WMA
**Video:** MP4, AVI, MKV, MOV, WMV, FLV, WEBM

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB+ recommended)
- 2GB free disk space
- NVIDIA GPU (optional, for acceleration)

## Troubleshooting

### Environment Setup Issues:
1. Run as administrator/sudo if permission errors
2. Check internet connection for package downloads
3. Ensure Python and pip are in PATH

### GPU Not Detected:
1. Install NVIDIA drivers
2. Verify CUDA installation
3. Restart application

### ffmpeg Not Found:
Download from: https://ffmpeg.org/download.html

## Manual Installation

If automatic setup fails:

```bash
# Create virtual environment
python -m venv whisper_env

# Activate environment
# Windows: whisper_env\\Scripts\\activate
# Linux/Mac: source whisper_env/bin/activate

# Install packages
pip install openai-whisper torch torchaudio tqdm

# For GPU support:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Support

For issues and updates, check the application's help menu or documentation.

---
Generated by installer on {self.get_timestamp()}
'''
        
        readme_file = self.app_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print("   SUCCESS: README.md created")
        return True
    
    def get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def setup_environment(self):
        """Set up the virtual environment"""
        print("\nSetting up virtual environment...")
        
        # Import environment manager
        sys.path.append(str(self.app_dir / "src"))
        from src.environment_manager import EnvironmentManager
        
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
    
    def create_uninstaller(self):
        """Create uninstall script"""
        print("\nCreating uninstaller...")
        
        uninstall_content = f'''#!/usr/bin/env python3
"""
Whisper Transcriber Pro - Uninstaller
"""

import shutil
import os
from pathlib import Path

def uninstall():
    app_dir = Path(__file__).parent
    
    print("Uninstalling Whisper Transcriber Pro...")
    
    # Remove virtual environment
    venv_dir = app_dir / "whisper_env"
    if venv_dir.exists():
        print("   Removing virtual environment...")
        shutil.rmtree(venv_dir)
    
    # Remove logs and temp
    for directory in ["logs", "temp"]:
        dir_path = app_dir / directory
        if dir_path.exists():
            print(f"   Removing {{directory}}...")
            shutil.rmtree(dir_path)
    
    # Remove settings
    settings_file = app_dir / "settings.json"
    if settings_file.exists():
        print("   Removing settings...")
        settings_file.unlink()
    
    print("SUCCESS: Uninstallation completed")
    print("INFO: Application files remain in:", app_dir)
    print("   You can safely delete the entire folder")
    
    input("Press Enter to exit...")

if __name__ == "__main__":
    uninstall()
'''
        
        uninstall_file = self.app_dir / "uninstall.py"
        with open(uninstall_file, 'w', encoding='utf-8') as f:
            f.write(uninstall_content)
        
        print("   SUCCESS: uninstall.py created")
        return True
    
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
            
            # Create structure
            if not self.create_directory_structure():
                print("\nERROR: Failed to create directories")
                return False
            
            # Create scripts
            if not self.create_batch_files():
                print("\nERROR: Failed to create launcher scripts")
                return False
            
            # Create README
            if not self.create_readme():
                print("\nERROR: Failed to create README")
                return False
            
            # Create uninstaller
            if not self.create_uninstaller():
                print("\nERROR: Failed to create uninstaller")
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
                        self.download_default_model()
            
            # Create desktop shortcut
            create_shortcut = input("\nCreate desktop shortcut? (Y/n): ").lower()
            if create_shortcut != 'n':
                self.create_desktop_shortcut()
            
            print("\n" + "="*50)
            print("INSTALLATION COMPLETED!")
            print("="*50)
            
            print("\nNEXT STEPS:")
            if self.is_windows:
                print(f"   1. Double-click 'Whisper_Transcriber.bat' to run")
            else:
                print(f"   1. Run './whisper_transcriber.sh' to start")
            
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
            print("\nReady to transcribe!")
            input("\nPress Enter to exit...")
        else:
            print("\nInstallation incomplete")
            input("\nPress Enter to exit...")
        
    except Exception as e:
        print(f"\nFatal error: {e}")
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()