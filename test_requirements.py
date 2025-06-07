
#!/usr/bin/env python3
"""
Test requirements.txt validity
Quick script to check if all package versions in requirements.txt are valid
"""

import subprocess
import sys
from pathlib import Path

def test_package_availability():
    """Test if packages in requirements.txt are available on PyPI"""
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    print("🔍 Testing package availability...")
    print("=" * 50)
    
    failed_packages = []
    
    with open(requirements_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        # Skip comments and empty lines
        if not line or line.startswith('#'):
            continue
            
        # Extract package name (before >= or ==)
        if '>=' in line:
            package_name = line.split('>=')[0].strip()
            version_spec = line.split('>=')[1].strip()
        elif '==' in line:
            package_name = line.split('==')[0].strip()
            version_spec = line.split('==')[1].strip()
        else:
            package_name = line.strip()
            version_spec = "any"
        
        print(f"Testing {package_name} {version_spec}...", end=" ")
        
        try:
            # Test if package exists on PyPI
            result = subprocess.run(
                [sys.executable, "-m", "pip", "index", "versions", package_name],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("✅ Available")
            else:
                print("❌ Not found")
                failed_packages.append(f"{package_name} {version_spec}")
                
        except subprocess.TimeoutExpired:
            print("⏰ Timeout")
            failed_packages.append(f"{package_name} {version_spec} (timeout)")
        except Exception as e:
            print(f"❌ Error: {e}")
            failed_packages.append(f"{package_name} {version_spec} (error)")
    
    print("=" * 50)
    
    if failed_packages:
        print("❌ FAILED PACKAGES:")
        for pkg in failed_packages:
            print(f"  • {pkg}")
        return False
    else:
        print("✅ ALL PACKAGES AVAILABLE!")
        return True

def suggest_fixes():
    """Suggest fixes for common package issues"""
    print("\n💡 COMMON FIXES:")
    print("1. For openai-whisper issues:")
    print("   pip install openai-whisper --no-cache-dir")
    print("2. For torch issues:")
    print("   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("3. For general issues:")
    print("   pip install --upgrade pip")
    print("   pip cache purge")

def main():
    print("Requirements.txt Package Validator")
    print("=" * 50)
    
    success = test_package_availability()
    
    if not success:
        suggest_fixes()
    
    print(f"\n{'✅ Ready to install!' if success else '❌ Fix issues before installing'}")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()