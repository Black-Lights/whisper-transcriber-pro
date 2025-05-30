"""
Utility Functions - Common helper functions
Author: Black-Lights (https://github.com/Black-Lights)
Project: Whisper Transcriber Pro
"""

import os
import subprocess
import platform
from pathlib import Path
from datetime import datetime, timedelta

def get_file_info(file_path):
    """Get comprehensive file information"""
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            return "File not found"
        
        # Basic file info
        stat = file_path.stat()
        size_bytes = stat.st_size
        size_mb = size_bytes / (1024 * 1024)
        
        # Format file size
        if size_mb < 1:
            size_str = f"{size_bytes / 1024:.1f} KB"
        elif size_mb < 1024:
            size_str = f"{size_mb:.1f} MB"
        else:
            size_str = f"{size_mb / 1024:.1f} GB"
        
        # Get duration for media files
        duration_str = get_media_duration(file_path)
        
        # Format modification time
        mod_time = datetime.fromtimestamp(stat.st_mtime)
        mod_str = mod_time.strftime("%Y-%m-%d %H:%M")
        
        info_parts = [
            f"Size: {size_str}",
            f"Modified: {mod_str}"
        ]
        
        if duration_str:
            info_parts.insert(1, f"Duration: {duration_str}")
        
        return " | ".join(info_parts)
        
    except Exception as e:
        return f"Error reading file: {e}"

def get_media_duration(file_path):
    """Get duration of media file using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 
            'format=duration', '-of', 'csv=p=0', str(file_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            duration_seconds = float(result.stdout.strip())
            return format_duration(duration_seconds)
        
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError, FileNotFoundError):
        pass
    
    return None

def format_duration(seconds):
    """Format duration in seconds to readable string"""
    if seconds < 0:
        return "Unknown"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"

def format_time(seconds):
    """Format time for display"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def format_file_size(bytes_size):
    """Format file size in bytes to readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"

def check_ffmpeg_installed():
    """Check if ffmpeg is installed and accessible"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, timeout=10)
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def check_ffprobe_installed():
    """Check if ffprobe is installed and accessible"""
    try:
        result = subprocess.run(['ffprobe', '-version'], 
                              capture_output=True, timeout=10)
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def get_system_info():
    """Get system information"""
    import psutil
    
    try:
        # CPU info
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_total_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        
        # Disk info
        disk = psutil.disk_usage('/')
        disk_total_gb = disk.total / (1024**3)
        disk_free_gb = disk.free / (1024**3)
        
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.architecture()[0],
            'cpu_count': cpu_count,
            'cpu_percent': cpu_percent,
            'memory_total_gb': memory_total_gb,
            'memory_available_gb': memory_available_gb,
            'disk_total_gb': disk_total_gb,
            'disk_free_gb': disk_free_gb
        }
    except ImportError:
        # Fallback if psutil not available
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.architecture()[0]
        }

def validate_audio_file(file_path):
    """Validate if file is a supported audio/video format"""
    supported_extensions = {
        # Audio formats
        '.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma',
        # Video formats  
        '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'
    }
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        return False, "File does not exist"
    
    if not file_path.is_file():
        return False, "Path is not a file"
    
    extension = file_path.suffix.lower()
    if extension not in supported_extensions:
        return False, f"Unsupported format: {extension}"
    
    # Try to get basic info using ffprobe
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 
               'format=format_name', '-of', 'csv=p=0', str(file_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            return False, "File appears to be corrupted or unsupported"
            
    except (subprocess.SubprocessError, FileNotFoundError):
        # ffprobe not available, but extension is supported
        pass
    
    return True, "File appears to be valid"

def create_output_filename(input_file, output_dir, suffix="", extension=".txt"):
    """Create output filename based on input file"""
    input_path = Path(input_file)
    output_path = Path(output_dir)
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create filename
    base_name = input_path.stem
    if suffix:
        filename = f"{base_name}_{suffix}{extension}"
    else:
        filename = f"{base_name}{extension}"
    
    return output_path / filename

def safe_filename(filename):
    """Make filename safe for all operating systems"""
    import re
    
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    
    return filename

def estimate_processing_time(file_duration_seconds, model_size, use_gpu=True):
    """Estimate processing time based on file duration and settings"""
    # Speed multipliers (approximate)
    speed_multipliers = {
        'tiny': 30 if use_gpu else 5,
        'base': 20 if use_gpu else 3,
        'small': 10 if use_gpu else 2,
        'medium': 5 if use_gpu else 1.5,
        'large': 2 if use_gpu else 1
    }
    
    multiplier = speed_multipliers.get(model_size, 1)
    estimated_seconds = file_duration_seconds / multiplier
    
    return estimated_seconds

def check_disk_space(directory, required_mb=1000):
    """Check if there's enough disk space in directory"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(directory)
        free_mb = free / (1024 * 1024)
        
        return free_mb >= required_mb, free_mb
    except:
        return True, 0  # Assume enough space if check fails

def open_file_explorer(path):
    """Open file explorer at given path"""
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", path])
        else:  # Linux
            subprocess.run(["xdg-open", path])
        return True
    except:
        return False

def copy_to_clipboard(text):
    """Copy text to system clipboard"""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # Hide window
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()  # Required for clipboard to work
        root.destroy()
        return True
    except:
        return False

def get_temp_directory():
    """Get temporary directory for the application"""
    import tempfile
    temp_dir = Path(tempfile.gettempdir()) / "whisper_transcriber"
    temp_dir.mkdir(exist_ok=True)
    return temp_dir

def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        temp_dir = get_temp_directory()
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
        return True
    except:
        return False

def log_error(error_message, error_type="General"):
    """Log error to file"""
    try:
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / "errors.log"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {error_type}: {error_message}\n")
        
        return True
    except:
        return False