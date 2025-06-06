"""
Unit tests for utility functions
"""

import pytest
from unittest.mock import patch, Mock, mock_open
from pathlib import Path
import subprocess
import platform
import tempfile
import os
from datetime import datetime

from src.utils import (
    get_file_info, get_media_duration, format_duration, format_time, 
    format_file_size, check_ffmpeg_installed, check_ffprobe_installed,
    get_system_info, validate_audio_file, create_output_filename,
    safe_filename, estimate_processing_time, check_disk_space,
    open_file_explorer, copy_to_clipboard, get_temp_directory,
    cleanup_temp_files, log_error
)


class TestUtilityFunctions:
    """Test cases for utility functions"""
    
    def test_format_duration_seconds(self):
        """Test duration formatting for seconds"""
        assert format_duration(30) == "0:30"
        assert format_duration(59) == "0:59"
        assert format_duration(0) == "0:00"
    
    def test_format_duration_minutes(self):
        """Test duration formatting for minutes"""
        assert format_duration(60) == "1:00"
        assert format_duration(65) == "1:05"
        assert format_duration(125) == "2:05"
    
    def test_format_duration_hours(self):
        """Test duration formatting for hours"""
        assert format_duration(3600) == "1:00:00"
        assert format_duration(3665) == "1:01:05"
        assert format_duration(7325) == "2:02:05"
    
    def test_format_duration_edge_cases(self):
        """Test duration formatting edge cases"""
        assert format_duration(-1) == "Unknown"
        assert format_duration(0.5) == "0:00"
        assert format_duration(59.9) == "0:59"
    
    def test_format_time_seconds(self):
        """Test time formatting for seconds"""
        assert format_time(30) == "30.0s"
        assert format_time(45.5) == "45.5s"
        assert format_time(1) == "1.0s"
    
    def test_format_time_minutes(self):
        """Test time formatting for minutes"""
        assert format_time(90) == "1.5m"
        assert format_time(120) == "2.0m"
        assert format_time(150) == "2.5m"
    
    def test_format_time_hours(self):
        """Test time formatting for hours"""
        assert format_time(3600) == "1.0h"
        assert format_time(7200) == "2.0h"
        assert format_time(5400) == "1.5h"
    
    def test_format_file_size_bytes(self):
        """Test file size formatting for bytes"""
        assert format_file_size(512) == "512.0 B"
        assert format_file_size(1000) == "1000.0 B"
        assert format_file_size(0) == "0.0 B"
    
    def test_format_file_size_kilobytes(self):
        """Test file size formatting for kilobytes"""
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(2048) == "2.0 KB"
        assert format_file_size(1536) == "1.5 KB"
    
    def test_format_file_size_megabytes(self):
        """Test file size formatting for megabytes"""
        assert format_file_size(1024*1024) == "1.0 MB"
        assert format_file_size(1024*1024*2.5) == "2.5 MB"
    
    def test_format_file_size_gigabytes(self):
        """Test file size formatting for gigabytes"""
        assert format_file_size(1024*1024*1024) == "1.0 GB"
        assert format_file_size(1024*1024*1024*1.5) == "1.5 GB"
    
    def test_format_file_size_large(self):
        """Test file size formatting for very large sizes"""
        tb_size = 1024*1024*1024*1024
        assert format_file_size(tb_size) == "1.0 TB"
        
        pb_size = tb_size * 1024
        assert format_file_size(pb_size) == "1.0 PB"
    
    @patch('subprocess.run')
    def test_check_ffmpeg_installed_true(self, mock_run):
        """Test FFmpeg detection when installed"""
        mock_run.return_value = Mock(returncode=0)
        
        result = check_ffmpeg_installed()
        
        assert result is True
        mock_run.assert_called_once_with(['ffmpeg', '-version'], capture_output=True, timeout=10)
    
    @patch('subprocess.run')
    def test_check_ffmpeg_installed_false(self, mock_run):
        """Test FFmpeg detection when not installed"""
        mock_run.side_effect = FileNotFoundError()
        
        result = check_ffmpeg_installed()
        
        assert result is False
    
    @patch('subprocess.run')
    def test_check_ffmpeg_installed_error(self, mock_run):
        """Test FFmpeg detection with subprocess error"""
        mock_run.return_value = Mock(returncode=1)
        
        result = check_ffmpeg_installed()
        
        assert result is False
    
    @patch('subprocess.run')
    def test_check_ffprobe_installed_true(self, mock_run):
        """Test FFprobe detection when installed"""
        mock_run.return_value = Mock(returncode=0)
        
        result = check_ffprobe_installed()
        
        assert result is True
        mock_run.assert_called_once_with(['ffprobe', '-version'], capture_output=True, timeout=10)
    
    @patch('subprocess.run')
    def test_check_ffprobe_installed_false(self, mock_run):
        """Test FFprobe detection when not installed"""
        mock_run.side_effect = FileNotFoundError()
        
        result = check_ffprobe_installed()
        
        assert result is False
    
    @patch('subprocess.run')
    def test_get_media_duration_success(self, mock_run):
        """Test successful media duration extraction"""
        mock_run.return_value = Mock(returncode=0, stdout="125.5")
        
        result = get_media_duration("test.mp3")
        
        assert result == "2:05"
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_get_media_duration_failure(self, mock_run):
        """Test media duration extraction failure"""
        mock_run.return_value = Mock(returncode=1, stdout="")
        
        result = get_media_duration("test.mp3")
        
        assert result is None
    
    @patch('subprocess.run')
    def test_get_media_duration_exception(self, mock_run):
        """Test media duration extraction with exception"""
        mock_run.side_effect = subprocess.TimeoutExpired(['ffprobe'], 30)
        
        result = get_media_duration("test.mp3")
        
        assert result is None
    
    def test_get_file_info_nonexistent(self):
        """Test file info for non-existent file"""
        result = get_file_info("/nonexistent/file.mp3")
        
        assert result == "File not found"
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.stat')
    @patch('src.utils.get_media_duration')
    def test_get_file_info_success(self, mock_duration, mock_stat, mock_exists):
        """Test successful file info extraction"""
        mock_exists.return_value = True
        mock_stat.return_value = Mock(
            st_size=1024*1024*5,  # 5MB
            st_mtime=1640995200   # 2022-01-01 00:00:00
        )
        mock_duration.return_value = "2:30"
        
        result = get_file_info("test.mp3")
        
        assert "5.0 MB" in result
        assert "2:30" in result
        assert "2022-01-01" in result
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.stat')
    def test_get_file_info_no_duration(self, mock_stat, mock_exists):
        """Test file info without duration"""
        mock_exists.return_value = True
        mock_stat.return_value = Mock(
            st_size=1024*500,  # 500KB
            st_mtime=1640995200
        )
        
        with patch('src.utils.get_media_duration', return_value=None):
            result = get_file_info("test.txt")
        
        assert "500.0 KB" in result
        assert "Duration" not in result
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.stat')
    def test_get_file_info_exception(self, mock_stat, mock_exists):
        """Test file info with exception"""
        mock_exists.return_value = True
        mock_stat.side_effect = OSError("Permission denied")
        
        result = get_file_info("test.mp3")
        
        assert "Error reading file" in result
    
    @patch('psutil.cpu_count')
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_get_system_info_with_psutil(self, mock_disk, mock_memory, mock_cpu_percent, mock_cpu_count):
        """Test system info collection with psutil"""
        mock_cpu_count.return_value = 8
        mock_cpu_percent.return_value = 25.5
        mock_memory.return_value = Mock(
            total=8*1024**3,     # 8GB
            available=4*1024**3   # 4GB available
        )
        mock_disk.return_value = Mock(
            total=500*1024**3,    # 500GB
            free=200*1024**3      # 200GB free
        )
        
        info = get_system_info()
        
        assert info['cpu_count'] == 8
        assert info['cpu_percent'] == 25.5
        assert info['memory_total_gb'] == pytest.approx(8, rel=1e-2)
        assert info['memory_available_gb'] == pytest.approx(4, rel=1e-2)
        assert info['disk_total_gb'] == pytest.approx(500, rel=1e-2)
        assert info['disk_free_gb'] == pytest.approx(200, rel=1e-2)
        assert 'platform' in info
    
    def test_get_system_info_without_psutil(self):
        """Test system info collection without psutil"""
        with patch.dict('sys.modules', {'psutil': None}):
            # This will trigger the ImportError fallback
            with patch('src.utils.psutil', side_effect=ImportError()):
                info = get_system_info()
        
        # Should have basic info even without psutil
        assert 'platform' in info
        assert 'platform_version' in info
        assert 'architecture' in info
    
    def test_validate_audio_file_nonexistent(self):
        """Test audio file validation for non-existent file"""
        is_valid, message = validate_audio_file("/nonexistent/file.mp3")
        
        assert is_valid is False
        assert "does not exist" in message
    
    def test_validate_audio_file_not_file(self, temp_dir):
        """Test audio file validation for directory"""
        is_valid, message = validate_audio_file(str(temp_dir))
        
        assert is_valid is False
        assert "not a file" in message
    
    def test_validate_audio_file_unsupported_format(self, temp_dir):
        """Test audio file validation for unsupported format"""
        unsupported_file = temp_dir / "test.xyz"
        unsupported_file.write_text("fake content")
        
        is_valid, message = validate_audio_file(str(unsupported_file))
        
        assert is_valid is False
        assert "Unsupported format" in message
    
    @patch('subprocess.run')
    def test_validate_audio_file_supported_format(self, mock_run, temp_dir):
        """Test audio file validation for supported format"""
        mock_run.return_value = Mock(returncode=0, stdout="mp3")
        
        audio_file = temp_dir / "test.mp3"
        audio_file.write_bytes(b"fake audio content")
        
        is_valid, message = validate_audio_file(str(audio_file))
        
        assert is_valid is True
        assert "valid" in message
    
    @patch('subprocess.run')
    def test_validate_audio_file_corrupted(self, mock_run, temp_dir):
        """Test audio file validation for corrupted file"""
        mock_run.return_value = Mock(returncode=1, stderr="Invalid data")
        
        audio_file = temp_dir / "test.mp3"
        audio_file.write_bytes(b"corrupted content")
        
        is_valid, message = validate_audio_file(str(audio_file))
        
        assert is_valid is False
        assert "corrupted" in message
    
    @patch('subprocess.run')
    def test_validate_audio_file_no_ffprobe(self, mock_run, temp_dir):
        """Test audio file validation without ffprobe"""
        mock_run.side_effect = FileNotFoundError()
        
        audio_file = temp_dir / "test.mp3"
        audio_file.write_bytes(b"fake audio content")
        
        is_valid, message = validate_audio_file(str(audio_file))
        
        # Should still be valid if extension is supported
        assert is_valid is True
    
    def test_create_output_filename_simple(self, temp_dir):
        """Test output filename creation"""
        result = create_output_filename("/path/to/audio.mp3", str(temp_dir))
        
        expected = temp_dir / "audio.txt"
        assert result == expected
        assert temp_dir.exists()  # Directory should be created
    
    def test_create_output_filename_with_suffix(self, temp_dir):
        """Test output filename creation with suffix"""
        result = create_output_filename("/path/to/audio.mp3", str(temp_dir), "detailed", ".json")
        
        expected = temp_dir / "audio_detailed.json"
        assert result == expected
    
    def test_create_output_filename_no_suffix(self, temp_dir):
        """Test output filename creation without suffix"""
        result = create_output_filename("/path/to/audio.mp3", str(temp_dir), "", ".srt")
        
        expected = temp_dir / "audio.srt"
        assert result == expected
    
    def test_safe_filename_invalid_characters(self):
        """Test filename sanitization with invalid characters"""
        unsafe = 'test<>:"/\\|?*file.txt'
        safe = safe_filename(unsafe)
        
        # Should replace invalid characters with underscores
        assert '<' not in safe
        assert '>' not in safe
        assert ':' not in safe
        assert '"' not in safe
        assert '/' not in safe
        assert '\\' not in safe
        assert '|' not in safe
        assert '?' not in safe
        assert '*' not in safe
        assert 'test' in safe
        assert 'file.txt' in safe
    
    def test_safe_filename_control_characters(self):
        """Test filename sanitization with control characters"""
        unsafe = 'test\x00\x1f\x7ffile.txt'
        safe = safe_filename(unsafe)
        
        # Should remove control characters
        assert '\x00' not in safe
        assert '\x1f' not in safe
        assert '\x7f' not in safe
        assert 'testfile.txt' == safe
    
    def test_safe_filename_long_name(self):
        """Test filename sanitization with very long name"""
        long_name = 'a' * 300 + '.txt'
        safe = safe_filename(long_name)
        
        # Should be limited to 255 characters
        assert len(safe) <= 255
        assert safe.endswith('.txt')
    
    def test_safe_filename_already_safe(self):
        """Test filename sanitization with already safe name"""
        safe_name = 'perfectly_safe_filename.mp3'
        result = safe_filename(safe_name)
        
        assert result == safe_name
    
    def test_estimate_processing_time_gpu(self):
        """Test processing time estimation with GPU"""
        # 1 hour audio with medium model on GPU
        estimated = estimate_processing_time(3600, 'medium', use_gpu=True)
        
        # Should be much faster than real-time (5x speed = 720 seconds)
        assert estimated == pytest.approx(720, rel=0.1)
    
    def test_estimate_processing_time_cpu(self):
        """Test processing time estimation with CPU"""
        # 1 hour audio with medium model on CPU
        estimated = estimate_processing_time(3600, 'medium', use_gpu=False)
        
        # Should be slower than GPU (1.5x speed = 2400 seconds)
        assert estimated == pytest.approx(2400, rel=0.1)
    
    def test_estimate_processing_time_different_models(self):
        """Test processing time estimation for different models"""
        duration = 600  # 10 minutes
        
        tiny_time = estimate_processing_time(duration, 'tiny', use_gpu=True)
        large_time = estimate_processing_time(duration, 'large', use_gpu=True)
        
        # Tiny should be much faster than large
        assert tiny_time < large_time
    
    def test_estimate_processing_time_unknown_model(self):
        """Test processing time estimation for unknown model"""
        estimated = estimate_processing_time(3600, 'unknown', use_gpu=True)
        
        # Should use default multiplier of 1 (same as real-time)
        assert estimated == 3600
    
    @patch('shutil.disk_usage')
    def test_check_disk_space_sufficient(self, mock_disk_usage):
        """Test disk space check with sufficient space"""
        # Mock 10GB free space
        mock_disk_usage.return_value = (1000*1024**3, 500*1024**3, 10*1024**3)
        
        has_space, free_mb = check_disk_space("/tmp", required_mb=1000)
        
        assert has_space is True
        assert free_mb == pytest.approx(10*1024, rel=1e-2)  # ~10GB in MB
    
    @patch('shutil.disk_usage')
    def test_check_disk_space_insufficient(self, mock_disk_usage):
        """Test disk space check with insufficient space"""
        # Mock 100MB free space
        mock_disk_usage.return_value = (1000*1024**3, 999*1024**3, 100*1024**2)
        
        has_space, free_mb = check_disk_space("/tmp", required_mb=1000)
        
        assert has_space is False
        assert free_mb == pytest.approx(100, rel=1e-2)
    
    @patch('shutil.disk_usage')
    def test_check_disk_space_exception(self, mock_disk_usage):
        """Test disk space check with exception"""
        mock_disk_usage.side_effect = OSError("Access denied")
        
        has_space, free_mb = check_disk_space("/tmp", required_mb=1000)
        
        # Should assume enough space if check fails
        assert has_space is True
        assert free_mb == 0
    
    @patch('platform.system')
    @patch('os.startfile')
    def test_open_file_explorer_windows(self, mock_startfile, mock_system):
        """Test file explorer opening on Windows"""
        mock_system.return_value = "Windows"
        
        result = open_file_explorer("C:/test/path")
        
        assert result is True
        mock_startfile.assert_called_once_with("C:/test/path")
    
    @patch('platform.system')
    @patch('subprocess.run')
    def test_open_file_explorer_macos(self, mock_run, mock_system):
        """Test file explorer opening on macOS"""
        mock_system.return_value = "Darwin"
        
        result = open_file_explorer("/test/path")
        
        assert result is True
        mock_run.assert_called_once_with(["open", "/test/path"])
    
    @patch('platform.system')
    @patch('subprocess.run')
    def test_open_file_explorer_linux(self, mock_run, mock_system):
        """Test file explorer opening on Linux"""
        mock_system.return_value = "Linux"
        
        result = open_file_explorer("/test/path")
        
        assert result is True
        mock_run.assert_called_once_with(["xdg-open", "/test/path"])
    
    @patch('platform.system')
    @patch('os.startfile')
    def test_open_file_explorer_exception(self, mock_startfile, mock_system):
        """Test file explorer opening with exception"""
        mock_system.return_value = "Windows"
        mock_startfile.side_effect = OSError("No application associated")
        
        result = open_file_explorer("C:/test/path")
        
        assert result is False
    
    @patch('tkinter.Tk')
    def test_copy_to_clipboard_success(self, mock_tk):
        """Test successful clipboard copy"""
        # Mock Tkinter root window
        mock_root = Mock()
        mock_tk.return_value = mock_root
        
        result = copy_to_clipboard("Test text")
        
        assert result is True
        mock_root.clipboard_clear.assert_called_once()
        mock_root.clipboard_append.assert_called_once_with("Test text")
        mock_root.update.assert_called_once()
        mock_root.destroy.assert_called_once()
    
    @patch('tkinter.Tk')
    def test_copy_to_clipboard_exception(self, mock_tk):
        """Test clipboard copy with exception"""
        mock_tk.side_effect = Exception("Clipboard error")
        
        result = copy_to_clipboard("Test text")
        
        assert result is False
    
    @patch('tempfile.gettempdir')
    @patch('pathlib.Path.mkdir')
    def test_get_temp_directory(self, mock_mkdir, mock_tempdir):
        """Test temporary directory creation"""
        mock_tempdir.return_value = "/tmp"
        
        temp_dir = get_temp_directory()
        
        assert temp_dir.name == "whisper_transcriber"
        assert str(temp_dir).startswith("/tmp")
        mock_mkdir.assert_called_once_with(exist_ok=True)
    
    @patch('pathlib.Path.exists')
    @patch('shutil.rmtree')
    @patch('src.utils.get_temp_directory')
    def test_cleanup_temp_files_success(self, mock_get_temp, mock_rmtree, mock_exists):
        """Test successful temp file cleanup"""
        mock_temp_dir = Mock()
        mock_get_temp.return_value = mock_temp_dir
        mock_exists.return_value = True
        
        result = cleanup_temp_files()
        
        assert result is True
        mock_rmtree.assert_called_once_with(mock_temp_dir)
    
    @patch('pathlib.Path.exists')
    @patch('src.utils.get_temp_directory')
    def test_cleanup_temp_files_no_directory(self, mock_get_temp, mock_exists):
        """Test temp file cleanup when directory doesn't exist"""
        mock_temp_dir = Mock()
        mock_get_temp.return_value = mock_temp_dir
        mock_exists.return_value = False
        
        result = cleanup_temp_files()
        
        assert result is True
    
    @patch('shutil.rmtree')
    @patch('pathlib.Path.exists')
    @patch('src.utils.get_temp_directory')
    def test_cleanup_temp_files_exception(self, mock_get_temp, mock_exists, mock_rmtree):
        """Test temp file cleanup with exception"""
        mock_temp_dir = Mock()
        mock_get_temp.return_value = mock_temp_dir
        mock_exists.return_value = True
        mock_rmtree.side_effect = OSError("Permission denied")
        
        result = cleanup_temp_files()
        
        assert result is False
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    @patch('datetime.datetime')
    def test_log_error_success(self, mock_datetime, mock_mkdir, mock_file):
        """Test successful error logging"""
        mock_datetime.now.return_value.strftime.return_value = "2023-01-01 12:00:00"
        
        result = log_error("Test error message", "TestType")
        
        assert result is True
        mock_mkdir.assert_called_once_with(exist_ok=True)
        mock_file.assert_called_once()
        
        # Check that error was written with timestamp
        written_content = mock_file().write.call_args[0][0]
        assert "2023-01-01 12:00:00" in written_content
        assert "TestType" in written_content
        assert "Test error message" in written_content
    
    @patch('builtins.open')
    def test_log_error_exception(self, mock_file):
        """Test error logging with exception"""
        mock_file.side_effect = OSError("Write error")
        
        result = log_error("Test error message")
        
        assert result is False


class TestUtilityFunctionsIntegration:
    """Integration tests for utility functions"""
    
    def test_file_operations_real_files(self, temp_dir):
        """Test file operations with real files"""
        # Create test file
        test_file = temp_dir / "test_audio.mp3"
        test_content = b"fake audio content" * 1000  # Make it larger
        test_file.write_bytes(test_content)
        
        # Test file info
        info = get_file_info(str(test_file))
        assert "KB" in info or "MB" in info
        assert "Modified:" in info
        
        # Test file validation
        is_valid, message = validate_audio_file(str(test_file))
        # Should be valid based on extension even without ffprobe
        assert is_valid is True or "corrupted" in message
    
    def test_filename_safety_comprehensive(self):
        """Test comprehensive filename safety"""
        dangerous_names = [
            "normal_file.txt",
            "file with spaces.mp3",
            "file<>:|?*.wav",
            "very" + "long" * 100 + ".mp4",
            "file\x00with\x1fcontrol\x7fchars.flac",
            "файл_с_unicode.txt",  # Unicode filename
            ".hidden_file",
            "file..with..dots.mp3",
            "CON.txt",  # Windows reserved name
            "file\nwith\nnewlines.wav"
        ]
        
        for dangerous in dangerous_names:
            safe = safe_filename(dangerous)
            
            # Should not contain dangerous characters
            dangerous_chars = '<>:"/\\|?*\x00-\x1f\x7f-\x9f'
            for char in dangerous_chars:
                assert char not in safe
            
            # Should not be too long
            assert len(safe) <= 255
            
            # Should not be empty (unless original was empty)
            if dangerous.strip():
                assert safe.strip()
    
    def test_size_formatting_comprehensive(self):
        """Test comprehensive size formatting"""
        test_cases = [
            (0, "0.0 B"),
            (1, "1.0 B"),
            (512, "512.0 B"),
            (1023, "1023.0 B"),
            (1024, "1.0 KB"),
            (1536, "1.5 KB"),
            (1024*1024, "1.0 MB"),
            (1024*1024*1.5, "1.5 MB"),
            (1024*1024*1024, "1.0 GB"),
            (1024*1024*1024*1024, "1.0 TB"),
            (1024*1024*1024*1024*1024, "1.0 PB")
        ]
        
        for size_bytes, expected in test_cases:
            result = format_file_size(size_bytes)
            assert result == expected
    
    def test_time_formatting_comprehensive(self):
        """Test comprehensive time formatting"""
        test_cases = [
            # (seconds, expected_duration, expected_time)
            (0, "0:00", "0.0s"),
            (30, "0:30", "30.0s"),
            (60, "1:00", "1.0m"),
            (90, "1:30", "1.5m"),
            (3600, "1:00:00", "1.0h"),
            (3665, "1:01:05", "1.0h"),  # Slight over 1 hour
            (7200, "2:00:00", "2.0h")
        ]
        
        for seconds, expected_duration, expected_time in test_cases:
            duration_result = format_duration(seconds)
            time_result = format_time(seconds)
            
            assert duration_result == expected_duration
            assert time_result == expected_time
    
    def test_processing_time_estimation_realistic(self):
        """Test realistic processing time estimations"""
        # Test various file durations and model combinations
        durations = [60, 300, 1800, 3600]  # 1min, 5min, 30min, 1hour
        models = ['tiny', 'base', 'small', 'medium', 'large']
        
        for duration in durations:
            gpu_times = []
            cpu_times = []
            
            for model in models:
                gpu_time = estimate_processing_time(duration, model, use_gpu=True)
                cpu_time = estimate_processing_time(duration, model, use_gpu=False)
                
                gpu_times.append(gpu_time)
                cpu_times.append(cpu_time)
                
                # GPU should always be faster than CPU
                assert gpu_time <= cpu_time
                
                # Processing time should be positive
                assert gpu_time > 0
                assert cpu_time > 0
            
            # Smaller models should generally be faster
            # (though this is a rough heuristic, not always true)
            for i in range(len(models) - 1):
                # Allow some tolerance for the heuristic
                assert gpu_times[i] <= gpu_times[i + 1] * 2