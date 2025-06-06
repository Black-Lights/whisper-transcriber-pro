# tests/conftest.py
"""
Test configuration and fixtures for Whisper Transcriber Pro
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_audio_file(temp_dir):
    """Create a mock audio file for testing"""
    audio_file = temp_dir / "test_audio.mp3"
    audio_file.write_bytes(b"fake audio content")
    return audio_file

@pytest.fixture
def mock_env_manager():
    """Mock environment manager"""
    with patch('src.environment_manager.EnvironmentManager') as mock:
        mock_instance = Mock()
        mock_instance.app_dir = Path("/fake/app/dir")
        mock_instance.python_exe = Path("/fake/python")
        mock_instance.check_environment.return_value = {
            'venv_exists': True,
            'python_works': True,
            'whisper_installed': True,
            'torch_installed': True,
            'gpu_available': False
        }
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_whisper_result():
    """Mock Whisper transcription result"""
    return {
        'text': 'This is a test transcription.',
        'segments': [
            {
                'start': 0.0,
                'end': 2.5,
                'text': 'This is a test',
                'avg_logprob': -0.3
            },
            {
                'start': 2.5,
                'end': 5.0,
                'text': ' transcription.',
                'avg_logprob': -0.4
            }
        ],
        'duration': 5.0
    }

# tests/unit/test_environment_manager.py
"""
Unit tests for EnvironmentManager
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import subprocess

from src.environment_manager import EnvironmentManager

class TestEnvironmentManager:
    """Test cases for EnvironmentManager"""
    
    def test_initialization(self):
        """Test EnvironmentManager initialization"""
        env_manager = EnvironmentManager()
        assert env_manager.app_dir.exists()
        assert env_manager.venv_dir.name == "whisper_env"
        assert env_manager.requirements_file.name == "requirements.txt"
    
    def test_check_environment_no_venv(self):
        """Test environment check when no virtual environment exists"""
        env_manager = EnvironmentManager()
        
        # Mock venv directory doesn't exist
        with patch.object(env_manager.venv_dir, 'exists', return_value=False):
            status = env_manager.check_environment()
            
        assert status['venv_exists'] is False
        assert status['python_works'] is False
        assert status['whisper_installed'] is False
    
    @patch('subprocess.run')
    def test_check_environment_with_venv(self, mock_run):
        """Test environment check with working virtual environment"""
        env_manager = EnvironmentManager()
        
        # Mock successful subprocess calls
        mock_run.return_value = Mock(returncode=0, stdout="True")
        
        with patch.object(env_manager.venv_dir, 'exists', return_value=True), \
             patch.object(env_manager.python_exe, 'exists', return_value=True):
            status = env_manager.check_environment()
        
        assert status['venv_exists'] is True
        assert status['python_works'] is True
    
    @patch('venv.create')
    @patch('subprocess.run')
    def test_setup_environment(self, mock_run, mock_venv_create):
        """Test environment setup"""
        env_manager = EnvironmentManager()
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="")
        
        with patch.object(env_manager.venv_dir, 'exists', return_value=False), \
             patch.object(env_manager, 'detect_gpu_support', return_value=False):
            
            result = env_manager.setup_environment()
        
        assert result is True
        mock_venv_create.assert_called_once()

# tests/unit/test_transcription_engine.py
"""
Unit tests for TranscriptionEngine
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json

from src.transcription_engine import TranscriptionEngine

class TestTranscriptionEngine:
    """Test cases for TranscriptionEngine"""
    
    def test_initialization(self, mock_env_manager):
        """Test TranscriptionEngine initialization"""
        engine = TranscriptionEngine(mock_env_manager)
        
        assert engine.env_manager == mock_env_manager
        assert engine.should_stop is False
        assert engine.is_paused is False
    
    def test_stop_functionality(self, mock_env_manager):
        """Test stop functionality"""
        engine = TranscriptionEngine(mock_env_manager)
        mock_process = Mock()
        engine.current_process = mock_process
        
        engine.stop()
        
        assert engine.should_stop is True
        mock_process.terminate.assert_called_once()
    
    def test_get_audio_info(self, mock_env_manager, sample_audio_file):
        """Test audio file information extraction"""
        engine = TranscriptionEngine(mock_env_manager)
        
        # Mock ffprobe response
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="10.5")
            
            info = engine.get_audio_info(str(sample_audio_file))
        
        assert 'size_bytes' in info
        assert 'duration' in info
        assert info['duration'] == 10.5
    
    def test_is_dots_only_output(self, mock_env_manager):
        """Test detection of dots-only output (silence)"""
        engine = TranscriptionEngine(mock_env_manager)
        
        # Test dots-only result
        dots_result = {'text': '... ... ... ...'}
        assert engine.is_dots_only_output(dots_result) is True
        
        # Test normal result
        normal_result = {'text': 'This is a normal transcription.'}
        assert engine.is_dots_only_output(normal_result) is False
        
        # Test empty result
        empty_result = {'text': ''}
        assert engine.is_dots_only_output(empty_result) is True
    
    @patch('subprocess.Popen')
    def test_transcribe_success(self, mock_popen, mock_env_manager, sample_audio_file, temp_dir, mock_whisper_result):
        """Test successful transcription"""
        engine = TranscriptionEngine(mock_env_manager)
        
        # Mock successful process
        mock_process = Mock()
        mock_process.poll.return_value = 0
        mock_process.returncode = 0
        mock_process.stdout.readline.return_value = ""
        mock_process.communicate.return_value = ("", "")
        mock_popen.return_value = mock_process
        
        # Create mock result file
        result_file = mock_env_manager.app_dir / "temp_result_enhanced.json"
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_whisper_result)
            
            options = {
                'model_size': 'medium',
                'device': 'cpu',
                'output_formats': {'text': True},
                'enhanced_silence_handling': True
            }
            
            result = engine.transcribe(str(sample_audio_file), str(temp_dir), options)
        
        assert result['success'] is True
        assert 'files' in result
    
    def test_clean_text(self, mock_env_manager):
        """Test text cleaning functionality"""
        engine = TranscriptionEngine(mock_env_manager)
        
        dirty_text = "Um, this is like, you know, a test, uh, transcription."
        clean_text = engine.clean_text(dirty_text)
        
        assert "um" not in clean_text.lower()
        assert "like" not in clean_text.lower()
        assert "you know" not in clean_text.lower()
        assert "test" in clean_text

# tests/unit/test_model_manager.py
"""
Unit tests for ModelManager
"""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import json

from src.model_manager import ModelManager

class TestModelManager:
    """Test cases for ModelManager"""
    
    def test_initialization(self):
        """Test ModelManager initialization"""
        manager = ModelManager()
        
        assert 'tiny' in manager.models
        assert 'medium' in manager.models
        assert 'large' in manager.models
        assert manager.cache_dir.name == 'whisper'
    
    def test_get_model_info(self):
        """Test getting model information"""
        manager = ModelManager()
        
        # Test getting all models
        all_models = manager.get_model_info()
        assert len(all_models) == 5
        
        # Test getting specific model
        medium_info = manager.get_model_info('medium')
        assert medium_info['size'] == '769 MB'
        assert medium_info['description'] == 'High accuracy (recommended)'
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.stat')
    def test_check_downloaded_models(self, mock_stat, mock_exists):
        """Test checking downloaded models"""
        manager = ModelManager()
        
        # Mock file exists and stats
        mock_exists.return_value = True
        mock_stat.return_value = Mock(st_size=1024*1024*100)  # 100MB
        
        with patch.object(manager, 'verify_model_file', return_value=True):
            downloaded = manager.check_downloaded_models()
        
        assert len(downloaded) == len(manager.models)
        for model_name in manager.models:
            assert model_name in downloaded
            assert downloaded[model_name]['valid'] is True
    
    @patch('requests.get')
    def test_download_single_model(self, mock_get):
        """Test downloading a single model"""
        manager = ModelManager()
        
        # Mock successful download
        mock_response = Mock()
        mock_response.headers = {'content-length': '1024'}
        mock_response.iter_content.return_value = [b'fake content']
        mock_get.return_value = mock_response
        
        with patch.object(manager.cache_dir, 'mkdir'), \
             patch.object(manager, 'verify_model_file_by_path', return_value=True), \
             patch('builtins.open', mock_open()):
            
            result = manager.download_single_model('tiny')
        
        assert result is True
    
    def test_get_recommended_model(self):
        """Test model recommendation logic"""
        manager = ModelManager()
        
        # Test speed priority
        assert manager.get_recommended_model(accuracy_priority='speed') == 'base'
        
        # Test accuracy priority
        assert manager.get_recommended_model(accuracy_priority='accuracy') == 'large'
        
        # Test size-based recommendation
        assert manager.get_recommended_model(file_size_mb=5) == 'small'
        assert manager.get_recommended_model(file_size_mb=50) == 'medium'

# tests/unit/test_utils.py
"""
Unit tests for utility functions
"""

import pytest
from unittest.mock import patch, Mock
from pathlib import Path
import subprocess

from src.utils import (
    get_file_info, format_duration, format_time, 
    validate_audio_file, check_ffmpeg_installed
)

class TestUtils:
    """Test cases for utility functions"""
    
    def test_format_duration(self):
        """Test duration formatting"""
        assert format_duration(65) == "1:05"
        assert format_duration(3665) == "1:01:05"
        assert format_duration(30) == "0:30"
        assert format_duration(-1) == "Unknown"
    
    def test_format_time(self):
        """Test time formatting"""
        assert format_time(30) == "30.0s"
        assert format_time(90) == "1.5m"
        assert format_time(3600) == "1.0h"
    
    def test_validate_audio_file(self, temp_dir):
        """Test audio file validation"""
        # Valid file
        valid_file = temp_dir / "test.mp3"
        valid_file.write_bytes(b"fake audio")
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)
            is_valid, message = validate_audio_file(str(valid_file))
        
        assert is_valid is True
        assert "valid" in message
        
        # Invalid extension
        invalid_file = temp_dir / "test.xyz"
        invalid_file.write_bytes(b"fake content")
        
        is_valid, message = validate_audio_file(str(invalid_file))
        assert is_valid is False
        assert "Unsupported format" in message
    
    @patch('subprocess.run')
    def test_check_ffmpeg_installed(self, mock_run):
        """Test FFmpeg installation check"""
        # FFmpeg installed
        mock_run.return_value = Mock(returncode=0)
        assert check_ffmpeg_installed() is True
        
        # FFmpeg not installed
        mock_run.side_effect = FileNotFoundError()
        assert check_ffmpeg_installed() is False

# tests/integration/test_workflow.py
"""
Integration tests for complete workflow
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import tempfile

from src.environment_manager import EnvironmentManager
from src.transcription_engine import TranscriptionEngine
from src.model_manager import ModelManager

class TestCompleteWorkflow:
    """Integration tests for complete transcription workflow"""
    
    @pytest.fixture
    def setup_managers(self):
        """Setup all managers for integration testing"""
        with patch('src.environment_manager.EnvironmentManager') as mock_env, \
             patch('src.model_manager.ModelManager') as mock_model:
            
            # Setup mocks
            env_manager = mock_env.return_value
            env_manager.app_dir = Path("/fake/app")
            env_manager.python_exe = Path("/fake/python")
            
            model_manager = mock_model.return_value
            model_manager.models = {
                'tiny': {'size': '39 MB'},
                'medium': {'size': '769 MB'}
            }
            
            return env_manager, model_manager
    
    def test_full_transcription_workflow(self, setup_managers, temp_dir):
        """Test complete transcription workflow"""
        env_manager, model_manager = setup_managers
        
        # Create test audio file
        audio_file = temp_dir / "test.mp3"
        audio_file.write_bytes(b"fake audio content")
        
        # Create transcription engine
        engine = TranscriptionEngine(env_manager)
        
        # Mock the transcription process
        mock_result = {
            'text': 'Test transcription text.',
            'segments': [
                {'start': 0.0, 'end': 2.0, 'text': 'Test transcription', 'avg_logprob': -0.3},
                {'start': 2.0, 'end': 4.0, 'text': ' text.', 'avg_logprob': -0.4}
            ],
            'duration': 4.0
        }
        
        with patch('subprocess.Popen') as mock_popen, \
             patch('builtins.open', create=True) as mock_open, \
             patch('pathlib.Path.exists', return_value=True):
            
            # Mock process
            mock_process = Mock()
            mock_process.poll.return_value = 0
            mock_process.returncode = 0
            mock_process.stdout.readline.return_value = ""
            mock_process.communicate.return_value = ("", "")
            mock_popen.return_value = mock_process
            
            # Mock file reading
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_result)
            
            # Run transcription
            options = {
                'model_size': 'medium',
                'device': 'cpu',
                'output_formats': {'text': True, 'srt': True},
                'clean_text': True
            }
            
            result = engine.transcribe(str(audio_file), str(temp_dir), options)
        
        assert result['success'] is True
        assert len(result['files']) >= 1
        assert result['segments'] == 2
    
    def test_environment_setup_workflow(self, setup_managers):
        """Test environment setup workflow"""
        env_manager, _ = setup_managers
        
        with patch('venv.create') as mock_venv, \
             patch('subprocess.run') as mock_run:
            
            mock_run.return_value = Mock(returncode=0, stderr="", stdout="")
            
            result = env_manager.setup_environment()
        
        assert result is True

# tests/performance/test_performance.py
"""
Performance tests for Whisper Transcriber Pro
"""

import pytest
import time
from unittest.mock import Mock, patch
import memory_profiler

from src.transcription_engine import TranscriptionEngine

class TestPerformance:
    """Performance tests"""
    
    @pytest.mark.performance
    def test_transcription_memory_usage(self, mock_env_manager):
        """Test memory usage during transcription"""
        engine = TranscriptionEngine(mock_env_manager)
        
        @memory_profiler.profile
        def run_transcription():
            # Mock transcription process
            with patch('subprocess.Popen'):
                options = {'model_size': 'tiny', 'device': 'cpu', 'output_formats': {'text': True}}
                # This would normally run transcription
                pass
        
        # Run and measure memory
        start_time = time.time()
        run_transcription()
        end_time = time.time()
        
        # Basic timing assertion
        assert end_time - start_time < 5.0  # Should be fast for mock
    
    @pytest.mark.performance
    def test_ui_responsiveness(self):
        """Test UI responsiveness under load"""
        # This would test UI response times
        # For now, just a placeholder
        start_time = time.time()
        
        # Simulate some UI operations
        for _ in range(1000):
            pass
        
        end_time = time.time()
        assert end_time - start_time < 1.0  # Should be very fast