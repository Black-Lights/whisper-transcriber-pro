"""
Test configuration and fixtures for Whisper Transcriber Pro
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

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
def sample_video_file(temp_dir):
    """Create a mock video file for testing"""
    video_file = temp_dir / "test_video.mp4"
    video_file.write_bytes(b"fake video content")
    return video_file


@pytest.fixture
def mock_env_manager():
    """Mock environment manager"""
    with patch("src.environment_manager.EnvironmentManager") as mock:
        mock_instance = Mock()
        mock_instance.app_dir = Path("/fake/app/dir")
        mock_instance.python_exe = Path("/fake/python")
        mock_instance.pip_exe = Path("/fake/pip")
        mock_instance.venv_dir = Path("/fake/venv")
        mock_instance.check_environment.return_value = {
            "venv_exists": True,
            "python_works": True,
            "whisper_installed": True,
            "torch_installed": True,
            "gpu_available": False,
        }
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_whisper_result():
    """Mock Whisper transcription result"""
    return {
        "text": "This is a test transcription.",
        "segments": [
            {"start": 0.0, "end": 2.5, "text": "This is a test", "avg_logprob": -0.3},
            {"start": 2.5, "end": 5.0, "text": " transcription.", "avg_logprob": -0.4},
        ],
        "duration": 5.0,
    }


@pytest.fixture
def mock_model_info():
    """Mock model information"""
    return {
        "tiny": {
            "size": "39 MB",
            "description": "Fastest, least accurate",
            "speed": "~32x real-time",
            "accuracy": "Basic",
        },
        "medium": {
            "size": "769 MB",
            "description": "High accuracy (recommended)",
            "speed": "~2x real-time",
            "accuracy": "High",
        },
    }


@pytest.fixture
def mock_subprocess_success():
    """Mock successful subprocess call"""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock(returncode=0, stdout="Success", stderr="")
        yield mock_run


@pytest.fixture
def mock_subprocess_failure():
    """Mock failed subprocess call"""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="Error occurred")
        yield mock_run


@pytest.fixture
def mock_psutil():
    """Mock psutil for process management tests"""
    with patch("psutil.process_iter") as mock_iter, patch(
        "psutil.virtual_memory"
    ) as mock_memory, patch("psutil.cpu_count") as mock_cpu:

        mock_memory.return_value = Mock(
            total=8 * 1024**3,  # 8GB
            available=4 * 1024**3,  # 4GB available
            percent=50.0,
        )
        mock_cpu.return_value = 8
        mock_iter.return_value = []

        yield {
            "process_iter": mock_iter,
            "virtual_memory": mock_memory,
            "cpu_count": mock_cpu,
        }


@pytest.fixture
def sample_settings():
    """Sample application settings"""
    return {
        "general": {
            "default_model": "medium",
            "default_language": "auto",
            "default_device": "gpu",
            "default_output_dir": "/tmp/output",
        },
        "output": {
            "formats": {"text": True, "detailed": True, "srt": True, "vtt": False}
        },
    }


@pytest.fixture
def mock_file_operations(temp_dir):
    """Mock file operations with temp directory"""

    def create_mock_file(filename, content="mock content"):
        file_path = temp_dir / filename
        file_path.write_text(content, encoding="utf-8")
        return file_path

    return create_mock_file


@pytest.fixture
def mock_torch():
    """Mock PyTorch functionality"""
    with patch("torch.cuda.is_available") as mock_cuda, patch(
        "torch.cuda.get_device_name"
    ) as mock_device_name:

        mock_cuda.return_value = True
        mock_device_name.return_value = "NVIDIA GeForce RTX 3060"

        yield {"cuda_available": mock_cuda, "device_name": mock_device_name}


@pytest.fixture
def capture_output():
    """Capture stdout/stderr for testing"""
    import contextlib
    import io

    @contextlib.contextmanager
    def _capture():
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(
            stderr_capture
        ):
            yield {"stdout": stdout_capture, "stderr": stderr_capture}

    return _capture


# Pytest markers
pytest_markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "performance: Performance tests",
    "slow: Slow running tests",
    "gpu: Tests requiring GPU",
    "network: Tests requiring network access",
]


def pytest_configure(config):
    """Configure pytest with custom markers"""
    for marker in pytest_markers:
        config.addinivalue_line("markers", marker)


# Test data constants
TEST_AUDIO_FORMATS = [".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"]
TEST_VIDEO_FORMATS = [".mp4", ".avi", ".mkv", ".mov", ".wmv"]
TEST_MODEL_SIZES = ["tiny", "base", "small", "medium", "large"]
TEST_LANGUAGES = ["auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"]


# Helper functions for tests
def create_mock_transcription_result(text="Test transcription", segments=None):
    """Create a mock transcription result"""
    if segments is None:
        segments = [
            {
                "start": 0.0,
                "end": len(text.split()) * 0.5,
                "text": text,
                "avg_logprob": -0.3,
            }
        ]

    return {
        "text": text,
        "segments": segments,
        "duration": segments[-1]["end"] if segments else 0.0,
    }


def assert_file_exists_with_content(file_path, expected_content=None):
    """Assert that file exists and optionally check content"""
    assert file_path.exists(), f"File {file_path} does not exist"

    if expected_content is not None:
        actual_content = file_path.read_text(encoding="utf-8")
        assert (
            expected_content in actual_content
        ), f"Expected content not found in {file_path}"


def assert_valid_json_file(file_path):
    """Assert that file exists and contains valid JSON"""
    assert file_path.exists(), f"JSON file {file_path} does not exist"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json.load(f)
    except json.JSONDecodeError as e:
        pytest.fail(f"Invalid JSON in {file_path}: {e}")


def mock_successful_transcription():
    """Create a mock for successful transcription"""
    return create_mock_transcription_result(
        text="This is a successful test transcription with multiple segments.",
        segments=[
            {
                "start": 0.0,
                "end": 2.0,
                "text": "This is a successful",
                "avg_logprob": -0.2,
            },
            {
                "start": 2.0,
                "end": 4.0,
                "text": " test transcription",
                "avg_logprob": -0.3,
            },
            {
                "start": 4.0,
                "end": 6.0,
                "text": " with multiple segments.",
                "avg_logprob": -0.25,
            },
        ],
    )
