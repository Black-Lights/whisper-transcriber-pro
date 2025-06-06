"""
Unit tests for ModelManager
"""

import pytest
from unittest.mock import Mock, patch, mock_open, MagicMock
from pathlib import Path
import json
import hashlib
import requests
import time

from src.model_manager import ModelManager


class TestModelManager:
    """Test cases for ModelManager"""
    
    def test_initialization(self):
        """Test ModelManager initialization"""
        manager = ModelManager()
        
        # Check model definitions
        assert 'tiny' in manager.models
        assert 'base' in manager.models
        assert 'small' in manager.models
        assert 'medium' in manager.models
        assert 'large' in manager.models
        assert len(manager.models) == 5
        
        # Check cache directory
        assert manager.cache_dir.name == 'whisper'
        assert manager.cache_dir.is_absolute()
        
        # Check each model has required fields
        for model_name, model_info in manager.models.items():
            assert 'size' in model_info
            assert 'description' in model_info
            assert 'speed' in model_info
            assert 'accuracy' in model_info
            assert 'url' in model_info
            assert 'sha256' in model_info
    
    def test_get_model_info_all_models(self):
        """Test getting information for all models"""
        manager = ModelManager()
        
        all_models = manager.get_model_info()
        
        assert len(all_models) == 5
        assert 'tiny' in all_models
        assert 'medium' in all_models
        assert 'large' in all_models
    
    def test_get_model_info_specific_model(self):
        """Test getting information for specific model"""
        manager = ModelManager()
        
        medium_info = manager.get_model_info('medium')
        
        assert medium_info['size'] == '769 MB'
        assert medium_info['description'] == 'High accuracy (recommended)'
        assert medium_info['speed'] == '~2x real-time'
        assert medium_info['accuracy'] == 'High'
        assert 'url' in medium_info
        assert 'sha256' in medium_info
    
    def test_get_model_info_nonexistent_model(self):
        """Test getting info for non-existent model"""
        manager = ModelManager()
        
        result = manager.get_model_info('nonexistent')
        
        assert result == {}
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.stat')
    def test_check_downloaded_models_with_valid_models(self, mock_stat, mock_exists):
        """Test checking downloaded models with valid files"""
        manager = ModelManager()
        
        # Mock file exists and stats
        mock_exists.return_value = True
        mock_stat.return_value = Mock(st_size=1024*1024*100, st_mtime=time.time())  # 100MB
        
        with patch.object(manager, 'verify_model_file', return_value=True):
            downloaded = manager.check_downloaded_models()
        
        assert len(downloaded) == len(manager.models)
        for model_name in manager.models:
            assert model_name in downloaded
            assert downloaded[model_name]['valid'] is True
            assert downloaded[model_name]['size_mb'] == pytest.approx(100, rel=1e-2)
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.stat')
    def test_check_downloaded_models_with_invalid_models(self, mock_stat, mock_exists):
        """Test checking downloaded models with corrupted files"""
        manager = ModelManager()
        
        mock_exists.return_value = True
        mock_stat.return_value = Mock(st_size=1024*1024*50, st_mtime=time.time())
        
        with patch.object(manager, 'verify_model_file', return_value=False):
            downloaded = manager.check_downloaded_models()
        
        for model_name in downloaded:
            assert downloaded[model_name]['valid'] is False
    
    @patch('pathlib.Path.exists')
    def test_check_downloaded_models_none_exist(self, mock_exists):
        """Test checking downloaded models when none exist"""
        manager = ModelManager()
        mock_exists.return_value = False
        
        downloaded = manager.check_downloaded_models()
        
        assert downloaded == {}
    
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake file content')
    @patch('hashlib.sha256')
    def test_verify_model_file_valid(self, mock_sha256, mock_file):
        """Test model file verification with valid hash"""
        manager = ModelManager()
        
        # Mock hash calculation
        mock_hash = Mock()
        mock_hash.hexdigest.return_value = manager.models['tiny']['sha256']
        mock_sha256.return_value = mock_hash
        
        result = manager.verify_model_file('tiny', Path('/fake/path'))
        
        assert result is True
        mock_file.assert_called_once()
    
    @patch('builtins.open', new_callable=mock_open, read_data=b'fake file content')
    @patch('hashlib.sha256')
    def test_verify_model_file_invalid(self, mock_sha256, mock_file):
        """Test model file verification with invalid hash"""
        manager = ModelManager()
        
        # Mock hash calculation with wrong hash
        mock_hash = Mock()
        mock_hash.hexdigest.return_value = 'wrong_hash'
        mock_sha256.return_value = mock_hash
        
        result = manager.verify_model_file('tiny', Path('/fake/path'))
        
        assert result is False
    
    def test_verify_model_file_exception(self):
        """Test model file verification with file read error"""
        manager = ModelManager()
        
        with patch('builtins.open', side_effect=IOError("File error")):
            result = manager.verify_model_file('tiny', Path('/fake/path'))
        
        assert result is False
    
    @patch.object(ModelManager, 'download_single_model')
    def test_download_models_success(self, mock_download):
        """Test successful model downloads"""
        manager = ModelManager()
        mock_download.return_value = True
        
        progress_messages = []
        def progress_callback(msg):
            progress_messages.append(msg)
        
        result = manager.download_models(['tiny', 'medium'], progress_callback)
        
        assert result is True
        assert mock_download.call_count == 2
        assert len(progress_messages) > 0
    
    @patch.object(ModelManager, 'download_single_model')
    def test_download_models_partial_failure(self, mock_download):
        """Test model downloads with some failures"""
        manager = ModelManager()
        mock_download.side_effect = [True, False]  # First succeeds, second fails
        
        result = manager.download_models(['tiny', 'medium'])
        
        assert result is True  # At least one succeeded
        assert mock_download.call_count == 2
    
    @patch.object(ModelManager, 'download_single_model')
    def test_download_models_all_failures(self, mock_download):
        """Test model downloads with all failures"""
        manager = ModelManager()
        mock_download.return_value = False
        
        result = manager.download_models(['tiny', 'medium'])
        
        assert result is False
    
    def test_download_models_default(self):
        """Test downloading models with default parameters"""
        manager = ModelManager()
        
        with patch.object(manager, 'download_single_model', return_value=True) as mock_download:
            result = manager.download_models()
        
        assert result is True
        mock_download.assert_called_once_with('medium', None)
    
    def test_download_models_string_input(self):
        """Test downloading models with string input"""
        manager = ModelManager()
        
        with patch.object(manager, 'download_single_model', return_value=True) as mock_download:
            result = manager.download_models('tiny')
        
        assert result is True
        mock_download.assert_called_once_with('tiny', None)
    
    def test_download_models_unknown_model(self):
        """Test downloading unknown model"""
        manager = ModelManager()
        
        with patch.object(manager, 'download_single_model') as mock_download:
            result = manager.download_models(['unknown_model'])
        
        mock_download.assert_not_called()
        assert result is False
    
    @patch('requests.get')
    @patch('pathlib.Path.mkdir')
    def test_download_single_model_success(self, mock_mkdir, mock_get):
        """Test successful single model download"""
        manager = ModelManager()
        
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.headers = {'content-length': '1024'}
        mock_response.iter_content.return_value = [b'fake content chunk']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        with patch.object(manager, 'verify_model_file_by_path', return_value=True), \
             patch('builtins.open', mock_open()), \
             patch('pathlib.Path.exists', return_value=False), \
             patch('pathlib.Path.rename'):
            
            result = manager.download_single_model('tiny')
        
        assert result is True
        mock_get.assert_called_once()
        mock_mkdir.assert_called_once()
    
    @patch('requests.get')
    def test_download_single_model_already_exists(self, mock_get):
        """Test download when model already exists and is valid"""
        manager = ModelManager()
        
        model_file = manager.cache_dir / "tiny.pt"
        
        with patch.object(model_file, 'exists', return_value=True), \
             patch.object(manager, 'verify_model_file', return_value=True):
            
            result = manager.download_single_model('tiny')
        
        assert result is True
        mock_get.assert_not_called()  # Should not download
    
    @patch('requests.get')
    def test_download_single_model_corrupted_existing(self, mock_get):
        """Test download when existing model is corrupted"""
        manager = ModelManager()
        
        model_file = manager.cache_dir / "tiny.pt"
        
        # Mock successful download after removing corrupted file
        mock_response = Mock()
        mock_response.headers = {'content-length': '1024'}
        mock_response.iter_content.return_value = [b'fake content']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        with patch.object(model_file, 'exists', return_value=True), \
             patch.object(manager, 'verify_model_file', return_value=False), \
             patch.object(model_file, 'unlink'), \
             patch.object(manager, 'verify_model_file_by_path', return_value=True), \
             patch('builtins.open', mock_open()), \
             patch('pathlib.Path.mkdir'), \
             patch('pathlib.Path.rename'):
            
            result = manager.download_single_model('tiny')
        
        assert result is True
        model_file.unlink.assert_called_once()  # Should remove corrupted file
    
    @patch('requests.get')
    def test_download_single_model_network_error(self, mock_get):
        """Test download with network error"""
        manager = ModelManager()
        
        mock_get.side_effect = requests.RequestException("Network error")
        
        result = manager.download_single_model('tiny')
        
        assert result is False
    
    @patch('requests.get')
    @patch('pathlib.Path.mkdir')
    def test_download_single_model_verification_failure(self, mock_mkdir, mock_get):
        """Test download with verification failure"""
        manager = ModelManager()
        
        # Mock successful download but failed verification
        mock_response = Mock()
        mock_response.headers = {'content-length': '1024'}
        mock_response.iter_content.return_value = [b'fake content']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        with patch.object(manager, 'verify_model_file_by_path', return_value=False), \
             patch('builtins.open', mock_open()), \
             patch('pathlib.Path.exists', return_value=False), \
             patch('pathlib.Path.unlink') as mock_unlink:
            
            result = manager.download_single_model('tiny')
        
        assert result is False
        mock_unlink.assert_called()  # Should clean up invalid file
    
    def test_download_single_model_invalid_model(self):
        """Test download of invalid model name"""
        manager = ModelManager()
        
        result = manager.download_single_model('invalid_model')
        
        assert result is False
    
    @patch('builtins.open', new_callable=mock_open, read_data=b'test content')
    @patch('hashlib.sha256')
    def test_verify_model_file_by_path_valid(self, mock_sha256, mock_file):
        """Test file verification by path with valid hash"""
        manager = ModelManager()
        
        mock_hash = Mock()
        mock_hash.hexdigest.return_value = 'expected_hash'
        mock_sha256.return_value = mock_hash
        
        result = manager.verify_model_file_by_path(Path('/fake/path'), 'expected_hash')
        
        assert result is True
    
    @patch('builtins.open', new_callable=mock_open, read_data=b'test content')
    @patch('hashlib.sha256')
    def test_verify_model_file_by_path_invalid(self, mock_sha256, mock_file):
        """Test file verification by path with invalid hash"""
        manager = ModelManager()
        
        mock_hash = Mock()
        mock_hash.hexdigest.return_value = 'actual_hash'
        mock_sha256.return_value = mock_hash
        
        result = manager.verify_model_file_by_path(Path('/fake/path'), 'expected_hash')
        
        assert result is False
    
    def test_verify_model_file_by_path_exception(self):
        """Test file verification with read exception"""
        manager = ModelManager()
        
        with patch('builtins.open', side_effect=IOError("Read error")):
            result = manager.verify_model_file_by_path(Path('/fake/path'), 'hash')
        
        assert result is False
    
    @patch('tempfile.NamedTemporaryFile')
    @patch('subprocess.run')
    @patch('os.unlink')
    def test_download_model_whisper_native_success(self, mock_unlink, mock_run, mock_temp):
        """Test native Whisper download success"""
        manager = ModelManager()
        
        # Mock temporary file
        mock_temp.return_value.__enter__.return_value.name = '/tmp/script.py'
        
        # Mock successful subprocess
        mock_run.return_value = Mock(returncode=0, stderr="", stdout="Model loaded successfully!")
        
        result = manager.download_model_whisper_native('medium')
        
        assert result is True
        mock_run.assert_called_once()
        mock_unlink.assert_called_once()
    
    @patch('tempfile.NamedTemporaryFile')
    @patch('subprocess.run')
    @patch('os.unlink')
    def test_download_model_whisper_native_failure(self, mock_unlink, mock_run, mock_temp):
        """Test native Whisper download failure"""
        manager = ModelManager()
        
        mock_temp.return_value.__enter__.return_value.name = '/tmp/script.py'
        mock_run.return_value = Mock(returncode=1, stderr="Download failed")
        
        result = manager.download_model_whisper_native('medium')
        
        assert result is False
    
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.exists')
    def test_delete_model_success(self, mock_exists, mock_unlink):
        """Test successful model deletion"""
        manager = ModelManager()
        
        mock_exists.return_value = True
        
        result = manager.delete_model('tiny')
        
        assert result is True
        mock_unlink.assert_called_once()
    
    @patch('pathlib.Path.exists')
    def test_delete_model_not_exists(self, mock_exists):
        """Test deletion of non-existent model"""
        manager = ModelManager()
        
        mock_exists.return_value = False
        
        result = manager.delete_model('tiny')
        
        assert result is False
    
    @patch('pathlib.Path.unlink')
    @patch('pathlib.Path.exists')
    def test_delete_model_exception(self, mock_exists, mock_unlink):
        """Test model deletion with exception"""
        manager = ModelManager()
        
        mock_exists.return_value = True
        mock_unlink.side_effect = OSError("Permission denied")
        
        result = manager.delete_model('tiny')
        
        assert result is False
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.rglob')
    @patch('pathlib.Path.stat')
    def test_get_cache_size(self, mock_stat, mock_rglob, mock_exists):
        """Test cache size calculation"""
        manager = ModelManager()
        
        mock_exists.return_value = True
        
        # Mock multiple model files
        mock_files = [Mock(), Mock(), Mock()]
        for i, mock_file in enumerate(mock_files):
            mock_file.stat.return_value = Mock(st_size=1024 * (i + 1))
        
        mock_rglob.return_value = mock_files
        
        total_size = manager.get_cache_size()
        
        expected_size = 1024 + 2048 + 3072  # Sum of mock file sizes
        assert total_size == expected_size
    
    @patch('pathlib.Path.exists')
    def test_get_cache_size_no_cache(self, mock_exists):
        """Test cache size when cache directory doesn't exist"""
        manager = ModelManager()
        
        mock_exists.return_value = False
        
        total_size = manager.get_cache_size()
        
        assert total_size == 0
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.glob')
    @patch('pathlib.Path.unlink')
    def test_clear_cache_success(self, mock_unlink, mock_glob, mock_exists):
        """Test successful cache clearing"""
        manager = ModelManager()
        
        mock_exists.return_value = True
        
        # Mock model files and temp files
        mock_pt_files = [Mock(), Mock()]
        mock_tmp_files = [Mock()]
        
        mock_glob.side_effect = [mock_pt_files, mock_tmp_files]
        
        result = manager.clear_cache()
        
        assert result is True
        assert mock_unlink.call_count == 3  # 2 .pt files + 1 .tmp file
    
    @patch('pathlib.Path.exists')
    def test_clear_cache_no_directory(self, mock_exists):
        """Test cache clearing when directory doesn't exist"""
        manager = ModelManager()
        
        mock_exists.return_value = False
        
        result = manager.clear_cache()
        
        assert result is True  # Should succeed even if dir doesn't exist
    
    @patch.object(ModelManager, 'delete_model')
    @patch.object(ModelManager, 'download_single_model')
    def test_repair_model_success(self, mock_download, mock_delete):
        """Test successful model repair"""
        manager = ModelManager()
        
        mock_delete.return_value = True
        mock_download.return_value = True
        
        progress_messages = []
        def progress_callback(msg):
            progress_messages.append(msg)
        
        result = manager.repair_model('tiny', progress_callback)
        
        assert result is True
        mock_delete.assert_called_once_with('tiny')
        mock_download.assert_called_once_with('tiny', progress_callback)
        assert len(progress_messages) > 0
    
    @patch.object(ModelManager, 'delete_model')
    @patch.object(ModelManager, 'download_single_model')
    def test_repair_model_failure(self, mock_download, mock_delete):
        """Test model repair failure"""
        manager = ModelManager()
        
        mock_delete.return_value = True
        mock_download.return_value = False
        
        result = manager.repair_model('tiny')
        
        assert result is False
    
    def test_estimate_download_time(self):
        """Test download time estimation"""
        manager = ModelManager()
        
        # Test for tiny model (39 MB)
        time_estimate = manager.estimate_download_time('tiny', 10)  # 10 Mbps
        assert "seconds" in time_estimate or "minutes" in time_estimate
        
        # Test for large model (1550 MB)
        time_estimate = manager.estimate_download_time('large', 10)
        assert "minutes" in time_estimate
        
        # Test for unknown model
        time_estimate = manager.estimate_download_time('unknown', 10)
        assert time_estimate == "Unknown"
    
    def test_get_download_progress_existing(self):
        """Test getting download progress for active download"""
        manager = ModelManager()
        
        # Set some progress data
        manager.download_progress['tiny'] = {
            'downloaded': 512,
            'total': 1024,
            'percent': 50.0
        }
        
        progress = manager.get_download_progress('tiny')
        
        assert progress['downloaded'] == 512
        assert progress['total'] == 1024
        assert progress['percent'] == 50.0
    
    def test_get_download_progress_nonexistent(self):
        """Test getting download progress for non-active download"""
        manager = ModelManager()
        
        progress = manager.get_download_progress('nonexistent')
        
        assert progress['downloaded'] == 0
        assert progress['total'] == 0
        assert progress['percent'] == 0
    
    @patch.object(ModelManager, 'check_downloaded_models')
    @patch.object(ModelManager, 'get_cache_size')
    def test_get_model_status_summary(self, mock_cache_size, mock_downloaded):
        """Test comprehensive model status summary"""
        manager = ModelManager()
        
        mock_downloaded.return_value = {
            'tiny': {'valid': True, 'size_mb': 39, 'last_modified': time.time()},
            'medium': {'valid': False, 'size_mb': 769, 'last_modified': time.time()}
        }
        mock_cache_size.return_value = 1024 * 1024 * 800  # 800 MB
        
        summary = manager.get_model_status_summary()
        
        assert summary['total_models'] == 5
        assert summary['downloaded_count'] == 2
        assert 'tiny' in summary['downloaded_models']
        assert 'medium' in summary['downloaded_models']
        assert 'base' in summary['missing_models']
        assert 'medium' in summary['corrupted_models']  # Invalid model
        assert summary['total_cache_size_mb'] == 800
        assert 'model_details' in summary
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch.object(ModelManager, 'get_model_status_summary')
    def test_export_model_info_success(self, mock_summary, mock_json, mock_file):
        """Test successful model info export"""
        manager = ModelManager()
        
        mock_summary.return_value = {'test': 'data'}
        
        result = manager.export_model_info('/tmp/export.json')
        
        assert result is True
        mock_file.assert_called_once()
        mock_json.assert_called_once()
    
    @patch('builtins.open', side_effect=IOError("Write error"))
    def test_export_model_info_failure(self, mock_file):
        """Test model info export failure"""
        manager = ModelManager()
        
        result = manager.export_model_info('/tmp/export.json')
        
        assert result is False
    
    def test_get_recommended_model_speed_priority(self):
        """Test model recommendation with speed priority"""
        manager = ModelManager()
        
        result = manager.get_recommended_model(accuracy_priority='speed')
        
        assert result == 'base'
    
    def test_get_recommended_model_accuracy_priority(self):
        """Test model recommendation with accuracy priority"""
        manager = ModelManager()
        
        result = manager.get_recommended_model(accuracy_priority='accuracy')
        
        assert result == 'large'
    
    def test_get_recommended_model_size_based(self):
        """Test model recommendation based on file size"""
        manager = ModelManager()
        
        # Small file
        result = manager.get_recommended_model(file_size_mb=5)
        assert result == 'small'
        
        # Medium file
        result = manager.get_recommended_model(file_size_mb=50)
        assert result == 'medium'
        
        # Large file
        result = manager.get_recommended_model(file_size_mb=500)
        assert result == 'medium'  # Still medium for balance
    
    def test_get_recommended_model_default(self):
        """Test default model recommendation"""
        manager = ModelManager()
        
        result = manager.get_recommended_model()
        
        assert result == 'medium'
    
    @patch.object(ModelManager, 'download_single_model')
    def test_batch_download_models_success(self, mock_download):
        """Test successful batch model download"""
        manager = ModelManager()
        
        mock_download.return_value = True
        
        progress_messages = []
        def progress_callback(msg):
            progress_messages.append(msg)
        
        results = manager.batch_download_models(['tiny', 'base'], progress_callback)
        
        assert results['tiny'] is True
        assert results['base'] is True
        assert mock_download.call_count == 2
        assert len(progress_messages) > 0
    
    @patch.object(ModelManager, 'download_single_model')
    def test_batch_download_models_mixed_results(self, mock_download):
        """Test batch download with mixed success/failure"""
        manager = ModelManager()
        
        mock_download.side_effect = [True, False, True]  # Mixed results
        
        results = manager.batch_download_models(['tiny', 'base', 'small'])
        
        assert results['tiny'] is True
        assert results['base'] is False
        assert results['small'] is True
    
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.glob')
    @patch('pathlib.Path.unlink')
    def test_cleanup_temp_files_success(self, mock_unlink, mock_glob, mock_exists):
        """Test successful temp file cleanup"""
        manager = ModelManager()
        
        mock_exists.return_value = True
        mock_temp_files = [Mock(), Mock()]
        mock_glob.return_value = mock_temp_files
        
        result = manager.cleanup_temp_files()
        
        assert result is True
        assert mock_unlink.call_count == 2
    
    @patch('pathlib.Path.exists')
    def test_cleanup_temp_files_no_cache_dir(self, mock_exists):
        """Test temp file cleanup when cache dir doesn't exist"""
        manager = ModelManager()
        
        mock_exists.return_value = False
        
        result = manager.cleanup_temp_files()
        
        assert result is True


class TestModelManagerIntegration:
    """Integration tests for ModelManager"""
    
    def test_model_definitions_consistency(self):
        """Test that all model definitions have consistent structure"""
        manager = ModelManager()
        
        required_fields = ['size', 'description', 'speed', 'accuracy', 'url', 'sha256']
        
        for model_name, model_info in manager.models.items():
            for field in required_fields:
                assert field in model_info, f"Model {model_name} missing field {field}"
            
            # Check size format
            assert 'MB' in model_info['size'] or 'GB' in model_info['size']
            
            # Check URL format
            assert model_info['url'].startswith('https://')
            
            # Check SHA256 format (64 hex characters)
            assert len(model_info['sha256']) == 64
            assert all(c in '0123456789abcdef' for c in model_info['sha256'])
    
    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Fixture to provide a temporary directory for tests"""
        return tmp_path
    
    def test_cache_directory_structure(self, temp_dir):
        """Test cache directory handling"""
        manager = ModelManager()
        
        # Temporarily change cache directory
        original_cache_dir = manager.cache_dir
        manager.cache_dir = temp_dir / 'test_cache'
        
        try:
            # Test cache size calculation with empty directory
            size = manager.get_cache_size()
            assert size == 0
            
            # Create mock model files
            manager.cache_dir.mkdir(exist_ok=True)
            (manager.cache_dir / 'tiny.pt').write_bytes(b'x' * 1024)
            (manager.cache_dir / 'medium.pt').write_bytes(b'x' * 2048)
            
            # Test cache size calculation
            size = manager.get_cache_size()
            assert size == 3072
            
            # Test cache clearing
            result = manager.clear_cache()
            assert result is True
            
            # Verify files were removed
            assert not (manager.cache_dir / 'tiny.pt').exists()
            assert not (manager.cache_dir / 'medium.pt').exists()
            
        finally:
            manager.cache_dir = original_cache_dir
    
    def test_download_progress_tracking(self):
        """Test download progress tracking functionality"""
        manager = ModelManager()
        
        # Test initial state
        progress = manager.get_download_progress('tiny')
        assert progress['downloaded'] == 0
        assert progress['total'] == 0
        assert progress['percent'] == 0
        
        # Simulate download progress
        manager.download_progress['tiny'] = {
            'downloaded': 256,
            'total': 1024,
            'percent': 25.0
        }
        
        progress = manager.get_download_progress('tiny')
        assert progress['downloaded'] == 256
        assert progress['total'] == 1024
        assert progress['percent'] == 25.0
        
        # Test progress for different model
        progress = manager.get_download_progress('medium')
        assert progress['downloaded'] == 0  # Should be default values
        assert progress['total'] == 0
        assert progress['percent'] == 0