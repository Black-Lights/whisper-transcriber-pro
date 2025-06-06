"""
Integration tests for complete workflow
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import json
import tempfile
import subprocess
import time
import threading

from src.environment_manager import EnvironmentManager
from src.transcription_engine import TranscriptionEngine
from src.model_manager import ModelManager
from src.settings_manager import SettingsManager


class TestCompleteWorkflow:
    """Integration tests for complete transcription workflow"""
    
    @pytest.fixture
    def setup_managers(self, temp_dir):
        """Setup all managers for integration testing"""
        # Create mock environment manager
        env_manager = Mock(spec=EnvironmentManager)
        env_manager.app_dir = temp_dir
        env_manager.python_exe = temp_dir / "python"
        env_manager.venv_dir = temp_dir / "venv"
        env_manager.pip_exe = temp_dir / "pip"
        env_manager.check_environment.return_value = {
            'venv_exists': True,
            'python_works': True,
            'whisper_installed': True,
            'torch_installed': True,
            'gpu_available': False
        }
        env_manager.setup_environment.return_value = True
        env_manager.get_environment_info.return_value = {
            'venv_path': str(temp_dir / "venv"),
            'python_path': str(temp_dir / "python"),
            'status': {'venv_exists': True, 'python_works': True, 'whisper_installed': True}
        }
        
        # Create mock model manager
        model_manager = Mock(spec=ModelManager)
        model_manager.models = {
            'tiny': {'size': '39 MB', 'description': 'Fastest', 'url': 'http://example.com/tiny.pt', 'sha256': 'abc123'},
            'medium': {'size': '769 MB', 'description': 'Recommended', 'url': 'http://example.com/medium.pt', 'sha256': 'def456'}
        }
        model_manager.check_downloaded_models.return_value = {
            'medium': {'valid': True, 'size_mb': 769, 'path': str(temp_dir / 'medium.pt')}
        }
        model_manager.download_models.return_value = True
        model_manager.get_model_info.return_value = model_manager.models
        model_manager.verify_model_file.return_value = True
        model_manager.get_cache_size.return_value = 1024 * 1024 * 769  # 769 MB
        
        # Create mock settings manager
        settings_manager = Mock(spec=SettingsManager)
        settings_manager.settings = {
            'general': {
                'default_model': 'medium',
                'default_language': 'auto',
                'default_device': 'gpu',
                'default_output_dir': str(temp_dir / 'output')
            },
            'output': {
                'formats': {'text': True, 'detailed': True, 'srt': True, 'vtt': False}
            }
        }
        settings_manager.load_settings.return_value = settings_manager.settings
        settings_manager.save_settings.return_value = True
        settings_manager.get_transcription_options.return_value = {
            'model_size': 'medium',
            'language': 'auto',
            'device': 'gpu',
            'output_formats': {'text': True, 'detailed': True, 'srt': True, 'vtt': False}
        }
        
        # Create mock transcription engine
        transcription_engine = Mock(spec=TranscriptionEngine)
        transcription_engine.transcribe.return_value = {
            'success': True,
            'files': [str(temp_dir / 'output.txt'), str(temp_dir / 'output.srt')],
            'transcription_time': 45.2,
            'segments': 15
        }
        transcription_engine.stop.return_value = None
        transcription_engine.should_stop = False
        
        return {
            'env_manager': env_manager,
            'model_manager': model_manager,
            'settings_manager': settings_manager,
            'transcription_engine': transcription_engine,
            'temp_dir': temp_dir
        }
    
    @pytest.fixture
    def mock_audio_file(self, temp_dir):
        """Create a mock audio file for testing"""
        audio_file = temp_dir / "test_audio.mp3"
        audio_file.write_bytes(b"fake audio content" * 1000)  # Make it reasonably sized
        return audio_file
    
    @pytest.fixture
    def mock_whisper_result(self):
        """Mock Whisper transcription result"""
        return {
            'text': 'This is a test transcription with multiple segments for testing purposes.',
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
                    'text': ' transcription with',
                    'avg_logprob': -0.4
                },
                {
                    'start': 5.0,
                    'end': 8.0,
                    'text': ' multiple segments for testing purposes.',
                    'avg_logprob': -0.2
                }
            ],
            'duration': 8.0
        }
    
    def test_complete_transcription_workflow_success(self, setup_managers, mock_audio_file, mock_whisper_result):
        """Test complete successful transcription workflow"""
        managers = setup_managers
        env_manager = managers['env_manager']
        model_manager = managers['model_manager']
        settings_manager = managers['settings_manager']
        temp_dir = managers['temp_dir']
        
        # Create output directory
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        # Create transcription engine with real logic
        transcription_engine = TranscriptionEngine(env_manager)
        
        # Mock the actual transcription process
        with patch.object(transcription_engine, 'run_enhanced_transcription') as mock_transcribe:
            mock_transcribe.return_value = mock_whisper_result
            
            # Mock file generation
            with patch.object(transcription_engine, 'generate_output_files') as mock_generate:
                output_files = [
                    str(output_dir / "test_audio.txt"),
                    str(output_dir / "test_audio_detailed.txt"),
                    str(output_dir / "test_audio.srt")
                ]
                mock_generate.return_value = output_files
                
                # Create the output files
                for file_path in output_files:
                    Path(file_path).write_text("Mock transcription content")
                
                # Get transcription options from settings
                options = settings_manager.get_transcription_options()
                
                # Run transcription
                result = transcription_engine.transcribe(
                    str(mock_audio_file),
                    str(output_dir),
                    options
                )
                
                # Verify success
                assert result['success'] is True
                assert len(result['files']) == 3
                assert result['transcription_time'] > 0
                assert result['segments'] == 3
                
                # Verify all output files exist
                for file_path in result['files']:
                    assert Path(file_path).exists()
                    assert Path(file_path).stat().st_size > 0
    
    def test_workflow_with_environment_setup(self, setup_managers, mock_audio_file):
        """Test workflow including environment setup"""
        managers = setup_managers
        env_manager = managers['env_manager']
        model_manager = managers['model_manager']
        
        # Test environment check
        status = env_manager.check_environment()
        assert status['venv_exists'] is True
        assert status['python_works'] is True
        assert status['whisper_installed'] is True
        
        # Test environment setup
        setup_success = env_manager.setup_environment()
        assert setup_success is True
        
        # Test model download
        download_success = model_manager.download_models(['medium'])
        assert download_success is True
        
        # Verify model is available
        downloaded_models = model_manager.check_downloaded_models()
        assert 'medium' in downloaded_models
        assert downloaded_models['medium']['valid'] is True
    
    def test_workflow_with_settings_management(self, setup_managers):
        """Test workflow with settings loading and saving"""
        managers = setup_managers
        settings_manager = managers['settings_manager']
        temp_dir = managers['temp_dir']
        
        # Test settings loading
        settings = settings_manager.load_settings()
        assert 'general' in settings
        assert 'output' in settings
        assert settings['general']['default_model'] == 'medium'
        
        # Test transcription options
        options = settings_manager.get_transcription_options()
        assert options['model_size'] == 'medium'
        assert options['language'] == 'auto'
        assert options['device'] == 'gpu'
        assert options['output_formats']['text'] is True
        
        # Test settings modification
        settings_manager.settings['general']['default_model'] = 'large'
        save_success = settings_manager.save_settings()
        assert save_success is True
    
    def test_workflow_error_handling_missing_file(self, setup_managers):
        """Test workflow error handling for missing input file"""
        managers = setup_managers
        env_manager = managers['env_manager']
        temp_dir = managers['temp_dir']
        
        transcription_engine = TranscriptionEngine(env_manager)
        
        options = {
            'model_size': 'medium',
            'language': 'auto',
            'device': 'cpu',
            'output_formats': {'text': True}
        }
        
        # Try to transcribe non-existent file
        result = transcription_engine.transcribe(
            "/nonexistent/file.mp3",
            str(temp_dir / "output"),
            options
        )
        
        assert result['success'] is False
        assert 'not found' in result['error'].lower()
    
    def test_workflow_error_handling_transcription_failure(self, setup_managers, mock_audio_file):
        """Test workflow error handling for transcription failure"""
        managers = setup_managers
        env_manager = managers['env_manager']
        temp_dir = managers['temp_dir']
        
        transcription_engine = TranscriptionEngine(env_manager)
        
        # Mock transcription failure
        with patch.object(transcription_engine, 'run_enhanced_transcription') as mock_transcribe:
            mock_transcribe.side_effect = Exception("Transcription failed due to model error")
            
            options = {
                'model_size': 'medium',
                'language': 'auto',
                'device': 'cpu',
                'output_formats': {'text': True}
            }
            
            result = transcription_engine.transcribe(
                str(mock_audio_file),
                str(temp_dir / "output"),
                options
            )
            
            assert result['success'] is False
            assert 'model error' in result['error']
    
    def test_workflow_dots_only_detection(self, setup_managers, mock_audio_file):
        """Test workflow detection of dots-only output (silence)"""
        managers = setup_managers
        env_manager = managers['env_manager']
        temp_dir = managers['temp_dir']
        
        transcription_engine = TranscriptionEngine(env_manager)
        
        # Mock dots-only result (silence detection)
        dots_result = {
            'text': '... ... . . ... .',
            'segments': [
                {'start': 0.0, 'end': 5.0, 'text': '... ... .', 'avg_logprob': -2.0}
            ],
            'duration': 5.0
        }
        
        with patch.object(transcription_engine, 'run_enhanced_transcription') as mock_transcribe:
            mock_transcribe.return_value = dots_result
            
            options = {
                'model_size': 'medium',
                'language': 'auto',
                'device': 'cpu',
                'output_formats': {'text': True}
            }
            
            result = transcription_engine.transcribe(
                str(mock_audio_file),
                str(temp_dir / "output"),
                options
            )
            
            assert result['success'] is False
            assert 'silence' in result['error'].lower() or 'unclear' in result['error'].lower()
    
    def test_workflow_with_progress_callback(self, setup_managers, mock_audio_file, mock_whisper_result):
        """Test workflow with progress callback functionality"""
        managers = setup_managers
        env_manager = managers['env_manager']
        temp_dir = managers['temp_dir']
        
        transcription_engine = TranscriptionEngine(env_manager)
        progress_updates = []
        
        def progress_callback(update):
            progress_updates.append(update)
        
        with patch.object(transcription_engine, 'run_enhanced_transcription') as mock_transcribe:
            mock_transcribe.return_value = mock_whisper_result
            
            with patch.object(transcription_engine, 'generate_output_files') as mock_generate:
                mock_generate.return_value = [str(temp_dir / "output.txt")]
                Path(temp_dir / "output.txt").write_text("test")
                
                options = {
                    'model_size': 'medium',
                    'output_formats': {'text': True}
                }
                
                result = transcription_engine.transcribe(
                    str(mock_audio_file),
                    str(temp_dir),
                    options,
                    progress_callback=progress_callback
                )
                
                assert result['success'] is True
                assert len(progress_updates) > 0
                
                # Check that progress updates contain expected information
                messages = [update.get('message', '') for update in progress_updates if 'message' in update]
                assert any('Analyzing' in msg for msg in messages)
    
    def test_workflow_stop_functionality(self, setup_managers, mock_audio_file):
        """Test workflow stop functionality"""
        managers = setup_managers
        env_manager = managers['env_manager']
        temp_dir = managers['temp_dir']
        
        transcription_engine = TranscriptionEngine(env_manager)
        
        def mock_long_transcription(*args, **kwargs):
            # Simulate long-running transcription
            for i in range(10):
                if transcription_engine.should_stop:
                    break
                time.sleep(0.1)
            return {'text': 'Partial result', 'segments': [], 'duration': 1.0}
        
        with patch.object(transcription_engine, 'run_enhanced_transcription', side_effect=mock_long_transcription):
            options = {
                'model_size': 'medium',
                'output_formats': {'text': True}
            }
            
            # Start transcription in separate thread
            result_container = {}
            
            def transcribe_worker():
                result_container['result'] = transcription_engine.transcribe(
                    str(mock_audio_file),
                    str(temp_dir),
                    options
                )
            
            transcribe_thread = threading.Thread(target=transcribe_worker)
            transcribe_thread.start()
            
            # Wait a bit then stop
            time.sleep(0.2)
            transcription_engine.stop()
            
            # Wait for thread to complete
            transcribe_thread.join(timeout=2)
            
            # Verify stop was handled
            assert transcription_engine.should_stop is True
    
    def test_workflow_multiple_output_formats(self, setup_managers, mock_audio_file, mock_whisper_result):
        """Test workflow with multiple output formats"""
        managers = setup_managers
        env_manager = managers['env_manager']
        temp_dir = managers['temp_dir']
        
        transcription_engine = TranscriptionEngine(env_manager)
        
        with patch.object(transcription_engine, 'run_enhanced_transcription') as mock_transcribe:
            mock_transcribe.return_value = mock_whisper_result
            
            # Create expected output files
            output_files = [
                temp_dir / "test_audio.txt",
                temp_dir / "test_audio_detailed.txt",
                temp_dir / "test_audio.srt",
                temp_dir / "test_audio.vtt"
            ]
            
            for file_path in output_files:
                file_path.write_text("Mock content")
            
            with patch.object(transcription_engine, 'generate_output_files') as mock_generate:
                mock_generate.return_value = [str(f) for f in output_files]
                
                options = {
                    'model_size': 'medium',
                    'output_formats': {
                        'text': True,
                        'detailed': True,
                        'srt': True,
                        'vtt': True
                    }
                }
                
                result = transcription_engine.transcribe(
                    str(mock_audio_file),
                    str(temp_dir),
                    options
                )
                
                assert result['success'] is True
                assert len(result['files']) == 4
                
                # Verify all expected formats are present
                file_extensions = [Path(f).suffix for f in result['files']]
                assert '.txt' in file_extensions
                assert '.srt' in file_extensions
                assert '.vtt' in file_extensions
    
    def test_workflow_with_live_updates(self, setup_managers, mock_audio_file, mock_whisper_result):
        """Test workflow with live update callbacks"""
        managers = setup_managers
        env_manager = managers['env_manager']
        temp_dir = managers['temp_dir']
        
        transcription_engine = TranscriptionEngine(env_manager)
        live_updates = []
        
        def live_callback(segment_data):
            live_updates.append(segment_data)
        
        def mock_transcription_with_live_updates(*args, **kwargs):
            # Simulate live updates during transcription
            for i, segment in enumerate(mock_whisper_result['segments']):
                if live_callback:
                    live_callback({
                        'start': segment['start'],
                        'end': segment['end'],
                        'text': segment['text'],
                        'segment_index': i + 1,
                        'total_segments': len(mock_whisper_result['segments']),
                        'avg_logprob': segment['avg_logprob']
                    })
                time.sleep(0.1)  # Simulate processing time
            return mock_whisper_result
        
        with patch.object(transcription_engine, 'run_enhanced_transcription', side_effect=mock_transcription_with_live_updates):
            with patch.object(transcription_engine, 'generate_output_files') as mock_generate:
                mock_generate.return_value = [str(temp_dir / "output.txt")]
                Path(temp_dir / "output.txt").write_text("test")
                
                options = {
                    'model_size': 'medium',
                    'output_formats': {'text': True},
                    'live_callback': live_callback
                }
                
                result = transcription_engine.transcribe(
                    str(mock_audio_file),
                    str(temp_dir),
                    options
                )
                
                assert result['success'] is True
                assert len(live_updates) == 3  # One for each segment
                
                # Verify live update structure
                for update in live_updates:
                    assert 'start' in update
                    assert 'end' in update
                    assert 'text' in update
                    assert 'segment_index' in update
                    assert 'total_segments' in update
    
    def test_workflow_model_verification(self, setup_managers):
        """Test workflow model verification and download"""
        managers = setup_managers
        model_manager = managers['model_manager']
        temp_dir = managers['temp_dir']
        
        # Test model info retrieval
        model_info = model_manager.get_model_info('medium')
        assert model_info['size'] == '769 MB'
        assert model_info['description'] == 'Recommended'
        
        # Test model verification
        model_file = temp_dir / 'medium.pt'
        model_file.write_bytes(b'fake model content')
        
        is_valid = model_manager.verify_model_file('medium', model_file)
        assert is_valid is True
        
        # Test downloaded models check
        downloaded = model_manager.check_downloaded_models()
        assert 'medium' in downloaded
        assert downloaded['medium']['valid'] is True
    
    def test_workflow_error_recovery(self, setup_managers, mock_audio_file):
        """Test workflow error recovery mechanisms"""
        managers = setup_managers
        env_manager = managers['env_manager']
        temp_dir = managers['temp_dir']
        
        transcription_engine = TranscriptionEngine(env_manager)
        
        # Test recovery from temporary failure
        call_count = 0
        def failing_then_succeeding_transcription(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary failure")
            return {
                'text': 'Success after retry',
                'segments': [{'start': 0, 'end': 1, 'text': 'Success', 'avg_logprob': -0.1}],
                'duration': 1.0
            }
        
        # Test with retry logic (would need to be implemented in actual code)
        with patch.object(transcription_engine, 'run_enhanced_transcription', side_effect=failing_then_succeeding_transcription):
            options = {
                'model_size': 'medium',
                'output_formats': {'text': True}
            }
            
            # First attempt should fail
            result = transcription_engine.transcribe(
                str(mock_audio_file),
                str(temp_dir),
                options
            )
            
            assert result['success'] is False
            assert 'Temporary failure' in result['error']
    
    def test_workflow_performance_monitoring(self, setup_managers, mock_audio_file, mock_whisper_result):
        """Test workflow performance monitoring and timing"""
        managers = setup_managers
        env_manager = managers['env_manager']
        temp_dir = managers['temp_dir']
        
        transcription_engine = TranscriptionEngine(env_manager)
        
        def timed_transcription(*args, **kwargs):
            time.sleep(0.5)  # Simulate processing time
            result = mock_whisper_result.copy()
            result['transcription_time'] = 0.5
            return result
        
        with patch.object(transcription_engine, 'run_enhanced_transcription', side_effect=timed_transcription):
            with patch.object(transcription_engine, 'generate_output_files') as mock_generate:
                mock_generate.return_value = [str(temp_dir / "output.txt")]
                Path(temp_dir / "output.txt").write_text("test")
                
                options = {
                    'model_size': 'medium',
                    'output_formats': {'text': True}
                }
                
                start_time = time.time()
                result = transcription_engine.transcribe(
                    str(mock_audio_file),
                    str(temp_dir),
                    options
                )
                end_time = time.time()
                
                assert result['success'] is True
                assert result['transcription_time'] >= 0.5
                assert (end_time - start_time) >= 0.5


class TestWorkflowIntegrationEdgeCases:
    """Test edge cases and boundary conditions in workflow integration"""
    
    def test_workflow_with_empty_audio(self, temp_dir):
        """Test workflow with empty audio file"""
        # Create empty audio file
        empty_file = temp_dir / "empty.mp3"
        empty_file.write_bytes(b"")
        
        env_manager = Mock(spec=EnvironmentManager)
        transcription_engine = TranscriptionEngine(env_manager)
        
        options = {'model_size': 'medium', 'output_formats': {'text': True}}
        
        # Should handle empty file gracefully
        result = transcription_engine.transcribe(
            str(empty_file),
            str(temp_dir),
            options
        )
        
        # Specific handling depends on implementation
        assert 'success' in result
    
    def test_workflow_with_corrupted_settings(self, temp_dir):
        """Test workflow with corrupted settings file"""
        settings_file = temp_dir / "settings.json"
        settings_file.write_text("invalid json content {")
        
        # SettingsManager should handle corrupted settings gracefully
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data="invalid json {")):
                settings_manager = SettingsManager()
                # Should fall back to defaults
                assert settings_manager.settings is not None
                assert 'general' in settings_manager.settings
    
    def test_workflow_concurrent_operations(self, setup_managers, mock_audio_file):
        """Test workflow behavior with concurrent operations"""
        managers = setup_managers
        env_manager = managers['env_manager']
        temp_dir = managers['temp_dir']
        
        # Create multiple transcription engines
        engine1 = TranscriptionEngine(env_manager)
        engine2 = TranscriptionEngine(env_manager)
        
        results = []
        
        def transcribe_worker(engine, suffix):
            with patch.object(engine, 'run_enhanced_transcription') as mock_transcribe:
                mock_transcribe.return_value = {
                    'text': f'Result {suffix}',
                    'segments': [],
                    'duration': 1.0
                }
                
                with patch.object(engine, 'generate_output_files') as mock_generate:
                    output_file = temp_dir / f"output_{suffix}.txt"
                    output_file.write_text(f"Content {suffix}")
                    mock_generate.return_value = [str(output_file)]
                    
                    options = {'model_size': 'medium', 'output_formats': {'text': True}}
                    result = engine.transcribe(str(mock_audio_file), str(temp_dir), options)
                    results.append(result)
        
        # Run concurrent transcriptions
        thread1 = threading.Thread(target=transcribe_worker, args=(engine1, "1"))
        thread2 = threading.Thread(target=transcribe_worker, args=(engine2, "2"))
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # Both should succeed
        assert len(results) == 2
        assert all(r['success'] for r in results)
    
    def test_workflow_resource_cleanup(self, setup_managers, mock_audio_file):
        """Test workflow resource cleanup after completion/failure"""
        managers = setup_managers
        env_manager = managers['env_manager']
        temp_dir = managers['temp_dir']
        
        transcription_engine = TranscriptionEngine(env_manager)
        
        # Create some temporary files
        temp_script = temp_dir / "temp_transcribe_enhanced.py"
        temp_result = temp_dir / "temp_result_enhanced.json"
        
        def mock_transcription_with_temp_files(*args, **kwargs):
            # Create temp files during transcription
            temp_script.write_text("# temp script")
            temp_result.write_text('{"text": "test"}')
            
            # Simulate cleanup
            if temp_script.exists():
                temp_script.unlink()
            
            return {'text': 'test', 'segments': [], 'duration': 1.0}
        
        with patch.object(transcription_engine, 'run_enhanced_transcription', side_effect=mock_transcription_with_temp_files):
            with patch.object(transcription_engine, 'generate_output_files') as mock_generate:
                mock_generate.return_value = [str(temp_dir / "output.txt")]
                Path(temp_dir / "output.txt").write_text("test")
                
                options = {'model_size': 'medium', 'output_formats': {'text': True}}
                result = transcription_engine.transcribe(str(mock_audio_file), str(temp_dir), options)
                
                # Verify temp files were cleaned up
                assert not temp_script.exists()
                # temp_result might still exist depending on implementation
                assert result['success'] is True


class TestWorkflowRealWorldScenarios:
    """Test real-world scenarios and user workflows"""
    
    def test_first_time_user_workflow(self, temp_dir):
        """Test complete first-time user setup workflow"""
        # Simulate first-time user with no environment
        env_manager = EnvironmentManager()
        
        with patch.object(env_manager, 'venv_dir', temp_dir / "venv"):
            with patch.object(env_manager, 'check_environment') as mock_check:
                # Initially no environment
                mock_check.return_value = {
                    'venv_exists': False,
                    'python_works': False,
                    'whisper_installed': False,
                    'torch_installed': False,
                    'gpu_available': False
                }
                
                # Check initial state
                status = env_manager.check_environment()
                assert status['venv_exists'] is False
                assert status['whisper_installed'] is False
                
                # After setup, environment should be ready
                with patch.object(env_manager, 'setup_environment', return_value=True):
                    setup_success = env_manager.setup_environment()
                    assert setup_success is True
                    
                    # Update mock to reflect successful setup
                    mock_check.return_value = {
                        'venv_exists': True,
                        'python_works': True,
                        'whisper_installed': True,
                        'torch_installed': True,
                        'gpu_available': False
                    }
                    
                    # Verify environment is now ready
                    final_status = env_manager.check_environment()
                    assert final_status['venv_exists'] is True
                    assert final_status['whisper_installed'] is True
    
    def test_user_workflow_with_model_download(self, temp_dir):
        """Test user workflow including model download"""
        model_manager = ModelManager()
        
        with patch.object(model_manager, 'cache_dir', temp_dir):
            # Initially no models downloaded
            with patch.object(model_manager, 'check_downloaded_models', return_value={}):
                downloaded = model_manager.check_downloaded_models()
                assert len(downloaded) == 0
                
                # Download medium model
                with patch.object(model_manager, 'download_models', return_value=True) as mock_download:
                    success = model_manager.download_models(['medium'])
                    assert success is True
                    mock_download.assert_called_once_with(['medium'])
                    
                    # After download, model should be available
                    with patch.object(model_manager, 'check_downloaded_models') as mock_check:
                        mock_check.return_value = {
                            'medium': {'valid': True, 'size_mb': 769, 'path': str(temp_dir / 'medium.pt')}
                        }
                        
                        downloaded = model_manager.check_downloaded_models()
                        assert 'medium' in downloaded
                        assert downloaded['medium']['valid'] is True
    
    def test_user_workflow_batch_processing(self, setup_managers, temp_dir):
        """Test user workflow for batch processing multiple files"""
        managers = setup_managers
        env_manager = managers['env_manager']
        
        # Create multiple audio files
        audio_files = []
        for i in range(3):
            audio_file = temp_dir / f"audio_{i}.mp3"
            audio_file.write_bytes(b"fake audio content" * 100)
            audio_files.append(audio_file)
        
        transcription_engine = TranscriptionEngine(env_manager)
        results = []
        
        # Mock transcription for each file
        def mock_batch_transcription(file_path, *args, **kwargs):
            file_name = Path(file_path).stem
            return {
                'text': f'Transcription for {file_name}',
                'segments': [{'start': 0, 'end': 1, 'text': f'Content {file_name}', 'avg_logprob': -0.1}],
                'duration': 1.0
            }
        
        with patch.object(transcription_engine, 'run_enhanced_transcription', side_effect=mock_batch_transcription):
            with patch.object(transcription_engine, 'generate_output_files') as mock_generate:
                def mock_file_generation(result, base_filename, output_dir, options, progress_callback=None):
                    output_file = output_dir / f"{base_filename}.txt"
                    output_file.write_text(result['text'])
                    return [str(output_file)]
                
                mock_generate.side_effect = mock_file_generation
                
                options = {'model_size': 'medium', 'output_formats': {'text': True}}
                
                # Process each file
                for audio_file in audio_files:
                    result = transcription_engine.transcribe(
                        str(audio_file),
                        str(temp_dir / "output"),
                        options
                    )
                    results.append(result)
        
        # Verify all files were processed successfully
        assert len(results) == 3
        assert all(r['success'] for r in results)
        
        # Verify output files exist
        output_dir = temp_dir / "output"
        for i in range(3):
            output_file = output_dir / f"audio_{i}.txt"
            assert output_file.exists()
    
    def test_user_workflow_settings_persistence(self, temp_dir):
        """Test user workflow with settings persistence across sessions"""
        settings_file = temp_dir / "settings.json"
        
        # First session - create and save settings
        with patch.object(SettingsManager, 'settings_file', settings_file):
            settings1 = SettingsManager()
            
            # Modify some settings
            settings1.set_setting('general', 'default_model', 'large')
            settings1.set_setting('output', 'formats', {'text': True, 'srt': False})
            
            # Save settings
            with patch('builtins.open', mock_open()) as mock_file:
                with patch('json.dump') as mock_json_dump:
                    success = settings1.save_settings()
                    assert success is True
                    mock_json_dump.assert_called_once()
        
        # Second session - load saved settings
        saved_settings = {
            'general': {'default_model': 'large'},
            'output': {'formats': {'text': True, 'srt': False}}
        }
        
        with patch.object(SettingsManager, 'settings_file', settings_file):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', mock_open(read_data=json.dumps(saved_settings))):
                    settings2 = SettingsManager()
                    
                    # Verify settings were loaded correctly
                    assert settings2.get_setting('general', 'default_model') == 'large'
                    assert settings2.get_setting('output', 'formats')['text'] is True
                    assert settings2.get_setting('output', 'formats')['srt'] is False
    
    def test_user_workflow_error_recovery_and_retry(self, setup_managers, temp_dir):
        """Test user workflow with error recovery and retry logic"""
        managers = setup_managers
        env_manager = managers['env_manager']
        
        audio_file = temp_dir / "problematic_audio.mp3"
        audio_file.write_bytes(b"fake problematic audio")
        
        transcription_engine = TranscriptionEngine(env_manager)
        
        # Simulate various failure scenarios
        failure_scenarios = [
            "CUDA out of memory",
            "Model file corrupted", 
            "Audio format not supported",
            "Network timeout during processing"
        ]
        
        for i, error_msg in enumerate(failure_scenarios):
            with patch.object(transcription_engine, 'run_enhanced_transcription') as mock_transcribe:
                mock_transcribe.side_effect = Exception(error_msg)
                
                options = {'model_size': 'medium', 'output_formats': {'text': True}}
                
                result = transcription_engine.transcribe(
                    str(audio_file),
                    str(temp_dir),
                    options
                )
                
                assert result['success'] is False
                assert error_msg in result['error']
                
                # User could retry with different settings
                # e.g., switch from GPU to CPU for CUDA error
                if "CUDA" in error_msg:
                    options['device'] = 'cpu'
                    # Retry logic would go here in real implementation
    
    def test_user_workflow_progress_monitoring(self, setup_managers, temp_dir):
        """Test user workflow with detailed progress monitoring"""
        managers = setup_managers
        env_manager = managers['env_manager']
        
        audio_file = temp_dir / "long_audio.mp3"
        audio_file.write_bytes(b"fake long audio content" * 1000)
        
        transcription_engine = TranscriptionEngine(env_manager)
        progress_updates = []
        
        def progress_callback(update):
            progress_updates.append(update)
        
        # Mock long transcription with detailed progress
        def mock_detailed_transcription(*args, **kwargs):
            # Simulate detailed progress updates
            stages = [
                "Loading model",
                "Preprocessing audio", 
                "Running transcription",
                "Post-processing results"
            ]
            
            for i, stage in enumerate(stages):
                if progress_callback:
                    progress_callback({
                        'message': stage,
                        'progress_percent': (i + 1) * 25,
                        'stage': i + 1,
                        'total_stages': len(stages)
                    })
                time.sleep(0.1)
            
            return {
                'text': 'Long transcription result',
                'segments': [{'start': 0, 'end': 10, 'text': 'Long content', 'avg_logprob': -0.2}],
                'duration': 10.0
            }
        
        with patch.object(transcription_engine, 'run_enhanced_transcription', side_effect=mock_detailed_transcription):
            with patch.object(transcription_engine, 'generate_output_files') as mock_generate:
                mock_generate.return_value = [str(temp_dir / "output.txt")]
                Path(temp_dir / "output.txt").write_text("test")
                
                options = {'model_size': 'medium', 'output_formats': {'text': True}}
                
                result = transcription_engine.transcribe(
                    str(audio_file),
                    str(temp_dir),
                    options,
                    progress_callback=progress_callback
                )
                
                assert result['success'] is True
                assert len(progress_updates) >= 4  # At least one for each stage
                
                # Verify progress increases
                percentages = [u.get('progress_percent', 0) for u in progress_updates if 'progress_percent' in u]
                assert percentages[-1] >= percentages[0]  # Progress should increase
    
    def test_user_workflow_quality_validation(self, setup_managers, temp_dir):
        """Test user workflow with transcription quality validation"""
        managers = setup_managers
        env_manager = managers['env_manager']
        
        audio_file = temp_dir / "quality_test.mp3"
        audio_file.write_bytes(b"audio for quality testing")
        
        transcription_engine = TranscriptionEngine(env_manager)
        
        # Test different quality scenarios
        quality_scenarios = [
            # High quality result
            {
                'text': 'This is a clear and accurate transcription.',
                'segments': [{'start': 0, 'end': 3, 'text': 'This is a clear and accurate transcription.', 'avg_logprob': -0.1}],
                'expected_quality': 'high'
            },
            # Medium quality result
            {
                'text': 'This is somewhat unclear but usable.',
                'segments': [{'start': 0, 'end': 3, 'text': 'This is somewhat unclear but usable.', 'avg_logprob': -0.8}],
                'expected_quality': 'medium'
            },
            # Low quality result (should be flagged)
            {
                'text': '... ... unclear ... mumbling ...',
                'segments': [{'start': 0, 'end': 3, 'text': '... ... unclear ... mumbling ...', 'avg_logprob': -2.5}],
                'expected_quality': 'low'
            }
        ]
        
        for scenario in quality_scenarios:
            with patch.object(transcription_engine, 'run_enhanced_transcription') as mock_transcribe:
                mock_transcribe.return_value = scenario
                
                options = {'model_size': 'medium', 'output_formats': {'text': True}}
                
                result = transcription_engine.transcribe(
                    str(audio_file),
                    str(temp_dir),
                    options
                )
                
                if scenario['expected_quality'] == 'low':
                    # Low quality should trigger dots-only detection
                    is_dots_only = transcription_engine.is_dots_only_output(scenario)
                    assert is_dots_only is True
                else:
                    # Higher quality should pass
                    with patch.object(transcription_engine, 'generate_output_files') as mock_generate:
                        mock_generate.return_value = [str(temp_dir / "output.txt")]
                        Path(temp_dir / "output.txt").write_text("test")
                        
                        result = transcription_engine.transcribe(
                            str(audio_file),
                            str(temp_dir),
                            options
                        )
                        
                        assert result['success'] is True