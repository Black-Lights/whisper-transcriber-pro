"""
Performance tests for Whisper Transcriber Pro
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
import gc
import sys
from pathlib import Path

# Try to import memory profiler for memory tests
try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

from src.transcription_engine import TranscriptionEngine
from src.model_manager import ModelManager
from src.environment_manager import EnvironmentManager
from src.utils import get_file_info, format_file_size


class TestPerformance:
    """Performance tests for core functionality"""
    
    @pytest.mark.performance
    def test_startup_time_performance(self, mock_env_manager):
        """Test application startup time performance"""
        start_time = time.time()
        
        # Initialize core components
        engine = TranscriptionEngine(mock_env_manager)
        model_manager = ModelManager()
        
        # Simulate basic initialization tasks
        engine.should_stop = False
        model_manager.get_model_info()
        
        end_time = time.time()
        startup_time = end_time - start_time
        
        # Startup should be very fast for mocked components
        assert startup_time < 1.0, f"Startup took {startup_time:.2f}s, expected < 1.0s"
    
    @pytest.mark.performance
    def test_file_info_extraction_performance(self, temp_dir):
        """Test performance of file information extraction"""
        # Create test files of various sizes
        test_files = []
        sizes = [1024, 1024*100, 1024*1024]  # 1KB, 100KB, 1MB
        
        for i, size in enumerate(sizes):
            test_file = temp_dir / f"test_{i}.mp3"
            test_file.write_bytes(b"x" * size)
            test_files.append(test_file)
        
        # Measure time for file info extraction
        start_time = time.time()
        
        for test_file in test_files:
            info = get_file_info(str(test_file))
            assert "Size:" in info
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process all files quickly
        assert processing_time < 1.0, f"File info extraction took {processing_time:.2f}s"
    
    @pytest.mark.performance
    def test_model_cache_size_calculation_performance(self, temp_dir):
        """Test performance of model cache size calculation"""
        model_manager = ModelManager()
        original_cache_dir = model_manager.cache_dir
        model_manager.cache_dir = temp_dir
        
        try:
            # Create many mock model files
            num_files = 100
            for i in range(num_files):
                model_file = temp_dir / f"model_{i}.pt"
                model_file.write_bytes(b"x" * (1024 * 100))  # 100KB each
            
            # Measure cache size calculation time
            start_time = time.time()
            total_size = model_manager.get_cache_size()
            end_time = time.time()
            
            calculation_time = end_time - start_time
            expected_size = num_files * 1024 * 100
            
            assert total_size == expected_size
            assert calculation_time < 1.0, f"Cache calculation took {calculation_time:.2f}s"
            
        finally:
            model_manager.cache_dir = original_cache_dir
    
    @pytest.mark.performance
    def test_settings_load_save_performance(self, temp_dir):
        """Test performance of settings loading and saving"""
        from src.settings_manager import SettingsManager
        
        settings_manager = SettingsManager()
        settings_file = temp_dir / "perf_settings.json"
        settings_manager.settings_file = settings_file
        
        # Create large settings object
        for i in range(1000):
            settings_manager.set_setting("test_category", f"key_{i}", f"value_{i}")
        
        # Measure save performance
        start_time = time.time()
        save_result = settings_manager.save_settings()
        save_time = time.time() - start_time
        
        assert save_result is True
        assert save_time < 1.0, f"Settings save took {save_time:.2f}s"
        
        # Measure load performance
        new_manager = SettingsManager()
        new_manager.settings_file = settings_file
        
        start_time = time.time()
        loaded_settings = new_manager.load_settings()
        load_time = time.time() - start_time
        
        assert len(loaded_settings["test_category"]) == 1000
        assert load_time < 1.0, f"Settings load took {load_time:.2f}s"
    
    @pytest.mark.performance
    def test_concurrent_file_operations_performance(self, temp_dir):
        """Test performance under concurrent file operations"""
        import concurrent.futures
        
        def create_and_process_file(file_id):
            """Create a file and get its info"""
            test_file = temp_dir / f"concurrent_{file_id}.mp3"
            test_file.write_bytes(b"x" * (1024 * 10))  # 10KB
            
            start_time = time.time()
            info = get_file_info(str(test_file))
            processing_time = time.time() - start_time
            
            return processing_time, len(info)
        
        # Run concurrent operations
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_and_process_file, i) for i in range(50)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Verify all operations completed successfully
        assert len(results) == 50
        avg_processing_time = sum(result[0] for result in results) / len(results)
        
        assert total_time < 10.0, f"Concurrent operations took {total_time:.2f}s"
        assert avg_processing_time < 0.1, f"Average processing time {avg_processing_time:.3f}s too slow"
    
    @pytest.mark.performance
    @pytest.mark.skipif(not MEMORY_PROFILER_AVAILABLE, reason="memory_profiler not available")
    def test_memory_usage_during_processing(self, mock_env_manager):
        """Test memory usage during transcription processing"""
        engine = TranscriptionEngine(mock_env_manager)
        
        @memory_profiler.profile
        def run_mock_transcription():
            # Simulate memory-intensive operations
            large_data = []
            for i in range(1000):
                # Simulate segment data
                segment = {
                    'start': i * 0.5,
                    'end': (i + 1) * 0.5,
                    'text': f'This is segment {i} with some text content',
                    'avg_logprob': -0.3
                }
                large_data.append(segment)
            
            # Simulate processing
            for segment in large_data:
                engine.is_paused = False
                engine.should_stop = False
            
            return len(large_data)
        
        # Run and measure memory
        start_time = time.time()
        result = run_mock_transcription()
        end_time = time.time()
        
        # Force garbage collection
        gc.collect()
        
        assert result == 1000
        assert end_time - start_time < 1.0, "Memory test took too long"
    
    @pytest.mark.performance
    def test_large_file_size_formatting_performance(self):
        """Test performance of file size formatting with large numbers"""
        large_sizes = [
            1024**2,      # 1 MB
            1024**3,      # 1 GB  
            1024**4,      # 1 TB
            1024**5,      # 1 PB
            1024**2 * 1500,  # ~1.5 GB
            1024**3 * 500,   # ~500 GB
        ]
        
        start_time = time.time()
        
        results = []
        for size in large_sizes:
            formatted = format_file_size(size)
            results.append(formatted)
        
        end_time = time.time()
        formatting_time = end_time - start_time
        
        # Verify all formats are correct
        assert "1.0 MB" in results[0]
        assert "1.0 GB" in results[1]
        assert "1.0 TB" in results[2]
        assert "1.0 PB" in results[3]
        
        # Should be very fast
        assert formatting_time < 0.01, f"Size formatting took {formatting_time:.4f}s"
    
    @pytest.mark.performance
    def test_model_info_retrieval_performance(self):
        """Test performance of model information retrieval"""
        model_manager = ModelManager()
        
        # Test repeated info retrieval
        start_time = time.time()
        
        for _ in range(1000):
            all_models = model_manager.get_model_info()
            medium_info = model_manager.get_model_info('medium')
            tiny_info = model_manager.get_model_info('tiny')
        
        end_time = time.time()
        retrieval_time = end_time - start_time
        
        assert len(all_models) == 5
        assert medium_info['size'] == '769 MB'
        assert tiny_info['size'] == '39 MB'
        assert retrieval_time < 1.0, f"Model info retrieval took {retrieval_time:.2f}s"
    
    @pytest.mark.performance
    def test_transcription_state_management_performance(self, mock_env_manager):
        """Test performance of transcription state management"""
        engine = TranscriptionEngine(mock_env_manager)
        
        start_time = time.time()
        
        # Simulate rapid state changes
        for i in range(10000):
            engine.should_stop = False
            engine.is_paused = i % 2 == 0
            engine.current_process = Mock() if i % 3 == 0 else None
            
            # Simulate state checks
            if engine.is_paused:
                engine.resume()
            if not engine.should_stop:
                engine.pause()
        
        end_time = time.time()
        state_time = end_time - start_time
        
        assert state_time < 1.0, f"State management took {state_time:.2f}s"
    
    @pytest.mark.performance
    def test_environment_check_performance(self, temp_dir):
        """Test performance of environment checking"""
        env_manager = EnvironmentManager()
        original_venv_dir = env_manager.venv_dir
        env_manager.venv_dir = temp_dir / "test_venv"
        
        try:
            # Create mock venv structure
            scripts_dir = env_manager.venv_dir / ("Scripts" if env_manager.is_windows else "bin")
            scripts_dir.mkdir(parents=True, exist_ok=True)
            
            python_exe = scripts_dir / ("python.exe" if env_manager.is_windows else "python")
            python_exe.write_text("fake python")
            
            # Mock subprocess calls for performance
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = Mock(returncode=0, stdout="Python 3.9.0")
                
                start_time = time.time()
                
                # Run multiple environment checks
                for _ in range(100):
                    status = env_manager.check_environment()
                
                end_time = time.time()
                check_time = end_time - start_time
                
                assert status['venv_exists'] is True
                assert check_time < 5.0, f"Environment checks took {check_time:.2f}s"
                
        finally:
            env_manager.venv_dir = original_venv_dir


class TestMemoryPerformance:
    """Memory-specific performance tests"""
    
    @pytest.mark.performance
    def test_memory_cleanup_after_transcription(self, mock_env_manager, temp_dir):
        """Test memory cleanup after transcription"""
        engine = TranscriptionEngine(mock_env_manager)
        
        # Get initial memory usage
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Simulate transcription with large data
        audio_file = temp_dir / "test.mp3"
        audio_file.write_bytes(b"x" * (1024 * 1024))  # 1MB file
        
        large_segments = []
        for i in range(1000):
            segment = {
                'start': i * 0.1,
                'end': (i + 1) * 0.1,
                'text': f'This is a longer segment {i} with more detailed text content for memory testing',
                'avg_logprob': -0.3 - (i * 0.001)
            }
            large_segments.append(segment)
        
        mock_result = {
            'text': ' '.join(seg['text'] for seg in large_segments),
            'segments': large_segments,
            'duration': len(large_segments) * 0.1
        }
        
        # Mock transcription process
        with patch('subprocess.Popen') as mock_popen, \
             patch('builtins.open', create=True) as mock_file, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('json.loads', return_value=mock_result):
            
            mock_process = Mock()
            mock_process.poll.return_value = 0
            mock_process.returncode = 0
            mock_process.stdout.readline.return_value = ""
            mock_process.communicate.return_value = ("", "")
            mock_popen.return_value = mock_process
            
            options = {
                'model_size': 'medium',
                'device': 'cpu',
                'output_formats': {'text': True}
            }
            
            result = engine.transcribe(str(audio_file), str(temp_dir), options)
        
        # Force garbage collection
        del large_segments
        del mock_result
        gc.collect()
        
        # Check memory after cleanup
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        assert result['success'] is True
        # Memory increase should be reasonable (less than 50MB)
        assert memory_increase < 50 * 1024 * 1024, f"Memory increased by {memory_increase / (1024*1024):.1f}MB"
    
    @pytest.mark.performance
    def test_memory_usage_with_large_cache(self, temp_dir):
        """Test memory usage with large model cache"""
        model_manager = ModelManager()
        original_cache_dir = model_manager.cache_dir
        model_manager.cache_dir = temp_dir
        
        try:
            # Create many large mock files
            num_files = 1000
            file_size = 1024 * 100  # 100KB each
            
            for i in range(num_files):
                model_file = temp_dir / f"model_{i}.pt"
                model_file.write_bytes(b"x" * file_size)
            
            # Measure memory usage during cache operations
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            # Perform cache operations
            total_size = model_manager.get_cache_size()
            downloaded = model_manager.check_downloaded_models()
            
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            expected_total = num_files * file_size
            assert total_size == expected_total
            assert len(downloaded) == 0  # No valid models (wrong hash)
            
            # Memory increase should be minimal for cache operations
            assert memory_increase < 10 * 1024 * 1024, f"Memory increased by {memory_increase / (1024*1024):.1f}MB"
            
        finally:
            model_manager.cache_dir = original_cache_dir
    
    @pytest.mark.performance
    def test_memory_efficient_file_processing(self, temp_dir):
        """Test memory-efficient processing of large files"""
        # Create a large file
        large_file = temp_dir / "large_test.mp3"
        file_size = 10 * 1024 * 1024  # 10MB
        
        # Write file in chunks to avoid memory issues in test
        with open(large_file, 'wb') as f:
            chunk_size = 1024 * 1024  # 1MB chunks
            for _ in range(10):
                f.write(b"x" * chunk_size)
        
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Process file info
        info = get_file_info(str(large_file))
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        assert "10.0 MB" in info
        # Processing a 10MB file should not significantly increase memory
        assert memory_increase < 5 * 1024 * 1024, f"Memory increased by {memory_increase / (1024*1024):.1f}MB"


class TestScalabilityPerformance:
    """Scalability performance tests"""
    
    @pytest.mark.performance
    def test_scalability_with_many_models(self, temp_dir):
        """Test performance scalability with many models"""
        model_manager = ModelManager()
        original_models = model_manager.models.copy()
        
        try:
            # Add many test models
            for i in range(100):
                model_manager.models[f'test_model_{i}'] = {
                    'size': f'{i+1} MB',
                    'description': f'Test model {i}',
                    'speed': f'~{i+1}x real-time',
                    'accuracy': 'Test',
                    'url': f'https://example.com/model_{i}.pt',
                    'sha256': 'a' * 64
                }
            
            start_time = time.time()
            
            # Test operations with many models
            all_models = model_manager.get_model_info()
            summary = model_manager.get_model_status_summary()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            assert len(all_models) == 105  # Original 5 + 100 test models
            assert summary['total_models'] == 105
            assert processing_time < 1.0, f"Processing {len(all_models)} models took {processing_time:.2f}s"
            
        finally:
            model_manager.models = original_models
    
    @pytest.mark.performance
    def test_scalability_with_many_settings(self, temp_dir):
        """Test performance scalability with many settings"""
        from src.settings_manager import SettingsManager
        
        settings_manager = SettingsManager()
        settings_file = temp_dir / "scalability_settings.json"
        settings_manager.settings_file = settings_file
        
        start_time = time.time()
        
        # Add many settings
        num_categories = 100
        settings_per_category = 100
        
        for cat_i in range(num_categories):
            category = f"category_{cat_i}"
            for set_i in range(settings_per_category):
                key = f"setting_{set_i}"
                value = f"value_{cat_i}_{set_i}"
                settings_manager.set_setting(category, key, value)
        
        # Save and load
        save_result = settings_manager.save_settings()
        
        # Create new manager and load
        new_manager = SettingsManager()
        new_manager.settings_file = settings_file
        loaded_settings = new_manager.load_settings()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        total_settings = num_categories * settings_per_category
        assert save_result is True
        assert len(loaded_settings) >= num_categories
        assert processing_time < 5.0, f"Processing {total_settings} settings took {processing_time:.2f}s"
    
    @pytest.mark.performance
    def test_concurrent_transcription_operations(self, mock_env_manager, temp_dir):
        """Test concurrent transcription state operations"""
        import threading
        import queue
        
        engines = [TranscriptionEngine(mock_env_manager) for _ in range(10)]
        results_queue = queue.Queue()
        
        def worker(engine_id, engine):
            try:
                start_time = time.time()
                
                # Simulate rapid operations
                for i in range(100):
                    engine.should_stop = False
                    engine.is_paused = i % 2 == 0
                    
                    if engine.is_paused:
                        engine.resume()
                    else:
                        engine.pause()
                    
                    if i % 10 == 0:
                        engine.stop()
                        engine.should_stop = False
                
                end_time = time.time()
                results_queue.put((engine_id, end_time - start_time, True))
                
            except Exception as e:
                results_queue.put((engine_id, 0, False, str(e)))
        
        # Start all workers
        threads = []
        start_time = time.time()
        
        for i, engine in enumerate(engines):
            thread = threading.Thread(target=worker, args=(i, engine))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) == 10
        assert all(result[2] for result in results), "Some workers failed"
        assert total_time < 10.0, f"Concurrent operations took {total_time:.2f}s"
        
        # Average per-worker time should be reasonable
        avg_worker_time = sum(result[1] for result in results) / len(results)
        assert avg_worker_time < 2.0, f"Average worker time {avg_worker_time:.2f}s too slow"


class TestRealWorldPerformance:
    """Real-world scenario performance tests"""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_realistic_transcription_workflow_performance(self, mock_env_manager, temp_dir):
        """Test performance of realistic transcription workflow"""
        engine = TranscriptionEngine(mock_env_manager)
        
        # Create realistic test scenario
        audio_file = temp_dir / "realistic_test.mp3"
        audio_file.write_bytes(b"x" * (5 * 1024 * 1024))  # 5MB audio file
        
        # Create realistic segment data (10 minutes of audio)
        realistic_segments = []
        for i in range(200):  # 200 segments = ~3 second segments
            segment = {
                'start': i * 3.0,
                'end': (i + 1) * 3.0,
                'text': f'This is segment {i} with realistic speech content that would be typical in a real transcription scenario.',
                'avg_logprob': -0.3 + (i % 10) * 0.01  # Varying confidence
            }
            realistic_segments.append(segment)
        
        mock_result = {
            'text': ' '.join(seg['text'] for seg in realistic_segments),
            'segments': realistic_segments,
            'duration': 600.0  # 10 minutes
        }
        
        # Track performance metrics
        progress_updates = []
        live_updates = []
        
        def progress_callback(update):
            progress_updates.append((time.time(), update))
        
        def live_callback(segment_data):
            live_updates.append((time.time(), segment_data))
        
        # Run realistic workflow
        start_time = time.time()
        
        with patch('subprocess.Popen') as mock_popen, \
             patch('builtins.open', create=True) as mock_file, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('json.loads', return_value=mock_result):
            
            mock_process = Mock()
            mock_process.poll.side_effect = [None] * 10 + [0]  # Simulate processing time
            mock_process.returncode = 0
            
            # Simulate realistic live updates
            mock_process.stdout.readline.side_effect = [
                'LIVE_UPDATE: {"message": "Loading model"}\n',
                'LIVE_UPDATE: {"message": "Processing audio"}\n',
            ] + [
                f'LIVE_UPDATE: {{"message": "Segment {i}", "live_segment": {{"start": {i*3}, "end": {(i+1)*3}, "text": "Segment {i}", "segment_index": {i+1}, "total_segments": 200}}}}\n'
                for i in range(5)  # Sample of segments
            ] + [
                'TRANSCRIPTION_COMPLETE\n',
                ''
            ]
            
            mock_process.communicate.return_value = ("", "")
            mock_popen.return_value = mock_process
            
            options = {
                'model_size': 'medium',
                'device': 'cpu',
                'output_formats': {'text': True, 'srt': True, 'detailed': True},
                'live_callback': live_callback,
                'clean_text': True
            }
            
            result = engine.transcribe(str(audio_file), str(temp_dir), options, progress_callback)
        
        end_time = time.time()
        total_workflow_time = end_time - start_time
        
        # Performance assertions
        assert result['success'] is True
        assert len(progress_updates) > 0
        assert len(live_updates) >= 5
        assert total_workflow_time < 30.0, f"Realistic workflow took {total_workflow_time:.2f}s (too slow)"
        
        # Check update frequency
        if len(progress_updates) > 1:
            update_times = [update[0] for update in progress_updates]
            avg_update_interval = (update_times[-1] - update_times[0]) / (len(update_times) - 1)
            assert avg_update_interval < 5.0, f"Average update interval {avg_update_interval:.2f}s too slow"