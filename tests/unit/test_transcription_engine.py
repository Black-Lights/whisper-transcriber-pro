"""
Unit tests for TranscriptionEngine
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import json
import subprocess
import threading
import time

from src.transcription_engine import TranscriptionEngine


class TestTranscriptionEngine:
    """Test cases for TranscriptionEngine"""
    
    def test_initialization(self, mock_env_manager):
        """Test TranscriptionEngine initialization"""
        engine = TranscriptionEngine(mock_env_manager)
        
        assert engine.env_manager == mock_env_manager
        assert engine.should_stop is False
        assert engine.is_paused is False
        assert engine.current_process is None
        assert engine.progress_callback is None
        assert engine.live_callback is None
    
    def test_stop_functionality(self, mock_env_manager):
        """Test stop functionality"""
        engine = TranscriptionEngine(mock_env_manager)
        mock_process = Mock()
        engine.current_process = mock_process
        
        engine.stop()
        
        assert engine.should_stop is True
        mock_process.terminate.assert_called_once()
    
    def test_pause_resume_functionality(self, mock_env_manager):
        """Test pause and resume functionality"""
        engine = TranscriptionEngine(mock_env_manager)
        
        # Test pause
        engine.pause()
        assert engine.is_paused is True
        
        # Test resume
        engine.resume()
        assert engine.is_paused is False
    
    def test_get_audio_info_success(self, mock_env_manager, sample_audio_file):
        """Test successful audio file information extraction"""
        engine = TranscriptionEngine(mock_env_manager)
        
        # Mock ffprobe response
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="10.5")
            
            info = engine.get_audio_info(str(sample_audio_file))
        
        assert 'size_bytes' in info
        assert 'size_mb' in info
        assert 'duration' in info
        assert info['duration'] == 10.5
        assert 'duration_str' in info
        assert info['duration_str'] == "0:10"
    
    def test_get_audio_info_no_duration(self, mock_env_manager, sample_audio_file):
        """Test audio info extraction when duration detection fails"""
        engine = TranscriptionEngine(mock_env_manager)
        
        # Mock ffprobe failure
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.SubprocessError()
            
            info = engine.get_audio_info(str(sample_audio_file))
        
        assert 'size_bytes' in info
        assert 'size_mb' in info
        assert 'duration' not in info or info['duration'] is None
    
    def test_get_audio_info_file_not_found(self, mock_env_manager):
        """Test audio info extraction for non-existent file"""
        engine = TranscriptionEngine(mock_env_manager)
        
        info = engine.get_audio_info("/nonexistent/file.mp3")
        
        assert 'error' in info
    
    def test_is_dots_only_output_true_cases(self, mock_env_manager):
        """Test detection of dots-only output (silence)"""
        engine = TranscriptionEngine(mock_env_manager)
        
        # Test various dots-only patterns
        test_cases = [
            {'text': '... ... ... ...'},
            {'text': '. . . . . . .'},
            {'text': '...'},
            {'text': ''},
            {'text': '   '},
            {'text': '. . . some text . . . . . . . . .'}  # >80% dots
        ]
        
        for case in test_cases:
            assert engine.is_dots_only_output(case) is True, f"Failed for: {case}"
    
    def test_is_dots_only_output_false_cases(self, mock_env_manager):
        """Test detection of normal transcription output"""
        engine = TranscriptionEngine(mock_env_manager)
        
        # Test normal transcription patterns
        test_cases = [
            {'text': 'This is a normal transcription.'},
            {'text': 'Hello world. How are you?'},
            {'text': 'Speech with some... pauses.'},
            {'text': 'The meeting started at 3 p.m.'}
        ]
        
        for case in test_cases:
            assert engine.is_dots_only_output(case) is False, f"Failed for: {case}"
    
    def test_is_dots_only_output_edge_cases(self, mock_env_manager):
        """Test edge cases for dots detection"""
        engine = TranscriptionEngine(mock_env_manager)
        
        # Test edge cases
        assert engine.is_dots_only_output({}) is True  # No text key
        assert engine.is_dots_only_output({'text': None}) is True  # None text
        
        # Test exception handling
        with patch.object(engine, 'is_dots_only_output', side_effect=Exception("Test error")):
            try:
                engine.is_dots_only_output({'text': 'test'})
            except:
                pytest.fail("Should handle exceptions gracefully")
    
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
        result_file_content = json.dumps(mock_whisper_result)
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=result_file_content)), \
             patch('pathlib.Path.unlink'), \
             patch.object(engine, 'generate_output_files', return_value=['test.txt']):
            
            options = {
                'model_size': 'medium',
                'device': 'cpu',
                'output_formats': {'text': True},
                'enhanced_silence_handling': True
            }
            
            result = engine.transcribe(str(sample_audio_file), str(temp_dir), options)
        
        assert result['success'] is True
        assert 'files' in result
        assert result['files'] == ['test.txt']
    
    @patch('subprocess.Popen')
    def test_transcribe_file_not_found(self, mock_popen, mock_env_manager, temp_dir):
        """Test transcription with non-existent file"""
        engine = TranscriptionEngine(mock_env_manager)
        
        options = {'model_size': 'medium', 'device': 'cpu', 'output_formats': {'text': True}}
        
        result = engine.transcribe("/nonexistent/file.mp3", str(temp_dir), options)
        
        assert result['success'] is False
        assert 'error' in result
        assert "not found" in result['error']
    
    @patch('subprocess.Popen')
    def test_transcribe_user_stopped(self, mock_popen, mock_env_manager, sample_audio_file, temp_dir):
        """Test transcription when user stops the process"""
        engine = TranscriptionEngine(mock_env_manager)
        engine.should_stop = True  # Simulate user stop
        
        mock_process = Mock()
        mock_process.poll.return_value = 0
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        options = {'model_size': 'medium', 'device': 'cpu', 'output_formats': {'text': True}}
        
        result = engine.transcribe(str(sample_audio_file), str(temp_dir), options)
        
        assert result['success'] is False
        assert 'stopped by user' in result['error']
    
    @patch('subprocess.Popen')
    def test_transcribe_dots_only_result(self, mock_popen, mock_env_manager, sample_audio_file, temp_dir):
        """Test transcription that returns only dots (silence)"""
        engine = TranscriptionEngine(mock_env_manager)
        
        mock_process = Mock()
        mock_process.poll.return_value = 0
        mock_process.returncode = 0
        mock_process.stdout.readline.return_value = ""
        mock_process.communicate.return_value = ("", "")
        mock_popen.return_value = mock_process
        
        # Mock dots-only result
        dots_result = {'text': '... ... ... ...', 'segments': []}
        result_content = json.dumps(dots_result)
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=result_content)), \
             patch('pathlib.Path.unlink'):
            
            options = {'model_size': 'medium', 'device': 'cpu', 'output_formats': {'text': True}}
            
            result = engine.transcribe(str(sample_audio_file), str(temp_dir), options)
        
        assert result['success'] is False
        assert 'silence' in result['error'] or 'unclear' in result['error']
    
    def test_create_enhanced_transcription_script(self, mock_env_manager, sample_audio_file):
        """Test enhanced transcription script creation"""
        engine = TranscriptionEngine(mock_env_manager)
        
        options = {
            'model_size': 'medium',
            'device': 'gpu',
            'language': 'en',
            'word_timestamps': True,
            'enhanced_silence_handling': True
        }
        
        script = engine.create_enhanced_transcription_script(str(sample_audio_file), options)
        
        # Check script content
        assert 'import whisper' in script
        assert 'import torch' in script
        assert 'medium' in script
        assert 'gpu' in script
        assert 'word_timestamps' in script
        assert 'enhanced_silence_handling' in script
        assert str(sample_audio_file) in script
    
    def test_create_enhanced_transcription_script_auto_language(self, mock_env_manager, sample_audio_file):
        """Test script creation with auto language detection"""
        engine = TranscriptionEngine(mock_env_manager)
        
        options = {
            'model_size': 'small',
            'device': 'cpu',
            'language': 'auto',
            'word_timestamps': False
        }
        
        script = engine.create_enhanced_transcription_script(str(sample_audio_file), options)
        
        # Should not set language parameter for auto-detection
        assert 'Auto-detect language' in script
        assert 'language' not in script or 'transcribe_options["language"]' not in script
    
    def test_run_with_live_monitoring_success(self, mock_env_manager):
        """Test live monitoring of transcription process"""
        engine = TranscriptionEngine(mock_env_manager)
        
        # Mock process
        mock_process = Mock()
        mock_process.poll.side_effect = [None, None, 0]  # Running, then finished
        mock_process.returncode = 0
        mock_process.stdout.readline.side_effect = [
            'LIVE_UPDATE: {"message": "Loading model"}\n',
            'LIVE_UPDATE: {"message": "Processing", "live_segment": {"segment_index": 1, "total_segments": 2}}\n',
            'TRANSCRIPTION_COMPLETE\n',
            ''
        ]
        mock_process.communicate.return_value = ('', '')
        
        progress_updates = []
        def mock_progress_callback(update):
            progress_updates.append(update)
        
        with patch('subprocess.Popen', return_value=mock_process):
            result = engine.run_with_live_monitoring(
                ['python', 'script.py'], 
                mock_progress_callback, 
                time.time(), 
                {}
            )
        
        assert result['returncode'] == 0
        assert len(progress_updates) >= 2  # Should have received updates
    
    def test_run_with_live_monitoring_with_stop(self, mock_env_manager):
        """Test live monitoring when user stops transcription"""
        engine = TranscriptionEngine(mock_env_manager)
        engine.should_stop = True
        
        mock_process = Mock()
        mock_process.poll.return_value = None  # Never finishes naturally
        mock_process.stdout.readline.return_value = ''
        mock_process.communicate.return_value = ('', '')
        
        with patch('subprocess.Popen', return_value=mock_process):
            result = engine.run_with_live_monitoring(['python', 'script.py'], None, time.time(), {})
        
        mock_process.terminate.assert_called()
    
    def test_generate_output_files_all_formats(self, mock_env_manager, temp_dir, mock_whisper_result):
        """Test generation of all output formats"""
        engine = TranscriptionEngine(mock_env_manager)
        
        options = {
            'output_formats': {
                'text': True,
                'detailed': True,
                'srt': True,
                'vtt': True
            },
            'clean_text': True
        }
        
        with patch('builtins.open', mock_open()) as mock_file:
            files = engine.generate_output_files(
                mock_whisper_result, 
                "test_audio", 
                temp_dir, 
                options
            )
        
        assert len(files) == 4  # All formats enabled
        assert any('test_audio.txt' in f for f in files)
        assert any('test_audio_detailed.txt' in f for f in files)
        assert any('test_audio.srt' in f for f in files)
        assert any('test_audio.vtt' in f for f in files)
    
    def test_generate_output_files_selective_formats(self, mock_env_manager, temp_dir, mock_whisper_result):
        """Test generation of selective output formats"""
        engine = TranscriptionEngine(mock_env_manager)
        
        options = {
            'output_formats': {
                'text': True,
                'detailed': False,
                'srt': True,
                'vtt': False
            }
        }
        
        with patch('builtins.open', mock_open()):
            files = engine.generate_output_files(
                mock_whisper_result, 
                "test_audio", 
                temp_dir, 
                options
            )
        
        assert len(files) == 2  # Only text and srt enabled
        assert any('test_audio.txt' in f for f in files)
        assert any('test_audio.srt' in f for f in files)
    
    def test_clean_text_functionality(self, mock_env_manager):
        """Test text cleaning functionality"""
        engine = TranscriptionEngine(mock_env_manager)
        
        dirty_text = "Um, this is like, you know, a test, uh, transcription."
        clean_text = engine.clean_text(dirty_text)
        
        # Check filler words are removed
        assert "um" not in clean_text.lower()
        assert "like" not in clean_text.lower()
        assert "you know" not in clean_text.lower()
        assert "uh" not in clean_text.lower()
        
        # Check meaningful words remain
        assert "test" in clean_text
        assert "transcription" in clean_text
        
        # Check capitalization
        assert clean_text[0].isupper()
    
    def test_clean_text_edge_cases(self, mock_env_manager):
        """Test text cleaning with edge cases"""
        engine = TranscriptionEngine(mock_env_manager)
        
        # Empty text
        assert engine.clean_text("") == ""
        
        # Only filler words
        result = engine.clean_text("Um, uh, like, you know")
        assert len(result.strip()) == 0 or result.strip() == ","
        
        # Multiple spaces
        result = engine.clean_text("This   has    multiple    spaces")
        assert "  " not in result
    
    def test_generate_detailed_transcript(self, mock_env_manager, temp_dir, mock_whisper_result):
        """Test detailed transcript generation"""
        engine = TranscriptionEngine(mock_env_manager)
        
        output_file = temp_dir / "detailed.txt"
        options = {'clean_text': False}
        
        with patch('builtins.open', mock_open()) as mock_file:
            engine.generate_detailed_transcript(mock_whisper_result, output_file, options)
        
        # Verify file was opened for writing
        mock_file.assert_called_once_with(output_file, 'w', encoding='utf-8')
        
        # Verify content was written
        written_content = ''.join(call.args[0] for call in mock_file().write.call_args_list)
        assert "DETAILED TRANSCRIPTION" in written_content
        assert "This is a test" in written_content
        assert "transcription." in written_content
    
    def test_generate_srt_file(self, mock_env_manager, temp_dir, mock_whisper_result):
        """Test SRT subtitle file generation"""
        engine = TranscriptionEngine(mock_env_manager)
        
        output_file = temp_dir / "subtitles.srt"
        options = {'clean_text': False, 'merge_short': False}
        
        with patch('builtins.open', mock_open()) as mock_file:
            engine.generate_srt_file(mock_whisper_result, output_file, options)
        
        mock_file.assert_called_once_with(output_file, 'w', encoding='utf-8')
        
        # Verify SRT format content
        written_content = ''.join(call.args[0] for call in mock_file().write.call_args_list)
        assert "1\n" in written_content  # Subtitle index
        assert "-->" in written_content  # Time separator
        assert "00:00:00,000" in written_content  # Time format
    
    def test_generate_vtt_file(self, mock_env_manager, temp_dir, mock_whisper_result):
        """Test VTT subtitle file generation"""
        engine = TranscriptionEngine(mock_env_manager)
        
        output_file = temp_dir / "subtitles.vtt"
        options = {'clean_text': False}
        
        with patch('builtins.open', mock_open()) as mock_file:
            engine.generate_vtt_file(mock_whisper_result, output_file, options)
        
        mock_file.assert_called_once_with(output_file, 'w', encoding='utf-8')
        
        # Verify VTT format content
        written_content = ''.join(call.args[0] for call in mock_file().write.call_args_list)
        assert "WEBVTT\n\n" in written_content
        assert "-->" in written_content
        assert "00:00:00.000" in written_content  # VTT time format
    
    def test_format_timestamp_detailed(self, mock_env_manager):
        """Test detailed timestamp formatting"""
        engine = TranscriptionEngine(mock_env_manager)
        
        # Test various times
        assert engine.format_timestamp_detailed(65.5) == "01:05.50"
        assert engine.format_timestamp_detailed(3665.25) == "61:05.25"
        assert engine.format_timestamp_detailed(5.1) == "00:05.10"
    
    def test_format_timestamp_srt(self, mock_env_manager):
        """Test SRT timestamp formatting"""
        engine = TranscriptionEngine(mock_env_manager)
        
        # Test SRT format (HH:MM:SS,mmm)
        result = engine.format_timestamp_srt(3665.5)
        assert result == "01:01:05,500"
        
        result = engine.format_timestamp_srt(65.25)
        assert result == "00:01:05,250"
    
    def test_format_timestamp_vtt(self, mock_env_manager):
        """Test VTT timestamp formatting"""
        engine = TranscriptionEngine(mock_env_manager)
        
        # Test VTT format (HH:MM:SS.mmm)
        result = engine.format_timestamp_vtt(3665.5)
        assert result == "01:01:05.500"
        
        result = engine.format_timestamp_vtt(65.25)
        assert result == "00:01:05.250"
    
    def test_split_text_for_subtitles(self, mock_env_manager):
        """Test text splitting for subtitle display"""
        engine = TranscriptionEngine(mock_env_manager)
        
        # Test short text (no splitting needed)
        short_text = "This is short"
        result = engine.split_text_for_subtitles(short_text)
        assert result == [short_text]
        
        # Test long text (splitting needed)
        long_text = "This is a very long line of text that should be split into multiple lines for subtitle display"
        result = engine.split_text_for_subtitles(long_text, max_chars=40)
        assert len(result) <= 2  # Limited to 2 lines
        assert all(len(line) <= 40 for line in result)
        
        # Test edge case - single very long word
        very_long_word = "supercalifragilisticexpialidocious" * 3
        result = engine.split_text_for_subtitles(very_long_word, max_chars=40)
        assert len(result) <= 2
    
    def test_update_transcription_progress_with_live_callback(self, mock_env_manager):
        """Test progress updates with live callback"""
        engine = TranscriptionEngine(mock_env_manager)
        
        live_updates = []
        def mock_live_callback(segment_data):
            live_updates.append(segment_data)
        
        engine.live_callback = mock_live_callback
        
        # Mock progress update with live segment data
        progress_info = {
            'message': 'Processing segment',
            'live_segment': {
                'start': 0.0,
                'end': 2.0,
                'text': 'Test segment',
                'segment_index': 1,
                'total_segments': 5
            }
        }
        
        engine.update_transcription_progress(progress_info)
        
        # Live callback should have been called
        assert len(live_updates) == 1
        assert live_updates[0]['text'] == 'Test segment'
    
    def test_error_handling_in_transcription(self, mock_env_manager, sample_audio_file, temp_dir):
        """Test error handling during transcription"""
        engine = TranscriptionEngine(mock_env_manager)
        
        # Mock process that raises an exception
        with patch('subprocess.Popen', side_effect=Exception("Process error")):
            options = {'model_size': 'medium', 'device': 'cpu', 'output_formats': {'text': True}}
            
            result = engine.transcribe(str(sample_audio_file), str(temp_dir), options)
        
        assert result['success'] is False
        assert 'error' in result
        assert 'Process error' in result['error']


class TestTranscriptionEngineIntegration:
    """Integration tests for TranscriptionEngine"""
    
    def test_full_transcription_workflow_mock(self, mock_env_manager, sample_audio_file, temp_dir):
        """Test complete transcription workflow with mocked dependencies"""
        engine = TranscriptionEngine(mock_env_manager)
        
        # Mock successful transcription result
        mock_result = {
            'text': 'This is a complete test transcription.',
            'segments': [
                {'start': 0.0, 'end': 3.0, 'text': 'This is a complete', 'avg_logprob': -0.2},
                {'start': 3.0, 'end': 6.0, 'text': ' test transcription.', 'avg_logprob': -0.3}
            ],
            'duration': 6.0
        }
        
        # Mock all external dependencies
        with patch('subprocess.Popen') as mock_popen, \
             patch('builtins.open', mock_open(read_data=json.dumps(mock_result))), \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.unlink'), \
             patch('pathlib.Path.mkdir'):
            
            # Setup mock process
            mock_process = Mock()
            mock_process.poll.return_value = 0
            mock_process.returncode = 0
            mock_process.stdout.readline.return_value = ""
            mock_process.communicate.return_value = ("", "")
            mock_popen.return_value = mock_process
            
            options = {
                'model_size': 'medium',
                'device': 'cpu',
                'output_formats': {'text': True, 'srt': True},
                'clean_text': True,
                'enhanced_silence_handling': True
            }
            
            result = engine.transcribe(str(sample_audio_file), str(temp_dir), options)
        
        assert result['success'] is True
        assert len(result['files']) == 2  # text and srt files
        assert result['segments'] == 2
    
    def test_live_monitoring_threading(self, mock_env_manager):
        """Test that live monitoring works correctly with threading"""
        engine = TranscriptionEngine(mock_env_manager)
        
        events = []
        
        def mock_progress_callback(update):
            events.append(('progress', update))
        
        # Mock process that provides live updates
        mock_process = Mock()
        mock_process.poll.side_effect = [None, None, 0]
        mock_process.returncode = 0
        
        # Simulate live updates in stdout
        live_updates = [
            'LIVE_UPDATE: {"message": "Starting"}\n',
            'LIVE_UPDATE: {"message": "Processing", "live_segment": {"text": "Hello"}}\n',
            'TRANSCRIPTION_COMPLETE\n'
        ]
        mock_process.stdout.readline.side_effect = live_updates + ['']
        mock_process.communicate.return_value = ('', '')
        
        with patch('subprocess.Popen', return_value=mock_process):
            result = engine.run_with_live_monitoring(
                ['python', 'test.py'],
                mock_progress_callback,
                time.time(),
                {}
            )
        
        assert result['returncode'] == 0
        assert len(events) >= 2  # Should have received multiple updates
    
    def test_script_generation_with_various_options(self, mock_env_manager, sample_audio_file):
        """Test script generation with different option combinations"""
        engine = TranscriptionEngine(mock_env_manager)
        
        test_cases = [
            {
                'model_size': 'tiny',
                'device': 'cpu',
                'language': 'en',
                'word_timestamps': False,
                'enhanced_silence_handling': False
            },
            {
                'model_size': 'large',
                'device': 'gpu',
                'language': 'auto',
                'word_timestamps': True,
                'enhanced_silence_handling': True
            }
        ]
        
        for options in test_cases:
            script = engine.create_enhanced_transcription_script(str(sample_audio_file), options)
            
            # Verify all options are reflected in script
            assert options['model_size'] in script
            assert options['device'] in script
            
            if options['word_timestamps']:
                assert 'word_timestamps' in script
            
            if options['enhanced_silence_handling']:
                assert 'enhanced_silence_handling' in script
            
            # Verify script is valid Python (basic check)
            assert 'import whisper' in script
            assert 'def main()' in script
            assert 'if __name__ == "__main__"' in script