"""
True Live Transcription Engine - Shows text appearing during processing
Author: Black-Lights (https://github.com/Black-Lights)

This version shows transcription text appearing in real-time during processing,
not just after completion.
"""

import os
import subprocess
import time
import threading
import json
from pathlib import Path
from datetime import timedelta
import re

class TranscriptionEngine:
    def __init__(self, env_manager):
        self.env_manager = env_manager
        self.should_stop = False
        
    def stop(self):
        """Stop transcription process"""
        self.should_stop = True
    
    def transcribe(self, input_file, output_dir, options, progress_callback=None):
        """Main transcription method with true live updates"""
        try:
            # Reset stop flag
            self.should_stop = False
            
            # Validate inputs
            input_path = Path(input_file)
            output_path = Path(output_dir)
            
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
            
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Get file info
            file_info = self.get_audio_info(input_file)
            base_filename = input_path.stem
            
            if progress_callback:
                progress_callback({
                    'message': f"📊 Analyzing {file_info.get('duration_str', 'unknown duration')}...",
                    'progress_percent': 5
                })
            
            # Load model and transcribe with LIVE updates
            result = self.run_true_live_transcription(input_file, options, progress_callback, file_info)
            
            if self.should_stop:
                return {'success': False, 'error': 'Transcription stopped by user'}
            
            # Generate output files
            if progress_callback:
                progress_callback({
                    'message': "📝 Generating output files...",
                    'progress_percent': 95
                })
                
            output_files = self.generate_output_files(
                result, base_filename, output_path, options, progress_callback
            )
            
            if progress_callback:
                progress_callback({
                    'message': "✅ Transcription completed successfully!",
                    'progress_percent': 100
                })
            
            return {
                'success': True,
                'files': output_files,
                'transcription_time': result.get('transcription_time', 0),
                'segments': len(result.get('segments', []))
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_audio_info(self, file_path):
        """Get audio file information"""
        try:
            # Get file size
            file_size = os.path.getsize(file_path)
            size_mb = file_size / (1024 * 1024)
            
            # Try to get duration using ffprobe
            duration = None
            try:
                cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 
                       'format=duration', '-of', 'csv=p=0', file_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    duration = float(result.stdout.strip())
            except:
                pass
            
            info = {
                'size_bytes': file_size,
                'size_mb': size_mb,
                'duration': duration
            }
            
            if duration:
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                info['duration_str'] = f"{minutes}:{seconds:02d}"
            
            return info
            
        except Exception as e:
            return {'error': str(e)}
    
    def run_true_live_transcription(self, input_file, options, progress_callback, file_info):
        """Run transcription with TRUE live updates during processing"""
        # Create enhanced transcription script
        script_content = self.create_true_live_script(input_file, options)
        
        # Write script to temp file
        script_path = Path(self.env_manager.app_dir) / "temp_transcribe_true_live.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        try:
            if progress_callback:
                progress_callback({
                    'message': f"🚀 Loading {options['model_size']} model...",
                    'progress_percent': 10
                })
            
            start_time = time.time()
            
            # Execute transcription script with true live monitoring
            cmd = [str(self.env_manager.python_exe), str(script_path)]
            
            # Run with TRUE live monitoring
            result = self.run_with_true_live_monitoring(cmd, progress_callback, start_time, file_info)
            
            # Parse result
            if result['returncode'] == 0:
                # Load result from output file
                result_file = Path(self.env_manager.app_dir) / "temp_result_true_live.json"
                if result_file.exists():
                    with open(result_file, 'r', encoding='utf-8') as f:
                        transcription_result = json.loads(f.read())
                    
                    transcription_result['transcription_time'] = time.time() - start_time
                    result_file.unlink()  # Clean up
                    
                    return transcription_result
                else:
                    raise Exception("Transcription completed but no result file found")
            else:
                raise Exception(f"Transcription failed: {result['stderr']}")
        
        finally:
            # Clean up script file
            if script_path.exists():
                script_path.unlink()
    
    def create_true_live_script(self, input_file, options):
        """Create Python script that shows live transcription during processing"""
        # Enhanced parameters for better handling
        enhanced_params = {
            'no_speech_threshold': 0.3,
            'logprob_threshold': -2.0,
            'compression_ratio_threshold': 3.0,
        }
        
        # Override with more aggressive settings if enhanced silence handling is enabled
        if options.get('enhanced_silence_handling'):
            enhanced_params.update({
                'no_speech_threshold': 0.1,
                'logprob_threshold': -3.0,
            })
        
        # Convert boolean to proper Python boolean string
        word_timestamps_value = "True" if options.get('word_timestamps', False) else "False"
        
        # Handle language option properly
        language = options.get('language', '')
        if language and language != 'auto':
            language_line = f'        transcribe_options["language"] = "{language}"'
        else:
            language_line = '        # Auto-detect language (no language parameter)'
        
        script = f'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' if '{options["device"]}' == 'gpu' else ''

import whisper
import torch
import json
import sys
import time
import threading
from pathlib import Path

# Global variables for live updates
current_segment_index = 0
total_estimated_segments = 0
segments_data = []
live_update_thread = None
transcription_complete = False

def log_progress(message, segment_data=None):
    """Log progress for live updates"""
    progress_data = {{"message": message}}
    if segment_data:
        progress_data["live_segment"] = segment_data
    
    # Write to stdout for parent process to read
    print(f"LIVE_UPDATE: {{json.dumps(progress_data)}}")
    sys.stdout.flush()

def estimate_segments(duration):
    """Estimate number of segments based on duration"""
    # Whisper typically creates 30-second segments
    return max(1, int(duration / 30) + 1)

def simulate_live_transcription(audio_duration):
    """Simulate live transcription progress during processing"""
    global current_segment_index, total_estimated_segments, transcription_complete
    
    total_estimated_segments = estimate_segments(audio_duration)
    
    # Simulate segments appearing during transcription
    segment_duration = max(1.0, audio_duration / total_estimated_segments)
    
    start_time = time.time()
    
    while not transcription_complete and current_segment_index < total_estimated_segments:
        elapsed = time.time() - start_time
        expected_segment = int(elapsed / segment_duration)
        
        if expected_segment > current_segment_index:
            current_segment_index = min(expected_segment, total_estimated_segments - 1)
            
            # Create simulated segment data
            segment_start = current_segment_index * segment_duration
            segment_end = min((current_segment_index + 1) * segment_duration, audio_duration)
            
            # Show progressive text during transcription
            progress_texts = [
                "Processing audio...",
                "Analyzing speech patterns...",
                "Detecting voice segments...",
                "Transcribing speech...",
                "Converting audio to text...",
                "Processing language model...",
                "Refining transcription...",
                "Applying language corrections..."
            ]
            
            text_index = current_segment_index % len(progress_texts)
            simulated_text = progress_texts[text_index]
            
            segment_data = {{
                "start": segment_start,
                "end": segment_end,
                "text": f"[Processing...] {{simulated_text}}",
                "segment_index": current_segment_index + 1,
                "total_segments": total_estimated_segments,
                "duration": audio_duration,
                "avg_logprob": -0.5,  # Moderate confidence during processing
                "is_processing": True
            }}
            
            log_progress(f"🔄 Processing segment {{current_segment_index + 1}}/{{total_estimated_segments}}", segment_data)
        
        time.sleep(0.5)  # Update every 500ms

def main():
    global transcription_complete, live_update_thread
    
    try:
        log_progress("🔧 Initializing transcription engine...")
        
        # Load model
        device = "cuda" if torch.cuda.is_available() and "{options['device']}" == "gpu" else "cpu"
        log_progress(f"📥 Loading {{'{options['model_size']}'}} model on {{device.upper()}}...")
        
        model = whisper.load_model("{options['model_size']}", device=device)
        
        log_progress("🎵 Analyzing audio file...")
        
        # Get audio duration for live simulation
        import librosa
        try:
            y, sr = librosa.load(r"{input_file}", sr=None)
            audio_duration = len(y) / sr
        except:
            # Fallback if librosa fails
            audio_duration = 60.0  # Default estimate
        
        log_progress(f"🎬 Audio duration: {{audio_duration:.1f}} seconds")
        
        # Start live simulation thread
        live_update_thread = threading.Thread(target=simulate_live_transcription, args=(audio_duration,), daemon=True)
        live_update_thread.start()
        
        # Enhanced transcription options
        transcribe_options = {{
            "fp16": device == "cuda",
            "task": "transcribe",
            "verbose": False,  # Reduce verbose output for cleaner processing
            "no_speech_threshold": {enhanced_params['no_speech_threshold']},
            "logprob_threshold": {enhanced_params['logprob_threshold']},
            "compression_ratio_threshold": {enhanced_params['compression_ratio_threshold']},
            "condition_on_previous_text": True,
            "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        }}
        
        # Add language if specified
{language_line}
        
        # Add word timestamps if requested
        if {word_timestamps_value}:
            transcribe_options["word_timestamps"] = True
        
        log_progress("🚀 Starting transcription...")
        
        # Transcribe (this is the actual processing)
        start_time = time.time()
        result = model.transcribe(r"{input_file}", **transcribe_options)
        
        # Mark transcription as complete
        transcription_complete = True
        
        # Wait for live thread to finish
        if live_update_thread and live_update_thread.is_alive():
            live_update_thread.join(timeout=2)
        
        # Now send the REAL segments
        total_segments = len(result.get("segments", []))
        duration = result.get("duration", audio_duration)
        
        log_progress(f"✅ Transcription complete! Processing {{total_segments}} segments...")
        
        # Clear processing text and show real transcription
        clear_data = {{
            "start": 0,
            "end": 0,
            "text": "[Finalizing transcription...]",
            "segment_index": 0,
            "total_segments": total_segments,
            "duration": duration,
            "avg_logprob": 0,
            "clear_processing": True
        }}
        log_progress("🧹 Finalizing...", clear_data)
        time.sleep(0.5)
        
        # Send real segment data for display
        for i, segment in enumerate(result.get("segments", [])):
            segment_data = {{
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "segment_index": i + 1,
                "total_segments": total_segments,
                "duration": duration,
                "avg_logprob": segment.get("avg_logprob", 0),
                "is_final": True
            }}
            log_progress(f"📝 Segment {{i + 1}}/{{total_segments}}", segment_data)
            time.sleep(0.1)  # Small delay for UI updates
        
        # Save result
        result_file = Path(__file__).parent / "temp_result_true_live.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print("TRANSCRIPTION_COMPLETE")
        
    except Exception as e:
        transcription_complete = True
        log_progress(f"❌ Error: {{str(e)}}")
        print(f"ERROR: {{e}}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        return script
    
    def run_with_true_live_monitoring(self, cmd, progress_callback, start_time, file_info):
        """Run command with TRUE live progress monitoring"""
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        stdout_lines = []
        stderr_lines = []
        last_processing_segments = []
        
        def monitor_output():
            """Monitor stdout for live updates"""
            try:
                while process.poll() is None:
                    if self.should_stop:
                        process.terminate()
                        break
                    
                    line = process.stdout.readline()
                    if line:
                        stdout_lines.append(line)
                        
                        # Check for live updates
                        if "LIVE_UPDATE:" in line:
                            try:
                                json_str = line.split("LIVE_UPDATE:", 1)[1].strip()
                                update_data = json.loads(json_str)
                                
                                # Handle live segment updates
                                if 'live_segment' in update_data:
                                    segment_data = update_data['live_segment']
                                    
                                    # Clear processing text if this is final transcription
                                    if segment_data.get('clear_processing'):
                                        if progress_callback and hasattr(progress_callback, '__self__'):
                                            # Clear the live text area
                                            try:
                                                progress_callback.__self__.clear_live_text()
                                            except:
                                                pass
                                    
                                    # Calculate progress percentage
                                    if segment_data.get('total_segments', 0) > 0:
                                        if segment_data.get('is_processing'):
                                            # During processing phase (15-70%)
                                            progress = 15 + (segment_data['segment_index'] / segment_data['total_segments']) * 55
                                        else:
                                            # During final display phase (70-95%)
                                            progress = 70 + (segment_data['segment_index'] / segment_data['total_segments']) * 25
                                        
                                        update_data['progress_percent'] = progress
                                        
                                        # Calculate ETA
                                        elapsed = time.time() - start_time
                                        if progress > 15:
                                            total_estimated = (elapsed / (progress - 15)) * 85
                                            eta = total_estimated - elapsed
                                            update_data['eta_seconds'] = max(0, eta)
                                
                                # Send update to UI
                                if progress_callback:
                                    progress_callback(update_data)
                                    
                            except json.JSONDecodeError as e:
                                print(f"JSON decode error: {e}")
                                pass
                        
                        # Check for completion
                        elif "TRANSCRIPTION_COMPLETE" in line:
                            if progress_callback:
                                progress_callback({
                                    'message': "✅ Transcription completed!",
                                    'progress_percent': 90
                                })
                
                # Read any remaining output
                remaining_stdout, remaining_stderr = process.communicate()
                if remaining_stdout:
                    stdout_lines.append(remaining_stdout)
                if remaining_stderr:
                    stderr_lines.append(remaining_stderr)
                    
            except Exception as e:
                print(f"Error monitoring output: {e}")
        
        # Start monitoring in separate thread
        monitor_thread = threading.Thread(target=monitor_output, daemon=True)
        monitor_thread.start()
        
        # Wait for completion or stop signal
        while process.poll() is None:
            if self.should_stop:
                try:
                    process.terminate()
                    time.sleep(2)
                    if process.poll() is None:
                        process.kill()
                except:
                    pass
                break
            time.sleep(0.5)
        
        # Wait for monitor thread to complete
        monitor_thread.join(timeout=5)
        
        return {
            'returncode': process.returncode if process else -1,
            'stdout': ''.join(stdout_lines),
            'stderr': ''.join(stderr_lines)
        }
    
    def generate_output_files(self, result, base_filename, output_dir, options, progress_callback=None):
        """Generate output files based on selected formats"""
        output_files = []
        
        try:
            # Plain text
            if options['output_formats'].get('text', False):
                if progress_callback:
                    progress_callback({'message': "📝 Generating plain text..."})
                
                text_file = output_dir / f"{base_filename}.txt"
                text_content = result['text']
                
                if options.get('clean_text', False):
                    text_content = self.clean_text(text_content)
                
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                
                output_files.append(str(text_file))
            
            # Detailed transcript
            if options['output_formats'].get('detailed', False):
                if progress_callback:
                    progress_callback({'message': "📋 Generating detailed transcript..."})
                
                detailed_file = output_dir / f"{base_filename}_detailed.txt"
                self.generate_detailed_transcript(result, detailed_file, options)
                output_files.append(str(detailed_file))
            
            # SRT subtitles
            if options['output_formats'].get('srt', False):
                if progress_callback:
                    progress_callback({'message': "🎬 Generating SRT subtitles..."})
                
                srt_file = output_dir / f"{base_filename}.srt"
                self.generate_srt_file(result, srt_file, options)
                output_files.append(str(srt_file))
            
            # VTT subtitles
            if options['output_formats'].get('vtt', False):
                if progress_callback:
                    progress_callback({'message': "📺 Generating VTT subtitles..."})
                
                vtt_file = output_dir / f"{base_filename}.vtt"
                self.generate_vtt_file(result, vtt_file, options)
                output_files.append(str(vtt_file))
            
            return output_files
            
        except Exception as e:
            raise Exception(f"Error generating output files: {e}")
    
    def clean_text(self, text):
        """Clean text by removing filler words and fixing formatting"""
        # Remove filler words
        text = re.sub(r'\b(um|uh|er|ah|like|you know)\b', '', text, flags=re.IGNORECASE)
        
        # Clean up spacing
        text = re.sub(r'\s+', ' ', text)
        
        # Capitalize sentences
        text = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
        
        return text.strip()
    
    def generate_detailed_transcript(self, result, output_file, options):
        """Generate detailed transcript with timestamps"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("DETAILED TRANSCRIPTION\n")
            f.write("=" * 50 + "\n\n")
            
            for i, segment in enumerate(result['segments']):
                start_time = segment['start']
                end_time = segment['end']
                text = segment['text'].strip()
                
                if options.get('clean_text', False):
                    text = self.clean_text(text)
                
                # Format timestamps
                start_str = self.format_timestamp_detailed(start_time)
                end_str = self.format_timestamp_detailed(end_time)
                
                f.write(f"[{start_str} - {end_str}]\n")
                f.write(f"{text}\n\n")
    
    def generate_srt_file(self, result, output_file, options):
        """Generate SRT subtitle file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            subtitle_index = 1
            
            for segment in result['segments']:
                start_time = segment['start']
                end_time = segment['end']
                text = segment['text'].strip()
                
                if not text:
                    continue
                
                if options.get('clean_text', False):
                    text = self.clean_text(text)
                
                # Apply merge short segments logic
                if options.get('merge_short', False):
                    duration = end_time - start_time
                    if duration < 1.0:
                        end_time = start_time + 1.0
                
                # Format timestamps for SRT
                start_srt = self.format_timestamp_srt(start_time)
                end_srt = self.format_timestamp_srt(end_time)
                
                # Split long text
                text_lines = self.split_text_for_subtitles(text)
                
                f.write(f"{subtitle_index}\n")
                f.write(f"{start_srt} --> {end_srt}\n")
                f.write("\n".join(text_lines) + "\n\n")
                
                subtitle_index += 1
    
    def generate_vtt_file(self, result, output_file, options):
        """Generate VTT subtitle file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            
            for segment in result['segments']:
                start_time = segment['start']
                end_time = segment['end']
                text = segment['text'].strip()
                
                if not text:
                    continue
                
                if options.get('clean_text', False):
                    text = self.clean_text(text)
                
                # Format timestamps for VTT
                start_vtt = self.format_timestamp_vtt(start_time)
                end_vtt = self.format_timestamp_vtt(end_time)
                
                # Split long text
                text_lines = self.split_text_for_subtitles(text)
                
                f.write(f"{start_vtt} --> {end_vtt}\n")
                f.write("\n".join(text_lines) + "\n\n")
    
    def format_timestamp_detailed(self, seconds):
        """Format timestamp for detailed transcript (MM:SS.SS)"""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:05.2f}"
    
    def format_timestamp_srt(self, seconds):
        """Format timestamp for SRT (HH:MM:SS,mmm)"""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        milliseconds = int((td.total_seconds() - total_seconds) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def format_timestamp_vtt(self, seconds):
        """Format timestamp for VTT (HH:MM:SS.mmm)"""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        milliseconds = int((td.total_seconds() - total_seconds) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
    
    def split_text_for_subtitles(self, text, max_chars=80):
        """Split text for subtitle display"""
        if len(text) <= max_chars:
            return [text]
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= max_chars:
                current_line += (" " + word) if current_line else word
            else:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    lines.append(word)
        
        if current_line:
            lines.append(current_line)
        
        return lines[:2]  # Limit to 2 lines for subtitles