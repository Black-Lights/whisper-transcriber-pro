"""
Transcription Engine - Core transcription functionality
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
        """Main transcription method"""
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
                    'time_info': f"File: {file_info.get('size_mb', 0):.1f} MB"
                })
            
            # Load model and transcribe
            result = self.run_transcription(input_file, options, progress_callback)
            
            if self.should_stop:
                return {'success': False, 'error': 'Transcription stopped by user'}
            
            # Generate output files
            output_files = self.generate_output_files(
                result, base_filename, output_path, options, progress_callback
            )
            
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
    
    def run_transcription(self, input_file, options, progress_callback=None):
        """Run the actual transcription"""
        # Create transcription script
        script_content = self.create_transcription_script(input_file, options)
        
        # Write script to temp file
        script_path = Path(self.env_manager.app_dir) / "temp_transcribe.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        try:
            # Run transcription in virtual environment
            if progress_callback:
                progress_callback({
                    'message': f"🚀 Loading {options['model_size']} model...",
                    'time_info': f"Device: {options['device'].upper()}"
                })
            
            start_time = time.time()
            
            # Execute transcription script
            cmd = [str(self.env_manager.python_exe), str(script_path)]
            
            # Run with progress monitoring
            result = self.run_with_progress_monitoring(cmd, progress_callback, start_time)
            
            # Parse result
            if result['returncode'] == 0:
                # Load result from output file
                result_file = Path(self.env_manager.app_dir) / "temp_result.json"
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
    
    def create_transcription_script(self, input_file, options):
        """Create Python script for transcription"""
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
from pathlib import Path

def main():
    try:
        # Load model
        device = "cuda" if torch.cuda.is_available() and "{options['device']}" == "gpu" else "cpu"
        model = whisper.load_model("{options['model_size']}", device=device)
        
        # Transcription options
        transcribe_options = {{
            "fp16": device == "cuda",
            "task": "transcribe",
            "verbose": True
        }}
        
        # Add language if specified (not auto)
{language_line}
        
        # Add word timestamps if requested
        if {word_timestamps_value}:
            transcribe_options["word_timestamps"] = True
        
        # Transcribe
        result = model.transcribe(r"{input_file}", **transcribe_options)
        
        # Save result
        result_file = Path(__file__).parent / "temp_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print("TRANSCRIPTION_COMPLETE")
        
    except Exception as e:
        print(f"ERROR: {{e}}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        return script
    
    def run_with_progress_monitoring(self, cmd, progress_callback, start_time):
        """Run command with progress monitoring"""
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
        
        def monitor_progress():
            while process.poll() is None:
                if self.should_stop:
                    process.terminate()
                    break
                
                elapsed = time.time() - start_time
                
                if progress_callback:
                    progress_callback({
                        'message': f"📼 Transcribing audio... {elapsed:.0f}s elapsed",
                        'time_info': "Processing segments..."
                    })
                
                time.sleep(2)
        
        # Start progress monitoring
        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()
        
        # Wait for completion
        stdout, stderr = process.communicate()
        
        return {
            'returncode': process.returncode,
            'stdout': stdout,
            'stderr': stderr
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