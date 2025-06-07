"""
Enhanced Transcription Engine - Core transcription functionality with live updates
Author: Black-Lights (https://github.com/Black-Lights)

Features:
- Live transcription updates with real-time progress
- Better silence and poor audio quality handling
- Proper process management and cleanup
- Enhanced error detection and recovery
- Support for pause/resume functionality
"""

import json
import os
import re
import signal
import subprocess
import threading
import time
from datetime import timedelta
from pathlib import Path

import psutil


class TranscriptionEngine:
    def __init__(self, env_manager):
        self.env_manager = env_manager
        self.should_stop = False
        self.is_paused = False
        self.current_process = None
        self.progress_callback = None
        self.live_callback = None

    def stop(self):
        """Stop transcription process"""
        self.should_stop = True
        if self.current_process:
            try:
                # Terminate the process gracefully
                self.current_process.terminate()
                # Wait a bit for graceful termination
                try:
                    self.current_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful termination failed
                    self.current_process.kill()
            except:
                pass

    def pause(self):
        """Pause transcription (if supported)"""
        self.is_paused = True
        # Note: Actual pause implementation would depend on the transcription method

    def resume(self):
        """Resume transcription (if supported)"""
        self.is_paused = False

    def transcribe(self, input_file, output_dir, options, progress_callback=None):
        """Main transcription method with live updates"""
        try:
            # Reset stop flag
            self.should_stop = False
            self.progress_callback = progress_callback
            self.live_callback = options.get("live_callback")

            # Validate inputs
            input_path = Path(input_file)
            output_path = Path(output_dir)

            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")

            output_path.mkdir(parents=True, exist_ok=True)

            # Get file info for progress calculation
            file_info = self.get_audio_info(input_file)

            if progress_callback:
                progress_callback(
                    {
                        "message": f"📊 Analyzing {file_info.get('duration_str', 'unknown duration')}...",
                        "progress_percent": 0,
                    }
                )

            # Enhanced transcription with live updates
            result = self.run_enhanced_transcription(
                input_file, options, progress_callback
            )

            if self.should_stop:
                return {"success": False, "error": "Transcription stopped by user"}

            # Check for dots-only output (silence/poor audio issue)
            if self.is_dots_only_output(result):
                return {
                    "success": False,
                    "error": "Audio contains only silence or is too unclear to transcribe. Try using a larger model or check audio quality.",
                }

            # Generate output files
            base_filename = input_path.stem
            output_files = self.generate_output_files(
                result, base_filename, output_path, options, progress_callback
            )

            return {
                "success": True,
                "files": output_files,
                "transcription_time": result.get("transcription_time", 0),
                "segments": len(result.get("segments", [])),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def is_dots_only_output(self, result):
        """Check if transcription result contains only dots (silence indicator)"""
        try:
            text = result.get("text", "").strip()
            if not text:
                return True

            # Count dots vs other characters
            dot_count = text.count(".")
            total_chars = len(text.replace(" ", "").replace("\n", ""))

            if total_chars == 0:
                return True

            # If more than 80% dots, consider it a silence/poor audio issue
            dot_ratio = dot_count / total_chars
            return dot_ratio > 0.8

        except:
            return False

    def get_audio_info(self, file_path):
        """Get audio file information with duration"""
        try:
            # Get file size
            file_size = os.path.getsize(file_path)
            size_mb = file_size / (1024 * 1024)

            # Get duration using ffprobe
            duration = None
            try:
                cmd = [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    file_path,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    duration = float(result.stdout.strip())
            except:
                pass

            info = {"size_bytes": file_size, "size_mb": size_mb, "duration": duration}

            if duration:
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                info["duration_str"] = f"{minutes}:{seconds:02d}"

            return info

        except Exception as e:
            return {"error": str(e)}

    def run_enhanced_transcription(self, input_file, options, progress_callback=None):
        """Run transcription with enhanced settings and live updates"""
        # Create enhanced transcription script
        script_content = self.create_enhanced_transcription_script(input_file, options)

        # Write script to temp file
        script_path = Path(self.env_manager.app_dir) / "temp_transcribe_enhanced.py"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        try:
            if progress_callback:
                progress_callback(
                    {
                        "message": f"🚀 Loading {options['model_size']} model...",
                        "progress_percent": 5,
                    }
                )

            start_time = time.time()

            # Execute transcription script with live monitoring
            cmd = [str(self.env_manager.python_exe), str(script_path)]

            # Run with enhanced progress monitoring
            result = self.run_with_live_monitoring(
                cmd, progress_callback, start_time, options
            )

            # Parse result
            if result["returncode"] == 0:
                # Load result from output file
                result_file = (
                    Path(self.env_manager.app_dir) / "temp_result_enhanced.json"
                )
                if result_file.exists():
                    with open(result_file, "r", encoding="utf-8") as f:
                        transcription_result = json.loads(f.read())

                    transcription_result["transcription_time"] = (
                        time.time() - start_time
                    )
                    result_file.unlink()  # Clean up

                    return transcription_result
                else:
                    raise Exception("Transcription completed but no result file found")
            else:
                error_output = result.get("stderr", "Unknown error")
                # Check for specific error patterns
                if "charmap" in error_output or "unicode" in error_output.lower():
                    raise Exception(
                        "Unicode encoding error - audio may contain foreign language content"
                    )
                else:
                    raise Exception(f"Transcription failed: {error_output}")

        finally:
            # Clean up script file
            if script_path.exists():
                script_path.unlink()

    def create_enhanced_transcription_script(self, input_file, options):
        """Create enhanced Python script for transcription with live updates"""
        # Enhanced parameters for better silence and poor audio handling
        enhanced_params = {
            "no_speech_threshold": 0.3,  # More aggressive - detect speech in poor audio
            "logprob_threshold": -2.0,  # Accept much lower confidence
            "compression_ratio_threshold": 3.0,  # Allow more repetitive content
            "condition_on_previous_text": True,
            "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Multiple attempts
            "beam_size": 5,  # Better search
            "best_of": 5,  # Try multiple candidates
            "patience": 2.0,  # Wait longer for speech
            "length_penalty": 1.0,  # Don't penalize longer sequences
        }

        # Override with more aggressive settings if enhanced silence handling is enabled
        if options.get("enhanced_silence_handling"):
            enhanced_params.update(
                {
                    "no_speech_threshold": 0.1,  # Very aggressive
                    "logprob_threshold": -3.0,  # Accept very low confidence
                    "initial_prompt": "This is a lecture or presentation recording that may contain periods of silence, background noise, or unclear audio. Please transcribe all audible speech.",
                }
            )

        # Handle language option
        language = options.get("language", "")
        if language and language != "auto":
            language_line = f'        transcribe_options["language"] = "{language}"'
        else:
            language_line = "        # Auto-detect language"

        # Word timestamps
        word_timestamps_value = (
            "True" if options.get("word_timestamps", False) else "False"
        )

        script = f'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' if '{options["device"]}' == 'gpu' else ''

import whisper
import torch
import json
import sys
import time
from pathlib import Path

def log_progress(message, segment_data=None):
    """Log progress for live updates"""
    progress_data = {{"message": message}}
    if segment_data:
        progress_data["live_segment"] = segment_data

    # Write to stdout for parent process to read
    print(f"LIVE_UPDATE: {{json.dumps(progress_data)}}")
    sys.stdout.flush()

def main():
    try:
        log_progress("🔧 Initializing transcription engine...")

        # Load model
        device = "cuda" if torch.cuda.is_available() and "{options['device']}" == "gpu" else "cpu"
        log_progress(f"📥 Loading {{'{options['model_size']}'}} model on {{device.upper()}}...")

        model = whisper.load_model("{options['model_size']}", device=device)

        log_progress("🎵 Processing audio file...")

        # Enhanced transcription options
        transcribe_options = {{
            "fp16": device == "cuda",
            "task": "transcribe",
            "verbose": True,
            "no_speech_threshold": {enhanced_params['no_speech_threshold']},
            "logprob_threshold": {enhanced_params['logprob_threshold']},
            "compression_ratio_threshold": {enhanced_params['compression_ratio_threshold']},
            "condition_on_previous_text": {enhanced_params['condition_on_previous_text']},
            "temperature": {enhanced_params['temperature']},
            "beam_size": {enhanced_params['beam_size']},
            "best_of": {enhanced_params['best_of']},
            "patience": {enhanced_params['patience']},
            "length_penalty": {enhanced_params['length_penalty']}
        }}

        # Add language if specified
{language_line}

        # Add word timestamps if requested
        if {word_timestamps_value}:
            transcribe_options["word_timestamps"] = True

        # Add initial prompt for enhanced handling
        if "{options.get('enhanced_silence_handling', False)}":
            transcribe_options["initial_prompt"] = "{enhanced_params.get('initial_prompt', '')}"

        log_progress("🚀 Starting transcription...")

        # Transcribe with progress tracking
        start_time = time.time()
        result = model.transcribe(r"{input_file}", **transcribe_options)

        # Process segments for live updates
        total_segments = len(result.get("segments", []))
        duration = result.get("duration", 0)

        log_progress(f"✅ Transcription complete! Processed {{total_segments}} segments in {{time.time() - start_time:.1f}}s")

        # Send final segment data for live display
        for i, segment in enumerate(result.get("segments", [])):
            segment_data = {{
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "segment_index": i + 1,
                "total_segments": total_segments,
                "duration": duration,
                "avg_logprob": segment.get("avg_logprob", 0)
            }}
            log_progress(f"📝 Segment {{i + 1}}/{{total_segments}}", segment_data)
            time.sleep(0.1)  # Small delay for UI updates

        # Save result with UTF-8 encoding
        result_file = Path(__file__).parent / "temp_result_enhanced.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print("TRANSCRIPTION_COMPLETE")

    except Exception as e:
        log_progress(f"❌ Error: {{str(e)}}")
        print(f"ERROR: {{e}}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        return script

    def run_with_live_monitoring(self, cmd, progress_callback, start_time, options):
        """Run command with live progress monitoring and updates"""
        self.current_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        stdout_lines = []
        stderr_lines = []

        def monitor_output():
            """Monitor stdout for live updates"""
            try:
                while self.current_process.poll() is None:
                    if self.should_stop:
                        break

                    line = self.current_process.stdout.readline()
                    if line:
                        stdout_lines.append(line)

                        # Check for live updates
                        if "LIVE_UPDATE:" in line:
                            try:
                                json_str = line.split("LIVE_UPDATE:", 1)[1].strip()
                                update_data = json.loads(json_str)

                                # Calculate progress percentage
                                if "live_segment" in update_data:
                                    segment_data = update_data["live_segment"]
                                    if segment_data.get("total_segments", 0) > 0:
                                        progress = (
                                            segment_data["segment_index"]
                                            / segment_data["total_segments"]
                                        ) * 100
                                        update_data["progress_percent"] = progress

                                        # Calculate ETA
                                        elapsed = time.time() - start_time
                                        if progress > 0:
                                            eta = (elapsed / progress * 100) - elapsed
                                            update_data["eta_seconds"] = max(0, eta)

                                # Send update to UI
                                if progress_callback:
                                    progress_callback(update_data)

                            except json.JSONDecodeError:
                                pass

                        # Check for completion
                        if "TRANSCRIPTION_COMPLETE" in line:
                            if progress_callback:
                                progress_callback(
                                    {
                                        "message": "✅ Transcription completed successfully!",
                                        "progress_percent": 100,
                                    }
                                )

                # Read any remaining output
                remaining_stdout, remaining_stderr = self.current_process.communicate()
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
        while self.current_process.poll() is None:
            if self.should_stop:
                try:
                    self.current_process.terminate()
                    time.sleep(2)
                    if self.current_process.poll() is None:
                        self.current_process.kill()
                except:
                    pass
                break
            time.sleep(0.5)

        # Wait for monitor thread to complete
        monitor_thread.join(timeout=5)

        return {
            "returncode": (
                self.current_process.returncode if self.current_process else -1
            ),
            "stdout": "".join(stdout_lines),
            "stderr": "".join(stderr_lines),
        }

    def generate_output_files(
        self, result, base_filename, output_dir, options, progress_callback=None
    ):
        """Generate output files based on selected formats"""
        output_files = []

        try:
            # Plain text
            if options["output_formats"].get("text", False):
                if progress_callback:
                    progress_callback({"message": "📝 Generating plain text..."})

                text_file = output_dir / f"{base_filename}.txt"
                text_content = result["text"]

                if options.get("clean_text", False):
                    text_content = self.clean_text(text_content)

                with open(text_file, "w", encoding="utf-8") as f:
                    f.write(text_content)

                output_files.append(str(text_file))

            # Detailed transcript
            if options["output_formats"].get("detailed", False):
                if progress_callback:
                    progress_callback(
                        {"message": "📋 Generating detailed transcript..."}
                    )

                detailed_file = output_dir / f"{base_filename}_detailed.txt"
                self.generate_detailed_transcript(result, detailed_file, options)
                output_files.append(str(detailed_file))

            # SRT subtitles
            if options["output_formats"].get("srt", False):
                if progress_callback:
                    progress_callback({"message": "🎬 Generating SRT subtitles..."})

                srt_file = output_dir / f"{base_filename}.srt"
                self.generate_srt_file(result, srt_file, options)
                output_files.append(str(srt_file))

            # VTT subtitles
            if options["output_formats"].get("vtt", False):
                if progress_callback:
                    progress_callback({"message": "📺 Generating VTT subtitles..."})

                vtt_file = output_dir / f"{base_filename}.vtt"
                self.generate_vtt_file(result, vtt_file, options)
                output_files.append(str(vtt_file))

            return output_files

        except Exception as e:
            raise Exception(f"Error generating output files: {e}")

    def clean_text(self, text):
        """Clean text by removing filler words and fixing formatting"""
        # Remove filler words
        text = re.sub(r"\b(um|uh|er|ah|like|you know)\b", "", text, flags=re.IGNORECASE)

        # Clean up spacing
        text = re.sub(r"\s+", " ", text)

        # Capitalize sentences
        text = re.sub(
            r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text
        )

        return text.strip()

    def generate_detailed_transcript(self, result, output_file, options):
        """Generate detailed transcript with timestamps"""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("DETAILED TRANSCRIPTION\n")
            f.write("=" * 50 + "\n\n")

            for i, segment in enumerate(result["segments"]):
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"].strip()
                confidence = segment.get("avg_logprob", 0)

                if options.get("clean_text", False):
                    text = self.clean_text(text)

                # Format timestamps
                start_str = self.format_timestamp_detailed(start_time)
                end_str = self.format_timestamp_detailed(end_time)

                # Add confidence indicator
                confidence_indicator = (
                    "🟢" if confidence > -0.5 else "🟡" if confidence > -1.0 else "🔴"
                )

                f.write(f"[{start_str} - {end_str}] {confidence_indicator}\n")
                f.write(f"{text}\n\n")

    def generate_srt_file(self, result, output_file, options):
        """Generate SRT subtitle file"""
        with open(output_file, "w", encoding="utf-8") as f:
            subtitle_index = 1

            for segment in result["segments"]:
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"].strip()

                if not text:
                    continue

                if options.get("clean_text", False):
                    text = self.clean_text(text)

                # Apply merge short segments logic
                if options.get("merge_short", False):
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
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")

            for segment in result["segments"]:
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"].strip()

                if not text:
                    continue

                if options.get("clean_text", False):
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
