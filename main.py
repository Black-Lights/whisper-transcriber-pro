#!/usr/bin/env python3
"""
Whisper Transcriber Pro - Main Application
Author: Black-Lights (https://github.com/Black-Lights)
Description: Professional GUI for Whisper audio/video transcription using OpenAI's Whisper AI

This application provides a user-friendly interface for transcribing audio and video files
using OpenAI's Whisper model with GPU acceleration support.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.environment_manager import EnvironmentManager
from src.transcription_engine import TranscriptionEngine
from src.model_manager import ModelManager
from src.settings_manager import SettingsManager
from src.utils import get_file_info, format_time

class WhisperTranscriberGUI:
    """Main GUI application for Whisper Transcriber Pro."""
    
    def __init__(self, root):
        """Initialize the main GUI application.
        
        Args:
            root: The main tkinter window
        """
        self.root = root
        self.root.title("Whisper Transcriber Pro v1.1.0")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Initialize managers
        self.env_manager = EnvironmentManager()
        self.model_manager = ModelManager()
        self.settings_manager = SettingsManager()
        self.transcription_engine = None
        
        # Variables
        self.input_file = tk.StringVar()
        self.output_dir = tk.StringVar(value=str(Path.home() / "Documents"))
        self.model_size = tk.StringVar(value="medium")
        self.language = tk.StringVar(value="auto")
        self.device_type = tk.StringVar(value="gpu")
        self.output_formats = {
            'text': tk.BooleanVar(value=True),
            'detailed': tk.BooleanVar(value=True),
            'srt': tk.BooleanVar(value=True),
            'vtt': tk.BooleanVar(value=False)
        }
        
        # Processing state
        self.is_processing = False
        
        # Initialize UI
        self.create_widgets()
        self.check_environment()
    
    def create_widgets(self):
        """Create the main GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Whisper Transcriber Pro", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # File Selection Section
        self.create_file_section(main_frame, row=1)
        
        # Model Settings Section
        self.create_model_section(main_frame, row=2)
        
        # Output Settings Section
        self.create_output_section(main_frame, row=3)
        
        # Advanced Settings Section
        self.create_advanced_section(main_frame, row=4)
        
        # Progress Section
        self.create_progress_section(main_frame, row=5)
        
        # Control Buttons
        self.create_control_buttons(main_frame, row=6)
        
        # Status Bar
        self.create_status_bar(main_frame, row=7)
    
    def create_file_section(self, parent, row):
        """Create file selection section"""
        # File Input Frame
        file_frame = ttk.LabelFrame(parent, text="Input File", padding="10")
        file_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Label(file_frame, text="File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        self.file_entry = ttk.Entry(file_frame, textvariable=self.input_file, width=50)
        self.file_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=(5, 0))
        
        # File info label
        self.file_info_label = ttk.Label(file_frame, text="No file selected", foreground="gray")
        self.file_info_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
    
    def create_model_section(self, parent, row):
        """Create model settings section"""
        model_frame = ttk.LabelFrame(parent, text="Model Settings", padding="10")
        model_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        model_frame.columnconfigure(1, weight=1)
        
        # Model size selection
        ttk.Label(model_frame, text="Model Size:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_size, 
                                  values=["tiny", "base", "small", "medium", "large"], 
                                  state="readonly", width=15)
        model_combo.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Model info button
        ttk.Button(model_frame, text="Info", command=self.show_model_info).grid(row=0, column=2, padx=(5, 0))
        
        # Language selection
        ttk.Label(model_frame, text="Language:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        
        lang_combo = ttk.Combobox(model_frame, textvariable=self.language,
                                 values=["auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                                 state="readonly", width=15)
        lang_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=(5, 0))
        
        # Device selection
        ttk.Label(model_frame, text="Device:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        
        device_frame = ttk.Frame(model_frame)
        device_frame.grid(row=2, column=1, sticky=tk.W, padx=5, pady=(5, 0))
        
        ttk.Radiobutton(device_frame, text="GPU (Faster)", variable=self.device_type, 
                       value="gpu").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(device_frame, text="CPU (Slower)", variable=self.device_type, 
                       value="cpu").grid(row=0, column=1, sticky=tk.W, padx=(20, 0))
    
    def create_output_section(self, parent, row):
        """Create output settings section"""
        output_frame = ttk.LabelFrame(parent, text="Output Settings", padding="10")
        output_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        output_frame.columnconfigure(1, weight=1)
        
        # Output directory
        ttk.Label(output_frame, text="Output Dir:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_dir, width=50)
        self.output_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        ttk.Button(output_frame, text="Browse", command=self.browse_output_dir).grid(row=0, column=2, padx=(5, 0))
        
        # Output formats
        ttk.Label(output_frame, text="Formats:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(10, 0))
        
        formats_frame = ttk.Frame(output_frame)
        formats_frame.grid(row=1, column=1, sticky=tk.W, padx=5, pady=(10, 0))
        
        ttk.Checkbutton(formats_frame, text="Plain Text", 
                       variable=self.output_formats['text']).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(formats_frame, text="Detailed", 
                       variable=self.output_formats['detailed']).grid(row=0, column=1, sticky=tk.W, padx=(20, 0))
        ttk.Checkbutton(formats_frame, text="SRT Subtitles", 
                       variable=self.output_formats['srt']).grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        ttk.Checkbutton(formats_frame, text="VTT Subtitles", 
                       variable=self.output_formats['vtt']).grid(row=0, column=3, sticky=tk.W, padx=(20, 0))
    
    def create_advanced_section(self, parent, row):
        """Create advanced settings section"""
        advanced_frame = ttk.LabelFrame(parent, text="Advanced Settings", padding="10")
        advanced_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Settings will be expandable
        self.advanced_visible = tk.BooleanVar(value=False)
        
        toggle_btn = ttk.Button(advanced_frame, text="Show Advanced", 
                               command=self.toggle_advanced)
        toggle_btn.grid(row=0, column=0, sticky=tk.W)
        
        # Advanced options frame (initially hidden)
        self.advanced_options_frame = ttk.Frame(advanced_frame)
        
        # Clean text option
        self.clean_text = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.advanced_options_frame, text="Remove filler words (um, uh)", 
                       variable=self.clean_text).grid(row=0, column=0, sticky=tk.W, pady=2)
        
        # Merge short segments
        self.merge_short = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.advanced_options_frame, text="Merge short segments", 
                       variable=self.merge_short).grid(row=1, column=0, sticky=tk.W, pady=2)
        
        # Word-level timestamps
        self.word_timestamps = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.advanced_options_frame, text="Word-level timestamps (precise)", 
                       variable=self.word_timestamps).grid(row=2, column=0, sticky=tk.W, pady=2)
    
    def create_progress_section(self, parent, row):
        """Create progress section"""
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding="10")
        progress_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        progress_frame.columnconfigure(0, weight=1)
        
        # Progress bar (determinant mode for percentage)
        self.progress = ttk.Progressbar(progress_frame, mode='determinate', maximum=100)
        self.progress.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Main progress label
        self.progress_label = ttk.Label(progress_frame, text="Ready to start")
        self.progress_label.grid(row=1, column=0, sticky=tk.W)
        
        # Time and speed info
        self.time_label = ttk.Label(progress_frame, text="", foreground="gray")
        self.time_label.grid(row=2, column=0, sticky=tk.W)
        
        # Percentage label
        self.percentage_label = ttk.Label(progress_frame, text="", foreground="blue")
        self.percentage_label.grid(row=3, column=0, sticky=tk.W)
    
    def create_control_buttons(self, parent, row):
        """Create control buttons"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=row, column=0, columnspan=3, pady=20)
        
        # Start/Stop button
        self.start_button = ttk.Button(button_frame, text="Start Transcription", 
                                      command=self.start_transcription)
        self.start_button.grid(row=0, column=0, padx=5)
        
        # Setup button
        ttk.Button(button_frame, text="Setup Environment", 
                  command=self.setup_environment).grid(row=0, column=1, padx=5)
        
        # Download models button
        ttk.Button(button_frame, text="Download Models", 
                  command=self.download_models).grid(row=0, column=2, padx=5)
        
        # Settings button
        ttk.Button(button_frame, text="Settings", 
                  command=self.open_settings).grid(row=0, column=3, padx=5)
    
    def create_status_bar(self, parent, row):
        """Create status bar"""
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(parent, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
    
    # Event handlers
    def browse_file(self):
        """Browse for input file"""
        filetypes = [
            ("All Supported", "*.mp3;*.mp4;*.wav;*.flac;*.m4a;*.ogg;*.wma;*.aac;*.avi;*.mkv;*.mov;*.wmv"),
            ("Audio Files", "*.mp3;*.wav;*.flac;*.m4a;*.ogg;*.wma;*.aac"),
            ("Video Files", "*.mp4;*.avi;*.mkv;*.mov;*.wmv"),
            ("All Files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Audio/Video File",
            filetypes=filetypes
        )
        
        if filename:
            self.input_file.set(filename)
            self.update_file_info(filename)
    
    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)
    
    def update_file_info(self, filename):
        """Update file information display"""
        try:
            info = get_file_info(filename)
            self.file_info_label.config(text=info, foreground="black")
        except Exception as e:
            self.file_info_label.config(text=f"Error reading file: {e}", foreground="red")
    
    def toggle_advanced(self):
        """Toggle advanced settings visibility"""
        if self.advanced_visible.get():
            self.advanced_options_frame.grid_remove()
            self.advanced_visible.set(False)
        else:
            self.advanced_options_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
            self.advanced_visible.set(True)
    
    def show_model_info(self):
        """Show model information dialog"""
        info = """Model Size Information:

tiny: Fastest, least accurate (~39 MB)
   - Speed: ~32x real-time (GPU)
   - Quality: Basic transcription

base: Good speed/accuracy balance (~74 MB)
   - Speed: ~16x real-time (GPU)
   - Quality: Good for clear audio

small: Better accuracy (~244 MB)
   - Speed: ~6x real-time (GPU)
   - Quality: Good for most content

medium: High accuracy - RECOMMENDED (~769 MB)
   - Speed: ~2x real-time (GPU)
   - Quality: Excellent for professional use

large: Best accuracy, slowest (~1550 MB)
   - Speed: ~1x real-time (GPU)
   - Quality: Maximum accuracy"""
        
        messagebox.showinfo("Model Information", info)
    
    def check_environment(self):
        """Check if environment is properly set up"""
        status = self.env_manager.check_environment()
        if not status['venv_exists']:
            self.status_var.set("Environment not set up - Click 'Setup Environment'")
            self.start_button.config(state="disabled")
        elif not status['whisper_installed']:
            self.status_var.set("Whisper not installed - Click 'Setup Environment'")
            self.start_button.config(state="disabled")
        else:
            self.status_var.set("Ready to transcribe")
            self.start_button.config(state="normal")
    
    def setup_environment(self):
        """Setup virtual environment and dependencies"""
        def setup_worker():
            try:
                self.status_var.set("Setting up environment...")
                self.progress.start()
                
                # Setup environment
                success = self.env_manager.setup_environment(
                    progress_callback=self.update_progress
                )
                
                self.progress.stop()
                
                if success:
                    self.status_var.set("Environment setup complete")
                    self.start_button.config(state="normal")
                    messagebox.showinfo("Success", "Environment setup completed successfully!")
                else:
                    self.status_var.set("Environment setup failed")
                    messagebox.showerror("Error", "Environment setup failed. Check console for details.")
                    
            except Exception as e:
                self.progress.stop()
                self.status_var.set("Setup error")
                messagebox.showerror("Error", f"Setup failed: {e}")
        
        threading.Thread(target=setup_worker, daemon=True).start()
    
    def download_models(self):
        """Download Whisper models"""
        def download_worker():
            try:
                self.progress.start()
                self.status_var.set("Downloading models...")
                
                success = self.model_manager.download_models(
                    progress_callback=self.update_progress
                )
                
                self.progress.stop()
                
                if success:
                    self.status_var.set("Models downloaded")
                    messagebox.showinfo("Success", "Models downloaded successfully!")
                else:
                    self.status_var.set("Download failed")
                    messagebox.showerror("Error", "Model download failed.")
                    
            except Exception as e:
                self.progress.stop()
                self.status_var.set("Download error")
                messagebox.showerror("Error", f"Download failed: {e}")
        
        threading.Thread(target=download_worker, daemon=True).start()
    
    def start_transcription(self):
        """Start or stop transcription"""
        if not self.is_processing:
            self.start_transcription_process()
        else:
            self.stop_transcription_process()
    
    def start_transcription_process(self):
        """Start transcription process"""
        # Validate inputs
        if not self.input_file.get():
            messagebox.showerror("Error", "Please select an input file")
            return
        
        if not os.path.exists(self.input_file.get()):
            messagebox.showerror("Error", "Input file does not exist")
            return
        
        if not any(self.output_formats[fmt].get() for fmt in self.output_formats):
            messagebox.showerror("Error", "Please select at least one output format")
            return
        
        # Start transcription
        def transcription_worker():
            try:
                self.is_processing = True
                self.start_button.config(text="Stop")
                
                # Reset progress
                self.progress['value'] = 0
                self.percentage_label.config(text="0%")
                
                # Initialize transcription engine
                self.transcription_engine = TranscriptionEngine(self.env_manager)
                
                # Prepare options
                options = {
                    'model_size': self.model_size.get(),
                    'language': self.language.get() if self.language.get() != 'auto' else None,
                    'device': self.device_type.get(),
                    'output_formats': {fmt: var.get() for fmt, var in self.output_formats.items()},
                    'clean_text': self.clean_text.get(),
                    'merge_short': self.merge_short.get(),
                    'word_timestamps': self.word_timestamps.get()
                }
                
                # Start transcription
                result = self.transcription_engine.transcribe(
                    self.input_file.get(),
                    self.output_dir.get(),
                    options,
                    progress_callback=self.update_transcription_progress
                )
                
                if result['success']:
                    self.status_var.set("Transcription completed successfully")
                    
                    # Show results
                    files_created = "\n".join([f"• {f}" for f in result['files']])
                    messagebox.showinfo("Success", 
                                      f"Transcription completed!\n\nFiles created:\n{files_created}")
                else:
                    self.status_var.set("Transcription failed")
                    messagebox.showerror("Error", f"Transcription failed: {result['error']}")
                
            except Exception as e:
                self.status_var.set("Transcription error")
                messagebox.showerror("Error", f"Transcription failed: {e}")
            
            finally:
                self.is_processing = False
                self.start_button.config(text="Start Transcription")
                self.progress['value'] = 0
                self.percentage_label.config(text="")
        
        threading.Thread(target=transcription_worker, daemon=True).start()
    
    def stop_transcription_process(self):
        """Stop transcription process"""
        if self.transcription_engine:
            self.transcription_engine.stop()
        
        self.is_processing = False
        self.progress['value'] = 0
        self.percentage_label.config(text="")
        self.start_button.config(text="Start Transcription")
        self.status_var.set("Transcription stopped")
    
    def update_progress(self, message):
        """Update progress message"""
        self.root.after(0, lambda: self.progress_label.config(text=message))
    
    def update_transcription_progress(self, progress_info):
        """Update transcription progress with enhanced information"""
        def update_ui():
            # Update main progress message
            self.progress_label.config(text=progress_info['message'])
            
            # Update time information
            if 'time_info' in progress_info:
                self.time_label.config(text=progress_info['time_info'])
            
            # Update progress bar and percentage
            if 'progress_percent' in progress_info:
                progress_value = min(100, max(0, progress_info['progress_percent']))
                self.progress['value'] = progress_value
                self.percentage_label.config(text=f"{progress_value:.0f}%")
        
        self.root.after(0, update_ui)
    
    def open_settings(self):
        """Open settings dialog"""
        messagebox.showinfo("Settings", "Settings dialog coming soon!")

def main():
    """Main entry point"""
    root = tk.Tk()
    
    # Set theme
    style = ttk.Style()
    if "vista" in style.theme_names():
        style.theme_use("vista")
    elif "clam" in style.theme_names():
        style.theme_use("clam")
    
    # Create and run app
    app = WhisperTranscriberGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application closed by user")

if __name__ == "__main__":
    main()