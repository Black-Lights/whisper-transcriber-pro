#!/usr/bin/env python3
"""
Whisper Transcriber Pro - Complete GUI with Hidden Elements
Author: Black-Lights (https://github.com/Black-Lights)
Description: Professional GUI with all live features that show when data is available

Features:
- All live transcription elements (hidden until active)
- Clean, modern interface design
- Smart visibility management
- Complete live transcription display
- Professional appearance with working features
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
import time
import psutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.environment_manager import EnvironmentManager
from src.transcription_engine import TranscriptionEngine
from src.model_manager import ModelManager
from src.settings_manager import SettingsManager
from src.utils import get_file_info, format_time

class WhisperTranscriberGUI:
    """Main GUI application for Whisper Transcriber Pro with complete live features."""
    
    def __init__(self, root):
        """Initialize the main GUI application."""
        self.root = root
        self.root.title("Whisper Transcriber Pro v1.2.0 - Live Edition")
        self.root.geometry("1200x900") 
        self.root.resizable(True, True)
        self.root.configure(bg="#f0f0f0")
        
        # Initialize managers
        self.env_manager = EnvironmentManager()
        self.model_manager = ModelManager()
        self.settings_manager = SettingsManager()
        self.transcription_engine = None
        
        # Process management
        self.current_process = None
        self.process_pid = None
        self.user_stopped = False
        self.is_paused = False
        
        # Timer management
        self.start_time = None
        self.elapsed_time = 0
        self.timer_thread = None
        self.timer_running = False
        
        # Live transcription data
        self.current_segment = 0
        self.total_segments = 0
        self.current_timestamp = 0
        self.total_duration = 0
        self.transcribed_text = ""
        self.live_data_received = False
        
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
        self.setup_styles()
        self.create_widgets()
        self.check_environment()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_styles(self):
        """Setup custom styles for a modern look"""
        style = ttk.Style()
        
        # Use a modern theme
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
        
        # Configure custom styles
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground="#2c3e50")
        style.configure("Heading.TLabel", font=("Segoe UI", 12, "bold"), foreground="#34495e")
        style.configure("Info.TLabel", font=("Segoe UI", 9), foreground="#7f8c8d")
        style.configure("Success.TLabel", font=("Segoe UI", 9), foreground="#27ae60")
        style.configure("Error.TLabel", font=("Segoe UI", 9), foreground="#e74c3c")
        style.configure("Time.TLabel", font=("Segoe UI", 11, "bold"), foreground="#3498db")
        style.configure("LiveData.TLabel", font=("Segoe UI", 10, "bold"), foreground="#e67e22")
        
        # Button styles
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Secondary.TButton", font=("Segoe UI", 9))
        
        # Frame styles
        style.configure("Card.TFrame", relief="solid", borderwidth=1)
        style.configure("Highlight.TFrame", relief="ridge", borderwidth=2)
        style.configure("Live.TFrame", relief="solid", borderwidth=1, background="#ecf0f1")
    
    def create_widgets(self):
        """Create the main GUI widgets with complete live features"""
        # Main container with padding
        main_container = ttk.Frame(self.root, padding="15")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header section
        self.create_header(main_container)
        
        # Content area with paned window for resizable sections
        content_paned = ttk.PanedWindow(main_container, orient=tk.HORIZONTAL)
        content_paned.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Left panel (controls) - 400px width
        left_frame = ttk.Frame(content_paned, width=400)
        content_paned.add(left_frame, weight=1)
        
        # Right panel (live transcription) - expand to fill
        right_frame = ttk.Frame(content_paned, width=600)
        content_paned.add(right_frame, weight=2)
        
        # Create sections
        self.create_control_sections(left_frame)
        self.create_live_transcription_section(right_frame)
        
        # Footer status bar
        self.create_footer(main_container)
    
    def create_header(self, parent):
        """Create header section"""
        header_frame = ttk.Frame(parent, style="Card.TFrame", padding="15")
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Title and subtitle
        title_label = ttk.Label(header_frame, text="Whisper Transcriber Pro", style="Title.TLabel")
        title_label.pack()
        
        subtitle_label = ttk.Label(header_frame, text="Professional Audio & Video Transcription with Live Updates", style="Info.TLabel")
        subtitle_label.pack(pady=(5, 0))
    
    def create_control_sections(self, parent):
        """Create control sections in left column"""
        # File Selection
        self.create_file_section(parent)
        
        # Model Settings
        self.create_model_section(parent)
        
        # Output Settings
        self.create_output_section(parent)
        
        # Advanced Settings
        self.create_advanced_section(parent)
        
        # Progress Section
        self.create_progress_section(parent)
        
        # Control Buttons
        self.create_control_buttons(parent)
    
    def create_live_transcription_section(self, parent):
        """Create complete live transcription display section"""
        # Main live transcription frame
        live_frame = ttk.LabelFrame(parent, text="Live Transcription Monitor", padding="10", style="Highlight.TFrame")
        live_frame.pack(fill=tk.BOTH, expand=True)
        
        # === LIVE STATUS SECTION (Always visible) ===
        status_frame = ttk.Frame(live_frame)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.transcription_status = ttk.Label(status_frame, text="Ready to start transcription", style="Heading.TLabel")
        self.transcription_status.pack(side=tk.LEFT)
        
        # Live indicator (hidden until active)
        self.live_indicator = ttk.Label(status_frame, text="🔴 LIVE", style="LiveData.TLabel")
        # Don't pack initially - will show when live
        
        # === LIVE POSITION SECTION (Hidden until data available) ===
        self.position_frame = ttk.Frame(live_frame, style="Live.TFrame", padding="8")
        # Don't pack initially
        
        # Current position display (HIDDEN - create but don't show)
        pos_header = ttk.Frame(self.position_frame)
        pos_header.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(pos_header, text="Current Position:", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        
        # Position will remain hidden as requested
        self.position_label = ttk.Label(pos_header, text="00:00:00 / 00:00:00", 
                                       font=("Segoe UI", 12), foreground="blue")
        # Don't pack this - keep hidden
        
        # === SEGMENT PROGRESS (Hidden until data available) ===
        segment_frame = ttk.Frame(self.position_frame)
        segment_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(segment_frame, text="Segment Progress:", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        
        self.segment_label = ttk.Label(segment_frame, text="0 of 0", 
                                      font=("Segoe UI", 10), foreground="#27ae60")
        self.segment_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # === CONFIDENCE SECTION (Hidden until data available) ===
        confidence_frame = ttk.Frame(self.position_frame)
        confidence_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(confidence_frame, text="Confidence:", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        
        self.confidence_bar = ttk.Progressbar(confidence_frame, mode='determinate', 
                                            length=120, maximum=100)
        self.confidence_bar.pack(side=tk.LEFT, padx=(10, 5))
        
        self.confidence_label = ttk.Label(confidence_frame, text="--", 
                                         font=("Segoe UI", 9), foreground="#34495e")
        self.confidence_label.pack(side=tk.LEFT)
        
        # === ETA SECTION (Hidden until data available) ===
        eta_frame = ttk.Frame(self.position_frame)
        eta_frame.pack(fill=tk.X)
        
        ttk.Label(eta_frame, text="ETA:", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        
        self.eta_label = ttk.Label(eta_frame, text="Calculating...", 
                                  font=("Segoe UI", 10), foreground="#e67e22")
        self.eta_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # === LIVE TEXT DISPLAY ===
        text_frame = ttk.LabelFrame(live_frame, text="Live Transcribed Text", padding="5")
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Create scrolled text widget with enhanced styling
        self.live_text = scrolledtext.ScrolledText(
            text_frame, 
            wrap=tk.WORD, 
            width=60, 
            height=20,
            font=("Segoe UI", 11),
            bg="white",
            fg="#2c3e50",
            selectbackground="#3498db",
            selectforeground="white",
            borderwidth=1,
            relief="solid",
            state=tk.DISABLED,
            padx=10,
            pady=10
        )
        self.live_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags for different confidence levels and styling
        self.live_text.tag_configure("high_confidence", foreground="#27ae60", font=("Segoe UI", 11))
        self.live_text.tag_configure("medium_confidence", foreground="#f39c12", font=("Segoe UI", 11))
        self.live_text.tag_configure("low_confidence", foreground="#e74c3c", font=("Segoe UI", 11))
        self.live_text.tag_configure("current_segment", background="#ecf0f1")
        self.live_text.tag_configure("timestamp", foreground="#3498db", font=("Segoe UI", 9, "italic"))
        self.live_text.tag_configure("completion", foreground="#27ae60", font=("Segoe UI", 12, "bold"))
        self.live_text.tag_configure("processing", foreground="#95a5a6", font=("Segoe UI", 10, "italic"))
        
        # Text control buttons
        text_buttons = ttk.Frame(text_frame)
        text_buttons.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(text_buttons, text="Clear Text", command=self.clear_live_text, style="Secondary.TButton").pack(side=tk.LEFT)
        ttk.Button(text_buttons, text="Copy Text", command=self.copy_live_text, style="Secondary.TButton").pack(side=tk.LEFT, padx=(5, 0))
        ttk.Button(text_buttons, text="Save Text", command=self.save_live_text, style="Secondary.TButton").pack(side=tk.LEFT, padx=(5, 0))
        
        # Word count display
        self.word_count_label = ttk.Label(text_buttons, text="", style="Info.TLabel")
        self.word_count_label.pack(side=tk.RIGHT)
    
    def create_file_section(self, parent):
        """Create file selection section"""
        file_frame = ttk.LabelFrame(parent, text="Input File", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File input row
        input_frame = ttk.Frame(file_frame)
        input_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.file_entry = ttk.Entry(input_frame, textvariable=self.input_file, font=("Segoe UI", 9))
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ttk.Button(input_frame, text="Browse...", command=self.browse_file, style="Secondary.TButton").pack(side=tk.RIGHT)
        
        # File info display
        self.file_info_label = ttk.Label(file_frame, text="No file selected", style="Info.TLabel")
        self.file_info_label.pack(anchor=tk.W)
    
    def create_model_section(self, parent):
        """Create model settings section"""
        model_frame = ttk.LabelFrame(parent, text="Model Settings", padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Model size row
        model_row = ttk.Frame(model_frame)
        model_row.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(model_row, text="Model:", width=10, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        model_combo = ttk.Combobox(model_row, textvariable=self.model_size, 
                                  values=["tiny", "base", "small", "medium", "large"], 
                                  state="readonly", width=12, font=("Segoe UI", 9))
        model_combo.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(model_row, text="Info", command=self.show_model_info, style="Secondary.TButton").pack(side=tk.LEFT)
        
        # Language row
        lang_row = ttk.Frame(model_frame)
        lang_row.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(lang_row, text="Language:", width=10, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        ttk.Combobox(lang_row, textvariable=self.language,
                    values=["auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                    state="readonly", width=12, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        
        # Device row
        device_row = ttk.Frame(model_frame)
        device_row.pack(fill=tk.X)
        
        ttk.Label(device_row, text="Device:", width=10, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        ttk.Radiobutton(device_row, text="GPU", variable=self.device_type, value="gpu").pack(side=tk.LEFT)
        ttk.Radiobutton(device_row, text="CPU", variable=self.device_type, value="cpu").pack(side=tk.LEFT, padx=(15, 0))
    
    def create_output_section(self, parent):
        """Create output settings section"""
        output_frame = ttk.LabelFrame(parent, text="Output Settings", padding="10")
        output_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Output directory row
        dir_frame = ttk.Frame(output_frame)
        dir_frame.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Label(dir_frame, text="Output:", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.output_entry = ttk.Entry(dir_frame, textvariable=self.output_dir, font=("Segoe UI", 9))
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        ttk.Button(dir_frame, text="Browse...", command=self.browse_output_dir, style="Secondary.TButton").pack(side=tk.RIGHT)
        
        # Output formats (2x2 grid)
        formats_frame = ttk.Frame(output_frame)
        formats_frame.pack(fill=tk.X)
        
        ttk.Checkbutton(formats_frame, text="Plain Text", variable=self.output_formats['text']).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Checkbutton(formats_frame, text="Detailed", variable=self.output_formats['detailed']).grid(row=0, column=1, sticky=tk.W)
        ttk.Checkbutton(formats_frame, text="SRT Subtitles", variable=self.output_formats['srt']).grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        ttk.Checkbutton(formats_frame, text="VTT Subtitles", variable=self.output_formats['vtt']).grid(row=1, column=1, sticky=tk.W, pady=(5, 0))
    
    def create_advanced_section(self, parent):
        """Create advanced settings section"""
        advanced_frame = ttk.LabelFrame(parent, text="Advanced Options", padding="10")
        advanced_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Advanced options
        self.clean_text = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="Clean text (remove filler words)", variable=self.clean_text).pack(anchor=tk.W, pady=1)
        
        self.merge_short = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="Merge short segments", variable=self.merge_short).pack(anchor=tk.W, pady=1)
        
        self.word_timestamps = tk.BooleanVar(value=False)
        ttk.Checkbutton(advanced_frame, text="Word-level timestamps", variable=self.word_timestamps).pack(anchor=tk.W, pady=1)
    
    def create_progress_section(self, parent):
        """Create progress section"""
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding="10")
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Progress bar
        self.progress = ttk.Progressbar(progress_frame, mode='determinate', maximum=100)
        self.progress.pack(fill=tk.X, pady=(0, 8))
        
        # Progress info row
        info_frame = ttk.Frame(progress_frame)
        info_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.progress_label = ttk.Label(info_frame, text="Ready to start", style="Info.TLabel")
        self.progress_label.pack(side=tk.LEFT)
        
        # Progress percentage (right aligned)
        self.progress_percent = ttk.Label(info_frame, text="0%", style="Info.TLabel")
        self.progress_percent.pack(side=tk.RIGHT)
        
        # Timer row
        timer_frame = ttk.Frame(progress_frame)
        timer_frame.pack(fill=tk.X)
        
        ttk.Label(timer_frame, text="Elapsed:", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.timer_label = ttk.Label(timer_frame, text="00:00", style="Time.TLabel")
        self.timer_label.pack(side=tk.LEFT, padx=(5, 0))
    
    def create_control_buttons(self, parent):
        """Create control buttons"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Main action button
        self.start_button = ttk.Button(button_frame, text="Start Transcription", 
                                      command=self.start_transcription, style="Primary.TButton")
        self.start_button.pack(fill=tk.X, pady=(0, 8))
        
        # Control buttons row
        control_frame = ttk.Frame(button_frame)
        control_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.pause_button = ttk.Button(control_frame, text="Pause", 
                                      command=self.pause_transcription, state="disabled")
        self.pause_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.stop_button = ttk.Button(control_frame, text="Stop", 
                                     command=self.stop_transcription, state="disabled")
        self.stop_button.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Setup buttons row
        setup_frame = ttk.Frame(button_frame)
        setup_frame.pack(fill=tk.X)
        
        ttk.Button(setup_frame, text="Setup Environment", command=self.setup_environment, style="Secondary.TButton").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        ttk.Button(setup_frame, text="Download Models", command=self.download_models, style="Secondary.TButton").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 3))
        ttk.Button(setup_frame, text="Settings", command=self.open_settings, style="Secondary.TButton").pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(3, 0))
    
    def create_footer(self, parent):
        """Create footer status bar"""
        footer_frame = ttk.Frame(parent, style="Card.TFrame", padding="8")
        footer_frame.pack(fill=tk.X, pady=(15, 0))
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(footer_frame, textvariable=self.status_var, style="Info.TLabel")
        status_label.pack(side=tk.LEFT)
        
        # Version info
        version_label = ttk.Label(footer_frame, text="v1.2.0 - Complete Live Edition", style="Info.TLabel")
        version_label.pack(side=tk.RIGHT)
    
    # ===== LIVE DISPLAY MANAGEMENT =====
    def show_live_elements(self):
        """Show live elements when transcription data becomes available"""
        if not self.live_data_received:
            self.live_data_received = True
            
            # Show live indicator
            self.live_indicator.pack(side=tk.RIGHT)
            
            # Show position frame with all live data
            self.position_frame.pack(fill=tk.X, pady=(0, 10))
            
            # Update transcription status
            self.transcription_status.config(text="🎙️ Live Transcription Active")
    
    def hide_live_elements(self):
        """Hide live elements when transcription stops"""
        self.live_data_received = False
        
        # Hide live indicator
        self.live_indicator.pack_forget()
        
        # Hide position frame
        self.position_frame.pack_forget()
        
        # Reset transcription status
        self.transcription_status.config(text="Transcription completed")
    
    # ===== TIMER MANAGEMENT =====
    def start_timer(self):
        """Start the elapsed time timer"""
        self.start_time = time.time()
        self.timer_running = True
        self.timer_thread = threading.Thread(target=self.update_timer, daemon=True)
        self.timer_thread.start()
    
    def stop_timer(self):
        """Stop the elapsed time timer"""
        self.timer_running = False
        if self.timer_thread and self.timer_thread.is_alive():
            self.timer_thread.join(timeout=1)
    
    def pause_timer(self):
        """Pause the timer"""
        if self.start_time:
            self.elapsed_time += time.time() - self.start_time
            self.start_time = None
    
    def resume_timer(self):
        """Resume the timer"""
        self.start_time = time.time()
    
    def reset_timer(self):
        """Reset the timer completely"""
        self.stop_timer()
        self.elapsed_time = 0
        self.start_time = None
        self.root.after(0, lambda: self.timer_label.config(text="00:00"))
    
    def update_timer(self):
        """Update timer display (runs in separate thread)"""
        while self.timer_running:
            try:
                if self.start_time:
                    current_elapsed = self.elapsed_time + (time.time() - self.start_time)
                else:
                    current_elapsed = self.elapsed_time
                
                # Format time
                minutes = int(current_elapsed // 60)
                seconds = int(current_elapsed % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"
                
                # Update UI in main thread
                self.root.after(0, lambda: self.timer_label.config(text=time_str))
                
                time.sleep(1)
            except Exception as e:
                print(f"Timer update error: {e}")
                break
    
    # ===== PROCESS MANAGEMENT =====
    def cleanup_processes(self):
        """Clean up any existing transcription processes"""
        try:
            # Terminate current process if exists
            if self.current_process and self.current_process.poll() is None:
                self.current_process.terminate()
                try:
                    self.current_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.current_process.kill()
            
            # Kill any Python processes running transcription
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if (proc.info['name'] and 'python' in proc.info['name'].lower() and 
                        proc.info['cmdline'] and any('temp_transcribe' in arg for arg in proc.info['cmdline'])):
                        proc.terminate()
                        print(f"Terminated zombie transcription process: {proc.info['pid']}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            print(f"Error cleaning up processes: {e}")
    
    def reset_ui_state(self):
        """Reset UI to initial state"""
        self.is_processing = False
        self.is_paused = False
        self.user_stopped = False
        
        # Reset buttons
        self.start_button.config(text="Start Transcription", state="normal")
        self.pause_button.config(state="disabled")
        self.stop_button.config(state="disabled")
        
        # Reset progress
        self.progress['value'] = 0
        self.progress_label.config(text="Ready to start")
        self.progress_percent.config(text="0%")
        
        # Hide live elements
        self.hide_live_elements()
        
        # Reset live data
        self.current_segment = 0
        self.total_segments = 0
        self.segment_label.config(text="0 of 0")
        self.confidence_bar['value'] = 0
        self.confidence_label.config(text="--")
        self.eta_label.config(text="Calculating...")
        
        # Reset timer
        self.reset_timer()
    
    # ===== LIVE TRANSCRIPTION METHODS =====
    def update_live_transcription(self, segment_data):
        """Update live transcription display with all data"""
        try:
            # Show live elements if not already shown
            if not self.live_data_received:
                self.show_live_elements()
            
            # Update segment progress
            current_seg = segment_data.get('segment_index', 0)
            total_seg = segment_data.get('total_segments', 0)
            if total_seg > 0:
                self.segment_label.config(text=f"{current_seg} of {total_seg}")
                self.current_segment = current_seg
                self.total_segments = total_seg
            
            # Update confidence
            confidence = segment_data.get('avg_logprob', 0)
            # Convert logprob to percentage (more accurate conversion)
            confidence_percent = max(0, min(100, (confidence + 1) * 100))
            self.confidence_bar['value'] = confidence_percent
            self.confidence_label.config(text=f"{confidence_percent:.0f}%")
            
            # Update live text with proper formatting
            text = segment_data.get('text', '').strip()
            if text:
                # Determine confidence level for color coding
                if confidence > -0.3:
                    tag = "high_confidence"
                elif confidence > -0.7:
                    tag = "medium_confidence"
                else:
                    tag = "low_confidence"
                
                # Insert text with timestamp
                self.live_text.config(state=tk.NORMAL)
                
                # Add timestamp
                start_time = segment_data.get('start', 0)
                timestamp_text = f"\n[{self.format_time_display(start_time)}] "
                self.live_text.insert(tk.END, timestamp_text, "timestamp")
                
                # Add transcribed text
                self.live_text.insert(tk.END, text, tag)
                
                # Update word count
                current_text = self.live_text.get(1.0, tk.END)
                word_count = len(current_text.split()) if current_text.strip() else 0
                self.word_count_label.config(text=f"{word_count} words")
                
                # Auto-scroll to bottom
                self.live_text.see(tk.END)
                self.live_text.config(state=tk.DISABLED)
                
        except Exception as e:
            print(f"Error updating live transcription: {e}")
    
    def clear_live_text(self):
        """Clear the live transcription text"""
        self.live_text.config(state=tk.NORMAL)
        self.live_text.delete(1.0, tk.END)
        self.live_text.config(state=tk.DISABLED)
        self.word_count_label.config(text="")
    
    def copy_live_text(self):
        """Copy live transcription text to clipboard"""
        try:
            text_content = self.live_text.get(1.0, tk.END).strip()
            if text_content:
                self.root.clipboard_clear()
                self.root.clipboard_append(text_content)
                self.transcription_status.config(text="Text copied to clipboard")
            else:
                messagebox.showinfo("Info", "No text to copy")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy text: {e}")
    
    def save_live_text(self):
        """Save live transcription text to file"""
        try:
            text_content = self.live_text.get(1.0, tk.END).strip()
            if not text_content:
                messagebox.showinfo("Info", "No text to save")
                return
            
            filename = filedialog.asksaveasfilename(
                title="Save Live Transcription",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                self.transcription_status.config(text=f"Text saved to {Path(filename).name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save text: {e}")
    
    def format_time_display(self, seconds):
        """Format seconds to MM:SS or HH:MM:SS display"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    # ===== EVENT HANDLERS =====
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
            self.file_info_label.config(text=info, foreground="#27ae60")
        except Exception as e:
            self.file_info_label.config(text=f"Error reading file: {e}", foreground="#e74c3c")
    
    def show_model_info(self):
        """Show model information dialog"""
        info = """Model Size Information:

tiny: Fastest, least accurate (~39 MB)
   • Speed: ~32x real-time (GPU)
   • Quality: Basic transcription

base: Good speed/accuracy balance (~74 MB)
   • Speed: ~16x real-time (GPU)
   • Quality: Good for clear audio

small: Better accuracy (~244 MB)
   • Speed: ~6x real-time (GPU)
   • Quality: Good for most content

medium: High accuracy - RECOMMENDED (~769 MB)
   • Speed: ~2x real-time (GPU)
   • Quality: Excellent for professional use

large: Best accuracy, slowest (~1550 MB)
   • Speed: ~1x real-time (GPU)
   • Quality: Maximum accuracy"""
        
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
                
                success = self.env_manager.setup_environment(
                    progress_callback=lambda msg: self.root.after(0, lambda: self.progress_label.config(text=msg))
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
                    progress_callback=lambda msg: self.root.after(0, lambda: self.progress_label.config(text=msg))
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
        
        # Clean up any existing processes first
        self.cleanup_processes()
        
        # Reset state
        self.user_stopped = False
        self.is_paused = False
        self.clear_live_text()
        
        # Start transcription
        def transcription_worker():
            try:
                self.is_processing = True
                
                # Update UI
                self.start_button.config(text="Starting...", state="disabled")
                self.pause_button.config(state="normal")
                self.stop_button.config(state="normal")
                
                # Start timer
                self.start_timer()
                
                # Initialize transcription engine
                self.transcription_engine = TranscriptionEngine(self.env_manager)
                
                # Prepare options with live callback
                options = {
                    'model_size': self.model_size.get(),
                    'language': self.language.get() if self.language.get() != 'auto' else None,
                    'device': self.device_type.get(),
                    'output_formats': {fmt: var.get() for fmt, var in self.output_formats.items()},
                    'clean_text': self.clean_text.get(),
                    'merge_short': self.merge_short.get(),
                    'word_timestamps': self.word_timestamps.get(),
                    'enhanced_silence_handling': True,
                    'live_callback': self.update_live_transcription
                }
                
                # Update status
                self.root.after(0, lambda: self.status_var.set("Transcribing..."))
                self.root.after(0, lambda: self.start_button.config(text="Transcribing..."))
                self.root.after(0, lambda: self.transcription_status.config(text="Initializing transcription..."))
                
                # Show initial processing message
                self.root.after(0, lambda: self.show_processing_message())
                
                # Start transcription
                result = self.transcription_engine.transcribe(
                    self.input_file.get(),
                    self.output_dir.get(),
                    options,
                    progress_callback=self.update_transcription_progress
                )
                
                # Handle results
                if self.user_stopped:
                    self.root.after(0, lambda: self.status_var.set("Transcription stopped by user"))
                    self.root.after(0, lambda: self.transcription_status.config(text="Stopped by user"))
                    self.root.after(0, lambda: messagebox.showinfo("Stopped", "Transcription stopped by user"))
                elif result['success']:
                    self.root.after(0, lambda: self.status_var.set("Transcription completed successfully"))
                    
                    # Show completion message in live text
                    self.root.after(0, lambda: self.show_completion_message(result))
                    
                    # Show results dialog
                    files_created = "\n".join([f"• {Path(f).name}" for f in result['files']])
                    time_taken = result.get('transcription_time', 0)
                    segments = result.get('segments', 0)
                    
                    result_msg = (f"🎉 Transcription completed successfully!\n\n"
                                f"⏱️ Time taken: {time_taken:.1f} seconds\n"
                                f"📊 Segments processed: {segments}\n"
                                f"📝 Total words: {len(self.live_text.get(1.0, tk.END).split())}\n\n"
                                f"📁 Files created:\n{files_created}")
                    
                    self.root.after(0, lambda: messagebox.showinfo("Success", result_msg))
                else:
                    error_msg = result.get('error', 'Unknown error')
                    self.root.after(0, lambda: self.status_var.set("Transcription failed"))
                    self.root.after(0, lambda: self.transcription_status.config(text="Transcription failed"))
                    
                    # Check for specific error types
                    if "dots" in error_msg.lower() or "no speech" in error_msg.lower():
                        self.root.after(0, lambda: messagebox.showerror("Audio Quality Issue", 
                            f"❌ Transcription failed due to poor audio quality or extended silence.\n\n"
                            f"💡 Suggestions:\n"
                            f"• Check if audio contains speech\n"
                            f"• Try a larger model (medium/large)\n"
                            f"• Verify audio file is not corrupted\n"
                            f"• Check audio volume levels\n\n"
                            f"🔍 Error: {error_msg}"))
                    else:
                        self.root.after(0, lambda: messagebox.showerror("Error", f"❌ Transcription failed:\n\n{error_msg}"))
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self.status_var.set("Transcription error"))
                self.root.after(0, lambda: self.transcription_status.config(text="Error occurred"))
                self.root.after(0, lambda: messagebox.showerror("Error", f"❌ Transcription failed:\n\n{error_msg}"))
            
            finally:
                # Always reset UI state
                self.root.after(0, self.reset_ui_state)
        
        threading.Thread(target=transcription_worker, daemon=True).start()
    
    def show_processing_message(self):
        """Show initial processing message in live text"""
        self.live_text.config(state=tk.NORMAL)
        self.live_text.delete(1.0, tk.END)
        
        processing_msg = "🔄 Initializing transcription engine...\n"
        processing_msg += "📥 Loading model and preparing audio...\n"
        processing_msg += "⏳ Live updates will appear here during transcription...\n\n"
        
        self.live_text.insert(tk.END, processing_msg, "processing")
        self.live_text.config(state=tk.DISABLED)
    
    def show_completion_message(self, result):
        """Show completion message in live text"""
        self.live_text.config(state=tk.NORMAL)
        
        # Add completion header
        completion_msg = f"\n\n{'='*50}\n"
        completion_msg += "🎉 TRANSCRIPTION COMPLETED SUCCESSFULLY!\n"
        completion_msg += f"{'='*50}\n"
        completion_msg += f"⏱️ Processing time: {result.get('transcription_time', 0):.1f} seconds\n"
        completion_msg += f"📊 Total segments: {result.get('segments', 0)}\n"
        completion_msg += f"📁 Output files: {len(result.get('files', []))}\n"
        
        self.live_text.insert(tk.END, completion_msg, "completion")
        self.live_text.see(tk.END)
        self.live_text.config(state=tk.DISABLED)
    
    def pause_transcription(self):
        """Pause/Resume transcription"""
        if not self.is_processing:
            return
            
        if self.is_paused:
            # Resume
            self.is_paused = False
            self.resume_timer()
            self.pause_button.config(text="Pause")
            self.status_var.set("Transcription resumed")
            self.transcription_status.config(text="🎙️ Live Transcription Active (Resumed)")
            
            # Resume transcription engine if supported
            if hasattr(self.transcription_engine, 'resume'):
                self.transcription_engine.resume()
        else:
            # Pause
            self.is_paused = True
            self.pause_timer()
            self.pause_button.config(text="Resume")
            self.status_var.set("Transcription paused")
            self.transcription_status.config(text="⏸️ Transcription Paused")
            
            # Pause transcription engine if supported
            if hasattr(self.transcription_engine, 'pause'):
                self.transcription_engine.pause()
    
    def stop_transcription(self):
        """Stop transcription process"""
        self.user_stopped = True
        
        # Stop transcription engine
        if self.transcription_engine:
            self.transcription_engine.stop()
        
        # Clean up processes
        self.cleanup_processes()
        
        # Update status
        self.status_var.set("Stopping transcription...")
        self.transcription_status.config(text="🛑 Stopping...")
        
        # Reset UI will be handled by transcription worker thread
    
    def update_transcription_progress(self, progress_info):
        """Update transcription progress with enhanced information"""
        def update_ui():
            try:
                # Update main progress message
                if 'message' in progress_info:
                    self.progress_label.config(text=progress_info['message'])
                
                # Update progress bar and percentage
                if 'progress_percent' in progress_info:
                    progress_value = min(100, max(0, progress_info['progress_percent']))
                    self.progress['value'] = progress_value
                    self.progress_percent.config(text=f"{progress_value:.0f}%")
                
                # Update ETA when available
                if 'eta_seconds' in progress_info:
                    eta = progress_info['eta_seconds']
                    if eta > 0:
                        eta_str = self.format_time_display(eta)
                        self.eta_label.config(text=eta_str)
                    else:
                        self.eta_label.config(text="Almost done...")
                
                # Handle live transcription updates
                if 'live_segment' in progress_info:
                    self.update_live_transcription(progress_info['live_segment'])
                
            except Exception as e:
                print(f"Error updating progress: {e}")
        
        self.root.after(0, update_ui)
    
    def open_settings(self):
        """Open settings dialog"""
        messagebox.showinfo("Settings", "⚙️ Advanced Settings\n\nSettings dialog coming soon!\n\nCurrent features:\n• Live transcription updates\n• Multiple output formats\n• Enhanced audio processing\n• Real-time confidence monitoring")
    
    def on_closing(self):
        """Handle application closing"""
        if self.is_processing:
            if messagebox.askokcancel("Quit", "Transcription in progress. Stop and quit?"):
                self.user_stopped = True
                self.cleanup_processes()
                self.stop_timer()
                self.root.destroy()
        else:
            self.cleanup_processes()
            self.stop_timer()
            self.root.destroy()

def main():
    """Main entry point"""
    root = tk.Tk()
    
    # Create and run app
    app = WhisperTranscriberGUI(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application closed by user")
        app.cleanup_processes()

if __name__ == "__main__":
    main()