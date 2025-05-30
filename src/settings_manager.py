"""
Settings Manager - Handles application settings and preferences
Author: Black-Lights (https://github.com/Black-Lights)
Project: Whisper Transcriber Pro
"""

import json
import os
from pathlib import Path

class SettingsManager:
    def __init__(self):
        self.app_dir = Path(__file__).parent.parent
        self.settings_file = self.app_dir / "settings.json"
        self.default_settings = {
            "general": {
                "default_model": "medium",
                "default_language": "auto",
                "default_device": "gpu",
                "default_output_dir": str(Path.home() / "Documents" / "Whisper_Output")
            },
            "output": {
                "formats": {
                    "text": True,
                    "detailed": True, 
                    "srt": True,
                    "vtt": False
                },
                "text_processing": {
                    "clean_text": True,
                    "merge_short_segments": True,
                    "word_timestamps": False,
                    "max_chars_per_line": 80
                }
            },
            "advanced": {
                "gpu_memory_fraction": 0.8,
                "batch_size": 16,
                "beam_size": 5,
                "best_of": 5,
                "temperature": 0.0,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6
            },
            "ui": {
                "theme": "default",
                "window_size": "900x700",
                "auto_check_updates": True,
                "show_advanced_by_default": False
            }
        }
        
        self.settings = self.load_settings()
    
    def load_settings(self):
        """Load settings from file or create default"""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                
                # Merge with defaults to ensure all keys exist
                settings = self.deep_merge(self.default_settings.copy(), loaded_settings)
                return settings
            else:
                return self.default_settings.copy()
                
        except Exception as e:
            print(f"Error loading settings: {e}")
            return self.default_settings.copy()
    
    def save_settings(self):
        """Save current settings to file"""
        try:
            # Ensure directory exists
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False
    
    def get_setting(self, category, key, default=None):
        """Get a specific setting value"""
        try:
            return self.settings.get(category, {}).get(key, default)
        except:
            return default
    
    def set_setting(self, category, key, value):
        """Set a specific setting value"""
        try:
            if category not in self.settings:
                self.settings[category] = {}
            
            self.settings[category][key] = value
            return True
        except Exception as e:
            print(f"Error setting {category}.{key}: {e}")
            return False
    
    def get_all_settings(self):
        """Get all settings"""
        return self.settings.copy()
    
    def reset_to_defaults(self):
        """Reset all settings to defaults"""
        self.settings = self.default_settings.copy()
        return self.save_settings()
    
    def reset_category(self, category):
        """Reset a specific category to defaults"""
        if category in self.default_settings:
            self.settings[category] = self.default_settings[category].copy()
            return self.save_settings()
        return False
    
    def export_settings(self, file_path):
        """Export settings to a file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error exporting settings: {e}")
            return False
    
    def import_settings(self, file_path):
        """Import settings from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                imported_settings = json.load(f)
            
            # Validate and merge
            self.settings = self.deep_merge(self.default_settings.copy(), imported_settings)
            return self.save_settings()
        except Exception as e:
            print(f"Error importing settings: {e}")
            return False
    
    def deep_merge(self, dict1, dict2):
        """Deeply merge two dictionaries"""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def validate_settings(self):
        """Validate current settings"""
        errors = []
        
        # Validate model
        valid_models = ["tiny", "base", "small", "medium", "large"]
        model = self.get_setting("general", "default_model")
        if model not in valid_models:
            errors.append(f"Invalid model: {model}")
        
        # Validate device
        valid_devices = ["cpu", "gpu"]
        device = self.get_setting("general", "default_device")
        if device not in valid_devices:
            errors.append(f"Invalid device: {device}")
        
        # Validate output directory
        output_dir = self.get_setting("general", "default_output_dir")
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        except:
            errors.append(f"Invalid output directory: {output_dir}")
        
        # Validate numeric settings
        numeric_settings = [
            ("advanced", "gpu_memory_fraction", 0.1, 1.0),
            ("advanced", "temperature", 0.0, 1.0),
            ("output", "max_chars_per_line", 20, 200)
        ]
        
        for category, key, min_val, max_val in numeric_settings:
            value = self.get_setting(category, key)
            if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                errors.append(f"Invalid {category}.{key}: {value} (should be {min_val}-{max_val})")
        
        return errors
    
    def get_transcription_options(self):
        """Get transcription options from settings"""
        return {
            'model_size': self.get_setting("general", "default_model", "medium"),
            'language': self.get_setting("general", "default_language", "auto"),
            'device': self.get_setting("general", "default_device", "gpu"),
            'output_formats': self.get_setting("output", "formats", {
                "text": True, "detailed": True, "srt": True, "vtt": False
            }),
            'clean_text': self.get_setting("output", "text_processing")["clean_text"],
            'merge_short': self.get_setting("output", "text_processing")["merge_short_segments"],
            'word_timestamps': self.get_setting("output", "text_processing")["word_timestamps"],
            'max_chars_per_line': self.get_setting("output", "text_processing")["max_chars_per_line"]
        }
    
    def update_from_ui(self, ui_values):
        """Update settings from UI values"""
        # General settings
        self.set_setting("general", "default_model", ui_values.get("model_size", "medium"))
        self.set_setting("general", "default_language", ui_values.get("language", "auto"))
        self.set_setting("general", "default_device", ui_values.get("device", "gpu"))
        
        # Output formats
        formats = self.get_setting("output", "formats", {})
        for fmt in ["text", "detailed", "srt", "vtt"]:
            if fmt in ui_values:
                formats[fmt] = ui_values[fmt]
        self.set_setting("output", "formats", formats)
        
        # Text processing
        text_processing = self.get_setting("output", "text_processing", {})
        processing_keys = ["clean_text", "merge_short", "word_timestamps", "max_chars_per_line"]
        for key in processing_keys:
            if key in ui_values:
                text_processing[key] = ui_values[key]
        self.set_setting("output", "text_processing", text_processing)
        
        return self.save_settings()
    
    def get_recent_files(self, max_count=10):
        """Get list of recent input files"""
        return self.get_setting("ui", "recent_files", [])[:max_count]
    
    def add_recent_file(self, file_path):
        """Add file to recent files list"""
        recent = self.get_recent_files()
        
        # Remove if already exists
        if file_path in recent:
            recent.remove(file_path)
        
        # Add to beginning
        recent.insert(0, file_path)
        
        # Limit to 10 files
        recent = recent[:10]
        
        self.set_setting("ui", "recent_files", recent)
        self.save_settings()