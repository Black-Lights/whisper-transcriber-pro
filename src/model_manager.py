"""
Model Manager - Handles Whisper model downloading and management
Author: Black-Lights (https://github.com/Black-Lights)
Project: Whisper Transcriber Pro

This module provides management of faster-whisper CTranslate2 models including
downloading, caching, and maintenance operations.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import requests


class ModelManager:
    """Manages faster-whisper model downloading and caching.

    This class handles all aspects of Whisper model management including:
    - Downloading models via faster-whisper (from HuggingFace)
    - Caching models for offline use
    - Providing model information and recommendations
    """

    def __init__(self):
        """Initialize the ModelManager with model definitions and cache directory."""
        self.models = {
            "tiny": {
                "size": "75 MB",
                "description": "Fastest, least accurate",
                "speed": "~100x real-time (GPU batched)",
                "accuracy": "Basic",
                "repo_id": "Systran/faster-whisper-tiny",
            },
            "base": {
                "size": "145 MB",
                "description": "Good speed/accuracy balance",
                "speed": "~80x real-time (GPU batched)",
                "accuracy": "Good",
                "repo_id": "Systran/faster-whisper-base",
            },
            "small": {
                "size": "488 MB",
                "description": "Better accuracy",
                "speed": "~60x real-time (GPU batched)",
                "accuracy": "Better",
                "repo_id": "Systran/faster-whisper-small",
            },
            "medium": {
                "size": "1.5 GB",
                "description": "High accuracy",
                "speed": "~40x real-time (GPU batched)",
                "accuracy": "High",
                "repo_id": "Systran/faster-whisper-medium",
            },
            "large-v3": {
                "size": "3.1 GB",
                "description": "Best accuracy, needs 10GB+ VRAM",
                "speed": "~20x real-time (GPU batched)",
                "accuracy": "Maximum",
                "repo_id": "Systran/faster-whisper-large-v3",
            },
            "large-v3-turbo": {
                "size": "1.6 GB",
                "description": "Fast + accurate - RECOMMENDED",
                "speed": "~60x real-time (GPU batched)",
                "accuracy": "High (7.75% WER)",
                "repo_id": "deepdml/faster-whisper-large-v3-turbo-ct2",
            },
            "distil-large-v3": {
                "size": "1.5 GB",
                "description": "Distilled, very fast",
                "speed": "~70x real-time (GPU batched)",
                "accuracy": "High (within 1% of large-v3)",
                "repo_id": "Systran/faster-distil-whisper-large-v3",
            },
        }

        # Model cache directory (faster-whisper uses HuggingFace hub cache)
        self.cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        self.download_progress = {}

    def get_model_info(self, model_name=None):
        """Get information about available models.

        Args:
            model_name (str, optional): Specific model to get info for.
                                      If None, returns info for all models.

        Returns:
            dict: Model information dictionary
        """
        if model_name:
            return self.models.get(model_name, {})
        return self.models

    def check_downloaded_models(self):
        """Check which models are already downloaded and their status.

        Returns:
            dict: Dictionary mapping model names to their download status and info
        """
        downloaded = {}

        if not self.cache_dir.exists():
            return downloaded

        # Check for CTranslate2 model directories in HuggingFace cache
        for model_name, model_info in self.models.items():
            repo_id = model_info.get("repo_id", "")
            # HuggingFace stores models in models--org--name format
            repo_dir_name = "models--" + repo_id.replace("/", "--")
            model_dir = self.cache_dir / repo_dir_name

            if model_dir.exists():
                # Calculate total size of model directory
                total_size = sum(
                    f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
                )

                downloaded[model_name] = {
                    "size_bytes": total_size,
                    "size_mb": total_size / (1024 * 1024),
                    "path": str(model_dir),
                    "valid": True,  # If directory exists with files, consider valid
                    "last_modified": model_dir.stat().st_mtime,
                }

        return downloaded

    def verify_model_file(self, model_name, file_path):
        """Verify model directory exists and contains expected files.

        Args:
            model_name (str): Name of the model to verify
            file_path (Path): Path to the model directory

        Returns:
            bool: True if model directory is valid
        """
        try:
            path = Path(file_path)
            if path.is_dir():
                # Check for CTranslate2 model files
                return any(path.rglob("model.bin")) or any(path.rglob("*.bin"))
            return False
        except Exception as e:
            print(f"Error verifying {model_name}: {e}")
            return False

    def download_models(self, models=None, progress_callback=None):
        """Download specified models or default model.

        Args:
            models (list or str, optional): Model(s) to download. Defaults to ['large-v3-turbo'].
            progress_callback (callable, optional): Function to call with progress updates

        Returns:
            bool: True if at least one model was downloaded successfully
        """
        if models is None:
            models = ["large-v3-turbo"]  # Default to large-v3-turbo model

        if isinstance(models, str):
            models = [models]

        try:
            downloaded_models = []

            for model_name in models:
                if model_name not in self.models:
                    print(f"Unknown model: {model_name}")
                    continue

                if progress_callback:
                    progress_callback(
                        f"Starting download: {model_name} model ({self.models[model_name]['size']})..."
                    )

                success = self.download_single_model(model_name, progress_callback)

                if success:
                    downloaded_models.append(model_name)
                    if progress_callback:
                        progress_callback(f"Successfully downloaded {model_name} model")
                else:
                    print(f"Failed to download {model_name} model")
                    if progress_callback:
                        progress_callback(f"Failed to download {model_name} model")

            return len(downloaded_models) > 0

        except Exception as e:
            print(f"Model download failed: {e}")
            if progress_callback:
                progress_callback(f"Download error: {e}")
            return False

    def download_single_model(self, model_name, progress_callback=None):
        """Download a single model using faster-whisper's built-in download.

        faster-whisper automatically downloads CTranslate2 models from HuggingFace
        when WhisperModel is instantiated. This method triggers that download.

        Args:
            model_name (str): Name of the model to download
            progress_callback (callable, optional): Function to call with progress updates

        Returns:
            bool: True if download was successful, False otherwise
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Unknown model: {model_name}")

            # Check if already downloaded
            downloaded = self.check_downloaded_models()
            if model_name in downloaded and downloaded[model_name].get("valid", False):
                if progress_callback:
                    progress_callback(f"Model {model_name} already exists and is valid")
                return True

            if progress_callback:
                progress_callback(f"Downloading {model_name} model from HuggingFace...")

            # Create a script that loads the model (triggering download)
            download_script = f"""
import sys
try:
    from faster_whisper import WhisperModel
    print("Downloading {model_name} model...")
    model = WhisperModel("{model_name}", device="cpu", compute_type="int8")
    print("Model downloaded and loaded successfully!")
except Exception as e:
    print(f"Download failed: {{e}}", file=sys.stderr)
    sys.exit(1)
"""

            # Write script to temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(download_script)
                script_path = f.name

            try:
                # Run the download script
                result = subprocess.run(
                    ["python", script_path],
                    capture_output=True,
                    text=True,
                    timeout=1800,
                )

                if result.returncode == 0:
                    if progress_callback:
                        progress_callback(
                            f"Model {model_name} downloaded and verified successfully"
                        )
                    return True
                else:
                    print(f"Download error: {result.stderr}")
                    if progress_callback:
                        progress_callback(f"Download failed: {result.stderr}")
                    return False

            finally:
                if os.path.exists(script_path):
                    os.unlink(script_path)

        except Exception as e:
            print(f"Failed to download {model_name}: {e}")
            if progress_callback:
                progress_callback(f"Download failed for {model_name}: {e}")
            return False

    def delete_model(self, model_name):
        """Delete a downloaded model from cache.

        Args:
            model_name (str): Name of the model to delete

        Returns:
            bool: True if model was deleted successfully
        """
        try:
            model_info = self.models.get(model_name)
            if not model_info:
                return False

            repo_id = model_info.get("repo_id", "")
            repo_dir_name = "models--" + repo_id.replace("/", "--")
            model_dir = self.cache_dir / repo_dir_name

            if model_dir.exists():
                shutil.rmtree(model_dir)
                print(f"Deleted {model_name} model")
                return True
            return False
        except Exception as e:
            print(f"Failed to delete {model_name}: {e}")
            return False

    def get_cache_size(self):
        """Get total size of model cache in bytes.

        Returns:
            int: Total size of all cached model files in bytes
        """
        total_size = 0

        if self.cache_dir.exists():
            for model_info in self.models.values():
                repo_id = model_info.get("repo_id", "")
                repo_dir_name = "models--" + repo_id.replace("/", "--")
                model_dir = self.cache_dir / repo_dir_name
                if model_dir.exists():
                    for file_path in model_dir.rglob("*"):
                        if file_path.is_file():
                            try:
                                total_size += file_path.stat().st_size
                            except (OSError, FileNotFoundError):
                                continue

        return total_size

    def clear_cache(self):
        """Clear all downloaded whisper models from cache.

        Returns:
            bool: True if cache was cleared successfully
        """
        try:
            if self.cache_dir.exists():
                for model_info in self.models.values():
                    repo_id = model_info.get("repo_id", "")
                    repo_dir_name = "models--" + repo_id.replace("/", "--")
                    model_dir = self.cache_dir / repo_dir_name
                    if model_dir.exists():
                        try:
                            shutil.rmtree(model_dir)
                            print(f"Removed {repo_dir_name}")
                        except (OSError, FileNotFoundError):
                            continue

                print("Model cache cleared successfully")
                return True

            return True

        except Exception as e:
            print(f"Failed to clear cache: {e}")
            return False

    def repair_model(self, model_name, progress_callback=None):
        """Repair/re-download a corrupted model.

        Args:
            model_name (str): Name of the model to repair
            progress_callback (callable, optional): Function to call with progress updates

        Returns:
            bool: True if repair was successful
        """
        try:
            if progress_callback:
                progress_callback(f"Repairing {model_name} model...")

            # Delete existing file
            self.delete_model(model_name)

            # Re-download
            return self.download_single_model(model_name, progress_callback)

        except Exception as e:
            if progress_callback:
                progress_callback(f"Repair failed: {e}")
            return False

    def estimate_download_time(self, model_name, connection_speed_mbps=10):
        """Estimate download time for a model based on connection speed.

        Args:
            model_name (str): Name of the model
            connection_speed_mbps (float): Connection speed in Mbps

        Returns:
            str: Estimated download time as a human-readable string
        """
        model_info = self.models.get(model_name, {})
        size_str = model_info.get("size", "0 MB")

        try:
            size_mb = float(size_str.split()[0])
            size_bits = size_mb * 8  # Convert to megabits

            # Calculate time in seconds
            time_seconds = size_bits / connection_speed_mbps

            if time_seconds < 60:
                return f"{time_seconds:.0f} seconds"
            else:
                minutes = time_seconds / 60
                return f"{minutes:.1f} minutes"

        except:
            return "Unknown"

    def get_download_progress(self, model_name):
        """Get current download progress for a model.

        Args:
            model_name (str): Name of the model

        Returns:
            dict: Progress information with downloaded, total, and percent keys
        """
        return self.download_progress.get(
            model_name, {"downloaded": 0, "total": 0, "percent": 0}
        )

    def get_model_status_summary(self):
        """Get comprehensive summary of model download status.

        Returns:
            dict: Complete status information for all models
        """
        downloaded = self.check_downloaded_models()

        summary = {
            "total_models": len(self.models),
            "downloaded_count": len(downloaded),
            "downloaded_models": list(downloaded.keys()),
            "missing_models": [
                name for name in self.models.keys() if name not in downloaded
            ],
            "corrupted_models": [
                name for name, info in downloaded.items() if not info["valid"]
            ],
            "total_cache_size_mb": self.get_cache_size() / (1024 * 1024),
            "cache_directory": str(self.cache_dir),
            "model_details": {},
        }

        for model_name, model_info in self.models.items():
            is_downloaded = model_name in downloaded
            model_data = downloaded.get(model_name, {})

            summary["model_details"][model_name] = {
                "info": model_info,
                "downloaded": is_downloaded,
                "valid": model_data.get("valid", False) if is_downloaded else False,
                "file_size_mb": model_data.get("size_mb", 0) if is_downloaded else 0,
                "last_modified": (
                    model_data.get("last_modified", 0) if is_downloaded else 0
                ),
                "repo_id": model_info.get("repo_id", ""),
                "estimated_download_time": self.estimate_download_time(model_name),
            }

        return summary

    def export_model_info(self, file_path):
        """Export model information to JSON file.

        Args:
            file_path (str or Path): Path to save the export file

        Returns:
            bool: True if export was successful
        """
        try:
            summary = self.get_model_status_summary()

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"Failed to export model info: {e}")
            return False

    def get_recommended_model(self, file_size_mb=None, accuracy_priority="medium"):
        """Get recommended model based on file size and accuracy needs.

        Args:
            file_size_mb (float, optional): Size of input file in MB
            accuracy_priority (str): Priority level - 'speed', 'medium', or 'accuracy'

        Returns:
            str: Recommended model name
        """
        if accuracy_priority == "speed":
            return "base"
        elif accuracy_priority == "accuracy":
            return "large-v3"
        else:
            # Default: large-v3-turbo is the best speed/accuracy tradeoff
            return "large-v3-turbo"

    def batch_download_models(self, model_list, progress_callback=None):
        """Download multiple models in sequence.

        Args:
            model_list (list): List of model names to download
            progress_callback (callable, optional): Function to call with progress updates

        Returns:
            dict: Dictionary mapping model names to success status
        """
        results = {}
        total_models = len(model_list)

        for i, model_name in enumerate(model_list, 1):
            if progress_callback:
                progress_callback(f"Downloading model {i}/{total_models}: {model_name}")

            success = self.download_single_model(model_name, progress_callback)
            results[model_name] = success

            if not success and progress_callback:
                progress_callback(f"Failed to download {model_name}")

        successful_downloads = [model for model, success in results.items() if success]

        if progress_callback:
            progress_callback(
                f"Batch download complete: {len(successful_downloads)}/{total_models} successful"
            )

        return results

    def cleanup_temp_files(self):
        """Clean up any temporary files in the cache directory.

        Returns:
            bool: True if cleanup was successful
        """
        try:
            if not self.cache_dir.exists():
                return True

            # Remove temporary files
            temp_files = list(self.cache_dir.rglob("*.tmp"))
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                    print(f"Cleaned up temporary file: {temp_file.name}")
                except (OSError, FileNotFoundError):
                    continue

            return True

        except Exception as e:
            print(f"Failed to cleanup temp files: {e}")
            return False
