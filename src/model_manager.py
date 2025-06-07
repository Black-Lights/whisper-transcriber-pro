"""
Model Manager - Handles Whisper model downloading and management
Author: Black-Lights (https://github.com/Black-Lights)
Project: Whisper Transcriber Pro

This module provides comprehensive management of OpenAI Whisper models including
downloading, verification, caching, and maintenance operations.
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


class ModelManager:
    """Manages OpenAI Whisper model downloading, verification, and caching.

    This class handles all aspects of Whisper model management including:
    - Downloading models from OpenAI servers
    - Verifying model integrity using SHA256 checksums
    - Caching models for offline use
    - Providing model information and recommendations
    - Repairing corrupted model files
    """

    def __init__(self):
        """Initialize the ModelManager with model definitions and cache directory."""
        self.models = {
            "tiny": {
                "size": "39 MB",
                "description": "Fastest, least accurate",
                "speed": "~32x real-time",
                "accuracy": "Basic",
                "url": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
                "sha256": "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9",
            },
            "base": {
                "size": "74 MB",
                "description": "Good speed/accuracy balance",
                "speed": "~16x real-time",
                "accuracy": "Good",
                "url": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
                "sha256": "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e",
            },
            "small": {
                "size": "244 MB",
                "description": "Better accuracy",
                "speed": "~6x real-time",
                "accuracy": "Better",
                "url": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
                "sha256": "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794",
            },
            "medium": {
                "size": "769 MB",
                "description": "High accuracy (recommended)",
                "speed": "~2x real-time",
                "accuracy": "High",
                "url": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
                "sha256": "345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1",
            },
            "large": {
                "size": "1550 MB",
                "description": "Best accuracy, slowest",
                "speed": "~1x real-time",
                "accuracy": "Maximum",
                "url": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v3.pt",
                "sha256": "e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a",
            },
        }

        # Model cache directory (uses Whisper's default location)
        self.cache_dir = Path.home() / ".cache" / "whisper"
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

        # Check for model files
        for model_name in self.models.keys():
            model_file = self.cache_dir / f"{model_name}.pt"
            if model_file.exists():
                size = model_file.stat().st_size

                # Verify file integrity
                is_valid = self.verify_model_file(model_name, model_file)

                downloaded[model_name] = {
                    "size_bytes": size,
                    "size_mb": size / (1024 * 1024),
                    "path": str(model_file),
                    "valid": is_valid,
                    "last_modified": model_file.stat().st_mtime,
                }

        return downloaded

    def verify_model_file(self, model_name, file_path):
        """Verify model file integrity using SHA256 checksum.

        Args:
            model_name (str): Name of the model to verify
            file_path (Path): Path to the model file

        Returns:
            bool: True if file is valid, False otherwise
        """
        try:
            expected_hash = self.models[model_name]["sha256"]

            # Calculate file hash
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)

            actual_hash = sha256_hash.hexdigest()
            return actual_hash == expected_hash

        except Exception as e:
            print(f"Error verifying {model_name}: {e}")
            return False

    def download_models(self, models=None, progress_callback=None):
        """Download specified models or default model.

        Args:
            models (list or str, optional): Model(s) to download. Defaults to ['medium'].
            progress_callback (callable, optional): Function to call with progress updates

        Returns:
            bool: True if at least one model was downloaded successfully
        """
        if models is None:
            models = ["medium"]  # Default to medium model

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
        """Download a single model with progress tracking and verification.

        Args:
            model_name (str): Name of the model to download
            progress_callback (callable, optional): Function to call with progress updates

        Returns:
            bool: True if download was successful, False otherwise
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Unknown model: {model_name}")

            model_info = self.models[model_name]
            url = model_info["url"]
            expected_hash = model_info["sha256"]

            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{model_name}.pt"
            file_path = self.cache_dir / filename
            temp_path = self.cache_dir / f"{filename}.tmp"

            # Check if model already exists and is valid
            if file_path.exists():
                if self.verify_model_file(model_name, file_path):
                    if progress_callback:
                        progress_callback(
                            f"Model {model_name} already exists and is valid"
                        )
                    return True
                else:
                    print(
                        f"Existing {model_name} model is corrupted, re-downloading..."
                    )
                    file_path.unlink()

            # Download with progress tracking
            if progress_callback:
                progress_callback(f"Downloading {model_name} model...")

            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            # Reset progress tracking
            self.download_progress[model_name] = {
                "downloaded": 0,
                "total": total_size,
                "percent": 0,
            }

            start_time = time.time()

            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Update progress
                        self.download_progress[model_name]["downloaded"] = downloaded
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            self.download_progress[model_name]["percent"] = percent

                            # Calculate download speed
                            elapsed = time.time() - start_time
                            if elapsed > 0:
                                speed_mbps = (downloaded / (1024 * 1024)) / elapsed

                                if progress_callback:
                                    size_mb = downloaded / (1024 * 1024)
                                    total_mb = total_size / (1024 * 1024)
                                    progress_callback(
                                        f"Downloading {model_name}: {size_mb:.1f}/{total_mb:.1f} MB ({percent:.1f}%) - {speed_mbps:.1f} MB/s"
                                    )

            # Verify downloaded file
            if progress_callback:
                progress_callback(f"Verifying {model_name} model integrity...")

            if not self.verify_model_file_by_path(temp_path, expected_hash):
                temp_path.unlink()
                raise Exception(f"Downloaded {model_name} model failed verification")

            # Move to final location
            temp_path.rename(file_path)

            if progress_callback:
                progress_callback(
                    f"Model {model_name} downloaded and verified successfully"
                )

            return True

        except Exception as e:
            print(f"Failed to download {model_name}: {e}")
            if progress_callback:
                progress_callback(f"Download failed for {model_name}: {e}")

            # Clean up temp file
            temp_path = self.cache_dir / f"{model_name}.pt.tmp"
            if temp_path.exists():
                temp_path.unlink()

            return False

    def verify_model_file_by_path(self, file_path, expected_hash):
        """Verify file integrity by path and expected hash.

        Args:
            file_path (Path): Path to file to verify
            expected_hash (str): Expected SHA256 hash

        Returns:
            bool: True if file matches expected hash
        """
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)

            actual_hash = sha256_hash.hexdigest()
            return actual_hash == expected_hash

        except Exception:
            return False

    def download_model_whisper_native(self, model_name, progress_callback=None):
        """Download model using Whisper's built-in download mechanism.

        Args:
            model_name (str): Name of the model to download
            progress_callback (callable, optional): Function to call with progress updates

        Returns:
            bool: True if download was successful
        """
        try:
            # Create a temporary script to download the model
            download_script = f"""
import whisper
import sys
import os

try:
    print(f"Loading {model_name} model...")
    model = whisper.load_model("{model_name}")
    print("Model loaded successfully!")
    print(f"Model device: {{model.device}}")
    print("Download completed successfully!")
except Exception as e:
    print(f"Download failed: {{e}}", file=sys.stderr)
    sys.exit(1)
"""

            # Write script to temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(download_script)
                script_path = f.name

            try:
                if progress_callback:
                    progress_callback(
                        f"Using Whisper native download for {model_name}..."
                    )

                # Run the download script
                result = subprocess.run(
                    ["python", script_path],
                    capture_output=True,
                    text=True,
                    timeout=1800,
                )  # 30 min timeout

                if result.returncode == 0:
                    if progress_callback:
                        progress_callback(
                            f"Model {model_name} downloaded successfully via Whisper"
                        )
                    return True
                else:
                    print(f"Download error: {result.stderr}")
                    if progress_callback:
                        progress_callback(f"Whisper download failed: {result.stderr}")
                    return False

            finally:
                # Clean up script file
                if os.path.exists(script_path):
                    os.unlink(script_path)

        except Exception as e:
            print(f"Failed to download {model_name} via Whisper: {e}")
            if progress_callback:
                progress_callback(f"Whisper download error: {e}")
            return False

    def delete_model(self, model_name):
        """Delete a downloaded model from cache.

        Args:
            model_name (str): Name of the model to delete

        Returns:
            bool: True if model was deleted successfully
        """
        try:
            model_file = self.cache_dir / f"{model_name}.pt"
            if model_file.exists():
                model_file.unlink()
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
            for file_path in self.cache_dir.rglob("*.pt"):
                try:
                    total_size += file_path.stat().st_size
                except (OSError, FileNotFoundError):
                    # Handle case where file is deleted during iteration
                    continue

        return total_size

    def clear_cache(self):
        """Clear all downloaded models from cache.

        Returns:
            bool: True if cache was cleared successfully
        """
        try:
            if self.cache_dir.exists():
                # Remove all .pt files
                for file_path in self.cache_dir.glob("*.pt"):
                    try:
                        file_path.unlink()
                        print(f"Removed {file_path.name}")
                    except (OSError, FileNotFoundError):
                        # File might have been deleted already
                        continue

                # Remove any temporary files
                for file_path in self.cache_dir.glob("*.tmp"):
                    try:
                        file_path.unlink()
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
                "download_url": model_info["url"],
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
            return "large"
        else:
            # Default balanced recommendation
            if file_size_mb is None:
                return "medium"

            # Size-based recommendations
            if file_size_mb < 10:  # Small files
                return "small"
            elif file_size_mb < 100:  # Medium files
                return "medium"
            else:  # Large files
                return "medium"  # Still good balance for large files

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
            temp_files = list(self.cache_dir.glob("*.tmp"))
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
