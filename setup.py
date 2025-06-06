#!/usr/bin/env python3
"""
Setup script for Whisper Transcriber Pro
Author: Black-Lights (https://github.com/Black-Lights)
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read version from main.py
version = "1.2.0"
try:
    with open(this_directory / "main.py", 'r', encoding='utf-8') as f:
        content = f.read()
        for line in content.split('\n'):
            if 'Whisper Transcriber Pro v' in line and 'title' in line:
                # Extract version from title line
                import re
                match = re.search(r'v(\d+\.\d+\.\d+)', line)
                if match:
                    version = match.group(1)
                break
except:
    pass

setup(
    name="whisper-transcriber-pro",
    version=version,
    author="Black-Lights",
    author_email="",
    description="Professional AI-powered audio and video transcription with live preview",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Black-Lights/whisper-transcriber-pro",
    project_urls={
        "Bug Reports": "https://github.com/Black-Lights/whisper-transcriber-pro/issues",
        "Source": "https://github.com/Black-Lights/whisper-transcriber-pro",
        "Documentation": "https://github.com/Black-Lights/whisper-transcriber-pro/wiki",
    },
    packages=find_packages(),
    package_dir={"": "."},
    py_modules=["main", "install"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
        ],
        "performance": [
            "memory-profiler>=0.61.0",
            "librosa>=0.10.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "whisper-transcriber-pro=main:main",
            "wtp=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml", "*.ini"],
        "src": ["*.py"],
        "screenshots": ["*.png", "*.jpg"],
    },
    zip_safe=False,
    keywords=[
        "whisper", "transcription", "speech-to-text", "ai", "audio", "video",
        "subtitles", "srt", "vtt", "live-transcription", "gpu-acceleration",
        "openai", "machine-learning", "speech-recognition"
    ],
)