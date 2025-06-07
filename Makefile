# Makefile for Whisper Transcriber Pro v1.2.0
# Author: Black-Lights (https://github.com/Black-Lights)

.PHONY: help install install-dev test test-unit test-integration test-performance lint format type-check security clean build release

# Default target
help:
	@echo "Whisper Transcriber Pro v1.2.0 - Live Transcription Edition"
	@echo "============================================================="
	@echo ""
	@echo "Available commands:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-performance Run performance tests only"
	@echo "  lint             Run code linting (flake8)"
	@echo "  format           Format code (black + isort)"
	@echo "  format-check     Check code formatting"
	@echo "  type-check       Run type checking (mypy)"
	@echo "  security         Run security checks"
	@echo "  clean            Clean temporary files and caches"
	@echo "  build            Build distribution packages"
	@echo "  release          Create release packages"
	@echo "  setup-env        Set up development environment"
	@echo "  run              Run the application"
	@echo ""

# Installation targets
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

# Testing targets
test:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -v

test-unit:
	pytest tests/unit/ -v -m unit

test-integration:
	pytest tests/integration/ -v -m integration

test-performance:
	pytest tests/performance/ -v -m performance

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing --cov-fail-under=70

# Code quality targets
lint:
	flake8 src/ tests/ --statistics
	@echo "✅ Linting complete"

format:
	black src/ tests/
	isort src/ tests/
	@echo "✅ Code formatting complete"

format-check:
	black --check src/ tests/
	isort --check-only src/ tests/
	@echo "✅ Code formatting check complete"

type-check:
	mypy src/ --ignore-missing-imports
	@echo "✅ Type checking complete"

security:
	safety check
	bandit -r src/ -f json -o bandit-report.json
	@echo "✅ Security checks complete"

# Quality gate - run all checks
quality-check: format-check lint type-check security test-coverage
	@echo "✅ All quality checks passed!"

# Development environment setup
setup-env:
	python -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "  Windows: venv\\Scripts\\activate"
	@echo "  Linux/Mac: source venv/bin/activate"
	@echo "Then run: make install-dev"

# Application targets
run:
	python main.py

run-debug:
	python main.py --debug

# Build and release targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	find . -type f -name "temp_*.py" -delete
	find . -type f -name "temp_result*.json" -delete
	@echo "✅ Cleanup complete"

build: clean
	python setup.py sdist bdist_wheel
	@echo "✅ Distribution packages built"

release: quality-check
	python create_release.py
	@echo "✅ Release packages created in releases/ directory"

# Documentation targets
docs:
	@echo "Opening project documentation..."
	@echo "README.md - Main documentation"
	@echo "CHANGELOG.md - Version history"
	@echo "CONTRIBUTING.md - Contribution guidelines"

docs-serve:
	@echo "Documentation files:"
	@ls -la *.md

# Installation verification
verify-install:
	python -c "import src.transcription_engine; print('✅ Core modules import successfully')"
	python -c "import tkinter; print('✅ GUI framework available')"
	python -c "import torch; print(f'✅ PyTorch available: {torch.__version__}')"
	python -c "import whisper; print('✅ Whisper available')"
	python -c "import psutil; print(f'✅ Process management available: {psutil.__version__}')"
	@echo "✅ Installation verification complete"

# Performance benchmarks
benchmark:
	@echo "Running performance benchmarks..."
	python -c "
import time
import psutil
import sys
sys.path.append('src')
from transcription_engine import TranscriptionEngine
print(f'Memory usage: {psutil.virtual_memory().percent}%')
print(f'CPU count: {psutil.cpu_count()}')
print('✅ Basic benchmarks complete')
"

# Development workflow shortcuts
dev-setup: setup-env install-dev verify-install
	@echo "✅ Development environment ready!"
	@echo "Next steps:"
	@echo "1. Activate virtual environment"
	@echo "2. Run: make run"

# Pre-commit checks (run before committing)
pre-commit: format lint type-check test-unit
	@echo "✅ Pre-commit checks passed!"

# CI/CD simulation
ci: quality-check test
	@echo "✅ CI pipeline simulation complete!"

# Release preparation
pre-release: ci clean
	@echo "Running release preparation..."
	python -c "
import sys
sys.path.append('.')
from main import *
print('✅ Application can be imported')
"
	@echo "✅ Pre-release checks complete!"

# Docker targets (if Docker support is added)
docker-build:
	@echo "Docker support not yet implemented"

docker-run:
	@echo "Docker support not yet implemented"

# Platform-specific targets
windows-package: clean
	python create_release.py
	@echo "✅ Windows package created"

linux-package: clean
	python create_release.py
	@echo "✅ Linux package created"

# Debug and troubleshooting
debug-env:
	@echo "Environment debugging information:"
	@echo "Python version: $(shell python --version)"
	@echo "Pip version: $(shell pip --version)"
	@echo "Current directory: $(shell pwd)"
	@echo "Python path:"
	@python -c "import sys; [print(f'  {p}') for p in sys.path]"

debug-imports:
	@echo "Testing critical imports..."
	@python -c "
try:
    import tkinter
    print('✅ tkinter - GUI framework')
except ImportError as e:
    print(f'❌ tkinter - {e}')

try:
    import torch
    print(f'✅ torch - {torch.__version__}')
    if torch.cuda.is_available():
        print(f'  🎮 CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        print('  💻 CPU-only mode')
except ImportError as e:
    print(f'❌ torch - {e}')

try:
    import whisper
    print('✅ whisper - OpenAI Whisper')
except ImportError as e:
    print(f'❌ whisper - {e}')

try:
    import psutil
    print(f'✅ psutil - {psutil.__version__}')
except ImportError as e:
    print(f'❌ psutil - {e}')
"

# Version information
version:
	@python -c "
import re
with open('main.py', 'r') as f:
    content = f.read()
    match = re.search(r'v(\d+\.\d+\.\d+)', content)
    if match:
        print(f'Version: {match.group(1)}')
    else:
        print('Version: Unknown')
"

# Help for specific commands
help-test:
	@echo "Testing Commands:"
	@echo "  make test             - Run all tests with coverage"
	@echo "  make test-unit        - Run only unit tests"
	@echo "  make test-integration - Run only integration tests"
	@echo "  make test-performance - Run only performance tests"
	@echo "  make test-coverage    - Run tests with coverage requirements"

help-quality:
	@echo "Code Quality Commands:"
	@echo "  make lint        - Check code style with flake8"
	@echo "  make format      - Format code with black and isort"
	@echo "  make format-check- Check if code is properly formatted"
	@echo "  make type-check  - Run type checking with mypy"
	@echo "  make security    - Run security checks"
	@echo "  make quality-check - Run all quality checks"

help-release:
	@echo "Release Commands:"
	@echo "  make clean       - Remove build artifacts and caches"
	@echo "  make build       - Build Python distribution packages"
	@echo "  make release     - Create release packages for all platforms"
	@echo "  make pre-release - Run all checks before release"