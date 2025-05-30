# Contributing to Whisper Transcriber Pro

We love your input! We want to make contributing to Whisper Transcriber Pro as easy and transparent as possible, whether it's:

- Reporting bugs
- Discussing the current state of the code
- Submitting feature requests
- Proposing code changes
- Improving documentation
- Adding translations
- Writing tests

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How to Contribute](#how-to-contribute)
4. [Development Process](#development-process)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation](#documentation)
8. [Community](#community)
9. [Security](#security)
10. [Recognition](#recognition)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers through GitHub issues.

### Our Standards

- **Be respectful** and inclusive
- **Be collaborative** and constructive
- **Be patient** with newcomers
- **Focus on what's best** for the community
- **Show empathy** towards other community members

## Getting Started

### Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed
- **Git** for version control
- **Basic knowledge** of Python and GUI development
- **Familiarity** with OpenAI Whisper (helpful but not required)

### Development Environment Setup

1. **Fork the repository**
   ```bash
   # Click the "Fork" button on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/whisper-transcriber-pro.git
   cd whisper-transcriber-pro
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv dev_env
   
   # Activate environment
   # Windows:
   dev_env\Scripts\activate
   # Linux/Mac:
   source dev_env/bin/activate
   
   # Install development dependencies
   pip install -r requirements-dev.txt
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

4. **Run tests**
   ```bash
   python -m pytest tests/
   ```

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check the [existing issues](../../issues) to avoid duplicates.

**When submitting a bug report, please include:**

- **Clear title** and description
- **Steps to reproduce** the behavior
- **Expected behavior** vs. actual behavior
- **Screenshots** or error messages (if applicable)
- **System information**:
  - OS and version
  - Python version
  - GPU information (if relevant)
  - Application version

**Bug Report Template:**
```markdown
**Bug Description**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
A clear description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**System Information:**
 - OS: [e.g. Windows 10, Ubuntu 20.04, macOS 12.0]
 - Python Version: [e.g. 3.9.7]
 - GPU: [e.g. NVIDIA RTX 3060, None]
 - App Version: [e.g. 1.0.0]

**Additional Context**
Add any other context about the problem here.
```

### Feature Requests

We welcome feature suggestions! Please:

1. **Check existing requests** in [GitHub Issues](../../issues)
2. **Describe the problem** you're trying to solve
3. **Explain your proposed solution**
4. **Consider the impact** on existing users
5. **Provide examples** of how the feature would be used

**Feature Request Template:**
```markdown
**Feature Summary**
Brief description of the feature you'd like to see.

**Problem Statement**
What problem does this feature solve? Who would benefit?

**Proposed Solution**
Detailed description of how you envision this working.

**Alternatives Considered**
What other approaches have you considered?

**Additional Context**
Add any other context, mockups, or examples here.
```

### Code Contributions

#### Before You Start
1. **Discuss major changes** in an issue first
2. **Check the roadmap** to align with project direction
3. **Ensure your idea fits** the project scope
4. **Consider backward compatibility**

#### Contribution Process

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes**
   - Follow our [coding standards](#coding-standards)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run all tests
   python -m pytest tests/
   
   # Run specific test file
   python -m pytest tests/test_specific_module.py
   
   # Run with coverage
   python -m pytest --cov=src tests/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new transcription feature"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

## Development Process

### Git Workflow

We use **Git Flow** with the following branches:

- **`main`** - Production-ready code
- **`develop`** - Integration branch for features
- **`feature/*`** - New features
- **`fix/*`** - Bug fixes
- **`release/*`** - Release preparation
- **`hotfix/*`** - Critical production fixes

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- **`feat`** - New feature
- **`fix`** - Bug fix
- **`docs`** - Documentation changes
- **`style`** - Code style changes (formatting, etc.)
- **`refactor`** - Code refactoring
- **`test`** - Adding or updating tests
- **`chore`** - Maintenance tasks

**Examples:**
```bash
feat(ui): add real-time progress bar
fix(gpu): resolve CUDA memory leak issue
docs(readme): update installation instructions
test(engine): add unit tests for transcription engine
```

### Pull Request Process

1. **Update documentation** for any public API changes
2. **Add tests** for new functionality
3. **Ensure all tests pass** on all supported platforms
4. **Update CHANGELOG.md** following [Keep a Changelog](https://keepachangelog.com/)
5. **Request review** from maintainers
6. **Address feedback** promptly

**PR Template:**
```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have tested on multiple platforms (if applicable)

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] Any dependent changes have been merged and published
```

## Coding Standards

### Python Style Guide

We follow **PEP 8** with some project-specific conventions:

```python
# Good
class TranscriptionEngine:
    """Handles audio transcription using Whisper models."""
    
    def __init__(self, model_size: str = "medium"):
        self.model_size = model_size
        self._model = None
    
    def transcribe(self, audio_path: str) -> dict:
        """Transcribe audio file to text.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing transcription results
        """
        # Implementation here
        pass
```

### Code Formatting

We use automated formatting tools:

```bash
# Install formatting tools
pip install black flake8 isort mypy

# Format code
black src/ tests/
isort src/ tests/

# Check style
flake8 src/ tests/

# Type checking
mypy src/
```

### Configuration Files

**`.flake8`**
```ini
[flake8]
max-line-length = 88
exclude = __pycache__,venv,env,whisper_env
ignore = E203,W503
```

**`pyproject.toml`**
```toml
[tool.black]
line-length = 88
target-version = ['py38']

[tool.isort]
profile = "black"
line_length = 88
```

### Naming Conventions

- **Files**: `lowercase_with_underscores.py`
- **Classes**: `CamelCase`
- **Functions/Variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `_leading_underscore`

### Documentation Standards

#### Docstring Format
We use **Google Style** docstrings:

```python
def transcribe_audio(file_path: str, model_size: str = "medium") -> dict:
    """Transcribe audio file using Whisper model.
    
    This function processes an audio file and returns a transcription
    with timestamps and confidence scores.
    
    Args:
        file_path: Path to the audio file to transcribe.
        model_size: Size of the Whisper model to use. Options are
            'tiny', 'base', 'small', 'medium', 'large'.
    
    Returns:
        A dictionary containing:
            - text: The transcribed text
            - segments: List of segments with timestamps
            - language: Detected language
    
    Raises:
        FileNotFoundError: If the audio file doesn't exist.
        ValueError: If the model_size is invalid.
    
    Example:
        >>> result = transcribe_audio("speech.mp3", "medium")
        >>> print(result["text"])
        "Hello, this is a test transcription."
    """
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/                  # Unit tests
│   ├── test_engine.py
│   ├── test_models.py
│   └── test_utils.py
├── integration/           # Integration tests
│   ├── test_gui.py
│   └── test_workflow.py
├── fixtures/             # Test data
│   ├── sample_audio.wav
│   └── expected_output.json
└── conftest.py          # Pytest configuration
```

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch
from src.transcription_engine import TranscriptionEngine

class TestTranscriptionEngine:
    """Test cases for TranscriptionEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create a TranscriptionEngine instance for testing."""
        return TranscriptionEngine(model_size="tiny")
    
    def test_initialization(self, engine):
        """Test engine initialization."""
        assert engine.model_size == "tiny"
        assert engine._model is None
    
    @patch('src.transcription_engine.whisper.load_model')
    def test_model_loading(self, mock_load_model, engine):
        """Test model loading functionality."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        engine.load_model()
        
        mock_load_model.assert_called_once_with("tiny")
        assert engine._model == mock_model
    
    def test_invalid_model_size(self):
        """Test initialization with invalid model size."""
        with pytest.raises(ValueError, match="Invalid model size"):
            TranscriptionEngine(model_size="invalid")
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/unit/test_engine.py

# Run with verbose output
python -m pytest -v

# Run tests matching pattern
python -m pytest -k "test_model"
```

### Test Requirements

- **Minimum 80% code coverage** for new code
- **All tests must pass** before merging
- **Use mocks** for external dependencies
- **Include edge cases** and error conditions
- **Test on multiple platforms** when applicable

## Documentation

### Types of Documentation

1. **Code Documentation** - Docstrings and comments
2. **User Documentation** - README, user guides
3. **Developer Documentation** - Architecture, API docs
4. **Process Documentation** - Contributing, deployment

### Documentation Standards

- **Keep it up-to-date** with code changes
- **Use clear, simple language**
- **Include examples** where helpful
- **Structure logically** with headers
- **Link related content**

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

## Community

### Communication Channels

- **GitHub Discussions** - General questions and ideas
- **GitHub Issues** - Bug reports and feature requests
- **Project Repository** - [Main repository](https://github.com/Black-Lights/whisper-transcriber-pro)

### Support Levels

#### Community Support
- **GitHub Discussions** - Community-driven Q&A
- **GitHub Issues** - Direct contact with maintainers

### Response Times

| Type | Expected Response Time |
|------|----------------------|
| Bug Reports | 3-5 business days |
| Feature Requests | 1-2 weeks |
| Questions | 2-3 business days |
| Pull Requests | 3-7 business days |

## Security

### Reporting Security Issues

**DO NOT** open public issues for security vulnerabilities.

Instead:
1. **Contact** the maintainer through GitHub
2. **Include** detailed description and reproduction steps
3. **Wait** for acknowledgment before public disclosure
4. **Coordinate** with maintainers on fix timeline

### Secure Development

- **Input validation** for all user-provided data
- **Dependency scanning** for known vulnerabilities
- **Code analysis** with security-focused tools
- **Regular updates** of dependencies
- **Principle of least privilege** in system access

## Recognition

### Contributor Acknowledgment

We value all contributions and recognize them in several ways:

- **GitHub Contributors** - Automatic recognition in repository
- **Changelog Credits** - Named in release notes
- **Maintainer Status** - Outstanding contributors may become maintainers

### Types of Contributions

All contributions are valuable, including:

- **Bug Reports** - Help us improve quality
- **Code** - Direct feature and fix contributions
- **Documentation** - Make the project more accessible
- **Translation** - Expand global reach
- **Testing** - Ensure reliability across platforms
- **Ideas** - Shape the future direction

## Best Practices

### For New Contributors

1. **Start small** - Begin with documentation or simple bug fixes
2. **Ask questions** - Don't hesitate to seek clarification
3. **Read the code** - Understand existing patterns and conventions
4. **Test thoroughly** - Ensure your changes work across platforms
5. **Be patient** - Code review and feedback take time

### For Experienced Contributors

1. **Mentor newcomers** - Help onboard new contributors
2. **Review thoughtfully** - Provide constructive, detailed feedback
3. **Share knowledge** - Contribute to documentation and discussions
4. **Think long-term** - Consider maintenance and backward compatibility
5. **Stay updated** - Keep up with project direction and best practices

## Contact Information

### Project Maintainer

- **Author** - Black-Lights ([GitHub Profile](https://github.com/Black-Lights))
- **Repository** - [whisper-transcriber-pro](https://github.com/Black-Lights/whisper-transcriber-pro)

---

## Thank You

Thank you for your interest in contributing to Whisper Transcriber Pro! Every contribution, no matter how small, makes a difference. Together, we're building a tool that makes audio and video content more accessible to everyone.

**Remember:**
- Every expert was once a beginner
- We're here to help you succeed  
- Your unique perspective adds value
- Great things are built by great communities

Ready to contribute? Check out our [issues](../../issues) and join the community!

---

*This contributing guide is a living document. If you have suggestions for improvements, please open an issue or submit a pull request.*