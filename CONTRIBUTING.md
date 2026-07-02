# Contributing to Energivanu

Thank you for your interest in contributing! We welcome contributions from the community.

## How to Contribute

1. **Fork** the repository
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes**
4. **Run tests** (`pytest tests/`)
5. **Run linting** (`ruff check src/ tests/`)
6. **Commit** your changes (`git commit -m 'Add amazing feature'`)
7. **Push** to the branch (`git push origin feature/amazing-feature`)
8. **Open a Pull Request**

## Development Setup

```bash
git clone https://github.com/mysterious75/Energivanu.git
cd Energivanu
pip install -e ".[dev]"
pytest tests/
```

## Code Style

- Follow PEP 8
- Use type hints for all function signatures
- Write docstrings for public functions and classes
- Add tests for new features or bug fixes
- Keep functions focused and single-purpose

## Pull Request Guidelines

- Keep PRs focused on a single concern
- Update documentation if needed
- Ensure all tests pass
- Add or update tests to cover your changes
- Reference any related issues

## Reporting Bugs

Open a GitHub issue using the bug report template. Include:
- A clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, GPU model)

## Questions

Open a GitHub discussion or contact support@voraprotocol.com
