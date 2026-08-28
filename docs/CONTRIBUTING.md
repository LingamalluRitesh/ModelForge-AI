# Contributing to ModelForge AI

We welcome contributions from the open source AI and MLOps engineering community!

## 1. Development Workflow
1. Fork the repository and create your feature branch: `git checkout -b feature/amazing-feature`.
2. Ensure you have Python 3.12+ and Node.js 20+ installed.
3. Install backend dependencies: `pip install -r backend/requirements.txt`.
4. Install frontend dependencies: `cd frontend && npm install`.

## 2. Code Quality & Standards
- Python: Formatted with **Black** (line length 100) and checked with **Flake8** & **MyPy**.
- TypeScript: Strict type safety without `any` overrides where possible.
- Pytest: All new features require unit tests in `backend/tests/`.

## 3. Pull Request Guidelines
- Ensure all CI tests pass: `pytest backend/tests/`.
- Provide a clear PR description detailing feature motivation, architectural decisions, and screenshots for UI changes.
