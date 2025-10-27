## Development environment & code style

Formatting and import cleanup are enforced as default project rules. The following tools are used:
- Black: code formatter
- Ruff: linter and auto-fixes (including import cleanup)
- isort (optional): import sorting (can be covered by Ruff/Black configuration)

Recommended development setup (VS Code)
1. Install these extensions:
   - Python (ms-python.python)
   - Ruff (charliermarsh.ruff) — linter and auto-fix support

2. Recommended VS Code settings (add to workspace or user settings)
```json
{
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true,
    "source.fixAll": true
  },
  "[python]": {
    "editor.defaultFormatter": "ms-python.python"
  },
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "88"]
}