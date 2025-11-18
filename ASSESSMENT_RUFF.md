### Linting Issues Found

**Total Errors: 5 (all auto-fixable)**

#### app/services/engine.py (2 errors)
- Line 11: F401 - `tempfile` imported but unused
- Line 12: F401 - `os` imported but unused

#### main.py (3 errors)
- Line 10: F401 - `tempfile` imported but unused
- Line 16: F401 - `ErrorResponse` imported but unused
- Line 143: F541 - f-string without any placeholders

**STATUS: NOT RUFF COMPLIANT**
**Fix: Run `ruff check --fix .` to auto-fix all issues**
