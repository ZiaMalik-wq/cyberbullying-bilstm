---
trigger: always_on
---

# Python Rules

Use Python 3.

Follow PEP 8 where practical.

Use clear, descriptive names.

Prefer small functions with one responsibility.

Use type hints where useful.

Use docstrings for reusable public functions and classes.

Use pathlib for filesystem paths.

Avoid:
- wildcard imports
- global mutable state
- magic numbers
- duplicated code
- hard-coded paths
- unnecessary abstraction

Configuration should be separated from implementation.

Reusable project logic belongs in `src/`.

Notebooks should not contain the only implementation of important logic.

Raise informative exceptions for:
- missing files
- missing columns
- invalid labels
- invalid configuration
- incompatible shapes

Do not silently suppress errors.

Do not use broad exception handling unless there is a specific reason.

Do not hide failed experiments.

Use logging where appropriate for reusable code.