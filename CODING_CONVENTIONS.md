Repository coding conventions

This project follows a small set of pragmatic coding conventions to keep the codebase
consistent and maintainable. Additions are welcome via PRs.

1) Imports
- All imports should be at the top of the module (top-level) unless there is a
  justified reason to import inside a function (rare): e.g. to avoid heavy
  optional dependency cost on import, or to avoid circular imports. If you add a
  function-local import, add a short comment explaining why.
- Standard library imports first, then third-party, then local imports. Use
  alphabetical order inside each group where convenient.

2) Exception handling
- Keep try/except blocks narrow: only wrap the minimum statements that can raise
  the error you're handling.
- Prefer to log exceptions with `exc_info=True` at DEBUG when the error is
  non-fatal and with `logger.exception()` for unexpected errors.

3) Logging
- Use module-level `logger = logging.getLogger(__name__)` in modules.
- Use INFO for high-level progress, DEBUG for verbose internal state, WARNING/ERROR
  for problems and exceptions.

4) Tests
- Keep unit tests fast and deterministic by mocking network calls.
- Where filesystem state is involved, use `tmp_path` fixtures and clean up any
  files you create.

5) Overrides & extension points
- Provide overridable instance methods for behaviors you expect child classes to
  change (eg. `_attempt_fn` in generators). Avoid passing callbacks where a
  method override is more natural.

6) PRs and style
- Run the test suite before opening a PR.
- Keep diffs focused and avoid large unrelated reformatting.

If you want, I can add a short pre-commit hook or a `CONTRIBUTING.md` entry to
remind contributors about these rules and check for inline imports automatically.
*** End Patch