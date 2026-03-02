# AQSE_v3: Unit Tests + GitHub CI + Coverage Badge + Merge Protection

This guide describes how unit tests, CI, coverage, and merge protection are set up **specifically for the `AQSE_v3` workflow** in this repository.

It covers:

1. Unit tests (current layout and how to extend)
2. GitHub Actions CI pipeline (using **UV**)
3. Coverage reporting + badge in `AQSE_v3/README.md`
4. Merge protection for `main` (only merge when tests pass)

All examples assume:

- Python **3.12** for `AQSE_v3`
- Dependency management via **UV**
- Core package: `aqse_modelling`

---

## 1. Current Test Layout (AQSE_v3)

### Directory structure

```text
AQSE_v3/
  aqse_modelling/
    ...
  tests/
    conftest.py
    test_config_and_utils.py
  pyproject.toml
  README.md
```

- `aqse_modelling/` contains the main library code (data loaders, models, utils, reporting).
- `tests/` contains unit tests for core utilities and helpers.

### Import configuration for tests

`tests/conftest.py` ensures `aqse_modelling` is importable during test runs by putting the `AQSE_v3` root on `sys.path`:

```python
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

### Existing tests

`tests/test_config_and_utils.py` currently covers:

- **`config_loader`**:
  - Resolving relative paths in `config.yaml` to absolute paths.
  - Precedence between CLI config argument and `CONFIG_FILE` env var.
- **`physicochemical_descriptors`**:
  - Ensuring basic descriptor keys are present and numeric.
  - Ensuring `validate_descriptors` returns the expected boolean flags.
- **`data_splitting`**:
  - `split_data_stratified` with `use_fixed_test=True`:
    - Produces non-empty train/val/test sets.
    - Ensures no overlap between sets, and no rows are lost or duplicated.

You can add more tests by creating additional files like `tests/test_data_loaders.py`, `tests/test_random_forest_trainer.py`, etc.

---

## 2. Running Tests Locally (UV + pytest)

From the repo root:

```bash
cd AQSE_v3

# One-time setup or when dependencies change
uv sync

# Run tests
uv run pytest
```

This uses the `test` dependency group in `pyproject.toml`, which includes `pytest` and `pytest-cov`.

### Running with coverage

Coverage is collected for the `aqse_modelling` package:

```bash
uv run pytest --cov=aqse_modelling --cov-report=term-missing --cov-report=xml
```

- `--cov=aqse_modelling` limits coverage to the core library.
- `--cov-report=xml` creates `coverage.xml` in `AQSE_v3/`, which is used by Codecov.
- No `--cov-fail-under` threshold is enforced yet; you can add it later once coverage stabilizes.

Example with a quality gate (optional, not currently enabled):

```bash
uv run pytest --cov=aqse_modelling --cov-report=term-missing --cov-report=xml --cov-fail-under=70
```

---

## 3. GitHub Actions CI for AQSE_v3 (Using UV)

The CI workflow is defined at the repository root:

```text
.github/
  workflows/
    ci.yml
```

Key parts of `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  tests:
    runs-on: ubuntu-latest

    defaults:
      run:
        working-directory: AQSE_v3

    steps:
      - name: Checkout
        uses: actions/checkout@v5

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install uv
        run: |
          curl -LsSf https://astral.sh/uv/install.sh | sh
          echo "${HOME}/.local/bin" >> $GITHUB_PATH

      - name: Sync dependencies
        run: uv sync --all-extras --dev

      - name: Run tests with coverage
        run: uv run pytest --cov=aqse_modelling --cov-report=term-missing --cov-report=xml

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v5
        with:
          files: ./coverage.xml
          fail_ci_if_error: false
        env:
          CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }}
```

Notes:

- The job is named **`tests`** (used later in branch protection).
- CI runs from within the `AQSE_v3` directory via `defaults.run.working-directory`.
- `coverage.xml` is generated and then uploaded to Codecov.

If you later want to enforce a coverage minimum in CI, you can add `--cov-fail-under=<N>` to the pytest command here.

---

## 4. Coverage Badge in `AQSE_v3/README.md`

After Codecov is set up for your GitHub repository, a badge has been added near the top of `AQSE_v3/README.md`:

```md
[![codecov](https://codecov.io/gh/<OWNER>/<REPO>/branch/main/graph/badge.svg)](https://codecov.io/gh/<OWNER>/<REPO>)
```

Replace `<OWNER>` and `<REPO>` with your GitHub org/user and repository name.

If you prefer a static badge (manual updates), you can use:

```md
[![Coverage](https://img.shields.io/badge/coverage-70%25-yellow)](https://github.com/<OWNER>/<REPO>)
```

However, for `AQSE_v3` the Codecov badge is recommended and already wired into CI.

---

## 5. Merge Protection: Only Merge if Tests Pass

To enforce that `main` only receives changes when tests pass:

1. In GitHub, open the repository’s **Settings**.
2. Go to **Branches** (or **Rulesets**, depending on your UI).
3. Add or edit a branch protection rule for `main`.
4. Enable **“Require a pull request before merging”**.
5. Enable **“Require status checks to pass before merging”**.
6. In the list of checks, select the CI job named **`tests`** (from `.github/workflows/ci.yml`).

Optional hardening:

- Enable **“Require branches to be up to date before merging”**.
- Enable **“Require conversation resolution before merging”**.
- Optionally disable **bypass for admins** if you want the rules to apply to everyone.

Important: the `tests` check only appears in the list after the workflow has run at least once on `main` or a PR targeting `main`.

---

## 6. Quick End-to-End Verification Flow

Use this flow to verify that tests, CI, coverage, and merge protection work together:

1. Create a new branch from `main`:
   - `git checkout -b feature/ci-verification`
2. Make a small change in `AQSE_v3` and open a PR against `main`.
3. Confirm that the **CI / tests** workflow runs on the PR.
4. Introduce a failing test locally (e.g. change an assertion), push, and verify in GitHub that:
   - The `tests` job fails.
   - The PR shows failing checks and **cannot be merged** (due to branch protection).
5. Fix the failing test, push again, and verify that:
   - The `tests` job passes.
   - The PR becomes mergeable.
6. Visit Codecov for your repo and confirm:
   - The latest coverage is shown.
   - The badge in `AQSE_v3/README.md` reflects the coverage status for `main`.

At this point, `AQSE_v3` has a fully working baseline for:

- Unit tests (with a clear path to expand coverage),
- Automated CI on every push/PR to `main`,
- Coverage reporting via Codecov,
- And protected merges to `main` gated on passing tests.

