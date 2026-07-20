# Lesson 11: Testing, Debugging, Logging, and Exception Handling

## 1. Learning Objectives

By the end of this lesson, you should be able to:

1. Explain why automated tests matter.
2. Write unit tests using `pytest`.
3. Test expected exceptions.
4. Debug using tracebacks and `breakpoint()`.
5. Define custom exceptions.
6. Use structured logging.
7. Understand logging levels.
8. Avoid hiding important errors.
9. Test file-processing code safely.
10. Mock external dependencies.

---

## 2. Theory

A reliable application needs more than code that works once.

```text
Input
  |
  v
Validation
  |
  v
Application logic
  |
  v
Output
  |
  +-- Tests verify behavior
  +-- Logs record events
  +-- Exceptions report failures
  +-- Debugger investigates problems
```

### Unit test

Tests one small unit such as a function.

### Integration test

Tests multiple components together.

### Exception

Represents an abnormal condition.

### Logging

Records what happened while the application ran.

### Debugging

Investigates why actual behavior differs from expected behavior.

---

## 3. Visual Explanation

```text
Test input
    |
    v
Function under test
    |
    v
Actual output
    |
    v
Compare with expected output
    |
    +-- Match    → Pass
    |
    +-- Mismatch → Fail
```

Testing pyramid:

```text
             /\
            /  \  End-to-end tests
           /----\
          /      \ Integration tests
         /--------\
        /          \ Unit tests
       /____________\
```

Most tests should normally be fast unit tests.

---

## 4. Mathematical Intuition

Test pass rate:

[
\text{Pass rate}
================

\frac{\text{Passed tests}}
{\text{Total tests}}
\times 100
]

If 18 of 20 tests pass:

[
\frac{18}{20} \times 100 = 90%
]

A 100% test pass rate does not prove that software has no bugs. It only proves that all implemented tests passed.

Code coverage also does not prove correctness. A line can execute while producing an incorrect result.

---

## 5. Python Implementation

Install testing tools:

```bash
pip install pytest
```

Project structure:

```text
lesson-11-project/
├── src/
│   └── data_loader/
│       ├── __init__.py
│       ├── errors.py
│       └── loader.py
└── tests/
    └── test_loader.py
```

---

### `src/data_loader/errors.py`

```python
class DataPipelineError(Exception):
    """Base exception for data-pipeline failures."""


class InvalidDatasetError(DataPipelineError):
    """Raised when dataset contents are invalid."""


class MissingColumnsError(InvalidDatasetError):
    """Raised when required columns are missing."""
```

Custom exceptions communicate application meaning better than generic errors.

---

### `src/data_loader/loader.py`

```python
import logging
from pathlib import Path

import pandas as pd

from data_loader.errors import InvalidDatasetError, MissingColumnsError


LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "student_id",
    "name",
    "exam_score",
}


def load_student_csv(
    file_path: Path,
) -> pd.DataFrame:
    """Load and validate a student CSV file."""
    LOGGER.info("Loading student dataset from %s", file_path)

    if not file_path.exists():
        LOGGER.error("Dataset does not exist: %s", file_path)
        raise FileNotFoundError(
            f"Dataset does not exist: {file_path}"
        )

    if not file_path.is_file():
        raise InvalidDatasetError(
            f"Expected a file: {file_path}"
        )

    try:
        data = pd.read_csv(file_path)
    except pd.errors.EmptyDataError as error:
        LOGGER.exception("Dataset is empty.")
        raise InvalidDatasetError(
            "Dataset is empty."
        ) from error
    except pd.errors.ParserError as error:
        LOGGER.exception("Dataset contains invalid CSV.")
        raise InvalidDatasetError(
            "Dataset contains invalid CSV."
        ) from error

    missing_columns = REQUIRED_COLUMNS - set(data.columns)

    if missing_columns:
        LOGGER.warning(
            "Dataset is missing columns: %s",
            sorted(missing_columns),
        )

        raise MissingColumnsError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    data["exam_score"] = pd.to_numeric(
        data["exam_score"],
        errors="coerce",
    )

    invalid_scores = (
        data["exam_score"].isna()
        | (data["exam_score"] < 0)
        | (data["exam_score"] > 100)
    )

    if invalid_scores.any():
        invalid_count = int(invalid_scores.sum())

        LOGGER.warning(
            "Removing %d invalid score records.",
            invalid_count,
        )

        data = data[~invalid_scores].copy()

    data = data.drop_duplicates(
        subset=["student_id"],
        keep="last",
    )

    LOGGER.info(
        "Student dataset loaded successfully with %d rows.",
        len(data),
    )

    return data.reset_index(drop=True)
```

---

### Logging configuration

```python
import logging


def configure_logging() -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )
```

Logging levels:

```text
DEBUG    Detailed diagnostic information
INFO     Normal application progress
WARNING  Unexpected but recoverable situation
ERROR    Operation failed
CRITICAL Application may not continue
```

---

### `tests/test_loader.py`

```python
from pathlib import Path

import pandas as pd
import pytest

from data_loader.errors import MissingColumnsError
from data_loader.loader import load_student_csv


def test_load_valid_student_csv(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "students.csv"

    pd.DataFrame(
        {
            "student_id": [1, 2],
            "name": ["Ali", "Sara"],
            "exam_score": [75, 88],
        }
    ).to_csv(
        input_path,
        index=False,
    )

    result = load_student_csv(input_path)

    assert len(result) == 2
    assert result["exam_score"].tolist() == [75, 88]


def test_missing_file_raises_error(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_student_csv(missing_path)


def test_missing_columns_raise_error(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "students.csv"

    pd.DataFrame(
        {
            "student_id": [1],
            "name": ["Ali"],
        }
    ).to_csv(
        input_path,
        index=False,
    )

    with pytest.raises(MissingColumnsError):
        load_student_csv(input_path)


def test_invalid_scores_are_removed(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "students.csv"

    pd.DataFrame(
        {
            "student_id": [1, 2, 3],
            "name": ["Ali", "Sara", "Ahmed"],
            "exam_score": [75, 150, "invalid"],
        }
    ).to_csv(
        input_path,
        index=False,
    )

    result = load_student_csv(input_path)

    assert len(result) == 1
    assert result.iloc[0]["name"] == "Ali"


def test_duplicate_ids_keep_last_record(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "students.csv"

    pd.DataFrame(
        {
            "student_id": [1, 1],
            "name": ["Ali", "Ali"],
            "exam_score": [70, 90],
        }
    ).to_csv(
        input_path,
        index=False,
    )

    result = load_student_csv(input_path)

    assert len(result) == 1
    assert result.iloc[0]["exam_score"] == 90
```

Run tests:

```bash
pytest -q
```

Verbose output:

```bash
pytest -v
```

Run one test:

```bash
pytest tests/test_loader.py::test_invalid_scores_are_removed
```

---

### Debugging with `breakpoint()`

```python
def calculate_pass_rate(
    scores: list[float],
) -> float:
    breakpoint()

    passed = sum(
        score >= 60
        for score in scores
    )

    return passed / len(scores) * 100
```

Debugger commands:

```text
n → Next line
s → Step into function
c → Continue
p variable → Print variable
q → Quit debugger
```

Remove accidental breakpoints before production deployment.

---

### Reading tracebacks

Start from the final line:

```text
ZeroDivisionError: division by zero
```

Then move upward to find:

* The failing file
* The failing line
* The calling function
* The input that caused the error

---

### Time complexity

Testing does not change function complexity.

For the CSV loader:

* Reading (n) rows: (O(n))
* Validating scores: (O(n))
* Duplicate removal: approximately (O(n))

Overall:

[
O(n)
]

Running (t) tests that each process (n) records is approximately:

[
O(tn)
]

---

## 6. Exercises

1. Test an empty dataset.
2. Test a path that is a directory.
3. Test scores equal to exactly 0 and 100.
4. Test whitespace-only student names.
5. Add a warning when more than 20% of rows are removed.
6. Add a test confirming the warning is logged.
7. Use `breakpoint()` to inspect an invalid score.

---

## 7. Mini Project: Robust Product Data Loader

Build a tested loader that:

* Reads product CSV files.
* Requires product ID, title, price, and stock.
* Converts price and stock to numeric.
* Rejects negative values.
* Removes duplicate product IDs.
* Logs the raw and cleaned row counts.
* Defines custom exceptions.
* Includes at least ten tests.
* Produces no network calls during testing.

---

## 8. Common Mistakes

### Catching every exception

```python
try:
    process_data()
except Exception:
    pass
```

This hides errors and makes debugging difficult.

### Logging and raising duplicate messages

Avoid creating several identical log entries for one failure.

### Using `print()` for production diagnostics

`print()` lacks levels, timestamps, module names, and configurable destinations.

### Testing implementation rather than behavior

A test should verify output and observable behavior, not unnecessary internal details.

### Depending on test order

Every test should create its own state.

### Tests using live services

Network-dependent tests can become slow and unreliable.

---

## 9. Interview Questions

1. What is a unit test?
2. What is an integration test?
3. What is a test fixture?
4. What does `pytest.raises()` do?
5. What is mocking?
6. Why should tests be independent?
7. What is a traceback?
8. What does `breakpoint()` do?
9. What is exception chaining?
10. Why create custom exceptions?
11. What are the main logging levels?
12. What is the difference between warning and error?
13. Why is broad exception handling dangerous?
14. Does 100% test coverage prove correctness?
15. Why should API calls be mocked?

---

## 10. Summary

* Automated tests protect expected behavior.
* Unit tests should be small, fast, and independent.
* `pytest` provides fixtures and readable assertions.
* Custom exceptions communicate domain-specific failures.
* Tracebacks should be read from the final error upward.
* `breakpoint()` allows interactive debugging.
* Logging records structured runtime information.
* Specific exceptions should be caught instead of every exception.
* External services should be mocked in unit tests.
* Test pass rate and coverage do not guarantee complete correctness.

## Lesson 11 Quiz

1. What is the difference between a unit and integration test?
2. What does `pytest.raises()` test?
3. Why should each test create its own data?
4. What is a fixture?
5. What does mocking mean?
6. Why should live APIs normally not be used in unit tests?
7. What is the difference between `INFO` and `ERROR` logging?
8. What does `LOGGER.exception()` include?
9. What is exception chaining?
10. Why are custom exceptions useful?
11. What does `breakpoint()` do?
12. Why is `except Exception: pass` dangerous?
13. Does 100% code coverage prove correctness?
14. What is the approximate complexity of scanning (n) CSV rows?
15. Why should secrets never be written to logs?

---
