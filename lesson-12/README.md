# Lesson 12: Phase 1 Capstone — Production-Style Student Data Pipeline

## 1. Learning Objectives

By the end of this lesson, you should be able to:

1. Combine Python, NumPy, Pandas, visualization, files, and APIs.
2. Build a modular `src/` project.
3. Load data from CSV or an API.
4. Clean and validate tabular data.
5. Calculate analytical metrics.
6. Save reports and charts.
7. Add a command-line interface.
8. Write automated tests.
9. Containerize the application.
10. Explain how a company would operate the system.

---

## 2. Theory

The capstone application will:

1. Load student data from CSV or API.
2. Validate required columns.
3. Clean invalid records.
4. Calculate performance metrics.
5. Classify passing students.
6. Save cleaned data.
7. Save JSON metrics.
8. Generate charts.
9. Run through a CLI.
10. Execute inside Docker.

Required columns:

```text
student_id
name
study_hours
attendance
exam_score
```

---

## 3. Visual Explanation

```text
CSV file or API
       |
       v
Input loader
       |
       v
Schema validation
       |
       v
Data cleaning
       |
       v
NumPy/Pandas analysis
       |
       +----------------------+
       |          |           |
       v          v           v
Clean CSV    JSON metrics   Charts
       |
       v
Logs and tests
       |
       v
Docker container
```

Project structure:

```text
phase-1-capstone/
├── Dockerfile
├── README.md
├── pyproject.toml
├── data/
│   └── raw/
│       └── students.csv
├── src/
│   └── student_pipeline/
│       ├── __init__.py
│       ├── analysis.py
│       ├── cleaning.py
│       ├── cli.py
│       ├── loaders.py
│       ├── pipeline.py
│       └── visualization.py
└── tests/
    ├── test_analysis.py
    ├── test_cleaning.py
    └── test_pipeline.py
```

---

## 4. Mathematical Intuition

### Average score

[
\bar{x} = \frac{\sum x_i}{n}
]

### Median score

The middle score after sorting.

### Pass rate

[
\text{Pass rate}
================

\frac{\text{Students with score} \geq t}
{\text{Total students}}
\times 100
]

where (t) is the passing threshold.

### Data retention

[
\text{Retention rate}
=====================

\frac{\text{Clean rows}}
{\text{Raw rows}}
\times 100
]

If 90 of 100 rows survive cleaning:

[
\text{Retention rate} = 90%
]

A low retention rate indicates poor source data or overly aggressive cleaning rules.

---

## 5. Python Implementation

### `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "student-performance-pipeline"
version = "0.1.0"
description = "Phase 1 AI and ML foundations capstone"
requires-python = ">=3.11"
dependencies = [
    "matplotlib>=3.9,<4.0",
    "numpy>=2.0,<3.0",
    "pandas>=2.2,<3.0",
    "requests>=2.32,<3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0,<9.0",
]

[project.scripts]
student-pipeline = "student_pipeline.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
```

Install:

```bash
pip install -e ".[dev]"
```

---

### `src/student_pipeline/loaders.py`

```python
from pathlib import Path
from typing import Any

import pandas as pd
import requests


def load_csv(
    file_path: Path,
) -> pd.DataFrame:
    """Load student records from a CSV file."""
    if not file_path.is_file():
        raise FileNotFoundError(
            f"CSV file does not exist: {file_path}"
        )

    try:
        return pd.read_csv(file_path)
    except pd.errors.EmptyDataError as error:
        raise ValueError("CSV file is empty.") from error
    except pd.errors.ParserError as error:
        raise ValueError("CSV file is malformed.") from error


def load_api(
    url: str,
    timeout: float = 10.0,
) -> pd.DataFrame:
    """Load student records from a JSON API."""
    response = requests.get(
        url,
        timeout=timeout,
    )

    response.raise_for_status()

    payload: Any = response.json()

    if isinstance(payload, dict):
        records = payload.get("items")
    else:
        records = payload

    if not isinstance(records, list):
        raise ValueError(
            "API response must be a list or contain an items list."
        )

    return pd.DataFrame(records)
```

---

### `src/student_pipeline/cleaning.py`

```python
import pandas as pd


REQUIRED_COLUMNS = {
    "student_id",
    "name",
    "study_hours",
    "attendance",
    "exam_score",
}


def clean_student_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and clean raw student records."""
    if data.empty:
        raise ValueError("Student dataset cannot be empty.")

    missing_columns = REQUIRED_COLUMNS - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    cleaned = data.copy()

    cleaned["name"] = (
        cleaned["name"]
        .replace(r"^\s*$", pd.NA, regex=True)
        .str.strip()
        .str.title()
    )

    numeric_columns = [
        "student_id",
        "study_hours",
        "attendance",
        "exam_score",
    ]

    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    cleaned = cleaned.dropna(
        subset=[
            "student_id",
            "name",
            "study_hours",
            "attendance",
            "exam_score",
        ]
    )

    valid_rows = (
        (cleaned["study_hours"] >= 0)
        & cleaned["attendance"].between(0, 100)
        & cleaned["exam_score"].between(0, 100)
    )

    cleaned = cleaned[valid_rows].copy()

    cleaned = cleaned.drop_duplicates(
        subset=["student_id"],
        keep="last",
    )

    cleaned["student_id"] = cleaned[
        "student_id"
    ].astype(int)

    return cleaned.reset_index(drop=True)
```

---

### `src/student_pipeline/analysis.py`

```python
from typing import Any

import numpy as np
import pandas as pd


def analyze_student_data(
    data: pd.DataFrame,
    passing_score: float = 60.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add predictions and calculate student metrics."""
    if not 0 <= passing_score <= 100:
        raise ValueError(
            "Passing score must be between 0 and 100."
        )

    analyzed = data.copy()

    analyzed["passed"] = (
        analyzed["exam_score"] >= passing_score
    )

    scores = analyzed["exam_score"].to_numpy(
        dtype=np.float64
    )

    study_hours = analyzed[
        "study_hours"
    ].to_numpy(
        dtype=np.float64
    )

    top_index = analyzed["exam_score"].idxmax()

    metrics: dict[str, Any] = {
        "student_count": int(len(analyzed)),
        "average_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "minimum_score": float(np.min(scores)),
        "maximum_score": float(np.max(scores)),
        "average_study_hours": float(
            np.mean(study_hours)
        ),
        "pass_rate": float(
            np.mean(
                analyzed["passed"].to_numpy(
                    dtype=np.float64
                )
            )
            * 100
        ),
        "top_student": str(
            analyzed.loc[top_index, "name"]
        ),
    }

    return analyzed, metrics
```

---

### `src/student_pipeline/visualization.py`

```python
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def save_score_histogram(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save the exam-score distribution."""
    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.hist(
        data["exam_score"],
        bins=8,
        edgecolor="black",
    )

    axis.set_title("Exam Score Distribution")
    axis.set_xlabel("Exam Score")
    axis.set_ylabel("Student Count")
    axis.grid(axis="y")

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_study_score_scatter(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save study-hours and exam-score relationship."""
    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.scatter(
        data["study_hours"],
        data["exam_score"],
    )

    axis.set_title("Study Hours vs Exam Score")
    axis.set_xlabel("Study Hours")
    axis.set_ylabel("Exam Score")
    axis.grid(True)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)
```

`matplotlib.use("Agg")` allows chart generation in containers and servers without a graphical desktop.

---

### `src/student_pipeline/pipeline.py`

```python
import json
from pathlib import Path
from typing import Any

import pandas as pd

from student_pipeline.analysis import analyze_student_data
from student_pipeline.cleaning import clean_student_data
from student_pipeline.visualization import (
    save_score_histogram,
    save_study_score_scatter,
)


def run_pipeline(
    raw_data: pd.DataFrame,
    output_directory: Path,
    passing_score: float = 60.0,
) -> dict[str, Any]:
    """Run the complete student-data pipeline."""
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_row_count = len(raw_data)

    cleaned = clean_student_data(raw_data)

    analyzed, metrics = analyze_student_data(
        cleaned,
        passing_score=passing_score,
    )

    clean_row_count = len(analyzed)

    metrics["raw_row_count"] = raw_row_count
    metrics["clean_row_count"] = clean_row_count
    metrics["removed_row_count"] = (
        raw_row_count - clean_row_count
    )

    metrics["retention_rate"] = float(
        clean_row_count / raw_row_count * 100
    )

    analyzed.to_csv(
        output_directory / "students_cleaned.csv",
        index=False,
    )

    with (
        output_directory / "metrics.json"
    ).open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    save_score_histogram(
        analyzed,
        output_directory / "score_distribution.png",
    )

    save_study_score_scatter(
        analyzed,
        output_directory / "study_score_relationship.png",
    )

    return metrics
```

---

### `src/student_pipeline/cli.py`

```python
import argparse
import logging
from pathlib import Path

from student_pipeline.loaders import load_api, load_csv
from student_pipeline.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Clean and analyze student-performance data."
        )
    )

    source_group = parser.add_mutually_exclusive_group(
        required=True
    )

    source_group.add_argument(
        "--input-csv",
        type=Path,
    )

    source_group.add_argument(
        "--api-url",
        type=str,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("reports"),
    )

    parser.add_argument(
        "--passing-score",
        type=float,
        default=60.0,
    )

    return parser


def main() -> None:
    """Run the command-line application."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )

    parser = build_parser()
    arguments = parser.parse_args()

    if arguments.input_csv:
        raw_data = load_csv(
            arguments.input_csv
        )
    else:
        raw_data = load_api(
            arguments.api_url
        )

    metrics = run_pipeline(
        raw_data=raw_data,
        output_directory=arguments.output_directory,
        passing_score=arguments.passing_score,
    )

    logging.info(
        "Pipeline completed for %d students.",
        metrics["student_count"],
    )

    logging.info(
        "Pass rate: %.2f%%",
        metrics["pass_rate"],
    )


if __name__ == "__main__":
    main()
```

---

### Example input CSV

```csv
student_id,name,study_hours,attendance,exam_score
1,Ali,2.5,70,58
2,Sara,5.5,88,84
3,Ahmed,3.0,75,65
4,Fatima,7.0,95,94
5,Omar,invalid,55,42
6,Ayesha,6.0,90,88
6,Ayesha,6.5,92,91
7,,4.0,80,72
8,Hassan,4.5,120,79
```

Run:

```bash
student-pipeline \
  --input-csv data/raw/students.csv \
  --output-directory reports \
  --passing-score 60
```

Windows PowerShell:

```powershell
student-pipeline `
  --input-csv data/raw/students.csv `
  --output-directory reports `
  --passing-score 60
```

---

### Tests

#### `tests/test_cleaning.py`

```python
import pandas as pd

from student_pipeline.cleaning import clean_student_data


def test_clean_student_data() -> None:
    raw_data = pd.DataFrame(
        {
            "student_id": [1, 2, 2, 3],
            "name": [" Ali ", "Sara", "Sara", ""],
            "study_hours": [2.0, 5.0, 6.0, 3.0],
            "attendance": [70, 85, 90, 75],
            "exam_score": [60, 80, 90, 70],
        }
    )

    cleaned = clean_student_data(raw_data)

    assert len(cleaned) == 2
    assert cleaned["student_id"].is_unique
    assert cleaned.loc[0, "name"] == "Ali"
    assert cleaned.loc[1, "exam_score"] == 90
```

#### `tests/test_analysis.py`

```python
import pandas as pd
import pytest

from student_pipeline.analysis import analyze_student_data


def test_analyze_student_data() -> None:
    data = pd.DataFrame(
        {
            "student_id": [1, 2],
            "name": ["Ali", "Sara"],
            "study_hours": [2.0, 6.0],
            "attendance": [70.0, 90.0],
            "exam_score": [50.0, 90.0],
        }
    )

    analyzed, metrics = analyze_student_data(
        data,
        passing_score=60,
    )

    assert analyzed["passed"].tolist() == [
        False,
        True,
    ]

    assert metrics["average_score"] == 70.0
    assert metrics["pass_rate"] == 50.0
    assert metrics["top_student"] == "Sara"


def test_invalid_passing_score() -> None:
    data = pd.DataFrame()

    with pytest.raises(ValueError):
        analyze_student_data(
            data,
            passing_score=120,
        )
```

#### `tests/test_pipeline.py`

```python
from pathlib import Path

import pandas as pd

from student_pipeline.pipeline import run_pipeline


def test_pipeline_creates_outputs(
    tmp_path: Path,
) -> None:
    raw_data = pd.DataFrame(
        {
            "student_id": [1, 2],
            "name": ["Ali", "Sara"],
            "study_hours": [2.0, 6.0],
            "attendance": [70.0, 90.0],
            "exam_score": [50.0, 90.0],
        }
    )

    metrics = run_pipeline(
        raw_data=raw_data,
        output_directory=tmp_path,
    )

    assert metrics["student_count"] == 2

    assert (
        tmp_path / "students_cleaned.csv"
    ).exists()

    assert (
        tmp_path / "metrics.json"
    ).exists()

    assert (
        tmp_path / "score_distribution.png"
    ).exists()

    assert (
        tmp_path / "study_score_relationship.png"
    ).exists()
```

Run:

```bash
pytest -q
```

---

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

ENTRYPOINT ["student-pipeline"]
```

Build:

```bash
docker build -t student-pipeline:0.1.0 .
```

Run with a mounted data directory:

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/reports:/app/reports" \
  student-pipeline:0.1.0 \
  --input-csv /app/data/raw/students.csv \
  --output-directory /app/reports
```

PowerShell:

```powershell
docker run --rm `
  -v "${PWD}/data:/app/data" `
  -v "${PWD}/reports:/app/reports" `
  student-pipeline:0.1.0 `
  --input-csv /app/data/raw/students.csv `
  --output-directory /app/reports
```

---

### Git workflow

```bash
git init
git checkout -b main
git add .
git commit -m "Initialize Phase 1 capstone"
```

Create feature branches:

```bash
git checkout -b feature/data-cleaning
git add .
git commit -m "Add student data cleaning pipeline"
```

```bash
git checkout main
git merge feature/data-cleaning
```

Useful `.gitignore`:

```text
.venv/
__pycache__/
.pytest_cache/
*.pyc
reports/
dist/
build/
*.egg-info/
```

---

### Documentation requirements

Your `README.md` should include:

```text
Project purpose
Architecture
Installation
Input schema
CLI usage
Testing commands
Docker commands
Output files
Cleaning decisions
Known limitations
```

---

### Time complexity

Let:

* (n) = number of students
* (m) = number of columns

Loading:

[
O(nm)
]

Cleaning:

[
O(nm)
]

Analysis:

[
O(n)
]

Chart creation:

[
O(n)
]

For a fixed number of columns:

[
\boxed{O(n)}
]

Memory:

[
\boxed{O(nm)}
]

because the complete DataFrame and several copies are held in memory.

---

## 6. Exercises

Extend the capstone with:

1. Grade classification.
2. Course or department columns.
3. Average score by course.
4. Missing-value quality reports.
5. A bar chart of pass rate by course.
6. JSON logging.
7. API-key authentication.
8. Pagination support.
9. Configuration loaded from JSON.
10. A command that prints metrics without generating charts.

---

## 7. Mini Project Deliverable

Your Phase 1 capstone submission should contain:

```text
Source code
Tests
README
Dockerfile
Sample CSV
Generated cleaned CSV
Generated metrics JSON
Generated charts
Git commit history
Quiz answers
```

Minimum acceptance requirements:

* PEP 8-compliant code
* Modular package
* Virtual environment
* No hardcoded absolute paths
* Input validation
* Specific exception handling
* Logging
* At least eight automated tests
* Docker build succeeds
* CLI runs successfully
* All output files are generated

### How a real company would build it

A company would normally add:

* Pull-request reviews
* CI tests on every commit
* Dependency vulnerability scans
* Centralized logging
* Cloud object storage
* Data lineage
* Schema versioning
* Access control
* Monitoring and alerts
* Scheduled pipeline execution
* Separate development, staging, and production environments

---

## 8. Common Mistakes

1. Keeping all logic inside `main.py`.
2. Cleaning the raw DataFrame in place.
3. Accepting API data without schema validation.
4. Ignoring HTTP timeouts.
5. Saving charts without closing figures.
6. Using broad `except Exception`.
7. Testing against live APIs.
8. Hardcoding local paths.
9. Committing secrets or virtual environments.
10. Calculating cleaning statistics from future test data.
11. Using IDs as predictive features.
12. Claiming causation from a scatter plot.

---

## 9. Interview Questions

1. Describe the project architecture.
2. Why separate loading, cleaning, analysis, and visualization?
3. Why use a `src/` layout?
4. Why use NumPy when Pandas already exists?
5. How are invalid values handled?
6. How are duplicate IDs resolved?
7. Why use a CLI?
8. What does the Dockerfile provide?
9. How would you support one million rows?
10. How would you prevent API failures from stopping scheduled work?
11. How would you monitor data-quality degradation?
12. What tests belong in CI?
13. Why is `Agg` used with Matplotlib?
14. What does retention rate measure?
15. What are the project’s current limitations?

---

## 10. Summary

* Phase 1 combines Python fundamentals with production structure.
* Modules and packages separate responsibilities.
* Virtual environments isolate dependencies.
* Pandas handles tabular data.
* NumPy performs numerical analysis.
* Matplotlib creates reports.
* File loaders and APIs provide external data.
* Validation and cleaning protect downstream calculations.
* Tests verify important behavior.
* Logging records execution information.
* Docker provides a repeatable runtime.
* Git records project history.
* The complete pipeline remains approximately (O(n)) for fixed columns.

## Lesson 12 Quiz

1. What are the major pipeline stages?
2. Why are loading and cleaning separated?
3. Why should the raw DataFrame remain unchanged?
4. How are invalid numerical strings handled?
5. How are duplicate student IDs resolved?
6. What does retention rate measure?
7. Why is a CLI useful?
8. Why is `matplotlib.use("Agg")` used?
9. What output files does the pipeline generate?
10. Why should API tests use mocks?
11. What does Docker improve?
12. Why should reports not normally be committed?
13. What is the pipeline’s approximate time complexity?
14. How would you process a dataset too large for memory?
15. What would a company add before production deployment?

**Phase 1 delivery is now complete: Lessons 1–12.** Mastery remains unverified until your results are reviewed.
