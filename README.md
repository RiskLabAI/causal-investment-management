# Causal Investment Management

Executable notebook companion for the book *Causal Investment Management*.

This companion contains 30 notebooks arranged under 10 book subjects. The folder numbers follow the current book, and each notebook filename identifies its section within that subject.

The examples use RiskLabAI 3.2.0, public online data, and visible fixed-seed simulations when a known population or counterfactual is required. They are teaching and research examples, not evidence of market performance.

## Running the notebooks

Create a Python environment, install the listed requirements, and start Jupyter from the repository root. Public data are retrieved when first needed and accepted only when their SHA-256 values match the records in `DATA_SOURCES.md`.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
jupyter lab
```

Generated figures, tables, and intermediate files are written under `outputs/`. The notebooks use the light figure palette defined in `book_figure_style.py`.

## Contents

See `NOTEBOOK_CROSSWALK.md` for the complete subject list and notebook counts.

## Related material

Reusable method tutorials that are not tied to book order remain in the RiskLabAI Notebooks.py collection.

## License

This repository is released under the BSD 3-Clause License. See `LICENSE` for the complete terms.
