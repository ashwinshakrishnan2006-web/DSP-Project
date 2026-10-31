# Additional contribution: data quality and privacy checks

This add-on introduces two independent modules:

- `src/quality.py`: reports missing values, duplicate rows, possible direct
  identifier columns, and invalid latitude/longitude values.
- `src/privacy.py`: removes selected direct-identifier columns and optionally
  rounds coordinate precision before a classroom demo export.
- `tests/test_quality_privacy.py`: tests the new helper functions.

## How to integrate

Copy the new `src/quality.py` and `src/privacy.py` files into the existing
project's `src/` folder. Copy the test file into `tests/`. The existing
`requirements.txt` already includes pandas and pytest is only needed to run
tests; install it with `pip install pytest`.

Run from the project root:

```bash
python -m pytest -q
```

Example use:

```python
from src.quality import audit_dataset
from src.privacy import remove_sensitive_columns, round_coordinates

quality_table, summary = audit_dataset(crime_df)
safe_df, removed_columns = remove_sensitive_columns(crime_df)
safe_df = round_coordinates(safe_df, decimals=2)
```

These tools do not guarantee anonymity. Rounded coordinates can still identify
a place, and rare combinations of fields may identify a person. Review data
before sharing it and never commit private or victim-identifying records.

## Honest contribution record

A contributor should personally review the code, run the tests, fix any issues,
and document what they changed. Use GitHub commits under the contributor's own
account so the history accurately reflects who did the work. Do not add a
person's name as an author or contributor unless they actually contributed.
