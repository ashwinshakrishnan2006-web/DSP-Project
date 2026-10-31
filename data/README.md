# Data Directory

## Full Dataset

Download the 2023 Chicago Crime dataset from the City of Chicago Open Data Portal:

🔗 https://data.cityofchicago.org/Public-Safety/Crimes-2023/xguy-4ndq

Save the downloaded file as:
```
data/chicago_crimes_2023.csv
```

This file (~261,238 records, ~85 MB) is excluded from version control via `.gitignore`.

## Sample Dataset

A 500-row synthetic sample (`sample_chicago_crimes.csv`) is included for testing.

Generate a fresh sample at any time:
```bash
python data/generate_sample.py
```
