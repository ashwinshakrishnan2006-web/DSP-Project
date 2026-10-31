"""
Generate a small sample Chicago crime CSV for testing the pipeline
without downloading the full 261k-record dataset.
"""
import csv
import random
from pathlib import Path

random.seed(42)

CRIME_TYPES = [
    "THEFT", "BATTERY", "CRIMINAL DAMAGE", "ASSAULT", "BURGLARY",
    "MOTOR VEHICLE THEFT", "ROBBERY", "NARCOTICS", "OTHER OFFENSE",
    "CRIMINAL TRESPASS",
]

LOCATIONS = [
    "STREET", "APARTMENT", "RESIDENCE", "SIDEWALK", "PARKING LOT",
    "VEHICLE NON-COMMERCIAL", "RESIDENCE - PORCH / HALLWAY", "ALLEY",
    "SMALL RETAIL STORE", "GAS STATION",
]

HEADERS = [
    "ID", "Case Number", "Date", "Block", "IUCR", "Primary Type",
    "Description", "Location Description", "Arrest", "Domestic",
    "Beat", "District", "Ward", "Community Area", "FBI Code",
    "X Coordinate", "Y Coordinate", "Year", "Updated On",
    "Latitude", "Longitude", "Location",
]

output_path = Path(__file__).parent / "sample_chicago_crimes.csv"

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=HEADERS)
    writer.writeheader()

    for i in range(500):
        lat = round(random.uniform(41.65, 42.05), 6)
        lon = round(random.uniform(-87.90, -87.55), 6)
        hour = random.randint(0, 23)
        month = random.randint(1, 12)
        date_str = f"0{month}/15/2023 {hour:02d}:00:00 AM"
        writer.writerow({
            "ID": 10_000_000 + i,
            "Case Number": f"JG{100000 + i}",
            "Date": date_str,
            "Block": f"{random.randint(100, 9999):04d} N TEST ST",
            "IUCR": "0460",
            "Primary Type": random.choice(CRIME_TYPES),
            "Description": "SIMPLE",
            "Location Description": random.choice(LOCATIONS),
            "Arrest": random.choice([True, False]),
            "Domestic": random.choice([True, False]),
            "Beat": str(random.randint(100, 2500)),
            "District": str(random.randint(1, 25)),
            "Ward": str(random.randint(1, 50)),
            "Community Area": str(random.randint(1, 77)),
            "FBI Code": "08B",
            "X Coordinate": str(random.randint(1100000, 1200000)),
            "Y Coordinate": str(random.randint(1800000, 1950000)),
            "Year": "2023",
            "Updated On": "10/31/2023 03:40:55 PM",
            "Latitude": lat,
            "Longitude": lon,
            "Location": f"({lat}, {lon})",
        })

print(f"Sample data written to {output_path}  (500 rows)")
