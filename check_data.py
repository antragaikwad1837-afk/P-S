
import pandas as pd
from pathlib import Path

data_folder = Path("data")

files = [
    "india_2000_2024_daily_weather.csv",
    "Rainfall_Data_LL.csv",
    "rainfall_statewise_daily_imd.csv"
]

for filename in files:
    file_path = data_folder / filename

    print("\n" + "=" * 60)
    print("DATASET:", filename)
    print("=" * 60)

    try:
        df = pd.read_csv(file_path)

        print("Rows and columns:", df.shape)
        print("\nColumn names:")
        print(df.columns.tolist())

        print("\nFirst 5 rows:")
        print(df.head().to_string())

        print("\nMissing values:")
        print(df.isnull().sum().to_string())

    except Exception as e:
        print("Error:", e)
