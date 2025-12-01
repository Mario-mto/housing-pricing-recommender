import pandas as pd
from pathlib import Path

data = [
    {
        "surface": 45,
        "rooms": 2,
        "bathrooms": 1,
        "floor": 3,
        "building_age": 15,
        "neighbourhood": "Plateau",
        "city": "Montreal",
        "furnished": "yes",
        "has_elevator": "no",
        "price": 1700
    },
    {
        "surface": 30,
        "rooms": 1,
        "bathrooms": 1,
        "floor": 1,
        "building_age": 30,
        "neighbourhood": "Rosemont",
        "city": "Montreal",
        "furnished": "no",
        "has_elevator": "no",
        "price": 1200
    },
]

df = pd.DataFrame(data)

raw_dir = Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(raw_dir / "listings.csv", index=False)

print("✅ Faux CSV créé dans data/raw/listings.csv")
print(df)