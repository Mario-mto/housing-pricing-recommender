# create_fake_data.py

import pandas as pd
from pathlib import Path
import numpy as np

data = []

# On crée 20 logements fictifs
np.random.seed(42)

neighbourhoods = ["Plateau", "Rosemont", "Hochelaga", "Ville-Marie"]
cities = ["Montreal"]
yes_no = ["yes", "no"]

for i in range(20):
    surface = np.random.randint(15, 80)          # 15 à 80 m²
    rooms = np.random.randint(1, 4)              # 1 à 3 pièces
    bathrooms = np.random.randint(1, 2)          # 1 ou 2
    floor = np.random.randint(0, 10)             # 0 à 9
    building_age = np.random.randint(1, 60)      # 1 à 60 ans

    neighbourhood = np.random.choice(neighbourhoods)
    city = np.random.choice(cities)
    furnished = np.random.choice(yes_no)
    has_elevator = np.random.choice(yes_no)

    # Prix fictif : base + 10€/m² + un peu d'aléatoire
    price = 600 + surface * 10 + np.random.randint(-150, 150)

    data.append({
        "surface": surface,
        "rooms": rooms,
        "bathrooms": bathrooms,
        "floor": floor,
        "building_age": building_age,
        "neighbourhood": neighbourhood,
        "city": city,
        "furnished": furnished,
        "has_elevator": has_elevator,
        "price": price
    })

df = pd.DataFrame(data)

raw_dir = Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)
csv_path = raw_dir / "listings.csv"

df.to_csv(csv_path, index=False)
print(f"✅ Faux CSV créé avec {len(df)} lignes dans {csv_path}")
print(df.head())
print(df)