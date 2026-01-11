import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
import random

def generate_aligned_twin():
    print("--- 🏗️ Generating Schema-Aligned Synthetic Data ---")
    np.random.seed(42)
    n_samples = 5000
    
    # --- 1. Core Data (Existing Logic) ---
    coords, cluster_ids = make_blobs(n_samples=n_samples, centers=8, n_features=2, random_state=42, cluster_std=0.4)
    lat = coords[:, 0] * 0.1 - 37.81 
    lon = coords[:, 1] * 0.1 + 144.96
    
    rooms = np.random.randint(1, 6, n_samples)
    base_price = 300000
    price = base_price + (rooms * 150000) + (cluster_ids * 50000) + np.random.normal(0, 50000, n_samples)
    
    # --- 2. Filling the Missing Columns (To match your Header) ---
    
    # Address: Random string to match format "123 Main St"
    addresses = [f"{np.random.randint(1,999)} Fake St" for _ in range(n_samples)]
    
    # Method: S (Sold), SP (Sold Prior), PI (Passed In)
    methods = np.random.choice(['S', 'SP', 'PI', 'VB'], n_samples)
    
    # SellerG: Real estate agents
    sellers = np.random.choice(['Nelson', 'Jellis', 'hockingstuart', 'Barry'], n_samples)
    
    # Date: Random dates in 2024/2025
    dates = pd.date_range(start='1/1/2024', periods=n_samples).strftime('%d/%m/%Y')
    
    # CouncilArea: Match cluster to a Council
    councils = [f"Council_Area_{c}" for c in cluster_ids]
    
    # Regionname:
    regions = [f"Region_{c%3}" for c in cluster_ids]
    
    # Propertycount: Density of the suburb
    prop_counts = [int(1000 + (c*500)) for c in cluster_ids]

    # --- 3. Assemble Exact DataFrame ---
    df = pd.DataFrame({
        'Suburb': [f'Suburb_{c}' for c in cluster_ids],
        'Address': addresses,
        'Rooms': rooms,
        'Type': np.random.choice(['h', 'u', 't'], n_samples),
        'Price': price,
        'Method': methods,
        'SellerG': sellers,
        'Date': dates,
        'Distance': np.random.uniform(2, 15, n_samples),
        'Postcode': 3000 + cluster_ids,
        'Bedroom2': rooms, # Often duplicates Rooms
        'Bathroom': np.random.randint(1, 4, n_samples),
        'Car': np.random.randint(0, 3, n_samples),
        'Landsize': np.abs(np.random.normal(400, 100, n_samples)),
        'BuildingArea': np.random.normal(150, 50, n_samples), # Messy data often has NaNs here, but we fill for now
        'YearBuilt': np.random.randint(1900, 2020, n_samples),
        'CouncilArea': councils,
        'Lattitude': lat,
        'Longtitude': lon,
        'Regionname': regions,
        'Propertycount': prop_counts
    })
    
    # Inject some Dirty Data (Nulls) to make preprocessing fun later
    df.loc[df.sample(frac=0.1).index, 'BuildingArea'] = np.nan
    df.loc[df.sample(frac=0.05).index, 'YearBuilt'] = np.nan
    
    return df

# Generate synthetic part
df_synthetic = generate_aligned_twin()
print("✅ Synthetic Data Generated with correct columns.")

# Save the enhanced dataframe to CSV
df_synthetic.to_csv('melbourne_titan_enhanced.csv', index=False)

print("✅ File saved as 'melbourne_titan_enhanced.csv'")
print(f"Columns included: {list(df_synthetic.columns)}")