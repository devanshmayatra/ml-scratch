import pandas as pd
import numpy as np


df_real = pd.read_csv('melb_data.csv')
df_synthetic = pd.read_csv('melbourne_titan_enhanced.csv')
df_real['Source'] = 'Real'

# 2. Tag Synthetic Data
df_synthetic['Source'] = 'Synthetic'

# 3. CONCATENATE
# axis=0 means stack them on top of each other
df_combined = pd.concat([df_real, df_synthetic], axis=0, ignore_index=True)

print(f"\n✅ Merged Dataset Created!")
print(f"Total Rows: {len(df_combined)}")
print(f"Columns: {len(df_combined.columns)}")
print(df_combined.head())

# Save the enhanced dataframe to CSV
df_combined.to_csv('dataset.csv', index=False)

print("✅ File saved as 'melbourne_titan_enhanced.csv'")
print(f"Columns included: {list(df_combined.columns)}")