import pandas as pd

# Load CSV
df = pd.read_csv('output/landmark_features.csv')

print('=' * 70)
print('LANDMARK FEATURES CSV - SAMPLE VIEW')
print('=' * 70)

print(f'\nCSV Shape: {df.shape}')
print(f'Rows: {df.shape[0]}')
print(f'Columns: {df.shape[1]}')

print('\nColumn Names:')
print(f'  First 10: {list(df.columns[:10])}')
print(f'  Last 5: {list(df.columns[-5:])}')

print('\nFirst 5 rows (selected columns):')
print(df[['class_name', 'class_label', 'landmark_0_x', 'landmark_0_y', 'landmark_0_z', 
          'landmark_20_x', 'landmark_20_y', 'landmark_20_z']].head())

print('\nSample Data Statistics:')
print(f'  Total samples: {len(df)}')
print(f'  Total unique classes: {df["class_name"].nunique()}')
vc = df['class_name'].value_counts()
print(f'  Samples per class: min={vc.min()}, max={vc.max()}, mean={vc.mean():.2f}')

print('\nLandmark coordinate ranges:')
print(f'  X coordinates: [{df["landmark_0_x"].min():.4f}, {df["landmark_0_x"].max():.4f}]')
print(f'  Y coordinates: [{df["landmark_0_y"].min():.4f}, {df["landmark_0_y"].max():.4f}]')
print(f'  Z coordinates: [{df["landmark_0_z"].min():.4f}, {df["landmark_0_z"].max():.4f}]')

print('\nSample Tamil characters in dataset:')
print(df['class_name'].unique()[:20])
