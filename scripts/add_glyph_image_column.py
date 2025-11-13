import pandas as pd

# Read the relationships.csv file
df = pd.read_csv('relationships.csv')

# Create the glyph_image column by combining glyph_cote and codex_id
df['glyph_image'] = df['glyph_cote'] + '-' + df['codex_id'].astype(str) + '.jpg'

# Save the updated dataframe back to the CSV file
df.to_csv('relationships.csv', index=False)

print("Successfully added 'glyph_image' column to relationships.csv")
print(f"Total rows: {len(df)}")
print("\nFirst few rows with the new column:")
print(df[['codex_id', 'glyph_cote', 'glyph_image']].head(10))
