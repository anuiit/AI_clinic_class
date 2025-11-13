import pandas as pd

# Read the elements.csv file
df = pd.read_csv('elements.csv')

# Get unique elements and create a mapping to element_id
unique_elements = df['element'].unique()
element_to_id = {element: idx + 1 for idx, element in enumerate(sorted(unique_elements))}

# Add element_id column based on the element name
df['element_id'] = df['element'].map(element_to_id)

# Reorder columns to put element_id after element
cols = df.columns.tolist()
# Remove element_id from its current position
cols.remove('element_id')
# Find the position of 'element' column
element_pos = cols.index('element')
# Insert element_id right after element
cols.insert(element_pos + 1, 'element_id')
df = df[cols]

# Save the updated dataframe back to the CSV file
df.to_csv('elements.csv', index=False)

print("Successfully added 'element_id' column to elements.csv")
print(f"Total rows: {len(df)}")
print(f"Unique elements: {len(unique_elements)}")
print("\nFirst few rows:")
print(df.head(15))
print("\nExample of same element having same element_id:")
print(df[df['element'] == 'atl'][['id', 'element', 'element_id']].head(10))
