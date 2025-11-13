"""
SQL to CSV Converter for Codex Database
Converts the SQL dump to CSV files and creates a relationship mapping
"""

import re
import csv
from pathlib import Path


def parse_sql_value(value):
    """Parse a SQL value (handles strings, numbers, NULL)"""
    value = value.strip()
    if value == 'NULL':
        return ''
    # Remove surrounding quotes if present
    if (value.startswith("'") and value.endswith("'")) or \
       (value.startswith('"') and value.endswith('"')):
        value = value[1:-1]
        # Unescape escaped quotes
        value = value.replace("\\'", "'").replace('\\"', '"')
    return value


def extract_table_data(sql_content, table_name):
    """Extract INSERT statements for a specific table"""
    print(f"  Searching for table {table_name}...")
    
    # Find CREATE TABLE statement to get column names
    create_pattern = rf"CREATE TABLE IF NOT EXISTS `{table_name}` \((.*?)\) ENGINE"
    create_match = re.search(create_pattern, sql_content, re.DOTALL)
    
    if not create_match:
        print(f"  ERROR: Table {table_name} not found")
        return None, None
    
    # Extract column definitions
    columns_text = create_match.group(1)
    columns = []
    for line in columns_text.split('\n'):
        line = line.strip()
        if line.startswith('`'):
            col_name = line.split('`')[1]
            columns.append(col_name)
    
    print(f"  Found {len(columns)} columns: {', '.join(columns)}")
    
    # Find all INSERT statements for this table
    # Pattern: INSERT INTO `table_name` (...) VALUES ... ;
    insert_pattern = rf"INSERT INTO `{table_name}` \([^)]+\) VALUES\s+([\s\S]+?);"
    
    rows = []
    for insert_match in re.finditer(insert_pattern, sql_content):
        values_block = insert_match.group(1)
        
        # Extract individual rows - each row is wrapped in parentheses
        # We need to handle nested parentheses and quoted strings
        current_row = []
        current_value = ''
        paren_depth = 0
        in_quote = False
        quote_char = None
        escape_next = False
        
        for i, char in enumerate(values_block):
            if escape_next:
                current_value += char
                escape_next = False
                continue
            
            if char == '\\':
                current_value += char
                escape_next = True
                continue
            
            if char in ("'", '"') and paren_depth > 0:
                if not in_quote:
                    in_quote = True
                    quote_char = char
                elif char == quote_char:
                    in_quote = False
                    quote_char = None
                current_value += char
                continue
            
            if not in_quote:
                if char == '(':
                    paren_depth += 1
                    if paren_depth == 1:
                        current_row = []
                        current_value = ''
                        continue
                elif char == ')':
                    if paren_depth == 1:
                        # End of row
                        if current_value.strip():
                            current_row.append(parse_sql_value(current_value))
                        if len(current_row) == len(columns):
                            rows.append(current_row)
                        current_value = ''
                    paren_depth -= 1
                    continue
                elif char == ',' and paren_depth == 1:
                    # End of value
                    current_row.append(parse_sql_value(current_value))
                    current_value = ''
                    continue
            
            if paren_depth > 0:
                current_value += char
    
    print(f"  Extracted {len(rows)} rows")
    return columns, rows



def create_csv_file(filename, columns, rows):
    """Create a CSV file from columns and rows"""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"Created {filename} with {len(rows)} rows")


def create_relationships_csv(codex_data, glyphe_data, element_data):
    """Create a comprehensive relationships CSV"""
    relationships = []
    
    # Header
    relationships.append([
        'codex_id', 'codex_titre', 
        'glyph_id', 'glyph_cote', 'glyph_lecture',
        'element_id', 'element_name', 'element_theme'
    ])
    
    # Create dictionaries for quick lookup
    codex_dict = {row[0]: row for row in codex_data}
    glyphe_dict = {}
    for row in glyphe_data:
        key = (row[1], row[2])  # (codexid, cote)
        if key not in glyphe_dict:
            glyphe_dict[key] = []
        glyphe_dict[key].append(row)
    
    # Process elements and link them
    for element_row in element_data:
        element_id = element_row[0]
        codex_id = element_row[1]
        cote = element_row[2]
        element_name = element_row[3]
        theme = element_row[4]
        
        # Get codex info
        codex_info = codex_dict.get(codex_id, ['', '', '', '', ''])
        codex_titre = codex_info[1] if len(codex_info) > 1 else ''
        
        # Get glyph info
        glyph_key = (codex_id, cote)
        glyphs = glyphe_dict.get(glyph_key, [[None, None, None, '', '']])
        
        for glyph in glyphs:
            glyph_id = glyph[0] if glyph[0] else ''
            glyph_lecture = glyph[3] if len(glyph) > 3 else ''
            
            relationships.append([
                codex_id, codex_titre,
                glyph_id, cote, glyph_lecture,
                element_id, element_name, theme
            ])
    
    return relationships


def main():
    print("SQL to CSV Converter for Codex Database")
    print("=" * 50)
    
    # Read SQL file
    sql_file = Path("inst-supinfor-application.sql")
    if not sql_file.exists():
        print(f"Error: {sql_file} not found")
        return
    
    print(f"Reading {sql_file}...")
    with open(sql_file, 'r', encoding='utf-8') as f:
        sql_content = f.read()
    
    # Extract data for each table
    print("\nExtracting data from SQL...")
    
    # Extract ai_codex
    print("\n1. Processing ai_codex table...")
    codex_cols, codex_rows = extract_table_data(sql_content, 'ai_codex')
    if codex_cols:
        create_csv_file('codex.csv', codex_cols, codex_rows)
    
    # Extract ai_glyphe
    print("\n2. Processing ai_glyphe table...")
    glyphe_cols, glyphe_rows = extract_table_data(sql_content, 'ai_glyphe')
    if glyphe_cols:
        create_csv_file('glyphs.csv', glyphe_cols, glyphe_rows)
    
    # Extract ai_element
    print("\n3. Processing ai_element table...")
    element_cols, element_rows = extract_table_data(sql_content, 'ai_element')
    if element_cols:
        create_csv_file('elements.csv', element_cols, element_rows)
    
    # Create relationships file
    if codex_rows and glyphe_rows and element_rows:
        print("\n4. Creating relationships CSV...")
        relationships = create_relationships_csv(codex_rows, glyphe_rows, element_rows)
        with open('relationships.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(relationships)
        print(f"Created relationships.csv with {len(relationships)-1} relationships")
    
    print("\n" + "=" * 50)
    print("Conversion complete!")
    print("\nGenerated files:")
    print("  - codex.csv         : Main codex table")
    print("  - glyphs.csv        : Glyphs table")
    print("  - elements.csv      : Elements table")
    print("  - relationships.csv : Complete relationship mapping")
    print("\nRelationship structure:")
    print("  Codex (1) -> Glyphs (many) -> Elements (many)")


if __name__ == "__main__":
    main()
