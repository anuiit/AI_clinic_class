"""
SQL to CSV Converter for Codex Database - Optimized Version
Converts the SQL dump to CSV files and creates a relationship mapping
"""

import csv
from pathlib import Path


def extract_columns_from_create(line):
    """Extract column names from CREATE TABLE statement"""
    columns = []
    return columns


def parse_insert_line(line):
    """Parse a single line of INSERT statement"""
    # Look for lines with VALUES or data rows
    if 'INSERT INTO' in line and 'VALUES' in line:
        return 'START', None
    elif line.strip().startswith('(') and line.strip().endswith('),'):
        # This is a data row (not the last one)
        return 'ROW', line.strip()[1:-2]  # Remove ( and ),
    elif line.strip().startswith('(') and line.strip().endswith(');'):
        # This is the last data row
        return 'LAST_ROW', line.strip()[1:-2]  # Remove ( and );
    return None, None


def parse_sql_values(value_string):
    """Parse comma-separated SQL values"""
    values = []
    current_value = ''
    in_quote = False
    quote_char = None
    escape_next = False
    
    for char in value_string + ',':
        if escape_next:
            current_value += char
            escape_next = False
            continue
        
        if char == '\\' and in_quote:
            escape_next = True
            current_value += char
            continue
        
        if char in ("'", '"'):
            if not in_quote:
                in_quote = True
                quote_char = char
            elif char == quote_char:
                in_quote = False
                quote_char = None
            continue
        
        if char == ',' and not in_quote:
            # End of value
            value = current_value.strip()
            if value == 'NULL':
                values.append('')
            else:
                values.append(value)
            current_value = ''
        else:
            current_value += char
    
    return values[:-1]  # Remove last empty from trailing comma


def process_table(sql_file, table_name, output_file):
    """Extract data for a specific table and write to CSV"""
    print(f"\nProcessing {table_name}...")
    
    # Column mappings for our tables
    columns_map = {
        'ai_codex': ['id', 'titre', 'glyphes', 'personnages', 'elements'],
        'ai_glyphe': ['id', 'codexid', 'cote', 'lecture', 'element'],
        'ai_element': ['id', 'codexid', 'cote', 'element', 'theme']
    }
    
    columns = columns_map.get(table_name, [])
    if not columns:
        print(f"  Unknown table: {table_name}")
        return None
    
    print(f"  Columns: {', '.join(columns)}")
    
    rows = []
    in_table = False
    in_insert = False
    
    with open(sql_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            # Check if we're entering our table
            if f'CREATE TABLE IF NOT EXISTS `{table_name}`' in line:
                in_table = True
                print(f"  Found table at line {line_num}")
                continue
            
            # Check if we're in an INSERT for our table
            if in_table and f'INSERT INTO `{table_name}`' in line:
                in_insert = True
                print(f"  Found INSERT at line {line_num}")
                continue
            
            # If we hit another CREATE TABLE, we're done with this table
            if in_table and line.startswith('CREATE TABLE') and table_name not in line:
                break
            
            # Parse data rows
            if in_insert:
                line = line.strip()
                if line.startswith('('):
                    # Extract values
                    if line.endswith('),'):
                        value_string = line[1:-2]  # Remove ( and ),
                    elif line.endswith(');'):
                        value_string = line[1:-2]  # Remove ( and );
                        in_insert = False  # Last row
                    else:
                        continue
                    
                    values = parse_sql_values(value_string)
                    if len(values) == len(columns):
                        rows.append(values)
                    
                    if len(rows) % 1000 == 0:
                        print(f"    Processed {len(rows)} rows...")
    
    print(f"  Extracted {len(rows)} rows total")
    
    # Write to CSV
    if rows:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)
        print(f"  Created {output_file}")
    
    return columns, rows


def create_relationships_csv(codex_data, glyphe_data, element_data):
    """Create a comprehensive relationships CSV"""
    print("\nCreating relationships file...")
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
        theme = element_row[4] if len(element_row) > 4 else ''
        
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
        
        if len(relationships) % 5000 == 0:
            print(f"  Processed {len(relationships)} relationships...")
    
    return relationships


def main():
    print("=" * 60)
    print("SQL to CSV Converter for Codex Database")
    print("=" * 60)
    
    # Check SQL file
    sql_file = Path("inst-supinfor-application.sql")
    if not sql_file.exists():
        print(f"\nError: {sql_file} not found")
        return
    
    print(f"\nReading from: {sql_file}")
    print(f"File size: {sql_file.stat().st_size / (1024*1024):.1f} MB")
    
    # Extract data for each table
    codex_cols, codex_rows = process_table(sql_file, 'ai_codex', 'codex.csv')
    glyphe_cols, glyphe_rows = process_table(sql_file, 'ai_glyphe', 'glyphs.csv')
    element_cols, element_rows = process_table(sql_file, 'ai_element', 'elements.csv')
    
    # Create relationships file
    if codex_rows and glyphe_rows and element_rows:
        relationships = create_relationships_csv(codex_rows, glyphe_rows, element_rows)
        with open('relationships.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(relationships)
        print(f"  Created relationships.csv with {len(relationships)-1} relationships")
    
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("\nGenerated files:")
    print("  📄 codex.csv         : Main codex table ({} records)".format(len(codex_rows) if codex_rows else 0))
    print("  📄 glyphs.csv        : Glyphs table ({} records)".format(len(glyphe_rows) if glyphe_rows else 0))
    print("  📄 elements.csv      : Elements table ({} records)".format(len(element_rows) if element_rows else 0))
    print("  📄 relationships.csv : Complete relationship mapping")
    print("\nRelationship structure:")
    print("  Codex (1) ──→ Glyphs (many) ──→ Elements (many)")
    print("=" * 60)


if __name__ == "__main__":
    main()
