/**
 * CSV & Structured Data Section
 * Module 3: File Processing & Document Understanding
 */

export const csvstructureddataSection = {
  id: 'csv-structured-data',
  title: 'CSV & Structured Data',
  content: `# CSV & Structured Data

Master processing CSV files and structured data formats for building robust data pipelines in LLM applications.

## Overview

CSV files are the universal data exchange format. From datasets to exports to logs, CSV processing is fundamental for data-driven AI applications.

## Reading CSV Files

\`\`\`python
import csv
import pandas as pd
from pathlib import Path

# Basic CSV reading with csv module
def read_csv_basic (filepath: str):
    with open (filepath, 'r') as f:
        reader = csv.DictReader (f)
        for row in reader:
            print(row)

# pandas - most common approach
df = pd.read_csv('data.csv')

# Handle encoding
df = pd.read_csv('data.csv', encoding='utf-8')

# Skip rows
df = pd.read_csv('data.csv', skiprows=2)

# Specify delimiter
df = pd.read_csv('data.tsv', delimiter='\\t')
\`\`\`

## Production CSV Processor

\`\`\`python
import pandas as pd
import logging
from typing import Optional, Dict
from pathlib import Path

class CSVProcessor:
    """Production-grade CSV processor with validation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def read_csv_safe(
        self,
        filepath: str,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """Read CSV with error handling."""
        path = Path (filepath)
        
        if not path.exists():
            self.logger.error (f"File not found: {filepath}")
            return None
        
        # Check file size
        max_size = 100 * 1024 * 1024  # 100MB
        if path.stat().st_size > max_size:
            self.logger.warning (f"Large file: {path.stat().st_size} bytes")
        
        try:
            # Try UTF-8 first
            df = pd.read_csv (filepath, encoding='utf-8', **kwargs)
        except UnicodeDecodeError:
            # Fallback to latin-1
            df = pd.read_csv (filepath, encoding='latin-1', **kwargs)
        except Exception as e:
            self.logger.error (f"Failed to read CSV: {e}")
            return None
        
        return df
    
    def validate_csv (self, df: pd.DataFrame, required_columns: list) -> bool:
        """Validate CSV has required columns."""
        missing = set (required_columns) - set (df.columns)
        if missing:
            self.logger.error (f"Missing columns: {missing}")
            return False
        return True

# Usage
processor = CSVProcessor()
df = processor.read_csv_safe('data.csv')
if df is not None and processor.validate_csv (df, ['name', 'email']):
    # Process data
    pass
\`\`\`

## JSON and JSONL Processing

\`\`\`python
import json
import pandas as pd

# Read JSON
with open('data.json', 'r') as f:
    data = json.load (f)

# Read JSONL (newline-delimited JSON)
def read_jsonl (filepath: str):
    data = []
    with open (filepath, 'r') as f:
        for line in f:
            data.append (json.loads (line))
    return data

# pandas for JSON
df = pd.read_json('data.json')

# Convert to DataFrame
df = pd.DataFrame (data)
\`\`\`

## Key Takeaways

1. **Use pandas** for most CSV operations
2. **Handle encoding** explicitly (UTF-8 first)
3. **Validate data** after reading
4. **Check file sizes** before loading
5. **Use appropriate delimiters** (comma, tab, pipe)
6. **Clean data** after reading
7. **Handle missing values** appropriately
8. **Type conversion** for correct data types
9. **Chunk processing** for large files
10. **Error handling** for production systems`,
  videoUrl: undefined,
};
