/**
 * Binary File Handling Section
 * Module 3: File Processing & Document Understanding
 */

export const binaryfilehandlingSection = {
    id: 'binary-file-handling',
    title: 'Binary File Handling',
    content: `# Binary File Handling

Master binary file processing for comprehensive file manipulation in LLM applications.

## Overview

Binary files require special handling. Understanding binary formats enables processing any file type - from SQLite databases to ZIP archives to custom formats.

## Basic Binary Operations

\`\`\`python
from pathlib import Path

# Read binary file
def read_binary(filepath: str) -> bytes:
    return Path(filepath).read_bytes()

# Write binary file
def write_binary(filepath: str, data: bytes):
    Path(filepath).write_bytes(data)

# Hex inspection
def inspect_binary(filepath: str, num_bytes: int = 16):
    data = Path(filepath).read_bytes()[:num_bytes]
    hex_str = data.hex()
    print(f"Hex: {hex_str}")
    print(f"ASCII: {data}")
\`\`\`

## File Type Detection

\`\`\`python
import magic  # pip install python-magic

def detect_file_type(filepath: str):
    """Detect file type from magic numbers."""
    mime = magic.Magic(mime=True)
    file_type = mime.from_file(filepath)
    return file_type

# Common magic numbers
MAGIC_NUMBERS = {
    b'\\x50\\x4B\\x03\\x04': 'ZIP',
    b'\\x25\\x50\\x44\\x46': 'PDF',
    b'\\x89PNG': 'PNG',
    b'\\xFF\\xD8\\xFF': 'JPEG'
}
\`\`\`

## SQLite Database Processing

\`\`\`python
import sqlite3
import pandas as pd

def read_sqlite_table(db_path: str, table_name: str):
    """Read SQLite table to DataFrame."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def list_sqlite_tables(db_path: str):
    """List all tables in SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables
\`\`\`

## Archive Handling

\`\`\`python
import zipfile
import tarfile

# ZIP files
def extract_zip(zip_path: str, extract_to: str):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def list_zip_contents(zip_path: str):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        return zip_ref.namelist()

# TAR files
def extract_tar(tar_path: str, extract_to: str):
    with tarfile.open(tar_path, 'r:*') as tar_ref:
        tar_ref.extractall(extract_to)
\`\`\`

## Key Takeaways

1. **Use read_bytes/write_bytes** for binary files
2. **Magic numbers** identify file types
3. **python-magic** for file type detection
4. **SQLite** is a common binary format
5. **Handle archives** (ZIP, TAR) appropriately
6. **Binary data** requires bytes not strings
7. **Struct module** for binary parsing
8. **Never assume** text encoding for binary
9. **Check file signatures** before processing
10. **Use appropriate libraries** for each format`,
    videoUrl: null,
};

