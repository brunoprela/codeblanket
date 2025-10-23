/**
 * Excel File Manipulation Section
 * Module 3: File Processing & Document Understanding
 */

export const excelfilemanipulationSection = {
  id: 'excel-file-manipulation',
  title: 'Excel File Manipulation',
  content: `# Excel File Manipulation

Master Excel file processing for building AI applications that can read, understand, and modify spreadsheets through natural language.

## Overview: Excel in LLM Applications

Excel is ubiquitous in business - from financial models to data analysis. Building LLM applications that can intelligently manipulate Excel files unlocks enormous value. Imagine a Cursor-like tool for Excel where you can say "add a column calculating profit margin" or "format all cells with negative values in red."

**Use Cases:**
- Natural language Excel manipulation
- Automated report generation
- Data extraction and transformation
- Formula generation from descriptions
- Spreadsheet analysis and insights

**This section** covers the fundamentals using openpyxl and pandas. The next section covers advanced operations like charts, formulas, and pivot tables.

## Libraries Overview

### openpyxl vs pandas vs xlwings

\`\`\`python
# openpyxl: Full control over Excel structure
# - Read/write .xlsx files
# - Preserve formatting, formulas, charts
# - Cell-by-cell manipulation
# - Best for: Maintaining existing Excel files

# pandas: Data-centric operations
# - Fast data manipulation
# - DataFrames are powerful for analysis
# - Less control over formatting
# - Best for: Data processing and analysis

# xlwings: Full Excel automation (requires Excel)
# - Control Excel application directly
# - Run VBA macros
# - Real-time Excel interaction
# - Best for: Complex Excel automation, macros
\`\`\`

## Reading Excel Files with openpyxl

### Basic File Reading

\`\`\`python
from openpyxl import load_workbook
from pathlib import Path

def read_excel_basic(filepath: str) -> None:
    """Read Excel file and print basic information."""
    wb = load_workbook(filepath)
    
    # Get sheet names
    print(f"Sheet names: {wb.sheetnames}")
    
    # Get active sheet
    sheet = wb.active
    print(f"Active sheet: {sheet.title}")
    
    # Get dimensions
    print(f"Dimensions: {sheet.dimensions}")
    print(f"Max row: {sheet.max_row}, Max col: {sheet.max_column}")
    
    # Read specific cell
    cell_value = sheet['A1'].value
    print(f"A1 value: {cell_value}")
    
    # Or by row/column index (1-based!)
    cell_value = sheet.cell(row=1, column=1).value
    print(f"Cell(1,1) value: {cell_value}")

# Usage
read_excel_basic("data.xlsx")
\`\`\`

### Reading All Data from a Sheet

\`\`\`python
from openpyxl import load_workbook
from typing import List

def read_sheet_data(filepath: str, sheet_name: str = None) -> List[List]:
    """
    Read all data from a sheet as a list of lists.
    
    Returns: Each row as a list, cells as values
    """
    wb = load_workbook(filepath, data_only=True)  # data_only=True reads formula results
    
    # Get sheet
    if sheet_name:
        sheet = wb[sheet_name]
    else:
        sheet = wb.active
    
    # Read all rows
    data = []
    for row in sheet.iter_rows(values_only=True):
        data.append(list(row))
    
    return data

# Usage
data = read_sheet_data("sales.xlsx", "Q4 Sales")
print(f"Read {len(data)} rows")
print(f"First row: {data[0]}")
\`\`\`

### Reading with Headers

\`\`\`python
from openpyxl import load_workbook
from typing import List, Dict

def read_sheet_with_headers(
    filepath: str,
    sheet_name: str = None
) -> tuple[List[str], List[Dict]]:
    """
    Read sheet data with first row as headers.
    
    Returns: (headers, rows as dicts)
    """
    wb = load_workbook(filepath, data_only=True)
    sheet = wb[sheet_name] if sheet_name else wb.active
    
    # Get all rows
    rows = list(sheet.iter_rows(values_only=True))
    
    if not rows:
        return [], []
    
    # First row is headers
    headers = list(rows[0])
    
    # Convert remaining rows to dicts
    data = []
    for row in rows[1:]:
        row_dict = {header: value for header, value in zip(headers, row)}
        data.append(row_dict)
    
    return headers, data

# Usage
headers, data = read_sheet_with_headers("employees.xlsx")
print(f"Headers: {headers}")
print(f"First employee: {data[0]}")
\`\`\`

## Reading Excel with pandas

### Basic pandas Reading

\`\`\`python
import pandas as pd

# Simple read
df = pd.read_excel("data.xlsx")
print(df.head())

# Specify sheet
df = pd.read_excel("data.xlsx", sheet_name="Sheet2")

# Read all sheets
all_sheets = pd.read_excel("data.xlsx", sheet_name=None)
for sheet_name, df in all_sheets.items():
    print(f"{sheet_name}: {len(df)} rows")

# Skip rows
df = pd.read_excel("data.xlsx", skiprows=2)  # Skip first 2 rows

# Specify columns
df = pd.read_excel("data.xlsx", usecols="A:C")  # Only columns A-C

# No headers
df = pd.read_excel("data.xlsx", header=None)
\`\`\`

### Production-Grade Excel Reader

\`\`\`python
import pandas as pd
from pathlib import Path
from typing import Optional, Dict
import logging

class ExcelReader:
    """
    Production-grade Excel reader with error handling.
    
    Used for building Excel processing pipelines.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def read_excel_safe(
        self,
        filepath: str,
        sheet_name: str = 0,
        header_row: int = 0
    ) -> Optional[pd.DataFrame]:
        """
        Read Excel file safely with comprehensive error handling.
        
        Args:
            filepath: Path to Excel file
            sheet_name: Sheet name or index (0-based)
            header_row: Row number for headers (0-based)
        
        Returns:
            DataFrame or None if reading fails
        """
        path = Path(filepath)
        
        # Check file exists
        if not path.exists():
            self.logger.error(f"File not found: {filepath}")
            return None
        
        # Check file extension
        if path.suffix not in ['.xlsx', '.xls', '.xlsm']:
            self.logger.error(f"Invalid file type: {path.suffix}")
            return None
        
        # Check file size (avoid loading huge files)
        max_size = 50 * 1024 * 1024  # 50MB
        if path.stat().st_size > max_size:
            self.logger.error(f"File too large: {path.stat().st_size} bytes")
            return None
        
        try:
            df = pd.read_excel(
                filepath,
                sheet_name=sheet_name,
                header=header_row,
                engine='openpyxl'  # Specify engine for .xlsx
            )
            
            self.logger.info(
                f"Successfully read {filepath}: {len(df)} rows, {len(df.columns)} columns"
            )
            return df
            
        except ValueError as e:
            self.logger.error(f"Invalid sheet name or format: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to read Excel file: {e}")
            return None
    
    def read_all_sheets(self, filepath: str) -> Dict[str, pd.DataFrame]:
        """Read all sheets from Excel file."""
        try:
            return pd.read_excel(filepath, sheet_name=None, engine='openpyxl')
        except Exception as e:
            self.logger.error(f"Failed to read sheets: {e}")
            return {}
    
    def get_sheet_info(self, filepath: str) -> Dict:
        """Get metadata about Excel file."""
        from openpyxl import load_workbook
        
        try:
            wb = load_workbook(filepath, read_only=True, data_only=True)
            
            info = {
                "filename": Path(filepath).name,
                "sheet_count": len(wb.sheetnames),
                "sheet_names": wb.sheetnames,
                "active_sheet": wb.active.title,
                "sheets": {}
            }
            
            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                info["sheets"][sheet_name] = {
                    "rows": sheet.max_row,
                    "columns": sheet.max_column,
                    "dimensions": sheet.dimensions
                }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get file info: {e}")
            return {}

# Usage
reader = ExcelReader()
df = reader.read_excel_safe("data.xlsx", sheet_name="Sales")
info = reader.get_sheet_info("data.xlsx")
print(info)
\`\`\`

## Writing Excel Files

### Basic Writing with openpyxl

\`\`\`python
from openpyxl import Workbook
from pathlib import Path

def write_excel_basic(filepath: str, data: List[List]) -> None:
    """Write data to Excel file."""
    wb = Workbook()
    sheet = wb.active
    sheet.title = "Data"
    
    # Write data
    for row in data:
        sheet.append(row)
    
    # Save
    wb.save(filepath)

# Usage
data = [
    ["Name", "Age", "City"],
    ["Alice", 30, "New York"],
    ["Bob", 25, "San Francisco"],
    ["Charlie", 35, "Chicago"]
]
write_excel_basic("output.xlsx", data)
\`\`\`

### Writing with Cell Formatting

\`\`\`python
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

def write_formatted_excel(filepath: str, data: List[List]) -> None:
    """
    Write Excel with formatting.
    
    Similar to what an AI Excel tool like Cursor would generate.
    """
    wb = Workbook()
    sheet = wb.active
    sheet.title = "Formatted Data"
    
    # Write data
    for row_idx, row in enumerate(data, start=1):
        for col_idx, value in enumerate(row, start=1):
            cell = sheet.cell(row=row_idx, column=col_idx, value=value)
            
            # Format header row
            if row_idx == 1:
                cell.font = Font(bold=True, size=12, color="FFFFFF")
                cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
                cell.alignment = Alignment(horizontal="center", vertical="center")
    
    # Auto-size columns
    for column in sheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(cell.value)
            except:
                pass
        
        adjusted_width = min(max_length + 2, 50)
        sheet.column_dimensions[column_letter].width = adjusted_width
    
    wb.save(filepath)

# Usage
data = [
    ["Product", "Price", "Quantity", "Total"],
    ["Widget A", 10.50, 100, 1050.00],
    ["Widget B", 25.00, 50, 1250.00],
    ["Widget C", 15.75, 75, 1181.25]
]
write_formatted_excel("formatted_output.xlsx", data)
\`\`\`

### Writing with pandas

\`\`\`python
import pandas as pd

# Simple write
df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [30, 25, 35],
    "City": ["NY", "SF", "Chicago"]
})

df.to_excel("output.xlsx", index=False)

# Write multiple sheets
with pd.ExcelWriter("multi_sheet.xlsx", engine='openpyxl') as writer:
    df1.to_excel(writer, sheet_name="Sheet1", index=False)
    df2.to_excel(writer, sheet_name="Sheet2", index=False)
    df3.to_excel(writer, sheet_name="Sheet3", index=False)

# Write with formatting using ExcelWriter
with pd.ExcelWriter("formatted.xlsx", engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name="Data", index=False)
    
    # Get workbook and sheet
    workbook = writer.book
    worksheet = writer.sheets["Data"]
    
    # Apply formatting
    from openpyxl.styles import Font
    for cell in worksheet[1]:  # Header row
        cell.font = Font(bold=True)
\`\`\`

## Modifying Existing Excel Files

### Reading, Modifying, and Saving

\`\`\`python
from openpyxl import load_workbook
from pathlib import Path
import shutil

def modify_excel_safe(
    filepath: str,
    modifications: callable,
    create_backup: bool = True
) -> bool:
    """
    Modify Excel file safely with backup.
    
    Args:
        filepath: Path to Excel file
        modifications: Function that modifies the workbook
        create_backup: Whether to create backup
    
    Returns:
        Success status
    """
    path = Path(filepath)
    
    if not path.exists():
        print(f"File not found: {filepath}")
        return False
    
    try:
        # Create backup
        if create_backup:
            backup_path = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, backup_path)
            print(f"Created backup: {backup_path}")
        
        # Load workbook
        wb = load_workbook(filepath)
        
        # Apply modifications
        modifications(wb)
        
        # Save
        wb.save(filepath)
        print(f"Successfully modified {filepath}")
        return True
        
    except Exception as e:
        print(f"Failed to modify file: {e}")
        # Restore from backup if available
        if create_backup and backup_path.exists():
            shutil.copy2(backup_path, path)
            print("Restored from backup")
        return False

# Usage: Add a column
def add_total_column(wb):
    """Add a 'Total' column that sums Price * Quantity."""
    sheet = wb.active
    
    # Find last column
    last_col = sheet.max_column
    
    # Add header
    sheet.cell(row=1, column=last_col + 1, value="Total")
    
    # Add formulas
    for row in range(2, sheet.max_row + 1):
        price_cell = f"B{row}"  # Assuming price in column B
        qty_cell = f"C{row}"    # Assuming quantity in column C
        formula = f"={price_cell}*{qty_cell}"
        sheet.cell(row=row, column=last_col + 1, value=formula)

modify_excel_safe("sales.xlsx", add_total_column)
\`\`\`

### Updating Cell Values

\`\`\`python
from openpyxl import load_workbook

def update_cell_values(filepath: str, updates: Dict[str, any]) -> None:
    """
    Update specific cells in Excel file.
    
    Args:
        filepath: Path to Excel file
        updates: Dict mapping cell references to new values
                 e.g., {"A1": "New Value", "B2": 100}
    """
    wb = load_workbook(filepath)
    sheet = wb.active
    
    for cell_ref, new_value in updates.items():
        sheet[cell_ref] = new_value
    
    wb.save(filepath)

# Usage
update_cell_values("data.xlsx", {
    "A1": "Updated Header",
    "B2": 999,
    "C3": "=SUM(A1:A10)"
})
\`\`\`

### Adding/Deleting Rows and Columns

\`\`\`python
from openpyxl import load_workbook

def manipulate_rows_cols(filepath: str) -> None:
    """Demonstrate adding and deleting rows/columns."""
    wb = load_workbook(filepath)
    sheet = wb.active
    
    # Insert row at position 2
    sheet.insert_rows(2)
    
    # Insert multiple rows
    sheet.insert_rows(5, amount=3)
    
    # Delete row
    sheet.delete_rows(10)
    
    # Insert column at position B
    sheet.insert_cols(2)
    
    # Delete column
    sheet.delete_cols(5)
    
    wb.save(filepath)
\`\`\`

## Data Extraction and Transformation

### Extracting Specific Data

\`\`\`python
from openpyxl import load_workbook
from typing import List, Dict

def extract_column_data(
    filepath: str,
    column: str,
    sheet_name: str = None
) -> List:
    """Extract all values from a specific column."""
    wb = load_workbook(filepath, data_only=True)
    sheet = wb[sheet_name] if sheet_name else wb.active
    
    # Get column values
    values = []
    for row in sheet.iter_rows(min_col=ord(column) - ord('A') + 1,
                                max_col=ord(column) - ord('A') + 1,
                                values_only=True):
        if row[0] is not None:
            values.append(row[0])
    
    return values

def find_cells_by_value(
    filepath: str,
    search_value: any,
    sheet_name: str = None
) -> List[str]:
    """Find all cells containing a specific value."""
    wb = load_workbook(filepath, data_only=True)
    sheet = wb[sheet_name] if sheet_name else wb.active
    
    matches = []
    for row in sheet.iter_rows():
        for cell in row:
            if cell.value == search_value:
                matches.append(cell.coordinate)
    
    return matches

# Usage
prices = extract_column_data("sales.xlsx", "B")
error_cells = find_cells_by_value("data.xlsx", "#ERROR!")
\`\`\`

### Converting Between pandas and openpyxl

\`\`\`python
import pandas as pd
from openpyxl import load_workbook

def excel_to_dataframe(filepath: str, sheet_name: str = None) -> pd.DataFrame:
    """Convert Excel sheet to pandas DataFrame."""
    return pd.read_excel(filepath, sheet_name=sheet_name)

def dataframe_to_excel(
    df: pd.DataFrame,
    filepath: str,
    sheet_name: str = "Sheet1"
) -> None:
    """Write DataFrame to Excel."""
    df.to_excel(filepath, sheet_name=sheet_name, index=False)

def append_dataframe_to_sheet(
    df: pd.DataFrame,
    filepath: str,
    sheet_name: str = None
) -> None:
    """Append DataFrame to existing Excel sheet."""
    wb = load_workbook(filepath)
    sheet = wb[sheet_name] if sheet_name else wb.active
    
    # Find next empty row
    next_row = sheet.max_row + 1
    
    # Append data
    for r_idx, row in enumerate(df.values):
        for c_idx, value in enumerate(row):
            sheet.cell(row=next_row + r_idx, column=c_idx + 1, value=value)
    
    wb.save(filepath)
\`\`\`

## Real-World Example: Excel Processor for LLM Applications

\`\`\`python
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
from openpyxl import load_workbook, Workbook
from openpyxl.styles import Font, PatternFill
import logging

class ExcelProcessor:
    """
    Production-grade Excel processor for LLM applications.
    
    Enables natural language manipulation of Excel files.
    Foundation for building a Cursor-like tool for Excel.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def read_file(self, filepath: str, sheet_name: str = None) -> Optional[pd.DataFrame]:
        """Read Excel file into DataFrame."""
        try:
            return pd.read_excel(filepath, sheet_name=sheet_name or 0)
        except Exception as e:
            self.logger.error(f"Failed to read file: {e}")
            return None
    
    def write_file(
        self,
        df: pd.DataFrame,
        filepath: str,
        sheet_name: str = "Sheet1",
        include_formatting: bool = True
    ) -> bool:
        """Write DataFrame to Excel with optional formatting."""
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                if include_formatting:
                    self._apply_formatting(writer, sheet_name)
            
            self.logger.info(f"Successfully wrote {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write file: {e}")
            return False
    
    def _apply_formatting(self, writer, sheet_name: str) -> None:
        """Apply basic formatting to sheet."""
        worksheet = writer.sheets[sheet_name]
        
        # Format header row
        for cell in worksheet[1]:
            cell.font = Font(bold=True, size=12)
            cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    
    def add_column(
        self,
        filepath: str,
        column_name: str,
        formula_or_values: any,
        sheet_name: str = None
    ) -> bool:
        """Add a column to existing Excel file."""
        try:
            df = self.read_file(filepath, sheet_name)
            if df is None:
                return False
            
            # Add column
            if callable(formula_or_values):
                df[column_name] = formula_or_values(df)
            else:
                df[column_name] = formula_or_values
            
            # Write back
            return self.write_file(df, filepath, sheet_name or "Sheet1")
            
        except Exception as e:
            self.logger.error(f"Failed to add column: {e}")
            return False
    
    def filter_rows(
        self,
        filepath: str,
        condition: callable,
        output_path: str = None
    ) -> Optional[pd.DataFrame]:
        """Filter rows based on condition."""
        df = self.read_file(filepath)
        if df is None:
            return None
        
        filtered = df[condition(df)]
        
        if output_path:
            self.write_file(filtered, output_path)
        
        return filtered
    
    def get_summary(self, filepath: str, sheet_name: str = None) -> Dict:
        """Get summary of Excel file for LLM context."""
        df = self.read_file(filepath, sheet_name)
        if df is None:
            return {}
        
        summary = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_data": df.head(3).to_dict('records'),
            "null_counts": df.isnull().sum().to_dict()
        }
        
        return summary

# Usage Example: Natural Language Processing
processor = ExcelProcessor()

# Read file
df = processor.read_file("sales.xlsx")

# Get summary for LLM context
summary = processor.get_summary("sales.xlsx")
print("Excel file summary:", summary)

# LLM generates instruction: "Add a column calculating profit margin"
# We parse and execute:
processor.add_column(
    "sales.xlsx",
    "Profit Margin",
    lambda df: (df["Revenue"] - df["Cost"]) / df["Revenue"] * 100
)

# LLM generates: "Filter rows where revenue > 1000"
filtered = processor.filter_rows(
    "sales.xlsx",
    lambda df: df["Revenue"] > 1000,
    output_path="high_revenue.xlsx"
)

print(f"Filtered to {len(filtered)} rows")
\`\`\`

## Best Practices for Excel Processing

### 1. Always Use data_only=True for Reading Formulas

\`\`\`python
# ❌ Reads formula strings
wb = load_workbook("file.xlsx")

# ✅ Reads calculated values
wb = load_workbook("file.xlsx", data_only=True)
\`\`\`

### 2. Handle Missing Values

\`\`\`python
import pandas as pd

# Read with NA values handled
df = pd.read_excel("data.xlsx", na_values=["", "N/A", "null"])

# Fill missing values
df = df.fillna(0)  # or df.fillna(method='ffill')
\`\`\`

### 3. Validate Data Types

\`\`\`python
# Convert columns to appropriate types
df['Date'] = pd.to_datetime(df['Date'])
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
\`\`\`

### 4. Preserve Existing Formatting When Modifying

\`\`\`python
from openpyxl import load_workbook

# Load with formatting preserved
wb = load_workbook("file.xlsx")
# ... make changes ...
wb.save("file.xlsx")  # Formatting is preserved
\`\`\`

## Key Takeaways

1. **Use openpyxl** for full control over Excel structure and formatting
2. **Use pandas** for data manipulation and analysis
3. **Always create backups** before modifying files
4. **Use data_only=True** when reading to get calculated values
5. **Handle errors gracefully** with try/except blocks
6. **Validate file size** before loading to avoid memory issues
7. **Convert between pandas and openpyxl** as needed
8. **Preserve formatting** when modifying existing files
9. **Use ExcelWriter** for writing multiple sheets efficiently
10. **Build modular functions** for reuse in LLM pipelines

These patterns form the foundation for building AI applications that can intelligently manipulate Excel files, enabling natural language interfaces to spreadsheet operations.`,
  videoUrl: undefined,
};
