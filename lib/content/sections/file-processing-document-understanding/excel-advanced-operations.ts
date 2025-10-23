/**
 * Excel Advanced Operations Section
 * Module 3: File Processing & Document Understanding
 */

export const exceladvancedoperationsSection = {
  id: 'excel-advanced-operations',
  title: 'Excel Advanced Operations',
  content: `# Excel Advanced Operations

Master advanced Excel manipulation for building sophisticated AI applications that handle formulas, charts, pivot tables, and complex formatting.

## Overview: Advanced Excel Features

Building on basic Excel manipulation, this section covers advanced features that make Excel powerful: formulas, charts, conditional formatting, data validation, and pivot tables. These are essential for creating production-ready Excel automation tools.

**Advanced Features Covered:**
- Formula generation and manipulation
- Charts and visualizations
- Conditional formatting
- Data validation rules
- Pivot tables
- Named ranges
- Cell styles and themes
- Excel tables

## Working with Formulas

### Reading Formulas

\`\`\`python
from openpyxl import load_workbook

def read_formulas(filepath: str, sheet_name: str = None):
    """
    Read both formulas and their calculated values.
    
    Important for understanding Excel file logic.
    """
    # Load WITHOUT data_only to see formulas
    wb_formulas = load_workbook(filepath, data_only=False)
    # Load WITH data_only to see values
    wb_values = load_workbook(filepath, data_only=True)
    
    sheet_f = wb_formulas[sheet_name] if sheet_name else wb_formulas.active
    sheet_v = wb_values[sheet_name] if sheet_name else wb_values.active
    
    for row_idx in range(1, sheet_f.max_row + 1):
        for col_idx in range(1, sheet_f.max_column + 1):
            cell_f = sheet_f.cell(row_idx, col_idx)
            cell_v = sheet_v.cell(row_idx, col_idx)
            
            if cell_f.value and str(cell_f.value).startswith('='):
                print(f"{cell_f.coordinate}: {cell_f.value} = {cell_v.value}")

# Usage
read_formulas("financial_model.xlsx", "Projections")
\`\`\`

### Writing Formulas

\`\`\`python
from openpyxl import Workbook

def create_sheet_with_formulas():
    """Create Excel sheet with various formulas."""
    wb = Workbook()
    sheet = wb.active
    
    # Headers
    sheet['A1'] = 'Product'
    sheet['B1'] = 'Price'
    sheet['C1'] = 'Quantity'
    sheet['D1'] = 'Total'
    sheet['E1'] = 'Tax (10%)'
    sheet['F1'] = 'Grand Total'
    
    # Data
    data = [
        ['Widget A', 10.50, 100],
        ['Widget B', 25.00, 50],
        ['Widget C', 15.75, 75]
    ]
    
    for idx, row in enumerate(data, start=2):
        sheet[f'A{idx}'] = row[0]
        sheet[f'B{idx}'] = row[1]
        sheet[f'C{idx}'] = row[2]
        
        # Formula: Price * Quantity
        sheet[f'D{idx}'] = f'=B{idx}*C{idx}'
        
        # Formula: Total * 0.1
        sheet[f'E{idx}'] = f'=D{idx}*0.1'
        
        # Formula: Total + Tax
        sheet[f'F{idx}'] = f'=D{idx}+E{idx}'
    
    # Sum formulas at bottom
    last_row = len(data) + 2
    sheet[f'D{last_row}'] = f'=SUM(D2:D{last_row-1})'
    sheet[f'E{last_row}'] = f'=SUM(E2:E{last_row-1})'
    sheet[f'F{last_row}'] = f'=SUM(F2:F{last_row-1})'
    
    wb.save('formulas_demo.xlsx')

create_sheet_with_formulas()
\`\`\`

### Dynamic Formula Generation

\`\`\`python
from openpyxl import load_workbook
from typing import Dict, List

class FormulaGenerator:
    """
    Generate Excel formulas programmatically.
    
    Essential for building natural language Excel tools.
    """
    
    @staticmethod
    def sum_formula(start_cell: str, end_cell: str) -> str:
        """Generate SUM formula."""
        return f"=SUM({start_cell}:{end_cell})"
    
    @staticmethod
    def average_formula(start_cell: str, end_cell: str) -> str:
        """Generate AVERAGE formula."""
        return f"=AVERAGE({start_cell}:{end_cell})"
    
    @staticmethod
    def if_formula(condition: str, true_val: str, false_val: str) -> str:
        """Generate IF formula."""
        return f"=IF({condition}, {true_val}, {false_val})"
    
    @staticmethod
    def vlookup_formula(lookup_value: str, table_range: str, col_index: int) -> str:
        """Generate VLOOKUP formula."""
        return f"=VLOOKUP({lookup_value}, {table_range}, {col_index}, FALSE)"
    
    @staticmethod
    def concat_formula(cells: List[str], delimiter: str = "") -> str:
        """Generate concatenation formula."""
        if delimiter:
            parts = [f'"{delimiter}"' if i > 0 else '' for i in range(len(cells))]
            combined = []
            for i, cell in enumerate(cells):
                if i > 0:
                    combined.append(f'"{delimiter}"')
                combined.append(cell)
            return f"=CONCATENATE({', '.join(combined)})"
        return f"=CONCATENATE({', '.join(cells)})"
    
    @staticmethod
    def calculate_column(column_letter: str, rows: int, operation: str) -> str:
        """Generate formula for entire column operation."""
        start = f"{column_letter}2"
        end = f"{column_letter}{rows}"
        
        operations = {
            'sum': FormulaGenerator.sum_formula(start, end),
            'average': FormulaGenerator.average_formula(start, end),
            'max': f"=MAX({start}:{end})",
            'min': f"=MIN({start}:{end})",
            'count': f"=COUNT({start}:{end})"
        }
        
        return operations.get(operation.lower(), "")

# Usage: AI generates "calculate average of column B"
formula_gen = FormulaGenerator()
avg_formula = formula_gen.calculate_column('B', 100, 'average')
print(f"Generated formula: {avg_formula}")
\`\`\`

## Creating Charts

### Basic Chart Creation

\`\`\`python
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, PieChart, Reference

def create_bar_chart():
    """Create Excel file with bar chart."""
    wb = Workbook()
    sheet = wb.active
    
    # Add data
    sheet['A1'] = 'Month'
    sheet['B1'] = 'Sales'
    
    data = [
        ['Jan', 1000],
        ['Feb', 1500],
        ['Mar', 1200],
        ['Apr', 1800],
        ['May', 2000]
    ]
    
    for idx, row in enumerate(data, start=2):
        sheet[f'A{idx}'] = row[0]
        sheet[f'B{idx}'] = row[1]
    
    # Create chart
    chart = BarChart()
    chart.title = "Monthly Sales"
    chart.x_axis.title = "Month"
    chart.y_axis.title = "Sales ($)"
    
    # Set data range
    data_ref = Reference(sheet, min_col=2, min_row=1, max_row=len(data)+1)
    cats_ref = Reference(sheet, min_col=1, min_row=2, max_row=len(data)+1)
    
    chart.add_data(data_ref, titles_from_data=True)
    chart.set_categories(cats_ref)
    
    # Add chart to sheet
    sheet.add_chart(chart, "D2")
    
    wb.save('chart_demo.xlsx')

create_bar_chart()
\`\`\`

### Multiple Chart Types

\`\`\`python
from openpyxl import Workbook
from openpyxl.chart import (
    BarChart, LineChart, PieChart, ScatterChart,
    Reference, Series
)

class ChartBuilder:
    """
    Build various chart types for Excel.
    
    Used by AI tools to visualize data on command.
    """
    
    def __init__(self, workbook, sheet):
        self.wb = workbook
        self.sheet = sheet
    
    def create_line_chart(
        self,
        data_range: str,
        title: str = "Line Chart",
        position: str = "E2"
    ):
        """Create line chart."""
        chart = LineChart()
        chart.title = title
        chart.style = 13  # Preset style
        
        data = Reference(self.sheet, range_string=data_range)
        chart.add_data(data, titles_from_data=True)
        
        self.sheet.add_chart(chart, position)
        return chart
    
    def create_pie_chart(
        self,
        data_range: str,
        labels_range: str,
        title: str = "Pie Chart",
        position: str = "E2"
    ):
        """Create pie chart."""
        chart = PieChart()
        chart.title = title
        
        data = Reference(self.sheet, range_string=data_range)
        labels = Reference(self.sheet, range_string=labels_range)
        
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(labels)
        
        self.sheet.add_chart(chart, position)
        return chart
    
    def create_combo_chart(self):
        """Create combination chart (bar + line)."""
        chart1 = BarChart()
        chart2 = LineChart()
        
        # Configure and combine...
        # chart1 += chart2
        # self.sheet.add_chart(chart1, "E2")
        pass

# Usage
wb = Workbook()
sheet = wb.active
builder = ChartBuilder(wb, sheet)

# Add data...
# builder.create_line_chart("B1:B10", "Sales Trend")

wb.save('charts.xlsx')
\`\`\`

## Conditional Formatting

### Basic Conditional Formatting

\`\`\`python
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font
from openpyxl.formatting.rule import CellIsRule

def apply_conditional_formatting():
    """Apply conditional formatting to highlight values."""
    wb = Workbook()
    sheet = wb.active
    
    # Add data
    sheet['A1'] = 'Sales'
    for i in range(2, 12):
        sheet[f'A{i}'] = i * 100
    
    # Define fills
    green_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    red_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    
    # Rule: Greater than 500 -> Green
    rule_high = CellIsRule(
        operator='greaterThan',
        formula=['500'],
        fill=green_fill
    )
    
    # Rule: Less than 300 -> Red
    rule_low = CellIsRule(
        operator='lessThan',
        formula=['300'],
        fill=red_fill
    )
    
    # Apply rules
    sheet.conditional_formatting.add('A2:A11', rule_high)
    sheet.conditional_formatting.add('A2:A11', rule_low)
    
    wb.save('conditional_formatting.xlsx')

apply_conditional_formatting()
\`\`\`

### Advanced Conditional Formatting

\`\`\`python
from openpyxl import Workbook
from openpyxl.styles import Color, PatternFill, Font, Border
from openpyxl.formatting.rule import ColorScaleRule, DataBarRule, IconSetRule

def advanced_conditional_formatting():
    """Apply advanced conditional formatting."""
    wb = Workbook()
    sheet = wb.active
    
    # Add data
    for i in range(1, 21):
        sheet[f'A{i}'] = i * 5
    
    # Color scale: Red to Yellow to Green
    color_scale = ColorScaleRule(
        start_type='min',
        start_color='F8696B',  # Red
        mid_type='percentile',
        mid_value=50,
        mid_color='FFEB84',    # Yellow
        end_type='max',
        end_color='63BE7B'     # Green
    )
    
    # Data bars
    data_bar = DataBarRule(
        start_type='min',
        end_type='max',
        color="638EC6"  # Blue
    )
    
    # Icon set (3 arrows)
    icon_set = IconSetRule(
        icon_style='3Arrows',
        type='percent',
        values=[0, 33, 67]
    )
    
    # Apply
    sheet.conditional_formatting.add('A1:A20', color_scale)
    sheet.conditional_formatting.add('B1:B20', data_bar)
    sheet.conditional_formatting.add('C1:C20', icon_set)
    
    wb.save('advanced_formatting.xlsx')

advanced_conditional_formatting()
\`\`\`

## Data Validation

### Creating Dropdown Lists

\`\`\`python
from openpyxl import Workbook
from openpyxl.worksheet.datavalidation import DataValidation

def create_dropdowns():
    """Create cells with dropdown validation."""
    wb = Workbook()
    sheet = wb.active
    
    # Create dropdown for category selection
    category_dv = DataValidation(
        type="list",
        formula1='"Product,Service,Support"',
        allow_blank=False
    )
    
    # Add validation to column A
    sheet.add_data_validation(category_dv)
    category_dv.add('A2:A100')
    
    # Create number validation (must be between 1-100)
    number_dv = DataValidation(
        type="whole",
        operator="between",
        formula1=1,
        formula2=100,
        error='Please enter a number between 1 and 100'
    )
    
    sheet.add_data_validation(number_dv)
    number_dv.add('B2:B100')
    
    # Add headers
    sheet['A1'] = 'Category'
    sheet['B1'] = 'Quantity'
    
    wb.save('validation_demo.xlsx')

create_dropdowns()
\`\`\`

### Custom Validation Rules

\`\`\`python
from openpyxl import Workbook
from openpyxl.worksheet.datavalidation import DataValidation

def custom_validation():
    """Create custom validation rules."""
    wb = Workbook()
    sheet = wb.active
    
    # Date validation (only future dates)
    date_dv = DataValidation(
        type="date",
        operator="greaterThan",
        formula1="TODAY()",
        error="Date must be in the future"
    )
    sheet.add_data_validation(date_dv)
    date_dv.add('A2:A100')
    
    # Email validation (custom formula)
    email_dv = DataValidation(
        type="custom",
        formula1='=AND(FIND("@",A2)>0, FIND(".",A2)>FIND("@",A2))',
        error="Please enter a valid email"
    )
    sheet.add_data_validation(email_dv)
    email_dv.add('B2:B100')
    
    # Add headers
    sheet['A1'] = 'Future Date'
    sheet['B1'] = 'Email'
    
    wb.save('custom_validation.xlsx')

custom_validation()
\`\`\`

## Named Ranges

### Creating and Using Named Ranges

\`\`\`python
from openpyxl import Workbook
from openpyxl.workbook.defined_name import DefinedName

def create_named_ranges():
    """Create named ranges for easier formula management."""
    wb = Workbook()
    sheet = wb.active
    
    # Add data
    sheet['A1'] = 'Revenue'
    for i in range(2, 12):
        sheet[f'A{i}'] = i * 1000
    
    # Create named range
    wb.create_named_range('TotalRevenue', sheet, 'A2:A11')
    
    # Use named range in formula
    sheet['B1'] = 'Total'
    sheet['B2'] = '=SUM(TotalRevenue)'
    
    sheet['C1'] = 'Average'
    sheet['C2'] = '=AVERAGE(TotalRevenue)'
    
    wb.save('named_ranges.xlsx')

create_named_ranges()
\`\`\`

## Excel Tables

### Creating Excel Tables

\`\`\`python
from openpyxl import Workbook
from openpyxl.worksheet.table import Table, TableStyleInfo

def create_excel_table():
    """
    Create Excel Table (formatted range with built-in features).
    
    Tables provide automatic filtering, sorting, and structured references.
    """
    wb = Workbook()
    sheet = wb.active
    
    # Add data
    headers = ['Name', 'Department', 'Salary', 'Start Date']
    data = [
        ['Alice', 'Engineering', 95000, '2020-01-15'],
        ['Bob', 'Sales', 75000, '2019-06-01'],
        ['Charlie', 'Marketing', 65000, '2021-03-10'],
        ['Diana', 'Engineering', 105000, '2018-09-20']
    ]
    
    # Write headers and data
    sheet.append(headers)
    for row in data:
        sheet.append(row)
    
    # Create table
    table = Table(displayName="EmployeeTable", ref="A1:D5")
    
    # Add style
    style = TableStyleInfo(
        name="TableStyleMedium2",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False
    )
    table.tableStyleInfo = style
    
    # Add table to sheet
    sheet.add_table(table)
    
    wb.save('excel_table.xlsx')

create_excel_table()
\`\`\`

## Advanced Styling

### Cell Styles and Themes

\`\`\`python
from openpyxl import Workbook
from openpyxl.styles import (
    Font, Alignment, Border, Side, PatternFill,
    NamedStyle
)

def create_custom_styles():
    """Create reusable custom styles."""
    wb = Workbook()
    sheet = wb.active
    
    # Create custom style for headers
    header_style = NamedStyle(name="header")
    header_style.font = Font(bold=True, size=14, color="FFFFFF")
    header_style.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_style.alignment = Alignment(horizontal="center", vertical="center")
    header_style.border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    
    # Register style
    wb.add_named_style(header_style)
    
    # Create currency style
    currency_style = NamedStyle(name="currency", number_format='$#,##0.00')
    wb.add_named_style(currency_style)
    
    # Apply styles
    sheet['A1'] = 'Product'
    sheet['B1'] = 'Price'
    sheet['A1'].style = 'header'
    sheet['B1'].style = 'header'
    
    sheet['A2'] = 'Widget'
    sheet['B2'] = 19.99
    sheet['B2'].style = 'currency'
    
    wb.save('custom_styles.xlsx')

create_custom_styles()
\`\`\`

## Production Example: Advanced Excel Automation

\`\`\`python
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from openpyxl import load_workbook, Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.formatting.rule import ColorScaleRule

class AdvancedExcelAutomation:
    """
    Production-grade advanced Excel automation.
    
    Foundation for building Cursor-like Excel tools.
    """
    
    def __init__(self):
        self.formula_gen = FormulaGenerator()
    
    def create_dashboard(
        self,
        data: pd.DataFrame,
        output_path: str,
        title: str = "Dashboard"
    ) -> bool:
        """
        Create Excel dashboard with data, charts, and formatting.
        
        Example: "Create a sales dashboard" command
        """
        try:
            wb = Workbook()
            sheet = wb.active
            sheet.title = title
            
            # Write title
            sheet['A1'] = title
            sheet['A1'].font = Font(size=16, bold=True)
            sheet.merge_cells('A1:F1')
            
            # Write data
            for r_idx, row in enumerate(data.itertuples(index=False), start=3):
                for c_idx, value in enumerate(row, start=1):
                    sheet.cell(row=r_idx, column=c_idx, value=value)
            
            # Write headers
            for c_idx, col_name in enumerate(data.columns, start=1):
                cell = sheet.cell(row=2, column=c_idx, value=col_name)
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            
            # Add summary formulas
            last_row = len(data) + 3
            sheet[f'A{last_row}'] = 'Total'
            sheet[f'A{last_row}'].font = Font(bold=True)
            
            # Add SUM formulas for numeric columns
            for c_idx in range(2, len(data.columns) + 1):
                if pd.api.types.is_numeric_dtype(data.iloc[:, c_idx-1]):
                    formula = self.formula_gen.sum_formula(
                        f"{chr(64+c_idx)}3",
                        f"{chr(64+c_idx)}{last_row-1}"
                    )
                    sheet.cell(row=last_row, column=c_idx, value=formula)
            
            # Add chart if numeric data exists
            numeric_cols = [i for i, col in enumerate(data.columns) if pd.api.types.is_numeric_dtype(data[col])]
            if numeric_cols:
                self._add_chart(sheet, len(data), numeric_cols[0] + 1)
            
            # Apply conditional formatting
            if numeric_cols:
                col_letter = chr(65 + numeric_cols[0])
                self._apply_conditional_formatting(sheet, f"{col_letter}3:{col_letter}{last_row-1}")
            
            wb.save(output_path)
            return True
            
        except Exception as e:
            print(f"Failed to create dashboard: {e}")
            return False
    
    def _add_chart(self, sheet, data_rows: int, data_col: int):
        """Add bar chart to sheet."""
        chart = BarChart()
        chart.title = "Data Visualization"
        
        data = Reference(sheet, min_col=data_col, min_row=2, max_row=data_rows + 2)
        cats = Reference(sheet, min_col=1, min_row=3, max_row=data_rows + 2)
        
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(cats)
        
        sheet.add_chart(chart, f"{chr(65 + data_col + 2)}3")
    
    def _apply_conditional_formatting(self, sheet, range_str: str):
        """Apply color scale formatting."""
        rule = ColorScaleRule(
            start_type='min',
            start_color='F8696B',
            mid_type='percentile',
            mid_value=50,
            mid_color='FFEB84',
            end_type='max',
            end_color='63BE7B'
        )
        sheet.conditional_formatting.add(range_str, rule)

# Usage: Natural language command
automation = AdvancedExcelAutomation()

# Create sample data
data = pd.DataFrame({
    'Product': ['A', 'B', 'C', 'D'],
    'Sales': [1000, 1500, 1200, 1800],
    'Profit': [200, 300, 250, 400]
})

# "Create a dashboard from this data"
automation.create_dashboard(data, 'dashboard.xlsx', 'Sales Dashboard')
\`\`\`

## Key Takeaways

1. **Use formula generators** for dynamic formula creation
2. **Create charts programmatically** for data visualization
3. **Apply conditional formatting** to highlight important data
4. **Use data validation** to ensure data quality
5. **Named ranges** make formulas more readable
6. **Excel tables** provide built-in filtering and sorting
7. **Custom styles** ensure consistent formatting
8. **Combine features** to create professional dashboards
9. **Test formulas** before applying to production files
10. **Document formula logic** for maintenance

These advanced Excel operations enable building sophisticated AI tools that can manipulate spreadsheets at an expert level, essential for a Cursor-like Excel application.`,
  videoUrl: undefined,
};
