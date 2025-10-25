export const section1 = {
  id: '2-1',
  title: 'Excel Power User for Finance',
  content: `
# Excel Power User for Finance

## Why Excel Still Dominates Finance

Despite the rise of Python, R, and sophisticated analytics platforms, Microsoft Excel remains the dominant tool in finance. Understanding why is crucial for any aspiring finance professional:

### Industry Reality
- **95%+ of financial professionals** use Excel daily
- **Investment banking models** are built in Excel
- **Private equity LBO models** are Excel-based
- **Corporate FP&A** relies heavily on Excel
- **Interviews** often include Excel modeling tests

### Why Excel Persists
1. **Universal accessibility**: Everyone has it
2. **Visual transparency**: See formulas and data together
3. **Flexibility**: Can model anything
4. **Client expectations**: Clients expect Excel deliverables
5. **Audit trail**: Easy to verify calculations
6. **Real-time collaboration**: With Microsoft 365

### When to Use Excel vs Python
- **Excel for**: Ad-hoc analysis, small datasets, sharing with non-technical stakeholders
- **Python for**: Large datasets, automated pipelines, complex algorithms, production systems

## Essential Financial Formulas

### Time Value of Money Functions

Excel's financial functions are the backbone of valuation and corporate finance:

\`\`\`excel
=NPV(rate, value1, [value2], ...)
// Net Present Value - discounts future cash flows

=IRR(values, [guess])
// Internal Rate of Return - finds discount rate where NPV = 0

=PV(rate, nper, pmt, [fv], [type])
// Present Value - calculates today's value

=FV(rate, nper, pmt, [pv], [type])
// Future Value - calculates future value

=PMT(rate, nper, pv, [fv], [type])
// Payment - calculates loan payment

=RATE(nper, pmt, pv, [fv], [type], [guess])
// Interest rate calculation

=NPER(rate, pmt, pv, [fv], [type])
// Number of periods calculation
\`\`\`

**Real-World Example**: Calculating mortgage payment
\`\`\`excel
=PMT(5%/12, 30*12, -500000)
// Result: $2,684.11 monthly payment
// 5% annual rate divided by 12 months
// 30 years times 12 months = 360 periods
// -$500,000 loan amount (negative = outflow)
\`\`\`

### Lookup and Reference Functions

Critical for building dynamic models that reference other data:

\`\`\`excel
=VLOOKUP(lookup_value, table_array, col_index_num, [range_lookup])
// Vertical lookup - searches first column, returns from specified column

=HLOOKUP(lookup_value, table_array, row_index_num, [range_lookup])
// Horizontal lookup - searches first row

=INDEX(array, row_num, [col_num])
// Returns value at specific position

=MATCH(lookup_value, lookup_array, [match_type])
// Returns position of value in array

=INDEX(array, MATCH(lookup_value, lookup_array, 0))
// INDEX-MATCH combo - more flexible than VLOOKUP

=XLOOKUP(lookup_value, lookup_array, return_array, [if_not_found])
// Modern replacement for VLOOKUP (Excel 365)
\`\`\`

**Real-World Example**: Stock price lookup
\`\`\`excel
// With data: A2:A100 (Tickers), B2:B100 (Prices)
=VLOOKUP("AAPL", A2:B100, 2, FALSE)
// Returns AAPL's price

// Better approach with INDEX-MATCH:
=INDEX(B2:B100, MATCH("AAPL", A2:A100, 0))
// More flexible - can look left or right
\`\`\`

### Logical Functions

Build sophisticated decision trees and conditional logic:

\`\`\`excel
=IF(logical_test, value_if_true, value_if_false)
// Basic conditional logic

=IFS(logical_test1, value1, logical_test2, value2, ...)
// Multiple conditions (Excel 2019+)

=AND(logical1, [logical2], ...)
// Returns TRUE if all arguments are TRUE

=OR(logical1, [logical2], ...)
// Returns TRUE if any argument is TRUE

=NOT(logical)
// Reverses logic

=IFERROR(value, value_if_error)
// Handle errors gracefully
\`\`\`

**Real-World Example**: Credit rating logic
\`\`\`excel
=IFS(
  D2 < 1.5, "Distressed",
  D2 < 2.0, "Weak",
  D2 < 3.0, "Adequate",
  D2 < 4.0, "Strong",
  D2 >= 4.0, "Excellent"
)
// Where D2 = Interest Coverage Ratio
\`\`\`

### Statistical Functions

Essential for financial analysis and risk management:

\`\`\`excel
=AVERAGE(number1, [number2], ...)
// Arithmetic mean

=MEDIAN(number1, [number2], ...)
// Middle value

=STDEV.S(number1, [number2], ...)
// Standard deviation (sample)

=STDEV.P(number1, [number2], ...)
// Standard deviation (population)

=VAR.S(number1, [number2], ...)
// Variance (sample)

=CORREL(array1, array2)
// Correlation coefficient

=COVARIANCE.S(array1, array2)
// Covariance (sample)

=PERCENTILE.INC(array, k)
// Returns k-th percentile
\`\`\`

**Real-World Example**: Portfolio volatility
\`\`\`excel
// Daily returns in column B
=STDEV.S(B2:B252) * SQRT(252)
// Annualized volatility (252 trading days)
\`\`\`

### Array Formulas (Dynamic Arrays in Excel 365)

Powerful formulas that operate on ranges:

\`\`\`excel
=SORT(array, [sort_index], [sort_order], [by_col])
// Sort data dynamically

=FILTER(array, include, [if_empty])
// Filter data based on criteria

=UNIQUE(array, [by_col], [exactly_once])
// Extract unique values

=SEQUENCE(rows, [columns], [start], [step])
// Generate number sequences

=RANDARRAY([rows], [columns], [min], [max], [integer])
// Generate random numbers
\`\`\`

**Real-World Example**: Filter stocks by criteria
\`\`\`excel
=FILTER(A2:E100, (C2:C100 > 1000000000) * (D2:D100 < 20))
// Returns stocks with market cap > $1B and P/E < 20
\`\`\`

## Data Analysis Techniques

### Pivot Tables for Financial Analysis

Pivot tables are indispensable for summarizing and analyzing large datasets:

**Creating a Pivot Table**:
1. Select your data range
2. Insert → PivotTable
3. Drag fields to Rows, Columns, Values areas

**Real-World Example**: Revenue Analysis by Product and Region
\`\`\`
Source Data:
Date | Product | Region | Sales
2024-01-01 | Widget A | North | 10000
2024-01-01 | Widget B | South | 15000
...

Pivot Table Setup:
- Rows: Product
- Columns: Region
- Values: Sum of Sales
- Filters: Date (by quarter)

Result: Dynamic table showing product sales by region
\`\`\`

**Advanced Pivot Table Features**:
- **Calculated Fields**: Create custom calculations
  \`\`\`excel
  // Add calculated field for Margin %
  = (Revenue - COGS) / Revenue
  \`\`\`
- **Grouping**: Group dates by month, quarter, year
- **Show Values As**: % of Total, % of Row/Column, Running Total
- **Slicers**: Visual filters for easy interaction

### What-If Analysis Tools

Excel's suite for scenario modeling and optimization:

#### Goal Seek
Find the input needed to achieve a target output.

**Example**: What interest rate gives me a $2,000 monthly payment?
\`\`\`
Data → What-If Analysis → Goal Seek
- Set cell: B5 (payment formula)
- To value: -2000
- By changing cell: B1 (interest rate)
\`\`\`

#### Data Tables
Test multiple input scenarios simultaneously.

**One-Variable Data Table** (Sensitivity Analysis):
\`\`\`
          | NPV Formula
----------|------------
5%        | =$B$10
6%        |
7%        |
8%        |
9%        |

Select range → Data → What-If Analysis → Data Table
Column input cell: Interest rate cell
\`\`\`

**Two-Variable Data Table**:
\`\`\`
          | 5%    | 6%    | 7%    | 8%
----------|-------|-------|-------|------
Year 1    |       |       |       |
Year 2    |       |       |       |
Year 3    |       |       |       |

Test two variables (growth rate and discount rate)
\`\`\`

#### Solver
Optimization tool for complex problems with constraints.

**Example**: Portfolio Optimization
\`\`\`
Objective: Maximize portfolio return
By changing: Stock allocations (A1:A10)
Constraints:
  - Sum(A1:A10) = 1 (weights sum to 100%)
  - A1:A10 >= 0 (no short selling)
  - Portfolio volatility <= 15%
\`\`\`

## Financial Modeling Best Practices

### Model Structure

Professional financial models follow a consistent structure:

\`\`\`
[Assumptions & Inputs] (Blue text)
    ↓
[Calculations] (Black text)
    ↓
[Outputs & Summaries] (Green text)
\`\`\`

**Color Coding Conventions**:
- **Blue**: Hard-coded inputs (can be changed)
- **Black**: Formulas (don't touch)
- **Green**: Key outputs/summaries
- **Red**: Links from other sheets/files

### Three-Statement Model Layout

Investment banking standard structure:

\`\`\`
Sheet 1: Assumptions
- Revenue growth rates
- Margin assumptions
- CapEx as % of revenue
- Working capital assumptions
- WACC components

Sheet 2: Historical Financials
- Income statement (3-5 years)
- Balance sheet
- Cash flow statement
- Financial ratios

Sheet 3: Income Statement Projection
- Revenue build
- COGS and gross profit
- Operating expenses
- EBITDA, EBIT
- Interest and taxes
- Net income

Sheet 4: Balance Sheet Projection
- Assets (current and long-term)
- Liabilities
- Shareholders' equity
- Working capital schedule

Sheet 5: Cash Flow Statement
- Operating activities
- Investing activities
- Financing activities
- Free cash flow calculation

Sheet 6: DCF Valuation
- Unlevered free cash flow
- Terminal value
- PV of cash flows
- Enterprise value → Equity value

Sheet 7: Outputs & Sensitivity
- Valuation summary
- Key metrics
- Sensitivity tables
\`\`\`

### Formula Best Practices

**1. Avoid Hard-Coding Numbers**
\`\`\`excel
// ❌ Bad
=100000 * 1.05

// ✅ Good
=B10 * (1 + $Assumptions!$B$5)
\`\`\`

**2. Use Named Ranges**
\`\`\`excel
// Define: WACC = Assumptions!B15

// Instead of:
=NPV(Assumptions!$B$15, C20:G20)

// Use:
=NPV(WACC, C20:G20)
\`\`\`

**3. Absolute vs Relative References**
\`\`\`excel
// Relative (changes when copied)
=A1 * B1

// Absolute (never changes)
=$A$1 * $B$1

// Mixed (row or column fixed)
=$A1 * B$1
\`\`\`

**4. Avoid Circular References**
\`\`\`excel
// ❌ Circular reference
// Cell A1: =A1 * 1.05

// ✅ Use iterative calculation or restructure
\`\`\`

**5. Error Handling**
\`\`\`excel
=IFERROR(VLOOKUP(A2, Data!A:B, 2, FALSE), "Not Found")
// Returns "Not Found" instead of #N/A error
\`\`\`

### Documentation

Professional models include:
- **Cover page**: Model purpose, author, version, date
- **Instructions**: How to use the model
- **Assumptions page**: All key assumptions in one place
- **Comments**: Alt + M + M to add cell comments
- **Color coding**: Consistent throughout

## Essential Keyboard Shortcuts

### Navigation
\`\`\`
Ctrl + Home          // Go to A1
Ctrl + End           // Go to last used cell
Ctrl + Arrow         // Jump to edge of data region
Ctrl + Page Down/Up  // Switch between sheets
Alt + Page Down/Up   // Scroll one screen right/left
\`\`\`

### Selection
\`\`\`
Ctrl + Space         // Select entire column
Shift + Space        // Select entire row
Ctrl + Shift + End   // Select to last used cell
Ctrl + A             // Select all (or current region)
\`\`\`

### Editing
\`\`\`
F2                   // Edit cell
F4                   // Toggle absolute/relative references
Ctrl + D             // Fill down
Ctrl + R             // Fill right
Ctrl + K             // Insert hyperlink
Ctrl + ;             // Insert today's date
Ctrl + Shift + :     // Insert current time
\`\`\`

### Formatting
\`\`\`
Ctrl + 1             // Format cells dialog
Ctrl + Shift + $     // Currency format
Ctrl + Shift + %     // Percentage format
Ctrl + Shift + #     // Date format
Alt + H + O + I      // Auto-fit column width
\`\`\`

### Formulas
\`\`\`
Ctrl + \`             // Show formulas
Ctrl + Shift + Enter // Array formula (legacy)
F9                   // Calculate all sheets
Shift + F9           // Calculate active sheet
Alt + =              // AutoSum
\`\`\`

### Time Savers
\`\`\`
Ctrl + T             // Create table
Alt + ; (semicolon)  // Select visible cells only
Ctrl + Shift + L     // Toggle filters
F5 or Ctrl + G       // Go To dialog
Alt + Enter          // New line in cell
\`\`\`

## Real-World Example: DCF Model in Excel

Let's build a simplified DCF model step-by-step:

### Step 1: Assumptions Sheet

\`\`\`excel
// A1: "DCF Valuation Model"
// A3: "Operating Assumptions"
// A4: "Revenue Growth (Years 1-5)"
// B4: 10%, 9%, 8%, 7%, 6%
// A5: "EBITDA Margin"
// B5: 25%
// A6: "Tax Rate"
// B6: 21%
// A7: "CapEx as % Revenue"
// B7: 5%
// A8: "NWC as % Revenue"
// B8: 10%

// A10: "Valuation Assumptions"
// A11: "WACC"
// B11: 10%
// A12: "Terminal Growth Rate"
// B12: 2.5%
// A13: "Current Year Revenue"
// B13: $1,000,000
\`\`\`

### Step 2: Financial Projections

\`\`\`excel
// Income Statement
         Year 1    Year 2    Year 3    Year 4    Year 5
Revenue  =B13*1.1  =C15*1.09 =D15*1.08 =E15*1.07 =F15*1.06
EBITDA   =C15*$B$5 =D15*$B$5 =E15*$B$5 =F15*$B$5 =G15*$B$5
D&A      =-C17*0.1 =-D17*0.1 =-E17*0.1 =-F17*0.1 =-G17*0.1
EBIT     =C17+C18  =D17+D18  =E17+E18  =F17+F18  =G17+G18
Tax      =C19*$B$6 =D19*$B$6 =E19*$B$6 =F19*$B$6 =G19*$B$6
NOPAT    =C19-C20  =D19-D20  =E19-E20  =F19-F20  =G19-G20

// Free Cash Flow
NOPAT    (link from above)
Add: D&A (link from above)
Less: CapEx    =-C15*$B$7
Less: ΔChg NWC =-(C15*$B$8-B15*$B$8)
-----------
FCF      =C21+C18+C24+C25
\`\`\`

### Step 3: DCF Calculation

\`\`\`excel
// A30: "DCF Valuation"
// A31: "PV of Explicit FCF"
// B31: =NPV($B$11, C26:G26)

// A32: "Terminal Value"
// B32: =G26 * (1+$B$12) / ($B$11-$B$12)

// A33: "PV of Terminal Value"
// B33: =B32 / (1+$B$11)^5

// A34: "Enterprise Value"
// B34: =B31+B33

// A36: "Less: Net Debt"
// B36: 100000

// A37: "Equity Value"
// B37: =B34-B36

// A39: "Shares Outstanding"
// B39: 1000000

// A40: "Value Per Share"
// B40: =B37/B39
\`\`\`

### Step 4: Sensitivity Analysis

\`\`\`excel
// Two-way data table: WACC vs Terminal Growth

           2.0%   2.5%   3.0%   3.5%
8%         $X     $X     $X     $X
9%         $X     $X     $X     $X
10%        $X     $X     $X     $X
11%        $X     $X     $X     $X
12%        $X     $X     $X     $X

// Setup: Put valuation formula in top-left
// Select entire table
// Data → What-If Analysis → Data Table
// Row input: Terminal growth rate cell
// Column input: WACC cell
\`\`\`

## Transitioning from Excel to Python

### When to Make the Switch

**Stay in Excel when**:
- Dataset < 100,000 rows
- Ad-hoc analysis
- Sharing with non-technical stakeholders
- Client deliverable expected in Excel

**Move to Python when**:
- Dataset > 100,000 rows
- Repeated analysis (automation)
- Complex algorithms
- Production systems
- API integration needed
- Version control important

### Reading Excel Files in Python

\`\`\`python
import pandas as pd
import openpyxl

# Read Excel file
df = pd.read_excel('financial_model.xlsx', sheet_name='Income Statement')

# Read multiple sheets
sheets = pd.read_excel('model.xlsx', sheet_name=None)  # Dictionary of all sheets

# Read specific range
df = pd.read_excel('model.xlsx', 
                   sheet_name='Data',
                   skiprows=2,  # Skip header rows
                   usecols='A:E',  # Only columns A through E
                   nrows=100)  # Only first 100 rows

# Read with specific dtypes
df = pd.read_excel('model.xlsx',
                   dtype={'Ticker': str, 'Price': float})
\`\`\`

### Writing Excel Files from Python

\`\`\`python
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows

# Simple write
df.to_excel('output.xlsx', sheet_name='Results', index=False)

# Write multiple sheets
with pd.ExcelWriter('multi_sheet.xlsx') as writer:
    df_income.to_excel(writer, sheet_name='Income Statement')
    df_balance.to_excel(writer, sheet_name='Balance Sheet')
    df_cashflow.to_excel(writer, sheet_name='Cash Flow')

# Advanced formatting
wb = load_workbook('template.xlsx')
ws = wb['Sheet1']

# Write data
for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), 1):
    for c_idx, value in enumerate(row, 1):
        cell = ws.cell(row=r_idx, column=c_idx, value=value)
        
        # Format header
        if r_idx == 1:
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color='366092', end_color='366092', fill_type='solid')

# Add formulas
ws['E2'] = '=C2*D2'  # Price * Quantity

# Save
wb.save('formatted_output.xlsx')
\`\`\`

### Replicating Excel Functions in Python

\`\`\`python
import numpy as np
import pandas as pd
from scipy import stats

# Excel: =VLOOKUP()
# Python:
df.merge(lookup_df, on='key', how='left')
# or
df.set_index('key').join(lookup_df.set_index('key'))

# Excel: =IF()
# Python:
df['result'] = np.where(df['value'] > 100, 'High', 'Low')
# or for multiple conditions
df['result'] = np.select(
    [df['value'] < 50, df['value'] < 100, df['value'] >= 100],
    ['Low', 'Medium', 'High']
)

# Excel: =SUMIF()
# Python:
df[df['category'] == 'A']['amount'].sum()
# or
df.groupby('category')['amount'].sum()

# Excel: =NPV()
# Python:
import numpy_financial as npf
npv = npf.npv(rate, cash_flows)

# Excel: =IRR()
# Python:
irr = npf.irr(cash_flows)

# Excel: Pivot Table
# Python:
pivot = df.pivot_table(
    values='sales',
    index='product',
    columns='region',
    aggfunc='sum'
)
\`\`\`

## Common Pitfalls and How to Avoid Them

### 1. Circular References
**Problem**: Formula refers to its own cell
\`\`\`excel
// ❌ Cell A1: =A1 * 1.05
\`\`\`

**Solution**: 
- Restructure model to break circularity
- Use iterative calculation (File → Options → Formulas) only if absolutely necessary

### 2. Broken Links
**Problem**: Model links to external files that moved

**Solution**:
\`\`\`excel
// Check links: Data → Edit Links
// Break links: Copy → Paste Values
// Use INDIRECT() for dynamic references
\`\`\`

### 3. Hidden Rows/Columns with Data
**Problem**: Calculations include hidden cells

**Solution**:
\`\`\`excel
// Use SUBTOTAL() instead of SUM()
=SUBTOTAL(109, A1:A100)  // 109 = SUM, ignores hidden

// Or use AGGREGATE()
=AGGREGATE(9, 5, A1:A100)  // 9 = SUM, 5 = ignore hidden
\`\`\`

### 4. Volatile Functions Slowing Model
**Problem**: Functions that recalculate on every change

**Volatile functions**:
- NOW(), TODAY()
- RAND(), RANDBETWEEN()
- OFFSET()
- INDIRECT()

**Solution**: Use sparingly or convert to values after calculation

### 5. Array Formula Confusion
**Problem**: Legacy array formulas require Ctrl+Shift+Enter

**Solution**: Upgrade to Excel 365 for dynamic arrays or use helper columns

### 6. Dates Stored as Text
**Problem**: "01/01/2024" is text, not a date

**Solution**:
\`\`\`excel
=DATEVALUE("01/01/2024")
// or
=DATE(2024, 1, 1)
\`\`\`

## Production Checklist

Before delivering an Excel model:

- [ ] **Assumptions clearly documented** on dedicated sheet
- [ ] **Color coding consistent** (Blue inputs, Black formulas, Green outputs)
- [ ] **Error checks** throughout model (check sums, balances)
- [ ] **Sensitivity analysis** for key assumptions
- [ ] **Scenario analysis** (Base, Upside, Downside)
- [ ] **Version control**: Save with date/version in filename
- [ ] **Password protection** for sensitive sections
- [ ] **Formulas audited**: Check with Ctrl + \`
- [ ] **Remove unnecessary sheets** and clutter
- [ ] **Test with different inputs** to verify logic
- [ ] **Print area set** if model will be printed
- [ ] **Documentation** sheet with instructions

## Regulatory Considerations

### SOX Compliance (Public Companies)
- **Audit trail**: Document all model changes
- **Version control**: Keep historical versions
- **Access controls**: Limit who can modify
- **Testing**: Validate formulas are correct

### Model Risk Management (Banks)
- **Independent validation**: Second person checks model
- **Documentation**: Assumptions and methodology documented
- **Stress testing**: Test extreme scenarios
- **Limitations**: Clearly state model limitations

## Practice Exercises

### Exercise 1: Build a Mortgage Calculator
Create a mortgage calculator with:
- Loan amount, interest rate, term inputs
- Monthly payment calculation
- Amortization schedule
- Total interest paid
- What-if analysis for different rates

### Exercise 2: Portfolio Tracker
Build a stock portfolio tracker with:
- Stock ticker, shares, purchase price inputs
- Current prices via VLOOKUP or web query
- P&L calculation
- Portfolio allocation (pie chart)
- Performance metrics (return, volatility)

### Exercise 3: Company Valuation
Create a simplified DCF model with:
- 5-year revenue projection
- EBITDA and FCF calculation
- DCF valuation
- Sensitivity table (WACC vs growth)
- Output summary page

## Summary

Excel remains the lingua franca of finance. Mastering these skills will:
- Make you immediately productive in any finance role
- Enable rapid financial modeling and analysis
- Facilitate communication with stakeholders
- Complement your Python skills for comprehensive toolkit

**Next Steps**:
1. Practice building models from scratch
2. Replicate Wall Street models (find templates online)
3. Take on progressively complex analyses
4. Learn keyboard shortcuts until they're muscle memory
5. Start building your model library

Remember: Excel proficiency is expected, not impressive. It's table stakes for finance roles, but combined with Python and data science skills, you become truly valuable.
`,
  quiz: '2-1-quiz',
  discussionQuestions: '2-1-discussion',
};
