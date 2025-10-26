export const financialModelingFundamentals = {
  title: 'Financial Modeling Fundamentals',
  id: 'financial-modeling-fundamentals',
  content: `
# Financial Modeling Fundamentals

## Introduction

Financial modeling is the art and science of building quantitative representations of business operations and financial performance. Unlike simple spreadsheets, **professional financial models** are structured, flexible, auditable tools that support critical decisions involving millions or billions of dollars.

**Why financial modeling matters:**

- **Investment Banking**: M&A models, DCF valuations, LBO analysis
- **Private Equity**: Acquisition models, portfolio company monitoring
- **Corporate Finance**: Capital budgeting, strategic planning, forecasting
- **Equity Research**: Stock valuations, earnings models
- **Asset Management**: Portfolio construction, risk analysis

A single modeling error can cost millions. In 2012, JPMorgan's "London Whale" trading loss was partly attributed to a flawed Value-at-Risk model with copy-paste errors. The lesson: **model quality matters**.

By the end of this section, you'll understand:
- What makes a professional financial model
- Best practices for model structure and documentation
- Excel vs Python trade-offs
- Common pitfalls and how to avoid them
- How to build flexible, auditable models

**Philosophy**: Build models that someone else (or future you) can understand, audit, and modify without breaking.

---

## What is Financial Modeling?

### Definition

**Financial model**: A mathematical representation of a company's operations, expressed through linked financial statements and supporting schedules, used to forecast performance and value the business.

**Key characteristics:**
1. **Forward-looking**: Projects future performance based on assumptions
2. **Integrated**: Income statement, balance sheet, and cash flow link together
3. **Flexible**: Allows scenario analysis by changing assumptions
4. **Documented**: Clear labeling, color coding, and notes
5. **Auditable**: Calculations can be traced and verified

### Types of Financial Models

\`\`\`python
"""
Financial Model Types and Use Cases
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict

class ModelType(Enum):
    """Common financial model types"""
    THREE_STATEMENT = "3-Statement Model"
    DCF = "Discounted Cash Flow"
    LBO = "Leveraged Buyout"
    MERGER = "Merger & Acquisition"
    BUDGET = "Budget/Forecast"
    PROJECT_FINANCE = "Project Finance"
    CONSOLIDATION = "Financial Consolidation"
    SCENARIO = "Scenario Planning"

@dataclass
class ModelUseCase:
    """Financial model use case"""
    model_type: ModelType
    primary_users: List[str]
    time_horizon: str
    complexity: str
    typical_size: str
    
    def describe(self) -> str:
        return f"""
{self.model_type.value}
Primary Users: {', '.join(self.primary_users)}
Time Horizon: {self.time_horizon}
Complexity: {self.complexity}
Typical Size: {self.typical_size}
"""

# Define use cases
use_cases = [
    ModelUseCase(
        ModelType.THREE_STATEMENT,
        ["Corporate finance", "FP&A teams", "Investors"],
        "3-5 years",
        "Medium",
        "50-200 rows"
    ),
    ModelUseCase(
        ModelType.DCF,
        ["Investment bankers", "Equity researchers", "Private equity"],
        "5-10 years",
        "Medium-High",
        "100-300 rows"
    ),
    ModelUseCase(
        ModelType.LBO,
        ["Private equity", "Investment banks (M&A)"],
        "5-7 years",
        "High",
        "200-500 rows"
    ),
    ModelUseCase(
        ModelType.MERGER,
        ["Corporate development", "Investment banks"],
        "3-5 years",
        "High",
        "300-600 rows"
    ),
]

print("Financial Model Types:")
print("=" * 70)
for uc in use_cases:
    print(uc.describe())
\`\`\`

**Output:**
\`\`\`
Financial Model Types:
======================================================================

3-Statement Model
Primary Users: Corporate finance, FP&A teams, Investors
Time Horizon: 3-5 years
Complexity: Medium
Typical Size: 50-200 rows

DCF
Primary Users: Investment bankers, Equity researchers, Private equity
Time Horizon: 5-10 years
Complexity: Medium-High
Typical Size: 100-300 rows

[... continues for all model types]
\`\`\`

---

## Model Structure: Best Practices

### The Golden Rule: Separation of Concerns

**Professional models separate:**

1. **Inputs (Assumptions)**: All user-changeable assumptions in one place
2. **Calculations**: Formulas and logic (reference inputs, never hard-code)
3. **Outputs (Results)**: Summary tables, charts, key metrics

**Why this matters**: Enables easy scenario analysis and reduces errors.

### Standard Model Layout

\`\`\`
Model Structure (Excel or Code):

├── Cover Page
│   ├── Model purpose and description
│   ├── Author, date, version
│   └── Table of contents
│
├── Assumptions & Inputs
│   ├── Revenue drivers (growth rates, pricing, volume)
│   ├── Operating assumptions (margins, expenses %)
│   ├── Capital assumptions (CapEx, depreciation, working capital)
│   ├── Financing assumptions (debt, interest rates)
│   └── Valuation assumptions (WACC, terminal growth)
│
├── Historical Financials
│   ├── Income statement (3-5 years)
│   ├── Balance sheet
│   └── Cash flow statement
│
├── Calculations
│   ├── Revenue build-up
│   ├── Operating expense schedules
│   ├── Working capital schedule
│   ├── PP&E and depreciation schedule
│   ├── Debt schedule
│   └── Tax calculation
│
├── Projected Financial Statements
│   ├── Income statement (5-10 years)
│   ├── Balance sheet
│   └── Cash flow statement
│
├── Valuation
│   ├── DCF analysis
│   ├── Comparable company analysis
│   └── Sensitivity tables
│
└── Outputs & Summary
    ├── Key metrics dashboard
    ├── Executive summary
    └── Charts and visualizations
\`\`\`

### Implementation in Python

\`\`\`python
"""
Model Structure Framework
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

@dataclass
class ModelInputs:
    """
    All model assumptions and inputs.
    Centralized for easy scenario analysis.
    """
    # Revenue assumptions
    revenue_growth_rate: float = 0.10  # 10% annually
    price_inflation: float = 0.02
    volume_growth: float = 0.08
    
    # Operating assumptions
    cogs_pct_revenue: float = 0.40
    opex_pct_revenue: float = 0.25
    
    # Capital assumptions
    capex_pct_revenue: float = 0.05
    depreciation_pct_ppe: float = 0.10
    
    # Working capital assumptions
    days_receivable: float = 45.0
    days_inventory: float = 60.0
    days_payable: float = 30.0
    
    # Financing assumptions
    interest_rate: float = 0.065
    tax_rate: float = 0.21
    
    # Valuation assumptions
    wacc: float = 0.095
    terminal_growth: float = 0.025
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export"""
        return {k: v for k, v in self.__dict__.items()}
    
    def update(self, **kwargs) -> 'ModelInputs':
        """Update specific assumptions"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid assumption: {key}")
        return self

@dataclass
class ModelMetadata:
    """Model documentation and versioning"""
    name: str
    version: str
    author: str
    created_date: datetime
    modified_date: datetime
    purpose: str
    assumptions_summary: str
    
    def __str__(self) -> str:
        return f"""
Financial Model: {self.name}
Version: {self.version}
Author: {self.author}
Created: {self.created_date.strftime('%Y-%m-%d')}
Last Modified: {self.modified_date.strftime('%Y-%m-%d')}

Purpose:
{self.purpose}

Key Assumptions:
{self.assumptions_summary}
"""

class FinancialModel:
    """
    Base financial model structure.
    Implements best practices for organization and auditability.
    """
    
    def __init__(
        self,
        name: str,
        author: str,
        purpose: str,
        inputs: Optional[ModelInputs] = None
    ):
        self.metadata = ModelMetadata(
            name=name,
            version="1.0",
            author=author,
            created_date=datetime.now(),
            modified_date=datetime.now(),
            purpose=purpose,
            assumptions_summary="See inputs for details"
        )
        
        self.inputs = inputs or ModelInputs()
        self.historical_data: Optional[pd.DataFrame] = None
        self.projections: Optional[pd.DataFrame] = None
        self.calculations: Dict[str, pd.DataFrame] = {}
        self.outputs: Dict[str, Any] = {}
    
    def load_historical_data(self, data: pd.DataFrame) -> None:
        """Load historical financials"""
        self.historical_data = data.copy()
        print(f"Loaded historical data: {len(data)} periods")
    
    def run_projections(self, periods: int = 5) -> pd.DataFrame:
        """
        Generate financial projections.
        Override in specific model implementations.
        """
        raise NotImplementedError("Subclass must implement run_projections()")
    
    def calculate_outputs(self) -> Dict[str, Any]:
        """
        Calculate key outputs and metrics.
        Override in specific model implementations.
        """
        raise NotImplementedError("Subclass must implement calculate_outputs()")
    
    def export_summary(self) -> str:
        """Export model summary"""
        return str(self.metadata)
    
    def version_bump(self, change_description: str) -> None:
        """Increment version and update metadata"""
        major, minor = map(int, self.metadata.version.split('.'))
        self.metadata.version = f"{major}.{minor + 1}"
        self.metadata.modified_date = datetime.now()
        print(f"Version updated to {self.metadata.version}")
        print(f"Changes: {change_description}")

# Example usage
model = FinancialModel(
    name="ACME Corp Valuation Model",
    author="Jane Analyst",
    purpose="DCF valuation for potential acquisition"
)

print(model.export_summary())

# Update assumptions for sensitivity analysis
model.inputs.update(
    revenue_growth_rate=0.12,
    wacc=0.10
)

print("\\nUpdated Assumptions:")
print(f"Revenue Growth: {model.inputs.revenue_growth_rate:.1%}")
print(f"WACC: {model.inputs.wacc:.1%}")
\`\`\`

---

## Excel vs Python for Financial Modeling

### The Excel Paradigm

**Excel dominates finance for reasons:**

✅ **Visual and intuitive**: See formulas, trace precedents/dependents  
✅ **Universal**: Everyone has Excel, easy to share  
✅ **Flexible**: Quick ad-hoc analysis, no programming required  
✅ **Familiar**: Industry standard, expected by clients/colleagues  
✅ **Built-in functions**: Financial functions (NPV, IRR, PMT), data tables

**But Excel has limitations:**

❌ **Error-prone**: Easy to accidentally overwrite formulas, hard to debug  
❌ **Not scalable**: Large models slow down, crash with too much data  
❌ **Version control nightmare**: Multiple versions floating around  
❌ **Limited automation**: Can't easily batch-process 100 companies  
❌ **Black box**: Hard to audit complex nested formulas

### The Python Advantage

**Python for financial modeling offers:**

✅ **Reproducible**: Code is version-controlled (Git), traceable  
✅ **Scalable**: Handle millions of rows, run 1000 Monte Carlo simulations  
✅ **Automated**: Build models for 100 companies overnight  
✅ **Testable**: Unit tests ensure calculations are correct  
✅ **Powerful**: Advanced analytics, machine learning, API integration  
✅ **Documented**: Comments, docstrings explain logic

**But Python has trade-offs:**

❌ **Steeper learning curve**: Requires programming skills  
❌ **Less intuitive**: Can't "see" the model like in Excel  
❌ **Not universal**: Clients/colleagues may not have Python setup  
❌ **Requires more structure**: Need to design architecture upfront

### The Hybrid Approach

**Best practice**: Use both tools strategically.

\`\`\`python
"""
Hybrid Workflow: Python + Excel
"""

import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows

class ExcelExporter:
    """
    Export Python model outputs to Excel.
    Combines Python's power with Excel's familiarity.
    """
    
    def __init__(self, filename: str):
        self.filename = filename
        self.wb = Workbook()
        # Remove default sheet
        self.wb.remove(self.wb.active)
    
    def add_sheet_from_dataframe(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        freeze_panes: str = "B2",
        format_as_table: bool = True
    ) -> None:
        """Add DataFrame as Excel sheet with formatting"""
        ws = self.wb.create_sheet(sheet_name)
        
        # Write data
        for r in dataframe_to_rows(df, index=True, header=True):
            ws.append(r)
        
        # Format header row
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True)
        
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
        
        # Freeze panes
        ws.freeze_panes = freeze_panes
        
        # Auto-adjust column widths
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column_letter].width = adjusted_width
    
    def save(self) -> None:
        """Save workbook"""
        self.wb.save(self.filename)
        print(f"Excel file saved: {self.filename}")

# Example: Run model in Python, export to Excel
def hybrid_workflow_example():
    """Demonstrate Python + Excel workflow"""
    
    # 1. Build model in Python (fast, automated)
    companies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Simulate valuation analysis
    results = []
    for ticker in companies:
        results.append({
            'Ticker': ticker,
            'Current Price': np.random.uniform(50, 500),
            'DCF Value': np.random.uniform(50, 500),
            'P/E Ratio': np.random.uniform(10, 50),
            'Recommendation': np.random.choice(['Buy', 'Hold', 'Sell'])
        })
    
    df_valuations = pd.DataFrame(results)
    
    # 2. Export to Excel (client-friendly)
    exporter = ExcelExporter('valuation_summary.xlsx')
    exporter.add_sheet_from_dataframe(df_valuations, 'Valuation Summary')
    
    # Add supporting schedules
    df_assumptions = pd.DataFrame({
        'Parameter': ['WACC', 'Terminal Growth', 'Tax Rate'],
        'Value': [0.095, 0.025, 0.21],
        'Source': ['CAPM', 'Historical GDP', 'Statutory']
    })
    exporter.add_sheet_from_dataframe(df_assumptions, 'Assumptions')
    
    exporter.save()
    
    return df_valuations

# Run hybrid workflow
results = hybrid_workflow_example()
print("\\nValuation Results:")
print(results.to_string(index=False))
\`\`\`

**Output:**
\`\`\`
Excel file saved: valuation_summary.xlsx

Valuation Results:
Ticker  Current Price  DCF Value  P/E Ratio Recommendation
  AAPL         134.52     287.43      23.41            Buy
  MSFT         423.18     156.89      45.23           Hold
[...]
\`\`\`

**When to use each:**

| Use Case | Excel | Python | Hybrid |
|----------|-------|--------|--------|
| Quick ad-hoc analysis | ✓ | | |
| Client-facing model | ✓ | | ✓ |
| Batch processing 100+ companies | | ✓ | |
| Complex Monte Carlo (10K simulations) | | ✓ | |
| Version-controlled, auditable | | ✓ | ✓ |
| Team collaboration | | ✓ | ✓ |
| Financial forecasting | ✓ | | ✓ |
| Algorithmic trading models | | ✓ | |

---

## Color Coding and Documentation

### Excel Color Conventions

**Industry-standard color scheme:**

- **Blue**: Input cells (user can change)
- **Black**: Calculated formulas (reference other cells)
- **Green**: Links to other sheets or external files
- **Red**: Hard-coded numbers in formulas (should be minimized)

**Why color coding matters:**

> "Color coding allowed me to audit a 500-row M&A model in 2 hours instead of 2 days. I could instantly identify inputs vs calculations."  
> — Investment Banking Analyst

\`\`\`python
"""
Color Coding in Excel via Python
"""

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill

def create_color_coded_model():
    """Create Excel model with color coding"""
    wb = Workbook()
    ws = wb.active
    ws.title = "Color Coded Model"
    
    # Define color schemes
    input_fill = PatternFill(start_color="B4C7E7", end_color="B4C7E7", fill_type="solid")  # Blue
    input_font = Font(color="000000", bold=True)  # Black text
    
    calc_fill = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")  # White
    calc_font = Font(color="000000")  # Black text
    
    # Assumptions section (inputs)
    ws['A1'] = 'ASSUMPTIONS'
    ws['A1'].font = Font(bold=True, size=14)
    
    ws['A3'] = 'Revenue Growth Rate'
    ws['B3'] = 0.10
    ws['B3'].fill = input_fill
    ws['B3'].font = input_font
    ws['B3'].number_format = '0.0%'
    
    ws['A4'] = 'Operating Margin'
    ws['B4'] = 0.25
    ws['B4'].fill = input_fill
    ws['B4'].font = input_font
    ws['B4'].number_format = '0.0%'
    
    # Calculations section
    ws['A7'] = 'CALCULATIONS'
    ws['A7'].font = Font(bold=True, size=14)
    
    ws['A9'] = 'Base Revenue'
    ws['B9'] = 1000000
    ws['B9'].fill = input_fill
    ws['B9'].font = input_font
    ws['B9'].number_format = '$#,##0'
    
    ws['A10'] = 'Projected Revenue (Year 1)'
    ws['B10'] = '=B9*(1+B3)'  # Formula references inputs
    ws['B10'].fill = calc_fill
    ws['B10'].font = calc_font
    ws['B10'].number_format = '$#,##0'
    
    ws['A11'] = 'Operating Income'
    ws['B11'] = '=B10*B4'
    ws['B11'].fill = calc_fill
    ws['B11'].font = calc_font
    ws['B11'].number_format = '$#,##0'
    
    # Add legend
    ws['A15'] = 'Legend:'
    ws['A16'] = 'Blue Background = Input (change freely)'
    ws['A16'].fill = input_fill
    ws['A17'] = 'White Background = Calculated (do not overwrite)'
    ws['A17'].fill = calc_fill
    
    wb.save('color_coded_model.xlsx')
    print("Color-coded model created: color_coded_model.xlsx")

create_color_coded_model()
\`\`\`

### Documentation Best Practices

**Every professional model should include:**

1. **Cover Page**
   - Model purpose and scope
   - Author and date
   - Version history
   - Key assumptions summary

2. **Assumption Documentation**
   - Source for each assumption
   - Rationale (why this number?)
   - Sensitivity to changes

3. **Formula Comments**
   - Excel: Add cell comments (Alt+M+M)
   - Python: Inline comments and docstrings

4. **Audit Trail**
   - Change log (what changed, when, why)
   - Version control (Git for Python, Excel compare for Excel)

\`\`\`python
"""
Self-Documenting Model Code
"""

from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime

class DocumentedModel:
    """
    Financial model with built-in documentation.
    Every calculation includes rationale and source.
    """
    
    def __init__(self):
        self.assumptions: Dict[str, Tuple[float, str, str]] = {}
        self.calculations: List[str] = []
        self.change_log: List[Dict] = []
    
    def add_assumption(
        self,
        name: str,
        value: float,
        source: str,
        rationale: str
    ) -> None:
        """
        Add documented assumption.
        
        Args:
            name: Assumption name
            value: Numeric value
            source: Where did this number come from?
            rationale: Why did we choose this value?
        """
        self.assumptions[name] = (value, source, rationale)
        self.log_change(f"Added assumption: {name} = {value}")
    
    def get_assumption(self, name: str) -> float:
        """Retrieve assumption value"""
        if name not in self.assumptions:
            raise ValueError(f"Assumption '{name}' not found")
        return self.assumptions[name][0]
    
    def document_calculation(self, description: str) -> None:
        """Log a calculation step"""
        self.calculations.append(description)
    
    def log_change(self, description: str) -> None:
        """Log a model change"""
        self.change_log.append({
            'timestamp': datetime.now(),
            'change': description
        })
    
    def export_documentation(self) -> str:
        """Generate complete model documentation"""
        doc = ["=" * 70, "MODEL DOCUMENTATION", "=" * 70, ""]
        
        # Assumptions
        doc.append("ASSUMPTIONS:")
        doc.append("-" * 70)
        for name, (value, source, rationale) in self.assumptions.items():
            doc.append(f"\\n{name}: {value}")
            doc.append(f"  Source: {source}")
            doc.append(f"  Rationale: {rationale}")
        
        # Calculations
        doc.append("\\n" + "=" * 70)
        doc.append("CALCULATION STEPS:")
        doc.append("-" * 70)
        for i, calc in enumerate(self.calculations, 1):
            doc.append(f"{i}. {calc}")
        
        # Change log
        doc.append("\\n" + "=" * 70)
        doc.append("CHANGE LOG:")
        doc.append("-" * 70)
        for entry in self.change_log[-10:]:  # Last 10 changes
            doc.append(f"{entry['timestamp']}: {entry['change']}")
        
        return "\\n".join(doc)

# Example usage
model = DocumentedModel()

model.add_assumption(
    name="revenue_growth",
    value=0.15,
    source="Company guidance (Q3 2024 earnings call)",
    rationale="Management projects 15% growth driven by new product launch and market expansion"
)

model.add_assumption(
    name="wacc",
    value=0.095,
    source="CAPM calculation (risk-free rate 4.5%, beta 1.2, market risk premium 6%)",
    rationale="Beta calculated using 5-year weekly returns vs S&P 500"
)

model.document_calculation("Calculate projected revenue: Base × (1 + growth)^years")
model.document_calculation("Calculate free cash flow: NOPAT + D&A - CapEx - ∆NWC")
model.document_calculation("Discount FCF to present value using WACC")

print(model.export_documentation())
\`\`\`

---

## Common Pitfalls and How to Avoid Them

### 1. Hard-Coded Numbers in Formulas

**The Problem:**

\`\`\`excel
# ❌ BAD: Hard-coded tax rate
=EBIT * (1 - 0.21)

# ✅ GOOD: Reference tax rate assumption
=EBIT * (1 - TaxRate)
\`\`\`

**Why it's bad**: If tax rate changes, you must find and update every instance. Easy to miss one.

**Python equivalent:**

\`\`\`python
# ❌ BAD
net_income = ebit * (1 - 0.21)

# ✅ GOOD
TAX_RATE = 0.21  # Single source of truth
net_income = ebit * (1 - TAX_RATE)

# ✅ BETTER
class Assumptions:
    tax_rate: float = 0.21

net_income = ebit * (1 - assumptions.tax_rate)
\`\`\`

### 2. Circular References

**The Problem**: Balance sheet cash depends on debt, which depends on cash balance. Creates circular dependency.

**Solution**: Use Excel's iterative calculation or, in Python, solve iteratively.

\`\`\`python
"""
Handling Circular References
"""

def solve_circular_model(
    initial_cash: float,
    target_min_cash: float,
    interest_rate: float,
    fcf: float,
    max_iterations: int = 100,
    tolerance: float = 0.01
) -> Dict[str, float]:
    """
    Solve model with circular reference between cash and debt.
    
    Circular logic:
    - If cash < minimum, draw debt
    - Debt accrues interest
    - Interest reduces cash
    - Need to iterate to convergence
    """
    
    cash = initial_cash
    debt = 0.0
    
    for iteration in range(max_iterations):
        old_cash = cash
        
        # Calculate debt needed
        if cash < target_min_cash:
            debt = target_min_cash - cash
        else:
            debt = 0.0
        
        # Interest expense reduces cash
        interest = debt * interest_rate
        
        # Update cash
        cash = initial_cash + fcf - interest
        
        # Check convergence
        if abs(cash - old_cash) < tolerance:
            return {
                'cash': cash,
                'debt': debt,
                'interest': interest,
                'iterations': iteration + 1
            }
    
    raise ValueError("Model did not converge")

# Example
result = solve_circular_model(
    initial_cash=50_000,
    target_min_cash=100_000,
    interest_rate=0.05,
    fcf=30_000
)

print("Circular Model Solution:")
for key, value in result.items():
    if key != 'iterations':
        print(f"{key.title()}: ${value:,.0f}")
    else:
        print(f"{key.title()}: {value}")
\`\`\`

### 3. Copy-Paste Errors

**The Problem**: Copying formulas without adjusting references correctly.

**Prevention**:
- Use named ranges (Excel) or variables (Python)
- Test edge cases
- Implement validation checks

\`\`\`python
"""
Model Validation Checks
"""

class ModelValidator:
    """Automated validation for financial models"""
    
    @staticmethod
    def check_balance_sheet(
        assets: float,
        liabilities: float,
        equity: float,
        tolerance: float = 1.0
    ) -> None:
        """Verify Assets = Liabilities + Equity"""
        difference = abs(assets - (liabilities + equity))
        if difference > tolerance:
            raise ValueError(
                f"Balance sheet doesn't balance! "
                f"Assets: ${assets:,.0f}, L+E: ${liabilities + equity:,.0f}, "
                f"Difference: ${difference:,.0f}"
            )
        print(f"✓ Balance sheet balances (diff: ${difference:.2f})")
    
    @staticmethod
    def check_cash_flow(
        beginning_cash: float,
        cash_from_operations: float,
        cash_from_investing: float,
        cash_from_financing: float,
        ending_cash: float,
        tolerance: float = 1.0
    ) -> None:
        """Verify cash flow statement ties"""
        calculated_ending = (
            beginning_cash + 
            cash_from_operations + 
            cash_from_investing + 
            cash_from_financing
        )
        difference = abs(ending_cash - calculated_ending)
        if difference > tolerance:
            raise ValueError(
                f"Cash flow doesn't tie! "
                f"Expected: ${calculated_ending:,.0f}, "
                f"Actual: ${ending_cash:,.0f}, "
                f"Difference: ${difference:,.0f}"
            )
        print(f"✓ Cash flow ties (diff: ${difference:.2f})")
    
    @staticmethod
    def check_reasonableness(
        metric_name: str,
        value: float,
        expected_range: Tuple[float, float]
    ) -> None:
        """Check if metric is in reasonable range"""
        min_val, max_val = expected_range
        if not (min_val <= value <= max_val):
            print(
                f"⚠ WARNING: {metric_name} = {value:.1%} "
                f"outside expected range [{min_val:.1%}, {max_val:.1%}]"
            )
        else:
            print(f"✓ {metric_name} = {value:.1%} is reasonable")

# Example validation
validator = ModelValidator()

validator.check_balance_sheet(
    assets=1_000_000,
    liabilities=600_000,
    equity=400_000
)

validator.check_reasonableness(
    "Gross Margin",
    value=0.42,
    expected_range=(0.20, 0.70)
)

validator.check_reasonableness(
    "Revenue Growth",
    value=0.95,  # 95% - suspicious!
    expected_range=(- 0.10, 0.30)
)
\`\`\`

### 4. Inconsistent Time Periods

**The Problem**: Mixing annual and quarterly numbers, or misaligning fiscal years.

**Solution**: Clearly label time periods and convert consistently.

\`\`\`python
"""
Time Period Management
"""

from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

class TimePeriodManager:
    """Manage time periods in financial models"""
    
    def __init__(self, start_date: str, fiscal_year_end: str = "12-31"):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.fiscal_year_end = fiscal_year_end
    
    def generate_periods(
        self,
        n_years: int,
        frequency: str = 'annual'
    ) -> pd.DatetimeIndex:
        """
        Generate time periods for model.
        
        Args:
            n_years: Number of years to project
            frequency: 'annual', 'quarterly', or 'monthly'
        """
        if frequency == 'annual':
            periods = pd.date_range(
                self.start_date,
                periods=n_years,
                freq='Y'
            )
        elif frequency == 'quarterly':
            periods = pd.date_range(
                self.start_date,
                periods=n_years * 4,
                freq='Q'
            )
        elif frequency == 'monthly':
            periods = pd.date_range(
                self.start_date,
                periods=n_years * 12,
                freq='M'
            )
        else:
            raise ValueError(f"Invalid frequency: {frequency}")
        
        return periods
    
    @staticmethod
    def convert_quarterly_to_annual(quarterly_values: pd.Series) -> pd.Series:
        """Convert quarterly to annual (sum 4 quarters)"""
        annual = quarterly_values.resample('Y').sum()
        return annual
    
    @staticmethod
    def convert_annual_to_quarterly(annual_values: pd.Series) -> pd.Series:
        """Convert annual to quarterly (divide by 4 - rough approximation)"""
        # This is simplified; real conversion needs more context
        quarterly = annual_values / 4
        return quarterly

# Example
tm = TimePeriodManager(start_date="2024-01-01")

annual_periods = tm.generate_periods(5, 'annual')
print("Annual Periods:")
print(annual_periods.strftime("%Y-%m-%d").tolist())

quarterly_periods = tm.generate_periods(5, 'quarterly')
print("\\nQuarterly Periods (first 8):")
print(quarterly_periods[:8].strftime("%Y-%m-%d").tolist())
\`\`\`

---

## pandas for Financial Modeling

### Why pandas?

**pandas** is Python's premier data manipulation library, perfect for financial models:

- **DataFrame structure**: Like Excel tables, but programmatic
- **Time series support**: Built-in date handling
- **Aggregation**: Easy calculations across rows/columns
- **Integration**: Works with numpy, matplotlib, Excel

\`\`\`python
"""
Financial Modeling with pandas
"""

import pandas as pd
import numpy as np

class PandasFinancialModel:
    """Build financial models using pandas DataFrames"""
    
    def __init__(self, base_year: int, projection_years: int = 5):
        self.base_year = base_year
        self.projection_years = projection_years
        self.years = list(range(base_year, base_year + projection_years + 1))
        
        # Initialize DataFrames
        self.income_statement = pd.DataFrame(index=self.years)
        self.balance_sheet = pd.DataFrame(index=self.years)
        self.cash_flow = pd.DataFrame(index=self.years)
    
    def project_revenue(
        self,
        base_revenue: float,
        growth_rates: List[float]
    ) -> None:
        """
        Project revenue with year-specific growth rates.
        
        Args:
            base_revenue: Starting revenue
            growth_rates: Growth rate for each projection year
        """
        if len(growth_rates) != self.projection_years:
            raise ValueError(
                f"Need {self.projection_years} growth rates, got {len(growth_rates)}"
            )
        
        revenue = [base_revenue]
        for growth in growth_rates:
            revenue.append(revenue[-1] * (1 + growth))
        
        self.income_statement['Revenue'] = revenue
    
    def calculate_operating_income(
        self,
        cogs_pct: float,
        opex_pct: float
    ) -> None:
        """Calculate operating income from revenue"""
        self.income_statement['COGS'] = -self.income_statement['Revenue'] * cogs_pct
        self.income_statement['Gross Profit'] = (
            self.income_statement['Revenue'] + self.income_statement['COGS']
        )
        self.income_statement['Operating Expenses'] = (
            -self.income_statement['Revenue'] * opex_pct
        )
        self.income_statement['Operating Income'] = (
            self.income_statement['Gross Profit'] + 
            self.income_statement['Operating Expenses']
        )
    
    def calculate_net_income(
        self,
        interest_expense: float,
        tax_rate: float
    ) -> None:
        """Calculate net income"""
        self.income_statement['Interest Expense'] = -interest_expense
        self.income_statement['EBT'] = (
            self.income_statement['Operating Income'] + 
            self.income_statement['Interest Expense']
        )
        self.income_statement['Taxes'] = (
            -self.income_statement['EBT'] * tax_rate
        )
        self.income_statement['Net Income'] = (
            self.income_statement['EBT'] + self.income_statement['Taxes']
        )
    
    def get_summary(self) -> pd.DataFrame:
        """Get income statement summary"""
        return self.income_statement.T  # Transpose for better viewing

# Example usage
model = PandasFinancialModel(base_year=2024, projection_years=5)

model.project_revenue(
    base_revenue=1_000_000,
    growth_rates=[0.10, 0.12, 0.12, 0.10, 0.08]
)

model.calculate_operating_income(
    cogs_pct=0.40,
    opex_pct=0.25
)

model.calculate_net_income(
    interest_expense=50_000,
    tax_rate=0.21
)

print("Income Statement ($ thousands):")
print(model.get_summary().apply(lambda x: x / 1000))
\`\`\`

---

## Key Takeaways

### Core Principles

1. **Separate inputs, calculations, and outputs** - Makes models flexible and auditable

2. **Document everything** - Future you (or colleagues) will thank you

3. **Use color coding** - Visual distinction between inputs and formulas prevents errors

4. **Validate your model** - Balance sheet must balance, cash flow must tie

5. **Version control** - Track changes, especially for collaborative models

6. **Test edge cases** - What if growth is negative? What if margins compress?

### Excel vs Python

- **Excel**: Quick, visual, universal - great for presentations
- **Python**: Scalable, reproducible, automated - great for complex analysis
- **Hybrid**: Best of both worlds - build in Python, present in Excel

### Common Mistakes

❌ Hard-coding numbers in formulas  
❌ Not documenting assumptions  
❌ Ignoring circular references  
❌ Copy-paste errors  
❌ Inconsistent time periods  
❌ No validation checks

### Professional Standards

✅ Clear structure (inputs → calculations → outputs)  
✅ Color-coded and labeled  
✅ Documented assumptions with sources  
✅ Built-in checks and validation  
✅ Version-controlled  
✅ Tested and stress-tested

---

## Next Steps

Master these fundamentals, and you're ready for:

- **Three-Statement Model** (Section 2): Build integrated financials
- **DCF Valuation** (Section 3): Value companies using cash flows
- **Comps Analysis** (Section 4): Relative valuation techniques

Financial modeling is a **craft**. The difference between amateur and professional models isn't complexity—it's **structure, documentation, and rigor**.

**Practice**: Build a simple 3-year projection model for any company. Focus on structure, not perfection. The skills compound.

---

## Additional Resources

**Books:**
- *Financial Modeling* by Simon Benninga
- *Investment Banking: Valuation, LBOs, M&A* by Rosenbaum & Pearl (Greenbook)

**Online:**
- Wall Street Prep: Financial modeling courses
- Breaking Into Wall Street: Excel models and tutorials
- Corporate Finance Institute: Free templates

**Tools:**
- **Excel**: Advanced functions, data tables, VBA
- **Python**: pandas, numpy, scipy, matplotlib
- **Git**: Version control for models

**Next Section**: [Three-Statement Model Building](./three-statement-model) →
`,
};

