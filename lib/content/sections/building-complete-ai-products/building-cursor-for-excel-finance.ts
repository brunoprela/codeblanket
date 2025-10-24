export const buildingCursorForExcelFinance = {
  title: 'Building Cursor for Excel & Finance',
  id: 'building-cursor-for-excel-finance',
  content: `
# Building Cursor for Excel & Finance

## Introduction

Imagine Cursor, but for Excel and financial analysis—an AI assistant that understands spreadsheets, generates formulas, analyzes financial data, and automates complex Excel tasks through natural language. This is one of the highest-value AI applications for business users.

This section covers building an AI-powered Excel assistant that:
- Generates Excel formulas from natural language
- Understands and explains existing formulas
- Performs financial analysis (DCF, ratios, modeling)
- Automates repetitive Excel tasks
- Integrates real-time financial data
- Provides version control for spreadsheets

### Why Excel + AI is Valuable

**Market Size**: 
- 1.2 billion Microsoft Office users
- 750 million Excel users worldwide
- Financial analysts spend 50%+ of time in Excel

**Pain Points**:
- Complex formulas are hard to write
- Financial modeling is time-consuming
- Data updates require manual work
- Error-prone manual calculations
- No version control for spreadsheets

**Market Opportunity**: Tools addressing this space (Coefficient, Rows, Equals) raising significant funding.

### Architecture Overview

\`\`\`
┌──────────────────────────────────────────────────────────┐
│          Cursor for Excel & Finance                       │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐        ┌──────────────┐               │
│  │  Excel File  │───────▶│  Parser &    │               │
│  │  (.xlsx)     │        │  Analyzer    │               │
│  └──────────────┘        └──────┬───────┘               │
│                                  │                        │
│                          ┌───────▼────────┐              │
│                          │  Formula       │              │
│                          │  Understanding │              │
│                          └───────┬────────┘              │
│                                  │                        │
│                          ┌───────▼────────┐              │
│                          │  LLM Engine    │              │
│                          │  (GPT-4)       │              │
│                          └───────┬────────┘              │
│                                  │                        │
│              ┌───────────────────┼──────────────┐        │
│              │                   │              │        │
│       ┌──────▼──────┐    ┌──────▼──────┐ ┌────▼─────┐  │
│       │  Formula    │    │  Financial  │ │  Data    │  │
│       │  Generator  │    │  Analysis   │ │  Fetcher │  │
│       └──────┬──────┘    └──────┬──────┘ └────┬─────┘  │
│              │                   │              │        │
│              └───────────────────┼──────────────┘        │
│                                  │                        │
│                          ┌───────▼────────┐              │
│                          │  Excel Writer  │              │
│                          │  (openpyxl)    │              │
│                          └───────┬────────┘              │
│                                  │                        │
│                          ┌───────▼────────┐              │
│                          │  Version       │              │
│                          │  Control       │              │
│                          └────────────────┘              │
└──────────────────────────────────────────────────────────┘
\`\`\`

---

## Excel Parsing & Understanding

### Comprehensive Excel Parser

\`\`\`python
"""
Parse and understand Excel spreadsheets
"""

from openpyxl import load_workbook
from openpyxl.formula import Tokenizer
import pandas as pd
from typing import Dict, List, Set

class CellReference(BaseModel):
    """Reference to an Excel cell"""
    sheet: str
    cell: str
    value: any
    formula: Optional[str] = None
    has_formula: bool = False

class ExcelStructure(BaseModel):
    """Analyzed Excel structure"""
    sheets: List[str]
    named_ranges: Dict[str, str]
    tables: List[Dict]
    formulas: Dict[str, List[str]]  # sheet -> formulas
    dependencies: Dict[str, Set[str]]  # cell -> dependencies
    data_summary: Dict[str, any]

class ExcelParser:
    """
    Parse Excel files and understand structure
    """
    
    def __init__(self):
        self.workbook = None
        
    def parse(self, file_path: str) -> ExcelStructure:
        """
        Parse Excel file and analyze structure
        """
        self.workbook = load_workbook(file_path, data_only=False)
        
        # Extract all components
        sheets = self.workbook.sheetnames
        named_ranges = self._extract_named_ranges()
        tables = self._extract_tables()
        formulas = self._extract_formulas()
        dependencies = self._build_dependency_graph()
        data_summary = self._summarize_data()
        
        return ExcelStructure(
            sheets=sheets,
            named_ranges=named_ranges,
            tables=tables,
            formulas=formulas,
            dependencies=dependencies,
            data_summary=data_summary
        )
    
    def _extract_named_ranges(self) -> Dict[str, str]:
        """Extract all named ranges"""
        named_ranges = {}
        
        for name, cells in self.workbook.defined_names.items():
            if cells.value:
                named_ranges[name] = cells.value
        
        return named_ranges
    
    def _extract_tables(self) -> List[Dict]:
        """Extract Excel tables"""
        tables = []
        
        for sheet_name in self.workbook.sheetnames:
            sheet = self.workbook[sheet_name]
            
            # Check for table structures
            if hasattr(sheet, 'tables'):
                for table in sheet.tables.values():
                    tables.append({
                        "name": table.name,
                        "sheet": sheet_name,
                        "range": table.ref,
                        "columns": [col.name for col in table.tableColumns]
                    })
        
        return tables
    
    def _extract_formulas(self) -> Dict[str, List[str]]:
        """Extract all formulas by sheet"""
        formulas = {}
        
        for sheet_name in self.workbook.sheetnames:
            sheet = self.workbook[sheet_name]
            sheet_formulas = []
            
            for row in sheet.iter_rows():
                for cell in row:
                    if cell.value and isinstance(cell.value, str) and cell.value.startswith('='):
                        sheet_formulas.append({
                            "cell": cell.coordinate,
                            "formula": cell.value,
                            "parsed": self._parse_formula(cell.value)
                        })
            
            formulas[sheet_name] = sheet_formulas
        
        return formulas
    
    def _parse_formula(self, formula: str) -> Dict:
        """Parse formula into components"""
        tokenizer = Tokenizer(formula)
        
        functions = []
        cell_refs = []
        operators = []
        
        for token in tokenizer.items:
            if token.type == 'FUNC':
                functions.append(token.value)
            elif token.type == 'OPERAND' and token.subtype == 'RANGE':
                cell_refs.append(token.value)
            elif token.type == 'OPERATOR':
                operators.append(token.value)
        
        return {
            "functions": functions,
            "cell_references": cell_refs,
            "operators": operators
        }
    
    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """Build graph of cell dependencies"""
        dependencies = {}
        
        for sheet_name in self.workbook.sheetnames:
            sheet = self.workbook[sheet_name]
            
            for row in sheet.iter_rows():
                for cell in row:
                    if cell.value and isinstance(cell.value, str) and cell.value.startswith('='):
                        cell_key = f"{sheet_name}!{cell.coordinate}"
                        parsed = self._parse_formula(cell.value)
                        
                        # Extract dependencies
                        deps = set()
                        for ref in parsed["cell_references"]:
                            if '!' in ref:
                                deps.add(ref)
                            else:
                                deps.add(f"{sheet_name}!{ref}")
                        
                        dependencies[cell_key] = deps
        
        return dependencies
    
    def _summarize_data(self) -> Dict:
        """Summarize data in spreadsheet"""
        summary = {}
        
        for sheet_name in self.workbook.sheetnames:
            df = pd.read_excel(
                self.workbook,
                sheet_name=sheet_name,
                engine='openpyxl'
            )
            
            summary[sheet_name] = {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "data_types": df.dtypes.astype(str).to_dict(),
                "numeric_columns": list(df.select_dtypes(include=['number']).columns),
                "has_headers": True  # Heuristic
            }
        
        return summary

# Usage
parser = ExcelParser()
structure = parser.parse("financial_model.xlsx")

print(f"Sheets: {structure.sheets}")
print(f"Named ranges: {len(structure.named_ranges)}")
print(f"Total formulas: {sum(len(f) for f in structure.formulas.values())}")
\`\`\`

---

## Natural Language to Formula

### Formula Generator

\`\`\`python
"""
Generate Excel formulas from natural language
"""

from openai import AsyncOpenAI

class FormulaGenerator:
    """
    Generate Excel formulas from natural language descriptions
    """
    
    def __init__(self, llm_client: AsyncOpenAI):
        self.llm = llm_client
        
    async def generate_formula(
        self,
        description: str,
        context: ExcelStructure = None
    ) -> Dict[str, any]:
        """
        Generate Excel formula from description
        """
        # Build context about spreadsheet
        context_info = self._build_context(context) if context else ""
        
        prompt = f"""Generate an Excel formula based on this description.

Description: {description}

{context_info}

Provide:
1. The Excel formula
2. Explanation of what it does
3. Any assumptions made

Format as JSON:
{{
  "formula": "=FORMULA_HERE",
  "explanation": "detailed explanation",
  "assumptions": ["assumption 1", "assumption 2"]
}}"""

        response = await self.llm.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Excel formula writer. Generate precise, working Excel formulas."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Validate formula syntax
        is_valid = self._validate_formula(result["formula"])
        result["is_valid"] = is_valid
        
        return result
    
    def _build_context(self, context: ExcelStructure) -> str:
        """Build context string about spreadsheet"""
        context_parts = []
        
        context_parts.append(f"Available sheets: {', '.join(context.sheets)}")
        
        if context.named_ranges:
            context_parts.append(f"Named ranges: {', '.join(context.named_ranges.keys())}")
        
        for sheet, summary in context.data_summary.items():
            context_parts.append(
                f"Sheet '{sheet}': {summary['rows']} rows, columns: {', '.join(summary['column_names'][:10])}"
            )
        
        return "\\n".join(context_parts)
    
    def _validate_formula(self, formula: str) -> bool:
        """Basic formula validation"""
        if not formula.startswith('='):
            return False
        
        # Check balanced parentheses
        if formula.count('(') != formula.count(')'):
            return False
        
        # Check for common errors
        invalid_patterns = ['==', ',,', '())', '(()']
        return not any(pattern in formula for pattern in invalid_patterns)
    
    async def explain_formula(self, formula: str) -> str:
        """Explain what an existing formula does"""
        prompt = f"""Explain this Excel formula in simple terms:

Formula: {formula}

Provide a clear, step-by-step explanation of what this formula does."""

        response = await self.llm.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content

# Usage
formula_gen = FormulaGenerator(llm_client)

# Generate formula
result = await formula_gen.generate_formula(
    "Calculate the compound annual growth rate (CAGR) from cell A1 (beginning value) to A10 (ending value) over 10 years",
    context=structure
)

print(f"Formula: {result['formula']}")
print(f"Explanation: {result['explanation']}")

# Explain existing formula
explanation = await formula_gen.explain_formula(
    "=VLOOKUP(A2,Sheet2!A:B,2,FALSE)"
)
print(f"Explanation: {explanation}")
\`\`\`

---

## Financial Analysis

### Financial Modeling Assistant

\`\`\`python
"""
Financial analysis and modeling functions
"""

class FinancialAnalyzer:
    """
    Perform financial analysis on Excel data
    """
    
    def __init__(self, llm_client: AsyncOpenAI):
        self.llm = llm_client
    
    async def analyze_financial_statements(
        self,
        file_path: str
    ) -> Dict[str, any]:
        """
        Analyze financial statements in Excel
        """
        # Parse Excel
        parser = ExcelParser()
        structure = parser.parse(file_path)
        
        # Load data
        all_data = {}
        for sheet in structure.sheets:
            df = pd.read_excel(file_path, sheet_name=sheet)
            all_data[sheet] = df
        
        # Identify financial statement types
        statement_types = self._identify_statements(all_data)
        
        # Perform analysis
        analysis = {}
        
        for sheet, stmt_type in statement_types.items():
            if stmt_type == "income_statement":
                analysis[sheet] = self._analyze_income_statement(all_data[sheet])
            elif stmt_type == "balance_sheet":
                analysis[sheet] = self._analyze_balance_sheet(all_data[sheet])
            elif stmt_type == "cash_flow":
                analysis[sheet] = self._analyze_cash_flow(all_data[sheet])
        
        # Generate insights with LLM
        insights = await self._generate_insights(analysis)
        
        return {
            "analysis": analysis,
            "insights": insights,
            "statement_types": statement_types
        }
    
    def _identify_statements(self, data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Identify what type of financial statement each sheet contains"""
        statement_types = {}
        
        for sheet_name, df in data.items():
            # Look for key terms in column headers
            columns = ' '.join(df.columns.astype(str)).lower()
            
            if any(term in columns for term in ['revenue', 'net income', 'expenses', 'gross profit']):
                statement_types[sheet_name] = "income_statement"
            elif any(term in columns for term in ['assets', 'liabilities', 'equity']):
                statement_types[sheet_name] = "balance_sheet"
            elif any(term in columns for term in ['operating activities', 'cash flow', 'investing activities']):
                statement_types[sheet_name] = "cash_flow"
            else:
                statement_types[sheet_name] = "unknown"
        
        return statement_types
    
    def _analyze_income_statement(self, df: pd.DataFrame) -> Dict:
        """Calculate key income statement metrics"""
        # Find revenue and net income rows
        revenue_row = None
        net_income_row = None
        
        for idx, row in df.iterrows():
            row_text = str(row[0]).lower()
            if 'revenue' in row_text or 'sales' in row_text:
                revenue_row = idx
            if 'net income' in row_text:
                net_income_row = idx
        
        if revenue_row is None or net_income_row is None:
            return {"error": "Could not identify key line items"}
        
        # Extract numeric columns (typically years)
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        metrics = {}
        for col in numeric_cols:
            revenue = df.iloc[revenue_row][col]
            net_income = df.iloc[net_income_row][col]
            
            if pd.notna(revenue) and pd.notna(net_income) and revenue != 0:
                metrics[str(col)] = {
                    "revenue": float(revenue),
                    "net_income": float(net_income),
                    "net_margin": (net_income / revenue) * 100
                }
        
        return metrics
    
    def _analyze_balance_sheet(self, df: pd.DataFrame) -> Dict:
        """Calculate key balance sheet metrics"""
        # Similar approach to income statement
        # Calculate liquidity ratios, leverage ratios, etc.
        return {}
    
    def _analyze_cash_flow(self, df: pd.DataFrame) -> Dict:
        """Calculate cash flow metrics"""
        return {}
    
    async def _generate_insights(self, analysis: Dict) -> str:
        """Generate AI insights from analysis"""
        prompt = f"""Analyze these financial metrics and provide insights:

{json.dumps(analysis, indent=2)}

Provide:
1. Key trends
2. Areas of concern
3. Strengths
4. Recommendations

Be specific and reference the actual numbers."""

        response = await self.llm.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        
        return response.choices[0].message.content
    
    async def build_dcf_model(
        self,
        projections: Dict[str, List[float]],
        discount_rate: float,
        terminal_growth_rate: float
    ) -> Dict:
        """
        Build a Discounted Cash Flow (DCF) model
        """
        free_cash_flows = projections.get('free_cash_flow', [])
        
        # Calculate present values
        present_values = []
        for i, fcf in enumerate(free_cash_flows, start=1):
            pv = fcf / ((1 + discount_rate) ** i)
            present_values.append(pv)
        
        # Calculate terminal value
        last_fcf = free_cash_flows[-1]
        terminal_value = (last_fcf * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
        terminal_pv = terminal_value / ((1 + discount_rate) ** len(free_cash_flows))
        
        # Enterprise value
        enterprise_value = sum(present_values) + terminal_pv
        
        return {
            "free_cash_flows": free_cash_flows,
            "present_values": present_values,
            "terminal_value": terminal_value,
            "terminal_pv": terminal_pv,
            "enterprise_value": enterprise_value,
            "assumptions": {
                "discount_rate": discount_rate,
                "terminal_growth_rate": terminal_growth_rate,
                "projection_years": len(free_cash_flows)
            }
        }

# Usage
analyzer = FinancialAnalyzer(llm_client)

# Analyze financial statements
analysis = await analyzer.analyze_financial_statements("financials.xlsx")
print(f"Insights: {analysis['insights']}")

# Build DCF model
dcf = await analyzer.build_dcf_model(
    projections={'free_cash_flow': [100, 110, 121, 133, 146]},
    discount_rate=0.10,
    terminal_growth_rate=0.03
)
print(f"Enterprise Value: \${dcf['enterprise_value']:, .2f
}")
\`\`\`

---

## Real-Time Data Integration

### Financial Data Fetcher

\`\`\`python
"""
Fetch real-time financial data into Excel
"""

import yfinance as yf
import pandas_datareader as pdr
from datetime import datetime, timedelta

class FinancialDataFetcher:
    """
    Fetch real-time market data for Excel
    """
    
    async def get_stock_data(
        self,
        ticker: str,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Fetch stock price data
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        return df
    
    async def get_financial_statements(self, ticker: str) -> Dict:
        """
        Fetch company financials
        """
        stock = yf.Ticker(ticker)
        
        return {
            "income_statement": stock.financials,
            "balance_sheet": stock.balance_sheet,
            "cash_flow": stock.cashflow,
            "info": stock.info
        }
    
    async def get_market_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetch current market data for multiple tickers
        """
        data = []
        
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            data.append({
                "ticker": ticker,
                "price": info.get("currentPrice"),
                "change": info.get("regularMarketChangePercent"),
                "volume": info.get("volume"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE")
            })
        
        return pd.DataFrame(data)
    
    async def update_excel_with_live_data(
        self,
        file_path: str,
        ticker: str,
        target_range: str
    ):
        """
        Update Excel cells with live market data
        """
        # Get data
        data = await self.get_market_data([ticker])
        
        # Load workbook
        wb = load_workbook(file_path)
        ws = wb.active
        
        # Parse range (e.g., "A1:F1")
        # Write data to range
        start_cell = ws[target_range.split(':')[0]]
        row = start_cell.row
        col = start_cell.column
        
        for i, (key, value) in enumerate(data.iloc[0].items()):
            ws.cell(row=row, column=col + i, value=value)
        
        # Save
        wb.save(file_path)

# Usage
fetcher = FinancialDataFetcher()

# Get stock data
stock_data = await fetcher.get_stock_data("AAPL")
print(stock_data.head())

# Update Excel with live data
await fetcher.update_excel_with_live_data(
    "portfolio.xlsx",
    "AAPL",
    "A2:F2"
)
\`\`\`

---

## Version Control for Spreadsheets

### Git-like Versioning

\`\`\`python
"""
Version control system for Excel files
"""

import hashlib
from datetime import datetime
import difflib

class SpreadsheetVersion(BaseModel):
    """A version of a spreadsheet"""
    version_id: str
    timestamp: datetime
    author: str
    message: str
    file_hash: str
    changes: Dict[str, any]

class ExcelVersionControl:
    """
    Git-like version control for Excel files
    """
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.versions_path = self.base_path / ".excel_versions"
        self.versions_path.mkdir(exist_ok=True)
        self.versions: List[SpreadsheetVersion] = []
        
    def commit(
        self,
        file_path: Path,
        message: str,
        author: str
    ) -> str:
        """
        Create a new version
        """
        # Calculate file hash
        file_hash = self._hash_file(file_path)
        
        # Detect changes from previous version
        changes = self._detect_changes(file_path)
        
        # Create version
        version = SpreadsheetVersion(
            version_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            author=author,
            message=message,
            file_hash=file_hash,
            changes=changes
        )
        
        # Store version
        version_file = self.versions_path / f"{version.version_id}.json"
        version_file.write_text(version.model_dump_json())
        
        # Copy file
        shutil.copy(
            file_path,
            self.versions_path / f"{version.version_id}.xlsx"
        )
        
        self.versions.append(version)
        
        return version.version_id
    
    def _hash_file(self, file_path: Path) -> str:
        """Calculate file hash"""
        return hashlib.sha256(file_path.read_bytes()).hexdigest()
    
    def _detect_changes(self, file_path: Path) -> Dict:
        """
        Detect changes from previous version
        """
        if not self.versions:
            return {"type": "initial_commit"}
        
        # Load current and previous
        current = load_workbook(file_path, data_only=True)
        previous_path = self.versions_path / f"{self.versions[-1].version_id}.xlsx"
        previous = load_workbook(previous_path, data_only=True)
        
        changes = {
            "modified_cells": [],
            "new_sheets": [],
            "deleted_sheets": []
        }
        
        # Compare sheets
        current_sheets = set(current.sheetnames)
        previous_sheets = set(previous.sheetnames)
        
        changes["new_sheets"] = list(current_sheets - previous_sheets)
        changes["deleted_sheets"] = list(previous_sheets - current_sheets)
        
        # Compare cells in common sheets
        for sheet_name in current_sheets & previous_sheets:
            curr_sheet = current[sheet_name]
            prev_sheet = previous[sheet_name]
            
            # Compare each cell
            for row in curr_sheet.iter_rows():
                for cell in row:
                    prev_cell = prev_sheet[cell.coordinate]
                    
                    if cell.value != prev_cell.value:
                        changes["modified_cells"].append({
                            "sheet": sheet_name,
                            "cell": cell.coordinate,
                            "old_value": prev_cell.value,
                            "new_value": cell.value
                        })
        
        return changes
    
    def list_versions(self) -> List[SpreadsheetVersion]:
        """List all versions"""
        return sorted(self.versions, key=lambda v: v.timestamp, reverse=True)
    
    def checkout(self, version_id: str, target_path: Path):
        """
        Restore a specific version
        """
        version_file = self.versions_path / f"{version_id}.xlsx"
        
        if not version_file.exists():
            raise ValueError(f"Version {version_id} not found")
        
        shutil.copy(version_file, target_path)
    
    def diff(self, version_id_1: str, version_id_2: str) -> Dict:
        """
        Show differences between two versions
        """
        # Load both versions
        file1 = self.versions_path / f"{version_id_1}.xlsx"
        file2 = self.versions_path / f"{version_id_2}.xlsx"
        
        # Compare (similar to _detect_changes but between any two versions)
        return self._compare_files(file1, file2)
    
    def _compare_files(self, file1: Path, file2: Path) -> Dict:
        """Compare two Excel files"""
        # Implementation similar to _detect_changes
        pass

# Usage
vcs = ExcelVersionControl("/path/to/project")

# Commit changes
version_id = vcs.commit(
    Path("financial_model.xlsx"),
    message="Updated revenue projections",
    author="analyst@company.com"
)

# List versions
versions = vcs.list_versions()
for v in versions:
    print(f"{v.version_id[:8]} - {v.timestamp} - {v.message}")

# Restore previous version
vcs.checkout(versions[1].version_id, Path("financial_model.xlsx"))
\`\`\`

---

## Complete Integration

### Excel AI Assistant API

\`\`\`python
"""
Complete Excel AI Assistant API
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

app = FastAPI(title="Excel AI Assistant")

class ExcelAssistant:
    """
    Complete Excel AI assistant
    """
    
    def __init__(self):
        self.parser = ExcelParser()
        self.formula_gen = FormulaGenerator(llm_client)
        self.financial_analyzer = FinancialAnalyzer(llm_client)
        self.data_fetcher = FinancialDataFetcher()
        self.vcs = ExcelVersionControl("/tmp/excel_vcs")
    
    async def process_request(
        self,
        file_path: str,
        request: str
    ) -> Dict:
        """
        Process natural language request on Excel file
        """
        # Parse Excel to understand structure
        structure = self.parser.parse(file_path)
        
        # Determine intent
        intent = await self._classify_intent(request)
        
        if intent == "generate_formula":
            result = await self.formula_gen.generate_formula(request, structure)
        elif intent == "analyze_financials":
            result = await self.financial_analyzer.analyze_financial_statements(file_path)
        elif intent == "fetch_data":
            # Extract ticker from request
            ticker = await self._extract_ticker(request)
            result = await self.data_fetcher.get_financial_statements(ticker)
        elif intent == "explain_formula":
            # Extract formula from request
            formula = request.split(":")[-1].strip()
            result = await self.formula_gen.explain_formula(formula)
        else:
            result = {"error": "Could not understand request"}
        
        return result
    
    async def _classify_intent(self, request: str) -> str:
        """Classify user intent"""
        prompt = f"""Classify this Excel-related request into one category:
- generate_formula: User wants to create a new formula
- analyze_financials: User wants financial analysis
- fetch_data: User wants to fetch external data
- explain_formula: User wants to understand a formula

Request: {request}

Return only the category name."""

        response = await llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
    
    async def _extract_ticker(self, request: str) -> str:
        """Extract stock ticker from request"""
        # Simple extraction, could be improved
        words = request.upper().split()
        for word in words:
            if len(word) <= 5 and word.isalpha():
                return word
        return ""

@app.post("/api/excel/analyze")
async def analyze_excel(file: UploadFile = File(...)):
    """Analyze uploaded Excel file"""
    # Save file
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    # Parse and analyze
    assistant = ExcelAssistant()
    structure = assistant.parser.parse(temp_path)
    
    return {
        "sheets": structure.sheets,
        "total_formulas": sum(len(f) for f in structure.formulas.values()),
        "named_ranges": len(structure.named_ranges),
        "data_summary": structure.data_summary
    }

@app.post("/api/excel/request")
async def process_request(
    file: UploadFile = File(...),
    request: str = ""
):
    """Process natural language request"""
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    assistant = ExcelAssistant()
    result = await assistant.process_request(temp_path, request)
    
    return result
\`\`\`

---

## Conclusion

Building Cursor for Excel & Finance requires:

1. **Excel Parsing**: Deep understanding of spreadsheet structure
2. **Formula Generation**: Natural language to Excel formulas
3. **Financial Analysis**: Domain-specific knowledge (DCF, ratios, modeling)
4. **Real-Time Data**: Integration with market data APIs
5. **Version Control**: Track changes to spreadsheets
6. **LLM Integration**: GPT-4 for understanding and generation

**Key Technologies**:
- **openpyxl**: Excel file manipulation
- **pandas**: Data analysis
- **yfinance**: Market data
- **OpenAI GPT-4**: Natural language understanding
- **FastAPI**: API server

This creates a powerful tool for financial analysts, replacing hours of manual Excel work with natural language commands.
`,
};
