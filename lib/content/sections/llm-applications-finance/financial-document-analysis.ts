export const financialDocumentAnalysis = {
  title: 'LLMs for Financial Document Analysis',
  id: 'financial-document-analysis',
  content: `
# LLMs for Financial Document Analysis

## Introduction

Financial documents like 10-K and 10-Q filings contain critical information for investment decisions, but they're dense, complex, and time-consuming to analyze manually. A single 10-K filing can contain hundreds of pages of text, tables, and financial statements. Large Language Models (LLMs) can automate the extraction, analysis, and summarization of this information at scale.

This section covers how to use LLMs to parse SEC filings, extract key metrics automatically, analyze risk factors, process Management Discussion & Analysis (MD&A) sections, and compare filings over time to identify trends.

### Why LLMs for Financial Documents

**Volume**: Thousands of filings published daily across public markets
**Complexity**: Legal language, financial jargon, complex structures
**Context**: Understanding requires domain knowledge and cross-document reasoning
**Speed**: Human analysts can only cover a fraction of available information
**Consistency**: LLMs can apply the same analytical framework across all documents

---

## Understanding SEC Filings

### Key SEC Filing Types

\`\`\`python
"""
Overview of SEC filing types and their significance
"""

SEC_FILING_TYPES = {
    '10-K': {
        'description': 'Annual comprehensive report',
        'frequency': 'Annual',
        'sections': [
            'Business Overview',
            'Risk Factors',
            'MD&A (Management Discussion & Analysis)',
            'Financial Statements',
            'Notes to Financial Statements',
        ],
        'pages': '100-300+',
        'use_cases': 'Complete company analysis, risk assessment, long-term trends'
    },
    '10-Q': {
        'description': 'Quarterly financial report',
        'frequency': 'Quarterly',
        'sections': [
            'Financial Statements',
            'MD&A',
            'Risk Factors (if material changes)',
        ],
        'pages': '40-80',
        'use_cases': 'Short-term performance, quarterly trends, timely updates'
    },
    '8-K': {
        'description': 'Current report for material events',
        'frequency': 'As needed',
        'sections': ['Event-specific disclosure'],
        'pages': '5-20',
        'use_cases': 'Breaking news, acquisitions, leadership changes'
    },
    'DEF 14A': {
        'description': 'Proxy statement',
        'frequency': 'Annual',
        'sections': ['Executive compensation', 'Governance', 'Voting matters'],
        'pages': '50-150',
        'use_cases': 'Executive compensation analysis, governance assessment'
    }
}

# What makes financial documents challenging:
CHALLENGES = {
    'structure': 'Inconsistent formatting across companies and time',
    'tables': 'Complex financial tables embedded in text',
    'jargon': 'Industry-specific terminology and accounting standards',
    'length': 'Extremely long documents (100-300 pages)',
    'legal': 'Dense legal language and disclaimers',
    'references': 'Cross-references to other sections and exhibits',
}
\`\`\`

---

## Retrieving SEC Filings

### Using SEC EDGAR API

\`\`\`python
"""
Retrieve SEC filings programmatically
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time

class SECFilingRetriever:
    """
    Retrieve SEC filings from EDGAR database
    """
    
    BASE_URL = "https://www.sec.gov"
    
    def __init__(self, user_agent: str):
        """
        Initialize with required user agent (SEC requirement)
        
        Args:
            user_agent: Your contact info (e.g., "YourName your@email.com")
        """
        self.headers = {
            'User-Agent': user_agent
        }
        
    def get_company_cik(self, ticker: str) -> str:
        """
        Get CIK (Central Index Key) from ticker symbol
        
        Args:
            ticker: Stock ticker (e.g., 'AAPL')
            
        Returns:
            CIK number (padded to 10 digits)
        """
        # SEC provides a ticker to CIK mapping
        url = f"{self.BASE_URL}/cgi-bin/browse-edgar"
        params = {
            'action': 'getcompany',
            'ticker': ticker,
            'output': 'json'
        }
        
        response = requests.get(url, params=params, headers=self.headers)
        time.sleep(0.1)  # Respect rate limits
        
        data = response.json()
        cik = data['cik']
        
        # Pad to 10 digits
        return cik.zfill(10)
    
    def get_filings(self, cik: str, filing_type: str = '10-K', 
                    count: int = 10) -> list:
        """
        Get recent filings for a company
        
        Args:
            cik: Company CIK
            filing_type: Type of filing (10-K, 10-Q, 8-K, etc.)
            count: Number of filings to retrieve
            
        Returns:
            List of filing metadata
        """
        url = f"{self.BASE_URL}/cgi-bin/browse-edgar"
        params = {
            'action': 'getcompany',
            'CIK': cik,
            'type': filing_type,
            'dateb': '',
            'owner': 'exclude',
            'count': count,
            'output': 'json'
        }
        
        response = requests.get(url, params=params, headers=self.headers)
        time.sleep(0.1)
        
        filings = response.json()['filings']['recent']
        
        return [{
            'filing_type': filing_type,
            'filing_date': date,
            'accession_number': acc_num,
            'url': f"{self.BASE_URL}/Archives/edgar/data/{cik}/{acc_num.replace('-', '')}/{acc_num}-index.htm"
        } for date, acc_num in zip(filings['filingDate'], filings['accessionNumber'])]
    
    def download_filing_text(self, accession_number: str, cik: str) -> str:
        """
        Download the full text of a filing
        
        Args:
            accession_number: SEC accession number
            cik: Company CIK
            
        Returns:
            Full text content
        """
        # Remove dashes from accession number for URL
        acc_num_no_dash = accession_number.replace('-', '')
        
        # Primary document is usually the .txt file
        url = f"{self.BASE_URL}/Archives/edgar/data/{cik}/{acc_num_no_dash}/{accession_number}.txt"
        
        response = requests.get(url, headers=self.headers)
        time.sleep(0.1)
        
        if response.status_code == 200:
            return response.text
        else:
            raise ValueError(f"Failed to download filing: {response.status_code}")

# Example usage
if __name__ == "__main__":
    retriever = SECFilingRetriever(user_agent="YourName your@email.com")
    
    # Get Apple's CIK
    cik = retriever.get_company_cik('AAPL')
    print(f"Apple CIK: {cik}")
    
    # Get recent 10-K filings
    filings = retriever.get_filings(cik, filing_type='10-K', count=3)
    print(f"\\nFound {len(filings)} recent 10-K filings")
    
    # Download most recent filing
    if filings:
        filing = filings[0]
        print(f"\\nDownloading filing from {filing['filing_date']}...")
        text = retriever.download_filing_text(filing['accession_number'], cik)
        print(f"Downloaded {len(text)} characters")
\`\`\`

---

## Parsing SEC Filings

### Extracting Structured Data

\`\`\`python
"""
Parse SEC filings to extract structured sections
"""

import re
from bs4 import BeautifulSoup

class SECFilingParser:
    """
    Parse SEC filings to extract key sections
    """
    
    # Common section headers in 10-K filings
    SECTION_PATTERNS = {
        'business': r'ITEM\\s+1[.:\\s]+BUSINESS',
        'risk_factors': r'ITEM\\s+1A[.:\\s]+RISK\\s+FACTORS',
        'properties': r'ITEM\\s+2[.:\\s]+PROPERTIES',
        'legal': r'ITEM\\s+3[.:\\s]+LEGAL\\s+PROCEEDINGS',
        'mda': r'ITEM\\s+7[.:\\s]+MANAGEMENT.?S\\s+DISCUSSION',
        'financials': r'ITEM\\s+8[.:\\s]+FINANCIAL\\s+STATEMENTS',
        'controls': r'ITEM\\s+9A[.:\\s]+CONTROLS\\s+AND\\s+PROCEDURES',
    }
    
    def clean_text(self, text: str) -> str:
        """
        Clean HTML and formatting from filing text
        """
        # Parse HTML
        soup = BeautifulSoup(text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def extract_section(self, text: str, section_name: str) -> str:
        """
        Extract a specific section from the filing
        
        Args:
            text: Full filing text
            section_name: Section key from SECTION_PATTERNS
            
        Returns:
            Extracted section text
        """
        pattern = self.SECTION_PATTERNS.get(section_name)
        if not pattern:
            raise ValueError(f"Unknown section: {section_name}")
        
        # Find start of section
        start_match = re.search(pattern, text, re.IGNORECASE)
        if not start_match:
            return ""
        
        start_pos = start_match.end()
        
        # Find start of next section (any ITEM)
        next_section = re.search(r'ITEM\\s+\\d+[A-Z]?[.:\\s]+', 
                                text[start_pos:], re.IGNORECASE)
        
        if next_section:
            end_pos = start_pos + next_section.start()
            return text[start_pos:end_pos].strip()
        else:
            # If no next section found, take rest of document (unlikely)
            return text[start_pos:].strip()
    
    def extract_all_sections(self, text: str) -> dict:
        """
        Extract all major sections from filing
        
        Returns:
            Dictionary mapping section names to content
        """
        # Clean text first
        cleaned = self.clean_text(text)
        
        sections = {}
        for section_name in self.SECTION_PATTERNS.keys():
            try:
                sections[section_name] = self.extract_section(cleaned, section_name)
            except Exception as e:
                print(f"Warning: Could not extract {section_name}: {e}")
                sections[section_name] = ""
        
        return sections
    
    def extract_tables(self, text: str) -> list:
        """
        Extract financial tables from HTML filing
        
        Returns:
            List of pandas DataFrames
        """
        import pandas as pd
        
        soup = BeautifulSoup(text, 'html.parser')
        tables = soup.find_all('table')
        
        dfs = []
        for table in tables:
            try:
                df = pd.read_html(str(table))[0]
                # Only keep tables with reasonable size
                if len(df) > 2 and len(df.columns) > 1:
                    dfs.append(df)
            except Exception:
                continue
        
        return dfs

# Example usage
parser = SECFilingParser()

# Assuming we have filing text from previous example
# text = retriever.download_filing_text(...)

# Extract all sections
sections = parser.extract_all_sections(text)

# Display section lengths
for section_name, content in sections.items():
    print(f"{section_name}: {len(content)} characters")

# Extract risk factors section specifically
risk_factors = sections['risk_factors']
print(f"\\nRisk Factors section preview:")
print(risk_factors[:500])
\`\`\`

---

## LLM-Based Analysis of Financial Documents

### Risk Factor Analysis

\`\`\`python
"""
Use LLMs to analyze risk factors in SEC filings
"""

import anthropic
from typing import List, Dict

class FinancialDocumentAnalyzer:
    """
    Analyze financial documents using Claude LLM
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def analyze_risk_factors(self, risk_factors_text: str) -> Dict:
        """
        Analyze risk factors section and categorize risks
        
        Args:
            risk_factors_text: Text from Risk Factors section
            
        Returns:
            Structured analysis of risks
        """
        prompt = f"""Analyze the following Risk Factors section from a 10-K filing.

Please provide:
1. A list of the top 5 most material risks (with brief explanation)
2. Risk categorization (market, operational, financial, regulatory, etc.)
3. New risks compared to typical disclosure (if any stand out)
4. Overall risk level assessment (Low/Medium/High)
5. Key risk themes or patterns

Risk Factors:
{risk_factors_text[:8000]}  # Limit to fit in context

Provide your analysis in JSON format."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def extract_key_metrics(self, mda_text: str) -> Dict:
        """
        Extract key financial metrics mentioned in MD&A
        
        Args:
            mda_text: Management Discussion & Analysis text
            
        Returns:
            Dictionary of extracted metrics
        """
        prompt = f"""Extract key financial metrics and KPIs mentioned in this MD&A section.

Please identify and extract:
1. Revenue/sales figures and growth rates
2. Profit margins and profitability metrics
3. Cash flow metrics
4. Guidance for next period (if mentioned)
5. Major business drivers mentioned
6. Key operational metrics (users, units sold, etc.)

Present findings in a structured JSON format with metric names and values.

MD&A:
{mda_text[:10000]}

Provide your analysis as structured JSON."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def compare_filings(self, current_text: str, previous_text: str, 
                       section: str = 'risk_factors') -> str:
        """
        Compare two filings to identify changes
        
        Args:
            current_text: Current period filing text
            previous_text: Previous period filing text
            section: Which section to compare
            
        Returns:
            Analysis of changes
        """
        prompt = f"""Compare these two versions of the {section} section from consecutive filings.

Identify:
1. New risks or concerns added
2. Risks that were removed or de-emphasized
3. Changes in language or tone
4. Material changes in company circumstances
5. Overall trend (improving/worsening)

Previous Filing:
{previous_text[:6000]}

Current Filing:
{current_text[:6000]}

Provide a detailed comparison analysis."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def summarize_filing(self, sections: Dict[str, str]) -> str:
        """
        Create executive summary of entire filing
        
        Args:
            sections: Dictionary of section names to content
            
        Returns:
            Executive summary
        """
        # Combine key sections (truncated to fit context)
        combined_text = f"""
Business Overview:
{sections.get('business', '')[:3000]}

Risk Factors:
{sections.get('risk_factors', '')[:3000]}

MD&A:
{sections.get('mda', '')[:4000]}
"""

        prompt = f"""Create a comprehensive executive summary of this 10-K filing.

Include:
1. Company business overview (2-3 sentences)
2. Financial performance highlights
3. Top 3 risks
4. Management's outlook
5. Key takeaways for investors
6. Notable changes from previous filings (if evident)

Keep the summary concise but informative (400-600 words).

Filing Sections:
{combined_text}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text

# Example usage
analyzer = FinancialDocumentAnalyzer(api_key="your-key")

# Analyze risk factors
risk_analysis = analyzer.analyze_risk_factors(sections['risk_factors'])
print("Risk Analysis:")
print(risk_analysis)

# Extract metrics from MD&A
metrics = analyzer.extract_key_metrics(sections['mda'])
print("\\nExtracted Metrics:")
print(metrics)

# Generate executive summary
summary = analyzer.summarize_filing(sections)
print("\\nExecutive Summary:")
print(summary)
\`\`\`

---

## Advanced: Long Document Processing

### Chunking and Map-Reduce Pattern

\`\`\`python
"""
Handle very long documents that exceed context limits
"""

from typing import List
import tiktoken

class LongDocumentProcessor:
    """
    Process documents longer than LLM context window
    """
    
    def __init__(self, client, model: str = "claude-3-5-sonnet-20241022"):
        self.client = client
        self.model = model
        # Claude has 200k token context, but we'll be conservative
        self.max_chunk_tokens = 8000
        
    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text
        """
        # Use GPT tokenizer as approximation
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    
    def chunk_text(self, text: str, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Full text to chunk
            overlap: Number of tokens to overlap between chunks
            
        Returns:
            List of text chunks
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = start + self.max_chunk_tokens
            chunk_tokens = tokens[start:end]
            chunk_text = encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move forward, accounting for overlap
            start = end - overlap
        
        return chunks
    
    def map_reduce_analysis(self, text: str, analysis_type: str) -> str:
        """
        Analyze long document using map-reduce pattern
        
        Args:
            text: Long document text
            analysis_type: Type of analysis to perform
            
        Returns:
            Final aggregated analysis
        """
        # Step 1: MAP - Analyze each chunk
        chunks = self.chunk_text(text)
        print(f"Processing {len(chunks)} chunks...")
        
        chunk_analyses = []
        for i, chunk in enumerate(chunks):
            print(f"Analyzing chunk {i+1}/{len(chunks)}...")
            
            prompt = f"""Analyze this section of a financial document.
Focus on: {analysis_type}

Extract key information, insights, and important details.

Text:
{chunk}"""

            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            chunk_analyses.append(response.content[0].text)
        
        # Step 2: REDUCE - Synthesize all analyses
        print("Synthesizing results...")
        
        combined_analyses = "\\n\\n---\\n\\n".join([
            f"Chunk {i+1} Analysis:\\n{analysis}"
            for i, analysis in enumerate(chunk_analyses)
        ])
        
        synthesis_prompt = f"""Synthesize these analyses of different sections of a financial document.

Create a comprehensive, coherent analysis that:
1. Combines insights from all sections
2. Identifies patterns and themes
3. Highlights the most important findings
4. Provides actionable conclusions

Focus on: {analysis_type}

Section Analyses:
{combined_analyses}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2500,
            messages=[{"role": "user", "content": synthesis_prompt}]
        )
        
        return response.content[0].text

# Example usage
processor = LongDocumentProcessor(analyzer.client)

# Analyze very long MD&A section
long_mda = sections['mda']  # Could be 50+ pages

analysis = processor.map_reduce_analysis(
    text=long_mda,
    analysis_type="financial performance, business drivers, and management outlook"
)

print("Comprehensive MD&A Analysis:")
print(analysis)
\`\`\`

---

## Structured Data Extraction

### Using Function Calling for Structured Output

\`\`\`python
"""
Extract structured data from financial documents using function calling
"""

import json
from typing import List, Optional
from pydantic import BaseModel, Field

class RiskFactor(BaseModel):
    """Structured risk factor"""
    category: str = Field(description="Risk category (market, operational, regulatory, etc.)")
    description: str = Field(description="Brief description of the risk")
    severity: str = Field(description="Severity level (Low/Medium/High)")
    likelihood: str = Field(description="Likelihood of occurrence (Low/Medium/High)")

class FinancialMetrics(BaseModel):
    """Structured financial metrics"""
    revenue: Optional[float] = Field(description="Revenue in millions")
    revenue_growth: Optional[float] = Field(description="Revenue growth rate as percentage")
    net_income: Optional[float] = Field(description="Net income in millions")
    profit_margin: Optional[float] = Field(description="Profit margin as percentage")
    eps: Optional[float] = Field(description="Earnings per share")
    free_cash_flow: Optional[float] = Field(description="Free cash flow in millions")

class FilingAnalysis(BaseModel):
    """Complete filing analysis"""
    summary: str = Field(description="Executive summary of the filing")
    key_risks: List[RiskFactor] = Field(description="Top 5 key risks")
    financial_metrics: FinancialMetrics = Field(description="Key financial metrics")
    outlook: str = Field(description="Management's outlook and guidance")
    sentiment: str = Field(description="Overall sentiment (Positive/Neutral/Negative)")
    red_flags: List[str] = Field(description="Any concerning signals or red flags")

class StructuredDocumentAnalyzer:
    """
    Extract structured data from financial documents
    """
    
    def __init__(self, client):
        self.client = client
        self.model = "claude-3-5-sonnet-20241022"
    
    def extract_structured_analysis(self, sections: Dict[str, str]) -> FilingAnalysis:
        """
        Extract structured analysis from filing sections
        
        Args:
            sections: Dictionary of section names to content
            
        Returns:
            Structured FilingAnalysis object
        """
        # Combine key sections
        combined_text = f"""
Business: {sections.get('business', '')[:2000]}
Risk Factors: {sections.get('risk_factors', '')[:4000]}
MD&A: {sections.get('mda', '')[:4000]}
"""

        prompt = f"""Analyze this 10-K filing and extract structured information.

Filing Content:
{combined_text}

Provide a comprehensive analysis including:
- Executive summary
- Top 5 key risks with categories, severity, and likelihood
- Financial metrics (extract actual numbers mentioned)
- Management's outlook
- Overall sentiment
- Any red flags or concerns

Return your analysis as a JSON object matching this structure:
{json.dumps(FilingAnalysis.schema(), indent=2)}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse JSON response
        response_text = response.content[0].text
        
        # Extract JSON from response (handling markdown code blocks)
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text
        
        # Parse into structured object
        data = json.loads(json_str)
        return FilingAnalysis(**data)

# Example usage
structured_analyzer = StructuredDocumentAnalyzer(analyzer.client)

# Extract structured analysis
analysis = structured_analyzer.extract_structured_analysis(sections)

print("Structured Analysis:")
print(f"Summary: {analysis.summary}")
print(f"\\nFinancial Metrics:")
print(f"  Revenue: ${analysis.financial_metrics.revenue}M")
print(f"  Growth: {analysis.financial_metrics.revenue_growth}%")
print(f"  Profit Margin: {analysis.financial_metrics.profit_margin}%")
print(f"\\nTop Risks:")
for risk in analysis.key_risks:
    print(f"  - {risk.category}: {risk.description}")
    print(f"    Severity: {risk.severity}, Likelihood: {risk.likelihood}")
print(f"\\nOutlook: {analysis.outlook}")
print(f"Sentiment: {analysis.sentiment}")

if analysis.red_flags:
    print(f"\\nRed Flags:")
    for flag in analysis.red_flags:
        print(f"  - {flag}")
\`\`\`

---

## Production Pipeline

### End-to-End Filing Analysis System

\`\`\`python
"""
Complete production pipeline for analyzing SEC filings
"""

import pandas as pd
from datetime import datetime
import sqlite3

class FilingAnalysisPipeline:
    """
    Complete pipeline for retrieving and analyzing SEC filings
    """
    
    def __init__(self, sec_user_agent: str, anthropic_api_key: str, 
                 db_path: str = "filings.db"):
        self.retriever = SECFilingRetriever(sec_user_agent)
        self.parser = SECFilingParser()
        self.analyzer = FinancialDocumentAnalyzer(anthropic_api_key)
        self.structured_analyzer = StructuredDocumentAnalyzer(self.analyzer.client)
        
        # Initialize database
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Create database tables for storing analyses"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS filings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                cik TEXT,
                filing_type TEXT,
                filing_date TEXT,
                accession_number TEXT UNIQUE,
                processed_date TEXT,
                summary TEXT,
                sentiment TEXT,
                outlook TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filing_id INTEGER,
                category TEXT,
                description TEXT,
                severity TEXT,
                likelihood TEXT,
                FOREIGN KEY (filing_id) REFERENCES filings(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filing_id INTEGER,
                revenue REAL,
                revenue_growth REAL,
                net_income REAL,
                profit_margin REAL,
                eps REAL,
                free_cash_flow REAL,
                FOREIGN KEY (filing_id) REFERENCES filings(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def process_company(self, ticker: str, filing_type: str = '10-K', 
                       count: int = 1) -> pd.DataFrame:
        """
        Process recent filings for a company
        
        Args:
            ticker: Stock ticker
            filing_type: Type of filing to process
            count: Number of recent filings to process
            
        Returns:
            DataFrame with analysis results
        """
        print(f"Processing {ticker} {filing_type} filings...")
        
        # Get company CIK
        cik = self.retriever.get_company_cik(ticker)
        
        # Get recent filings
        filings = self.retriever.get_filings(cik, filing_type, count)
        
        results = []
        
        for filing in filings:
            print(f"\\nProcessing {filing['filing_date']}...")
            
            # Check if already processed
            if self._is_processed(filing['accession_number']):
                print(f"Already processed, skipping...")
                continue
            
            try:
                # Download filing
                text = self.retriever.download_filing_text(
                    filing['accession_number'], cik
                )
                
                # Parse sections
                sections = self.parser.extract_all_sections(text)
                
                # Analyze with LLM
                analysis = self.structured_analyzer.extract_structured_analysis(sections)
                
                # Store in database
                filing_id = self._store_analysis(
                    ticker=ticker,
                    cik=cik,
                    filing=filing,
                    analysis=analysis
                )
                
                results.append({
                    'ticker': ticker,
                    'filing_date': filing['filing_date'],
                    'filing_type': filing_type,
                    'summary': analysis.summary,
                    'sentiment': analysis.sentiment,
                    'revenue': analysis.financial_metrics.revenue,
                    'revenue_growth': analysis.financial_metrics.revenue_growth,
                    'filing_id': filing_id
                })
                
                print(f"Successfully processed and stored analysis")
                
            except Exception as e:
                print(f"Error processing filing: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def _is_processed(self, accession_number: str) -> bool:
        """Check if filing already processed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT COUNT(*) FROM filings WHERE accession_number = ?",
            (accession_number,)
        )
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
    
    def _store_analysis(self, ticker: str, cik: str, filing: Dict, 
                       analysis: FilingAnalysis) -> int:
        """Store analysis in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store filing
        cursor.execute("""
            INSERT INTO filings 
            (ticker, cik, filing_type, filing_date, accession_number, 
             processed_date, summary, sentiment, outlook)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker, cik, filing['filing_type'], filing['filing_date'],
            filing['accession_number'], datetime.now().isoformat(),
            analysis.summary, analysis.sentiment, analysis.outlook
        ))
        
        filing_id = cursor.lastrowid
        
        # Store risks
        for risk in analysis.key_risks:
            cursor.execute("""
                INSERT INTO risks 
                (filing_id, category, description, severity, likelihood)
                VALUES (?, ?, ?, ?, ?)
            """, (
                filing_id, risk.category, risk.description,
                risk.severity, risk.likelihood
            ))
        
        # Store metrics
        m = analysis.financial_metrics
        cursor.execute("""
            INSERT INTO metrics 
            (filing_id, revenue, revenue_growth, net_income, 
             profit_margin, eps, free_cash_flow)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            filing_id, m.revenue, m.revenue_growth, m.net_income,
            m.profit_margin, m.eps, m.free_cash_flow
        ))
        
        conn.commit()
        conn.close()
        
        return filing_id
    
    def get_company_history(self, ticker: str) -> pd.DataFrame:
        """Get historical analysis for a company"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                f.filing_date,
                f.filing_type,
                f.sentiment,
                m.revenue,
                m.revenue_growth,
                m.profit_margin
            FROM filings f
            LEFT JOIN metrics m ON f.id = m.filing_id
            WHERE f.ticker = ?
            ORDER BY f.filing_date DESC
        """
        
        df = pd.read_sql_query(query, conn, params=(ticker,))
        conn.close()
        
        return df

# Example usage
if __name__ == "__main__":
    pipeline = FilingAnalysisPipeline(
        sec_user_agent="YourName your@email.com",
        anthropic_api_key="your-key"
    )
    
    # Process recent filings for multiple companies
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    for ticker in tickers:
        try:
            results = pipeline.process_company(ticker, filing_type='10-K', count=3)
            print(f"\\n{ticker} Analysis Summary:")
            print(results)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    # Get historical analysis
    history = pipeline.get_company_history('AAPL')
    print("\\nApple Historical Filings:")
    print(history)
\`\`\`

---

## Best Practices and Considerations

### Key Takeaways

1. **Use Structured Outputs**: Always extract structured data (JSON) rather than freeform text for downstream analysis

2. **Handle Long Documents**: Use chunking and map-reduce patterns for documents exceeding context limits

3. **Validate Extractions**: Cross-reference LLM extractions with actual financial tables when possible

4. **Cache Analyses**: Store analyses in a database to avoid reprocessing the same documents

5. **Rate Limiting**: Respect both SEC API rate limits (10 requests/second) and LLM API rate limits

6. **Error Handling**: Financial documents have inconsistent formats; robust error handling is essential

7. **Prompt Engineering**: Use specific, structured prompts to get consistent output format

8. **Comparative Analysis**: Compare filings over time to identify trends and changes

9. **Multi-Source Validation**: Use multiple sections (business, MD&A, risk factors) to validate findings

10. **Human Review**: LLMs can hallucinate; critical decisions should involve human review

### Common Pitfalls

- **Context Overflow**: 10-K filings often exceed even large context windows
- **Table Extraction**: Financial tables in HTML can be complex to parse
- **Hallucination Risk**: LLMs may generate plausible but incorrect financial metrics
- **Inconsistent Formats**: Companies format filings differently
- **XBRL vs Text**: Consider using XBRL data for precise financial metrics

---

## Summary

We've covered:
- Retrieving SEC filings programmatically from EDGAR
- Parsing and extracting sections from complex financial documents
- Using LLMs to analyze risk factors, MD&A, and other sections
- Handling long documents with chunking and map-reduce
- Extracting structured data with function calling
- Building production pipelines for automated analysis
- Best practices for financial document processing

In the next section, we'll apply similar techniques to earnings call transcripts and real-time earnings analysis.
`,
};

