export const section6 = {
  title: 'SEC EDGAR & Automated Filing Analysis',
  content: `
# SEC EDGAR & Automated Filing Analysis

The SEC EDGAR database contains every public company's filings. Master automated extraction and analysis of:
- **10-K/10-Q**: Annual and quarterly reports
- **8-K**: Material events
- **DEF 14A**: Proxy statements
- **Form 4**: Insider transactions
- **13F**: Institutional holdings

## Section 1: EDGAR API & Data Extraction

\`\`\`python
import requests
import pandas as pd
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
import re
from typing import Dict, List
import json

class EDGARDownloader:
    """Download filings from SEC EDGAR."""
    
    def __init__(self, email: str, company_name: str):
        self.dl = Downloader(company_name, email)
        self.base_url = "https://www.sec.gov"
        self.headers = {'User-Agent': f'{company_name} {email}'}
    
    def get_cik(self, ticker: str) -> str:
        """Get CIK number from ticker symbol."""
        url = f"{self.base_url}/cgi-bin/browse-edgar"
        params = {
            'action': 'getcompany',
            'CIK': ticker,
            'type': '10-K',
            'dateb': '',
            'owner': 'exclude',
            'count': '1',
            'search_text': ''
        }
        
        response = requests.get(url, params=params, headers=self.headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        cik_element = soup.find('span', {'class': 'companyName'})
        
        if cik_element:
            cik = re.search(r'CIK=(\d+)', str(cik_element)).group(1)
            return cik.zfill(10)
        return None
    
    def download_10k(self, ticker: str, num_filings: int = 5):
        """Download recent 10-K filings."""
        self.dl.get("10-K", ticker, amount=num_filings)
        print(f"Downloaded {num_filings} 10-K filings for {ticker}")
    
    def download_10q(self, ticker: str, num_filings: int = 8):
        """Download recent 10-Q filings."""
        self.dl.get("10-Q", ticker, amount=num_filings)
        print(f"Downloaded {num_filings} 10-Q filings for {ticker}")
    
    def get_company_facts(self, cik: str) -> Dict:
        """Get company facts using new SEC API."""
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        return None

# Example usage
downloader = EDGARDownloader("your.email@example.com", "YourCompany")
cik = downloader.get_cik("AAPL")
print(f"Apple CIK: {cik}")

# Download filings
# downloader.download_10k("AAPL", num_filings=3)

# Get company facts
facts = downloader.get_company_facts(cik)
\`\`\`

## Section 2: XBRL Parsing for Financial Data

\`\`\`python
from secedgar import CompanyFilings, FilingType
import xml.etree.ElementTree as ET

class XBRLParser:
    """Parse XBRL financial data from SEC filings."""
    
    def __init__(self, filing_path: str):
        self.filing_path = filing_path
        self.namespaces = {
            'xbrli': 'http://www.xbrl.org/2003/instance',
            'us-gaap': 'http://fasb.org/us-gaap/2021-01-31',
            'dei': 'http://xbrl.sec.gov/dei/2021'
        }
    
    def extract_financial_statements(self, facts_json: Dict) -> pd.DataFrame:
        """Extract financial statements from company facts JSON."""
        
        us_gaap = facts_json.get('facts', {}).get('us-gaap', {})
        
        # Key financial metrics
        metrics = {
            'Revenue': 'Revenues',
            'NetIncome': 'NetIncomeLoss',
            'TotalAssets': 'Assets',
            'TotalLiabilities': 'Liabilities',
            'StockholdersEquity': 'StockholdersEquity',
            'CashAndCashEquivalents': 'CashAndCashEquivalentsAtCarryingValue',
            'OperatingCashFlow': 'NetCashProvidedByUsedInOperatingActivities',
            'CapEx': 'PaymentsToAcquirePropertyPlantAndEquipment'
        }
        
        data = []
        for metric_name, xbrl_tag in metrics.items():
            if xbrl_tag in us_gaap:
                for unit_type in us_gaap[xbrl_tag]['units']:
                    for item in us_gaap[xbrl_tag]['units'][unit_type]:
                        if 'fy' in item:  # Annual data
                            data.append({
                                'metric': metric_name,
                                'date': item['end'],
                                'fiscal_year': item['fy'],
                                'value': item['val'],
                                'form': item['form']
                            })
        
        df = pd.DataFrame(data)
        return df.pivot(index='date', columns='metric', values='value')
    
    def extract_with_context(self, facts_json: Dict, metric: str, 
                            fiscal_year: int) -> float:
        """Extract specific metric for specific fiscal year."""
        
        us_gaap = facts_json.get('facts', {}).get('us-gaap', {})
        
        if metric in us_gaap:
            for unit_type in us_gaap[metric]['units']:
                for item in us_gaap[metric]['units'][unit_type]:
                    if item.get('fy') == fiscal_year and item.get('form') == '10-K':
                        return item['val']
        
        return None

# Example
parser = XBRLParser("")
# df = parser.extract_financial_statements(facts)
# print(df)
\`\`\`

## Section 3: Automated Financial Statement Extraction

\`\`\`python
class FinancialStatementExtractor:
    """Extract and structure financial statements."""
    
    @staticmethod
    def extract_income_statement(facts: Dict, years: List[int]) -> pd.DataFrame:
        """Build income statement from XBRL data."""
        
        us_gaap = facts.get('facts', {}).get('us-gaap', {})
        
        income_statement_items = {
            'Revenue': 'Revenues',
            'Cost of Revenue': 'CostOfRevenue',
            'Gross Profit': 'GrossProfit',
            'Operating Expenses': 'OperatingExpenses',
            'Operating Income': 'OperatingIncomeLoss',
            'Interest Expense': 'InterestExpense',
            'Income Before Tax': 'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
            'Income Tax Expense': 'IncomeTaxExpenseBenefit',
            'Net Income': 'NetIncomeLoss',
            'EPS Basic': 'EarningsPerShareBasic',
            'EPS Diluted': 'EarningsPerShareDiluted'
        }
        
        data = {}
        for item_name, xbrl_tag in income_statement_items.items():
            if xbrl_tag in us_gaap:
                yearly_values = {}
                for unit_type in us_gaap[xbrl_tag]['units']:
                    for entry in us_gaap[xbrl_tag]['units'][unit_type]:
                        if entry.get('form') == '10-K' and entry.get('fy') in years:
                            yearly_values[entry['fy']] = entry['val']
                data[item_name] = yearly_values
        
        df = pd.DataFrame(data).T
        return df
    
    @staticmethod
    def extract_balance_sheet(facts: Dict, years: List[int]) -> pd.DataFrame:
        """Build balance sheet from XBRL data."""
        
        balance_sheet_items = {
            'Cash': 'CashAndCashEquivalentsAtCarryingValue',
            'Accounts Receivable': 'AccountsReceivableNetCurrent',
            'Inventory': 'InventoryNet',
            'Current Assets': 'AssetsCurrent',
            'PP&E': 'PropertyPlantAndEquipmentNet',
            'Total Assets': 'Assets',
            'Current Liabilities': 'LiabilitiesCurrent',
            'Long-term Debt': 'LongTermDebtNoncurrent',
            'Total Liabilities': 'Liabilities',
            'Stockholders Equity': 'StockholdersEquity'
        }
        
        # Similar extraction logic as income statement
        pass
    
    @staticmethod
    def extract_cash_flow(facts: Dict, years: List[int]) -> pd.DataFrame:
        """Build cash flow statement from XBRL data."""
        
        cash_flow_items = {
            'Net Income': 'NetIncomeLoss',
            'Depreciation': 'DepreciationDepletionAndAmortization',
            'Stock-Based Comp': 'ShareBasedCompensation',
            'Changes in Working Capital': 'IncreaseDecreaseInOperatingCapital',
            'Operating Cash Flow': 'NetCashProvidedByUsedInOperatingActivities',
            'CapEx': 'PaymentsToAcquirePropertyPlantAndEquipment',
            'Investing Cash Flow': 'NetCashProvidedByUsedInInvestingActivities',
            'Financing Cash Flow': 'NetCashProvidedByUsedInFinancingActivities'
        }
        
        # Extraction logic
        pass

extractor = FinancialStatementExtractor()
\`\`\`

## Section 4: MD&A and Footnote Analysis with NLP

\`\`\`python
from transformers import pipeline
import spacy

class MDAAanalyzer:
    """Analyze Management Discussion & Analysis section."""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment_analyzer = pipeline("sentiment-analysis")
    
    def extract_mda_section(self, html_content: str) -> str:
        """Extract MD&A section from 10-K HTML."""
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # MD&A is typically in Item 7
        patterns = [
            r'ITEM\s+7[\.:]?\s*MANAGEMENT',
            r'MANAGEMENT.*DISCUSSION.*ANALYSIS',
            r'MD&A'
        ]
        
        text = soup.get_text()
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start_idx = match.start()
                # Find end (usually Item 8)
                end_match = re.search(r'ITEM\s+8[\.:]?', text[start_idx:], re.IGNORECASE)
                if end_match:
                    end_idx = start_idx + end_match.start()
                    return text[start_idx:end_idx]
        
        return ""
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of MD&A text."""
        
        # Split into chunks (models have token limits)
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        
        sentiments = []
        for chunk in chunks[:10]:  # Analyze first 5000 chars
            if chunk.strip():
                result = self.sentiment_analyzer(chunk)[0]
                sentiments.append(result)
        
        # Aggregate
        positive = sum(1 for s in sentiments if s['label'] == 'POSITIVE')
        negative = sum(1 for s in sentiments if s['label'] == 'NEGATIVE')
        
        return {
            'positive_pct': positive / len(sentiments) if sentiments else 0,
            'negative_pct': negative / len(sentiments) if sentiments else 0,
            'overall': 'POSITIVE' if positive > negative else 'NEGATIVE'
        }
    
    def extract_risk_factors(self, text: str) -> List[str]:
        """Extract key risk factors mentioned."""
        
        doc = self.nlp(text)
        
        # Look for risk-related sentences
        risk_keywords = ['risk', 'uncertainty', 'adverse', 'challenge', 'volatile']
        
        risk_sentences = []
        for sent in doc.sents:
            if any(keyword in sent.text.lower() for keyword in risk_keywords):
                risk_sentences.append(sent.text.strip())
        
        return risk_sentences[:10]  # Top 10

analyzer = MDAAanalyzer()
\`\`\`

## Section 5: Form 4 Insider Trading Analysis

\`\`\`python
class InsiderTradingAnalyzer:
    """Analyze Form 4 insider transactions."""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.transactions = []
    
    def download_form4_filings(self, num_filings: int = 50) -> List[Dict]:
        """Download recent Form 4 filings."""
        
        # Use sec-edgar-downloader or SEC API
        # Parse XML Form 4 files
        pass
    
    def analyze_insider_sentiment(self, transactions: pd.DataFrame) -> Dict:
        """Analyze insider buying/selling patterns."""
        
        # Calculate net insider activity
        buys = transactions[transactions['transaction_type'] == 'P']
        sells = transactions[transactions['transaction_type'] == 'S']
        
        buy_volume = buys['shares'].sum()
        sell_volume = sells['shares'].sum()
        buy_value = (buys['shares'] * buys['price']).sum()
        sell_value = (sells['shares'] * sells['price']).sum()
        
        net_volume = buy_volume - sell_volume
        net_value = buy_value - sell_value
        
        # Insider sentiment score
        if net_value > 1_000_000:
            sentiment = "STRONG BUY - Significant insider buying"
        elif net_value > 0:
            sentiment = "BULLISH - Net insider buying"
        elif net_value > -1_000_000:
            sentiment = "BEARISH - Net insider selling"
        else:
            sentiment = "STRONG SELL - Heavy insider selling"
        
        return {
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'net_volume': net_volume,
            'buy_value': buy_value,
            'sell_value': sell_value,
            'net_value': net_value,
            'sentiment': sentiment
        }
\`\`\`

## Section 6: Automated Multi-Company Analysis Pipeline

\`\`\`python
class MultiCompanyPipeline:
    """Automated pipeline for analyzing multiple companies."""
    
    def __init__(self, tickers: List[str], email: str):
        self.tickers = tickers
        self.downloader = EDGARDownloader(email, "AnalysisBot")
        self.results = {}
    
    def run_pipeline(self):
        """Run complete analysis pipeline."""
        
        for ticker in self.tickers:
            print(f"\\nAnalyzing {ticker}...")
            
            # 1. Get CIK
            cik = self.downloader.get_cik(ticker)
            
            # 2. Get company facts
            facts = self.downloader.get_company_facts(cik)
            
            # 3. Extract financials
            parser = XBRLParser("")
            financials = parser.extract_financial_statements(facts)
            
            # 4. Calculate ratios
            metrics = self.calculate_metrics(financials)
            
            # 5. Store results
            self.results[ticker] = {
                'cik': cik,
                'financials': financials,
                'metrics': metrics
            }
        
        # 6. Generate comparison
        comparison = self.generate_comparison()
        return comparison
    
    def calculate_metrics(self, financials: pd.DataFrame) -> Dict:
        """Calculate financial metrics."""
        
        latest = financials.iloc[-1]
        
        return {
            'revenue': latest.get('Revenue', 0),
            'net_income': latest.get('NetIncome', 0),
            'total_assets': latest.get('TotalAssets', 0),
            'net_margin': latest.get('NetIncome', 0) / latest.get('Revenue', 1),
            'roe': latest.get('NetIncome', 0) / latest.get('StockholdersEquity', 1)
        }
    
    def generate_comparison(self) -> pd.DataFrame:
        """Generate peer comparison table."""
        
        data = []
        for ticker, result in self.results.items():
            data.append({
                'Ticker': ticker,
                **result['metrics']
            })
        
        return pd.DataFrame(data)

# Example usage
# pipeline = MultiCompanyPipeline(['AAPL', 'MSFT', 'GOOGL'], "email@example.com")
# results = pipeline.run_pipeline()
# print(results)
\`\`\`

## Key Takeaways

1. **SEC API is powerful** - Direct access to structured financial data
2. **XBRL standardizes data** - Consistent tags across companies
3. **Automate at scale** - Analyze hundreds of companies efficiently
4. **MD&A reveals insights** - NLP can extract sentiment and risks
5. **Insider transactions matter** - Form 4 shows management confidence
6. **Build pipelines** - Automate repetitive analysis tasks

Master EDGAR and you can analyze any public company instantly!
`,
  discussionQuestions: [],
  multipleChoiceQuestions: [],
};
