export const section9 = {
  slug: 'peers',
  title: 'Peer Comparison & Relative Valuation',
  content: `
# Peer Comparison & Relative Valuation

Compare companies using financial metrics and valuation multiples to identify mispriced securities.

## Section 1: Building a Comp Set

\`\`\`python
import pandas as pd
import numpy as np
from typing import List, Dict

class PeerComparator:
    """Build and analyze peer comparison groups."""
    
    @staticmethod
    def build_comp_table (companies: List[Dict]) -> pd.DataFrame:
        """Create peer comparison table."""
        
        comp_data = []
        for company in companies:
            comp_data.append({
                'Company': company['name'],
                'Market Cap ($B)': company['market_cap'] / 1e9,
                'Revenue ($B)': company['revenue'] / 1e9,
                'EBITDA ($M)': company['ebitda'] / 1e6,
                'Net Margin': company['net_income'] / company['revenue'],
                'ROE': company['net_income'] / company['equity'],
                'Debt/EBITDA': company['total_debt'] / company['ebitda'],
                'P/E': company['market_cap'] / company['net_income'],
                'EV/EBITDA': company['enterprise_value'] / company['ebitda'],
                'Revenue Growth': company.get('revenue_growth', 0)
            })
        
        df = pd.DataFrame (comp_data)
        
        # Add median/mean rows
        medians = df.select_dtypes (include=[np.number]).median()
        means = df.select_dtypes (include=[np.number]).mean()
        
        return df
    
    @staticmethod
    def identify_outliers (df: pd.DataFrame, metric: str) -> pd.DataFrame:
        """Identify outliers in specific metric."""
        
        q1 = df[metric].quantile(0.25)
        q3 = df[metric].quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        df['Is_Outlier'] = (df[metric] < lower_bound) | (df[metric] > upper_bound)
        
        return df

# Example peer group
tech_peers = [
    {'name': 'Company A', 'market_cap': 2e12, 'revenue': 400e9, 'ebitda': 130e9,
     'net_income': 100e9, 'equity': 60e9, 'total_debt': 100e9, 'enterprise_value': 2.1e12,
     'revenue_growth': 0.10},
    {'name': 'Company B', 'market_cap': 2.5e12, 'revenue': 200e9, 'ebitda': 80e9,
     'net_income': 60e9, 'equity': 150e9, 'total_debt': 50e9, 'enterprise_value': 2.55e12,
     'revenue_growth': 0.15},
    {'name': 'Company C', 'market_cap': 1.5e12, 'revenue': 150e9, 'ebitda': 40e9,
     'net_income': 30e9, 'equity': 200e9, 'total_debt': 80e9, 'enterprise_value': 1.58e12,
     'revenue_growth': 0.08}
]

comparator = PeerComparator()
comp_table = comparator.build_comp_table (tech_peers)
print("Peer Comparison Table:")
print(comp_table)
\`\`\`

## Section 2: Valuation Multiple Analysis

\`\`\`python
class ValuationAnalyzer:
    """Analyze valuation multiples across peers."""
    
    @staticmethod
    def calculate_relative_valuation(
        target_metric: float,
        peer_multiples: List[float]
    ) -> Dict:
        """Calculate implied valuation using peer multiples."""
        
        median_multiple = np.median (peer_multiples)
        mean_multiple = np.mean (peer_multiples)
        
        implied_value_median = target_metric * median_multiple
        implied_value_mean = target_metric * mean_multiple
        
        return {
            'median_multiple': median_multiple,
            'mean_multiple': mean_multiple,
            'implied_value_median': implied_value_median,
            'implied_value_mean': implied_value_mean
        }
    
    @staticmethod
    def peg_ratio_analysis (pe: float, growth_rate: float) -> Dict:
        """
        PEG Ratio = P/E / (Growth Rate × 100)
        < 1.0 = Undervalued
        ~ 1.0 = Fairly valued
        > 1.0 = Overvalued
        """
        
        peg = pe / (growth_rate * 100) if growth_rate > 0 else None
        
        if peg and peg < 0.75:
            assessment = "UNDERVALUED"
        elif peg and peg < 1.25:
            assessment = "FAIRLY VALUED"
        elif peg:
            assessment = "OVERVALUED"
        else:
            assessment = "N/A"
        
        return {'peg': peg, 'assessment': assessment}
\`\`\`

## Section 3: Regression-Based Valuation

\`\`\`python
from sklearn.linear_model import LinearRegression

class RegressionValuation:
    """Use regression to model valuation based on fundamentals."""
    
    @staticmethod
    def ev_ebitda_model (peers_data: pd.DataFrame) -> Dict:
        """
        Model: EV/EBITDA = f(Margin, Growth, ROIC)
        """
        
        X = peers_data[['Net Margin', 'Revenue Growth', 'ROE']].values
        y = peers_data['EV/EBITDA'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Coefficients
        coefficients = {
            'Net Margin': model.coef_[0],
            'Revenue Growth': model.coef_[1],
            'ROE': model.coef_[2],
            'Intercept': model.intercept_
        }
        
        return {
            'model': model,
            'coefficients': coefficients,
            'r_squared': model.score(X, y)
        }
    
    @staticmethod
    def predict_fair_multiple (model: LinearRegression, company_metrics: Dict) -> float:
        """Predict fair EV/EBITDA multiple for a company."""
        
        X = np.array([[
            company_metrics['net_margin'],
            company_metrics['revenue_growth'],
            company_metrics['roe']
        ]])
        
        return model.predict(X)[0]
\`\`\`

## Key Takeaways

1. **Peer selection matters** - Similar size, geography, business model
2. **Use multiple metrics** - P/E, EV/EBITDA, P/S, PEG
3. **Adjust for differences** - Growth, margins, leverage
4. **Median > Mean** - Less affected by outliers
5. **Context is critical** - Industry norms vary widely
6. **Relative doesn't mean cheap** - Entire sector can be overvalued

Master peer comparison and you can identify mispriced opportunities!
`,
  discussionQuestions: [],
  multipleChoiceQuestions: [],
};
