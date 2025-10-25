export const projectDiscussionQuestions = [
  {
    id: 1,
    question:
      'For the capstone project, design the database schema to store 10 years of financial data for 500 companies. What tables, relationships, and indexes?',
    answer: `**Database Schema Design**:

\`\`\`sql
-- Core tables
CREATE TABLE companies (id, ticker, cik, name, sector, industry);
CREATE TABLE income_statements (id, company_id, period_end, revenue, cogs, gross_profit, ...);
CREATE TABLE balance_sheets (id, company_id, period_end, assets, liabilities, equity, ...);
CREATE TABLE cash_flows (id, company_id, period_end, cfo, cfi, cff, fcf, ...);
CREATE TABLE computed_metrics (id, company_id, period_end, roe, roa, current_ratio, ...);
CREATE TABLE quality_scores (id, company_id, period_end, beneish, altman, piotroski, ...);

-- Indexes for performance
CREATE INDEX idx_company_period ON income_statements (company_id, period_end);
CREATE INDEX idx_ticker ON companies (ticker);

-- Constraints
ALTER TABLE income_statements ADD CONSTRAINT fk_company FOREIGN KEY (company_id) REFERENCES companies (id);
UNIQUE(company_id, period_end, fiscal_quarter);
\`\`\`

**Why**: (1) Normalized structure avoids duplication, (2) Foreign keys ensure referential integrity, (3) Indexes speed up queries by company+date, (4) Unique constraints prevent duplicate data, (5) Separate tables for raw vs computed allows recalculation.`,
  },

  {
    id: 2,
    question:
      "How would you implement automated alerts when any monitored company's Interest Coverage drops below 3.0x?",
    answer: `**Alert System Implementation**:

\`\`\`python
class AlertSystem:
    def check_interest_coverage (self, company_id, new_data):
        current_coverage = new_data['ebit'] / new_data['interest_expense']
        
        if current_coverage < 3.0:
            prior_coverage = self.get_prior_quarter_coverage (company_id)
            
            alert = {
                'severity': 'HIGH' if current_coverage < 2.0 else 'MEDIUM',
                'company': self.get_company_name (company_id),
                'metric': 'Interest Coverage',
                'current': current_coverage,
                'prior': prior_coverage,
                'threshold': 3.0,
                'message': f"Interest coverage dropped to {current_coverage:.1f}x from {prior_coverage:.1f}x"
            }
            
            self.send_email (alert)
            self.post_to_slack (alert)
            self.log_to_database (alert)
\`\`\`

**Trigger**: Run after each quarterly data update. **Channels**: Email, Slack, database logging. **Escalation**: Critical (<2x) alerts go to senior analysts immediately.`,
  },

  {
    id: 3,
    question:
      'Your dashboard needs to show 5-year trend of ROE, Debt/EBITDA, and FCF Margin for a company. How do you design the API endpoint and visualization?',
    answer: `**API Design**:

\`\`\`python
# API endpoint
@app.get("/api/companies/{ticker}/trends")
def get_trends (ticker: str, metrics: List[str], years: int = 5):
    company_id = get_company_id (ticker)
    
    data = db.query("""
        SELECT period_end, roe, debt_to_ebitda, fcf_margin
        FROM computed_metrics
        WHERE company_id = %s
        AND period_end >= CURRENT_DATE - INTERVAL '%s years'
        ORDER BY period_end
    """, (company_id, years))
    
    return {'ticker': ticker, 'trends': data}
\`\`\`

**Frontend Visualization** (React + Plotly):
\`\`\`javascript
const TrendsChart = ({data}) => {
  const traces = [
    {y: data.roe, name: 'ROE', yaxis: 'y'},
    {y: data.debt_to_ebitda, name: 'Debt/EBITDA', yaxis: 'y2'},
    {y: data.fcf_margin, name: 'FCF Margin', yaxis: 'y3'}
  ];
  
  return <Plot data={traces} layout={{    yaxis: {title: 'ROE'},
    yaxis2: {title: 'Leverage', overlaying: 'y', side: 'right'},
    yaxis3: {title: 'FCF Margin', overlaying: 'y', side: 'right'}
  }} />;
};
\`\`\`

**Result**: Multi-axis chart showing all trends simultaneously for easy comparison.`,
  },
];
