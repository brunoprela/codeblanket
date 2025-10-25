export const pipelineDiscussionQuestions = [
  {
    id: 1,
    question:
      'Design database schema for storing 10 years of financial statements for 500 companies. What tables, indexes, and constraints?',
    answer: `**Schema Design**:

\`\`\`sql
-- Companies table
CREATE TABLE companies (
    company_id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) UNIQUE,
    cik VARCHAR(10),
    name VARCHAR(255),
    sector VARCHAR(50),
    industry VARCHAR(100)
);

-- Financial statements (normalized)
CREATE TABLE income_statements (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(company_id),
    period_end DATE,
    fiscal_year INTEGER,
    fiscal_quarter INTEGER,
    revenue NUMERIC(20,2),
    cost_of_revenue NUMERIC(20,2),
    gross_profit NUMERIC(20,2),
    operating_expenses NUMERIC(20,2),
    operating_income NUMERIC(20,2),
    net_income NUMERIC(20,2),
    eps_basic NUMERIC(10,2),
    eps_diluted NUMERIC(10,2),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(company_id, period_end, fiscal_quarter)
);

-- Indexes for performance
CREATE INDEX idx_income_company_period ON income_statements(company_id, period_end);
CREATE INDEX idx_income_fiscal ON income_statements(fiscal_year, fiscal_quarter);

-- Similar tables for balance_sheets, cash_flow_statements

-- Computed metrics table
CREATE TABLE financial_metrics (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES companies(company_id),
    period_end DATE,
    roe NUMERIC(10,4),
    roa NUMERIC(10,4),
    current_ratio NUMERIC(10,2),
    debt_to_equity NUMERIC(10,2),
    fcf NUMERIC(20,2),
    cfo_to_ni NUMERIC(10,2),
    created_at TIMESTAMP DEFAULT NOW()
);
\`\`\`

**Why This Design**: Normalized structure, foreign keys for referential integrity, indexes on query patterns (company+date), unique constraints prevent duplicates, timestamps for audit trail.`,
  },

  {
    id: 2,
    question:
      "Your pipeline processes 500 companies nightly. One company's XBRL parsing fails. How do you handle errors without stopping entire pipeline?",
    answer: `**Error Handling Strategy**:

\`\`\`python
def run_pipeline_with_error_handling(companies):
    results = {'success': [], 'failed': []}
    
    for company in companies:
        try:
            # Extract
            data = extract_data(company)
            
            # Transform
            clean = transform(data)
            
            # Load
            load_to_db(clean)
            
            results['success'].append(company)
            
        except XBRLParseError as e:
            logger.error(f"XBRL parse failed for {company}: {e}")
            results['failed'].append({'company': company, 'error': str(e)})
            continue  # Skip this company, process others
        
        except DatabaseError as e:
            logger.critical(f"DB error for {company}: {e}")
            # Retry logic or alert DevOps
        
        except Exception as e:
            logger.exception(f"Unexpected error for {company}: {e}")
            results['failed'].append({'company': company, 'error': str(e)})
    
    # Summary email
    send_summary_email(results)
    
    return results
\`\`\`

**Key Principles**: (1) Try-except per company, not whole pipeline, (2) Different handling for different error types, (3) Log everything, (4) Continue processing others, (5) Summary report at end, (6) Retry failed companies next run.`,
  },

  {
    id: 3,
    question:
      "Build alert system that notifies when any company's Debt/EBITDA crosses 5.0x. What's the architecture?",
    answer: `**Alert System Architecture**:

\`\`\`python
class AlertSystem:
    def __init__(self):
        self.thresholds = {
            'debt_to_ebitda': {'critical': 5.0, 'warning': 4.0},
            'interest_coverage': {'critical': 2.0, 'warning': 3.0},
            'current_ratio': {'critical': 1.0, 'warning': 1.2}
        }
    
    def check_metrics(self, company_id, metrics):
        alerts = []
        
        # Debt/EBITDA check
        if metrics['debt_to_ebitda'] > self.thresholds['debt_to_ebitda']['critical']:
            alerts.append({
                'company_id': company_id,
                'metric': 'Debt/EBITDA',
                'value': metrics['debt_to_ebitda'],
                'threshold': 5.0,
                'severity': 'CRITICAL',
                'message': f"Debt/EBITDA at {metrics['debt_to_ebitda']:.1f}x, above 5.0x threshold"
            })
        
        return alerts
    
    def send_alert(self, alert):
        # Email
        send_email(to='analyst@firm.com', subject=f"ALERT: {alert['company']}", body=alert['message'])
        
        # Slack
        post_to_slack(channel='#credit-alerts', message=alert['message'])
        
        # Database logging
        log_alert_to_db(alert)
\`\`\`

**Components**: (1) Threshold configuration, (2) Real-time metric checking after each pipeline run, (3) Multi-channel alerts (email, Slack, SMS), (4) Alert history in database, (5) Escalation for critical alerts.`,
  },
];
