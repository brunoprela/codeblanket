export const riskAssessmentLLMs = {
    title: 'Risk Assessment with LLMs',
    id: 'risk-assessment-llms',
    content: `
# Risk Assessment with LLMs

## Introduction

Risk management is critical in finance-poor risk assessment can lead to catastrophic losses. Traditional risk models rely on quantitative metrics (VaR, Beta, volatility), but miss qualitative risks from documents, news, and complex scenarios. LLMs can analyze unstructured risk information, assess counterparty risk, monitor geopolitical threats, evaluate supply chain risks, and build comprehensive early warning systems.

This section covers using LLMs for multi-dimensional risk assessment: credit risk from documents, counterparty analysis, geopolitical risk monitoring, supply chain risk evaluation, and building integrated risk management systems.

### Why LLMs for Risk Assessment

**Unstructured Data**: Analyze documents, news, and reports
**Early Detection**: Identify risks before they materialize in numbers
**Comprehensive Analysis**: Combine multiple risk dimensions
**Contextual Understanding**: Understand nuanced risk factors
**Proactive Management**: Move from reactive to predictive risk management

---

## Credit Risk Analysis from Documents

### Analyzing Financial Documents for Credit Risk

\`\`\`python
"""
Credit risk assessment using LLM analysis of financial documents
"""

import anthropic
from typing import Dict, List, Optional
from datetime import datetime
import json

class CreditRiskAnalyzer:
    """
    Assess credit risk using LLM analysis
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def analyze_credit_risk(self, company_data: Dict) -> Dict:
        """
        Comprehensive credit risk analysis
        
        Args:
            company_data: Financial statements, ratios, and context
            
        Returns:
            Credit risk assessment
        """
        prompt = f"""Analyze the credit risk of this company based on financial data.

Company: {company_data.get('name')}
Industry: {company_data.get('industry')}

Financial Metrics:
- Revenue: \${company_data['financials']['revenue']}M
- Revenue Trend: {company_data['financials']['revenue_trend']}
- EBITDA: \${company_data['financials']['ebitda']}M
- EBITDA Margin: {company_data['financials']['ebitda_margin']}%
- Total Debt: \${company_data['financials']['total_debt']}M
- Cash & Equivalents: \${company_data['financials']['cash']}M
- Debt/EBITDA: {company_data['financials']['debt_ebitda']}x
- Interest Coverage: {company_data['financials']['interest_coverage']}x
- Current Ratio: {company_data['financials']['current_ratio']}
- Quick Ratio: {company_data['financials']['quick_ratio']}

Recent Developments:
{company_data.get('recent_developments', 'None')}

Management Discussion (MD&A excerpts):
{company_data.get('mda_excerpts', 'Not available')}

Industry Context:
{company_data.get('industry_context', 'Not available')}

Provide comprehensive credit risk assessment as JSON:
{{
  "credit_rating": "AAA/AA/A/BBB/BB/B/CCC/CC/C/D",
  "risk_level": "Low/Medium/High/Very High",
  "probability_of_default": "Estimated % over 1 year",
  "key_strengths": ["Financial strengths"],
  "key_weaknesses": ["Financial weaknesses and concerns"],
  "liquidity_assessment": {{
    "score": 0-10,
    "runway": "Estimated months of cash runway",
    "concerns": ["Any liquidity concerns"]
  }},
  "leverage_assessment": {{
    "score": 0-10,
    "debt_sustainability": "Assessment of debt load",
    "refinancing_risk": "High/Medium/Low"
  }},
  "profitability_assessment": {{
    "score": 0-10,
    "margin_trend": "Improving/Stable/Declining",
    "quality_of_earnings": "Assessment"
  }},
  "industry_risk": {{
    "score": 0-10,
    "cyclicality": "High/Medium/Low",
    "competitive_position": "Strong/Average/Weak"
  }},
  "management_quality": {{
    "score": 0-10,
    "assessment": "Based on MD&A tone and decisions"
  }},
  "red_flags": ["Any warning signs"],
  "covenants_at_risk": ["Any financial covenants that might be breached"],
  "overall_assessment": "Comprehensive summary",
  "recommendation": "Invest/Hold/Avoid/Monitor Closely",
  "monitoring_priorities": ["What to watch closely"]
}}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        assessment = self._parse_json(response.content[0].text)
        assessment['company'] = company_data.get('name')
        assessment['analysis_date'] = datetime.now().isoformat()
        
        return assessment
    
    def analyze_debt_covenants(self, covenant_text: str, 
                               financial_metrics: Dict) -> Dict:
        """
        Analyze debt covenant compliance risk
        
        Args:
            covenant_text: Covenant language from debt agreement
            financial_metrics: Current financial metrics
            
        Returns:
            Covenant compliance analysis
        """
        prompt = f"""Analyze debt covenant compliance and risk.

Covenant Language:
{covenant_text}

Current Financial Metrics:
{json.dumps(financial_metrics, indent=2)}

Analyze:
1. What are the specific covenant requirements?
2. Current compliance status for each covenant
3. Cushion/headroom on each covenant
4. Risk of covenant breach in next 12 months
5. Consequences of breach
6. Recommendations for management

Return JSON:
{{
  "covenants": [
    {{
      "covenant_type": "Type (leverage, coverage, etc.)",
      "requirement": "Specific requirement",
      "current_value": "Current metric value",
      "threshold": "Covenant threshold",
      "cushion": "How close to breach (%)",
      "status": "Compliant/At Risk/Breached",
      "risk_level": "Low/Medium/High"
    }}
  ],
  "overall_compliance": "Compliant/At Risk/Breached",
  "highest_risk_covenant": "Which covenant is closest to breach",
  "breach_scenarios": ["Scenarios that could trigger breach"],
  "mitigation_actions": ["Actions to improve covenant position"],
  "recommendations": "Overall recommendations"
}}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return self._parse_json(response.content[0].text)
    
    def assess_going_concern_risk(self, company_filings: Dict,
                                  news_articles: List[str]) -> Dict:
        """
        Assess risk of company's going concern status
        
        Args:
            company_filings: Recent 10-K/10-Q excerpts
            news_articles: Recent news about the company
            
        Returns:
            Going concern risk assessment
        """
        # Combine information
        filing_text = company_filings.get('going_concern_note', '')
        audit_opinion = company_filings.get('audit_opinion', '')
        mda_liquidity = company_filings.get('liquidity_discussion', '')
        
        news_summary = "\\n".join([f"- {article}" for article in news_articles[:10]])
        
        prompt = f"""Assess going concern risk for this company.

Audit Opinion:
{audit_opinion[:1000]}

Going Concern Note (if any):
{filing_text[:2000]}

MD&A Liquidity Discussion:
{mda_liquidity[:2000]}

Recent News:
{news_summary}

Assess going concern risk as JSON:
{{
  "going_concern_risk": "Low/Medium/High/Critical",
  "auditor_concerns": "What auditor highlighted (if anything)",
  "liquidity_runway": "Estimated months until cash crisis",
  "funding_options": ["Available funding sources"],
  "warning_signs": ["Specific warning signs identified"],
  "stress_scenarios": [
    {{
      "scenario": "Description",
      "likelihood": "High/Medium/Low",
      "time_to_crisis": "Estimated timeline",
      "severity": "Impact severity"
    }}
  ],
  "mitigating_factors": ["Factors that reduce risk"],
  "overall_assessment": "Detailed assessment",
  "recommendation": "Action to take",
  "monitoring_triggers": ["Specific events to watch for"]
}}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return self._parse_json(response.content[0].text)
    
    def _parse_json(self, response_text: str) -> Dict:
        """Parse JSON from LLM response"""
        try:
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
    json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
        json_str = response_text
            return json.loads(json_str)
        except:
            return {}

# Example usage
credit_analyzer = CreditRiskAnalyzer(api_key = "your-key")

company_data = {
        'name': 'Example Corp',
        'industry': 'Retail',
        'financials': {
            'revenue': 1200,
            'revenue_trend': 'Declining 5% YoY',
            'ebitda': 150,
            'ebitda_margin': 12.5,
            'total_debt': 800,
            'cash': 100,
            'debt_ebitda': 5.3,
            'interest_coverage': 2.1,
            'current_ratio': 1.2,
            'quick_ratio': 0.8
        },
        'recent_developments': 'Store closures announced, CEO departure',
        'mda_excerpts': 'Management discussing challenging retail environment...',
        'industry_context': 'Retail sector under pressure from e-commerce'
    }

# Analyze credit risk
credit_assessment = credit_analyzer.analyze_credit_risk(company_data)
print("Credit Risk Assessment:")
print(json.dumps(credit_assessment, indent = 2))
\`\`\`

---

## Counterparty Risk Assessment

### Evaluating Business Partner Risk

\`\`\`python
"""
Assess counterparty risk for business relationships
"""

class CounterpartyRiskAssessor:
    """
    Assess risk of business counterparties
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def assess_counterparty(self, counterparty_data: Dict,
                           relationship_type: str) -> Dict:
        """
        Assess counterparty risk
        
        Args:
            counterparty_data: Information about the counterparty
            relationship_type: Type of relationship (supplier, customer, partner, etc.)
            
        Returns:
            Risk assessment
        """
        prompt = f"""Assess counterparty risk for this {relationship_type} relationship.

Counterparty: {counterparty_data.get('name')}
Type: {relationship_type}
Relationship Duration: {counterparty_data.get('relationship_years', 0)} years
Annual Business Volume: \${counterparty_data.get('annual_volume', 0)}M

Financial Health:
- Credit Rating: {counterparty_data.get('credit_rating', 'Unknown')}
- Recent Financial Performance: {counterparty_data.get('financial_performance', 'Unknown')}
- Debt Levels: {counterparty_data.get('debt_levels', 'Unknown')}

Operational Factors:
- Geographic Concentration: {counterparty_data.get('geographic_concentration', 'Unknown')}
- Industry Exposure: {counterparty_data.get('industry_exposure', 'Unknown')}
- Key Person Dependencies: {counterparty_data.get('key_person_risk', 'Unknown')}

Recent Events:
{counterparty_data.get('recent_events', 'None')}

Contractual Terms:
- Payment Terms: {counterparty_data.get('payment_terms', 'Unknown')}
- Contract Duration: {counterparty_data.get('contract_duration', 'Unknown')}
- Termination Clauses: {counterparty_data.get('termination_clauses', 'Unknown')}

Provide risk assessment as JSON:
{{
  "overall_risk": "Low/Medium/High/Critical",
  "risk_score": 0-100,
  "financial_risk": {{
    "score": 0-10,
    "concerns": ["Specific financial concerns"],
    "probability_of_failure": "Estimated %"
  }},
  "operational_risk": {{
    "score": 0-10,
    "concerns": ["Operational concerns"],
    "single_points_of_failure": ["Concentration risks"]
  }},
  "legal_regulatory_risk": {{
    "score": 0-10,
    "concerns": ["Legal or regulatory issues"]
  }},
  "reputational_risk": {{
    "score": 0-10,
    "concerns": ["Reputational concerns"]
  }},
  "concentration_risk": {{
    "score": 0-10,
    "dependency_level": "How dependent are we on them",
    "alternatives": "Availability of alternatives"
  }},
  "red_flags": ["Critical warning signs"],
  "impact_of_failure": {{
    "financial_impact": "Estimated financial impact",
    "operational_impact": "Operational disruption",
    "timeline_to_replace": "Time needed to find alternative"
  }},
  "mitigation_strategies": ["Actions to reduce risk"],
  "monitoring_requirements": ["What to monitor"],
  "recommendation": "Continue/Reduce Exposure/Exit/Monitor Closely"
}}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        assessment = self._parse_json(response.content[0].text)
        assessment['counterparty'] = counterparty_data.get('name')
        assessment['assessment_date'] = datetime.now().isoformat()
        
        return assessment
    
    def assess_supply_chain_risk(self, supplier_data: List[Dict],
                                critical_components: List[str]) -> Dict:
        """
        Assess supply chain risk across multiple suppliers
        
        Args:
            supplier_data: List of supplier information
            critical_components: List of critical components/materials
            
        Returns:
            Supply chain risk assessment
        """
        supplier_summary = json.dumps([
            {
                'name': s['name'],
                'components': s['supplies'],
                'location': s['location'],
                'alternatives': s.get('alternatives', 0)
            }
            for s in supplier_data
        ], indent=2)
        
        prompt = f"""Assess supply chain risk across this supplier network.

Suppliers:
{supplier_summary}

Critical Components:
{json.dumps(critical_components)}

Analyze:
1. Single points of failure in supply chain
2. Geographic concentration risk
3. Supplier financial health concerns
4. Lead time vulnerabilities
5. Alternative source availability
6. Risk mitigation priorities

Return comprehensive supply chain risk assessment as JSON."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return self._parse_json(response.content[0].text)
    
    def _parse_json(self, response_text: str) -> Dict:
        """Parse JSON from response"""
        import json
        try:
            if "```json" in response_text:
json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
json_str = response_text
return json.loads(json_str)
except:
return {}

# Example usage
counterparty_assessor = CounterpartyRiskAssessor(api_key = "your-key")

counterparty_data = {
    'name': 'Key Supplier Inc',
    'relationship_years': 5,
    'annual_volume': 50,
    'credit_rating': 'BB',
    'financial_performance': 'Declining margins, rising debt',
    'debt_levels': 'High leverage (6x EBITDA)',
    'geographic_concentration': 'Single manufacturing facility in Country X',
    'industry_exposure': 'Heavily concentrated in automotive',
    'key_person_risk': 'Founder-CEO owns 60% of company',
    'recent_events': 'Recently lost major customer, announced layoffs',
    'payment_terms': 'Net 60',
    'contract_duration': '2 years remaining',
    'termination_clauses': '90-day notice required'
}

assessment = counterparty_assessor.assess_counterparty(
    counterparty_data,
    relationship_type = 'critical supplier'
)

print("Counterparty Risk Assessment:")
print(json.dumps(assessment, indent = 2))
\`\`\`

---

## Geopolitical Risk Monitoring

### Tracking Global Risk Events

\`\`\`python
"""
Monitor and assess geopolitical risks
"""

from typing import List, Dict
from datetime import datetime, timedelta

class GeopoliticalRiskMonitor:
    """
    Monitor geopolitical risks and assess impact
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def assess_geopolitical_event(self, event_data: Dict,
                                  portfolio_exposure: Dict) -> Dict:
        """
        Assess impact of geopolitical event on portfolio
        
        Args:
            event_data: Information about geopolitical event
            portfolio_exposure: Portfolio exposures by region, industry, etc.
            
        Returns:
            Impact assessment
        """
        prompt = f"""Assess the impact of this geopolitical event on the investment portfolio.

Event:
- Type: {event_data.get('event_type')}
- Location: {event_data.get('location')}
- Description: {event_data.get('description')}
- Severity: {event_data.get('severity')}
- Timeline: {event_data.get('timeline')}

Portfolio Exposure:
- Geographic: {json.dumps(portfolio_exposure.get('geographic', {}))}
- Industry: {json.dumps(portfolio_exposure.get('industry', {}))}
- Key Holdings: {json.dumps(portfolio_exposure.get('top_holdings', []))}
- Currency Exposure: {json.dumps(portfolio_exposure.get('currencies', {}))}

Provide impact assessment as JSON:
{{
  "overall_impact": "High/Medium/Low",
  "timeframe": "Immediate/Short-term/Long-term",
  "affected_holdings": [
    {{
      "ticker": "Stock ticker",
      "company": "Company name",
      "exposure_type": "How exposed (operations, supply chain, demand)",
      "impact_severity": "High/Medium/Low",
      "estimated_impact": "Potential % impact on price",
      "reason": "Why this company is affected"
    }}
  ],
  "affected_sectors": [
    {{
      "sector": "Sector name",
      "impact": "Positive/Negative/Neutral",
      "magnitude": "Impact magnitude",
      "reasoning": "Why sector is affected"
    }}
  ],
  "currency_impact": {{
    "affected_currencies": ["List of currencies"],
    "direction": "Strengthen/Weaken",
    "magnitude": "Expected movement",
    "portfolio_effect": "How it affects portfolio"
  }},
  "supply_chain_disruptions": ["Potential disruptions"],
  "regulatory_implications": ["Regulatory changes or risks"],
  "scenario_analysis": [
    {{
      "scenario": "Description",
      "probability": "High/Medium/Low",
      "portfolio_impact": "Estimated impact",
      "timeline": "When it might occur"
    }}
  ],
  "hedging_recommendations": ["Actions to hedge risk"],
  "position_adjustments": ["Suggested portfolio changes"],
  "monitoring_priorities": ["What to monitor closely"],
  "overall_recommendation": "Detailed recommendation"
}}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=3500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        assessment = self._parse_json(response.content[0].text)
        assessment['event'] = event_data.get('description')
        assessment['assessment_date'] = datetime.now().isoformat()
        
        return assessment
    
    def monitor_risk_indicators(self, news_feed: List[Dict],
                               current_risks: List[Dict]) -> Dict:
        """
        Monitor news for emerging geopolitical risks
        
        Args:
            news_feed: Recent news articles
            current_risks: Currently tracked risks
            
        Returns:
            Updated risk assessment
        """
        # Summarize recent news
        news_summary = "\\n".join([
            f"- {article['title']} ({article['source']})"
            for article in news_feed[:20]
        ])
        
        current_risks_summary = json.dumps([
            {
                'risk': r['name'],
                'level': r['level'],
                'status': r['status']
            }
            for r in current_risks
        ], indent=2)
        
        prompt = f"""Monitor geopolitical risk landscape based on recent news.

Recent News (Last 24 hours):
{news_summary}

Currently Tracked Risks:
{current_risks_summary}

Analyze:
1. Are existing risks escalating or de-escalating?
2. Any new emerging risks?
3. Any false alarms or risks to downgrade?
4. What requires immediate attention?

Return JSON:
{{
  "risk_changes": [
    {{
      "risk_name": "Name of risk",
      "change": "Escalating/De-escalating/New/Resolved",
      "previous_level": "Previous risk level",
      "current_level": "Current risk level",
      "key_developments": ["What changed"],
      "urgency": "High/Medium/Low"
    }}
  ],
  "new_risks": [
    {{
      "name": "Risk name",
      "type": "Type of risk",
      "affected_regions": ["Regions"],
      "affected_industries": ["Industries"],
      "severity": "High/Medium/Low",
      "likelihood": "High/Medium/Low",
      "timeframe": "When it might materialize",
      "description": "What the risk is"
    }}
  ],
  "attention_required": ["Risks needing immediate action"],
  "recommendations": ["Overall recommendations"],
  "monitoring_focus": ["What to watch closely"]
}}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return self._parse_json(response.content[0].text)
    
    def _parse_json(self, response_text: str) -> Dict:
        """Parse JSON from response"""
        import json
        try:
            if "```json" in response_text:
json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
json_str = response_text
return json.loads(json_str)
except:
return {}

# Example usage
geo_monitor = GeopoliticalRiskMonitor(api_key = "your-key")

event_data = {
    'event_type': 'Trade Dispute',
    'location': 'US-China',
    'description': 'New tariffs announced on technology imports',
    'severity': 'High',
    'timeline': 'Effective in 30 days'
}

portfolio_exposure = {
    'geographic': {
        'US': 60,
        'China': 15,
        'Europe': 20,
        'Other': 5
    },
    'industry': {
        'Technology': 35,
        'Consumer': 25,
        'Financials': 20,
        'Healthcare': 15,
        'Other': 5
    },
    'top_holdings': [
        { 'ticker': 'AAPL', 'weight': 8.5, 'china_exposure': 'High' },
        { 'ticker': 'NVDA', 'weight': 6.2, 'china_exposure': 'Medium' }
    ],
    'currencies': {
        'USD': 75,
        'CNY': 10,
        'EUR': 15
    }
}

impact = geo_monitor.assess_geopolitical_event(event_data, portfolio_exposure)
print("Geopolitical Impact Assessment:")
print(json.dumps(impact, indent = 2))
\`\`\`

---

## Early Warning System

### Building Comprehensive Risk Alert System

\`\`\`python
"""
Early warning system for multiple risk types
"""

import threading
import queue
from datetime import datetime
import time

class RiskEarlyWarningSystem:
    """
    Comprehensive early warning system for financial risks
    """
    
    def __init__(self, api_key: str):
        self.credit_analyzer = CreditRiskAnalyzer(api_key)
        self.counterparty_assessor = CounterpartyRiskAssessor(api_key)
        self.geo_monitor = GeopoliticalRiskMonitor(api_key)
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
        
        # Risk tracking
        self.active_risks = {}
        self.risk_history = []
        
        # Alert queue
        self.alert_queue = queue.Queue()
        
        self.running = False
    
    def start(self):
        """Start the early warning system"""
        self.running = True
        
        # Start monitoring threads
        threads = [
            threading.Thread(target=self._credit_monitor, daemon=True),
            threading.Thread(target=self._news_monitor, daemon=True),
            threading.Thread(target=self._alert_processor, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        print("Risk Early Warning System started")
    
    def add_watchlist_company(self, ticker: str, company_data: Dict):
        """Add company to watchlist"""
        self.active_risks[ticker] = {
            'data': company_data,
            'last_assessment': None,
            'risk_level': 'Unknown',
            'alerts': []
        }
    
    def _credit_monitor(self):
        """Monitor credit risk for watchlist companies"""
        while self.running:
            for ticker, risk_data in self.active_risks.items():
                try:
                    # Assess credit risk
                    assessment = self.credit_analyzer.analyze_credit_risk(
                        risk_data['data']
                    )
                    
                    # Check for deterioration
                    if risk_data['last_assessment']:
                        self._check_risk_change(ticker, 
                                               risk_data['last_assessment'],
                                               assessment)
                    
                    # Update
                    risk_data['last_assessment'] = assessment
                    risk_data['risk_level'] = assessment.get('risk_level')
                    
                except Exception as e:
                    print(f"Error monitoring {ticker}: {e}")
            
            time.sleep(3600)  # Check hourly
    
    def _news_monitor(self):
        """Monitor news for risk signals"""
        while self.running:
            # Fetch news for watchlist companies
            # Check for risk-related news
            # Generate alerts if needed
            time.sleep(300)  # Check every 5 minutes
    
    def _check_risk_change(self, ticker: str, previous: Dict, current: Dict):
        """Check if risk level has changed"""
        prev_level = previous.get('risk_level')
        curr_level = current.get('risk_level')
        
        if prev_level != curr_level:
            self._generate_risk_alert(ticker, prev_level, curr_level, current)
    
    def _generate_risk_alert(self, ticker: str, prev_level: str,
                            curr_level: str, assessment: Dict):
        """Generate risk alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'alert_type': 'RISK_CHANGE',
            'previous_level': prev_level,
            'current_level': curr_level,
            'severity': self._calculate_severity(prev_level, curr_level),
            'details': assessment,
            'action_required': self._determine_action(curr_level)
        }
        
        self.alert_queue.put(alert)
    
    def _calculate_severity(self, prev_level: str, curr_level: str) -> str:
        """Calculate alert severity"""
        risk_levels = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
        
        prev_score = risk_levels.get(prev_level, 0)
        curr_score = risk_levels.get(curr_level, 0)
        
        diff = curr_score - prev_score
        
        if diff >= 2:
            return 'CRITICAL'
        elif diff == 1:
            return 'HIGH'
        elif diff < 0:
            return 'INFO'
        else:
            return 'MEDIUM'
    
    def _determine_action(self, risk_level: str) -> str:
        """Determine required action based on risk level"""
        actions = {
            'Very High': 'IMMEDIATE ACTION - Consider exit',
            'High': 'URGENT - Reduce exposure',
            'Medium': 'MONITOR CLOSELY - Review position',
            'Low': 'MAINTAIN - Continue monitoring'
        }
        return actions.get(risk_level, 'REVIEW')
    
    def _alert_processor(self):
        """Process and distribute alerts"""
        while self.running:
            try:
                alert = self.alert_queue.get(timeout=1)
                
                # Process alert
                self._emit_alert(alert)
                
                # Store in history
                self.risk_history.append(alert)
                
                self.alert_queue.task_done()
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing alert: {e}")
    
    def _emit_alert(self, alert: Dict):
        """Emit alert to various channels"""
        print(f"\\n{'='*70}")
        print(f"RISK ALERT - {alert['severity']}")
        print(f"{'='*70}")
        print(f"Ticker: {alert['ticker']}")
        print(f"Alert Type: {alert['alert_type']}")
        print(f"Risk Change: {alert['previous_level']} → {alert['current_level']}")
        print(f"Action Required: {alert['action_required']}")
        print(f"Time: {alert['timestamp']}")
        print(f"{'='*70}\\n")
        
        # In production:
        # - Send email alerts
        # - Push notifications
        # - Update dashboard
        # - Log to database
        # - Trigger automated actions if configured
    
    def generate_risk_report(self) -> str:
        """Generate comprehensive risk report"""
        prompt = f"""Generate a comprehensive risk report based on current risk monitoring.

Active Risks:
{json.dumps([
    {
        'ticker': ticker,
        'risk_level': data['risk_level'],
        'key_concerns': data['last_assessment'].get('key_weaknesses', [])[:3] if data['last_assessment'] else []
    }
    for ticker, data in self.active_risks.items()
], indent=2)}

Recent Alerts (Last 7 days):
{json.dumps([
    {
        'ticker': alert['ticker'],
        'severity': alert['severity'],
        'change': f"{alert['previous_level']} → {alert['current_level']}"
    }
    for alert in self.risk_history[-10:]
], indent=2)}

Generate risk report covering:
1. Executive Summary
2. High-Priority Risks
3. Emerging Concerns
4. Risk Trends
5. Recommended Actions
6. Monitoring Focus

Format as professional risk report."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text

# Example usage
ews = RiskEarlyWarningSystem(api_key="your-key")
ews.start()

# Add companies to monitor
ews.add_watchlist_company('XYZ', {
    'name': 'XYZ Corp',
    'industry': 'Retail',
    'financials': {
        'revenue': 500,
        'ebitda': 50,
        'total_debt': 400,
        'cash': 30,
        # ... more data
    }
})

# System runs continuously
# time.sleep(10)
# report = ews.generate_risk_report()
# print(report)
\`\`\`

---

## Best Practices

1. **Multi-Dimensional Analysis**: Assess financial, operational, and qualitative risks
2. **Early Detection**: Monitor leading indicators, not just lagging metrics
3. **Continuous Monitoring**: Risk is dynamic; assess regularly
4. **Scenario Analysis**: Consider multiple risk scenarios
5. **Mitigation Planning**: Have action plans for identified risks
6. **Human Oversight**: Critical risk decisions need human judgment
7. **Documentation**: Maintain audit trail of risk assessments
8. **Validation**: Validate LLM risk assessments with quantitative models
9. **Alert Fatigue**: Calibrate alert thresholds to avoid false alarms
10. **Integration**: Integrate risk assessment into investment process

---

## Summary

We covered:
- Credit risk analysis from financial documents
- Counterparty and supply chain risk assessment
- Geopolitical risk monitoring and impact assessment
- Building comprehensive early warning systems
- Best practices for LLM-powered risk management

Next: Market research automation with LLMs.
`,
};

