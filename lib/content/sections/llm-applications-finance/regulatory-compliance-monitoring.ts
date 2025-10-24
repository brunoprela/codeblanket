export const regulatoryComplianceMonitoring = {
  title: 'Regulatory Compliance & Monitoring',
  id: 'regulatory-compliance-monitoring',
  content: `
# Regulatory Compliance & Monitoring

## Introduction

Financial services are heavily regulated. Compliance requires monitoring trades, communications, disclosures, and ensuring adherence to complex rules (FINRA, SEC, MiFID II, etc.). Manual compliance review is expensive, time-consuming, and error-prone. LLMs can automate much of this process while maintaining accuracy and providing explainability.

This section covers using LLMs for regulatory compliance: trade surveillance, communication monitoring, regulatory change tracking, automated compliance reporting, and building comprehensive compliance monitoring systems.

### Why LLMs for Compliance

**Automation**: Process vast amounts of data automatically
**Accuracy**: Consistent application of rules
**Explainability**: Provide reasoning for flagged items
**Adaptability**: Quickly update for new regulations
**Cost Efficiency**: Reduce manual review burden

---

## Trade Surveillance

### Automated Market Abuse Detection

\`\`\`python
"""
LLM-powered trade surveillance system
"""

import anthropic
from typing import Dict, List
import json
from datetime import datetime

class TradeSurveillanceSystem:
    """
    Monitor trading activity for suspicious patterns
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def analyze_trade_pattern(self, trades: List[Dict],
                             context: Dict) -> Dict:
        """
        Analyze trading patterns for suspicious activity
        
        Args:
            trades: List of trades to analyze
            context: Market and account context
            
        Returns:
            Surveillance report
        """
        trades_summary = json.dumps([
            {
                'time': t['timestamp'],
                'ticker': t['ticker'],
                'action': t['action'],
                'quantity': t['quantity'],
                'price': t['price']
            }
            for t in trades
        ], indent=2)
        
        prompt = f"""Analyze these trades for potential regulatory violations.

Trades:
{trades_summary}

Context:
- Account Type: \${context.get('account_type')}
- Account Size: \\$\${context.get('account_size'):,.2f}
- Typical Trade Size: \\$\${context.get('typical_size'):,.2f}
- Trading History: \${context.get('history', 'Normal')}

Check for:
    1. ** Wash Sales **: Selling and repurchasing same security within 30 days
2. ** Front Running **: Trading ahead of large client orders
3. ** Insider Trading Patterns **: Unusual timing before material events
4. ** Market Manipulation **: Pump and dump, spoofing, layering
5. ** Excessive Trading **: Churning for commissions
6. ** Position Limits **: Exceeding regulatory position limits
7. ** Marking the Close **: Trades near market close to influence price

Provide analysis as JSON:
{
    {
        "risk_level": "High/Medium/Low/None",
            "flags": [
                {{
                    "violation_type": "Type of potential violation",
                    "severity": "High/Medium/Low",
                    "affected_trades": ["Trade IDs or descriptions"],
                    "explanation": "Why this is flagged",
                    "evidence": "Specific evidence",
                    "false_positive_likelihood": "High/Medium/Low",
                    "recommended_action": "What to do"
                }}
  ],
    "patterns_detected": ["List of patterns found"],
        "requires_investigation": true / false,
            "escalation_needed": true / false,
                "summary": "Brief summary of findings"
}}"""

response = self.client.messages.create(
    model = self.model,
    max_tokens = 2500,
    messages = [{ "role": "user", "content": prompt }]
)

report = self._parse_json(response.content[0].text)
report['analysis_date'] = datetime.now().isoformat()

return report
    
    def detect_insider_trading_risk(self, trades: List[Dict],
    material_events: List[Dict],
    account_info: Dict) -> Dict:
"""
        Detect potential insider trading

Args:
trades: Recent trades
material_events: Material corporate events
account_info: Information about account holder

Returns:
            Insider trading risk assessment
"""
events_summary = json.dumps([
    {
        'date': e['date'],
        'company': e['company'],
        'event': e['event_type'],
        'public_date': e['announcement_date']
    }
            for e in material_events
        ], indent = 2)

prompt = f"""Assess insider trading risk based on trading activity and corporate events.

Account Information:
- Account Holder: { account_info.get('name', 'Anonymous') }
- Relationship to Companies: { account_info.get('relationships', 'Unknown') }
- Access to Material Information: { account_info.get('access_level', 'Unknown') }

Recent Trades:
{ json.dumps([{ 'date': t['timestamp'], 'ticker': t['ticker'], 'action': t['action'] } for t in trades], indent = 2) }

Material Corporate Events:
{ events_summary }

Analyze:
1. Were trades made before material event announcements ?
    2. Timing suspicious(concentrated activity before events) ?
        3. Unusual size or aggressiveness of trades ?
            4. Pattern of trading ahead of news ?
                5. Any legitimate explanations ?

                    Return JSON assessment with risk level and explanation."""

response = self.client.messages.create(
    model = self.model,
    max_tokens = 2000,
    messages = [{ "role": "user", "content": prompt }]
)

return self._parse_json(response.content[0].text)
    
    def check_best_execution(self, order: Dict,
    execution: Dict,
    market_data: Dict) -> Dict:
"""
        Verify best execution requirements

Args:
order: Original order details
execution: How order was executed
market_data: Market conditions at time

Returns:
            Best execution assessment
"""
prompt = f"""Assess whether this order achieved best execution.

Order:
- Type: { order.get('order_type') }
- Quantity: { order.get('quantity') }
- Limit Price: { order.get('limit_price', 'N/A') }
- Time: { order.get('timestamp') }

Execution:
- Execution Price: { execution.get('price') }
- Venue: { execution.get('venue') }
- Time to Fill: { execution.get('time_to_fill') } seconds
    - Slippage: { execution.get('slippage') }%

        Market Data:
- Bid - Ask Spread: { market_data.get('spread') }
- Volume: { market_data.get('volume') }
- Alternative Venue Prices: { market_data.get('other_venues', 'Not available') }

Assess:
1. Was execution price reasonable vs market ?
    2. Was venue selection appropriate ?
        3. Could execution have been better at another venue ?
            4. Was time to fill acceptable ?
                5. Does this meet best execution obligations ?

                    Return assessment as JSON."""

response = self.client.messages.create(
    model = self.model,
    max_tokens = 1500,
    messages = [{ "role": "user", "content": prompt }]
)

return self._parse_json(response.content[0].text)
    
    def _parse_json(self, response_text: str) -> Dict:
"""Parse JSON from response"""
try:
if "\`\`\`json" in response_text:
    json_str = response_text.split("\`\`\`json")[1].split("\`\`\`")[0].strip()
            elif "\`\`\`" in response_text:
json_str = response_text.split("\`\`\`")[1].split("\`\`\`")[0].strip()
            else:
json_str = response_text
return json.loads(json_str)
except:
return {}

# Example usage
surveillance = TradeSurveillanceSystem(api_key = "your-key")

trades = [
    {
        'timestamp': '2024-01-15 10:30:00',
        'ticker': 'AAPL',
        'action': 'SELL',
        'quantity': 100,
        'price': 180.50
    },
    {
        'timestamp': '2024-01-16 14:45:00',
        'ticker': 'AAPL',
        'action': 'BUY',
        'quantity': 100,
        'price': 179.00
    }
]

context = {
    'account_type': 'Individual',
    'account_size': 250000,
    'typical_size': 5000,
    'history': 'Active trader, usually holds 2-4 weeks'
}

report = surveillance.analyze_trade_pattern(trades, context)
print("Surveillance Report:")
print(json.dumps(report, indent = 2))
\`\`\`

---

## Communication Monitoring

### Monitoring Electronic Communications

\`\`\`python
"""
Monitor communications for compliance violations
"""

class CommunicationMonitor:
    """
    Monitor electronic communications for compliance
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def analyze_communication(self, message: str,
                             metadata: Dict) -> Dict:
        """
        Analyze communication for compliance issues
        
        Args:
            message: Communication content
            metadata: Sender, recipient, context
            
        Returns:
            Compliance analysis
        """
        prompt = f"""Analyze this communication for regulatory compliance issues.

Message:
{message}

Metadata:
- From: {metadata.get('sender')} ({metadata.get('sender_role', 'Unknown')})
- To: {metadata.get('recipient')} ({metadata.get('recipient_role', 'Unknown')})
- Date: {metadata.get('timestamp')}
- Channel: {metadata.get('channel', 'Email')}

Check for:
1. **Material Non-Public Information (MNPI)**: Sharing insider information
2. **Unsuitable Recommendations**: Recommending unsuitable investments
3. **False/Misleading Statements**: Misrepresentations
4. **Promises of Performance**: Guarantees of returns
5. **Personal Trading Info**: Inappropriate sharing of positions
6. **Market Manipulation**: Coordinating trades, spreading rumors
7. **Unapproved Communications**: Using unauthorized channels
8. **Record Keeping**: Communications that should be retained

Provide analysis as JSON:
{{
  "risk_level": "High/Medium/Low/None",
  "violations": [
    {{
      "type": "Violation type",
      "severity": "High/Medium/Low",
      "text_excerpt": "Problematic text",
      "explanation": "Why this violates rules",
      "regulation": "Specific regulation violated",
      "recommended_action": "What to do"
    }}
  ],
  "contains_material_info": true/false,
  "requires_retention": true/false,
  "requires_escalation": true/false,
  "context_needed": "Additional context needed for assessment",
  "summary": "Brief summary"
}}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return self._parse_json(response.content[0].text)
    
    def detect_collusion(self, messages: List[Dict]) -> Dict:
        """
        Detect potential collusion in communications
        
        Args:
            messages: Series of messages between parties
            
        Returns:
            Collusion analysis
        """
        conversation = "\\n".join([
            f"[{msg['timestamp']}] {msg['sender']}: {msg['content']}"
            for msg in messages
        ])
        
        prompt = f"""Analyze this conversation for signs of collusion or coordination.

Conversation:
{conversation}

Look for:
1. Coordination of trading activity
2. Information sharing that could manipulate markets
3. Agreements to fix prices
4. Front-running arrangements
5. Wash sale coordination
6. Pump and dump schemes

Return JSON assessment of collusion risk."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return self._parse_json(response.content[0].text)
    
    def _parse_json(self, response_text: str) -> Dict:
        """Parse JSON from response"""
        import json
        try:
            if "\`\`\`json" in response_text:
json_str = response_text.split("\`\`\`json")[1].split("\`\`\`")[0].strip()
            elif "\`\`\`" in response_text:
json_str = response_text.split("\`\`\`")[1].split("\`\`\`")[0].strip()
            else:
json_str = response_text
return json.loads(json_str)
except:
return {}

# Example usage
comm_monitor = CommunicationMonitor(api_key = "your-key")

message = """
Hey John, I just heard from a friend at XYZ Corp that they're announcing
a major acquisition next week.Might want to load up on some calls before
the announcement.This is confidential so keep it between us!
"""

metadata = {
    'sender': 'Mike Smith',
    'sender_role': 'Trader',
    'recipient': 'John Doe',
    'recipient_role': 'Portfolio Manager',
    'timestamp': '2024-01-15 14:30:00',
    'channel': 'Instant Message'
}

analysis = comm_monitor.analyze_communication(message, metadata)
print("Communication Analysis:")
print(json.dumps(analysis, indent = 2))
\`\`\`

---

## Regulatory Change Tracking

### Monitoring Regulatory Updates

\`\`\`python
"""
Track and interpret regulatory changes
"""

class RegulatoryChangeMonitor:
    """
    Monitor and interpret regulatory changes
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def analyze_regulatory_change(self, change_document: str,
                                  current_policies: Dict) -> Dict:
        """
        Analyze regulatory change and implications
        
        Args:
            change_document: New regulation or amendment text
            current_policies: Current compliance policies
            
        Returns:
            Impact analysis
        """
        prompt = f"""Analyze this regulatory change and assess impact.

Regulatory Change:
{change_document[:5000]}

Current Policies:
{json.dumps(current_policies, indent=2)}

Provide analysis as JSON:
{{
  "summary": "Plain language summary of change",
  "effective_date": "When this takes effect",
  "impact_level": "High/Medium/Low",
  "affected_areas": [
    {{
      "area": "Business area affected",
      "impact": "How it's affected",
      "action_required": "What must be done"
    }}
  ],
  "policy_updates_needed": [
    {{
      "policy": "Policy to update",
      "current_approach": "Current policy",
      "required_change": "What needs to change",
      "urgency": "High/Medium/Low"
    }}
  ],
  "systems_impacted": ["Systems needing updates"],
  "training_required": "What training is needed",
  "estimated_cost": "Estimated compliance cost",
  "implementation_timeline": "Recommended timeline",
  "key_risks": ["Risks of non-compliance"],
  "opportunities": ["Any beneficial aspects"]
}}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return self._parse_json(response.content[0].text)
    
    def generate_compliance_checklist(self, regulation: str) -> List[Dict]:
        """
        Generate compliance checklist for regulation
        
        Args:
            regulation: Regulation to comply with
            
        Returns:
            Compliance checklist
        """
        prompt = f"""Generate a comprehensive compliance checklist for this regulation.

Regulation: {regulation}

Create checklist with:
1. Specific requirements
2. Evidence needed to demonstrate compliance
3. Responsible party
4. Frequency of review
5. Documentation requirements

Return as structured JSON list."""

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
            if "\`\`\`json" in response_text:
json_str = response_text.split("\`\`\`json")[1].split("\`\`\`")[0].strip()
            elif "\`\`\`" in response_text:
json_str = response_text.split("\`\`\`")[1].split("\`\`\`")[0].strip()
            else:
json_str = response_text
return json.loads(json_str)
except:
return {}

# Example usage
reg_monitor = RegulatoryChangeMonitor(api_key = "your-key")

change_doc = """
SEC Amendment to Rule 10b5 - 1: Trading Plans and Affirmative Defense

The Securities and Exchange Commission has amended Rule 10b5 - 1 to address
concerns about potential abuse of trading plans.Key changes include:

1. Mandatory cooling - off periods before trading plans take effect
2. Limitations on overlapping trading plans
3. Enhanced disclosure requirements
4. Certifications from plan participants

Effective Date: April 1, 2024
"""

current_policies = {
    '10b5-1_plans': 'Current policy allows immediate execution of trading plans',
    'disclosure': 'Annual disclosure in proxy statements'
}

analysis = reg_monitor.analyze_regulatory_change(change_doc, current_policies)
print("Regulatory Change Analysis:")
print(json.dumps(analysis, indent = 2))
\`\`\`

---

## Automated Compliance Reporting

### Generating Regulatory Reports

\`\`\`python
"""
Generate automated compliance reports
"""

class ComplianceReporter:
    """
    Generate compliance reports automatically
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def generate_surveillance_report(self, period: str,
                                    surveillance_data: Dict) -> str:
        """
        Generate periodic surveillance report
        
        Args:
            period: Reporting period
            surveillance_data: Surveillance findings
            
        Returns:
            Formatted report
        """
        prompt = f"""Generate a comprehensive surveillance report for {period}.

Surveillance Data:
{json.dumps(surveillance_data, indent=2)}

Generate report (1500-2000 words) including:

# Executive Summary
- Key findings
- High-priority items
- Actions taken

# Surveillance Activities
- Trade surveillance volume
- Communication review volume
- Systems monitored

# Findings
- High risk items identified
- Medium risk items
- Trends observed

# Investigations
- Investigations opened
- Investigations closed
- Current status of open items

# Remediation
- Actions taken
- Policy updates
- Training conducted

# Forward-Looking
- Areas of focus for next period
- Recommended enhancements
- Resource needs

Format as professional regulatory report."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def generate_exception_report(self, exceptions: List[Dict]) -> str:
        """
        Generate report of compliance exceptions
        
        Args:
            exceptions: List of compliance exceptions
            
        Returns:
            Exception report
        """
        exceptions_summary = json.dumps([
            {
                'date': e['date'],
                'type': e['exception_type'],
                'severity': e['severity'],
                'description': e['description'][:100]
            }
            for e in exceptions
        ], indent=2)
        
        prompt = f"""Generate compliance exceptions report.

Exceptions:
{exceptions_summary}

Create report covering:
1. Summary of exceptions by type
2. Root cause analysis
3. Remediation status
4. Systemic issues identified
5. Recommendations

Format as executive summary suitable for board presentation."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def generate_regulatory_filing(self, filing_type: str,
                                  data: Dict) -> str:
        """
        Generate regulatory filing document
        
        Args:
            filing_type: Type of filing (Form ADV, etc.)
            data: Data for filing
            
        Returns:
            Formatted filing
        """
        prompt = f"""Generate regulatory filing: {filing_type}

Data:
{json.dumps(data, indent=2)}

Generate complete filing with:
1. All required sections
2. Proper formatting
3. Required disclosures
4. Accurate data presentation
5. Compliance with filing requirements

Return formatted filing document."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text

# Example usage
reporter = ComplianceReporter(api_key="your-key")

surveillance_data = {
    'period': 'Q1 2024',
    'trades_reviewed': 15420,
    'communications_reviewed': 8930,
    'high_risk_items': 12,
    'investigations_opened': 3,
    'investigations_closed': 5,
    'training_sessions': 8
}

report = reporter.generate_surveillance_report('Q1 2024', surveillance_data)
print("Surveillance Report:")
print(report)
\`\`\`

---

## Complete Compliance System

### Integrated Compliance Platform

\`\`\`python
"""
Complete integrated compliance monitoring system
"""

import schedule
from datetime import datetime

class ComplianceMonitoringPlatform:
    """
    Complete compliance monitoring platform
    """
    
    def __init__(self, api_key: str):
        self.surveillance = TradeSurveillanceSystem(api_key)
        self.comm_monitor = CommunicationMonitor(api_key)
        self.reg_monitor = RegulatoryChangeMonitor(api_key)
        self.reporter = ComplianceReporter(api_key)
        
        # Storage
        self.alerts = []
        self.investigations = []
    
    def monitor_continuous(self):
        """
        Run continuous compliance monitoring
        """
        # Schedule monitoring tasks
        schedule.every(5).minutes.do(self._check_trades)
        schedule.every(5).minutes.do(self._check_communications)
        schedule.every().day.at("09:00").do(self._check_regulatory_updates)
        schedule.every().monday.at("08:00").do(self._generate_weekly_report)
        
        print("Compliance monitoring system started")
        
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    def _check_trades(self):
        """Monitor recent trades"""
        # Fetch recent trades
        # Analyze with surveillance system
        # Flag suspicious activity
        pass
    
    def _check_communications(self):
        """Monitor communications"""
        # Fetch recent communications
        # Analyze with comm monitor
        # Flag violations
        pass
    
    def _check_regulatory_updates(self):
        """Check for regulatory changes"""
        # Check regulatory websites
        # Analyze changes
        # Generate action items
        pass
    
    def _generate_weekly_report(self):
        """Generate weekly compliance report"""
        # Aggregate weekly data
        # Generate report
        # Distribute to stakeholders
        pass
    
    def investigate_alert(self, alert_id: str) -> Dict:
        """
        Conduct investigation of alert
        
        Args:
            alert_id: Alert to investigate
            
        Returns:
            Investigation report
        """
        # Gather all related information
        # Analyze with LLM
        # Generate investigation report
        pass

# Initialize platform
# platform = ComplianceMonitoringPlatform(api_key="your-key")
# platform.monitor_continuous()
\`\`\`

---

## Best Practices

1. **Human Review**: Always have compliance officers review flagged items
2. **Audit Trail**: Maintain complete records of all compliance activities
3. **Regular Testing**: Regularly test surveillance systems
4. **False Positive Management**: Tune systems to reduce false positives
5. **Timely Escalation**: Escalate high-risk items immediately
6. **Training**: Regular compliance training for all staff
7. **Documentation**: Document all compliance decisions
8. **System Monitoring**: Monitor compliance systems themselves
9. **Regulatory Updates**: Stay current with regulatory changes
10. **Third-Party Review**: Periodic external compliance audits

### Compliance Disclaimer

These LLM systems are tools to assist compliance programs, not replacements for qualified compliance professionals. Always ensure:
- Human oversight of all compliance decisions
- Regular validation of LLM outputs
- Proper documentation
- Qualified legal review when needed
- Compliance with all applicable regulations

---

## Summary

We covered:
- Trade surveillance and market abuse detection
- Communication monitoring for compliance
- Regulatory change tracking and interpretation
- Automated compliance reporting
- Building integrated compliance monitoring systems
- Best practices and limitations

This completes Module 18: LLM Applications in Finance. We've covered the complete spectrum of applying LLMs to financial analysis, trading, risk management, research, and compliance.
`,
};
