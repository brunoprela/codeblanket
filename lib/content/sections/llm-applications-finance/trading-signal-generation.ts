export const tradingSignalGeneration = {
  title: 'Trading Signal Generation',
  id: 'trading-signal-generation',
  content: `
# Trading Signal Generation

## Introduction

The ultimate goal of financial analysis is actionable trading signals-specific buy, sell, or hold recommendations with confidence levels and reasoning. LLMs can synthesize multiple data sources (fundamental analysis, technical indicators, news sentiment, market context) to generate sophisticated trading signals that combine quantitative and qualitative analysis.

This section covers building LLM-powered trading signal generation systems that combine multiple analysis streams, provide explainable recommendations, assess confidence levels, and integrate with automated trading systems.

### Why LLM-Generated Trading Signals

**Multi-Modal Analysis**: Combine quantitative data with qualitative insights
**Contextual Understanding**: Consider market regime, news flow, and fundamentals
**Explainability**: Provide clear reasoning for each signal
**Adaptability**: Adjust to changing market conditions
**Scale**: Generate signals for thousands of securities

---

## Signal Generation Framework

### LLM-Powered Signal System Architecture

\`\`\`python
"""
Complete trading signal generation system using LLMs
"""

import anthropic
from typing import Dict, List, Optional
from datetime import datetime
import json

class TradingSignalGenerator:
    """
    Generate trading signals using LLM analysis
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def generate_signal(self, ticker: str, analysis_data: Dict) -> Dict:
        """
        Generate trading signal from comprehensive analysis
        
        Args:
            ticker: Stock ticker
            analysis_data: Dict containing all analysis streams
            
        Returns:
            Trading signal with reasoning
        """
        prompt = f"""Generate a trading signal for {ticker} based on comprehensive analysis.

Technical Analysis:
- Current Price: \${analysis_data['technical']['price']}
- RSI: {analysis_data['technical']['rsi']}
- MACD: {analysis_data['technical']['macd']}
- Moving Averages: {analysis_data['technical']['ma_analysis']}
- Support/Resistance: {analysis_data['technical']['levels']}
- Trend: {analysis_data['technical']['trend']}

Fundamental Analysis:
- P/E Ratio: {analysis_data['fundamental']['pe']}
- Revenue Growth: {analysis_data['fundamental']['revenue_growth']}%
- Profit Margin: {analysis_data['fundamental']['profit_margin']}%
- Debt/Equity: {analysis_data['fundamental']['debt_equity']}
- Recent Earnings: {analysis_data['fundamental']['earnings_surprise']}

Sentiment Analysis:
- News Sentiment: {analysis_data['sentiment']['news_score']}
- Social Media Sentiment: {analysis_data['sentiment']['social_score']}
- Analyst Ratings: {analysis_data['sentiment']['analyst_consensus']}

Market Context:
- Market Regime: {analysis_data['market']['regime']}
- Sector Performance: {analysis_data['market']['sector_performance']}
- VIX Level: {analysis_data['market']['vix']}

Generate JSON response:
{{
  "signal": "STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL",
  "confidence": 0.0-1.0,
  "target_price": "Price target if applicable",
  "stop_loss": "Recommended stop loss",
  "time_horizon": "Short-term/Medium-term/Long-term",
  "position_size": "Suggested position size (Small/Medium/Large)",
  "key_catalysts": ["Reasons supporting the trade"],
  "risks": ["Key risks to the trade"],
  "technical_score": 0.0-1.0,
  "fundamental_score": 0.0-1.0,
  "sentiment_score": 0.0-1.0,
  "overall_reasoning": "Comprehensive explanation of the signal"
}}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        signal = self._parse_json(response.content[0].text)
        signal['ticker'] = ticker
        signal['timestamp'] = datetime.now().isoformat()
        
        return signal
    
    def generate_multi_signal(self, watchlist: List[str],
                             market_data: Dict) -> List[Dict]:
        """
        Generate signals for multiple tickers
        
        Args:
            watchlist: List of tickers to analyze
            market_data: Market context data
            
        Returns:
            List of trading signals
        """
        signals = []
        
        for ticker in watchlist:
            # Fetch analysis data for ticker (simplified)
            analysis_data = self._get_analysis_data(ticker, market_data)
            
            # Generate signal
            signal = self.generate_signal(ticker, analysis_data)
            signals.append(signal)
        
        # Rank signals by confidence
        signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return signals
    
    def generate_portfolio_signals(self, current_holdings: List[Dict],
                                  watchlist: List[str],
                                  portfolio_constraints: Dict) -> Dict:
        """
        Generate portfolio-level trading signals
        
        Args:
            current_holdings: Current portfolio positions
            watchlist: Candidates for new positions
            portfolio_constraints: Risk limits, sector limits, etc.
            
        Returns:
            Portfolio action recommendations
        """
        # Format current holdings
        holdings_summary = json.dumps([
            {
                'ticker': h['ticker'],
                'weight': h['weight'],
                'return': h['return'],
                'cost_basis': h['cost_basis']
            }
            for h in current_holdings
        ])
        
        # Format constraints
        constraints_summary = json.dumps(portfolio_constraints)
        
        prompt = f"""Generate portfolio-level trading recommendations.

Current Holdings:
{holdings_summary}

Portfolio Constraints:
{constraints_summary}

Analysis:
- Total portfolio value: \${sum(h['value'] for h in current_holdings):,.2f}
- Number of holdings: {len(current_holdings)}
- Largest position: {max(current_holdings, key=lambda x: x['weight'])['ticker']} ({max(h['weight'] for h in current_holdings):.1f}%)

Provide recommendations in JSON:
{{
  "actions": [
    {{
      "action": "BUY/SELL/TRIM/ADD",
      "ticker": "Ticker symbol",
      "current_weight": "Current weight or 0 for new",
      "target_weight": "Recommended weight",
      "rationale": "Why this action",
      "priority": "High/Medium/Low",
      "urgency": "Immediate/This Week/Month"
    }}
  ],
  "portfolio_adjustments": {{
    "rebalancing_needed": true/false,
    "risk_assessment": "Current portfolio risk level",
    "diversification_score": 0-10,
    "recommendations": ["Overall portfolio suggestions"]
  }}
}}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return self._parse_json(response.content[0].text)
    
    def validate_signal(self, signal: Dict, risk_params: Dict) -> Dict:
        """
        Validate signal against risk parameters
        
        Args:
            signal: Generated trading signal
            risk_params: Risk management parameters
            
        Returns:
            Validated signal with risk checks
        """
        prompt = f"""Validate this trading signal against risk parameters.

Signal:
{json.dumps(signal, indent=2)}

Risk Parameters:
- Max position size: {risk_params.get('max_position_size')}%
- Max sector exposure: {risk_params.get('max_sector_exposure')}%
- Min confidence threshold: {risk_params.get('min_confidence', 0.6)}
- Stop loss requirement: {risk_params.get('stop_loss_pct')}%
- Current portfolio volatility: {risk_params.get('portfolio_volatility')}%

Validate:
1. Does signal meet minimum confidence?
2. Is position size appropriate for risk?
3. Does it fit within portfolio constraints?
4. Is stop loss reasonable?
5. Any red flags?

Return JSON:
{{
  "approved": true/false,
  "adjusted_position_size": "If adjustment needed",
  "adjusted_stop_loss": "If adjustment needed",
  "warnings": ["Any warnings"],
  "risk_score": 0.0-1.0,
  "recommendation": "Execute as-is / Modify / Reject"
}}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        validation = self._parse_json(response.content[0].text)
        validation['original_signal'] = signal
        
        return validation
    
    def _get_analysis_data(self, ticker: str, market_data: Dict) -> Dict:
        """Get comprehensive analysis data for ticker"""
        # In production, fetch from various data sources
        return {
            'technical': {},
            'fundamental': {},
            'sentiment': {},
            'market': market_data
        }
    
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
signal_generator = TradingSignalGenerator(api_key="your-key")

# Sample analysis data
analysis_data = {
    'technical': {
        'price': 180.50,
        'rsi': 45,
        'macd': 'Bullish crossover',
        'ma_analysis': '50-day MA crossed above 200-day MA (Golden Cross)',
        'levels': 'Support at $175, Resistance at $185',
        'trend': 'Uptrend'
    },
    'fundamental': {
        'pe': 28.5,
        'revenue_growth': 15.2,
        'profit_margin': 24.1,
        'debt_equity': 0.35,
        'earnings_surprise': 'Beat by 5%'
    },
    'sentiment': {
        'news_score': 0.7,
        'social_score': 0.6,
        'analyst_consensus': 'Buy (15 Buy, 5 Hold, 2 Sell)'
    },
    'market': {
        'regime': 'Bull market',
        'sector_performance': 'Technology outperforming',
        'vix': 15.2
    }
}

# Generate signal
signal = signal_generator.generate_signal('AAPL', analysis_data)
print("Generated Signal:")
print(json.dumps(signal, indent=2))

# Validate against risk parameters
risk_params = {
    'max_position_size': 5.0,
    'max_sector_exposure': 30.0,
    'min_confidence': 0.7,
    'stop_loss_pct': 5.0,
    'portfolio_volatility': 15.0
}

validation = signal_generator.validate_signal(signal, risk_params)
print("\\nValidation Result:")
print(json.dumps(validation, indent=2))
\`\`\`

---

## Multi-Factor Signal Integration

### Combining Multiple Signal Sources

\`\`\`python
"""
Integrate multiple signal sources for robust trading decisions
"""

from typing import List, Dict
import numpy as np

class MultiFactorSignalIntegrator:
    """
    Integrate signals from multiple sources
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def integrate_signals(self, ticker: str, signal_sources: Dict) -> Dict:
        """
        Integrate signals from multiple analysis methods
        
        Args:
            ticker: Stock ticker
            signal_sources: Dict of different signal types
            
        Returns:
            Integrated final signal
        """
        prompt = f"""Integrate multiple trading signals for {ticker} into a unified recommendation.

Quantitative Signals:
- ML Model Prediction: {signal_sources['ml_model']}
- Technical Indicators: {signal_sources['technical']}
- Statistical Arbitrage: {signal_sources['stat_arb']}

Fundamental Signals:
- Value Score: {signal_sources['value_score']}
- Growth Score: {signal_sources['growth_score']}
- Quality Score: {signal_sources['quality_score']}

Alternative Data Signals:
- News Sentiment: {signal_sources['news_sentiment']}
- Social Sentiment: {signal_sources['social_sentiment']}
- Satellite Data: {signal_sources.get('alternative_data', 'N/A')}

Analyst Signals:
- Consensus Rating: {signal_sources['analyst_consensus']}
- Price Target Implied Return: {signal_sources['price_target_return']}

Provide integrated signal as JSON:
{{
  "final_signal": "STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL",
  "confidence": 0.0-1.0,
  "signal_agreement": "How much signals agree (High/Medium/Low)",
  "dominant_factors": ["Which factors most influenced decision"],
  "conflicting_signals": ["Any conflicting signals to note"],
  "weight_breakdown": {{
    "quantitative": 0.0-1.0,
    "fundamental": 0.0-1.0,
    "sentiment": 0.0-1.0,
    "analyst": 0.0-1.0
  }},
  "edge_assessment": "Why this signal has edge over market",
  "conviction_level": "High/Medium/Low",
  "recommended_action": "Specific action to take"
}}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        integrated = self._parse_json(response.content[0].text)
        integrated['ticker'] = ticker
        integrated['timestamp'] = datetime.now().isoformat()
        
        return integrated
    
    def detect_signal_conflicts(self, signals: List[Dict]) -> Dict:
        """
        Detect and analyze conflicts between signals
        
        Args:
            signals: List of signals from different sources
            
        Returns:
            Conflict analysis
        """
        # Analyze signal direction conflicts
        signal_directions = [s.get('signal', 'HOLD') for s in signals]
        
        prompt = f"""Analyze conflicts between these trading signals.

Signals:
{json.dumps(signals, indent=2)}

Signal Directions: {signal_directions}

Analyze:
1. How much disagreement exists?
2. Which signals are most reliable historically?
3. Why might signals conflict?
4. How to resolve the conflict?
5. Should we trade despite conflict?

Return JSON analysis with recommended resolution."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
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
integrator = MultiFactorSignalIntegrator(api_key="your-key")

signal_sources = {
    'ml_model': {'signal': 'BUY', 'confidence': 0.75, 'predicted_return': 8.5},
    'technical': {'signal': 'BUY', 'confidence': 0.65, 'score': 7.5},
    'stat_arb': {'signal': 'HOLD', 'confidence': 0.50, 'z_score': 0.5},
    'value_score': 6.5,
    'growth_score': 8.2,
    'quality_score': 7.8,
    'news_sentiment': {'score': 0.7, 'signal': 'POSITIVE'},
    'social_sentiment': {'score': 0.6, 'signal': 'POSITIVE'},
    'analyst_consensus': 'BUY',
    'price_target_return': 12.5
}

integrated_signal = integrator.integrate_signals('NVDA', signal_sources)
print("Integrated Signal:")
print(json.dumps(integrated_signal, indent=2))
\`\`\`

---

## Real-Time Signal Updates

### Dynamic Signal Adjustment

\`\`\`python
"""
Real-time signal updates based on market events
"""

import threading
import queue
from datetime import datetime

class RealTimeSignalUpdater:
    """
    Update trading signals in real-time as new data arrives
    """
    
    def __init__(self, api_key: str):
        self.signal_generator = TradingSignalGenerator(api_key)
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
        
        # Store active signals
        self.active_signals = {}
        
        # Event queue
        self.event_queue = queue.Queue()
        
        self.running = False
    
    def start(self):
        """Start real-time signal updater"""
        self.running = True
        
        # Start event processor thread
        thread = threading.Thread(target=self._event_processor, daemon=True)
        thread.start()
        
        print("Real-time signal updater started")
    
    def add_event(self, event: Dict):
        """
        Add market event to processing queue
        
        Args:
            event: Market event (news, price move, etc.)
        """
        event['timestamp'] = datetime.now()
        self.event_queue.put(event)
    
    def _event_processor(self):
        """Process market events and update signals"""
        while self.running:
            try:
                event = self.event_queue.get(timeout=1)
                
                # Determine which signals are affected
                affected_tickers = self._get_affected_tickers(event)
                
                for ticker in affected_tickers:
                    if ticker in self.active_signals:
                        self._update_signal(ticker, event)
                
                self.event_queue.task_done()
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing event: {e}")
    
    def _get_affected_tickers(self, event: Dict) -> List[str]:
        """Determine which tickers are affected by event"""
        # Extract tickers from event
        return event.get('tickers', [])
    
    def _update_signal(self, ticker: str, event: Dict):
        """
        Update existing signal based on new event
        
        Args:
            ticker: Stock ticker
            event: Market event
        """
        current_signal = self.active_signals[ticker]
        
        prompt = f"""Update this trading signal based on new market event.

Current Signal:
{json.dumps(current_signal, indent=2)}

New Event:
- Type: {event.get('type')}
- Description: {event.get('description')}
- Impact: {event.get('impact')}
- Time: {event.get('timestamp')}

Should the signal change? Return JSON:
{{
  "signal_change": "UPGRADE/DOWNGRADE/MAINTAIN",
  "new_signal": "New signal if changed",
  "new_confidence": "New confidence if changed",
  "new_stop_loss": "New stop loss if needed",
  "reasoning": "Why signal changed or stayed same",
  "urgency": "How urgently to act (Immediate/Soon/Normal)",
  "event_impact_score": 0.0-1.0
}}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        update = self._parse_json(response.content[0].text)
        
        # Apply update if signal changed
        if update.get('signal_change') != 'MAINTAIN':
            self._apply_signal_update(ticker, update)
            self._emit_signal_update(ticker, current_signal, update)
    
    def _apply_signal_update(self, ticker: str, update: Dict):
        """Apply signal update"""
        signal = self.active_signals[ticker]
        
        if 'new_signal' in update:
            signal['signal'] = update['new_signal']
        if 'new_confidence' in update:
            signal['confidence'] = update['new_confidence']
        if 'new_stop_loss' in update:
            signal['stop_loss'] = update['new_stop_loss']
        
        signal['last_updated'] = datetime.now().isoformat()
        signal['update_reason'] = update.get('reasoning')
    
    def _emit_signal_update(self, ticker: str, old_signal: Dict, update: Dict):
        """Emit signal update notification"""
        print(f"\\n{'='*60}")
        print(f"SIGNAL UPDATE: {ticker}")
        print(f"Change: {update['signal_change']}")
        print(f"Old: {old_signal.get('signal')} (confidence: {old_signal.get('confidence')})")
        print(f"New: {update.get('new_signal')} (confidence: {update.get('new_confidence')})")
        print(f"Reason: {update.get('reasoning')}")
        print(f"Urgency: {update.get('urgency')}")
        print(f"{'='*60}\\n")
        
        # In production:
        # - Send to trading system
        # - Update dashboard
        # - Send alerts
    
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
updater = RealTimeSignalUpdater(api_key="your-key")
updater.start()

# Add initial signal
updater.active_signals['AAPL'] = {
    'signal': 'BUY',
    'confidence': 0.75,
    'stop_loss': 175.00
}

# Simulate market event
event = {
    'type': 'NEWS',
    'tickers': ['AAPL'],
    'description': 'Apple announces new product delay',
    'impact': 'Negative',
    'sentiment': -0.6
}

updater.add_event(event)

import time
time.sleep(5)  # Wait for processing
\`\`\`

---

## Best Practices

1. **Multi-Source Validation**: Never rely on single signal source
2. **Confidence Calibration**: Regularly validate confidence scores against outcomes
3. **Risk-Adjusted Sizing**: Scale position size with confidence level
4. **Real-Time Updates**: Adjust signals as new information arrives
5. **Explainability**: Always provide clear reasoning for signals
6. **Backtesting**: Validate signal generation logic historically
7. **Human Oversight**: Critical trades need human approval
8. **Stop Losses**: Always include risk management
9. **Market Regime**: Adjust signal thresholds by market conditions
10. **Performance Tracking**: Monitor signal accuracy over time

---

## Summary

We covered:
- LLM-powered trading signal generation
- Multi-factor signal integration
- Portfolio-level signal recommendations
- Signal validation and risk checks
- Real-time signal updates
- Best practices for signal generation

Next: Risk assessment with LLMs for portfolio management.
`,
};

