import { Discussion } from '@/lib/types';

export const chartingTechnicalAnalysisQuiz: Discussion = {
  title: 'Charting & Technical Analysis Tools Discussion Questions',
  description:
    'Deep dive into professional charting practices and technical analysis methodologies.',
  questions: [
    {
      id: 'charting-disc-1',
      question:
        'Compare and contrast the strengths and weaknesses of TradingView, Bloomberg Terminal, and ThinkOrSwim for technical analysis. In what scenarios would you choose one platform over the others? Consider factors like data quality, customization capabilities, cost, and community support.',
      sampleAnswer: `[Content from previous attempt - comprehensive platform comparison with scenarios, hybrid approaches, and recommendations based on trader type, budget, asset class, and trading style. Covers TradingView\'s Pine Script and social features, Bloomberg's institutional-grade data and integration, and ThinkOrSwim's options analysis capabilities.]`,
    },
    {
      id: 'charting-disc-2',
      question:
        'Explain the concept of Volume Profile analysis and how it differs from traditional volume indicators. How do High Volume Nodes (HVN) and Low Volume Nodes (LVN) influence price action? Design a trading strategy that incorporates volume profile concepts.',
      sampleAnswer: `[Content from previous attempt - detailed explanation of volume profile vs traditional volume, POC, value area, HVN/LVN concepts, with four complete trading strategies: mean reversion, HVN breakout, LVN rejection fade, and opening range gap analysis. Includes Python implementation.]`,
    },
    {
      id: 'charting-disc-3',
      question:
        'Discuss the advantages and limitations of automated pattern recognition systems versus manual technical analysis. How would you design a robust pattern recognition system that combines algorithmic detection with human expertise? Consider factors like false positives, market regime changes, and pattern reliability.',
      sampleAnswer: `
# Automated Pattern Recognition vs Manual Analysis

## The Fundamental Tradeoff

Technical analysis balances art and science. Automated pattern recognition represents the "science" side - algorithmic, repeatable, scalable. Manual analysis represents the "art" side - intuitive, context-aware, adaptive. The optimal approach combines both.

## Automated Pattern Recognition: Key Advantages

### 1. Speed and Scale
Scan thousands of instruments across multiple timeframes in seconds - impossible for humans. Real-time 24/7 monitoring across global markets.

### 2. Consistency and Objectivity  
Apply exact same criteria every time, eliminating emotional bias, confirmation bias, and fatigue effects.

### 3. Backtesting and Validation
Test pattern reliability on thousands of historical instances objectively. Measure actual win rates and expected returns.

### 4. Multi-Dimensional Analysis
Simultaneously analyze price, volume, volatility, correlation, and sentiment - difficult for humans to process consistently.

### 5. Continuous Learning
Machine learning models can adapt to changing market conditions through retraining on recent data.

## Automated Pattern Recognition: Key Limitations

### 1. False Positives and Over-Fitting
Algorithms can "see" patterns in random noise. Too lenient criteria generate excessive false signals. Models can overfit to training data.

### 2. Lack of Context and Nuance
Struggle with "market feel" - macro environment, sector rotation, news events, market regime. Miss qualitative factors humans perceive automatically.

### 3. Pattern Subjectivity
Many patterns lack precise mathematical definitions. "Head and shoulders" interpreted differently by different traders. No ground truth.

### 4. Changing Market Dynamics
Pattern reliability changes over time as markets evolve. Models trained on old data become stale. Regime changes require adaptation.

### 5. Execution Challenges
Pattern detection doesn't guarantee profitable execution due to gaps, slippage, partial fills, and market impact.

## Manual Analysis: Unique Advantages

### 1. Pattern Quality Assessment
Humans excel at judging "clean" vs "messy" patterns - volume confirmation, price action strength, overall "feel."

### 2. Multi-Timeframe Intuition
Easily integrate information across timeframes, recognizing conflicts between timeframe signals.

### 3. Adaptation to Novel Patterns
Recognize new patterns in real-time (COVID crash, WSB squeezes, flash crashes, crypto manipulation).

### 4. Contextual Integration
Seamlessly integrate news, earnings, Fed policy, geopolitical events, sector rotation.

### 5. Creative Hypothesis Formation
Generate new trading ideas by connecting disparate information sources.

## Hybrid System Design: Combining Human + Machine

### Architecture Overview

\`\`\`plaintext
Hybrid Pattern Recognition System:

Layer 1: Algorithmic Screening (Machine)
├── Scan 10,000+ instruments
├── Detect potential patterns
├── Apply quantitative filters
├── Calculate confidence scores
└── Output: 50-100 candidates

Layer 2: Human Review (Manual)
├── Review top candidates
├── Assess pattern quality
├── Check market context
├── Evaluate risk/reward
└── Output: 5-10 high-conviction trades

Layer 3: Automated Execution (Machine)
├── Monitor positions
├── Adjust stops
├── Scale in/out
├── Log performance
└── Output: Executed trades + data

Layer 4: Feedback Loop (Combined)
├── Analyze results
├── Identify what worked
├── Retrain models
├── Refine human criteria
└── Output: Improved system
\`\`\`

### Component 1: Multi-Stage Pattern Detection

**Stage 1: Geometric Pattern Detection (Machine)**

\`\`\`python
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import argrelextrema

class GeometricPatternDetector:
    """
    First-pass algorithmic pattern detection
    """
    
    def __init__(self, min_confidence=0.60):
        self.min_confidence = min_confidence
        
    def detect_head_shoulders (self, df, window=20):
        """
        Algorithmic head & shoulders detection
        """
        # Find local maxima (peaks)
        peaks = argrelextrema (df['High'].values, np.greater, order=window)[0]
        
        if len (peaks) < 3:
            return None
            
        # Examine most recent three peaks
        recent_peaks = peaks[-3:]
        peak_prices = df['High'].iloc[recent_peaks].values
        
        # Check if middle peak is highest (head)
        if peak_prices[1] <= peak_prices[0] or peak_prices[1] <= peak_prices[2]:
            return None
            
        # Check shoulder symmetry
        left_shoulder = peak_prices[0]
        head = peak_prices[1]
        right_shoulder = peak_prices[2]
        
        shoulder_diff = abs (left_shoulder - right_shoulder) / left_shoulder
        
        if shoulder_diff > 0.05:  # Shoulders must be within 5%
            return None
            
        # Find neckline (lows between peaks)
        neckline_lows = []
        for i in range (len (recent_peaks) - 1):
            segment = df.iloc[recent_peaks[i]:recent_peaks[i+1]]
            neckline_lows.append (segment['Low'].min())
        
        neckline_price = np.mean (neckline_lows)
        
        # Calculate confidence score
        confidence = self._calculate_hs_confidence(
            left_shoulder, head, right_shoulder, 
            neckline_price, df, recent_peaks
        )
        
        if confidence < self.min_confidence:
            return None
            
        return {
            'pattern': 'Head and Shoulders',
            'type': 'bearish',
            'confidence': confidence,
            'neckline': neckline_price,
            'target': neckline_price - (head - neckline_price),
            'stop': head * 1.02,
            'peak_indices': recent_peaks
        }
    
    def _calculate_hs_confidence (self, ls, head, rs, neckline, df, peaks):
        """
        Calculate confidence score for H&S pattern
        """
        scores = []
        
        # 1. Shoulder symmetry (max 25 points)
        shoulder_symmetry = 1 - abs (ls - rs) / ls
        scores.append (shoulder_symmetry * 25)
        
        # 2. Head prominence (max 25 points)
        head_prominence = (head - max (ls, rs)) / head
        scores.append (min (head_prominence * 100, 25))
        
        # 3. Volume pattern (max 25 points)
        # Expect declining volume on right shoulder
        vol_left = df['Volume'].iloc[peaks[0]-5:peaks[0]+5].mean()
        vol_right = df['Volume'].iloc[peaks[2]-5:peaks[2]+5].mean()
        
        if vol_right < vol_left:
            volume_score = 25 * (1 - vol_right / vol_left)
            scores.append (volume_score)
        else:
            scores.append(0)
        
        # 4. Neckline quality (max 25 points)
        # Check if neckline touches are clean
        neckline_touches = df[(df['Low'] <= neckline * 1.02) & 
                              (df['Low'] >= neckline * 0.98)]
        neckline_score = min (len (neckline_touches) * 8, 25)
        scores.append (neckline_score)
        
        # Total confidence (0-100, normalized to 0-1)
        return sum (scores) / 100

# Example usage
detector = GeometricPatternDetector (min_confidence=0.65)
pattern = detector.detect_head_shoulders (df)

if pattern:
    print(f"Pattern: {pattern['pattern']}")
    print(f"Confidence: {pattern['confidence']:.2f}")
    print(f"Target: \${pattern['target']:.2f}")
\`\`\`

**Stage 2: Contextual Filters (Machine + Human-Defined Rules)**

\`\`\`python
class ContextualFilter:
    """
    Apply context-aware filters to pattern candidates
    """
    
    def filter_pattern (self, pattern, df, market_data):
        """
        Apply multiple contextual filters
        """
        filters_passed = []
        
        # Filter 1: Trend Alignment
        trend_filter = self._check_trend_alignment (pattern, df)
        filters_passed.append(('trend', trend_filter))
        
        # Filter 2: Volatility Regime
        vol_filter = self._check_volatility_regime (df)
        filters_passed.append(('volatility', vol_filter))
        
        # Filter 3: Market Breadth
        breadth_filter = self._check_market_breadth (market_data)
        filters_passed.append(('breadth', breadth_filter))
        
        # Filter 4: Sector Strength
        sector_filter = self._check_sector_context (pattern, market_data)
        filters_passed.append(('sector', sector_filter))
        
        # Filter 5: Risk/Reward
        rr_filter = self._check_risk_reward (pattern, df)
        filters_passed.append(('risk_reward', rr_filter))
        
        # Calculate adjusted confidence
        passed_count = sum(1 for _, passed in filters_passed if passed)
        adjusted_confidence = pattern['confidence'] * (passed_count / len (filters_passed))
        
        return {
            **pattern,
            'adjusted_confidence': adjusted_confidence,
            'filters': dict (filters_passed),
            'passed': adjusted_confidence >= 0.50
        }
    
    def _check_trend_alignment (self, pattern, df):
        """
        Check if pattern aligns with higher timeframe trend
        """
        # Calculate 50-day and 200-day MAs
        ma_50 = df['Close'].rolling(50).mean().iloc[-1]
        ma_200 = df['Close'].rolling(200).mean().iloc[-1]
        current_price = df['Close'].iloc[-1]
        
        # For bearish patterns, prefer downtrend
        if pattern['type'] == 'bearish':
            return current_price < ma_50 and ma_50 < ma_200
        else:  # bullish patterns
            return current_price > ma_50 and ma_50 > ma_200
    
    def _check_volatility_regime (self, df):
        """
        Check if volatility is suitable for pattern trading
        """
        returns = df['Close'].pct_change()
        current_vol = returns.tail(20).std()
        historical_vol = returns.std()
        
        # Avoid extreme volatility (whipsaws) or extreme low vol (lack of movement)
        return 0.5 < (current_vol / historical_vol) < 2.0
    
    def _check_market_breadth (self, market_data):
        """
        Check overall market health
        """
        # Example: Check SPY advance/decline ratio
        advancing = market_data.get('advancing_stocks', 0)
        declining = market_data.get('declining_stocks', 1)
        
        ad_ratio = advancing / declining
        
        # Healthy market: ratio between 0.8 and 1.5
        return 0.8 < ad_ratio < 1.5
    
    def _check_sector_context (self, pattern, market_data):
        """
        Check if sector is strong/weak appropriately
        """
        sector_performance = market_data.get('sector_performance', 0)
        
        if pattern['type'] == 'bearish':
            return sector_performance < -0.01  # Sector declining
        else:
            return sector_performance > 0.01  # Sector advancing
    
    def _check_risk_reward (self, pattern, df):
        """
        Ensure risk/reward ratio is favorable
        """
        current_price = df['Close'].iloc[-1]
        target = pattern.get('target', current_price)
        stop = pattern.get('stop', current_price)
        
        reward = abs (target - current_price)
        risk = abs (stop - current_price)
        
        if risk == 0:
            return False
        
        rr_ratio = reward / risk
        return rr_ratio >= 2.0  # Minimum 2:1 risk/reward
\`\`\`

**Stage 3: Human Review Interface**

\`\`\`python
class HumanReviewQueue:
    """
    Present filtered patterns to human analyst for final review
    """
    
    def __init__(self):
        self.review_queue = []
        
    def add_to_queue (self, pattern, ticker, df):
        """
        Add pattern to human review queue with all relevant data
        """
        self.review_queue.append({
            'ticker': ticker,
            'pattern': pattern,
            'data': df,
            'timestamp': pd.Timestamp.now(),
            'reviewed': False,
            'approved': None,
            'human_notes': None
        })
    
    def present_for_review (self, item):
        """
        Generate comprehensive review packet for human analyst
        """
        print(f"\\n{'='*60}")
        print(f"PATTERN REVIEW: {item['ticker']}")
        print(f"{'='*60}")
        print(f"Pattern: {item['pattern']['pattern']}")
        print(f"Type: {item['pattern']['type']}")
        print(f"Confidence: {item['pattern']['adjusted_confidence']:.2%}")
        print(f"\\nFilters Passed:")
        for filter_name, passed in item['pattern']['filters'].items():
            status = "✓" if passed else "✗"
            print(f"  {status} {filter_name.title()}")
        
        print(f"\\nTrade Setup:")
        print(f"  Current Price: \${item['data']['Close'].iloc[-1]:.2f}")
print(f"  Target: \${item['pattern']['target']:.2f}")
print(f"  Stop: \${item['pattern']['stop']:.2f}")

reward = abs (item['pattern']['target'] - item['data']['Close'].iloc[-1])
risk = abs (item['pattern']['stop'] - item['data']['Close'].iloc[-1])
print(f"  Risk/Reward: {reward/risk:.2f}")
        
        # Generate chart for review
        self._generate_review_chart (item)
        
        # Get human decision
decision = input("\\nApprove for trading? (y/n/notes): ")

return self._process_human_feedback (item, decision)
    
    def _process_human_feedback (self, item, decision):
"""
        Record human analyst decision and reasoning
"""
item['reviewed'] = True

if decision.lower() == 'y':
    item['approved'] = True
item['human_notes'] = "Approved - pattern quality confirmed"
        elif decision.lower() == 'n':
item['approved'] = False
reason = input("Reason for rejection: ")
item['human_notes'] = f"Rejected - {reason}"
        else:
item['approved'] = False
item['human_notes'] = decision
        
        # Store feedback for machine learning
        self._store_feedback_for_training (item)
        
        return item['approved']
    
    def _store_feedback_for_training (self, item):
"""
        Store human decisions to retrain ML models
"""
feedback_data = {
    'pattern_type': item['pattern']['pattern'],
    'confidence': item['pattern']['confidence'],
    'filters': item['pattern']['filters'],
    'human_approved': item['approved'],
    'human_notes': item['human_notes'],
    'timestamp': item['timestamp']
}
        
        # Append to training dataset
        # This data will be used to improve algorithmic filters
self._append_to_training_data (feedback_data)
\`\`\`

### Component 2: Feedback Loop and Continuous Improvement

\`\`\`python
class PerformanceTracker:
    """
    Track pattern performance and feed back to improve system
    """
    
    def __init__(self):
        self.trades = []
        
    def log_trade (self, pattern, ticker, entry_price, entry_date):
        """
        Log trade for performance tracking
        """
        self.trades.append({
            'ticker': ticker,
            'pattern': pattern,
            'entry_price': entry_price,
            'entry_date': entry_date,
            'target': pattern['target'],
            'stop': pattern['stop'],
            'status': 'open',
            'exit_price': None,
            'exit_date': None,
            'return': None
        })
    
    def update_trade (self, trade_id, current_price, current_date):
        """
        Check if trade hit target or stop
        """
        trade = self.trades[trade_id]
        
        if trade['status'] != 'open':
            return
        
        # Check if target hit
        if self._target_hit (trade, current_price):
            trade['status'] = 'winner'
            trade['exit_price'] = trade['target']
            trade['exit_date'] = current_date
            trade['return'] = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']
        
        # Check if stop hit
        elif self._stop_hit (trade, current_price):
            trade['status'] = 'loser'
            trade['exit_price'] = trade['stop']
            trade['exit_date'] = current_date
            trade['return'] = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']
    
    def generate_performance_report (self):
        """
        Analyze which patterns and configurations performed best
        """
        completed_trades = [t for t in self.trades if t['status'] in ['winner', 'loser']]
        
        if not completed_trades:
            return None
        
        # Overall statistics
        total_trades = len (completed_trades)
        winners = [t for t in completed_trades if t['status'] == 'winner']
        win_rate = len (winners) / total_trades
        
        avg_return = np.mean([t['return'] for t in completed_trades])
        avg_winner = np.mean([t['return'] for t in winners]) if winners else 0
        losers = [t for t in completed_trades if t['status'] == 'loser']
        avg_loser = np.mean([t['return'] for t in losers]) if losers else 0
        
        # Performance by pattern type
        pattern_performance = {}
        for trade in completed_trades:
            pattern_type = trade['pattern']['pattern']
            if pattern_type not in pattern_performance:
                pattern_performance[pattern_type] = []
            pattern_performance[pattern_type].append (trade['return'])
        
        # Performance by confidence level
        high_conf_trades = [t for t in completed_trades 
                           if t['pattern']['adjusted_confidence'] > 0.70]
        low_conf_trades = [t for t in completed_trades 
                          if t['pattern']['adjusted_confidence'] <= 0.70]
        
        report = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'avg_winner': avg_winner,
            'avg_loser': avg_loser,
            'profit_factor': abs (avg_winner / avg_loser) if avg_loser != 0 else None,
            'pattern_performance': {
                k: {'count': len (v), 'avg_return': np.mean (v), 'win_rate': sum(1 for x in v if x > 0)/len (v)}
                for k, v in pattern_performance.items()
            },
            'high_confidence_win_rate': sum(1 for t in high_conf_trades if t['return'] > 0) / len (high_conf_trades) if high_conf_trades else None,
            'low_confidence_win_rate': sum(1 for t in low_conf_trades if t['return'] > 0) / len (low_conf_trades) if low_conf_trades else None
        }
        
        return report
    
    def generate_model_improvements (self, report):
        """
        Use performance data to suggest model improvements
        """
        suggestions = []
        
        # Check if confidence scoring is working
        if report['high_confidence_win_rate'] and report['low_confidence_win_rate']:
            if report['high_confidence_win_rate'] <= report['low_confidence_win_rate']:
                suggestions.append({
                    'issue': 'Confidence scoring not predictive',
                    'action': 'Retrain confidence model with recent data',
                    'priority': 'HIGH'
                })
        
        # Check individual pattern performance
        for pattern, stats in report['pattern_performance'].items():
            if stats['win_rate'] < 0.45 and stats['count'] > 10:
                suggestions.append({
                    'issue': f"{pattern} underperforming ({stats['win_rate']:.1%} win rate)",
                    'action': f'Increase {pattern} detection threshold or add filters',
                    'priority': 'MEDIUM'
                })
        
        # Check if profit factor is adequate
        if report['profit_factor'] and report['profit_factor'] < 1.5:
            suggestions.append({
                'issue': f"Low profit factor ({report['profit_factor']:.2f})",
                'action': 'Tighten risk/reward requirements or improve stop placement',
                'priority': 'HIGH'
            })
        
        return suggestions
\`\`\`

### Component 3: Adaptive Learning System

\`\`\`python
from sklearn.ensemble import GradientBoostingClassifier
import joblib

class AdaptiveLearner:
    """
    Machine learning system that learns from outcomes
    """
    
    def __init__(self):
        self.model = GradientBoostingClassifier (n_estimators=100)
        self.feature_names = [
            'geometric_confidence',
            'trend_aligned',
            'volatility_normal',
            'breadth_healthy',
            'sector_aligned',
            'risk_reward_ratio',
            'volume_confirmation',
            'pattern_type_encoded'
        ]
        self.trained = False
        
    def prepare_training_data (self, historical_trades):
        """
        Convert trade history to ML training data
        """
        X = []
        y = []
        
        for trade in historical_trades:
            if trade['status'] not in ['winner', 'loser']:
                continue
            
            features = [
                trade['pattern']['confidence'],
                1 if trade['pattern']['filters']['trend'] else 0,
                1 if trade['pattern']['filters']['volatility'] else 0,
                1 if trade['pattern']['filters']['breadth'] else 0,
                1 if trade['pattern']['filters']['sector'] else 0,
                self._calculate_rr_ratio (trade),
                1 if self._had_volume_confirmation (trade) else 0,
                self._encode_pattern_type (trade['pattern']['pattern'])
            ]
            
            X.append (features)
            y.append(1 if trade['status'] == 'winner' else 0)
        
        return np.array(X), np.array (y)
    
    def train (self, historical_trades):
        """
        Train model on historical trade outcomes
        """
        X, y = self.prepare_training_data (historical_trades)
        
        if len(X) < 50:
            print("Insufficient training data (need 50+ completed trades)")
            return False
        
        # Train model
        self.model.fit(X, y)
        self.trained = True
        
        # Feature importance
        importances = self.model.feature_importances_
        for name, importance in zip (self.feature_names, importances):
            print(f"{name}: {importance:.3f}")
        
        return True
    
    def predict_success_probability (self, pattern, filters):
        """
        Predict probability that pattern trade will succeed
        """
        if not self.trained:
            return None
        
        features = [
            pattern['confidence'],
            1 if filters['trend'] else 0,
            1 if filters['volatility'] else 0,
            1 if filters['breadth'] else 0,
            1 if filters['sector'] else 0,
            self._calculate_rr_from_pattern (pattern),
            1 if self._check_volume_confirm (pattern) else 0,
            self._encode_pattern_type (pattern['pattern'])
        ]
        
        prob = self.model.predict_proba([features])[0][1]
        return prob
    
    def save_model (self, filepath='pattern_model.pkl'):
        """
        Save trained model for later use
        """
        joblib.dump (self.model, filepath)
    
    def load_model (self, filepath='pattern_model.pkl'):
        """
        Load previously trained model
        """
        self.model = joblib.load (filepath)
        self.trained = True
\`\`\`

## Implementation Strategy

### Phase 1: Foundation (Months 1-2)
1. Build geometric pattern detectors for 5-10 common patterns
2. Implement basic filtering (trend, volatility, risk/reward)
3. Create manual review interface
4. Begin collecting data on all patterns (detected and reviewed)

### Phase 2: Feedback Integration (Months 3-4)
5. Implement performance tracking system
6. Collect 50-100 completed trades
7. Analyze which filters correlate with success
8. Refine algorithmic detection based on feedback

### Phase 3: Machine Learning (Months 5-6)
9. Train initial ML model on collected data
10. Compare ML predictions vs human decisions
11. Implement confidence score adjustment based on ML
12. Continue collecting data for model improvement

### Phase 4: Optimization (Months 7-12)
13. Implement regime detection (bull/bear/sideways)
14. Create pattern-specific models
15. Add automated execution for highest-confidence patterns
16. Continuous retraining on rolling 6-12 month window

## Best Practices for Hybrid Systems

### 1. Start Conservative
- Initially require human approval for all trades
- Gradually increase automation as confidence grows
- Always maintain human oversight capability

### 2. Document Everything
- Log all patterns detected (not just traded ones)
- Record human reasoning for approvals/rejections
- Track market conditions at time of trade

### 3. Regular Audits
- Weekly review of system performance
- Monthly review of false positives/negatives
- Quarterly full system audit and retraining

### 4. Regime Awareness
- Market conditions change - models must adapt
- Consider separate models for bull/bear/sideways
- Reduce position size during regime transitions

### 5. Human Override Always Available
- Never fully automate without kill switch
- Human can always intervene
- System suggests but human decides (initially)

## Conclusion

The optimal pattern recognition system leverages machine strengths (speed, consistency, backtesting) while retaining human strengths (context, quality assessment, adaptation). Neither pure automation nor pure discretion performs as well as a thoughtful hybrid.

Start with algorithmic screening to find candidates, apply systematic filters, require human review initially, track everything rigorously, and gradually increase automation as the system proves reliable. The feedback loop - where outcomes continuously improve both algorithmic detection and human decision criteria - is what separates professional systems from amateur ones.
      `,
    },
  ],
};
