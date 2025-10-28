export const newsBasedTrading = {
  title: 'News-Based Trading',
  slug: 'news-based-trading',
  description:
    'Trade on news events: earnings, economic releases, and real-time news processing with NLP',
  content: `
# News-Based Trading

## Introduction: Information as Alpha

News moves markets. The announcement of an earnings beat can send a stock up 10% in seconds. A Fed rate decision can move the entire market. Algorithmic traders who can process and act on news faster than others capture significant alpha. This is the domain where microseconds matter and natural language processing meets high-frequency trading.

**What you'll learn:**
- Event-driven trading strategies (earnings, economic data, M&A)
- Natural Language Processing (NLP) for news
- Sentiment analysis with FinBERT and transformer models
- Latency arbitrage and speed advantages
- Post-Earnings Announcement Drift (PEAD)
- News impact measurement and prediction

**Why this matters for engineers:**
- Speed = alpha (first mover advantage is massive)
- NLP/ML skills directly applicable
- Infrastructure challenges (low latency, real-time processing)
- Measurable edge (compare execution before/after news)

**Performance Characteristics:**
- **Win Rate**: 60-70% on clear earnings surprises
- **Sharpe Ratio**: 1.5-3.0 (varies by strategy)
- **Holding Period**: Minutes to days
- **Alpha Decay**: Very fast (hours to days)
- **Latency Sensitivity**: Critical (<100ms for HFT, <1s for earnings drift)

---

## Types of News Events

### 1. Scheduled Events (Known Timing)

**Earnings Announcements:**
- Released quarterly after market close or pre-market
- Known date/time in advance
- High impact (5-15% moves common)
- Predictable opportunity

**Economic Data:**
- Non-Farm Payrolls (NFP): First Friday each month, 8:30 AM ET
- CPI (Inflation): Monthly, 8:30 AM ET
- Fed Announcements: 2:00 PM ET on Fed meeting days
- GDP: Quarterly

**Corporate Events:**
- Product launches (Apple events, Tesla reveals)
- FDA approvals (biotech)
- Analyst days
- Shareholder meetings

### 2. Unscheduled Events (Surprise)

**Breaking News:**
- M&A announcements
- Executive changes
- Regulatory actions
- Natural disasters
- Geopolitical events

**Social Media:**
- CEO tweets (Elon Musk, Trump)
- Rumors and leaks
- Analyst upgrades/downgrades
- Activist investor campaigns

\`\`\`python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from enum import Enum
import pandas as pd
import numpy as np
import requests
import json
from collections import deque
import logging

class EventType(Enum):
    """Types of news events"""
    EARNINGS = "EARNINGS"
    ECONOMIC_DATA = "ECONOMIC_DATA"
    FDA_APPROVAL = "FDA_APPROVAL"
    MERGER_ACQUISITION = "M&A"
    EXECUTIVE_CHANGE = "EXECUTIVE_CHANGE"
    PRODUCT_LAUNCH = "PRODUCT_LAUNCH"
    ANALYST_RATING = "ANALYST_RATING"
    BREAKING_NEWS = "BREAKING_NEWS"

class EventImpact(Enum):
    """Expected impact magnitude"""
    HIGH = "HIGH"  # >5% expected move
    MEDIUM = "MEDIUM"  # 2-5% expected move
    LOW = "LOW"  # <2% expected move

@dataclass
class NewsEvent:
    """
    Represents a news event
    """
    timestamp: datetime
    symbol: str
    event_type: EventType
    headline: str
    content: str
    source: str
    sentiment_score: float  # -1 to +1
    magnitude: float  # 0 to 1 (how significant)
    expected_impact: EventImpact
    
    def is_material(self) -> bool:
        """Determine if event is material (worth trading)"""
        return (
            abs(self.sentiment_score) > 0.5 and
            self.magnitude > 0.6 and
            self.expected_impact in [EventImpact.HIGH, EventImpact.MEDIUM]
        )

@dataclass
class EarningsSurprise:
    """
    Earnings announcement data
    """
    symbol: str
    report_date: datetime
    actual_eps: float
    expected_eps: float
    actual_revenue: float
    expected_revenue: float
    
    @property
    def eps_surprise_pct(self) -> float:
        """Calculate EPS surprise percentage"""
        if self.expected_eps == 0:
            return 0
        return (self.actual_eps - self.expected_eps) / abs(self.expected_eps)
    
    @property
    def revenue_surprise_pct(self) -> float:
        """Calculate revenue surprise percentage"""
        if self.expected_revenue == 0:
            return 0
        return (self.actual_revenue - self.expected_revenue) / self.expected_revenue
    
    @property
    def is_beat(self) -> bool:
        """Did company beat expectations?"""
        return self.eps_surprise_pct > 0.02  # >2% beat
    
    @property
    def is_miss(self) -> bool:
        """Did company miss expectations?"""
        return self.eps_surprise_pct < -0.02  # >2% miss

class NewsDataProvider:
    """
    Interface to news data providers
    
    Production providers:
    - Bloomberg Terminal
    - Reuters Machine Readable News
    - RavenPack
    - Alpha Vantage
    - NewsAPI
    """
    
    def __init__(self, api_key: str, provider: str = "newsapi"):
        self.api_key = api_key
        self.provider = provider
        self.logger = logging.getLogger(__name__)
        
    def fetch_real_time_news(self, symbols: List[str]) -> List[NewsEvent]:
        """
        Fetch real-time news for symbols
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            List of news events
        """
        news_events = []
        
        for symbol in symbols:
            try:
                if self.provider == "newsapi":
                    url = f"https://newsapi.org/v2/everything"
                    params = {
                        'q': symbol,
                        'apiKey': self.api_key,
                        'language': 'en',
                        'sortBy': 'publishedAt',
                        'pageSize': 20
                    }
                    
                    response = requests.get(url, params=params, timeout=5)
                    data = response.json()
                    
                    for article in data.get('articles', []):
                        event = NewsEvent(
                            timestamp=datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                            symbol=symbol,
                            event_type=EventType.BREAKING_NEWS,
                            headline=article['title'],
                            content=article.get('description', ''),
                            source=article['source']['name'],
                            sentiment_score=0.0,  # Will be calculated
                            magnitude=0.5,  # Default
                            expected_impact=EventImpact.MEDIUM
                        )
                        news_events.append(event)
                        
            except Exception as e:
                self.logger.error(f"Error fetching news for {symbol}: {e}")
                
        return news_events
    
    def fetch_earnings_calendar(self, date: datetime) -> List[Dict]:
        """
        Fetch earnings calendar for date
        
        Args:
            date: Target date
            
        Returns:
            List of earnings events
        """
        # In production: Use Alpha Vantage, Earnings Whispers, etc.
        # Placeholder implementation
        earnings = []
        
        # Example earnings event
        earnings.append({
            'symbol': 'AAPL',
            'report_date': date.replace(hour=16, minute=30),  # After close
            'expected_eps': 1.25,
            'expected_revenue': 90_000_000_000,
            'time': 'AMC'  # After Market Close
        })
        
        return earnings
    
    def fetch_economic_calendar(self) -> List[Dict]:
        """
        Fetch economic data release calendar
        
        Returns:
            List of upcoming economic releases
        """
        # Key economic indicators
        calendar = [
            {
                'event': 'Non-Farm Payrolls',
                'date': 'First Friday',
                'time': '8:30 AM ET',
                'impact': EventImpact.HIGH,
                'market': 'Equities, Bonds, FX'
            },
            {
                'event': 'CPI (Inflation)',
                'date': 'Mid-month',
                'time': '8:30 AM ET',
                'impact': EventImpact.HIGH,
                'market': 'All'
            },
            {
                'event': 'FOMC Decision',
                'date': '8 times per year',
                'time': '2:00 PM ET',
                'impact': EventImpact.HIGH,
                'market': 'All'
            },
            {
                'event': 'GDP',
                'date': 'Quarterly',
                'time': '8:30 AM ET',
                'impact': EventImpact.MEDIUM,
                'market': 'All'
            }
        ]
        
        return calendar

class NaturalLanguageProcessor:
    """
    NLP for financial news processing
    
    Uses transformer models (FinBERT) for sentiment analysis
    """
    
    def __init__(self, model_type: str = "finbert"):
        """
        Initialize NLP processor
        
        Args:
            model_type: "finbert", "distilbert", or "vader" (simple)
        """
        self.model_type = model_type
        self.sentiment_analyzer = None
        
        # Initialize model
        if model_type == "finbert":
            # In production: Load actual FinBERT model
            # from transformers import pipeline
            # self.sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            pass
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_sentiment(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
            
        Returns:
            (sentiment_score, confidence) where score is -1 to +1
        """
        if self.model_type == "vader":
            # Simple keyword-based approach
            return self._vader_sentiment(text)
        else:
            # Transformer-based (FinBERT)
            return self._finbert_sentiment(text)
    
    def _vader_sentiment(self, text: str) -> Tuple[float, float]:
        """
        Simple keyword-based sentiment (fast but less accurate)
        
        Args:
            text: Text to analyze
            
        Returns:
            (sentiment, confidence)
        """
        positive_keywords = [
            'beat', 'surge', 'soar', 'jump', 'rally', 'gain', 'profit',
            'strong', 'growth', 'expand', 'record', 'exceed', 'outperform'
        ]
        
        negative_keywords = [
            'miss', 'plunge', 'crash', 'fall', 'decline', 'loss', 'weak',
            'disappoint', 'concern', 'warning', 'cut', 'reduce', 'underperform'
        ]
        
        text_lower = text.lower()
        
        pos_count = sum(1 for word in positive_keywords if word in text_lower)
        neg_count = sum(1 for word in negative_keywords if word in text_lower)
        
        total = pos_count + neg_count
        
        if total == 0:
            return 0.0, 0.0
        
        sentiment = (pos_count - neg_count) / total
        confidence = min(total / 5, 1.0)  # More keywords = higher confidence
        
        return sentiment, confidence
    
    def _finbert_sentiment(self, text: str) -> Tuple[float, float]:
        """
        Transformer-based sentiment (slow but accurate)
        
        Args:
            text: Text to analyze
            
        Returns:
            (sentiment, confidence)
        """
        # Placeholder: In production, use actual FinBERT
        # result = self.sentiment_analyzer(text[:512])[0]
        # label = result['label']  # positive, negative, neutral
        # score = result['score']  # confidence
        
        # For now, use VADER as fallback
        return self._vader_sentiment(text)
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities (companies, people, amounts)
        
        Args:
            text: Text to analyze
            
        Returns:
            List of entities
        """
        # In production: Use spaCy or similar NER
        entities = []
        
        # Simple example: Extract dollar amounts
        import re
        amounts = re.findall(r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|M|B))?', text)
        
        for amount in amounts:
            entities.append({
                'type': 'MONEY',
                'value': amount,
                'context': 'financial'
            })
        
        return entities
    
    def calculate_news_importance(self, news: NewsEvent) -> float:
        """
        Calculate importance/materiality of news
        
        Factors:
        - Source credibility (Bloomberg > Random blog)
        - Sentiment magnitude
        - Entity mentions
        - Headline vs. detailed content
        
        Args:
            news: News event
            
        Returns:
            Importance score 0-1
        """
        score = 0.0
        
        # Source credibility (40 points)
        credible_sources = ['Bloomberg', 'Reuters', 'WSJ', 'FT']
        if any(source in news.source for source in credible_sources):
            score += 0.40
        else:
            score += 0.20
        
        # Sentiment magnitude (30 points)
        score += abs(news.sentiment_score) * 0.30
        
        # Content length (10 points) - longer = more detailed
        if len(news.content) > 200:
            score += 0.10
        elif len(news.content) > 100:
            score += 0.05
        
        # Headline quality (20 points)
        if any(word in news.headline.lower() for word in ['earnings', 'revenue', 'profit', 'loss']):
            score += 0.20
        
        return min(score, 1.0)

class EarningsSurpriseTrader:
    """
    Trade on earnings surprises
    
    Strategy: Post-Earnings Announcement Drift (PEAD)
    - Earnings beats drift up over next 30-60 days
    - Earnings misses drift down
    """
    
    def __init__(self,
                 surprise_threshold: float = 0.05,  # 5% surprise
                 hold_days: int = 30):
        """
        Initialize earnings trader
        
        Args:
            surprise_threshold: Minimum surprise to trade (5% default)
            hold_days: Holding period for PEAD (30 days default)
        """
        self.surprise_threshold = surprise_threshold
        self.hold_days = hold_days
        self.positions = {}
        
    def analyze_earnings(self, earnings: EarningsSurprise) -> Optional[int]:
        """
        Analyze earnings and generate signal
        
        Args:
            earnings: Earnings data
            
        Returns:
            Signal: 1 (long), -1 (short), 0 (no trade)
        """
        surprise = earnings.eps_surprise_pct
        
        # Check if surprise is significant
        if abs(surprise) < self.surprise_threshold:
            return 0  # Not significant enough
        
        # Positive surprise → Long
        if surprise > self.surprise_threshold:
            return 1
        
        # Negative surprise → Short
        if surprise < -self.surprise_threshold:
            return -1
        
        return 0
    
    def calculate_position_size(self,
                               surprise_magnitude: float,
                               capital: float,
                               max_position_pct: float = 0.05) -> float:
        """
        Calculate position size based on surprise magnitude
        
        Larger surprise = larger position
        
        Args:
            surprise_magnitude: Absolute surprise percentage
            capital: Total capital
            max_position_pct: Max position as % of capital (5% default)
            
        Returns:
            Position size in dollars
        """
        # Base position
        base_size = capital * max_position_pct
        
        # Scale by surprise magnitude
        # 5% surprise → 1.0x
        # 10% surprise → 2.0x
        # Cap at 2.0x
        scale = min(surprise_magnitude / 0.05, 2.0)
        
        position_size = base_size * scale
        
        return position_size
    
    def estimate_drift(self, surprise_pct: float, days_held: int) -> float:
        """
        Estimate expected drift based on historical PEAD
        
        Research: ~60% of surprise magnitude drifts over 30 days
        
        Args:
            surprise_pct: EPS surprise percentage
            days_held: Days to hold
            
        Returns:
            Expected price drift
        """
        # Historical PEAD: ~60% of surprise over 30 days
        drift_coefficient = 0.60
        
        # Time decay: Full drift over 30 days
        time_factor = min(days_held / 30, 1.0)
        
        expected_drift = surprise_pct * drift_coefficient * time_factor
        
        return expected_drift

class LatencyArbitrageTrader:
    """
    Trade on speed advantage in news processing
    
    Edge: Process news faster than others
    Typical advantage: 1-100 milliseconds
    """
    
    def __init__(self, latency_advantage_ms: float = 10.0):
        """
        Initialize latency arbitrage trader
        
        Args:
            latency_advantage_ms: Speed advantage in milliseconds
        """
        self.latency_advantage_ms = latency_advantage_ms
        self.logger = logging.getLogger(__name__)
        
    def estimate_latency_edge(self,
                             news_timestamp: datetime,
                             current_time: datetime,
                             expected_market_reaction_ms: float = 100.0) -> bool:
        """
        Determine if we have latency edge
        
        Args:
            news_timestamp: When news was published
            current_time: Current time
            expected_market_reaction_ms: How fast market typically reacts
            
        Returns:
            True if we have edge
        """
        # How long ago was news released?
        time_since_news = (current_time - news_timestamp).total_seconds() * 1000
        
        # Do we have time to trade before market fully reacts?
        edge_window = expected_market_reaction_ms - self.latency_advantage_ms
        
        has_edge = time_since_news < edge_window
        
        return has_edge
    
    def calculate_expected_profit(self,
                                 latency_advantage_ms: float,
                                 price_impact: float,
                                 position_size: int) -> float:
        """
        Calculate expected profit from latency advantage
        
        Args:
            latency_advantage_ms: Speed advantage
            price_impact: Expected price move (in %)
            position_size: Shares to trade
            
        Returns:
            Expected profit
        """
        # Profit = % of price move captured before competition
        # If 10ms advantage and 100ms total reaction time:
        # Capture 10% of move
        
        # Typical market reaction time
        market_reaction_ms = 100.0
        
        # Fraction of move we capture
        capture_rate = latency_advantage_ms / market_reaction_ms
        capture_rate = min(capture_rate, 0.5)  # Cap at 50%
        
        # Expected profit
        expected_profit = position_size * price_impact * capture_rate
        
        return expected_profit

# Example usage
if __name__ == "__main__":
    print("\\n=== News-Based Trading System ===\\n")
    
    # 1. Earnings Surprise Trading
    print("1. Earnings Surprise Analysis")
    earnings = EarningsSurprise(
        symbol="AAPL",
        report_date=datetime(2024, 1, 30, 16, 30),
        actual_eps=2.50,
        expected_eps=2.30,
        actual_revenue=125_000_000_000,
        expected_revenue=120_000_000_000
    )
    
    print(f"   EPS Surprise: {earnings.eps_surprise_pct:.2%}")
    print(f"   Revenue Surprise: {earnings.revenue_surprise_pct:.2%}")
    print(f"   Beat/Miss: {'BEAT' if earnings.is_beat else 'MISS' if earnings.is_miss else 'INLINE'}")
    
    trader = EarningsSurpriseTrader()
    signal = trader.analyze_earnings(earnings)
    drift = trader.estimate_drift(earnings.eps_surprise_pct, 30)
    
    print(f"   Trading Signal: {signal}")
    print(f"   Expected 30-day drift: {drift:.2%}")
    
    # 2. NLP Sentiment Analysis
    print("\\n2. NLP Sentiment Analysis")
    nlp = NaturalLanguageProcessor(model_type="vader")
    
    news_text = "Apple beats earnings expectations with record revenue growth of 25%, strong iPhone sales"
    sentiment, confidence = nlp.analyze_sentiment(news_text)
    
    print(f"   Text: {news_text[:60]}...")
    print(f"   Sentiment: {sentiment:.2f} (confidence: {confidence:.2f})")
    
    # 3. Latency Arbitrage
    print("\\n3. Latency Arbitrage")
    lat_trader = LatencyArbitrageTrader(latency_advantage_ms=10.0)
    
    news_time = datetime.now() - timedelta(milliseconds=5)
    has_edge = lat_trader.estimate_latency_edge(news_time, datetime.now())
    
    expected_profit = lat_trader.calculate_expected_profit(
        latency_advantage_ms=10.0,
        price_impact=0.02,  # 2% expected move
        position_size=10000
    )
    
    print(f"   Latency Advantage: 10ms")
    print(f"   Has Edge: {has_edge}")
    print(f"   Expected Profit: \\$\{expected_profit:.2f})"
\`\`\`

---

## Post-Earnings Announcement Drift (PEAD)

### The Anomaly

**Research Finding**: Stocks that beat earnings expectations drift up over the next 30-60 days. Stocks that miss drift down.

**Why It Exists:**1. **Under-reaction**: Investors slow to fully incorporate news
2. **Anchoring**: Previous price expectations stick
3. **Herding**: Others follow initial movers
4. **Institutional buying**: Takes time to accumulate positions

**Profitability:**
- ~60% of earnings surprise drifts over 30 days
- 5% surprise → ~3% drift over month
- Sharpe ratio: 1.5-2.5

**Implementation:**
- Enter immediately after earnings (pre-market if possible)
- Hold 20-40 days
- Exit before next earnings

---

## Real-World Examples

### Renaissance Technologies - News Processing

**Technology:**
- Process news in microseconds
- Machine learning for sentiment
- Historical pattern recognition
- Correlation with price moves

**Edge:**
- Speed (faster than humans)
- Scale (process thousands of sources)
- Patterns (historical relationships)

### Two Sigma - Alternative Data

**Data Sources:**
- Satellite imagery (retail parking lots)
- Credit card transactions
- Web scraping
- Social media sentiment

**Example:**
- Monitor Chipotle parking lots via satellite
- Predict quarterly revenue before earnings
- Trade before announcement

---

## Summary and Key Takeaways

**News Trading Works When:**
- Clear, material information
- Fast processing (first mover advantage)
- Predictable market reactions
- Post-announcement drift (PEAD)

**News Trading Fails When:**
- Already priced in
- Too slow to react
- False signals (fake news)
- Market already moved

**Critical Success Factors:**1. **Speed**: Sub-second processing for earnings
2. **Quality**: Filter noise, focus on material news
3. **NLP**: Accurate sentiment analysis
4. **Execution**: Fast order routing
5. **Risk Management**: False signals happen

**Next Section:** Sentiment Analysis Trading
`,
};
