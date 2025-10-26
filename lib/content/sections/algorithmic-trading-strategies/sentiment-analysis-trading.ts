export const sentimentAnalysisTrading = {
    title: 'Sentiment Analysis Trading',
    slug: 'sentiment-analysis-trading',
    description: 'Build sentiment-driven strategies using social media, analyst reports, and alternative data',
    content: `
# Sentiment Analysis Trading

## Introduction: Measuring Market Emotion

Markets are driven by fear and greed. Sentiment analysis quantifies these emotions by analyzing text from social media, news, analyst reports, and other sources. When everyone is euphoric, it might be time to sell. When panic reigns, opportunity emerges. Modern sentiment trading uses NLP, machine learning, and alternative data to gain an edge.

**What you'll learn:**
- Sentiment data sources (Twitter, Reddit, StockTwits, news)
- NLP techniques (FinBERT, VADER, transformers)
- Sentiment indicators (bull/bear ratio, put/call, VIX)
- Contrarian vs. momentum sentiment strategies
- Alternative data (social media, satellite, credit card)
- Building a sentiment pipeline

**Why this matters for engineers:**
- Big data problem (millions of tweets/day)
- Real-time streaming (Kafka, Flink)
- ML/NLP skills highly applicable
- Measurable signals (backtest sentiment vs. returns)

**Performance Characteristics:**
- **Win Rate**: 55-65% (varies by source)
- **Sharpe Ratio**: 0.8-2.0
- **Holding Period**: Hours to weeks
- **Alpha Decay**: Medium (days to weeks)
- **Signal Noise**: High (need filtering)

---

## Sentiment Data Sources

### 1. Social Media

**Twitter:**
- 500M tweets/day
- Real-time market reactions
- Influencer impact (Elon, Cathie Wood)
- $CASHTAGS track mentions

**Reddit (WallStreetBets):**
- 13M+ members
- Meme stock phenomena (GME, AMC)
- Sentiment often contrarian indicator
- Options discussion (YOLO, loss porn)

**StockTwits:**
- Finance-focused
- Bull/bear labels
- Ticker-specific streams
- Cleaner than Twitter

**Challenges:**
- Bots and spam
- Sarcasm detection
- Fake news
- Pump and dump schemes

### 2. News and Media

**Traditional News:**
- Bloomberg, Reuters, WSJ
- High credibility
- Slower than social
- Professional language

**Financial Blogs:**
- Seeking Alpha
- Zero Hedge
- Motley Fool
- Mixed quality

**Analyst Reports:**
- Sell-side research
- Ratings changes (upgrade/downgrade)
- Price targets
- Often lagging

### 3. Alternative Data

**Satellite Imagery:**
- Retail parking lots (foot traffic)
- Oil storage (supply levels)
- Agricultural production
- Shipping activity

**Credit Card Transactions:**
- Consumer spending trends
- Sector performance
- Geographic patterns
- Privacy concerns

**Web Scraping:**
- Job postings (company growth)
- Product reviews
- App downloads
- Website traffic

**App Usage Data:**
- Mobile app engagement
- User retention
- Feature adoption
- Competitive analysis

\`\`\`python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from enum import Enum
from collections import defaultdict, deque
import pandas as pd
import numpy as np
import re
import logging

class SentimentSource(Enum):
    """Sources of sentiment data"""
    TWITTER = "TWITTER"
    REDDIT = "REDDIT"
    STOCKTWITS = "STOCKTWITS"
    NEWS = "NEWS"
    ANALYST = "ANALYST"
    ALTERNATIVE = "ALTERNATIVE"

@dataclass
class SentimentData:
    """
    Individual sentiment datapoint
    """
    timestamp: datetime
    symbol: str
    source: SentimentSource
    text: str
    sentiment_score: float  # -1 (bearish) to +1 (bullish)
    confidence: float  # 0 to 1
    author: Optional[str] = None
    likes: int = 0
    retweets: int = 0
    followers: int = 0
    
    @property
    def weighted_score(self) -> float:
        """
        Calculate weighted sentiment considering engagement
        
        More likes/retweets = more influential
        """
        # Base weight
        weight = 1.0
        
        # Engagement weight (logarithmic)
        engagement = self.likes + self.retweets * 2
        if engagement > 0:
            weight *= (1 + np.log10(engagement))
        
        # Follower weight (credibility)
        if self.followers > 1000:
            weight *= (1 + np.log10(self.followers / 1000))
        
        return self.sentiment_score * weight

@dataclass
class AggregatedSentiment:
    """
    Aggregated sentiment for a symbol over time period
    """
    symbol: str
    start_time: datetime
    end_time: datetime
    mean_sentiment: float
    median_sentiment: float
    sentiment_std: float
    volume: int  # Number of mentions
    bullish_pct: float
    bearish_pct: float
    source_breakdown: Dict[SentimentSource, int]
    
    @property
    def sentiment_momentum(self) -> float:
        """
        Calculate sentiment momentum (direction of change)
        Positive = getting more bullish
        """
        # This would compare to previous period
        # Placeholder implementation
        return 0.0
    
    @property
    def is_extreme(self) -> bool:
        """
        Is sentiment at extreme levels?
        Extreme levels often mean reversal
        """
        return abs(self.mean_sentiment) > 0.8

class SentimentAnalyzer:
    """
    Analyze text sentiment using various NLP techniques
    """
    
    def __init__(self, model_type: str = "finbert"):
        """
        Initialize sentiment analyzer
        
        Args:
            model_type: "finbert", "vader", "distilbert"
        """
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)
        
        # Load model (placeholder)
        if model_type == "finbert":
            # from transformers import pipeline
            # self.model = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            pass
    
    def analyze(self, text: str) -> Tuple[float, float]:
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
            
        Returns:
            (sentiment_score, confidence)
        """
        if self.model_type == "vader":
            return self._vader_analysis(text)
        else:
            return self._transformer_analysis(text)
    
    def _vader_analysis(self, text: str) -> Tuple[float, float]:
        """
        VADER lexicon-based sentiment (fast, simple)
        
        Good for social media (handles emojis, slang)
        """
        # Financial keywords
        bullish_words = {
            'moon': 2.0, '🚀': 2.0, 'calls': 1.5, 'buy': 1.0,
            'bullish': 1.5, 'pump': 1.5, 'gains': 1.5, 'profit': 1.0,
            'beat': 1.0, 'surge': 1.5, 'rally': 1.5, 'breakout': 1.5
        }
        
        bearish_words = {
            'puts': 1.5, 'sell': 1.0, 'bearish': 1.5, 'dump': 1.5,
            'loss': 1.5, 'crash': 2.0, 'plunge': 2.0, 'miss': 1.0,
            'decline': 1.0, 'fall': 1.0, 'weak': 1.0
        }
        
        text_lower = text.lower()
        
        # Calculate scores
        bullish_score = sum(
            weight for word, weight in bullish_words.items()
            if word in text_lower
        )
        
        bearish_score = sum(
            weight for word, weight in bearish_words.items()
            if word in text_lower
        )
        
        # Normalize
        total = bullish_score + bearish_score
        
        if total == 0:
            return 0.0, 0.0
        
        sentiment = (bullish_score - bearish_score) / (bullish_score + bearish_score)
        confidence = min(total / 5.0, 1.0)
        
        return sentiment, confidence
    
    def _transformer_analysis(self, text: str) -> Tuple[float, float]:
        """
        Transformer-based sentiment (FinBERT)
        
        Slower but more accurate
        """
        # Placeholder: Use actual FinBERT in production
        # result = self.model(text[:512])[0]
        # label = result['label']  # positive, negative, neutral
        # score = result['score']
        
        # Map to -1 to +1
        # if label == 'positive':
        #     sentiment = score
        # elif label == 'negative':
        #     sentiment = -score
        # else:
        #     sentiment = 0.0
        
        # Fallback to VADER
        return self._vader_analysis(text)
    
    def detect_sarcasm(self, text: str) -> bool:
        """
        Detect sarcasm (flips sentiment)
        
        Simple heuristics:
        - "yeah right"
        - Excessive punctuation "!!!"
        - Mixed case "gReAt JoB"
        """
        text_lower = text.lower()
        
        sarcasm_indicators = [
            'yeah right',
            'sure thing',
            'oh great',
            '🙄'
        ]
        
        # Check for indicators
        has_sarcasm = any(indicator in text_lower for indicator in sarcasm_indicators)
        
        # Excessive punctuation
        if text.count('!') > 3 or text.count('?') > 2:
            has_sarcasm = True
        
        return has_sarcasm

class TwitterSentimentCollector:
    """
    Collect and process Twitter sentiment
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.analyzer = SentimentAnalyzer(model_type="vader")
        self.logger = logging.getLogger(__name__)
    
    def stream_tweets(self, symbols: List[str]) -> List[SentimentData]:
        """
        Stream tweets for symbols
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            List of sentiment data
        """
        sentiments = []
        
        for symbol in symbols:
            # In production: Use Twitter API v2
            # Search for $SYMBOL cashtag
            query = f"${symbol}"
            
            # Placeholder tweets
            example_tweets = [
                {
                    'text': f'${symbol} to the moon! 🚀 Calls printing',
                    'user': 'trader123',
                    'likes': 50,
                    'retweets': 10,
                    'followers': 5000,
                    'timestamp': datetime.now()
                },
                {
                    'text': f'${symbol} looking weak, might grab puts',
                    'user': 'bearish_bob',
                    'likes': 20,
                    'retweets': 5,
                    'followers': 2000,
                    'timestamp': datetime.now()
                }
            ]
            
            for tweet in example_tweets:
                # Analyze sentiment
                sentiment, confidence = self.analyzer.analyze(tweet['text'])
                
                # Check for sarcasm
                if self.analyzer.detect_sarcasm(tweet['text']):
                    sentiment *= -1  # Flip sentiment
                
                data = SentimentData(
                    timestamp=tweet['timestamp'],
                    symbol=symbol,
                    source=SentimentSource.TWITTER,
                    text=tweet['text'],
                    sentiment_score=sentiment,
                    confidence=confidence,
                    author=tweet['user'],
                    likes=tweet['likes'],
                    retweets=tweet['retweets'],
                    followers=tweet['followers']
                )
                
                sentiments.append(data)
        
        return sentiments
    
    def filter_bots(self, sentiments: List[SentimentData]) -> List[SentimentData]:
        """
        Filter out bot accounts
        
        Bot indicators:
        - New accounts (<30 days)
        - Few followers (<100)
        - High tweet frequency (>50/day)
        - Generic names
        """
        filtered = []
        
        for sentiment in sentiments:
            # Simple filtering (in production: ML model)
            if sentiment.followers < 50:
                continue  # Likely bot
            
            # Check for spam patterns
            if any(word in sentiment.text.lower() for word in ['buy now', 'guaranteed', 'profit']):
                continue  # Likely spam
            
            filtered.append(sentiment)
        
        return filtered

class RedditSentimentCollector:
    """
    Collect sentiment from Reddit (WallStreetBets)
    """
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.analyzer = SentimentAnalyzer(model_type="vader")
    
    def fetch_wsb_mentions(self, lookback_hours: int = 24) -> Dict[str, List[SentimentData]]:
        """
        Fetch WSB mentions for last N hours
        
        Args:
            lookback_hours: Hours to look back
            
        Returns:
            Dict mapping symbol to sentiment data
        """
        # In production: Use PRAW (Python Reddit API Wrapper)
        # reddit = praw.Reddit(client_id=..., client_secret=...)
        # subreddit = reddit.subreddit('wallstreetbets')
        
        mentions = defaultdict(list)
        
        # Placeholder data
        example_posts = [
            {
                'title': 'GME YOLO Update - $500k → $2M 🚀🚀🚀',
                'selftext': 'Bought calls, diamond hands, to the moon',
                'score': 5000,
                'num_comments': 500,
                'created_utc': datetime.now().timestamp()
            },
            {
                'title': 'SPY puts printing, bear market incoming',
                'selftext': 'Economy bad, puts good',
                'score': 1000,
                'num_comments': 200,
                'created_utc': datetime.now().timestamp()
            }
        ]
        
        for post in example_posts:
            # Extract tickers
            tickers = self._extract_tickers(post['title'] + ' ' + post['selftext'])
            
            for ticker in tickers:
                # Analyze sentiment
                text = post['title'] + ' ' + post['selftext']
                sentiment, confidence = self.analyzer.analyze(text)
                
                data = SentimentData(
                    timestamp=datetime.fromtimestamp(post['created_utc']),
                    symbol=ticker,
                    source=SentimentSource.REDDIT,
                    text=text,
                    sentiment_score=sentiment,
                    confidence=confidence,
                    likes=post['score'],
                    retweets=0,  # Not applicable
                    followers=0  # Not applicable
                )
                
                mentions[ticker].append(data)
        
        return mentions
    
    def _extract_tickers(self, text: str) -> List[str]:
        """
        Extract stock tickers from text
        
        Handles: $GME, GME, NYSE:GME
        """
        tickers = []
        
        # $TICKER pattern
        pattern1 = r'\$([A-Z]{1,5})\\b'
        tickers.extend(re.findall(pattern1, text))
        
        # Standalone TICKER (more false positives)
        # pattern2 = r'\\b([A-Z]{2,5})\\b'
        # tickers.extend(re.findall(pattern2, text))
        
        return list(set(tickers))
    
    def calculate_wsb_hype(self, symbol: str, sentiments: List[SentimentData]) -> float:
        """
        Calculate WSB hype score
        
        High hype = contrarian indicator (often means top)
        
        Args:
            symbol: Stock symbol
            sentiments: Sentiment data
            
        Returns:
            Hype score 0-100
        """
        if not sentiments:
            return 0.0
        
        # Volume factor (mentions)
        volume_score = min(len(sentiments) / 100, 1.0) * 40
        
        # Sentiment extremity
        avg_sentiment = np.mean([s.sentiment_score for s in sentiments])
        extremity_score = abs(avg_sentiment) * 30
        
        # Engagement (upvotes)
        total_engagement = sum(s.likes for s in sentiments)
        engagement_score = min(total_engagement / 10000, 1.0) * 30
        
        hype_score = volume_score + extremity_score + engagement_score
        
        return hype_score

class SentimentAggregator:
    """
    Aggregate sentiment from multiple sources
    """
    
    def __init__(self, window_minutes: int = 60):
        """
        Initialize aggregator
        
        Args:
            window_minutes: Time window for aggregation
        """
        self.window_minutes = window_minutes
        self.sentiment_buffer = defaultdict(deque)
    
    def add_sentiment(self, sentiment: SentimentData):
        """Add sentiment datapoint to buffer"""
        self.sentiment_buffer[sentiment.symbol].append(sentiment)
        
        # Clean old data
        self._clean_old_data(sentiment.symbol)
    
    def _clean_old_data(self, symbol: str):
        """Remove data outside time window"""
        cutoff = datetime.now() - timedelta(minutes=self.window_minutes)
        
        buffer = self.sentiment_buffer[symbol]
        
        while buffer and buffer[0].timestamp < cutoff:
            buffer.popleft()
    
    def get_aggregated(self, symbol: str) -> Optional[AggregatedSentiment]:
        """
        Get aggregated sentiment for symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Aggregated sentiment or None
        """
        sentiments = list(self.sentiment_buffer[symbol])
        
        if not sentiments:
            return None
        
        # Calculate statistics
        scores = [s.sentiment_score for s in sentiments]
        
        mean_sentiment = np.mean(scores)
        median_sentiment = np.median(scores)
        sentiment_std = np.std(scores)
        
        bullish_pct = len([s for s in scores if s > 0]) / len(scores)
        bearish_pct = len([s for s in scores if s < 0]) / len(scores)
        
        # Source breakdown
        source_breakdown = defaultdict(int)
        for s in sentiments:
            source_breakdown[s.source] += 1
        
        start_time = min(s.timestamp for s in sentiments)
        end_time = max(s.timestamp for s in sentiments)
        
        return AggregatedSentiment(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            mean_sentiment=mean_sentiment,
            median_sentiment=median_sentiment,
            sentiment_std=sentiment_std,
            volume=len(sentiments),
            bullish_pct=bullish_pct,
            bearish_pct=bearish_pct,
            source_breakdown=dict(source_breakdown)
        )

class SentimentTradingStrategy:
    """
    Trade based on aggregated sentiment
    
    Two approaches:
    1. Momentum: Follow sentiment (bullish → long)
    2. Contrarian: Fade sentiment (extreme bullish → short)
    """
    
    def __init__(self,
                 strategy_type: str = "momentum",
                 sentiment_threshold: float = 0.5,
                 volume_threshold: int = 100):
        """
        Initialize strategy
        
        Args:
            strategy_type: "momentum" or "contrarian"
            sentiment_threshold: Minimum sentiment magnitude
            volume_threshold: Minimum mentions required
        """
        self.strategy_type = strategy_type
        self.sentiment_threshold = sentiment_threshold
        self.volume_threshold = volume_threshold
    
    def generate_signal(self, sentiment: AggregatedSentiment) -> int:
        """
        Generate trading signal
        
        Args:
            sentiment: Aggregated sentiment
            
        Returns:
            1 (long), -1 (short), 0 (no trade)
        """
        # Check volume threshold
        if sentiment.volume < self.volume_threshold:
            return 0
        
        # Check sentiment magnitude
        if abs(sentiment.mean_sentiment) < self.sentiment_threshold:
            return 0
        
        if self.strategy_type == "momentum":
            # Follow sentiment
            if sentiment.mean_sentiment > self.sentiment_threshold:
                return 1
            elif sentiment.mean_sentiment < -self.sentiment_threshold:
                return -1
        
        else:  # contrarian
            # Fade extreme sentiment
            if sentiment.is_extreme:
                # Extreme bullish → short
                if sentiment.mean_sentiment > 0.8:
                    return -1
                # Extreme bearish → long
                elif sentiment.mean_sentiment < -0.8:
                    return 1
        
        return 0
    
    def calculate_conviction(self, sentiment: AggregatedSentiment) -> float:
        """
        Calculate conviction level (for position sizing)
        
        Factors:
        - Sentiment magnitude
        - Volume of mentions
        - Source diversity
        - Sentiment consistency (low std)
        
        Args:
            sentiment: Aggregated sentiment
            
        Returns:
            Conviction 0-1
        """
        conviction = 0.0
        
        # Sentiment magnitude (40%)
        conviction += abs(sentiment.mean_sentiment) * 0.4
        
        # Volume (30%)
        volume_factor = min(sentiment.volume / self.volume_threshold, 1.0)
        conviction += volume_factor * 0.3
        
        # Consistency (20%) - low std = high conviction
        if sentiment.sentiment_std < 0.3:
            conviction += 0.2
        elif sentiment.sentiment_std < 0.5:
            conviction += 0.1
        
        # Source diversity (10%)
        if len(sentiment.source_breakdown) > 1:
            conviction += 0.1
        
        return min(conviction, 1.0)

# Example usage
if __name__ == "__main__":
    print("\\n=== Sentiment Analysis Trading System ===\\n")
    
    # 1. Analyze tweet sentiment
    print("1. Twitter Sentiment Analysis")
    analyzer = SentimentAnalyzer(model_type="vader")
    
    tweets = [
        "TSLA to the moon! 🚀 Elon is a genius, buying more calls",
        "TSLA puts printing, overvalued garbage company",
        "Just bought AAPL shares, strong fundamentals and solid growth"
    ]
    
    for tweet in tweets:
        sentiment, confidence = analyzer.analyze(tweet)
        print(f"   Tweet: {tweet[:50]}...")
        print(f"   Sentiment: {sentiment:.2f} (confidence: {confidence:.2f})\\n")
    
    # 2. Aggregate sentiment
    print("2. Aggregated Sentiment")
    
    aggregator = SentimentAggregator(window_minutes=60)
    
    # Add sample sentiments
    for i in range(100):
        sentiment_data = SentimentData(
            timestamp=datetime.now() - timedelta(minutes=i),
            symbol="GME",
            source=SentimentSource.REDDIT,
            text="Sample text",
            sentiment_score=np.random.normal(0.6, 0.2),  # Bullish bias
            confidence=0.8,
            likes=np.random.randint(10, 1000)
        )
        aggregator.add_sentiment(sentiment_data)
    
    agg = aggregator.get_aggregated("GME")
    
    print(f"   Symbol: {agg.symbol}")
    print(f"   Mean Sentiment: {agg.mean_sentiment:.2f}")
    print(f"   Volume: {agg.volume} mentions")
    print(f"   Bullish: {agg.bullish_pct:.1%}")
    print(f"   Bearish: {agg.bearish_pct:.1%}")
    print(f"   Extreme: {agg.is_extreme}")
    
    # 3. Generate trading signals
    print("\\n3. Trading Signals")
    
    # Momentum strategy
    momentum_strategy = SentimentTradingStrategy(
        strategy_type="momentum",
        sentiment_threshold=0.5
    )
    
    signal = momentum_strategy.generate_signal(agg)
    conviction = momentum_strategy.calculate_conviction(agg)
    
    print(f"   Strategy: Momentum")
    print(f"   Signal: {signal} ({'LONG' if signal == 1 else 'SHORT' if signal == -1 else 'NEUTRAL'})")
    print(f"   Conviction: {conviction:.2f}")
    
    # Contrarian strategy
    contrarian_strategy = SentimentTradingStrategy(
        strategy_type="contrarian",
        sentiment_threshold=0.5
    )
    
    signal_contrarian = contrarian_strategy.generate_signal(agg)
    
    print(f"\\n   Strategy: Contrarian")
    print(f"   Signal: {signal_contrarian} ({'LONG' if signal_contrarian == 1 else 'SHORT' if signal_contrarian == -1 else 'NEUTRAL'})")
    
    # 4. WSB Hype Score
    print("\\n4. WallStreetBets Hype")
    
    reddit = RedditSentimentCollector(client_id="fake", client_secret="fake")
    sentiments = list(aggregator.sentiment_buffer["GME"])
    
    hype_score = reddit.calculate_wsb_hype("GME", sentiments)
    
    print(f"   Symbol: GME")
    print(f"   Hype Score: {hype_score:.1f}/100")
    print(f"   Interpretation: {'EXTREME HYPE (contrarian short)' if hype_score > 70 else 'MODERATE' if hype_score > 40 else 'LOW'}")
\`\`\`

---

## Sentiment Indicators

### 1. Traditional Indicators

**VIX (Fear Index):**
- Measures implied volatility of S&P 500 options
- High VIX = fear (often buying opportunity)
- Low VIX = complacency (potential risk)

**Put/Call Ratio:**
- Puts / Calls traded
- High ratio = bearish (contrarian bullish)
- Low ratio = bullish (contrarian bearish)

**Bull/Bear Surveys:**
- AAII Sentiment Survey
- Investors Intelligence
- Extreme readings = contrarian signals

### 2. Modern Indicators

**Social Media Sentiment:**
- Twitter mentions and sentiment
- Reddit upvotes and comments
- StockTwits bull/bear ratio

**News Sentiment:**
- Bloomberg sentiment scores
- Reuters sentiment
- RavenPack analytics

**Alternative Data:**
- Google search trends
- App store rankings
- Web traffic analytics

---

## Contrarian vs. Momentum

### Contrarian Approach

**Logic**: When everyone is bullish, who's left to buy?

**Implementation:**
- Extreme sentiment = reversal signal
- Short when WSB hype > 80
- Long when fear > 70

**Example:**
- GME at $300 with extreme WSB hype → short
- AAPL at $120 with extreme bearishness → long

**Risks:**
- "The market can stay irrational longer than you can stay solvent"
- Timing is everything

### Momentum Approach

**Logic**: Trend is your friend

**Implementation:**
- Follow positive sentiment
- Long when sentiment improving
- Short when sentiment deteriorating

**Example:**
- TSLA sentiment turning bullish → long
- Netflix sentiment declining → short

**Risks:**
- Late to the party
- Sentiment can reverse quickly

---

## Real-World Examples

### GameStop (GME) - January 2021

**What Happened:**
- WallStreetBets coordinated GME squeeze
- Sentiment went from 0 to 100
- Stock went $20 → $483 in weeks

**Lessons:**
- Extreme social media sentiment = risk
- Momentum worked (for a while)
- Contrarian worked (eventually)
- Risk management critical

### Tesla (TSLA) - Elon's Tweets

**Pattern:**
- Elon tweets positive → stock up
- Elon tweets negative → stock down
- "Funding secured" → SEC investigation

**Trading Strategy:**
- Monitor Elon's Twitter in real-time
- Immediate reaction (seconds matter)
- Fade after initial move

---

## Building a Sentiment Pipeline

### Architecture

**Data Collection:**
- Twitter Streaming API
- Reddit API (PRAW)
- News APIs (Bloomberg, Reuters)
- Alternative data providers

**Processing:**
- NLP sentiment analysis
- Entity extraction (tickers, companies)
- Spam/bot filtering
- Aggregation and storage

**Signal Generation:**
- Real-time sentiment scores
- Aggregation windows (1hr, 1day, 1week)
- Comparison to historical baselines
- Trading signals

**Execution:**
- Automated order placement
- Position sizing based on conviction
- Risk limits and circuit breakers

**Tech Stack:**
- **Streaming**: Kafka, Flink
- **NLP**: Hugging Face Transformers, spaCy
- **Storage**: TimescaleDB, Redis
- **Compute**: AWS Lambda, Kubernetes
- **Monitoring**: Grafana, Prometheus

---

## Summary and Key Takeaways

**Sentiment Analysis Works When:**
- Clear sentiment shifts
- Multiple sources confirm
- Actionable timeframe (not too late)
- Proper filtering (bots, spam)

**Sentiment Analysis Fails When:**
- Noisy data
- Already priced in
- Fake/manipulated sentiment
- Wrong strategy (momentum vs. contrarian)

**Critical Success Factors:**
1. **Data Quality**: Filter bots and spam
2. **Speed**: Real-time processing
3. **Multiple Sources**: Don't rely on one
4. **Context**: Understand why sentiment shifted
5. **Risk Management**: Sentiment can flip fast

**Best Practices:**
- Start with traditional indicators (VIX, Put/Call)
- Add social media cautiously
- Backtest extensively
- Use sentiment as filter, not sole signal
- Combine with technical/fundamental analysis

**Next Section:** Factor Investing Strategies
`,
};
