export const financialNewsAnalysis = {
  title: 'Financial News Analysis at Scale',
  id: 'financial-news-analysis',
  content: `
# Financial News Analysis at Scale

## Introduction

Financial news drives markets. A single breaking news story can move a stock 10%+ in minutes. However, thousands of news articles, tweets, blog posts, and press releases are published every hour across global markets. Manual monitoring is impossible-sophisticated, automated news analysis systems are essential for modern trading.

This section covers building production-scale news analysis systems using LLMs: real-time processing, event extraction, entity recognition, sentiment aggregation, news impact assessment, and generating actionable trading signals from news flow.

### Why News Analysis Matters

**Speed**: First movers capture most of the price movement
**Volume**: Too much news for humans to process
**Context**: Understanding implications requires domain knowledge
**Signals**: News contains trading signals before they appear in prices
**Risk**: Breaking negative news can cause sudden losses

---

## News Data Sources

### Financial News APIs and Feeds

\`\`\`python
"""
Connect to financial news data sources
"""

import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import feedparser

class NewsDataRetriever:
    """
    Retrieve news from multiple sources
    """
    
    def __init__(self):
        self.sources = {
            'alpha_vantage': 'News & sentiment API',
            'news_api': 'General news aggregator',
            'polygon': 'Financial news with market data',
            'benzinga': 'Trading news and analysis',
            'rss_feeds': 'Free RSS feeds from major outlets'
        }
    
    def get_alpha_vantage_news(self, api_key: str, tickers: List[str] = None,
                                topics: List[str] = None,
                                time_from: str = None,
                                limit: int = 50) -> List[Dict]:
        """
        Get news from Alpha Vantage News & Sentiment API
        
        Args:
            api_key: Alpha Vantage API key
            tickers: List of stock tickers to filter
            topics: List of topics (e.g., 'earnings', 'merger', 'ipo')
            time_from: Start time (format: YYYYMMDDTHHMM)
            limit: Max number of articles
            
        Returns:
            List of news articles
        """
        url = "https://www.alphavantage.co/query"
        
        params = {
            'function': 'NEWS_SENTIMENT',
            'apikey': api_key,
            'limit': limit
        }
        
        if tickers:
            params['tickers'] = ','.join(tickers)
        
        if topics:
            params['topics'] = ','.join(topics)
        
        if time_from:
            params['time_from'] = time_from
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'feed' in data:
                return data['feed']
            else:
                print(f"Error: {data}")
                return []
        
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []
    
    def get_news_api_articles(self, api_key: str, query: str = 'stock market',
                              from_date: str = None,
                              language: str = 'en',
                              sort_by: str = 'publishedAt') -> List[Dict]:
        """
        Get news from NewsAPI.org
        
        Args:
            api_key: NewsAPI key
            query: Search query
            from_date: Start date (YYYY-MM-DD)
            language: Language code
            sort_by: Sort option (publishedAt, relevancy, popularity)
            
        Returns:
            List of articles
        """
        url = "https://newsapi.org/v2/everything"
        
        if not from_date:
            from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        params = {
            'q': query,
            'from': from_date,
            'language': language,
            'sortBy': sort_by,
            'apiKey': api_key
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get('status') == 'ok':
                return data.get('articles', [])
            else:
                print(f"Error: {data}")
                return []
        
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []
    
    def get_rss_feeds(self, feed_urls: List[str]) -> List[Dict]:
        """
        Parse RSS feeds from financial news sites
        
        Args:
            feed_urls: List of RSS feed URLs
            
        Returns:
            List of articles
        """
        articles = []
        
        for url in feed_urls:
            try:
                feed = feedparser.parse(url)
                
                for entry in feed.entries:
                    articles.append({
                        'title': entry.get('title', ''),
                        'description': entry.get('description', ''),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'source': feed.feed.get('title', 'Unknown'),
                    })
            
            except Exception as e:
                print(f"Error parsing feed {url}: {e}")
                continue
        
        return articles
    
    def get_polygon_news(self, api_key: str, ticker: str = None,
                         limit: int = 100) -> List[Dict]:
        """
        Get news from Polygon.io
        
        Args:
            api_key: Polygon API key
            ticker: Specific stock ticker (optional)
            limit: Max number of articles
            
        Returns:
            List of news articles
        """
        if ticker:
            url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}"
        else:
            url = "https://api.polygon.io/v2/reference/news"
        
        params = {
            'limit': limit,
            'apiKey': api_key
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get('status') == 'OK':
                return data.get('results', [])
            else:
                return []
        
        except Exception as e:
            print(f"Error fetching Polygon news: {e}")
            return []

# Common RSS feeds for financial news
FINANCIAL_RSS_FEEDS = [
    'https://feeds.bloomberg.com/markets/news.rss',
    'https://www.cnbc.com/id/100003114/device/rss/rss.html',  # Markets
    'https://www.reuters.com/finance',
    'https://www.marketwatch.com/rss/topstories',
    'https://seekingalpha.com/feed.xml',
    'http://feeds.finance.yahoo.com/rss/2.0/headline',
]

# Example usage
retriever = NewsDataRetriever()

# Get news from RSS feeds (free)
rss_articles = retriever.get_rss_feeds(FINANCIAL_RSS_FEEDS)
print(f"Retrieved {len(rss_articles)} articles from RSS feeds")

# Show sample articles
for article in rss_articles[:3]:
    print(f"\\n{article['source']}: {article['title']}")
    print(f"Published: {article['published']}")
\`\`\`

---

## Entity Recognition and Extraction

### Identifying Companies, People, and Events

\`\`\`python
"""
Extract entities and events from news articles
"""

import anthropic
from typing import List, Dict
import json

class NewsEntityExtractor:
    """
    Extract structured information from news articles
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def extract_entities(self, article_text: str, article_title: str = "") -> Dict:
        """
        Extract entities from news article
        
        Args:
            article_text: Article content
            article_title: Article headline
            
        Returns:
            Structured entity data
        """
        prompt = f"""Extract key entities and information from this financial news article.

Title: {article_title}

Article:
{article_text}

Extract and return as JSON:
{{
  "companies": ["Company names mentioned"],
  "tickers": ["Stock tickers if mentioned"],
  "people": ["Names of people mentioned with their roles"],
  "event_type": "Type of event (earnings, merger, product_launch, etc.)",
  "event_description": "Brief description of main event",
  "financial_metrics": ["Any numbers/metrics mentioned"],
  "locations": ["Geographic locations mentioned"],
  "sentiment_target": "Main subject of the article",
  "urgency": "High/Medium/Low - how time-sensitive is this news"
}}

Return only the JSON."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        
        # Parse JSON from response
        try:
            if "\`\`\`json" in response_text:
                json_str = response_text.split("\`\`\`json")[1].split("\`\`\`")[0].strip()
            elif "\`\`\`" in response_text:
                json_str = response_text.split("\`\`\`")[1].split("\`\`\`")[0].strip()
            else:
                json_str = response_text
            
            return json.loads(json_str)
        except Exception as e:
            print(f"Error parsing JSON: {{e}}")
            return {}
    
    def classify_event_type(self, article_text: str) -> Dict:
        """
        Classify the type of event in the news
        
        Args:
            article_text: Article content
            
        Returns:
            Event classification
        """
        prompt = f"""Classify the main event type in this financial news article.

Categories:
- EARNINGS: Earnings releases, guidance updates
- M&A: Mergers, acquisitions, takeovers
- PRODUCT: Product launches, announcements
- REGULATORY: Regulatory actions, legal issues
- MANAGEMENT: Leadership changes, board changes
- FUNDING: Fundraising, IPO, secondary offerings
- ANALYST: Analyst upgrades/downgrades, price targets
- ECONOMIC: Economic data, policy changes
- CORPORATE: General corporate news
- MARKET: Market-wide news

Article:
{article_text[:2000]}

Return JSON:
{{
  "primary_category": "Main category",
  "secondary_categories": ["Other relevant categories"],
  "market_moving_potential": "High/Medium/Low",
  "explanation": "Brief reason for classification"
}}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return self._parse_json_response(response.content[0].text)
    
    def extract_key_facts(self, article_text: str) -> List[str]:
        """
        Extract key factual statements from article
        
        Args:
            article_text: Article content
            
        Returns:
            List of key facts
        """
        prompt = f"""Extract the top 5 most important factual statements from this article.

Focus on:
- Concrete facts and numbers
- Actions taken or announced
- Timeline information
- Impact statements

Article:
{article_text}

Return as JSON list of facts:
{{"facts": ["Fact 1", "Fact 2", ...]}}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        
        data = self._parse_json_response(response.content[0].text)
        return data.get('facts', [])
    
    def _parse_json_response(self, response_text: str) -> Dict:
        """Helper to parse JSON from LLM response"""
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
extractor = NewsEntityExtractor(api_key="your-key")

sample_article = """
Tesla Inc. (TSLA) shares surged 8% in after-hours trading following the company's third-quarter earnings report, which beat analyst expectations on both revenue and earnings per share. 

CEO Elon Musk announced that the company's new Gigafactory in Texas has reached full production capacity, producing 5,000 vehicles per week. The company also raised its full-year delivery guidance to 1.8 million vehicles, up from previous guidance of 1.6 million.

Despite ongoing supply chain challenges, Tesla reported automotive gross margins of 27.9%, exceeding Wall Street's estimate of 26.2%. The company also confirmed that production of the Cybertruck will begin in late Q1 2024.
"""

# Extract entities
entities = extractor.extract_entities(
    article_text=sample_article,
    article_title="Tesla Beats Earnings Expectations, Raises Guidance"
)

print("Extracted Entities:")
print(json.dumps(entities, indent=2))

# Classify event
event_class = extractor.classify_event_type(sample_article)
print("\\nEvent Classification:")
print(json.dumps(event_class, indent=2))

# Extract key facts
facts = extractor.extract_key_facts(sample_article)
print("\\nKey Facts:")
for i, fact in enumerate(facts, 1):
    print(f"{i}. {fact}")
\`\`\`

---

## Sentiment Analysis at Scale

### Analyzing Sentiment Across News Flow

\`\`\`python
"""
Sentiment analysis for financial news
"""

from typing import List, Dict
from collections import defaultdict
import pandas as pd

class NewsSentimentAnalyzer:
    """
    Analyze sentiment in financial news at scale
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def analyze_article_sentiment(self, article: Dict) -> Dict:
        """
        Analyze sentiment of a single article
        
        Args:
            article: Dict with 'title' and 'content'
            
        Returns:
            Sentiment analysis result
        """
        text = f"{article.get('title', '')}\\n\\n{article.get('content', '')}"
        
        prompt = f"""Analyze the sentiment of this financial news article.

Article:
{text[:4000]}

Provide JSON response:
{{
  "overall_sentiment": "Positive/Neutral/Negative",
  "sentiment_score": -1.0 to 1.0 (negative to positive),
  "confidence": 0.0 to 1.0 (how confident in the sentiment),
  "bullish_signals": ["List of bullish indicators"],
  "bearish_signals": ["List of bearish indicators"],
  "key_sentiment_drivers": ["Main factors driving sentiment"],
  "market_impact_estimate": "High/Medium/Low",
  "affected_companies": ["Company tickers that are affected"]
}}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return self._parse_json(response.content[0].text)
    
    def batch_analyze_sentiment(self, articles: List[Dict], 
                               max_concurrent: int = 5) -> List[Dict]:
        """
        Analyze sentiment for multiple articles concurrently
        
        Args:
            articles: List of article dicts
            max_concurrent: Maximum concurrent API calls
            
        Returns:
            List of sentiment analysis results
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all tasks
            future_to_article = {
                executor.submit(self.analyze_article_sentiment, article): article
                for article in articles
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_article):
                article = future_to_article[future]
                try:
                    sentiment = future.result()
                    results.append({
                        'article': article,
                        'sentiment': sentiment
                    })
                except Exception as e:
                    print(f"Error analyzing article: {e}")
        
        return results
    
    def aggregate_sentiment(self, articles_with_sentiment: List[Dict],
                           ticker: str) -> Dict:
        """
        Aggregate sentiment across multiple articles for a ticker
        
        Args:
            articles_with_sentiment: List of articles with sentiment analysis
            ticker: Stock ticker to focus on
            
        Returns:
            Aggregated sentiment metrics
        """
        # Filter articles mentioning the ticker
        relevant = [
            a for a in articles_with_sentiment
            if ticker in a['sentiment'].get('affected_companies', [])
        ]
        
        if not relevant:
            return {
                'ticker': ticker,
                'article_count': 0,
                'avg_sentiment': 0,
                'sentiment_trend': 'Neutral'
            }
        
        # Calculate aggregates
        sentiments = [a['sentiment']['sentiment_score'] for a in relevant]
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        positive_count = sum(1 for s in sentiments if s > 0.2)
        negative_count = sum(1 for s in sentiments if s < -0.2)
        
        # Determine trend
        if avg_sentiment > 0.3:
            trend = 'Strongly Positive'
        elif avg_sentiment > 0.1:
            trend = 'Positive'
        elif avg_sentiment < -0.3:
            trend = 'Strongly Negative'
        elif avg_sentiment < -0.1:
            trend = 'Negative'
        else:
            trend = 'Neutral'
        
        return {
            'ticker': ticker,
            'article_count': len(relevant),
            'avg_sentiment': avg_sentiment,
            'positive_articles': positive_count,
            'negative_articles': negative_count,
            'neutral_articles': len(relevant) - positive_count - negative_count,
            'sentiment_trend': trend,
            'recent_articles': [
                {
                    'title': a['article'].get('title'),
                    'sentiment': a['sentiment']['overall_sentiment'],
                    'score': a['sentiment']['sentiment_score']
                }
                for a in relevant[:5]
            ]
        }
    
    def detect_sentiment_shift(self, historical_sentiment: List[float],
                              window: int = 10) -> Dict:
        """
        Detect significant shifts in sentiment
        
        Args:
            historical_sentiment: Time series of sentiment scores
            window: Window size for moving average
            
        Returns:
            Shift detection results
        """
        if len(historical_sentiment) < window * 2:
            return {'shift_detected': False}
        
        # Compare recent average to previous average
        recent_avg = sum(historical_sentiment[-window:]) / window
        previous_avg = sum(historical_sentiment[-2*window:-window]) / window
        
        change = recent_avg - previous_avg
        
        # Significant shift threshold
        if abs(change) > 0.3:
            return {
                'shift_detected': True,
                'direction': 'Positive' if change > 0 else 'Negative',
                'magnitude': abs(change),
                'previous_avg': previous_avg,
                'recent_avg': recent_avg,
                'alert_level': 'High' if abs(change) > 0.5 else 'Medium'
            }
        
        return {'shift_detected': False}
    
    def _parse_json(self, response_text: str) -> Dict:
        """Parse JSON from LLM response"""
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
sentiment_analyzer = NewsSentimentAnalyzer(api_key="your-key")

# Analyze single article
sentiment = sentiment_analyzer.analyze_article_sentiment({
    'title': 'Tesla Beats Earnings Expectations',
    'content': sample_article
})

print("Article Sentiment:")
print(json.dumps(sentiment, indent=2))

# Batch analyze multiple articles
articles = [
    {'title': 'Tesla stock surges on strong earnings', 'content': '...'},
    {'title': 'Tesla faces production challenges', 'content': '...'},
    {'title': 'Analysts upgrade Tesla price target', 'content': '...'}
]

batch_results = sentiment_analyzer.batch_analyze_sentiment(articles)
print(f"\\nAnalyzed {len(batch_results)} articles")

# Aggregate sentiment for ticker
agg_sentiment = sentiment_analyzer.aggregate_sentiment(batch_results, 'TSLA')
print("\\nAggregated Sentiment for TSLA:")
print(json.dumps(agg_sentiment, indent=2))
\`\`\`

---

## Real-Time News Processing

### Stream Processing for Breaking News

\`\`\`python
"""
Real-time news processing and signal generation
"""

import threading
import queue
from datetime import datetime
import time

class RealTimeNewsProcessor:
    """
    Process financial news in real-time and generate trading signals
    """
    
    def __init__(self, anthropic_key: str):
        self.entity_extractor = NewsEntityExtractor(anthropic_key)
        self.sentiment_analyzer = NewsSentimentAnalyzer(anthropic_key)
        
        # Queue for incoming news
        self.news_queue = queue.Queue()
        
        # Store recent sentiment by ticker
        self.ticker_sentiment = defaultdict(list)
        
        # Trading signals
        self.signals = []
        
        self.running = False
    
    def start(self):
        """Start the real-time processor"""
        self.running = True
        
        # Start worker threads
        threads = [
            threading.Thread(target=self._news_processor, daemon=True),
            threading.Thread(target=self._signal_generator, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        print("Real-time news processor started")
    
    def stop(self):
        """Stop the processor"""
        self.running = False
    
    def add_news(self, article: Dict):
        """Add news article to processing queue"""
        article['received_at'] = datetime.now()
        self.news_queue.put(article)
    
    def _news_processor(self):
        """Worker thread that processes incoming news"""
        while self.running:
            try:
                # Get article from queue (with timeout)
                article = self.news_queue.get(timeout=1)
                
                # Process article
                self._process_article(article)
                
                self.news_queue.task_done()
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing news: {e}")
    
    def _process_article(self, article: Dict):
        """Process a single article"""
        print(f"\\nProcessing: {article.get('title', 'Untitled')}")
        
        # Extract entities
        entities = self.entity_extractor.extract_entities(
            article_text=article.get('content', ''),
            article_title=article.get('title', '')
        )
        
        # Analyze sentiment
        sentiment = self.sentiment_analyzer.analyze_article_sentiment(article)
        
        # Store analysis
        article['entities'] = entities
        article['sentiment'] = sentiment
        
        # Update ticker sentiment history
        for ticker in entities.get('tickers', []):
            self.ticker_sentiment[ticker].append({
                'timestamp': article['received_at'],
                'score': sentiment.get('sentiment_score', 0),
                'title': article.get('title', '')
            })
        
        # Check for trading signal
        self._check_for_signal(article)
    
    def _check_for_signal(self, article: Dict):
        """Check if article should generate trading signal"""
        entities = article.get('entities', {})
        sentiment = article.get('sentiment', {})
        
        # Signal criteria
        urgency = entities.get('urgency', 'Low')
        market_impact = sentiment.get('market_impact_estimate', 'Low')
        sentiment_score = sentiment.get('sentiment_score', 0)
        
        # Generate signal if high urgency and high impact
        if urgency == 'High' and market_impact in ['High', 'Medium']:
            for ticker in entities.get('tickers', []):
                signal = self._generate_signal(
                    ticker=ticker,
                    article=article,
                    sentiment_score=sentiment_score,
                    urgency=urgency,
                    impact=market_impact
                )
                
                self.signals.append(signal)
                self._emit_signal(signal)
    
    def _generate_signal(self, ticker: str, article: Dict,
                        sentiment_score: float, urgency: str,
                        impact: str) -> Dict:
        """Generate trading signal"""
        
        # Determine action
        if sentiment_score > 0.4:
            action = 'BUY'
            confidence = min(sentiment_score, 0.9)
        elif sentiment_score < -0.4:
            action = 'SELL'
            confidence = min(abs(sentiment_score), 0.9)
        else:
            action = 'HOLD'
            confidence = 0.3
        
        # Adjust confidence based on urgency and impact
        if urgency == 'High':
            confidence *= 1.2
        if impact == 'High':
            confidence *= 1.1
        
        confidence = min(confidence, 1.0)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'action': action,
            'confidence': round(confidence, 2),
            'reason': article['sentiment'].get('key_sentiment_drivers', []),
            'article_title': article.get('title', ''),
            'sentiment_score': sentiment_score,
            'urgency': urgency,
            'impact': impact
        }
    
    def _emit_signal(self, signal: Dict):
        """Emit trading signal"""
        print(f"\\n{'='*60}")
        print(f"TRADING SIGNAL: {signal['action']} {signal['ticker']}")
        print(f"Confidence: {signal['confidence']}")
        print(f"Reason: {', '.join(signal['reason'][:2])}")
        print(f"Article: {signal['article_title']}")
        print(f"{'='*60}")
        
        # In production, this would:
        # 1. Send to trading system
        # 2. Update dashboard
        # 3. Send push notifications
        # 4. Log to database
    
    def _signal_generator(self):
        """Background thread that checks for aggregate signals"""
        while self.running:
            time.sleep(60)  # Check every minute
            
            # Check for sentiment shifts across tracked tickers
            for ticker, history in self.ticker_sentiment.items():
                if len(history) >= 10:
                    scores = [h['score'] for h in history[-10:]]
                    shift = self.sentiment_analyzer.detect_sentiment_shift(scores)
                    
                    if shift.get('shift_detected'):
                        print(f"\\nSENTIMENT SHIFT DETECTED: {ticker}")
                        print(f"Direction: {shift['direction']}")
                        print(f"Magnitude: {shift['magnitude']:.2f}")
    
    def get_ticker_summary(self, ticker: str) -> Dict:
        """Get summary of recent news for a ticker"""
        history = self.ticker_sentiment.get(ticker, [])
        
        if not history:
            return {'ticker': ticker, 'article_count': 0}
        
        recent = history[-10:]
        avg_sentiment = sum(h['score'] for h in recent) / len(recent)
        
        return {
            'ticker': ticker,
            'article_count': len(history),
            'recent_sentiment': avg_sentiment,
            'latest_articles': [
                {'title': h['title'], 'score': h['score']}
                for h in recent[-5:]
            ]
        }

# Example usage
processor = RealTimeNewsProcessor(anthropic_key="your-key")
processor.start()

# Simulate news feed
simulated_news = [
    {
        'title': 'Tesla announces record quarterly deliveries',
        'content': 'Tesla delivered 500,000 vehicles in Q4, beating expectations...',
        'source': 'Reuters'
    },
    {
        'title': 'FDA approves new Tesla drug trial',
        'content': 'The FDA has approved...',  # Wrong Tesla!
        'source': 'Bloomberg'
    },
    {
        'title': 'Tesla faces investigation over safety concerns',
        'content': 'NHTSA announces investigation into Tesla vehicles following...',
        'source': 'CNBC'
    }
]

for article in simulated_news:
    processor.add_news(article)
    time.sleep(5)

# Wait for processing
time.sleep(10)

# Get summary
summary = processor.get_ticker_summary('TSLA')
print("\\nTSLA News Summary:")
print(json.dumps(summary, indent=2))

# Check signals
print(f"\\nTotal signals generated: {len(processor.signals)}")

# processor.stop()
\`\`\`

---

## News Impact Assessment

### Quantifying Market Impact of News

\`\`\`python
"""
Assess and predict market impact of news
"""

import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta

class NewsImpactAssessor:
    """
    Assess potential market impact of news
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def assess_impact(self, article: Dict, entities: Dict, 
                     sentiment: Dict) -> Dict:
        """
        Assess potential market impact of news article
        
        Args:
            article: Article dict
            entities: Extracted entities
            sentiment: Sentiment analysis
            
        Returns:
            Impact assessment
        """
        prompt = f"""Assess the potential market impact of this news.

Title: {article.get('title', '')}
Content: {article.get('content', '')[:2000]}

Entities: {json.dumps(entities)}
Sentiment: {json.dumps(sentiment)}

Provide JSON response:
{{
  "impact_magnitude": "High/Medium/Low",
  "impact_timeframe": "Immediate/Short-term/Long-term",
  "affected_stocks": ["List of stock tickers"],
  "expected_price_movement": {{
    "direction": "Up/Down/Neutral",
    "magnitude_percent": "Estimated % move (e.g., 2-5%)",
    "confidence": "High/Medium/Low"
  }},
  "sector_impact": ["Affected sectors"],
  "contagion_risk": "High/Medium/Low - will this spread to other stocks",
  "volatility_impact": "Expected increase in volatility",
  "key_factors": ["Factors that will determine impact"],
  "similar_historical_events": "Brief comparison to past similar events",
  "action_recommended": "Strong Buy/Buy/Hold/Sell/Strong Sell"
}}"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return self._parse_json(response.content[0].text)
    
    def compare_to_historical(self, current_event: Dict,
                             historical_events: List[Dict]) -> Dict:
        """
        Compare current event to historical similar events
        
        Args:
            current_event: Current news event
            historical_events: List of past similar events with outcomes
            
        Returns:
            Comparison analysis
        """
        historical_summary = "\\n".join([
            f"- {event['description']}: {event['outcome']}"
            for event in historical_events[:5]
        ])
        
        prompt = f"""Compare this current event to historical similar events.

Current Event:
{json.dumps(current_event, indent=2)}

Historical Similar Events:
{historical_summary}

Based on historical patterns:
1. What was the typical market reaction?
2. How long did the impact last?
3. Were there any surprises compared to expectations?
4. What should we expect this time?

Provide analysis with expected outcome."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
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
assessor = NewsImpactAssessor(api_key="your-key")

article = {
    'title': 'Fed Announces Emergency Rate Cut',
    'content': 'The Federal Reserve announced an emergency 50 basis point rate cut...'
}

entities = {
    'event_type': 'ECONOMIC',
    'urgency': 'High'
}

sentiment = {
    'sentiment_score': 0.7,
    'market_impact_estimate': 'High'
}

impact = assessor.assess_impact(article, entities, sentiment)
print("Market Impact Assessment:")
print(json.dumps(impact, indent=2))
\`\`\`

---

## Production Pipeline

### Complete News Analysis System

\`\`\`python
"""
Production news analysis pipeline
"""

import schedule
import sqlite3
from datetime import datetime

class NewsAnalysisPipeline:
    """
    Complete production pipeline for news analysis
    """
    
    def __init__(self, anthropic_key: str, news_api_keys: Dict[str, str],
                 db_path: str = "news_analysis.db"):
        
        self.retriever = NewsDataRetriever()
        self.processor = RealTimeNewsProcessor(anthropic_key)
        self.assessor = NewsImpactAssessor(anthropic_key)
        
        self.api_keys = news_api_keys
        self.db_path = db_path
        
        self._init_database()
    
    def _init_database(self):
        """Initialize database for storing news and analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                content TEXT,
                source TEXT,
                published_date TEXT,
                processed_date TEXT,
                url TEXT UNIQUE
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER,
                ticker TEXT,
                sentiment_score REAL,
                market_impact TEXT,
                FOREIGN KEY (article_id) REFERENCES news_articles(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                ticker TEXT,
                action TEXT,
                confidence REAL,
                reason TEXT,
                article_id INTEGER,
                FOREIGN KEY (article_id) REFERENCES news_articles(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def run_continuous(self, watchlist: List[str]):
        """
        Run continuous news monitoring
        
        Args:
            watchlist: List of tickers to monitor
        """
        # Start real-time processor
        self.processor.start()
        
        # Schedule periodic checks
        schedule.every(5).minutes.do(self._fetch_news, watchlist)
        
        print(f"Monitoring news for: {', '.join(watchlist)}")
        
        while True:
            schedule.run_pending()
            time.sleep(30)
    
    def _fetch_news(self, watchlist: List[str]):
        """Fetch latest news for watchlist"""
        print(f"\\n[{datetime.now()}] Fetching latest news...")
        
        all_articles = []
        
        # Fetch from multiple sources
        for ticker in watchlist:
            # Alpha Vantage
            if 'alpha_vantage' in self.api_keys:
                articles = self.retriever.get_alpha_vantage_news(
                    api_key=self.api_keys['alpha_vantage'],
                    tickers=[ticker],
                    limit=10
                )
                all_articles.extend(articles)
        
        # Process new articles
        for article in all_articles:
            if not self._is_processed(article.get('url')):
                self.processor.add_news(article)
                self._store_article(article)
        
        print(f"Added {len(all_articles)} articles to processing queue")
    
    def _is_processed(self, url: str) -> bool:
        """Check if article already processed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT COUNT(*) FROM news_articles WHERE url = ?",
            (url,)
        )
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
    
    def _store_article(self, article: Dict):
        """Store article in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR IGNORE INTO news_articles
            (title, content, source, published_date, processed_date, url)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            article.get('title'),
            article.get('content'),
            article.get('source'),
            article.get('published_date'),
            datetime.now().isoformat(),
            article.get('url')
        ))
        
        conn.commit()
        conn.close()

# Initialize pipeline
# pipeline = NewsAnalysisPipeline(
#     anthropic_key="your-key",
#     news_api_keys={
#         'alpha_vantage': 'your-av-key',
#         'news_api': 'your-newsapi-key'
#     }
# )

# Run continuous monitoring
# watchlist = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
# pipeline.run_continuous(watchlist)
\`\`\`

---

## Best Practices

1. **Low Latency**: Process news within seconds of publication
2. **Entity Linking**: Accurately link news to affected tickers
3. **Deduplication**: Handle same story from multiple sources
4. **False Positives**: Filter out rumors and unverified news
5. **Context Matters**: Consider broader market context
6. **Historical Patterns**: Learn from past similar events
7. **Sentiment Aggregation**: Don't rely on single article
8. **Source Quality**: Weight high-quality sources more heavily
9. **Event Classification**: Different events have different impacts
10. **Human Oversight**: Critical decisions need human validation

---

## Summary

We covered:
- Connecting to multiple financial news data sources
- Entity and event extraction from articles
- Sentiment analysis at scale with LLMs
- Real-time processing and signal generation
- Market impact assessment
- Building production news analysis pipelines
- Best practices for news-driven trading

Next: Automated report generation for portfolio and market analysis.
`,
};
