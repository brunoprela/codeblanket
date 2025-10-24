export const fundamentalAnalysisML = {
  title: 'Fundamental Analysis with ML',
  id: 'fundamental-analysis-ml',
  content: `
# Fundamental Analysis with ML

## Introduction

Fundamental analysis evaluates intrinsic value using financial statements, earnings, ratios, and macroeconomic factors. Machine learning enhances traditional fundamental analysis by processing vast amounts of data and discovering non-linear relationships.

This section covers:
- Financial statement analysis (10-K, 10-Q parsing)
- Key financial ratios and metrics
- Earnings predictions and surprises
- Alternative data (news sentiment, social media, options flow)
- ML models for fundamental factors
- Combining technical + fundamental signals

---

## Financial Statement Data

\`\`\`python
"""
Financial Statement Analysis with ML
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Get fundamental data for multiple stocks
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']

def get_fundamentals(ticker):
    """Extract fundamental metrics"""
    stock = yf.Ticker(ticker)
    info = stock.info
    
    fundamentals = {
        'ticker': ticker,
        'pe_ratio': info.get('trailingPE', np.nan),
        'forward_pe': info.get('forwardPE', np.nan),
        'peg_ratio': info.get('pegRatio', np.nan),
        'price_to_book': info.get('priceToBook', np.nan),
        'price_to_sales': info.get('priceToSalesTrailing12Months', np.nan),
        'profit_margin': info.get('profitMargins', np.nan),
        'operating_margin': info.get('operatingMargins', np.nan),
        'roe': info.get('returnOnEquity', np.nan),
        'roa': info.get('returnOnAssets', np.nan),
        'debt_to_equity': info.get('debtToEquity', np.nan),
        'current_ratio': info.get('currentRatio', np.nan),
        'quick_ratio': info.get('quickRatio', np.nan),
        'revenue_growth': info.get('revenueGrowth', np.nan),
        'earnings_growth': info.get('earningsGrowth', np.nan),
        'beta': info.get('beta', np.nan),
        'market_cap': info.get('marketCap', np.nan),
        'dividend_yield': info.get('dividendYield', 0),
    }
    
    return fundamentals

# Collect data
fundamental_data = []
for ticker in tickers:
    try:
        data = get_fundamentals(ticker)
        fundamental_data.append(data)
        print(f"✓ {ticker}")
    except Exception as e:
        print(f"✗ {ticker}: {e}")

df_fundamentals = pd.DataFrame(fundamental_data)

print("\\n=== Fundamental Data ===")
print(df_fundamentals)

# Calculate forward returns (target variable)
def get_forward_returns(ticker, days=90):
    """Calculate future returns"""
    prices = yf.download(ticker, period='1y', progress=False)['Close']
    current_price = prices.iloc[-1]
    past_price = prices.iloc[-days-1] if len(prices) > days else prices.iloc[0]
    return (current_price - past_price) / past_price

df_fundamentals['forward_return_90d'] = df_fundamentals['ticker'].apply(
    lambda x: get_forward_returns(x, 90)
)

print("\\n=== With Forward Returns ===")
print(df_fundamentals[['ticker', 'pe_ratio', 'roe', 'revenue_growth', 'forward_return_90d']])
\`\`\`

---

## Financial Ratios & Feature Engineering

\`\`\`python
"""
Financial Ratio Analysis
"""

class FundamentalFeatures:
    """
    Calculate fundamental features
    """
    
    @staticmethod
    def value_score(df):
        """
        Value score: Lower P/E, P/B, P/S = better value
        """
        # Normalize metrics (lower is better for value)
        pe_score = 1 / (1 + df['pe_ratio'].fillna(df['pe_ratio'].median()))
        pb_score = 1 / (1 + df['price_to_book'].fillna(df['price_to_book'].median()))
        ps_score = 1 / (1 + df['price_to_sales'].fillna(df['price_to_sales'].median()))
        
        # Combined value score (0-1, higher = better value)
        value_score = (pe_score + pb_score + ps_score) / 3
        return value_score
    
    @staticmethod
    def quality_score(df):
        """
        Quality score: Higher margins, ROE, ROA = better quality
        """
        # Normalize metrics (higher is better)
        margin_score = df['profit_margin'].fillna(0).clip(0, 1)
        roe_score = df['roe'].fillna(0).clip(0, 1)
        roa_score = df['roa'].fillna(0).clip(0, 1)
        
        # Combined quality score
        quality_score = (margin_score + roe_score + roa_score) / 3
        return quality_score
    
    @staticmethod
    def growth_score(df):
        """
        Growth score: Higher revenue/earnings growth = better
        """
        rev_growth = df['revenue_growth'].fillna(0).clip(-1, 1)
        earn_growth = df['earnings_growth'].fillna(0).clip(-1, 1)
        
        # Normalize to 0-1
        growth_score = ((rev_growth + earn_growth) / 2 + 1) / 2
        return growth_score
    
    @staticmethod
    def financial_health_score(df):
        """
        Financial health: Lower debt, higher current ratio = healthier
        """
        # Debt score (lower debt-to-equity is better)
        debt_score = 1 / (1 + df['debt_to_equity'].fillna(0) / 100)
        
        # Liquidity score (higher current ratio is better, cap at 3)
        liquidity_score = df['current_ratio'].fillna(1).clip(0, 3) / 3
        
        health_score = (debt_score + liquidity_score) / 2
        return health_score

# Calculate composite scores
df_fundamentals['value_score'] = FundamentalFeatures.value_score(df_fundamentals)
df_fundamentals['quality_score'] = FundamentalFeatures.quality_score(df_fundamentals)
df_fundamentals['growth_score'] = FundamentalFeatures.growth_score(df_fundamentals)
df_fundamentals['health_score'] = FundamentalFeatures.financial_health_score(df_fundamentals)

# Overall fundamental score
df_fundamentals['fundamental_score'] = (
    df_fundamentals['value_score'] * 0.25 +
    df_fundamentals['quality_score'] * 0.35 +
    df_fundamentals['growth_score'] * 0.25 +
    df_fundamentals['health_score'] * 0.15
)

print("\\n=== Fundamental Scores ===")
print(df_fundamentals[['ticker', 'value_score', 'quality_score', 'growth_score', 
                       'health_score', 'fundamental_score', 'forward_return_90d']]
      .sort_values('fundamental_score', ascending=False))

# Correlation with forward returns
correlation = df_fundamentals[['fundamental_score', 'forward_return_90d']].corr()
print("\\n=== Correlation with Forward Returns ===")
print(correlation)
\`\`\`

---

## Earnings Analysis

\`\`\`python
"""
Earnings Surprise Analysis
"""

def get_earnings_history(ticker):
    """Get historical earnings data"""
    stock = yf.Ticker(ticker)
    
    # Get earnings history
    earnings = stock.earnings_history
    
    if earnings is None or len(earnings) == 0:
        return None
    
    # Calculate surprise percentage
    earnings['surprise_pct'] = (
        (earnings['epsActual'] - earnings['epsEstimate']) / 
        abs(earnings['epsEstimate']) * 100
    )
    
    return earnings

# Example: AAPL earnings
aapl_earnings = get_earnings_history('AAPL')

if aapl_earnings is not None:
    print("=== AAPL Recent Earnings ===")
    print(aapl_earnings[['quarter', 'epsEstimate', 'epsActual', 'surprise_pct']].tail())
    
    # Average surprise
    avg_surprise = aapl_earnings['surprise_pct'].mean()
    print(f"\\nAverage earnings surprise: {avg_surprise:.2f}%")
    
    # Consistency (% of quarters with positive surprise)
    positive_surprise_pct = (aapl_earnings['surprise_pct'] > 0).mean() * 100
    print(f"Positive surprise rate: {positive_surprise_pct:.1f}%")

# Earnings surprise strategy
def earnings_surprise_feature(ticker, lookback=4):
    """
    Create earnings surprise features
    
    Returns:
        - avg_surprise: Average surprise over last N quarters
        - surprise_consistency: % of positive surprises
        - surprise_trend: Linear trend in surprises
    """
    earnings = get_earnings_history(ticker)
    
    if earnings is None or len(earnings) < lookback:
        return {'avg_surprise': 0, 'consistency': 0.5, 'trend': 0}
    
    recent = earnings.tail(lookback)
    
    avg_surprise = recent['surprise_pct'].mean()
    consistency = (recent['surprise_pct'] > 0).mean()
    
    # Trend: Fit linear regression to surprises
    x = np.arange(len(recent))
    y = recent['surprise_pct'].values
    trend = np.polyfit(x, y, 1)[0]  # Slope
    
    return {
        'avg_surprise': avg_surprise,
        'consistency': consistency,
        'trend': trend
    }

# Calculate for portfolio
earnings_features = []
for ticker in tickers[:3]:  # First 3 for demo
    features = earnings_surprise_feature(ticker)
    features['ticker'] = ticker
    earnings_features.append(features)

df_earnings = pd.DataFrame(earnings_features)
print("\\n=== Earnings Surprise Features ===")
print(df_earnings)
\`\`\`

---

## Alternative Data Integration

\`\`\`python
"""
Alternative Data: Sentiment, Social Media, Options
"""

import requests
from textblob import TextBlob

# 1. News Sentiment
def get_news_sentiment(ticker, days=7):
    """
    Get news sentiment from headlines
    
    Returns:
        Average sentiment score [-1, 1]
    """
    # In production, use NewsAPI, Bloomberg, or specialized services
    # Here: Demo with mock data
    
    # Mock headlines
    headlines = [
        f"{ticker} beats earnings estimates",
        f"{ticker} announces new product launch",
        f"Analysts upgrade {ticker} to buy",
        f"{ticker} faces regulatory scrutiny",
        f"{ticker} reports strong quarterly results"
    ]
    
    sentiments = []
    for headline in headlines:
        blob = TextBlob(headline)
        sentiments.append(blob.sentiment.polarity)
    
    avg_sentiment = np.mean(sentiments)
    return avg_sentiment

# 2. Social Media Mentions
def get_social_mentions(ticker):
    """
    Count social media mentions
    
    In production: Use Reddit API, Twitter API, StockTwits
    """
    # Mock data
    mentions = np.random.randint(100, 5000)
    sentiment = np.random.uniform(-0.2, 0.5)
    
    return {
        'mentions': mentions,
        'sentiment': sentiment,
        'mentions_change': np.random.uniform(-0.3, 0.8)  # vs yesterday
    }

# 3. Options Flow (requires paid data)
def get_options_flow(ticker):
    """
    Analyze unusual options activity
    
    Signals:
        - Large call buying = bullish
        - Large put buying = bearish
        - High put/call ratio = bearish
    """
    # In production: Use Market Chameleon, FlowAlgo, etc.
    # Mock data
    
    put_call_ratio = np.random.uniform(0.5, 1.5)
    unusual_activity = np.random.choice([True, False], p=[0.2, 0.8])
    
    # Interpret
    if put_call_ratio < 0.7:
        signal = 'bullish'  # More calls than puts
    elif put_call_ratio > 1.3:
        signal = 'bearish'  # More puts than calls
    else:
        signal = 'neutral'
    
    return {
        'put_call_ratio': put_call_ratio,
        'unusual_activity': unusual_activity,
        'signal': signal
    }

# Combine all alternative data
def get_alternative_data_score(ticker):
    """Combine all alternative data sources"""
    news_sent = get_news_sentiment(ticker)
    social = get_social_mentions(ticker)
    options = get_options_flow(ticker)
    
    # Normalize and combine
    # News sentiment: [-1, 1] → [0, 1]
    news_score = (news_sent + 1) / 2
    
    # Social sentiment: [-1, 1] → [0, 1], weighted by mentions
    social_score = (social['sentiment'] + 1) / 2
    social_weight = min(social['mentions'] / 1000, 1.0)  # Cap at 1000 mentions
    
    # Options signal: bullish=1, neutral=0.5, bearish=0
    options_score = {'bullish': 1, 'neutral': 0.5, 'bearish': 0}[options['signal']]
    
    # Combined score (weighted average)
    combined_score = (
        news_score * 0.4 +
        social_score * social_weight * 0.3 +
        options_score * 0.3
    )
    
    return {
        'news_sentiment': news_sent,
        'social_mentions': social['mentions'],
        'social_sentiment': social['sentiment'],
        'options_signal': options['signal'],
        'alternative_data_score': combined_score
    }

# Calculate for portfolio
alt_data = []
for ticker in tickers[:3]:
    data = get_alternative_data_score(ticker)
    data['ticker'] = ticker
    alt_data.append(data)

df_alt = pd.DataFrame(alt_data)
print("\\n=== Alternative Data Scores ===")
print(df_alt)
\`\`\`

---

## ML Model for Stock Selection

\`\`\`python
"""
Machine Learning for Fundamental Stock Selection
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Prepare larger dataset (in practice: 500+ stocks)
# For demo, we'll simulate data

def simulate_fundamental_dataset(n_stocks=200):
    """
    Simulate fundamental data for ML training
    """
    np.random.seed(42)
    
    data = {
        'pe_ratio': np.random.uniform(10, 40, n_stocks),
        'roe': np.random.uniform(0.05, 0.30, n_stocks),
        'debt_to_equity': np.random.uniform(0, 200, n_stocks),
        'profit_margin': np.random.uniform(0.05, 0.40, n_stocks),
        'revenue_growth': np.random.uniform(-0.10, 0.50, n_stocks),
        'earnings_growth': np.random.uniform(-0.20, 0.60, n_stocks),
        'peg_ratio': np.random.uniform(0.5, 3.0, n_stocks),
        'current_ratio': np.random.uniform(0.8, 3.0, n_stocks),
    }
    
    df = pd.DataFrame(data)
    
    # Create target: outperform (1) or underperform (0)
    # Simplified logic: Good fundamentals → outperform
    fundamental_score = (
        (40 - df['pe_ratio']) / 30 * 0.2 +  # Lower P/E better
        df['roe'] / 0.30 * 0.25 +  # Higher ROE better
        (200 - df['debt_to_equity']) / 200 * 0.15 +
        df['profit_margin'] / 0.40 * 0.2 +
        (df['revenue_growth'] + 0.10) / 0.60 * 0.2
    )
    
    # Add noise
    fundamental_score += np.random.normal(0, 0.1, n_stocks)
    
    # Binary target: top 40% = 1 (outperform)
    df['outperform'] = (fundamental_score > fundamental_score.quantile(0.6)).astype(int)
    
    return df

# Generate training data
df_train = simulate_fundamental_dataset(200)

print("=== Simulated Training Data ===")
print(df_train.head())
print(f"\\nOutperform distribution: {df_train['outperform'].value_counts()}")

# Prepare features and target
features = ['pe_ratio', 'roe', 'debt_to_equity', 'profit_margin', 
            'revenue_growth', 'earnings_growth', 'peg_ratio', 'current_ratio']

X = df_train[features]
y = df_train['outperform']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Predict
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\\n=== Model Performance ===")
print(f"Accuracy: {accuracy:.3f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\\n=== Feature Importance ===")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Fundamental Feature Importance')
plt.tight_layout()
plt.show()

# Use model for stock selection
# Predict on real data (our initial 7 stocks)
df_real_features = df_fundamentals[features].fillna(df_fundamentals[features].median())
df_fundamentals['ml_score'] = rf_model.predict_proba(df_real_features)[:, 1]
df_fundamentals['ml_prediction'] = rf_model.predict(df_real_features)

print("\\n=== ML Predictions for Portfolio ===")
print(df_fundamentals[['ticker', 'fundamental_score', 'ml_score', 
                       'ml_prediction', 'forward_return_90d']]
      .sort_values('ml_score', ascending=False))
\`\`\`

---

## Key Takeaways

1. **Fundamental Metrics**: P/E, ROE, debt ratios, growth rates
2. **Composite Scores**: Value, Quality, Growth, Health
3. **Earnings Surprises**: Positive surprises correlate with returns
4. **Alternative Data**: News sentiment, social media, options flow
5. **ML Models**: Random Forest for stock selection (60-70% accuracy)
6. **Feature Engineering**: Ratios and interactions matter
7. **Combining Signals**: Fundamental + Technical = better performance

**Best Practices**:
- Use quarterly updates (fundamentals change slowly)
- Combine with technical timing signals
- Rebalance monthly/quarterly (not daily)
- Focus on relative value within sectors
- Validate on out-of-sample data
`,
};
