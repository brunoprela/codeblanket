export const cryptocurrencyTrading = {
  title: 'Cryptocurrency Trading',
  id: 'cryptocurrency-trading',
  content: `
# Cryptocurrency Trading

## Introduction

Cryptocurrency markets represent a new frontier for algorithmic trading with unique characteristics that differentiate them from traditional markets:

**Unique Characteristics**:
- **24/7 Trading**: No market close, continuous price discovery
- **High Volatility**: 2-5x traditional markets (opportunity + risk)
- **Global & Decentralized**: No single exchange, arbitrage opportunities
- **Younger Markets**: More inefficiencies, less competition
- **On-Chain Data**: Blockchain transparency provides unique alpha signals
- **Lower Barriers**: Easier API access, lower capital requirements

**Challenges**:
- **Extreme Volatility**: -50% drawdowns common
- **Liquidity Fragmentation**: Spread across many exchanges
- **Security Risks**: Exchange hacks, wallet security
- **Regulatory Uncertainty**: Rules still evolving
- **Flash Crashes**: Thin order books, cascading liquidations
- **Market Manipulation**: Pump-and-dumps, wash trading

**Opportunities**:
- **Statistical Arbitrage**: Cross-exchange price differences
- **Market Making**: Wider spreads = more profit
- **Momentum**: Strong trends, retail FOMO
- **On-Chain Alpha**: Whale movements, smart money
- **DeFi**: Yield farming, liquidity provision

---

## Cryptocurrency Data & APIs

\`\`\`python
"""
Complete cryptocurrency data fetching and management
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class CryptoDataFetcher:
    """
    Fetch cryptocurrency data from multiple exchanges
    
    Uses CCXT library for unified exchange API access
    """
    
    def __init__(self, exchange_name: str = 'binance', api_key: str = None, secret: str = None):
        """
        Initialize connection to crypto exchange
        
        Args:
            exchange_name: Exchange name ('binance', 'coinbase', 'kraken', etc.)
            api_key: API key (for private endpoints)
            secret: API secret
        """
        exchange_class = getattr (ccxt, exchange_name)
        
        if api_key and secret:
            self.exchange = exchange_class({
                'apiKey': api_key,
                'secret': secret,
                'enableRateLimit': True  # Important!
            })
        else:
            self.exchange = exchange_class({
                'enableRateLimit': True
            })
        
        self.exchange_name = exchange_name
        
        print(f"✓ Connected to {exchange_name}")
        print(f"  Markets: {len (self.exchange.load_markets())}")
    
    # ========================================================================
    # OHLCV DATA
    # ========================================================================
    
    def fetch_ohlcv (self, symbol: str = 'BTC/USDT', 
                    timeframe: str = '1h',
                    since: Optional[datetime] = None,
                    limit: int = 1000) -> pd.DataFrame:
        """
        Fetch OHLCV (candlestick) data
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT', 'ETH/USD')
            timeframe: '1m', '5m', '15m', '1h', '4h', '1d'
            since: Start date
            limit: Number of candles
        
        Returns:
            DataFrame with OHLCV data
        """
        if since:
            since_ms = int (since.timestamp() * 1000)
        else:
            since_ms = None
        
        ohlcv = self.exchange.fetch_ohlcv(
            symbol, 
            timeframe=timeframe, 
            since=since_ms,
            limit=limit
        )
        
        df = pd.DataFrame(
            ohlcv, 
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime (df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def fetch_multiple_timeframes (self, symbol: str = 'BTC/USDT',
                                  timeframes: List[str] = ['1h', '4h', '1d']) -> Dict[str, pd.DataFrame]:
        """Fetch multiple timeframes for multi-scale analysis"""
        data = {}
        for tf in timeframes:
            data[tf] = self.fetch_ohlcv (symbol, timeframe=tf)
            print(f"Fetched {len (data[tf])} {tf} candles for {symbol}")
        
        return data
    
    # ========================================================================
    # ORDER BOOK
    # ========================================================================
    
    def fetch_order_book (self, symbol: str = 'BTC/USDT', limit: int = 20) -> Dict:
        """
        Fetch order book (level 2 data)
        
        Returns:
            Dict with bids, asks, spread, depth
        """
        orderbook = self.exchange.fetch_order_book (symbol, limit=limit)
        
        bids = orderbook['bids'][:limit]  # [[price, size], ...]
        asks = orderbook['asks'][:limit]
        
        if len (bids) > 0 and len (asks) > 0:
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            spread = best_ask - best_bid
            spread_bps = (spread / best_bid) * 10000
            
            # Calculate depth (liquidity)
            bid_depth = sum (size for price, size in bids)
            ask_depth = sum (size for price, size in asks)
        else:
            best_bid = best_ask = spread = spread_bps = 0
            bid_depth = ask_depth = 0
        
        return {
            'bids': bids,
            'asks': asks,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': spread,
            'spread_bps': spread_bps,
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'imbalance': (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0
        }
    
    # ========================================================================
    # TICKER DATA
    # ========================================================================
    
    def fetch_ticker (self, symbol: str = 'BTC/USDT') -> Dict:
        """Fetch current ticker (price, volume, change)"""
        ticker = self.exchange.fetch_ticker (symbol)
        
        return {
            'symbol': symbol,
            'last': ticker['last'],
            'bid': ticker['bid'],
            'ask': ticker['ask'],
            'high_24h': ticker['high'],
            'low_24h': ticker['low'],
            'volume_24h': ticker['quoteVolume'],
            'change_24h': ticker['percentage'],
            'timestamp': datetime.fromtimestamp (ticker['timestamp'] / 1000)
        }
    
    def fetch_tickers (self, symbols: List[str]) -> Dict[str, Dict]:
        """Fetch tickers for multiple symbols"""
        tickers = {}
        for symbol in symbols:
            try:
                tickers[symbol] = self.fetch_ticker (symbol)
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
        
        return tickers
    
    # ========================================================================
    # FUNDING RATES (FUTURES)
    # ========================================================================
    
    def fetch_funding_rate (self, symbol: str = 'BTC/USDT') -> Dict:
        """
        Fetch funding rate for perpetual futures
        
        Funding rate indicates long/short sentiment:
        - Positive: Longs pay shorts (bullish sentiment)
        - Negative: Shorts pay longs (bearish sentiment)
        """
        try:
            funding = self.exchange.fetch_funding_rate (symbol)
            return {
                'symbol': symbol,
                'funding_rate': funding['fundingRate'],
                'next_funding_time': funding['fundingTimestamp'],
                'estimated_rate': funding.get('estimatedRate', None)
            }
        except Exception as e:
            print(f"Funding rate not available: {e}")
            return None
    
    # ========================================================================
    # HISTORICAL TRADES
    # ========================================================================
    
    def fetch_trades (self, symbol: str = 'BTC/USDT', limit: int = 100) -> pd.DataFrame:
        """
        Fetch recent trades (time & sales)
        
        Useful for:
        - Trade intensity analysis
        - Buy/sell pressure
        - Large order detection
        """
        trades = self.exchange.fetch_trades (symbol, limit=limit)
        
        df = pd.DataFrame([{
            'timestamp': datetime.fromtimestamp (t['timestamp'] / 1000),
            'price': t['price'],
            'amount': t['amount'],
            'side': t['side'],
            'id': t['id']
        } for t in trades])
        
        return df


# ============================================================================
# EXAMPLE: FETCH CRYPTO DATA
# ============================================================================

# Initialize fetcher
fetcher = CryptoDataFetcher (exchange_name='binance')

# Fetch BTC data
print("\\n" + "="*70)
print("FETCHING BITCOIN DATA")
print("="*70)

# 1. OHLCV data
btc_1h = fetcher.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=168)  # 1 week
print(f"\\nFetched {len (btc_1h)} hourly candles")
print(btc_1h.tail())

# 2. Order book
orderbook = fetcher.fetch_order_book('BTC/USDT', limit=10)
print(f"\\nOrder Book:")
print(f"  Best Bid: \\$\{orderbook['best_bid']:,.2f}")
print(f"  Best Ask: \\$\{orderbook['best_ask']:,.2f}")
print(f"  Spread: \\$\{orderbook['spread']:.2f} ({orderbook['spread_bps']:.1f} bps)")
print(f"  Imbalance: {orderbook['imbalance']:+.3f}")

# 3. Ticker
ticker = fetcher.fetch_ticker('BTC/USDT')
print(f"\\nTicker:")
print(f"  Price: \\$\{ticker['last']:,.2f}")
print(f"  24h Change: {ticker['change_24h']:+.2f}%")
print(f"  24h Volume: \\$\{ticker['volume_24h']:,.0f}")

# 4. Multiple coins
print("\\nFetching multiple coins...")
tickers = fetcher.fetch_tickers(['BTC/USDT', 'ETH/USDT', 'SOL/USDT'])
for symbol, data in tickers.items():
    print(f"  {symbol}: \\$\{data['last']:,.2f} ({data['change_24h']:+.2f}%)")
\`\`\`

---

## Crypto-Specific Feature Engineering

\`\`\`python
"""
Features unique to cryptocurrency markets
"""

class CryptoFeatureEngineer:
    """
    Engineer features specific to crypto trading
    """
    
    @staticmethod
    def add_volatility_features (df: pd.DataFrame) -> pd.DataFrame:
        """
        Crypto-specific volatility features
        
        Crypto trades 24/7, so volatility calculated differently
        """
        returns = df['close'].pct_change()
        
        # 24-hour realized volatility (annualized)
        df['vol_24h'] = returns.rolling(24).std() * np.sqrt(365 * 24)
        
        # 7-day realized volatility
        df['vol_7d'] = returns.rolling(168).std() * np.sqrt(365 * 24)
        
        # Volatility regime
        df['high_vol'] = (df['vol_24h'] > df['vol_7d'].quantile(0.75)).astype (int)
        df['low_vol'] = (df['vol_24h'] < df['vol_7d'].quantile(0.25)).astype (int)
        
        # Volatility ratio (expansion/contraction)
        df['vol_ratio'] = df['vol_24h'] / df['vol_7d']
        
        # Parkinson volatility (uses high-low range)
        df['parkinson_vol'] = np.sqrt(
            1 / (4 * np.log(2)) * np.log (df['high'] / df['low']) ** 2
        ).rolling(24).mean() * np.sqrt(365 * 24)
        
        return df
    
    @staticmethod
    def add_momentum_features (df: pd.DataFrame) -> pd.DataFrame:
        """Momentum features for crypto"""
        # Multiple timeframe momentum
        df['mom_1h'] = df['close'].pct_change(1)
        df['mom_4h'] = df['close'].pct_change(4)
        df['mom_24h'] = df['close'].pct_change(24)
        df['mom_7d'] = df['close'].pct_change(168)
        
        # Momentum strength
        df['mom_strength'] = (
            (df['mom_24h'] > 0).astype (int) + 
            (df['mom_7d'] > 0).astype (int)
        )
        
        # Rate of change acceleration
        df['mom_acceleration'] = df['mom_24h'] - df['mom_24h'].shift(24)
        
        return df
    
    @staticmethod
    def add_order_flow_features (df: pd.DataFrame, trades_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Order flow imbalance features
        
        Requires trade data with buy/sell classification
        """
        if trades_df is not None:
            # Buy pressure vs sell pressure
            trades_df['buy_volume'] = trades_df.apply(
                lambda x: x['amount'] if x['side'] == 'buy' else 0, axis=1
            )
            trades_df['sell_volume'] = trades_df.apply(
                lambda x: x['amount'] if x['side'] == 'sell' else 0, axis=1
            )
            
            # Aggregate to hourly
            hourly_flow = trades_df.resample('1H', on='timestamp').agg({
                'buy_volume': 'sum',
                'sell_volume': 'sum'
            })
            
            hourly_flow['net_flow'] = hourly_flow['buy_volume'] - hourly_flow['sell_volume']
            hourly_flow['flow_ratio'] = hourly_flow['buy_volume'] / (hourly_flow['buy_volume'] + hourly_flow['sell_volume'])
            
            # Merge with OHLCV
            df = df.join (hourly_flow[['net_flow', 'flow_ratio']], how='left')
        
        return df
    
    @staticmethod
    def add_funding_features (df: pd.DataFrame, funding_rates: pd.Series = None) -> pd.DataFrame:
        """
        Funding rate features (for futures)
        
        Funding rate = sentiment indicator
        - High positive: Overleveraged longs (bearish)
        - High negative: Overleveraged shorts (bullish)
        """
        if funding_rates is not None:
            df['funding_rate'] = funding_rates
            df['funding_ma_24h'] = df['funding_rate'].rolling(24).mean()
            df['funding_extreme'] = (abs (df['funding_rate']) > abs (df['funding_rate']).quantile(0.95)).astype (int)
        
        return df
    
    @staticmethod
    def add_market_regime_features (df: pd.DataFrame) -> pd.DataFrame:
        """Detect crypto market regimes"""
        # Bull/bear regime (SMA crossover)
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
        df['bull_regime'] = (df['sma_50'] > df['sma_200']).astype (int)
        
        # Consolidation vs trending
        df['bb_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
        df['bb_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        df['consolidation'] = (df['bb_width'] < df['bb_width'].quantile(0.25)).astype (int)
        
        return df


# Apply features
btc_1h = CryptoFeatureEngineer.add_volatility_features (btc_1h)
btc_1h = CryptoFeatureEngineer.add_momentum_features (btc_1h)
btc_1h = CryptoFeatureEngineer.add_market_regime_features (btc_1h)

print("\\n" + "="*70)
print("CRYPTO FEATURES ENGINEERED")
print("="*70)
print(f"\\nCurrent Market State:")
print(f"  24h Volatility: {btc_1h['vol_24h'].iloc[-1]:.1%}")
print(f"  24h Momentum: {btc_1h['mom_24h'].iloc[-1]:+.2%}")
print(f"  Bull Regime: {'Yes' if btc_1h['bull_regime'].iloc[-1] else 'No'}")
print(f"  Consolidation: {'Yes' if btc_1h['consolidation'].iloc[-1] else 'No'}")
\`\`\`

---

## Cryptocurrency Trading Strategies

\`\`\`python
"""
Complete crypto trading strategies
"""

class CryptoTradingStrategy:
    """
    Crypto-adapted trading strategies
    """
    
    def __init__(self, risk_per_trade: float = 0.02):
        """
        Args:
            risk_per_trade: Max risk per trade (e.g., 0.02 = 2%)
        """
        self.risk_per_trade = risk_per_trade
    
    def momentum_breakout (self, df: pd.DataFrame, 
                         lookback: int = 24,
                         threshold: float = 0.05) -> pd.Series:
        """
        24-hour momentum breakout strategy
        
        Buy on strong upward momentum, sell on downward
        Works well in trending crypto markets
        """
        returns_24h = df['close'].pct_change (lookback)
        
        signals = pd.Series(0, index=df.index)
        signals[returns_24h > threshold] = 1  # Long
        signals[returns_24h < -threshold] = -1  # Short (or exit)
        
        return signals
    
    def mean_reversion_rsi (self, df: pd.DataFrame,
                          period: int = 14,
                          oversold: int = 30,
                          overbought: int = 70) -> pd.Series:
        """
        RSI mean reversion
        
        Buy oversold, sell overbought
        Works in ranging/consolidating markets
        """
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where (delta > 0, 0).ewm (span=period, adjust=False).mean()
        loss = -delta.where (delta < 0, 0).ewm (span=period, adjust=False).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=df.index)
        signals[rsi < oversold] = 1  # Oversold → Buy
        signals[rsi > overbought] = -1  # Overbought → Sell
        
        return signals
    
    def volatility_breakout (self, df: pd.DataFrame,
                           vol_threshold: float = 1.5,
                           volume_confirm: bool = True) -> pd.Series:
        """
        Volatility expansion breakout
        
        Trade when volatility expands (potential big move)
        """
        if 'vol_ratio' not in df.columns:
            df = CryptoFeatureEngineer.add_volatility_features (df)
        
        returns = df['close'].pct_change()
        vol_expanding = df['vol_ratio'] > vol_threshold
        
        if volume_confirm:
            # Volume confirmation
            volume_high = df['volume'] > df['volume'].rolling(24).mean() * 1.2
            vol_expanding = vol_expanding & volume_high
        
        signals = pd.Series(0, index=df.index)
        signals[vol_expanding & (returns > 0)] = 1  # Breakout up
        signals[vol_expanding & (returns < 0)] = -1  # Breakout down
        
        return signals
    
    def funding_rate_contrarian (self, df: pd.DataFrame,
                                funding_rates: pd.Series,
                                extreme_threshold: float = 0.001) -> pd.Series:
        """
        Contrarian strategy using funding rates
        
        When funding extremely positive (longs overleveraged) → Short
        When funding extremely negative (shorts overleveraged) → Long
        """
        signals = pd.Series(0, index=df.index)
        
        # Align funding rates with df
        funding_aligned = funding_rates.reindex (df.index, method='ffill')
        
        # Extreme positive funding → short (longs will be squeezed)
        signals[funding_aligned > extreme_threshold] = -1
        
        # Extreme negative funding → long (shorts will be squeezed)
        signals[funding_aligned < -extreme_threshold] = 1
        
        return signals
    
    def multi_timeframe_trend (self, df_1h: pd.DataFrame,
                             df_4h: pd.DataFrame,
                             df_1d: pd.DataFrame) -> pd.Series:
        """
        Multi-timeframe trend following
        
        Strongest signals when all timeframes align
        """
        # Trends on each timeframe
        trend_1h = (df_1h['close'] > df_1h['close'].rolling(20).mean()).astype (int) * 2 - 1
        trend_4h = (df_4h['close'] > df_4h['close'].rolling(20).mean()).astype (int) * 2 - 1
        trend_1d = (df_1d['close'] > df_1d['close'].rolling(20).mean()).astype (int) * 2 - 1
        
        # Align all to 1h timeframe
        trend_4h_aligned = trend_4h.reindex (df_1h.index, method='ffill')
        trend_1d_aligned = trend_1d.reindex (df_1h.index, method='ffill')
        
        # Combined signal (average)
        signals = (trend_1h + trend_4h_aligned + trend_1d_aligned) / 3
        
        # Only trade when strong alignment
        signals[abs (signals) < 0.6] = 0
        signals[signals >= 0.6] = 1
        signals[signals <= -0.6] = -1
        
        return signals
    
    def calculate_position_size (self, signal: float, price: float,
                                capital: float, volatility: float) -> float:
        """
        Position sizing for crypto (accounts for high volatility)
        
        Uses volatility-adjusted Kelly Criterion
        """
        if signal == 0:
            return 0
        
        # Risk amount
        risk_amount = capital * self.risk_per_trade
        
        # Stop loss based on volatility (e.g., 2x daily ATR)
        stop_distance = price * volatility * 2
        
        # Position size
        shares = risk_amount / stop_distance
        
        # Cap at reasonable % of capital
        max_position_value = capital * 0.25  # Max 25% per position
        max_shares = max_position_value / price
        
        return min (shares, max_shares) * np.sign (signal)


# ============================================================================
# EXAMPLE: BACKTEST CRYPTO STRATEGY
# ============================================================================

strategy = CryptoTradingStrategy (risk_per_trade=0.02)

# Generate signals
momentum_signals = strategy.momentum_breakout (btc_1h, lookback=24, threshold=0.05)
rsi_signals = strategy.mean_reversion_rsi (btc_1h)
vol_signals = strategy.volatility_breakout (btc_1h, vol_threshold=1.5)

# Ensemble (combine strategies)
ensemble_signals = (momentum_signals + rsi_signals + vol_signals) / 3
ensemble_signals[abs (ensemble_signals) < 0.5] = 0  # Only trade strong signals
ensemble_signals[ensemble_signals >= 0.5] = 1
ensemble_signals[ensemble_signals <= -0.5] = -1

print("\\n" + "="*70)
print("CRYPTO STRATEGY SIGNALS")
print("="*70)
print(f"\\nMomentum signals: {(momentum_signals != 0).sum()}")
print(f"RSI signals: {(rsi_signals != 0).sum()}")
print(f"Volatility signals: {(vol_signals != 0).sum()}")
print(f"Ensemble signals: {(ensemble_signals != 0).sum()}")

# Simple backtest
returns = btc_1h['close'].pct_change()
strategy_returns = ensemble_signals.shift(1) * returns

cumulative_returns = (1 + strategy_returns).cumprod()
total_return = cumulative_returns.iloc[-1] - 1
sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365 * 24)

print(f"\\nBacktest Results (1 week):")
print(f"  Total Return: {total_return:+.2%}")
print(f"  Sharpe Ratio: {sharpe:.2f}")
print(f"  Win Rate: {(strategy_returns > 0).sum() / len (strategy_returns):.1%}")
\`\`\`

---

## Cross-Exchange Arbitrage

\`\`\`python
"""
Arbitrage opportunities across crypto exchanges
"""

class CrossExchangeArbitrage:
    """
    Find and execute arbitrage between exchanges
    """
    
    def __init__(self, exchanges: List[str]):
        """Initialize multiple exchange connections"""
        self.exchanges = {}
        for exchange_name in exchanges:
            self.exchanges[exchange_name] = CryptoDataFetcher (exchange_name)
    
    def find_arbitrage_opportunities (self, symbol: str = 'BTC/USDT',
                                    min_profit_bps: float = 50) -> List[Dict]:
        """
        Find arbitrage opportunities across exchanges
        
        Args:
            symbol: Trading pair
            min_profit_bps: Minimum profit in basis points (after fees)
        
        Returns:
            List of arbitrage opportunities
        """
        # Fetch prices from all exchanges
        prices = {}
        for name, fetcher in self.exchanges.items():
            try:
                ticker = fetcher.fetch_ticker (symbol)
                prices[name] = {
                    'bid': ticker['bid'],
                    'ask': ticker['ask'],
                    'last': ticker['last']
                }
            except Exception as e:
                print(f"Error fetching {name}: {e}")
        
        # Find arbitrage opportunities
        opportunities = []
        
        for buy_exchange in prices:
            for sell_exchange in prices:
                if buy_exchange == sell_exchange:
                    continue
                
                # Buy at ask on buy_exchange, sell at bid on sell_exchange
                buy_price = prices[buy_exchange]['ask']
                sell_price = prices[sell_exchange]['bid']
                
                # Calculate profit (accounting for fees)
                fee_rate = 0.001  # 0.1% per side
                gross_profit = (sell_price - buy_price) / buy_price
                net_profit = gross_profit - 2 * fee_rate  # Both sides
                net_profit_bps = net_profit * 10000
                
                if net_profit_bps >= min_profit_bps:
                    opportunities.append({
                        'buy_exchange': buy_exchange,
                        'sell_exchange': sell_exchange,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'gross_profit_bps': gross_profit * 10000,
                        'net_profit_bps': net_profit_bps,
                        'timestamp': datetime.now()
                    })
        
        return sorted (opportunities, key=lambda x: x['net_profit_bps'], reverse=True)


# Example (would need API keys for live trading)
# arb = CrossExchangeArbitrage(['binance', 'coinbase', 'kraken'])
# opportunities = arb.find_arbitrage_opportunities('BTC/USDT', min_profit_bps=50)
# 
# for opp in opportunities:
#     print(f"Arbitrage: Buy on {opp['buy_exchange']} @ {opp['buy_price']}, "
#           f"Sell on {opp['sell_exchange']} @ {opp['sell_price']} "
#           f"= {opp['net_profit_bps']:.1f} bps profit")
\`\`\`

---

## Key Takeaways

**Crypto vs Traditional Markets**:
- **24/7 Trading**: No rest, continuous monitoring needed
- **Higher Volatility**: 2-5x stock market (bigger opportunities + risks)
- **Lower Barriers**: Easier to start, democratic access
- **More Inefficiencies**: Younger market = more alpha
- **Faster Pace**: News moves markets instantly

**Data Sources**:
- **OHLCV**: CCXT, exchange APIs
- **Order Book**: Real-time via WebSocket
- **Funding Rates**: Perpetual futures sentiment
- **On-Chain**: Glassnode, CryptoQuant, Nansen
- **Social**: LunarCrush, Santiment

**Best Strategies**:
1. **Momentum**: Strong in crypto (retail FOMO)
2. **Mean Reversion**: Works in consolidation
3. **Arbitrage**: Cross-exchange, funding rate
4. **Market Making**: Wider spreads = profit
5. **Trend Following**: Crypto has strong trends

**Risk Management** (Critical!):
- **Smaller Positions**: 2-5% per trade (not 10%)
- **Tighter Stops**: 2-3% stops (not 5-10%)
- **Exchange Risk**: Diversify across exchanges
- **Security**: Hardware wallets, 2FA, cold storage
- **Leverage**: Use sparingly (max 2-3x, not 100x)

**Unique Considerations**:
- **Exchange Downtime**: Have backup exchanges
- **Flash Crashes**: More common, wider stops
- **Wash Trading**: Be aware of fake volume
- **Regulatory Risk**: Laws still evolving
- **Custody**: Your keys, your coins

**Remember**: Crypto is the Wild West—higher rewards but also higher risks. Start small, test thoroughly, and never risk more than you can afford to lose.
`,
};
