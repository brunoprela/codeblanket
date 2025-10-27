export const meanReversionStrategiesQuiz = [
  {
    id: 'ats-3-1-q-1',
    question:
      "Design a complete Bollinger Bands mean reversion system for trading SPY with $500K capital. Include: (1) Entry rules with filters, (2) Position sizing with scaling (add to losers), (3) Stop loss at what level, (4) When to shut down strategy (regime change detection). Why is 'averaging into losers' dangerous for mean reversion but acceptable?",
    sampleAnswer: `**Complete Bollinger Bands Mean Reversion System for SPY:**

**System Specifications:**
- Asset: SPY (S&P 500 ETF)
- Capital: $500,000
- Holding Period: 2-10 days (mean reversion timeframe)
- Max Risk: 2% per position ($10,000)
- Target: 15-20% annual return, 60% win rate

**1. Entry Rules with Filters:**

\`\`\`python
class SPYBollingerMeanReversion:
    """
    Production-ready Bollinger Bands mean reversion for SPY
    """
    
    def __init__(self, capital: float = 500_000):
        self.capital = capital
        self.max_risk_per_trade = 0.02  # 2%
        self.max_position_size = 0.20  # Max 20% in single position
        
        # Bollinger Bands parameters
        self.bb_period = 20
        self.bb_std = 2.0
        
        # Filters
        self.min_volume = 70_000_000  # SPY typically 100M+ daily
        self.max_adx = 25  # Only trade when not trending
        self.max_vix = 30  # Avoid extreme volatility
        
    def check_entry_conditions(self, data: pd.DataFrame) -> dict:
        """
        Comprehensive entry filter checklist
        
        Returns dict with {can_enter: bool, reasons: list}
        """
        latest = data.iloc[-1]
        reasons = []
        
        # Calculate Bollinger Bands
        ma = data['close'].rolling(self.bb_period).mean().iloc[-1]
        std = data['close'].rolling(self.bb_period).std().iloc[-1]
        lower_band = ma - (self.bb_std * std)
        upper_band = ma + (self.bb_std * std)
        
        # Primary signal: Price < lower band (oversold)
        if latest['close'] >= lower_band:
            reasons.append("NOT_OVERSOLD")
            return {'can_enter': False, 'reasons': reasons}
        
        # Filter 1: Volume confirmation (avoid illiquid days)
        avg_volume = data['volume'].rolling(20).mean().iloc[-1]
        if latest['volume'] < avg_volume * 0.8:
            reasons.append("LOW_VOLUME")
            return {'can_enter': False, 'reasons': reasons}
        
        # Filter 2: ADX < 25 (ensure mean-reverting regime)
        adx = self.calculate_adx(data).iloc[-1]
        if adx > self.max_adx:
            reasons.append(f"TRENDING_MARKET_ADX_{adx:.1f}")
            return {'can_enter': False, 'reasons': reasons}
        
        # Filter 3: VIX not extreme (avoid crisis periods)
        vix = self.get_vix()
        if vix > self.max_vix:
            reasons.append(f"HIGH_VIX_{vix:.1f}")
            return {'can_enter': False, 'reasons': reasons}
        
        # Filter 4: Not at 52-week lows (avoid falling knives)
        low_52w = data['low'].rolling(252).min().iloc[-1]
        if latest['close'] <= low_52w * 1.02:  # Within 2% of 52w low
            reasons.append("NEAR_52W_LOW")
            return {'can_enter': False, 'reasons': reasons}
        
        # Filter 5: Bandwidth not at extreme (avoid volatility explosions)
        bandwidth = (upper_band - lower_band) / ma
        bandwidth_percentile = self.calculate_percentile(
            data['close'].rolling(self.bb_period).std() / ma,
            lookback=100
        )
        
        if bandwidth_percentile > 90:  # Top 10% volatility
            reasons.append("EXTREME_VOLATILITY")
            return {'can_enter': False, 'reasons': reasons}
        
        # All filters passed
        reasons.append("ALL_FILTERS_PASSED")
        return {
            'can_enter': True,
            'reasons': reasons,
            'entry_price': latest['close'],
            'lower_band': lower_band,
            'upper_band': upper_band,
            'zscore': (latest['close'] - ma) / std
        }
\`\`\`

**2. Position Sizing with Scaling (Averaging Down):**

\`\`\`python
def calculate_initial_position_size(self, entry_price: float, 
                                   stop_loss: float) -> int:
    """
    Initial position size based on 2% risk
    
    This is 1/3 of total position (will add 2 more if price drops)
    """
    risk_amount = self.capital * self.max_risk_per_trade
    risk_per_share = abs(entry_price - stop_loss)
    
    # Size for 2% risk
    full_size = int(risk_amount / risk_per_share)
    
    # Start with 1/3 position (plan to add 2x more if needed)
    initial_size = full_size // 3
    
    return initial_size

def scaling_plan(self, entry_price: float, atr: float) -> dict:
    """
    Plan for adding to position as it moves against us
    
    Mean reversion logic: Buy more as it gets cheaper
    (Dangerous! Only works if actually mean-reverting)
    
    Args:
        entry_price: Initial entry price
        atr: Average True Range
        
    Returns:
        Scaling plan with add levels and sizes
    """
    return {
        'entry_1': {
            'price': entry_price,
            'size': self.calculate_initial_position_size(entry_price, 
                                                         entry_price - 2*atr),
            'rationale': 'Initial position at lower Bollinger Band'
        },
        'entry_2': {
            'price': entry_price - 0.5*atr,  # Add if drops 0.5× ATR
            'size': self.calculate_initial_position_size(entry_price, 
                                                         entry_price - 2*atr),
            'rationale': 'Add as more oversold (averaging down)'
        },
        'entry_3': {
            'price': entry_price - 1.0*atr,  # Add if drops 1× ATR
            'size': self.calculate_initial_position_size(entry_price, 
                                                         entry_price - 2*atr),
            'rationale': 'Final add (maximum conviction)'
        },
        'max_position': self.calculate_initial_position_size(entry_price,
                                                             entry_price - 2*atr) * 3,
        'avg_entry': None,  # Calculate after all adds
        'total_risk': self.capital * self.max_risk_per_trade
    }
\`\`\`

**3. Stop Loss Levels:**

\`\`\`python
def calculate_stop_loss(self, entry_prices: list, sizes: list, 
                       atr: float) -> dict:
    """
    Stop loss for scaled position
    
    Rule: Stop at 2× ATR from average entry price
    
    Args:
        entry_prices: List of entry prices (if scaled in)
        sizes: List of position sizes
        atr: Average True Range
        
    Returns:
        Stop loss details
    """
    # Calculate weighted average entry
    total_cost = sum(p * s for p, s in zip(entry_prices, sizes))
    total_shares = sum(sizes)
    avg_entry = total_cost / total_shares
    
    # Stop at 2× ATR from average
    stop_price = avg_entry - 2*atr
    
    # Calculate maximum loss
    max_loss_dollars = (avg_entry - stop_price) * total_shares
    max_loss_pct = max_loss_dollars / self.capital
    
    return {
        'avg_entry': avg_entry,
        'stop_price': stop_price,
        'stop_distance': avg_entry - stop_price,
        'stop_pct': (avg_entry - stop_price) / avg_entry,
        'max_loss_dollars': max_loss_dollars,
        'max_loss_pct': max_loss_pct,
        'acceptable': max_loss_pct <= self.max_risk_per_trade
    }
\`\`\`

**4. Regime Change Detection (When to Shut Down):**

\`\`\`python
def detect_regime_change(self, data: pd.DataFrame, 
                        pnl_history: pd.Series) -> dict:
    """
    Detect when market regime changes from mean-reverting to trending
    
    Shut down strategy to avoid catastrophic losses
    
    Signals:
    1. ADX crosses above 30 (strong trend developing)
    2. Consecutive losses (5+ losing trades)
    3. Maximum drawdown exceeded (10%)
    4. VIX spike (>40, crisis mode)
    5. Hurst exponent > 0.5 (trending)
    """
    latest = data.iloc[-1]
    
    shutdown_signals = {}
    
    # Signal 1: ADX > 30 (strong trend)
    adx = self.calculate_adx(data).iloc[-1]
    if adx > 30:
        shutdown_signals['trending'] = {
            'triggered': True,
            'adx': adx,
            'message': f'ADX {adx:.1f} indicates strong trend - mean reversion unsafe'
        }
    
    # Signal 2: Consecutive losses (strategy stopped working)
    recent_trades = pnl_history.iloc[-10:]
    consecutive_losses = 0
    for pnl in reversed(recent_trades):
        if pnl < 0:
            consecutive_losses += 1
        else:
            break
    
    if consecutive_losses >= 5:
        shutdown_signals['consecutive_losses'] = {
            'triggered': True,
            'count': consecutive_losses,
            'message': f'{consecutive_losses} consecutive losses - edge disappeared'
        }
    
    # Signal 3: Maximum drawdown
    equity_curve = self.calculate_equity_curve(pnl_history)
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = drawdown.min()
    
    if max_dd < -0.10:  # -10% drawdown
        shutdown_signals['max_drawdown'] = {
            'triggered': True,
            'drawdown': max_dd,
            'message': f'{max_dd:.1%} drawdown exceeded threshold - stop trading'
        }
    
    # Signal 4: VIX spike
    vix = self.get_vix()
    if vix > 40:
        shutdown_signals['vix_spike'] = {
            'triggered': True,
            'vix': vix,
            'message': f'VIX {vix:.1f} indicates crisis - extreme volatility regime'
        }
    
    # Signal 5: Hurst exponent trending
    hurst = self.calculate_hurst_exponent(data['close'])
    if hurst > 0.55:
        shutdown_signals['hurst_trending'] = {
            'triggered': True,
            'hurst': hurst,
            'message': f'Hurst {hurst:.2f} > 0.5 - market transitioned to trending'
        }
    
    return {
        'should_shutdown': len(shutdown_signals) > 0,
        'signals': shutdown_signals,
        'recommendation': 'STOP TRADING' if shutdown_signals else 'CONTINUE'
    }
\`\`\`

**Why Averaging Into Losers is Dangerous (But Sometimes Acceptable):**

**Dangerous Because:**
1. **Trend Emergence**: What looks "oversold" gets more oversold in a trend
2. **Doubling Down**: Each add increases risk exposure
3. **Catastrophic Losses**: 3x position at entry, 2× ATR stop = 6% loss (vs 2% planned)
4. **Psychological**: Hard to admit you're wrong and take loss

**Acceptable When:**
1. **Statistical Edge**: Proven mean reversion (Hurst < 0.5, half-life < 30 days)
2. **Strict Limits**: Max 3 adds, then hard stop
3. **Small Sizes**: Each add is small (1/3 position)
4. **Filters Active**: Only in range-bound regime (ADX < 25)
5. **Diversified**: Multiple positions spread risk

**Comparison: Trend Following vs Mean Reversion Scaling**

| Aspect | Trend Following | Mean Reversion |
|--------|----------------|----------------|
| Scaling | Pyramid into WINNERS | Average into LOSERS |
| Logic | Add to profitable positions | Add as price improves (cheaper) |
| Risk | Each add is NEW risk | Each add increases SAME trade risk |
| Stop | Raised with each add | Fixed from average entry |
| Psychology | Feels great (adding to winner) | Feels terrible (adding to loser) |

**Example Scenario:**

\`\`\`
SPY Entry 1: $420 (100 shares, price at lower BB)
SPY Entry 2: $418 (100 shares, dropped 0.5× ATR)
SPY Entry 3: $416 (100 shares, dropped 1× ATR)

Average Entry: $418
Stop Loss: $414 (2× ATR from average)
Total Risk: ($418 - $414) × 300 shares = $1,200 (0.24% of $500K capital)

If stop hit: Lost $1,200 (acceptable)
If reverts to mean ($425): Gained ($425-$418) × 300 = $2,100 (0.42% gain)
Risk/Reward: 1:1.75 (acceptable)
\`\`\`

**Key Rule:**
Only average down if you have STATISTICAL EVIDENCE of mean reversion and STRICT RISK LIMITS. Otherwise, averaging down is a recipe for disaster (see: every blown-up trader ever).`,
    keyPoints: [
      'Entry filters: price < lower BB, ADX < 25, VIX < 30, volume > average, not at 52w lows',
      'Scaling: 3 entries (initial, -0.5× ATR, -1× ATR), each 1/3 size, stop 2× ATR from average entry',
      'Shutdown triggers: ADX > 30, 5+ consecutive losses, 10%+ drawdown, VIX > 40, Hurst > 0.55',
      'Averaging down dangerous: only acceptable with statistical edge, strict limits (max 3 adds), small sizes',
      'Key difference: trend following pyramids into winners, mean reversion averages into losers (riskier)',
    ],
  },
  {
    id: 'ats-3-1-q-2',
    question:
      'Calculate and interpret: (1) Hurst exponent for SPY (2010-2024), (2) Half-life of mean reversion, (3) What these metrics tell you about trading strategy choice. If Hurst = 0.52 and half-life = 45 days, would you use trend following or mean reversion? Why?',
    sampleAnswer: `**Statistical Analysis of SPY Mean Reversion Properties:**

**1. Hurst Exponent Calculation for SPY (2010-2024):**

\`\`\`python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def calculate_hurst_exponent(prices: pd.Series, max_lag: int = 100) -> dict:
    """
    Calculate Hurst Exponent using R/S analysis
    
    Interpretation:
    H < 0.5: Mean reverting (anti-persistent)
    H = 0.5: Random walk (geometric Brownian motion)
    H > 0.5: Trending (persistent)
    
    Returns detailed analysis
    """
    lags = range(2, max_lag)
    
    # Calculate standard deviation of log differences at each lag
    tau = []
    for lag in lags:
        # Log price differences
        diffs = np.subtract(prices[lag:], prices[:-lag])
        tau.append(np.std(diffs))
    
    # Fit power law: tau ~ lag^H
    # log(tau) = H * log(lag) + const
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst_exponent = poly[0]
    
    # Calculate confidence interval (bootstrap)
    hurst_estimates = []
    for _ in range(1000):
        # Resample prices
        sample = prices.sample(frac=1.0, replace=True)
        sample_tau = [np.std(np.subtract(sample[lag:], sample[:-lag])) 
                     for lag in lags]
        sample_poly = np.polyfit(np.log(lags), np.log(sample_tau), 1)
        hurst_estimates.append(sample_poly[0])
    
    confidence_interval = (
        np.percentile(hurst_estimates, 2.5),
        np.percentile(hurst_estimates, 97.5)
    )
    
    return {
        'hurst_exponent': hurst_exponent,
        'confidence_interval_95': confidence_interval,
        'interpretation': classify_hurst(hurst_exponent),
        'p_value': calculate_hurst_significance(hurst_exponent, len(prices))
    }

def classify_hurst(hurst: float) -> str:
    """Classify market behavior by Hurst exponent"""
    if hurst < 0.4:
        return "STRONGLY_MEAN_REVERTING"
    elif hurst < 0.5:
        return "MEAN_REVERTING"
    elif hurst < 0.55:
        return "NEAR_RANDOM_WALK"
    elif hurst < 0.6:
        return "TRENDING"
    else:
        return "STRONGLY_TRENDING"

# Example: Calculate for SPY
# (Assuming we have SPY daily data 2010-2024)
spy_hurst = calculate_hurst_exponent(spy_prices)

print(f"SPY Hurst Exponent: {spy_hurst['hurst_exponent']:.3f}")
print(f"95% CI: {spy_hurst['confidence_interval_95']}")
print(f"Interpretation: {spy_hurst['interpretation']}")
\`\`\`

**Expected Results for SPY (2010-2024):**
- **Hurst Exponent**: ~0.52 (slightly above 0.5)
- **95% Confidence Interval**: (0.48, 0.56)
- **Interpretation**: NEAR_RANDOM_WALK with slight trending bias

**What This Tells Us:**
- SPY exhibits weak momentum (H > 0.5)
- NOT strongly mean-reverting (H not < 0.4)
- Close to random walk (efficient market)
- Trend following has slight edge over mean reversion

**2. Half-Life of Mean Reversion:**

\`\`\`python
def calculate_half_life(prices: pd.Series) -> dict:
    """
    Calculate half-life of mean reversion
    
    Uses Ornstein-Uhlenbeck process:
    dX = θ(μ - X)dt + σdW
    
    Half-life = ln(2) / θ
    
    Returns:
        Half-life in days (for daily data)
    """
    # Log prices
    log_prices = np.log(prices)
    
    # Lagged prices and differences
    lagged = log_prices.shift(1).dropna()
    delta = log_prices.diff().dropna()
    
    # Align series
    lagged = lagged[delta.index]
    
    # Regression: Δp(t) = λ(μ - p(t-1)) + ε
    # Rearranging: Δp(t) = α + β*p(t-1) + ε
    # where β = -λ (mean reversion speed)
    
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(lagged.values.reshape(-1, 1), delta.values)
    
    beta = model.coef_[0]
    lambda_param = -beta  # Mean reversion speed
    
    # Calculate half-life
    if lambda_param <= 0:
        half_life = np.inf  # No mean reversion
        is_stationary = False
    else:
        half_life = np.log(2) / lambda_param
        is_stationary = True
    
    # Statistical significance test
    # Use t-statistic for beta coefficient
    from scipy import stats
    predictions = model.predict(lagged.values.reshape(-1, 1))
    residuals = delta.values - predictions
    mse = np.mean(residuals**2)
    se_beta = np.sqrt(mse / np.sum((lagged - lagged.mean())**2))
    t_stat = beta / se_beta
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(lagged)-2))
    
    return {
        'half_life_days': half_life,
        'lambda': lambda_param,
        'is_stationary': is_stationary,
        'p_value': p_value,
        'interpretation': interpret_half_life(half_life),
        'trading_implication': half_life_to_strategy(half_life)
    }

def interpret_half_life(half_life: float) -> str:
    """Interpret half-life value"""
    if half_life < 5:
        return "VERY_FAST_REVERSION (< 1 week)"
    elif half_life < 20:
        return "FAST_REVERSION (1-4 weeks)"
    elif half_life < 60:
        return "MODERATE_REVERSION (1-3 months)"
    elif half_life < 252:
        return "SLOW_REVERSION (3-12 months)"
    else:
        return "NO_REVERSION (>1 year or infinite)"

def half_life_to_strategy(half_life: float) -> str:
    """Recommend strategy based on half-life"""
    if half_life < 20:
        return "Mean reversion (hold 0.5× half-life)"
    elif half_life < 60:
        return "Mean reversion possible (hold 0.5-1× half-life)"
    else:
        return "Trend following better (reversion too slow)"

# Calculate for SPY
spy_half_life = calculate_half_life(spy_prices)

print(f"\\nSPY Half-Life: {spy_half_life['half_life_days']:.1f} days")
print(f"Interpretation: {spy_half_life['interpretation']}")
print(f"Strategy Recommendation: {spy_half_life['trading_implication']}")
\`\`\`

**Expected Results for SPY:**
- **Half-Life**: ~45 days (moderate mean reversion)
- **Lambda**: ~0.0154 (mean reversion speed)
- **Interpretation**: MODERATE_REVERSION (1-3 months)

**What This Tells Us:**
- Deviations from mean persist for ~45 days
- Takes 1.5 months for price to revert halfway
- Too slow for short-term mean reversion (need < 20 days)
- Better suited for longer-term strategies

**3. Given Hurst = 0.52, Half-Life = 45 Days: Which Strategy?**

**Analysis:**

\`\`\`python
def strategy_recommendation(hurst: float, half_life: float) -> dict:
    """
    Comprehensive strategy recommendation
    
    Args:
        hurst: Hurst exponent
        half_life: Half-life in days
        
    Returns:
        Detailed recommendation
    """
    # Score each strategy (0-100)
    scores = {
        'trend_following': 0,
        'mean_reversion': 0,
        'market_making': 0,
        'buy_hold': 0
    }
    
    # Hurst analysis
    if hurst < 0.45:
        scores['mean_reversion'] += 50
        scores['market_making'] += 30
    elif hurst < 0.5:
        scores['mean_reversion'] += 30
        scores['buy_hold'] += 20
    elif hurst < 0.55:
        scores['buy_hold'] += 30
        scores['trend_following'] += 20
        scores['mean_reversion'] += 10
    elif hurst < 0.6:
        scores['trend_following'] += 40
        scores['buy_hold'] += 20
    else:
        scores['trend_following'] += 50
    
    # Half-life analysis
    if half_life < 10:
        scores['mean_reversion'] += 40
        scores['market_making'] += 30
    elif half_life < 30:
        scores['mean_reversion'] += 30
    elif half_life < 60:
        scores['mean_reversion'] += 10
        scores['trend_following'] += 10
    else:
        scores['trend_following'] += 20
    
    # Normalize scores
    total = sum(scores.values())
    probabilities = {k: v/total for k, v in scores.items()}
    
    # Determine best strategy
    best_strategy = max(scores, key=scores.get)
    
    return {
        'recommended_strategy': best_strategy,
        'confidence': scores[best_strategy] / 100,
        'scores': scores,
        'probabilities': probabilities,
        'reasoning': generate_reasoning(hurst, half_life, best_strategy)
    }

def generate_reasoning(hurst: float, half_life: float, strategy: str) -> str:
    """Generate explanation for recommendation"""
    reasoning = []
    
    # Hurst reasoning
    if hurst < 0.5:
        reasoning.append(f"Hurst {hurst:.2f} < 0.5 suggests mean reversion")
    elif hurst > 0.55:
        reasoning.append(f"Hurst {hurst:.2f} > 0.55 suggests trending behavior")
    else:
        reasoning.append(f"Hurst {hurst:.2f} ≈ 0.5 suggests near-random walk")
    
    # Half-life reasoning
    if half_life < 30:
        reasoning.append(f"Half-life {half_life:.0f} days is fast enough for mean reversion")
    elif half_life < 60:
        reasoning.append(f"Half-life {half_life:.0f} days is moderate (borderline)")
    else:
        reasoning.append(f"Half-life {half_life:.0f} days too slow for mean reversion")
    
    # Strategy-specific advice
    if strategy == 'trend_following':
        reasoning.append("Use trend following: capture momentum with trailing stops")
    elif strategy == 'mean_reversion':
        reasoning.append("Use mean reversion: trade oversold/overbought with stops")
    elif strategy == 'buy_hold':
        reasoning.append("Market efficient: buy-and-hold or index investing best")
    
    return " | ".join(reasoning)

# Analyze H=0.52, HL=45
recommendation = strategy_recommendation(hurst=0.52, half_life=45)

print("\\n=== Strategy Recommendation ===")
print(f"Best Strategy: {recommendation['recommended_strategy']}")
print(f"Confidence: {recommendation['confidence']:.0%}")
print(f"\\nScores:")
for strat, score in recommendation['scores'].items():
    print(f"  {strat}: {score}")
print(f"\\nReasoning: {recommendation['reasoning']}")
\`\`\`

**VERDICT: Hurst = 0.52, Half-Life = 45 Days**

**Recommendation: TREND FOLLOWING (with caveat)**

**Why:**
1. **Hurst 0.52 > 0.5**: Slight trending bias (momentum exists)
2. **Half-Life 45 days**: Too slow for short-term mean reversion
   - Mean reversion works best when HL < 20 days
   - 45 days means holding 20-30 days for reversion (too long)
3. **Near Random Walk**: Market quite efficient
   - Traditional trend following also challenging
   - Consider longer-term trend following (50-200 day MAs)

**Optimal Strategy:**
- **Long-Term Trend Following**: 50/200-day MA crossover
- **Hold Period**: 2-6 months (match half-life)
- **Avoid**: Short-term mean reversion (HL too long)
- **Alternative**: Buy-and-hold (market efficient, H ≈ 0.5)

**Expected Performance:**
- **Trend Following (50/200 MA)**: 8-12% annual, Sharpe 0.8-1.2
- **Mean Reversion (BB)**: 5-8% annual, Sharpe 0.5-0.8 (not optimal)
- **Buy-Hold**: 10% annual (SPY historical), Sharpe 0.7

**Key Insight:**
When Hurst ≈ 0.5 and half-life > 30 days, market is quite efficient. Neither pure trend nor pure mean reversion has strong edge. Best approach:
1. Longer timeframes (months, not days)
2. Combine both (hybrid strategy)
3. Or just buy-and-hold index (lowest cost)`,
    keyPoints: [
      'Hurst exponent: <0.5 mean-reverting, =0.5 random walk, >0.5 trending; SPY typically 0.52 (slight trend bias)',
      'Half-life: time to revert halfway to mean; <20 days good for mean reversion, >60 days better for trend following',
      'H=0.52, HL=45: recommend long-term trend following (50/200 MA), hold 2-6 months, avoid short-term mean reversion',
      'Near random walk (H≈0.5) means market efficient: neither strategy has strong edge, consider buy-and-hold',
      'Optimal mean reversion needs H<0.45 AND half-life <20 days; otherwise trending or buy-hold better',
    ],
  },
  {
    id: 'ats-3-1-q-3',
    question:
      'Mean reversion strategies had Sharpe ratios of 2-3 in the 2010-2019 period but dropped to 0.5-1.0 in 2020-2024. Analyze: (1) Why performance degraded (regime change, crowding, volatility), (2) Specific changes in 2020-2024 that broke mean reversion, (3) Adaptations needed for modern mean reversion.',
    sampleAnswer: `**Mean Reversion Performance Collapse: 2010-2019 vs 2020-2024**

**Historical Performance:**
- **2010-2019**: Sharpe 2-3, steady returns, low drawdowns
- **2020-2024**: Sharpe 0.5-1.0, inconsistent, large drawdowns

**1. Why Performance Degraded:**

**A. Regime Change (Most Important)**

\`\`\`python
class RegimeAnalysis:
    """
    Analyze market regime changes 2010-2024
    """
    
    def compare_regimes(self):
        """
        Compare key metrics across periods
        """
        regimes = {
            '2010-2019': {
                'avg_vix': 14.5,
                'volatility': 0.10,  # 10% annual vol
                'fed_policy': 'QE + Zero rates',
                'trend_strength': 'Low (range-bound)',
                'mean_reversion_frequency': 'High (weekly)',
                'sharpe_mean_reversion': 2.5
            },
            '2020-2024': {
                'avg_vix': 19.5,
                'volatility': 0.18,  # 18% annual vol
                'fed_policy': 'Tightening + Rate hikes',
                'trend_strength': 'High (volatile trends)',
                'mean_reversion_frequency': 'Low (monthly)',
                'sharpe_mean_reversion': 0.8
            }
        }
        
        return regimes
    
    def analyze_volatility_regime_shift(self, prices: pd.Series) -> dict:
        """
        Compare volatility characteristics
        """
        # Split data
        pre_covid = prices['2010':'2019']
        post_covid = prices['2020':'2024']
        
        # Calculate metrics
        vol_pre = pre_covid.pct_change().std() * np.sqrt(252)
        vol_post = post_covid.pct_change().std() * np.sqrt(252)
        
        # Volatility clustering (GARCH effect)
        autocorr_pre = (pre_covid.pct_change()**2).autocorr()
        autocorr_post = (post_covid.pct_change()**2).autocorr()
        
        # Hurst exponent
        hurst_pre = calculate_hurst(pre_covid)
        hurst_post = calculate_hurst(post_covid)
        
        return {
            'volatility_increase': vol_post / vol_pre - 1,  # ~80% increase
            'pre_period': {
                'volatility': vol_pre,
                'vol_clustering': autocorr_pre,
                'hurst': hurst_pre,
                'regime': 'LOW_VOL_RANGE_BOUND'
            },
            'post_period': {
                'volatility': vol_post,
                'vol_clustering': autocorr_post,
                'hurst': hurst_post,
                'regime': 'HIGH_VOL_TRENDING'
            },
            'impact': 'Mean reversion works in low-vol, fails in high-vol trending'
        }

# Results
analysis = RegimeAnalysis()
regime_comparison = analysis.compare_regimes()

print("=== Regime Change Analysis ===")
print("\\n2010-2019 (Golden Age for Mean Reversion):")
print(f"  VIX: {regime_comparison['2010-2019']['avg_vix']}")
print(f"  Volatility: {regime_comparison['2010-2019']['volatility']:.0%}")
print(f"  Fed Policy: {regime_comparison['2010-2019']['fed_policy']}")
print(f"  Mean Reversion Sharpe: {regime_comparison['2010-2019']['sharpe_mean_reversion']}")

print("\\n2020-2024 (Mean Reversion Struggles):")
print(f"  VIX: {regime_comparison['2020-2024']['avg_vix']}")
print(f"  Volatility: {regime_comparison['2020-2024']['volatility']:.0%}")
print(f"  Fed Policy: {regime_comparison['2020-2024']['fed_policy']}")
print(f"  Mean Reversion Sharpe: {regime_comparison['2020-2024']['sharpe_mean_reversion']}")
\`\`\`

**Key Regime Differences:**

| Metric | 2010-2019 | 2020-2024 | Impact on Mean Reversion |
|--------|-----------|-----------|--------------------------|
| VIX Average | 14.5 | 19.5 | Higher vol = wider ranges, less predictable |
| Volatility | 10% | 18% | 80% increase breaks historical patterns |
| Fed Policy | QE, 0% rates | Rate hikes, QT | Removes "Fed put" safety net |
| Trend Strength | Low (ADX 15-20) | High (ADX 25-35) | More trending = less mean reversion |
| COVID Impact | None | Structural change | Market behavior permanently altered |

**B. Crowding (Secondary Factor)**

\`\`\`python
def analyze_strategy_crowding():
    """
    Estimate mean reversion strategy adoption
    """
    return {
        '2010': {
            'retail_traders': '5M',
            'algo_traders': '1K',
            'using_mean_reversion': '10%',
            'estimated_capital': '$50B'
        },
        '2024': {
            'retail_traders': '50M',  # 10x increase (Robinhood era)
            'algo_traders': '50K',  # 50x increase
            'using_mean_reversion': '40%',  # Everyone knows Bollinger Bands
            'estimated_capital': '$2T'  # 40x increase!
        },
        'impact': {
            'alpha_decay': '70%',  # Returns compressed
            'sharpe_decay': '60%',
            'reason': 'Everyone buys the dip, exhausts buying power'
        }
    }
\`\`\`

**C. Volatility Explosion (2020-2024 Specific)**

**Specific Events Breaking Mean Reversion:**

1. **March 2020: COVID Crash**
   - S&P 500: -35% in 23 days
   - Mean reversion traders destroyed (kept buying dips)
   - VIX hit 82 (extreme)

2. **2021: Meme Stock Mania**
   - GME, AMC: Traditional mean reversion failed
   - Reddit/social media created new dynamics
   - "Dumb money" overpowered "smart money"

3. **2022: Fed Tightening**
   - Rate hikes broke 10-year low-vol regime
   - Bonds and stocks correlated (both down)
   - Traditional hedges failed

4. **2023-2024: AI Hype**
   - NVDA, tech mega-caps: strong trends
   - Momentum > mean reversion
   - Concentration in few stocks

**2. Specific Changes in 2020-2024 That Broke Mean Reversion:**

\`\`\`python
class MeanReversionBreakers:
    """
    Identify specific factors that broke mean reversion
    """
    
    def analyze_breaking_factors(self) -> dict:
        return {
            'factor_1_retail_explosion': {
                'description': 'Robinhood + stimulus checks = millions buying dips',
                'impact': 'Exhausted mean reversion bounce',
                'example': 'Every dip in 2020-2021 bought aggressively',
                'result': 'Dips no longer oversold, immediately bought'
            },
            'factor_2_correlation_breakdown': {
                'description': '2022: Stocks+Bonds down together (first time since 1970s)',
                'impact': 'Traditional hedges failed',
                'example': 'SPY -18%, TLT -30% in 2022',
                'result': 'Mean reversion to what? Everything down'
            },
            'factor_3_volatility_regime': {
                'description': 'VIX permanently elevated vs 2010s',
                'impact': 'Wider Bollinger Bands, fewer signals',
                'example': 'VIX floor moved from 12 to 15',
                'result': 'Historical overbought/oversold levels obsolete'
            },
            'factor_4_momentum_dominance': {
                'description': 'AI/Tech created strong trends (not range-bound)',
                'impact': 'Trend > mean reversion',
                'example': 'NVDA +740% in 2023 (would destroy mean reversion shorts)',
                'result': 'Mega-cap momentum broke market dynamics'
            },
            'factor_5_zero_DTE_options': {
                'description': '0DTE options created gamma squeezes',
                'impact': 'Intraday mean reversion unpredictable',
                'example': 'SPEX options volume > SPY stock volume',
                'result': 'Traditional mean reversion signals unreliable'
            }
        }
\`\`\`

**3. Adaptations Needed for Modern Mean Reversion (2024+):**

\`\`\`python
class ModernMeanReversionAdaptations:
    """
    Adapt mean reversion for 2024+ markets
    """
    
    def __init__(self):
        self.adaptations = [
            'shorter_timeframes',
            'dynamic_thresholds',
            'regime_filtering',
            'options_integration',
            'crypto_diversification'
        ]
    
    def adaptation_1_shorter_timeframes(self) -> dict:
        """
        Adaptation 1: Move to intraday (hours vs days)
        
        Reason: Daily mean reversion broken, but intraday still works
        """
        return {
            'old_approach': 'Hold 3-10 days for reversion',
            'new_approach': 'Hold 1-6 hours for reversion',
            'logic': 'Market microstructure still causes intraday reversions',
            'example': 'RSI(14) on 15-min bars vs daily bars',
            'expected_sharpe': '1.5-2.0 (vs 0.8 on daily)'
        }
    
    def adaptation_2_dynamic_thresholds(self) -> dict:
        """
        Adaptation 2: Use dynamic vs fixed thresholds
        
        Reason: Fixed 30/70 RSI or ±2σ BB doesn't adapt to volatility
        """
        return {
            'old_approach': 'Buy when RSI < 30, BB < -2σ',
            'new_approach': 'Buy when RSI < 20th percentile (rolling)',
            'implementation': '''
# Dynamic Bollinger Bands
lookback = 100
std_percentile = data['std'].rolling(lookback).quantile(0.80)
dynamic_bands = ma ± (std_percentile * std_current)

# Dynamic RSI
rsi_oversold = rsi.rolling(252).quantile(0.20)  # Not fixed 30
rsi_overbought = rsi.rolling(252).quantile(0.80)  # Not fixed 70
            ''',
            'benefit': 'Adapts to new volatility regime automatically'
        }
    
    def adaptation_3_regime_filtering(self) -> dict:
        """
        Adaptation 3: Strict regime filtering (only trade in mean-reverting regimes)
        
        Reason: Can't fight trends (2020-2024 lesson)
        """
        return {
            'old_approach': 'Always on (trade every signal)',
            'new_approach': 'Only trade when multiple filters confirm range-bound',
            'filters': [
                'ADX < 20 (not trending)',
                'Hurst < 0.5 (mean-reverting last 60 days)',
                'VIX < 25 (not crisis)',
                'No 52-week highs/lows in last month',
                'Market not down >5% from peak (avoid crashes)'
            ],
            'expected_impact': 'Trade only 30% of time (vs 100% before)',
            'result': 'Lower frequency but higher quality signals'
        }
    
    def adaptation_4_options_for_defined_risk(self) -> dict:
        """
        Adaptation 4: Use options instead of stock (define max loss)
        
        Reason: Stock mean reversion has unlimited downside (COVID lesson)
        """
        return {
            'old_approach': 'Buy stock at lower BB, stop at -2× ATR',
            'new_approach': 'Buy call spread at lower BB (defined risk)',
            'example': '''
# Stock at $400, touched lower BB
# Old: Buy 100 shares, stop at $392 (risk $800)
# New: Buy $400/$410 call spread for $3 (risk $300 max)

If stock drops to $350 (another COVID):
  - Stock: Lose $5,000 (-50%)
  - Call spread: Lose $300 max (-100% of premium)
            ''',
            'benefit': 'Survive black swans, defined risk'
        }
    
    def adaptation_5_crypto_diversification(self) -> dict:
        """
        Adaptation 5: Add crypto mean reversion (24/7, high volatility)
        
        Reason: Crypto still exhibits intraday mean reversion (less crowded)
        """
        return {
            'rationale': 'Crypto mean reversion less crowded than stocks',
            'characteristics': {
                'volatility': '3-5x higher than stocks',
                'mean_reversion': 'Strong intraday (4-hour timeframe)',
                'hurst_exponent': '0.45-0.48 (mean-reverting)',
                'half_life': '8-24 hours (fast reversion)'
            },
            'strategy': 'RSI(14) on 1-hour BTC, 30/70 levels still work',
            'expected_sharpe': '1.5-2.5 (vs 0.8 for stock mean reversion)'
        }
\`\`\`

**Complete Modern Mean Reversion Strategy (2024+):**

| Aspect | Old (2010-2019) | New (2024+) |
|--------|-----------------|-------------|
| Timeframe | Daily (3-10 day hold) | Intraday (1-6 hour hold) |
| Thresholds | Fixed (RSI 30/70, ±2σ) | Dynamic (percentile-based) |
| Regime Filter | None (always on) | Strict (ADX, Hurst, VIX) |
| Instrument | Stock (unlimited downside) | Options (defined risk) |
| Universe | US stocks only | Stocks + Crypto |
| Expected Sharpe | 2.5 (2010s), 0.8 (2020s) | 1.5-2.0 (adapted) |

**Bottom Line:**

2010-2019 was the golden age for simple mean reversion (low vol, range-bound, Fed put). 2020-2024 broke it (COVID, retail explosion, volatility regime shift). To survive:
1. Move to intraday (hours not days)
2. Dynamic thresholds (adapt to vol)
3. Strict regime filters (only trade 30% of time)
4. Use options (define risk)
5. Diversify to crypto (less crowded)

Can't go back to 2010s (market changed). Must adapt or perish.`,
    keyPoints: [
      '2010-2019 golden age: VIX 14, low vol (10%), QE, range-bound (Sharpe 2-3); 2020-2024: VIX 19, high vol (18%), rate hikes, trending (Sharpe 0.8)',
      'Specific breakers: COVID crash (-35% in 23 days), retail explosion (Robinhood), correlation breakdown (stocks+bonds down), 0DTE options',
      'Adaptations: (1) Intraday timeframes (1-6 hours vs 3-10 days), (2) Dynamic thresholds (percentile-based vs fixed)',
      'More adaptations: (3) Strict regime filters (ADX<20, Hurst<0.5, VIX<25, trade only 30% of time), (4) Options for defined risk',
      'Diversification: Add crypto mean reversion (Hurst 0.45-0.48, 8-24hr half-life, Sharpe 1.5-2.5 vs stock 0.8)',
    ],
  },
];
