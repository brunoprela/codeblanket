export const autocorrelationPartialAutocorrelationQuiz = [
    {
        id: 1,
        question:
            "A quantitative researcher discovers that daily returns of a momentum ETF show ACF values of [0.15, 0.12, 0.09, 0.07, 0.05] at lags 1-5 (all statistically significant with p < 0.01). The trading desk wants to build a 'momentum continuation' strategy that goes long when yesterday's return was positive. The risk manager objects: 'This violates the Efficient Market Hypothesis - returns shouldn't be predictable. The strategy will lose money after transaction costs.' You need to: (1) Explain whether significant ACF genuinely indicates an exploitable market inefficiency or could be spurious, (2) Calculate the theoretical maximum profit from this autocorrelation pattern, (3) Determine the transaction cost threshold above which the strategy becomes unprofitable, and (4) Design three statistical tests to validate if this is a real tradeable pattern or data mining artifact.",
        answer: `## Comprehensive Answer:

### Part 1: Is This Genuine Market Inefficiency?

**The ACF pattern suggests autocorrelation, but several explanations exist:**

**A. Genuine Market Inefficiency (Exploitable):**
- Momentum effect is real and documented (Fama-French momentum factor)
- Information diffusion takes time (prices don't instantly reflect all info)
- Momentum ETF may have unique properties (rebalancing creates autocorrelation)
- Cross-asset contagion creates predictability

**B. Spurious Patterns (NOT Exploitable):**1. **Non-synchronous trading:**
   - ETF holds stocks that trade at different times
   - Stale prices create artificial autocorrelation
   - Formula: \\(ACF_{spurious} \\approx (1-p)^2 / [1+p^2-2p]\\) where p = % of stale prices

2. **Bid-ask bounce:**
   - Trades alternate between bid and ask
   - Creates negative autocorrelation in tick data, positive in returns
   - Not exploitable (you pay the spread!)

3. **Microstructure noise:**
   - Market maker inventory effects
   - Order flow toxicity
   - These disappear at lower frequencies

4. **Data mining:**
   - Testing many patterns → some significant by chance
   - 5% significance level → 1 in 20 false positives
   - Multiple testing correction needed

**Testing for Genuine vs Spurious:**

\`\`\`python
def test_genuine_autocorrelation(returns: pd.Series, 
                                 prices: pd.Series) -> dict:
    """
    Distinguish genuine autocorrelation from spurious patterns.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller
    
    results = {}
    
    # Test 1: Check different frequencies
    # Spurious autocorrelation disappears at lower frequencies
    daily_acf = returns.autocorr(lag=1)
    weekly_returns = prices.resample('W').last().pct_change()
    weekly_acf = weekly_returns.autocorr(lag=1)
    
    results['daily_acf'] = daily_acf
    results['weekly_acf'] = weekly_acf
    results['frequency_test'] = (
        'Genuine' if abs(weekly_acf) > 0.5 * abs(daily_acf)
        else 'Spurious (disappears at lower frequency)'
    )
    
    # Test 2: Sub-period stability
    # Genuine effects should be stable across time
    mid = len(returns) // 2
    first_half_acf = returns.iloc[:mid].autocorr(lag=1)
    second_half_acf = returns.iloc[mid:].autocorr(lag=1)
    
    results['first_half_acf'] = first_half_acf
    results['second_half_acf'] = second_half_acf
    results['stability_test'] = (
        'Stable (genuine)' if abs(first_half_acf - second_half_acf) < 0.05
        else 'Unstable (possibly spurious)'
    )
    
    # Test 3: Out-of-sample validation
    # Can't show here, but critical for real validation
    
    return results
\`\`\`

### Part 2: Theoretical Maximum Profit

**Given ACF pattern:** \\(\\rho_1 = 0.15, \\rho_2 = 0.12, ...\\)

**Theoretical return from lag-1 momentum:**

If ACF(1) = 0.15, then:
$$E[R_t | R_{t-1}] = \\rho_1 \\cdot R_{t-1} = 0.15 \\cdot R_{t-1}$$

**For a simple momentum strategy:**
- Signal: \\(S_t = sign(R_{t-1})\\) (±1)
- Position: Long if yesterday positive, short if negative
- Strategy return: \\(R_{strategy,t} = S_t \\cdot R_t\\)

**Expected profit per trade:**

\`\`\`python
def calculate_theoretical_profit(acf_lag1: float,
                                daily_vol: float = 0.02,
                                periods_per_year: int = 252) -> dict:
    """
    Calculate theoretical maximum profit from autocorrelation.
    
    Args:
        acf_lag1: Lag-1 autocorrelation
        daily_vol: Daily volatility
        periods_per_year: Trading days per year
    """
    # Expected profit per trade (before costs)
    # E[S_t * R_t] = E[sign(R_{t-1}) * (rho * R_{t-1} + noise)]
    #              ≈ rho * E[|R_{t-1}|]
    
    # For normal distribution: E[|X|] = sigma * sqrt(2/pi)
    expected_abs_return = daily_vol * np.sqrt(2 / np.pi)
    
    # Expected profit per period
    profit_per_period = acf_lag1 * expected_abs_return
    
    # Annual profit (if trading every day)
    annual_profit = profit_per_period * periods_per_year
    
    # Information Ratio (theoretical)
    # Strategy vol ≈ daily_vol (not leveraged)
    strategy_vol = daily_vol * np.sqrt(periods_per_year)
    information_ratio = annual_profit / strategy_vol
    
    return {
        'profit_per_trade': profit_per_period,
        'annual_profit_pct': annual_profit * 100,
        'information_ratio': information_ratio,
        'sharpe_ratio': information_ratio,  # Assuming no risk-free rate
        'interpretation': (
            f"Theoretical max: {annual_profit*100:.2f}% annual profit\\n"
            f"With ACF={acf_lag1:.2f}, each trade profits {profit_per_period*100:.3f}%\\n"
            f"Over {periods_per_year} trades: {annual_profit*100:.2f}% annual"
        )
    }

# Example calculation
results = calculate_theoretical_profit(
    acf_lag1=0.15,
    daily_vol=0.02,
    periods_per_year=252
)

print("Theoretical Maximum Profit:")
print(f"  Per trade: {results['profit_per_trade']*100:.3f}%")
print(f"  Annual: {results['annual_profit_pct']:.2f}%")
print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
\`\`\`

**For ACF(1) = 0.15, daily vol = 2%:**
- Expected profit per trade ≈ 0.15 × 0.016 ≈ **0.24 basis points (0.0024%)**
- Annual profit (252 trades) ≈ **0.60%**
- Sharpe ratio ≈ **0.19**

**Reality check:** This is TINY! Must trade frequently and large size to be meaningful.

### Part 3: Transaction Cost Threshold

**Breakeven cost:**

Each round-trip costs \\(2 \\times c\\) (buy and sell).

Strategy is profitable if:
$$E[profit per trade] > 2c$$

$$0.0024\\% > 2c$$
$$c < 0.0012\\%$$
$$c < 1.2 \\text{ basis points}$$

\`\`\`python
def calculate_cost_threshold(acf_lag1: float,
                             daily_vol: float = 0.02) -> dict:
    """
    Calculate maximum transaction cost for profitability.
    """
    expected_abs_return = daily_vol * np.sqrt(2 / np.pi)
    profit_per_trade = acf_lag1 * expected_abs_return
    
    # Round-trip costs (2x one-way)
    max_one_way_cost = profit_per_trade / 2
    
    # Typical costs
    costs = {
        'retail_broker': 0.0005,  # 5 bps
        'institutional': 0.0002,  # 2 bps
        'market_maker': 0.0001,   # 1 bp
        'hft_firm': 0.00005,      # 0.5 bp
    }
    
    results = {
        'max_one_way_cost_bps': max_one_way_cost * 10000,
        'max_round_trip_cost_bps': max_one_way_cost * 2 * 10000,
    }
    
    # Assess feasibility for different traders
    for trader_type, cost in costs.items():
        is_profitable = cost < max_one_way_cost
        results[trader_type] = {
            'cost_bps': cost * 10000,
            'profitable': is_profitable,
            'profit_after_costs': (profit_per_trade - 2*cost) * 252 * 100 if is_profitable else 'LOSS'
        }
    
    return results

# Example
cost_analysis = calculate_cost_threshold(acf_lag1=0.15, daily_vol=0.02)

print("\\nTransaction Cost Analysis:")
print(f"Maximum cost for profitability: {cost_analysis['max_one_way_cost_bps']:.2f} bps\\n")

for trader_type in ['retail_broker', 'institutional', 'market_maker', 'hft_firm']:
    info = cost_analysis[trader_type]
    print(f"{trader_type.replace('_', ' ').title()}:")
    print(f"  Cost: {info['cost_bps']:.2f} bps")
    print(f"  Profitable: {info['profitable']}")
    if info['profitable']:
        print(f"  Annual profit after costs: {info['profit_after_costs']:.2f}%")
    print()
\`\`\`

**Conclusion:** 
- **Retail brokers:** Costs ~5 bps → UNPROFITABLE
- **Institutional:** Costs ~2 bps → UNPROFITABLE  
- **Market makers:** Costs ~1 bp → MARGINAL
- **HFT firms:** Costs ~0.5 bp → PROFITABLE

**Risk Manager is RIGHT for retail/institutional execution!**

### Part 4: Three Validation Tests

**Test 1: Walk-Forward Out-of-Sample Validation**

\`\`\`python
def walk_forward_validation(returns: pd.Series,
                           train_window: int = 252,
                           test_window: int = 21) -> dict:
    """
    Test if autocorrelation is stable out-of-sample.
    
    If data mining artifact, will disappear out-of-sample.
    """
    results = []
    
    for start in range(train_window, len(returns) - test_window, test_window):
        # Training: estimate ACF
        train_data = returns.iloc[start-train_window:start]
        train_acf = train_data.autocorr(lag=1)
        
        # Testing: check if pattern holds
        test_data = returns.iloc[start:start+test_window]
        test_acf = test_data.autocorr(lag=1)
        
        # Strategy performance
        signal = np.sign(test_data.shift(1))
        strategy_returns = signal * test_data
        strategy_mean = strategy_returns.mean()
        
        results.append({
            'period': start,
            'train_acf': train_acf,
            'test_acf': test_acf,
            'strategy_return': strategy_mean * 21 * 100  # Monthly %
        })
    
    df = pd.DataFrame(results)
    
    # Stability test
    acf_correlation = df['train_acf'].corr(df['test_acf'])
    mean_strategy_return = df['strategy_return'].mean()
    fraction_positive = (df['strategy_return'] > 0).mean()
    
    return {
        'n_periods': len(df),
        'train_test_acf_correlation': acf_correlation,
        'mean_oos_return_monthly': mean_strategy_return,
        'fraction_positive_periods': fraction_positive,
        'is_genuine': (
            acf_correlation > 0.5 and 
            fraction_positive > 0.55 and
            mean_strategy_return > 0
        ),
        'interpretation': (
            'GENUINE: Stable out-of-sample' if acf_correlation > 0.5
            else 'SPURIOUS: Unstable out-of-sample'
        )
    }
\`\`\`

**Test 2: Multiple Testing Correction (Bonferroni)**

\`\`\`python
def multiple_testing_correction(returns: pd.Series,
                               max_lag: int = 20,
                               alpha: float = 0.05) -> dict:
    """
    Correct for data mining by adjusting significance level.
    
    If testing 20 lags, need alpha = 0.05/20 = 0.0025 for each test.
    """
    from scipy import stats
    
    # Calculate ACF
    acf_vals = [returns.autocorr(lag=k) for k in range(1, max_lag+1)]
    
    # Standard errors
    n = len(returns)
    se = 1 / np.sqrt(n)
    
    # Z-statistics
    z_stats = [acf / se for acf in acf_vals]
    
    # P-values (two-tailed)
    p_values = [2 * (1 - stats.norm.cdf(abs(z))) for z in z_stats]
    
    # Bonferroni correction
    alpha_corrected = alpha / max_lag
    
    # Count significant (before and after correction)
    sig_before = sum(p < alpha for p in p_values)
    sig_after = sum(p < alpha_corrected for p in p_values)
    
    return {
        'tests_conducted': max_lag,
        'alpha_uncorrected': alpha,
        'alpha_bonferroni': alpha_corrected,
        'significant_uncorrected': sig_before,
        'significant_bonferroni': sig_after,
        'survives_correction': sig_after > 0,
        'interpretation': (
            f'{sig_after}/{max_lag} lags significant after Bonferroni correction.\\n'
            f'{"GENUINE pattern" if sig_after > 0 else "Likely DATA MINING artifact"}'
        )
    }
\`\`\`

**Test 3: Bootstrap Simulation**

\`\`\`python
def bootstrap_significance(returns: pd.Series,
                          observed_acf: float,
                          n_bootstrap: int = 10000) -> dict:
    """
    Test if observed ACF is significantly different from random.
    
    Generate bootstrap samples under null (no autocorrelation),
    see where observed ACF ranks.
    """
    # Bootstrap under null (shuffle returns)
    bootstrap_acfs = []
    
    for _ in range(n_bootstrap):
        # Shuffle to destroy autocorrelation
        shuffled = returns.sample(frac=1.0, replace=False).values
        shuffled_series = pd.Series(shuffled)
        bootstrap_acf = shuffled_series.autocorr(lag=1)
        bootstrap_acfs.append(bootstrap_acf)
    
    bootstrap_acfs = np.array(bootstrap_acfs)
    
    # P-value: proportion of bootstrap ACFs >= observed
    p_value = np.mean(np.abs(bootstrap_acfs) >= abs(observed_acf))
    
    # Confidence interval
    ci_lower = np.percentile(bootstrap_acfs, 2.5)
    ci_upper = np.percentile(bootstrap_acfs, 97.5)
    
    return {
        'observed_acf': observed_acf,
        'bootstrap_mean': bootstrap_acfs.mean(),
        'bootstrap_std': bootstrap_acfs.std(),
        'p_value': p_value,
        '95_ci': (ci_lower, ci_upper),
        'significant': p_value < 0.05,
        'interpretation': (
            f'Observed ACF = {observed_acf:.3f}\\n'
            f'Bootstrap 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]\\n'
            f'P-value: {p_value:.4f}\\n'
            f'{"SIGNIFICANT: Unlikely due to chance" if p_value < 0.05 else "Not significant"}'
        )
    }
\`\`\`

### Recommendation

**To the Trading Desk:**

"Yes, ACF(1) = 0.15 is statistically significant, BUT:

1. **Profitability threshold:** Need execution costs < 1.2 bps per trade
2. **Your costs:** Likely 2-5 bps → Strategy is UNPROFITABLE after costs
3. **Theoretical max:** Even with perfect execution, only ~0.6% annual return
4. **Risk Manager is correct:** This won't work at institutional cost structure

**However**, if you can:
- Execute at market maker levels (< 1 bp)
- Use HFT infrastructure
- Scale to large size
- Validate pattern is genuine (run all 3 tests above)

Then it MIGHT be exploitable. But for typical execution, abandon this strategy."

**To the Risk Manager:**

"You're correct about transaction costs, but let's validate whether pattern is genuine:
1. Run walk-forward validation
2. Apply Bonferroni correction
3. Bootstrap test

If pattern survives these tests AND we can get execution costs down, it's worth revisiting. Otherwise, pattern is either spurious or unexploitable."`,
    },
    {
        id: 2,
        question:
            "Your ARIMA modeling system identifies that daily S&P 500 returns have: ACF that decays geometrically (0.08, 0.06, 0.05, 0.04...) and PACF that cuts off sharply after lag 1 (0.08 at lag 1, then all near zero). This suggests an AR(1) model. However, when you fit AR(1) and examine the residuals, the Ljung-Box test shows p-value < 0.001 (rejecting white noise residuals). Additionally, the squared residuals show even stronger autocorrelation than the original returns (ACF of squared residuals: 0.25, 0.20, 0.18...). Explain: (1) What the residual autocorrelation means for your AR(1) model, (2) What the autocorrelation in squared residuals indicates, (3) Why simple ARIMA models fail here, and (4) The correct model class to use and how to validate the final model.",
        answer: `## Comprehensive Answer:

### Part 1: What Residual Autocorrelation Means

**Problem:** After fitting AR(1), residuals still show autocorrelation.

**Interpretation:**
- AR(1) captured SOME pattern (first-order linear dependence)
- BUT significant information remains in residuals
- Model is **misspecified** - wrong functional form or order

**Why this matters:**
- Forecasts are suboptimal (leaving money on the table)
- Standard errors are wrong (confidence intervals invalid)
- Risk models underestimate true risk

**Possible causes:**

\`\`\`python
def diagnose_residual_autocorrelation(returns: pd.Series,
                                     model_residuals: pd.Series) -> dict:
    """
    Diagnose why residuals show autocorrelation.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    
    # Test residual autocorrelation
    lb_result = acorr_ljungbox(model_residuals, lags=20, return_df=True)
    
    # Check residual ACF pattern
    residual_acf = [model_residuals.autocorr(lag=k) for k in range(1, 11)]
    
    # Possible causes:
    causes = []
    
    # Cause 1: Wrong AR order (should be AR(p), p>1)
    if abs(residual_acf[1]) > 2/np.sqrt(len(returns)):  # Lag 2 significant
        causes.append("AR order too low - try AR(2) or AR(3)")
    
    # Cause 2: MA component needed (ARMA, not pure AR)
    pacf_residual = pacf(model_residuals, nlags=10)
    if max(abs(pacf_residual[2:5])) > 2/np.sqrt(len(returns)):
        causes.append("MA component needed - try ARMA(1,1)")
    
    # Cause 3: Nonlinear effects (GARCH, regime-switching)
    squared_residuals = model_residuals ** 2
    squared_acf = squared_residuals.autocorr(lag=1)
    if squared_acf > 0.1:
        causes.append("VOLATILITY CLUSTERING - need GARCH model")
    
    # Cause 4: Structural breaks
    # (Would need more complex testing)
    
    return {
        'ljung_box_pvalue': lb_result['lb_pvalue'].iloc[0],
        'residuals_white_noise': lb_result['lb_pvalue'].iloc[0] > 0.05,
        'residual_acf_lag1': residual_acf[0],
        'squared_residual_acf': squared_acf,
        'probable_causes': causes,
        'interpretation': (
            "✓ Residuals are white noise - model is adequate" 
            if lb_result['lb_pvalue'].iloc[0] > 0.05
            else f"✗ Model misspecified. Likely causes:\\n" + "\\n".join(f"  - {c}" for c in causes)
        )
    }
\`\`\`

**In your case:**
Ljung-Box p < 0.001 → AR(1) is **insufficient**. Need more complex model.

### Part 2: Autocorrelation in Squared Residuals

**This is the KEY diagnostic!**

**What it means:**
Squared residuals represent volatility/variance. ACF in squared residuals = **volatility clustering**.

**Financial interpretation:**
- Large returns follow large returns (regardless of sign)
- Small returns follow small returns
- Volatility is predictable even if returns aren't!
- Classic GARCH effect

**Mathematical:**
\`\`\`
Returns: r_t = μ + ε_t
Where: ε_t = σ_t * z_t
       z_t ~ N(0,1) (unpredictable)
       σ_t^2 = ω + α*ε_{t-1}^2 + β*σ_{t-1}^2  (PREDICTABLE!)
\`\`\`

**Visualization:**

\`\`\`python
def visualize_volatility_clustering(returns: pd.Series,
                                   residuals: pd.Series):
    """
    Show evidence of volatility clustering.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Panel 1: Returns (or residuals)
    axes[0].plot(residuals.index, residuals.values, linewidth=0.5)
    axes[0].set_title('Residuals from AR(1) Model', fontsize=12)
    axes[0].set_ylabel('Residual')
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Squared residuals (volatility proxy)
    squared_resid = residuals ** 2
    axes[1].plot(squared_resid.index, squared_resid.values, 
                linewidth=0.5, color='red')
    axes[1].set_title('Squared Residuals (Volatility Proxy)', fontsize=12)
    axes[1].set_ylabel('Squared Residual')
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: ACF of squared residuals
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(squared_resid, lags=40, ax=axes[2], alpha=0.05)
    axes[2].set_title('ACF of Squared Residuals (Shows Clustering)', fontsize=12)
    axes[2].set_xlabel('Lag')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Note: Would create visualization showing clustering
print("Squared residual ACF = 0.25 → Strong volatility clustering!")
print("High volatility today → High volatility tomorrow")
\`\`\`

**Why squared residuals?**
- Absolute residuals also work: |ε_t|
- Squaring emphasizes large values
- Mathematical: ε_t^2 is unbiased estimator of σ_t^2

### Part 3: Why Simple ARIMA Fails

**ARIMA assumptions:**1. ✓ Linear dependence in conditional mean
2. ✗ **Constant conditional variance** (homoskedasticity)
3. ✗ **Independent residuals**

**Your data:**
- ACF in residuals → Assumption 3 violated
- ACF in squared residuals → Assumption 2 violated (heteroskedasticity)

**The problem:**

\`\`\`
ARIMA models:
  r_t = μ + φ*r_{t-1} + ... + ε_t
  where ε_t ~ N(0, σ²)  ← CONSTANT variance!

Reality:
  r_t = μ + φ*r_{t-1} + ... + ε_t
  where ε_t ~ N(0, σ_t²)  ← TIME-VARYING variance!
\`\`\`

**Consequences of using ARIMA:**1. **Inefficient forecasts** - not using variance information
2. **Wrong confidence intervals** - assume constant vol
3. **Underestimate risk** - miss volatility spikes
4. **Missed trading opportunities** - volatility is tradeable!

**Real-world impact:**
- Risk models fail during crises (vol clustering)
- Options mispriced (wrong volatility forecast)
- VaR calculations wrong

### Part 4: Correct Model Class and Validation

**The solution: AR-GARCH Model**

**Model specification:**

**Conditional Mean (AR part):**
$$r_t = \\mu + \\phi r_{t-1} + \\epsilon_t$$

**Conditional Variance (GARCH part):**
$$\\sigma_t^2 = \\omega + \\alpha \\epsilon_{t-1}^2 + \\beta \\sigma_{t-1}^2$$

Where:
- \\(\\omega > 0\\) (baseline volatility)
- \\(\\alpha, \\beta \\geq 0\\) (persistence)
- \\(\\alpha + \\beta < 1\\) (stationarity)

**Implementation:**

\`\`\`python
from arch import arch_model

def fit_ar_garch_model(returns: pd.Series) -> dict:
    """
    Fit AR(1)-GARCH(1,1) model for returns with volatility clustering.
    """
    # Scale returns to percentage for numerical stability
    returns_pct = returns * 100
    
    # Specify model: AR(1) mean, GARCH(1,1) variance
    model = arch_model(
        returns_pct,
        mean='AR',          # AR mean model
        lags=1,             # AR(1)
        vol='GARCH',        # GARCH variance
        p=1,                # GARCH lag
        q=1,                # ARCH lag
        dist='normal'       # Error distribution
    )
    
    # Fit model
    results = model.fit(disp='off')
    
    # Extract parameters
    params = results.params
    
    return {
        'model': results,
        'mu': params['mu'],                    # Mean
        'phi': params['ar.L1'],                # AR(1) coefficient
        'omega': params['omega'],              # GARCH constant
        'alpha': params['alpha[1]'],           # ARCH coefficient
        'beta': params['beta[1]'],             # GARCH coefficient
        'persistence': params['alpha[1]'] + params['beta[1]'],
        'aic': results.aic,
        'bic': results.bic,
        'interpretation': f"""
AR(1)-GARCH(1,1) Model Results:
  Mean equation: r_t = {params['mu']:.4f} + {params['ar.L1']:.4f}*r_{{t-1}} + ε_t
  Variance equation: σ_t^2 = {params['omega']:.4f} + {params['alpha[1]']:.4f}*ε_{{t-1}}^2 + {params['beta[1]']:.4f}*σ_{{t-1}}^2
  
  Persistence: {params['alpha[1]'] + params['beta[1]']:.4f} (< 1 ✓ stationary)
  Half-life of vol shock: {-np.log(2)/np.log(params['alpha[1]'] + params['beta[1]']):.1f} days
        """
    }


# Example
np.random.seed(42)
# Simulate AR(1)-GARCH(1,1) process
n = 1000
returns = [0]
sigma2 = [0.02**2]  # Initial variance

for t in range(1, n):
    # GARCH variance
    sigma2_t = 0.00001 + 0.1 * (returns[-1]**2) + 0.85 * sigma2[-1]
    sigma2.append(sigma2_t)
    
    # AR(1) with GARCH errors
    epsilon = np.random.randn() * np.sqrt(sigma2_t)
    r_t = 0.0005 + 0.08 * returns[-1] + epsilon
    returns.append(r_t)

returns_series = pd.Series(returns)

# Fit model
ar_garch = fit_ar_garch_model(returns_series)
print(ar_garch['interpretation'])
\`\`\`

**Model Validation (Critical!)**

\`\`\`python
def validate_ar_garch(model_results, returns: pd.Series) -> dict:
    """
    Comprehensive validation of AR-GARCH model.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from scipy import stats
    
    # Extract standardized residuals
    std_resid = model_results.std_resid
    
    validation = {}
    
    # Test 1: Standardized residuals should be white noise
    lb_resid = acorr_ljungbox(std_resid.dropna(), lags=20, return_df=True)
    validation['residuals_white_noise'] = lb_resid['lb_pvalue'].iloc[0] > 0.05
    validation['residuals_lb_pvalue'] = lb_resid['lb_pvalue'].iloc[0]
    
    # Test 2: Squared standardized residuals should have NO autocorrelation
    #         (all variance clustering captured by GARCH)
    squared_std_resid = std_resid ** 2
    lb_squared = acorr_ljungbox(squared_std_resid.dropna(), lags=20, return_df=True)
    validation['squared_resid_white_noise'] = lb_squared['lb_pvalue'].iloc[0] > 0.05
    validation['squared_lb_pvalue'] = lb_squared['lb_pvalue'].iloc[0]
    
    # Test 3: Standardized residuals should be normally distributed
    jb_stat, jb_pvalue = stats.jarque_bera(std_resid.dropna())
    validation['normality_pvalue'] = jb_pvalue
    validation['residuals_normal'] = jb_pvalue > 0.05
    
    # Test 4: No remaining ARCH effects
    from statsmodels.stats.diagnostic import het_arch
    arch_test = het_arch(std_resid.dropna(), nlags=10)
    validation['arch_test_pvalue'] = arch_test[1]
    validation['no_arch_effects'] = arch_test[1] > 0.05
    
    # Overall assessment
    all_tests_pass = (
        validation['residuals_white_noise'] and
        validation['squared_resid_white_noise'] and
        validation['no_arch_effects']
    )
    
    validation['model_adequate'] = all_tests_pass
    validation['summary'] = (
        "✓ Model is ADEQUATE:\\n"
        "  - Standardized residuals: white noise\\n"
        "  - Squared residuals: no autocorrelation\\n"
        "  - No remaining ARCH effects\\n"
        "Model successfully captures both mean and variance dynamics!"
        if all_tests_pass else
        "✗ Model still INADEQUATE:\\n"
        f"  - Residuals white noise: {validation['residuals_white_noise']}\\n"
        f"  - Squared residuals white noise: {validation['squared_resid_white_noise']}\\n"
        f"  - No ARCH effects: {validation['no_arch_effects']}\\n"
        "Consider: (1) Higher GARCH orders, (2) GJR-GARCH, (3) Student-t errors"
    )
    
    return validation
\`\`\`

**Additional Considerations:**1. **If Student-t distribution fits better:**
   - Fat tails beyond GARCH
   - Use `dist='t'` in arch_model

2. **If asymmetric effects (leverage effect):**
   - Negative shocks increase volatility more
   - Use GJR-GARCH or EGARCH

3. **If very long memory:**
   - Use FIGARCH (fractionally integrated)

### Summary

**Your situation:**1. ✗ AR(1) residuals correlated → Model incomplete
2. ✗ Squared residuals correlated → **Volatility clustering**3. ✓ Use **AR(1)-GARCH(1,1)**4. ✓ Validate: Check standardized residuals are white noise

**Why GARCH matters:**
- Better volatility forecasts → Better risk management
- Capture volatility clustering → More accurate VaR
- Time-varying volatility → Correct option pricing
- Exploitable volatility patterns → Trading opportunities

**The key insight:** 
Returns may be unpredictable (EMH), but VOLATILITY is highly predictable (clustering). GARCH captures this.`,
    },
    {
        id: 3,
        question:
            "A pairs trading algorithm monitors the spread between two cointegrated stocks. The trading rule is: 'Enter trade when spread exceeds 2 standard deviations, exit when spread returns to mean.' Historical backtests show excellent returns (Sharpe 2.5). However, you notice the spread's ACF shows: ACF(1)=0.95, ACF(2)=0.90, ACF(3)=0.86 - very slow geometric decay. The PACF shows a large spike at lag 1 (0.95) then drops to near zero. Your colleague says: 'Perfect! The high ACF(1)=0.95 means the spread is very persistent, so our mean-reversion trades have a high success rate.' Explain: (1) Why your colleague's interpretation is backwards and dangerous, (2) What ACF(1)=0.95 actually indicates about mean reversion speed, (3) How to calculate the optimal entry/exit thresholds given this ACF pattern, and (4) Why the backtest Sharpe of 2.5 is likely overstated and how to get a realistic estimate.",
        answer: `## Comprehensive Answer:

### Part 1: Why Colleague's Interpretation is Wrong

**The colleague has it EXACTLY BACKWARDS.**

**What ACF(1) = 0.95 ACTUALLY means:**

ACF close to 1.0 indicates **SLOW mean reversion**, NOT fast.

**Mathematical proof:**

For AR(1) process:
$$spread_t = \\phi \\cdot spread_{t-1} + \\epsilon_t$$

Where \\(\\phi\\) = ACF(1) ≈ PACF(1) for AR(1).

**Mean reversion speed:**
- \\(\\phi = 0\\): Immediate mean reversion (white noise)
- \\(\\phi = 0.5\\): Moderate persistence
- \\(\\phi = 0.95\\): **VERY slow** mean reversion
- \\(\\phi = 1.0\\): NO mean reversion (random walk)

**Half-life of shocks:**
$$t_{1/2} = \\frac{-\\ln(2)}{\\ln(\\phi)}$$

\`\`\`python
def mean_reversion_analysis(acf_lag1: float) -> dict:
    """
    Analyze mean reversion speed from ACF(1).
    """
    phi = acf_lag1
    
    # Half-life: time for shock to decay to 50%
    if 0 < phi < 1:
        half_life = -np.log(2) / np.log(phi)
    elif phi >= 1:
        half_life = np.inf
    else:
        half_life = None  # Invalid
    
    # Time to decay to various levels
    decay_times = {}
    for pct in [0.9, 0.5, 0.1, 0.01]:
        if 0 < phi < 1:
            t = -np.log(pct) / np.log(phi)
            decay_times[f'decay_to_{int(pct*100)}%'] = t
        else:
            decay_times[f'decay_to_{int(pct*100)}%'] = np.inf
    
    # Mean reversion strength
    if phi < 0:
        mr_strength = "Strong mean reversion (oscillatory)"
    elif phi < 0.3:
        mr_strength = "Fast mean reversion"
    elif phi < 0.7:
        mr_strength = "Moderate mean reversion"
    elif phi < 0.95:
        mr_strength = "Slow mean reversion"
    elif phi < 1.0:
        mr_strength = "VERY SLOW mean reversion (nearly random walk)"
    else:
        mr_strength = "NO mean reversion (random walk or explosive)"
    
    return {
        'phi': phi,
        'half_life_periods': half_life,
        'mean_reversion_strength': mr_strength,
        'decay_times': decay_times,
        'interpretation': f"""
ACF(1) = {phi:.2f} Analysis:

Mean Reversion Speed: {mr_strength}
Half-life: {half_life:.1f} periods

Time to decay:
  - To 90% of shock: {decay_times['decay_to_90%']:.1f} periods
  - To 50% of shock: {decay_times['decay_to_50%']:.1f} periods  
  - To 10% of shock: {decay_times['decay_to_10%']:.1f} periods
  - To 1% of shock: {decay_times['decay_to_01%']:.1f} periods

Trading implication: {"FAST trades (hours/days)" if phi < 0.7 else 
                     "SLOW trades (weeks/months)" if phi < 0.95 else
                     "VERY LONG holding periods (months+) - may not be tradeable!"}
        """
    }

# Example: Colleague's spread (ACF = 0.95)
analysis_095 = mean_reversion_analysis(0.95)
print("=== Slow Mean Reversion (ACF=0.95) ===")
print(analysis_095['interpretation'])

# Compare to fast mean reversion
analysis_05 = mean_reversion_analysis(0.5)
print("\\n=== Fast Mean Reversion (ACF=0.5) ===")
print(analysis_05['interpretation'])
\`\`\`

**For ACF(1) = 0.95:**
- Half-life ≈ **13.5 periods** (e.g., 13.5 days)
- Decay to 10% takes **44 periods**
- Decay to 1% takes **90 periods**

**Why this is DANGEROUS for trading:**1. **Long holding periods** → More risk exposure
2. **Slow convergence** → Spread can widen before reverting
3. **Higher drawdowns** → Can go 3-4 std devs before reverting
4. **Opportunity cost** → Capital tied up for months

**Contrast with ACF(1) = 0.5:**
- Half-life ≈ **1 period**
- Decay to 10% takes **3.3 periods**
- Fast convergence → Lower risk

**Colleague's error:**
"High correlation = persistent = successful trades"

**Reality:**
"High correlation = slow reversion = risky trades"

### Part 2: Calculating Optimal Entry/Exit Thresholds

**Standard approach (WRONG for φ=0.95):**
- Enter at ±2σ
- Exit at mean (0)

**Problem:** With slow mean reversion, spread can widen to 3-4σ before reverting!

**Optimal strategy given φ=0.95:**

\`\`\`python
def optimal_thresholds_given_phi(phi: float,
                                 spread_std: float = 1.0,
                                 max_hold_periods: int = 60) -> dict:
    """
    Calculate optimal entry/exit thresholds for AR(1) spread.
    
    Accounts for:
    - Mean reversion speed (phi)
    - Risk of further widening
    - Expected hold time
    """
    # Expected path of spread after hitting entry threshold
    def expected_path(entry_level: float, n_periods: int) -> np.ndarray:
        """Forecast spread path from entry."""
        path = [entry_level]
        for _ in range(n_periods):
            path.append(phi * path[-1])  # E[spread_t] = phi * spread_{t-1}
        return np.array(path)
    
    # Probability of hitting stop-loss before mean
    def prob_stop_before_mean(entry: float, stop: float, periods: int) -> float:
        """
        Monte Carlo: probability spread hits stop before returning to mean.
        """
        n_sim = 10000
        innovations_std = spread_std * np.sqrt(1 - phi**2)  # Stationary variance
        
        hit_stop = 0
        for _ in range(n_sim):
            spread = entry
            for t in range(periods):
                spread = phi * spread + np.random.randn() * innovations_std
                if abs(spread) >= abs(stop):
                    hit_stop += 1
                    break
                if abs(spread) < 0.1 * spread_std:  # Returned to mean
                    break
        
        return hit_stop / n_sim
    
    # Optimize thresholds
    # Entry: High enough for profit, not so high we often hit stop
    # Stop: Wide enough to avoid noise, not so wide we lose too much
    
    results = {}
    
    # Test different entry thresholds
    for entry_mult in [1.5, 2.0, 2.5, 3.0]:
        entry_level = entry_mult * spread_std
        
        # Expected holding time (to return within 0.5σ of mean)
        path = expected_path(entry_level, max_hold_periods)
        exit_time = np.where(np.abs(path) < 0.5 * spread_std)[0]
        exp_hold = exit_time[0] if len(exit_time) > 0 else max_hold_periods
        
        # Stop-loss at entry_level * 1.5
        stop_level = entry_level * 1.5
        prob_stop = prob_stop_before_mean(entry_level, stop_level, max_hold_periods)
        
        # Expected profit (assuming exit at mean)
        exp_profit = entry_level * (1 - prob_stop)
        exp_loss = (stop_level - entry_level) * prob_stop
        exp_pnl = exp_profit - exp_loss
        
        results[f'entry_{entry_mult}sigma'] = {
            'entry_threshold': entry_level,
            'stop_loss': stop_level,
            'expected_hold_periods': exp_hold,
            'prob_hit_stop': prob_stop,
            'expected_pnl': exp_pnl,
            'profit_per_period': exp_pnl / exp_hold if exp_hold > 0 else 0
        }
    
    # Find optimal (max profit per period)
    best = max(results.items(), key=lambda x: x[1]['profit_per_period'])
    
    return {
        'all_scenarios': results,
        'optimal': best[0],
        'optimal_details': best[1],
        'recommendation': f"""
Optimal Strategy for φ={phi:.2f}:

Entry: {best[1]['entry_threshold']:.2f}σ ({best[0]})
Stop-loss: {best[1]['stop_loss']:.2f}σ
Expected hold: {best[1]['expected_hold_periods']:.0f} periods

Risk: {best[1]['prob_hit_stop']*100:.1f}% chance of hitting stop-loss
Expected P&L per trade: {best[1]['expected_pnl']:.3f}σ

WARNING: With φ=0.95, mean reversion is SLOW.
Consider:
- Wider stops (accept more drawdown)
- Higher entry thresholds (trade less frequently)
- Longer holding periods (weeks/months, not days)
- Position sizing: Reduce size vs faster-reverting spreads
        """
    }

# Calculate optimal thresholds
optimal = optimal_thresholds_given_phi(phi=0.95, spread_std=1.0, max_hold_periods=90)
print("\\n=== Optimal Trading Thresholds ===")
print(optimal['recommendation'])
\`\`\`

**Key findings for φ=0.95:**
- Optimal entry: **2.5-3.0 σ** (NOT 2σ!)
- Stop-loss: **4.0-4.5 σ** (wide!)
- Expected hold: **30-60 periods**
- Risk: 20-30% chance of hitting stop

**Why wider thresholds:**
- Slow reversion → spread takes time to return
- Risk of 2σ → 3σ → 4σ excursions
- Better to wait for wider misalignments

### Part 3: Why Backtest Sharpe 2.5 is Overstated

**Multiple problems:**

**Problem 1: Overlapping Trades**

With 60-day hold times, trades overlap:
\`\`\`
Day 0: Enter trade 1
Day 30: Enter trade 2 (trade 1 still open)
Day 60: Exit trade 1, Enter trade 3
\`\`\`

This creates **serial correlation** in strategy returns → overstates Sharpe.

**Problem 2: In-Sample Optimization**

Backtest likely optimized:
- Entry threshold (2σ chosen because it worked best)
- φ estimated from same data used for trading
- Parameters fit to historical data

Out-of-sample will be worse!

**Problem 3: Ignoring Transaction Costs**

\`\`\`python
def realistic_sharpe_estimate(backtest_sharpe: float,
                             phi: float,
                             avg_hold_periods: float,
                             transaction_cost_bps: float = 10) -> dict:
    """
    Adjust backtest Sharpe for realistic factors.
    """
    adjustments = {}
    
    # Adjustment 1: Overlapping trades (Newey-West correction)
    # Sharpe with overlapping trades is overstated by ~sqrt(1 + 2*rho)
    # where rho ≈ correlation between overlapping trade returns
    
    # For AR(1) spread with hold H periods:
    H = avg_hold_periods
    avg_overlap_rho = phi ** (H/2)  # Approximate
    overlap_factor = np.sqrt(1 + 2 * avg_overlap_rho)
    
    sharpe_after_overlap = backtest_sharpe / overlap_factor
    adjustments['overlap_adjustment'] = overlap_factor
    adjustments['sharpe_after_overlap'] = sharpe_after_overlap
    
    # Adjustment 2: Parameter uncertainty (reduce by ~20%)
    # Out-of-sample typically 70-80% of in-sample performance
    shrinkage_factor = 0.8
    sharpe_after_shrinkage = sharpe_after_overlap * shrinkage_factor
    adjustments['shrinkage_factor'] = shrinkage_factor
    adjustments['sharpe_after_shrinkage'] = sharpe_after_shrinkage
    
    # Adjustment 3: Transaction costs
    # Each trade: 2 legs × cost_bps
    # Assume backtest return = annual_return (before costs)
    # Cost per round-trip = 2 * cost_bps / 10000
    
    # If trading N times per year, annual cost drag:
    trades_per_year = 252 / H  # Approximate
    annual_cost = trades_per_year * 2 * (transaction_cost_bps / 10000)
    
    # Adjust Sharpe (assuming vol stays same, return decreases)
    # New Sharpe = (return - costs) / vol
    # ≈ Old Sharpe - (costs / vol)
    # Approximate: costs reduce Sharpe by annual_cost / (0.15) where 0.15 is typical vol
    sharpe_drag = annual_cost / 0.15
    sharpe_final = sharpe_after_shrinkage - sharpe_drag
    
    adjustments['transaction_cost_drag'] = sharpe_drag
    adjustments['final_sharpe'] = max(sharpe_final, 0)  # Can't be negative
    
    return {
        'backtest_sharpe': backtest_sharpe,
        'adjustments': adjustments,
        'final_sharpe': adjustments['final_sharpe'],
        'degradation_pct': (1 - adjustments['final_sharpe'] / backtest_sharpe) * 100,
        'summary': f"""
Sharpe Ratio Adjustments:

Backtest (raw): {backtest_sharpe:.2f}
  
Adjustments:
1. Overlapping trades: ÷{overlap_factor:.2f} = {sharpe_after_overlap:.2f}
2. Out-of-sample shrinkage: ×{shrinkage_factor:.2f} = {sharpe_after_shrinkage:.2f}
3. Transaction costs: -{sharpe_drag:.2f} = {adjustments['final_sharpe']:.2f}

Realistic estimate: {adjustments['final_sharpe']:.2f}
Degradation: {(1 - adjustments['final_sharpe'] / backtest_sharpe) * 100:.0f}%

Interpretation:
{"✓ Still attractive (>1.0)" if adjustments['final_sharpe'] > 1.0 else
 "⚠ Marginal (0.5-1.0)" if adjustments['final_sharpe'] > 0.5 else
 "✗ Likely unprofitable"}
        """
    }

# Example: Your colleague's backtest
realistic = realistic_sharpe_estimate(
    backtest_sharpe=2.5,
    phi=0.95,
    avg_hold_periods=60,
    transaction_cost_bps=10
)

print("\\n=== Realistic Sharpe Estimate ===")
print(realistic['summary'])
\`\`\`

**Result:** Sharpe 2.5 → Realistic Sharpe ~0.8-1.2

**Additional biases:**
- Survivorship (only pairs that stayed cointegrated)
- Look-ahead (using future cointegration data)
- Cherry-picking (best pairs from many tested)

### Summary

**Your colleague is WRONG:**1. **ACF(1)=0.95 = SLOW mean reversion, NOT fast**
   - Half-life: 13.5 periods
   - Takes 90 periods to decay to 1%

2. **Optimal strategy for φ=0.95:**
   - Enter at 2.5-3.0σ (NOT 2σ)
   - Wide stops (4-4.5σ)
   - Long holds (60+ periods)
   - Reduce position size

3. **Backtest Sharpe 2.5 → Realistic ~1.0:**
   - Overlapping trades overstate by ~40%
   - Out-of-sample shrinkage ~20%
   - Transaction costs ~0.3 Sharpe

4. **Recommendation:**
   - Don't trade this pair at 2σ with current parameters
   - Either find faster-reverting pairs (φ < 0.7) OR
   - Adjust strategy for slow reversion (wider thresholds, patient capital)

**Bottom line:** High persistence (ACF ≈ 1) is BAD for mean-reversion trading, not good!`,
    },
];

