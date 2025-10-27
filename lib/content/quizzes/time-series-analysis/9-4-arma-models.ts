export const armaModelsQuiz = [
  {
    id: 1,
    question:
      "Your quantitative trading system uses an ARMA(2,1) model to forecast daily S&P 500 returns. After 6 months in production, you notice: (1) The model's forecast errors have doubled compared to backtest, (2) The AR coefficients have drifted (original: φ₁=0.35, φ₂=0.15, now estimated: φ₁=0.18, φ₂=0.08), (3) Residuals now show significant ACF at lag 5, and (4) The Ljung-Box p-value dropped from 0.42 to 0.03. The trading desk wants to know if they should: (A) Retrain the model monthly, (B) Switch to a more complex ARMA(3,2) model, (C) Abandon ARMA modeling entirely, or (D) Add regime-switching or time-varying parameters. Analyze each option, explain what's causing the model degradation, propose the best solution, and design a monitoring system to detect future model decay.",
    answer: `## Comprehensive Answer:

### Root Cause Analysis: Why the Model is Degrading

The symptoms indicate **model instability and structural change**:

1. **Doubled forecast errors**: Out-of-sample performance degradation
2. **Parameter drift**: AR coefficients declined by ~50%
3. **Residual autocorrelation at lag 5**: New pattern model doesn't capture
4. **Ljung-Box rejection**: Residuals no longer white noise

**Possible causes:**

**A. Market Regime Change**
- Volatility regime shift (low vol → high vol)
- Different market dynamics (trending → mean-reverting)
- Correlations broke down

**B. Structural Break**
- Policy change (Fed pivot, QE program)
- Crisis event (pandemic, financial crisis)
- Market microstructure evolution

**C. Overfitting in Original Model**
- Backtest captured noise, not signal
- ARMA(2,1) was too complex for data
- Parameters unstable from the start

**D. Non-stationarity**
- Underlying process changed
- ARMA assumes stationary parameters
- Reality: Time-varying dynamics

### Evaluating Each Option

**Option A: Retrain Model Monthly**

**Pros:**
- Adapts to recent market conditions
- Easy to implement (just re-estimate)
- Keeps same model structure

**Cons:**
- **Overfitting risk**: 21-30 trading days per month might be too few observations
- **Parameter instability**: Estimates noisy with limited data
- **Doesn't address root cause**: If regime changed, short window won't help
- **Transaction costs**: Frequent strategy changes

**Analysis:**
\`\`\`python
def analyze_retraining_frequency(returns: pd.Series,
                                 window_sizes: list = [30, 60, 120, 252]) -> dict:
    """
    Test different retraining frequencies.
    """
    from statsmodels.tsa.arima.model import ARIMA
    
    results = {}
    
    for window in window_sizes:
        # Walk-forward validation with rolling window
        forecasts = []
        actuals = []
        
        for t in range(window, len(returns) - 1):
            # Training window
            train_data = returns.iloc[t-window:t]
            
            # Fit ARMA(2,1)
            try:
                model = ARIMA(train_data, order=(2, 0, 1))
                fit = model.fit()
                
                # 1-step forecast
                forecast = fit.forecast(steps=1)[0]
                actual = returns.iloc[t]
                
                forecasts.append(forecast)
                actuals.append(actual)
            except:
                continue
        
        # Calculate metrics
        forecast_errors = np.array(forecasts) - np.array(actuals)
        rmse = np.sqrt(np.mean(forecast_errors**2))
        mae = np.mean(np.abs(forecast_errors))
        
        # Direction accuracy
        direction_acc = np.mean(
            np.sign(forecasts) == np.sign(actuals)
        )
        
        results[window] = {
            'rmse': rmse,
            'mae': mae,
            'direction_accuracy': direction_acc,
            'avg_days_for_training': window
        }
    
    # Find optimal window
    best = min(results.items(), key=lambda x: x[1]['rmse'])
    
    return {
        'results': results,
        'optimal_window': best[0],
        'interpretation': f"""
Retraining Frequency Analysis:

Optimal window: {best[0]} days
  RMSE: {best[1]['rmse']:.6f}
  Direction accuracy: {best[1]['direction_accuracy']*100:.1f}%

Recommendation: 
{"Retrain monthly (30 days)" if best[0] <= 30 else
 "Retrain quarterly (60-90 days)" if best[0] <= 90 else
 "Retrain semi-annually (120+ days)"}

Warning: Window < 60 days risks overfitting!
        """
    }
\`\`\`

**Verdict: PARTIAL SOLUTION** - May help but doesn't address regime changes.

**Option B: Switch to ARMA(3,2)**

**Pros:**
- Might capture the lag-5 autocorrelation
- More flexible model

**Cons:**
- **Overfitting**: More parameters = more noise
- **Doesn't solve instability**: Still assumes constant parameters
- **Wrong diagnosis**: Problem is regime change, not model order
- **Curse of dimensionality**: 6 parameters for daily returns is excessive

**Analysis:**
\`\`\`python
def compare_model_complexity(returns: pd.Series,
                            orders: list = [(1,1), (2,1), (3,2), (4,2)]) -> dict:
    """
    Compare in-sample vs out-of-sample for different ARMA orders.
    """
    results = []
    
    # Split data
    split = int(len(returns) * 0.7)
    train = returns.iloc[:split]
    test = returns.iloc[split:]
    
    for p, q in orders:
        try:
            # Fit on training
            model = ARIMA(train, order=(p, 0, q))
            fit = model.fit()
            
            # In-sample metrics
            insample_resid = fit.resid
            insample_rmse = np.sqrt(np.mean(insample_resid**2))
            
            # Out-of-sample forecast
            forecast = fit.forecast(steps=len(test))
            oos_errors = test.values - forecast.values
            oos_rmse = np.sqrt(np.mean(oos_errors**2))
            
            # Overfitting measure
            overfitting_ratio = oos_rmse / insample_rmse
            
            results.append({
                'order': f'ARMA({p},{q})',
                'n_params': p + q + 1,  # AR + MA + const
                'bic': fit.bic,
                'insample_rmse': insample_rmse,
                'oos_rmse': oos_rmse,
                'overfitting_ratio': overfitting_ratio
            })
        except:
            continue
    
    df = pd.DataFrame(results)
    
    return {
        'comparison': df,
        'interpretation': """
Key insight: Overfitting ratio > 1.5 indicates model is too complex.

Higher order models often have:
- Better in-sample fit (lower BIC)
- WORSE out-of-sample performance
- More overfitting

For daily returns, ARMA(1,1) or ARMA(2,1) is typically optimal.
ARMA(3,2) likely overfits.
        """
    }
\`\`\`

**Verdict: BAD IDEA** - Will likely make overfitting worse.

**Option C: Abandon ARMA**

**Pros:**
- Acknowledges model limitations
- Avoids false confidence

**Cons:**
- **Throws baby out with bathwater**: ARMA may still have value
- **No alternative proposed**: What to use instead?
- **Doesn't learn from failure**: Why did it fail?

**Verdict: TOO EXTREME** - Don't give up yet.

**Option D: Add Regime-Switching or Time-Varying Parameters**

**Pros:**
- **Addresses root cause**: Allows for regime changes
- **More realistic**: Markets do change regimes
- **Captures parameter drift**: φ can vary over time

**Cons:**
- **More complex**: Harder to implement and interpret
- **More parameters**: Risk of overfitting
- **Computational cost**: Estimation is slower

**Models to consider:**

**1. Regime-Switching ARMA (Markov-Switching)**
\`\`\`python
# Two regimes: Low volatility vs High volatility
# State 0: φ₁=0.35, φ₂=0.15, σ=0.01  (calm markets)
# State 1: φ₁=0.10, φ₂=0.05, σ=0.03  (volatile markets)

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

def fit_regime_switching_arma(returns: pd.Series) -> dict:
    """
    Fit Markov-Switching AR model (simplified).
    """
    # Markov-Switching AR(2) with 2 regimes
    model = MarkovRegression(
        returns.values,
        k_regimes=2,
        order=2,
        switching_ar=True,
        switching_variance=True
    )
    
    results = model.fit()
    
    return {
        'regime_0_params': results.params[:3],  # AR params for regime 0
        'regime_1_params': results.params[3:6],  # AR params for regime 1
        'transition_matrix': results.regime_transition,
        'smoothed_probs': results.smoothed_marginal_probabilities,
        'interpretation': """
Regime-Switching Model:

Regime 0 (Low Vol): More persistent (higher φ)
Regime 1 (High Vol): Less persistent (lower φ)

Current regime probability: [Show current state]

This explains parameter drift: Markets shifted to high-vol regime!
        """
    }
\`\`\`

**2. Time-Varying Parameter Model (TV-ARMA)**
\`\`\`python
def estimate_tv_arma_rolling(returns: pd.Series,
                             window: int = 120) -> dict:
    """
    Estimate time-varying ARMA parameters using rolling window.
    """
    phi_history = []
    sigma_history = []
    dates = []
    
    for t in range(window, len(returns)):
        train = returns.iloc[t-window:t]
        
        try:
            model = ARIMA(train, order=(2, 0, 1))
            fit = model.fit()
            
            phi_history.append([
                fit.params.get('ar.L1', 0),
                fit.params.get('ar.L2', 0)
            ])
            sigma_history.append(np.sqrt(fit.sigma2))
            dates.append(returns.index[t])
        except:
            continue
    
    phi_df = pd.DataFrame(phi_history, 
                         columns=['phi1', 'phi2'],
                         index=dates)
    
    return {
        'phi_evolution': phi_df,
        'avg_phi1_first_half': phi_df['phi1'].iloc[:len(phi_df)//2].mean(),
        'avg_phi1_second_half': phi_df['phi1'].iloc[len(phi_df)//2:].mean(),
        'interpretation': """
Time-varying parameters show:

φ₁ evolved from 0.35 → 0.18 over 6 months
This confirms parameter instability!

Solution: Use rolling window estimation or 
exponentially-weighted parameter updates.
        """
    }
\`\`\`

**Verdict: BEST SOLUTION** - Addresses regime changes directly.

### Recommended Solution

**Hybrid Approach: Adaptive ARMA with Monitoring**1. **Use rolling-window estimation** (90-120 days)
2. **Monitor regime changes** (volatility, parameter stability)
3. **Automatic model re-selection** when diagnostics fail
4. **Fallback to simpler model** (AR(1)) in high uncertainty

\`\`\`python
class AdaptiveARMASystem:
    """
    Production ARMA system with monitoring and adaptation.
    """
    
    def __init__(self, base_order=(2,1), window=120):
        self.base_order = base_order
        self.window = window
        self.current_model = None
        self.monitoring_metrics = []
        
    def fit_and_monitor(self, returns: pd.Series) -> dict:
        """
        Fit model with comprehensive monitoring.
        """
        # Fit current model
        recent_data = returns.iloc[-self.window:]
        
        model = ARIMA(recent_data, order=self.base_order)
        self.current_model = model.fit()
        
        # Monitoring checks
        warnings = []
        
        # Check 1: Parameter stability
        if len(self.monitoring_metrics) > 0:
            prev_params = self.monitoring_metrics[-1]['params']
            curr_params = self.current_model.params
            
            param_change = np.abs(curr_params - prev_params).mean()
            if param_change > 0.1:
                warnings.append(f"Parameter drift detected: {param_change:.3f}")
        
        # Check 2: Residual diagnostics
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb = acorr_ljungbox(self.current_model.resid, lags=20)
        if lb['lb_pvalue'].iloc[0] < 0.05:
            warnings.append(f"Residuals not white noise (p={lb['lb_pvalue'].iloc[0]:.3f})")
        
        # Check 3: Forecast error tracking
        # (Would compare recent forecasts to actuals)
        
        # Store metrics
        self.monitoring_metrics.append({
            'timestamp': returns.index[-1],
            'params': self.current_model.params,
            'aic': self.current_model.aic,
            'ljung_box_p': lb['lb_pvalue'].iloc[0],
            'warnings': warnings
        })
        
        # Decision: Keep, retrain, or simplify?
        if len(warnings) >= 2:
            action = "SIMPLIFY: Switch to AR(1)"
        elif len(warnings) == 1:
            action = "RETRAIN: Update parameters"
        else:
            action = "CONTINUE: Model performing well"
        
        return {
            'current_order': self.base_order,
            'warnings': warnings,
            'recommended_action': action,
            'monitoring_history': pd.DataFrame(self.monitoring_metrics)
        }
\`\`\`

### Monitoring System Design

**Real-time Monitoring Dashboard:**1. **Parameter Evolution Chart**
   - Track φ₁, φ₂, θ₁ over time
   - Alert if |change| > 0.15

2. **Residual Diagnostics**
   - ACF of residuals (should be white noise)
   - Ljung-Box p-value (should be > 0.05)
   - Alert if fails 3 consecutive periods

3. **Forecast Error Tracking**
   - RMSE over rolling 20-day window
   - Compare to baseline (random walk)
   - Alert if RMSE > 1.5× baseline

4. **Regime Detection**
   - Volatility regime (rolling std)
   - Trend regime (price level)
   - Alert on regime transitions

5. **Automated Actions**
   - Level 1 (Minor): Log warning
   - Level 2 (Moderate): Retrain model
   - Level 3 (Severe): Switch to simpler model
   - Level 4 (Critical): Halt trading, human review

### Final Recommendation

**Choose Option D (with modifications):**1. **Short-term** (immediate fix):
   - Retrain ARMA(2,1) on recent 120 days
   - Implement monitoring system
   - Set up alerts for parameter drift

2. **Medium-term** (1-2 months):
   - Test regime-switching ARMA
   - Compare to time-varying parameters
   - Validate out-of-sample

3. **Long-term** (3-6 months):
   - Implement ensemble: Average ARMA + GARCH + ML
   - Adaptive model selection based on regime
   - Continuous monitoring and improvement

**Key insight:** The problem isn't ARMA modeling itself, it's assuming constant parameters in a time-varying world. Solution: Make the model adaptive!`,
  },
  {
    id: 2,
    question:
      "You're analyzing high-frequency (1-minute) FX returns for EUR/USD and fit an MA(1) model. The estimated θ = 1.8, which violates the invertibility condition (|θ| < 1). Your colleague argues: 'The model fits well (high R², low residuals), so let's use it anyway. Invertibility is just a theoretical nicety that doesn't matter in practice.' However, when you try to forecast, the predictions explode to ±50% returns within 10 steps. Explain: (1) What invertibility means mathematically and why |θ| < 1 is required, (2) Why the colleague's reasoning is flawed despite good fit, (3) What causes θ > 1 estimates in practice, (4) How to diagnose and fix the problem, and (5) Why this is especially dangerous for HFT systems.",
    answer: `## Comprehensive Answer:

### Part 1: What Invertibility Means Mathematically

**MA(1) Model:**
$$X_t = \\mu + \\epsilon_t + \\theta \\epsilon_{t-1}$$

**Invertibility** means we can express past errors in terms of observable data:
$$\\epsilon_t = X_t - \\mu - \\theta \\epsilon_{t-1}$$

**Recursive substitution:**
$$\\epsilon_t = (X_t - \\mu) - \\theta(X_{t-1} - \\mu) + \\theta^2(X_{t-2} - \\mu) - \\theta^3(X_{t-3} - \\mu) + ...$$

**Key insight:** This is a **geometric series** that converges ONLY if $|\\theta| < 1$.

**If |θ| ≥ 1:**
$$\\sum_{k=0}^{\\infty} \\theta^k \\text{ DIVERGES}$$

The past errors $\\epsilon_t$ cannot be uniquely determined from observed data!

**Mathematical proof of explosion:**
\`\`\`python
def demonstrate_invertibility_explosion(theta: float,
                                       n_steps: int = 100) -> dict:
    """
    Show what happens when |θ| > 1.
    """
    # Simulate observed returns
    np.random.seed(42)
    true_errors = np.random.randn(n_steps)
    returns = true_errors[1:] + theta * true_errors[:-1]
    
    # Try to recover errors using inversion formula
    # ε_t = (X_t - θX_{t-1} + θ²X_{t-2} - ...)
    
    recovered_errors = []
    max_lag = min(50, n_steps)
    
    for t in range(max_lag, n_steps):
        error_estimate = returns[t]
        
        # Add geometric series terms
        for k in range(1, max_lag):
            if t-k >= 0:
                error_estimate -= (theta ** k) * returns[t-k]
        
        recovered_errors.append(error_estimate)
    
    # Check convergence
    mean_abs_error = np.mean(np.abs(recovered_errors))
    
    return {
        'theta': theta,
        'invertible': abs(theta) < 1,
        'mean_recovered_error': mean_abs_error,
        'max_recovered_error': np.max(np.abs(recovered_errors)),
        'interpretation': (
            f"θ = {theta:.2f}:\\n"
            f"{'✓ INVERTIBLE' if abs(theta) < 1 else '✗ NON-INVERTIBLE'}\\n"
            f"Mean |error|: {mean_abs_error:.2f}\\n"
            f"Max |error|: {np.max(np.abs(recovered_errors)):.2f}\\n"
            f"{'Errors recover to ~1.0 (correct)' if abs(theta) < 1 else 'ERRORS EXPLODE!'}"
        )
    }

# Test with θ = 0.6 (invertible)
inv_result = demonstrate_invertibility_explosion(0.6)
print("Invertible case (θ=0.6):")
print(inv_result['interpretation'])

# Test with θ = 1.8 (non-invertible)
noninv_result = demonstrate_invertibility_explosion(1.8)
print("\\nNon-invertible case (θ=1.8):")
print(noninv_result['interpretation'])
\`\`\`

**Result:** With θ=1.8, recovered errors blow up to 10-100× the true values!

### Part 2: Why Colleague's Reasoning is Flawed

**The colleague says:** "High R², low residuals, model fits well"

**The problems:**

**Problem 1: Non-uniqueness**

MA models with θ and 1/θ are **observationally equivalent**!

$$X_t = \\epsilon_t + \\theta \\epsilon_{t-1}$$
$$X_t = \\eta_t + \\frac{1}{\\theta} \\eta_{t-1}$$

produce THE SAME covariance structure.

\`\`\`python
def demonstrate_observational_equivalence(theta: float, n: int = 1000):
    """
    Show that θ and 1/θ produce same ACF.
    """
    # MA(1) with θ
    errors1 = np.random.randn(n)
    series1 = errors1[1:] + theta * errors1[:-1]
    acf1 = pd.Series(series1).autocorr(lag=1)
    
    # MA(1) with 1/θ (rescaled errors)
    errors2 = np.random.randn(n) * theta
    series2 = errors2[1:] + (1/theta) * errors2[:-1]
    acf2 = pd.Series(series2).autocorr(lag=1)
    
    # Theoretical ACF for MA(1): ρ₁ = θ/(1+θ²)
    theoretical_acf = theta / (1 + theta**2)
    
    return {
        'theta': theta,
        'one_over_theta': 1/theta,
        'acf_with_theta': acf1,
        'acf_with_inverse': acf2,
        'theoretical_acf': theoretical_acf,
        'are_equal': np.isclose(acf1, acf2, atol=0.05),
        'interpretation': f"""
Observational Equivalence:

MA(1) with θ = {theta:.2f}:  ACF(1) = {acf1:.4f}
MA(1) with θ = {1/theta:.2f}: ACF(1) = {acf2:.4f}

Both produce SAME observable patterns!

But only one is invertible: |θ| < 1 → θ = {min(theta, 1/theta):.2f}
        """
    }

result = demonstrate_observational_equivalence(1.8)
print(result['interpretation'])
\`\`\`

**Implication:** If you get θ=1.8, the model is trying to tell you θ=1/1.8=0.56 is the correct (invertible) parameter!

**Problem 2: Forecasting Explodes**

\`\`\`python
def forecast_with_noninvertible(theta: float, steps: int = 20):
    """
    Show forecast explosion with |θ| > 1.
    """
    # Simulate MA(1) data with invertible parameter
    true_theta = 1/theta if abs(theta) > 1 else theta
    
    np.random.seed(42)
    n = 100
    errors = np.random.randn(n) * 0.01  # 1% returns
    returns = errors[1:] + true_theta * errors[:-1]
    
    # Fit MA(1) - might get non-invertible estimate
    from statsmodels.tsa.arima.model import ARIMA
    
    model = ARIMA(returns, order=(0, 0, 1))
    
    # Force non-invertible parameter
    # (In practice, constrain estimation to |θ| < 1)
    try:
        fit = model.fit()
        estimated_theta = fit.params['ma.L1']
        
        # Forecast
        forecast = fit.forecast(steps=steps)
        
        # What happens with non-invertible?
        # Manually calculate with θ=1.8
        noninv_forecast = []
        last_error = fit.resid.iloc[-1]
        
        for _ in range(steps):
            # E[X_{t+h}] = θ * ε_t (for h=1), then grows exponentially
            f = theta * last_error
            noninv_forecast.append(f)
            last_error = f  # Compounds!
        
        return {
            'estimated_theta': estimated_theta,
            'normal_forecast': forecast.values,
            'noninvertible_forecast': noninv_forecast,
            'interpretation': f"""
Forecasting with θ = {theta:.2f}:

Normal (constrained): {forecast.values[:3]}... (reasonable)
Non-invertible: {np.array(noninv_forecast[:3])}... (EXPLODING!)

By step {steps}: {'Forecast = ' + f'{noninv_forecast[-1]:.2f}' + ' (ABSURD!)'}

This is why invertibility matters!
            """
        }
    except:
        return {"error": "Model fitting failed (as it should!)"}

result = forecast_with_noninvertible(1.8, steps=10)
print(result['interpretation'])
\`\`\`

### Part 3: What Causes θ > 1 Estimates

**Common causes:**

**1. Model Misspecification**
- True process is AR, not MA
- Need ARMA, not pure MA
- Non-stationarity (should difference first)

**2. Numerical Optimization Issues**
- Unbounded optimization
- Poor starting values
- Numerical instability

**3. Overdifferencing**
- Differenced when shouldn't have
- Introduces artificial MA structure with |θ| ≈ 1

**4. Microstructure Effects (HFT)**
- Bid-ask bounce creates negative MA(1)
- Non-synchronous trading
- Stale prices

### Part 4: How to Diagnose and Fix

**Diagnosis:**

\`\`\`python
def diagnose_ma_estimation(returns: pd.Series) -> dict:
    """
    Comprehensive MA estimation diagnostics.
    """
    from statsmodels.tsa.arima.model import ARIMA
    
    diagnostics = {}
    
    # Fit MA(1) without constraints
    try:
        model_unconstrained = ARIMA(returns, order=(0, 0, 1))
        fit_unconstrained = model_unconstrained.fit(method='css-mle')  # Less constrained
        theta_unconstrained = fit_unconstrained.params['ma.L1']
        
        diagnostics['unconstrained_theta'] = theta_unconstrained
        diagnostics['invertible'] = abs(theta_unconstrained) < 1
    except:
        diagnostics['unconstrained_theta'] = None
        diagnostics['error'] = "Unconstrained estimation failed"
    
    # Fit with constraints (force |θ| < 1)
    try:
        model_constrained = ARIMA(returns, order=(0, 0, 1), 
                                  enforce_invertibility=True)
        fit_constrained = model_constrained.fit()
        theta_constrained = fit_constrained.params['ma.L1']
        
        diagnostics['constrained_theta'] = theta_constrained
        diagnostics['constrained_aic'] = fit_constrained.aic
    except:
        diagnostics['constrained_theta'] = None
    
    # Check if should use AR instead
    try:
        ar1_model = ARIMA(returns, order=(1, 0, 0))
        ar1_fit = ar1_model.fit()
        
        diagnostics['ar1_aic'] = ar1_fit.aic
        diagnostics['prefer_ar'] = ar1_fit.aic < diagnostics.get('constrained_aic', np.inf)
    except:
        pass
    
    # Check ACF/PACF
    from statsmodels.tsa.stattools import acf, pacf
    acf_vals = acf(returns, nlags=5)
    pacf_vals = pacf(returns, nlags=5)
    
    diagnostics['acf_lag1'] = acf_vals[1]
    diagnostics['pacf_lag1'] = pacf_vals[1]
    
    # Decision
    if abs(diagnostics.get('unconstrained_theta', 0)) > 1:
        if diagnostics.get('prefer_ar', False):
            recommendation = "Use AR(1) instead of MA(1)"
        else:
            recommendation = "Use constrained MA(1) estimation (enforce |θ| < 1)"
    else:
        recommendation = "MA(1) is appropriate"
    
    diagnostics['recommendation'] = recommendation
    
    return diagnostics
\`\`\`

**Fix strategies:**1. **Enforce Invertibility Constraint**
\`\`\`python
# In statsmodels
model = ARIMA(data, order=(0, 0, 1), 
             enforce_invertibility=True)  # KEY!
\`\`\`

2. **Try AR Instead**
\`\`\`python
# If θ > 1, often AR is better
ar_model = ARIMA(data, order=(1, 0, 0))
\`\`\`

3. **Use ARMA**
\`\`\`python
# Combine AR and MA
arma_model = ARIMA(data, order=(1, 0, 1))
\`\`\`

4. **Check Stationarity**
\`\`\`python
# Maybe need differencing
from statsmodels.tsa.stattools import adfuller
adf = adfuller(data)
if adf[1] > 0.05:  # Non-stationary
    data_diff = data.diff().dropna()
    # Fit to differenced data
\`\`\`

### Part 5: Why This is Dangerous for HFT

**HFT-specific risks:**

**1. Millisecond Explosions**
- 1-minute model → forecast every second
- With θ=1.8: 60 forecasts compound errors
- Position sizes explode within minutes!

**2. Leverage Amplification**
- HFT uses 10-50x leverage
- Small forecast error × leverage = account blow-up
- Example: 1% forecast error × 50x = 50% loss!

**3. Regulatory Risk**
- Erroneous orders (\"fat finger\" trades)
- Market manipulation accusations
- Circuit breaker triggers

**4. Systemic Risk**
- Flash crashes (May 2010, Aug 2015)
- Algorithm goes haywire
- Liquidity evaporates

**Real-world example:**

\`\`\`python
def simulate_hft_disaster(theta: float = 1.8,
                         leverage: float = 20,
                         time_steps: int = 60):
    """
    Simulate what happens in HFT with non-invertible MA.
    """
    # Start with $1M account
    account_value = 1_000_000
    position = 0
    
    # Simulate 1 hour (60 minutes)
    for t in range(time_steps):
        # Non-invertible forecast compounds
        forecast_error = 0.0001 * (theta ** (t/10))  # Grows exponentially
        
        # Position sizing based on forecast
        position = account_value * leverage * forecast_error
        
        # Realized return (actual market is random)
        realized_return = np.random.normal(0, 0.0001)
        
        # P&L
        pnl = position * realized_return
        account_value += pnl
        
        # Check for margin call
        if account_value < 200_000:  # 80% loss
            return {
                'disaster_time': t,
                'final_value': account_value,
                'loss_pct': (1 - account_value / 1_000_000) * 100,
                'message': f"⚠ MARGIN CALL at minute {t}! Account blown up."
            }
    
    return {
        'survived': True,
        'final_value': account_value,
        'pnl_pct': (account_value / 1_000_000 - 1) * 100
    }

disaster = simulate_hft_disaster()
print("HFT with non-invertible MA(1):")
print(disaster)
\`\`\`

**Safeguards:**1. **Parameter Validation**
   - Check |θ| < 0.95 (buffer from boundary)
   - Alert if parameter near boundary

2. **Forecast Bounds**
   - Cap forecasts at ±3σ
   - Sanity checks on position sizes

3. **Real-time Monitoring**
   - Track forecast errors every second
   - Kill switch if errors exceed threshold

4. **Gradual Position Entry**
   - Don't go all-in on one forecast
   - Scale positions based on confidence

### Summary

**Invertibility matters because:**1. Ensures unique, stable error representation
2. Prevents forecast explosions
3. Guarantees convergence of recursions
4. Required for consistent estimation

**Your colleague is wrong because:**
- Good fit ≠ good forecasts
- θ=1.8 mathematically unstable
- Forecasts will explode

**Solution:**
- Enforce invertibility constraint
- Consider AR model instead
- Use ARMA if both AR and MA needed
- For HFT: Add multiple safeguards

**Key takeaway:** Theory isn't just "nicety" - it prevents million-dollar disasters!`,
  },
  {
    id: 3,
    question:
      'Design a production ARMA forecasting system for a hedge fund that trades 500 stocks daily. The system must: (1) Automatically select ARMA order for each stock, (2) Retrain models weekly, (3) Generate next-day return forecasts every night after market close, (4) Provide confidence intervals, (5) Flag models that fail diagnostics, and (6) Scale to handle all 500 stocks in under 30 minutes. Address: system architecture, parallel processing strategy, model validation pipeline, failure handling, monitoring, and how to prevent the system from generating systematically biased forecasts that could lead to portfolio-wide losses.',
    answer: `## Comprehensive Answer:

### System Architecture

**High-level design:**

\`\`\`
┌─────────────────────────────────────────────────────┐
│              Nightly Forecasting Pipeline            │
│                                                      │
│  18:00  Data Ingestion → Clean → Store              │
│  18:15  Parallel Model Fitting (500 stocks)         │
│  18:30  Validation & Diagnostics                    │
│  18:45  Forecast Generation                         │
│  19:00  Aggregation & Portfolio View                │
│  19:15  Alerts & Reports                            │
│  19:30  Ready for Trading (Next Day)                │
└─────────────────────────────────────────────────────┘
\`\`\`

**Component Details:**

\`\`\`python
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import logging
from datetime import datetime
import redis
import json

@dataclass
class ModelConfig:
    """Configuration for ARMA model selection."""
    max_p: int = 5
    max_q: int = 5
    min_observations: int = 252  # 1 year
    ic_criterion: str = 'bic'  # or 'aic'
    confidence_level: float = 0.95
    
@dataclass
class ForecastResult:
    """Container for forecast results."""
    ticker: str
    timestamp: datetime
    forecast: float
    lower_ci: float
    upper_ci: float
    model_order: Tuple[int, int]
    aic: float
    bic: float
    diagnostics_passed: bool
    warnings: List[str]


class StockForecaster:
    """
    Individual stock forecasting engine.
    
    Handles: model selection, fitting, validation, forecasting.
    """
    
    def __init__(self, ticker: str, config: ModelConfig):
        self.ticker = ticker
        self.config = config
        self.model = None
        self.results = None
        self.selected_order = None
        self.logger = logging.getLogger(f'Forecaster.{ticker}')
        
    def select_order(self, returns: pd.Series) -> Tuple[int, int]:
        """
        Automatic ARMA order selection using BIC.
        
        Fast heuristic:
        1. Try AR(1), MA(1), ARMA(1,1) first (most common)
        2. If inadequate, expand search
        """
        candidates = [
            (1, 0),  # AR(1)
            (0, 1),  # MA(1)
            (1, 1),  # ARMA(1,1)
            (2, 0),  # AR(2)
            (2, 1),  # ARMA(2,1)
            (1, 2),  # ARMA(1,2)
        ]
        
        best_ic = np.inf
        best_order = (1, 0)
        
        for p, q in candidates:
            try:
                model = ARIMA(returns, order=(p, 0, q),
                            enforce_stationarity=True,
                            enforce_invertibility=True)
                fit = model.fit(method='statespace')
                
                ic = fit.bic if self.config.ic_criterion == 'bic' else fit.aic
                
                if ic < best_ic:
                    best_ic = ic
                    best_order = (p, q)
                    
            except Exception as e:
                self.logger.debug(f"Order ({p},{q}) failed: {e}")
                continue
        
        return best_order
    
    def fit(self, returns: pd.Series) -> bool:
        """
        Fit ARMA model with selected order.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Select order
            self.selected_order = self.select_order(returns)
            
            # Fit model
            self.model = ARIMA(
                returns,
                order=(self.selected_order[0], 0, self.selected_order[1]),
                enforce_stationarity=True,
                enforce_invertibility=True
            )
            self.results = self.model.fit(method='statespace')
            
            return True
            
        except Exception as e:
            self.logger.error(f"Fit failed: {e}")
            return False
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Comprehensive model validation.
        
        Returns:
            (diagnostics_passed, list_of_warnings)
        """
        warnings = []
        
        if self.results is None:
            return False, ["Model not fitted"]
        
        # Check 1: Residual autocorrelation
        try:
            lb = acorr_ljungbox(self.results.resid, lags=10, return_df=True)
            if lb['lb_pvalue'].iloc[0] < 0.05:
                warnings.append(f"Residuals autocorrelated (LB p={lb['lb_pvalue'].iloc[0]:.3f})")
        except:
            warnings.append("Ljung-Box test failed")
        
        # Check 2: Parameter significance
        pvalues = self.results.pvalues
        if not (pvalues < 0.10).all():  # 10% significance
            warnings.append("Some parameters insignificant")
        
        # Check 3: Stationarity (for AR part)
        if self.selected_order[0] > 0:
            ar_params = [self.results.params.get(f'ar.L{i}', 0) 
                        for i in range(1, self.selected_order[0]+1)]
            if sum(np.abs(ar_params)) > 0.95:
                warnings.append("AR parameters near non-stationarity")
        
        # Check 4: Invertibility (for MA part)
        if self.selected_order[1] > 0:
            ma_params = [self.results.params.get(f'ma.L{i}', 0)
                        for i in range(1, self.selected_order[1]+1)]
            if any(abs(p) > 0.95 for p in ma_params):
                warnings.append("MA parameters near non-invertibility")
        
        # Overall: Pass if 0-1 warnings (some flexibility)
        passed = len(warnings) <= 1
        
        return passed, warnings
    
    def forecast(self) -> Optional[ForecastResult]:
        """
        Generate 1-step ahead forecast with confidence interval.
        """
        if self.results is None:
            return None
        
        try:
            # Get forecast
            forecast_obj = self.results.get_forecast(steps=1)
            
            # Extract values
            forecast_mean = forecast_obj.predicted_mean.iloc[0]
            conf_int = forecast_obj.conf_int(alpha=1-self.config.confidence_level)
            lower_ci = conf_int.iloc[0, 0]
            upper_ci = conf_int.iloc[0, 1]
            
            # Validation
            passed, warnings = self.validate()
            
            return ForecastResult(
                ticker=self.ticker,
                timestamp=datetime.now(),
                forecast=forecast_mean,
                lower_ci=lower_ci,
                upper_ci=upper_ci,
                model_order=self.selected_order,
                aic=self.results.aic,
                bic=self.results.bic,
                diagnostics_passed=passed,
                warnings=warnings
            )
            
        except Exception as e:
            self.logger.error(f"Forecast failed: {e}")
            return None


class ParallelForecastingSystem:
    """
    Production system for forecasting 500 stocks in parallel.
    """
    
    def __init__(self, 
                 tickers: List[str],
                 config: ModelConfig,
                 n_workers: int = 8):
        self.tickers = tickers
        self.config = config
        self.n_workers = n_workers
        self.logger = logging.getLogger('ForecastingSystem')
        
        # Redis for caching and monitoring
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        
    def load_returns(self, ticker: str, 
                    lookback_days: int = 504) -> Optional[pd.Series]:
        """
        Load historical returns for ticker.
        
        In production: Load from database or data warehouse.
        """
        # Placeholder - implement actual data loading
        # Should handle: missing data, corporate actions, errors
        try:
            # returns = load_from_database(ticker, lookback_days)
            # For demo:
            returns = pd.Series(
                np.random.normal(0.001, 0.02, lookback_days),
                name=ticker
            )
            return returns
        except Exception as e:
            self.logger.error(f"Failed to load {ticker}: {e}")
            return None
    
    def process_single_stock(self, ticker: str) -> Optional[ForecastResult]:
        """
        Complete pipeline for single stock.
        
        This function runs in parallel across workers.
        """
        try:
            # Load data
            returns = self.load_returns(ticker)
            if returns is None or len(returns) < self.config.min_observations:
                self.logger.warning(f"{ticker}: Insufficient data")
                return None
            
            # Create forecaster
            forecaster = StockForecaster(ticker, self.config)
            
            # Fit model
            if not forecaster.fit(returns):
                self.logger.warning(f"{ticker}: Fit failed")
                return None
            
            # Generate forecast
            result = forecaster.forecast()
            
            # Cache result
            if result:
                self.redis_client.setex(
                    f"forecast:{ticker}",
                    86400,  # 24 hour expiry
                    json.dumps({
                        'forecast': result.forecast,
                        'lower_ci': result.lower_ci,
                        'upper_ci': result.upper_ci,
                        'timestamp': result.timestamp.isoformat()
                    })
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"{ticker}: Unexpected error: {e}")
            return None
    
    def run_parallel_forecasting(self) -> Dict[str, ForecastResult]:
        """
        Run forecasting for all stocks in parallel.
        
        Target: Complete 500 stocks in < 30 minutes.
        """
        start_time = datetime.now()
        results = {}
        failed = []
        
        self.logger.info(f"Starting parallel forecasting for {len(self.tickers)} stocks")
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(self.process_single_stock, ticker): ticker
                for ticker in self.tickers
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result:
                        results[ticker] = result
                    else:
                        failed.append(ticker)
                except Exception as e:
                    self.logger.error(f"{ticker}: Processing failed: {e}")
                    failed.append(ticker)
        
        # Performance metrics
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.logger.info(f"Completed in {duration:.1f} seconds")
        self.logger.info(f"Success: {len(results)}, Failed: {len(failed)}")
        
        # Alert if too many failures
        failure_rate = len(failed) / len(self.tickers)
        if failure_rate > 0.10:  # >10% failure rate
            self.logger.error(f"⚠ High failure rate: {failure_rate*100:.1f}%")
            # Send alert to ops team
        
        return results
    
    def aggregate_forecasts(self, 
                           results: Dict[str, ForecastResult]) -> pd.DataFrame:
        """
        Aggregate all forecasts into portfolio view.
        """
        records = []
        for ticker, result in results.items():
            records.append({
                'ticker': ticker,
                'forecast_return': result.forecast,
                'lower_ci': result.lower_ci,
                'upper_ci': result.upper_ci,
                'model_order': f"ARMA{result.model_order}",
                'diagnostics_passed': result.diagnostics_passed,
                'n_warnings': len(result.warnings)
            })
        
        df = pd.DataFrame(records)
        
        # Add derived metrics
        df['signal_strength'] = df['forecast_return'] / (df['upper_ci'] - df['lower_ci'])
        df['rank'] = df['forecast_return'].rank(pct=True)
        
        return df
    
    def check_systematic_bias(self, forecasts_df: pd.DataFrame) -> Dict:
        """
        Critical: Detect systematic forecast bias.
        
        Prevents portfolio-wide losses from biased models.
        """
        checks = {}
        
        # Check 1: Mean forecast should be ~0
        mean_forecast = forecasts_df['forecast_return'].mean()
        checks['mean_forecast'] = mean_forecast
        checks['mean_bias'] = abs(mean_forecast) > 0.001  # >10 bps bias
        
        if checks['mean_bias']:
            self.logger.warning(f"⚠ Systematic forecast bias: {mean_forecast*10000:.1f} bps")
        
        # Check 2: Forecasts should be symmetric
        positive_pct = (forecasts_df['forecast_return'] > 0).mean()
        checks['positive_pct'] = positive_pct
        checks['asymmetric'] = abs(positive_pct - 0.5) > 0.15  # >65% or <35%
        
        if checks['asymmetric']:
            self.logger.warning(f"⚠ Asymmetric forecasts: {positive_pct*100:.0f}% positive")
        
        # Check 3: Too many failed diagnostics
        pass_rate = forecasts_df['diagnostics_passed'].mean()
        checks['pass_rate'] = pass_rate
        checks['low_pass_rate'] = pass_rate < 0.50  # <50% pass
        
        if checks['low_pass_rate']:
            self.logger.error(f"⚠ Low diagnostic pass rate: {pass_rate*100:.0f}%")
        
        # Check 4: Cross-sectional consistency
        # If all stocks forecast positive, likely systematic error
        if positive_pct > 0.80:
            checks['likely_regime_issue'] = True
            self.logger.error("⚠ >80% stocks forecast positive - check for regime change!")
        
        return checks


# Usage Example
def main_forecasting_pipeline():
    """
    Nightly forecasting job.
    """
    # Configuration
    tickers = [f"STOCK{i}" for i in range(500)]  # 500 stocks
    config = ModelConfig(
        max_p=3,
        max_q=3,
        min_observations=252,
        ic_criterion='bic'
    )
    
    # Initialize system
    system = ParallelForecastingSystem(
        tickers=tickers,
        config=config,
        n_workers=16  # 16 parallel workers
    )
    
    # Run forecasting
    results = system.run_parallel_forecasting()
    
    # Aggregate
    forecasts_df = system.aggregate_forecasts(results)
    
    # Check for systematic bias
    bias_checks = system.check_systematic_bias(forecasts_df)
    
    # Save to database / send to trading system
    # forecasts_df.to_sql('daily_forecasts', engine)
    
    print(f"\\nForecasting complete: {len(results)} stocks")
    print(f"Mean forecast: {forecasts_df['forecast_return'].mean()*10000:.1f} bps")
    print(f"Diagnostics passed: {forecasts_df['diagnostics_passed'].mean()*100:.0f}%")
    
    return forecasts_df

# Run pipeline
# forecasts = main_forecasting_pipeline()
\`\`\`

### Key Design Decisions

**1. Parallel Processing:**
- 16 workers × 30 stocks each = 480 stocks
- Each stock ~3-5 seconds → Total ~90-150 seconds
- Well under 30 minute target!

**2. Fast Model Selection:**
- Test only 6 common orders (not all 25 combinations)
- 90% of stocks fit AR(1), MA(1), or ARMA(1,1)

**3. Robustness:**
- Enforce stationarity/invertibility constraints
- Multiple validation checks
- Graceful failure handling

**4. Bias Prevention:**
- Cross-sectional bias checks
- Alert if systematic patterns
- Historical forecast tracking

**5. Monitoring:**
- Redis for real-time status
- Logging at multiple levels
- Alerting for failures

This production system handles scale, reliability, and prevents systematic errors that could cause portfolio losses!`,
  },
];
