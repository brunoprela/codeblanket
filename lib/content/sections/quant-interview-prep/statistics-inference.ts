export const statisticsInference = {
  title: 'Statistics & Inference',
  id: 'statistics-inference',
  content: `
# Statistics & Inference

## Introduction

Statistical inference is the backbone of quantitative trading and risk management. Every trading decision involves uncertainty, and statistics provides the framework to reason about that uncertainty rigorously. In quant interviews, firms test your ability to:

**Core Competencies:**
- Design and interpret hypothesis tests
- Construct and interpret confidence intervals
- Understand sampling distributions
- Calculate statistical power and sample sizes
- Recognize and avoid statistical fallacies
- Apply inference to trading problems

**Why Interviewers Test This:**
1. **Strategy validation**: Is your backtest result real or luck?
2. **Risk management**: How confident are we in our VaR estimates?
3. **Parameter estimation**: What's the true expected return?
4. **A/B testing**: Is Strategy B better than Strategy A?
5. **Model selection**: Which factors are statistically significant?
6. **Scientific thinking**: Can you reason rigorously under uncertainty?

**Common Applications:**
- Backtesting statistical significance
- Sharpe ratio confidence intervals
- Comparing strategies (paired tests)
- Factor model significance
- Parameter stability testing
- Regime detection

This section covers 40+ problems across all aspects of statistical inference, with particular emphasis on trading applications, common pitfalls, and mental math techniques.

---

## Section 1: Sampling Distributions & Central Limit Theorem

### 1.1 Central Limit Theorem (CLT)

**Theorem:** For iid random variables X₁, X₂, ..., Xₙ with mean μ and finite variance σ²:

\`\`\`
X̄ = (X₁ + ... + Xₙ)/n

As n → ∞:
√n(X̄ - μ)/σ →^d N(0,1)

Or equivalently: X̄ ~approximately~ N(μ, σ²/n)
\`\`\`

**Key Insight:** Sample mean becomes approximately normal REGARDLESS of the original distribution (as long as σ² < ∞).

**How fast does it converge?**
- **Symmetric distributions**: n ≥ 15-20 usually sufficient
- **Skewed distributions**: n ≥ 30-40 needed
- **Heavy-tailed**: May need n >> 100

**Rule of thumb:** n ≥ 30 for "safe" approximation

### Problem 1.1a: Trading Returns Distribution

**Question:** Daily returns follow unknown distribution with mean 0.08% and std dev 1.5%. What's the distribution of average daily return over 100 trading days?

**Solution:**

By CLT:
\`\`\`
X̄₁₀₀ ~ N(0.08%, 1.5%²/100)
     ~ N(0.08%, 0.000225%)
     ~ N(0.08%, 0.015%)

Standard error: SE = 1.5%/√100 = 0.15%
\`\`\`

**95% Confidence Interval:**
\`\`\`
0.08% ± 1.96 × 0.15% = 0.08% ± 0.29%
                       = [-0.21%, 0.37%]
\`\`\`

**Interpretation:** Sample mean will fall in this range 95% of the time.

### Problem 1.1b: Annual Returns

**Question:** Daily returns have mean μ_d = 0.05% and σ_d = 2%. What's the distribution of cumulative annual return over 252 trading days?

**Solution:**

For small returns, cumulative return ≈ sum of returns.

\`\`\`
R_annual = Σ(i=1 to 252) R_i

E[R_annual] = 252 × 0.05% = 12.6%

Var(R_annual) = 252 × (2%)² = 1.008
                  (assuming independent returns)

σ_annual = √1.008 ≈ 31.7%
\`\`\`

By CLT:
\`\`\`
R_annual ~ N(12.6%, 31.7%²)
\`\`\`

**95% CI for annual return:**
\`\`\`
12.6% ± 1.96 × 31.7% = 12.6% ± 62.1%
                       = [-49.5%, 74.7%]
\`\`\`

**Key Insight:** Even with positive daily mean, there's substantial probability of annual loss!

### Problem 1.1c: Non-Normal Distribution

**Question:** Returns follow a uniform distribution U[-1%, 3%]. What's the approximate distribution of average return over 50 days?

**Solution:**

For uniform U[a,b]:
\`\`\`
μ = (a+b)/2 = (-1 + 3)/2 = 1%
σ² = (b-a)²/12 = 4²/12 = 16/12 = 1.333
σ = √1.333 ≈ 1.155%
\`\`\`

By CLT (n=50 is sufficient):
\`\`\`
X̄₅₀ ~ N(1%, 1.155%²/50)
    ~ N(1%, 0.163%)

SE = 1.155%/√50 ≈ 0.163%
\`\`\`

**Note:** Original distribution is uniform (not normal), but average is approximately normal!

### 1.2 Standard Error

**Definition:** Standard error of an estimator is its standard deviation.

For sample mean:
\`\`\`
SE(X̄) = σ/√n
\`\`\`

If σ unknown, estimate with sample std dev:
\`\`\`
SE(X̄) = s/√n
\`\`\`

**Key Properties:**
- SE decreases with √n (not n)
- To halve SE, need 4× sample size
- To reduce SE by 10×, need 100× sample size

### Problem 1.2a: Sample Size for SE

**Question:** Current SE = 0.5%. To reduce to 0.1%, how much more data needed?

**Solution:**

\`\`\`
SE = σ/√n

Want: SE_new = 0.1% = σ/√n_new

Currently: SE_current = 0.5% = σ/√n_current

Ratio: 0.1/0.5 = √n_current / √n_new
       1/5 = √(n_current/n_new)
       n_new = 25 × n_current
\`\`\`

**Answer:** Need 25× as much data!

**General formula:** To reduce SE by factor k, need k² × data.

### 1.3 T-Distribution

When σ is unknown and estimated from sample:

\`\`\`
T = (X̄ - μ) / (s/√n) ~ t_{n-1}
\`\`\`

**Properties:**
- Heavier tails than normal (accounts for estimation uncertainty)
- Converges to normal as n → ∞
- For n ≥ 30, practically indistinguishable from normal

**Critical values (two-tailed, α=0.05):**
- n = 10: t* = 2.262
- n = 20: t* = 2.086
- n = 30: t* = 2.042
- n = 100: t* = 1.984
- n = ∞: z* = 1.96

### Problem 1.3: Small Sample Inference

**Question:** Sample of 15 returns: mean = 0.5%, s = 2%. Construct 95% CI.

**Solution:**

Use t-distribution with df = 15-1 = 14:
\`\`\`
t* = 2.145  (from t-table)

SE = 2%/√15 ≈ 0.516%

95% CI: 0.5% ± 2.145 × 0.516%
       = 0.5% ± 1.11%
       = [-0.61%, 1.61%]
\`\`\`

**Compare with normal approximation:**
\`\`\`
Using z* = 1.96:
95% CI: 0.5% ± 1.96 × 0.516% = [0.01%, 1.99%]
\`\`\`

**Difference:** T-distribution gives wider CI (more conservative), accounting for small sample uncertainty.

---

## Section 2: Confidence Intervals

### 2.1 Construction and Interpretation

**General Form:**
\`\`\`
Estimate ± (Critical Value) × (Standard Error)
\`\`\`

**Common Confidence Levels:**
- 90%: z* = 1.645
- 95%: z* = 1.96
- 99%: z* = 2.576

**Interpretation:** "If we repeated this procedure many times, 95% of intervals would contain the true parameter."

**Common Misconception:** "95% probability the true value is in THIS interval" ← WRONG! The true value is fixed; the interval is random.

### Problem 2.1a: Mean Return CI

**Question:** Sample of 200 daily returns: X̄ = 0.08%, s = 1.2%. Construct 95% CI for true mean.

**Solution:**

\`\`\`
SE = s/√n = 1.2%/√200 ≈ 0.0849%

95% CI: 0.08% ± 1.96 × 0.0849%
       = 0.08% ± 0.166%
       = [-0.086%, 0.246%]
\`\`\`

**Interpretation:** True mean daily return is likely between -0.086% and +0.246%.

**Key Observation:** Interval includes zero! Cannot conclude positive expected return at 95% confidence.

### Problem 2.1b: Volatility CI

**Question:** Sample variance s² = 4 from n=100 observations. Construct 95% CI for true variance σ².

**Solution:**

For variance, use chi-square distribution:

\`\`\`
(n-1)s²/σ² ~ χ²_{n-1}

95% CI for σ²:
[(n-1)s²/χ²_{α/2}, (n-1)s²/χ²_{1-α/2}]

For n=100, α=0.05:
χ²_{0.025, 99} ≈ 129.6
χ²_{0.975, 99} ≈ 73.4

CI: [99×4/129.6, 99×4/73.4]
  = [3.05, 5.39]
\`\`\`

**For std dev σ:**
\`\`\`
CI: [√3.05, √5.39] = [1.75, 2.32]
\`\`\`

### Problem 2.1c: Sharpe Ratio CI

**Question:** Strategy has Sharpe ratio 1.2 over 252 days. Construct 95% CI.

**Solution:**

For Sharpe ratio S = μ/σ:

\`\`\`
SE(S) ≈ √((1 + S²/2)/n)
       = √((1 + 1.2²/2)/252)
       = √((1 + 0.72)/252)
       = √(1.72/252)
       = √0.00683
       ≈ 0.0826
\`\`\`

\`\`\`
95% CI: 1.2 ± 1.96 × 0.0826
       = 1.2 ± 0.162
       = [1.04, 1.36]
\`\`\`

**Interpretation:** True Sharpe likely between 1.04 and 1.36 (both excellent!).

### 2.2 Width of Confidence Intervals

**How to make CI narrower:**
1. **Increase sample size** (most important): Width ∝ 1/√n
2. **Lower confidence level**: 90% narrower than 95% narrower than 99%
3. **Reduce variance**: σ directly affects width

### Problem 2.2: Sample Size for Desired Precision

**Question:** Want 95% CI for mean return with width ≤ 0.1%. If σ = 1.5%, how many observations needed?

**Solution:**

\`\`\`
Width = 2 × 1.96 × (σ/√n)

Want: 2 × 1.96 × (1.5%/√n) ≤ 0.1%
      3.92 × 1.5%/√n ≤ 0.1%
      5.88%/√n ≤ 0.1%
      √n ≥ 58.8
      n ≥ 3457
\`\`\`

**Answer:** Need at least 3457 observations!

**Key Insight:** High precision requires large samples.

---

## Section 3: Hypothesis Testing Framework

### 3.1 The Classical Framework

**Five Steps:**

1. **State hypotheses**
   - H₀: Null hypothesis (status quo, no effect)
   - H₁: Alternative hypothesis (what you're trying to show)

2. **Choose significance level α**
   - Typically α = 0.05 or 0.01
   - α = P(Type I error) = P(reject H₀ | H₀ true)

3. **Compute test statistic**
   - Standardized measure (t, z, F, χ², etc.)
   - Measures distance from H₀ in standard errors

4. **Find p-value**
   - P(observing test statistic this extreme or more | H₀ true)
   - Smaller p-value = stronger evidence against H₀

5. **Make decision**
   - If p < α: Reject H₀
   - If p ≥ α: Fail to reject H₀ (not "accept H₀"!)

### Problem 3.1a: One-Sample T-Test

**Question:** Strategy backtested over 500 days. Mean daily return = 0.06%, s = 1.2%. Test H₀: μ = 0 vs H₁: μ > 0 at α = 0.05.

**Solution:**

**Step 1: Hypotheses**
\`\`\`
H₀: μ = 0  (strategy has no edge)
H₁: μ > 0  (strategy is profitable)

(One-sided test because we only care if μ > 0)
\`\`\`

**Step 2: Significance level**
\`\`\`
α = 0.05
\`\`\`

**Step 3: Test statistic**
\`\`\`
t = (X̄ - μ₀) / (s/√n)
  = (0.06 - 0) / (1.2/√500)
  = 0.06 / 0.0537
  = 1.117
\`\`\`

**Step 4: P-value**
\`\`\`
df = 499
P(T > 1.117) ≈ 0.132

(Use t-table or calculator)
\`\`\`

**Step 5: Decision**
\`\`\`
p = 0.132 > 0.05

Fail to reject H₀
\`\`\`

**Conclusion:** Insufficient evidence to conclude strategy is profitable. The observed mean of 0.06% could reasonably occur by chance (p=13.2%) if true mean is zero.

**Critical value approach (equivalent):**
\`\`\`
Critical value: t* = 1.648 (for df=499, one-tailed)
Test statistic: t = 1.117 < 1.648

Do not reject H₀
\`\`\`

### Problem 3.1b: Two-Sided Test

**Question:** Same data, but test H₀: μ = 0 vs H₁: μ ≠ 0 (two-sided).

**Solution:**

**Steps 1-3:** Same as before, t = 1.117

**Step 4: P-value (two-sided)**
\`\`\`
p = 2 × P(T > |1.117|) ≈ 2 × 0.132 = 0.264
\`\`\`

**Step 5: Decision**
\`\`\`
p = 0.264 > 0.05

Fail to reject H₀
\`\`\`

**Note:** Two-sided test is more conservative (harder to reject).

### 3.2 Type I and Type II Errors

**Truth Table:**

\`\`\`
                  H₀ True       H₀ False
--------------------------------------------
Reject H₀       Type I Error   Correct (Power)
                (False Pos)     (1-β)

Don't Reject H₀  Correct       Type II Error
                (1-α)          (False Neg) (β)
\`\`\`

**Tradeoffs:**
- Lower α → Harder to reject H₀ → More Type II errors
- Higher α → Easier to reject H₀ → More Type I errors
- Increase sample size → Lower both error rates

### Problem 3.2a: Error Rates

**Question:** You test 100 null hypotheses, all true. Using α=0.05, how many false rejections expected?

**Solution:**

\`\`\`
Expected Type I errors = 100 × 0.05 = 5
\`\`\`

**This is the multiple testing problem!** When testing many hypotheses, you'll get false positives even if nothing is real.

### Problem 3.2b: Power Calculation

**Question:** You want 80% power to detect mean return of 0.08% (vs H₀: μ=0) with σ=1.5% and α=0.05 (one-sided). How many observations needed?

**Solution:**

**Power formula:**
\`\`\`
n ≈ [(z_{1-α} + z_{1-β}) × σ / δ]²

where δ = effect size (difference from H₀)
\`\`\`

\`\`\`
z_{0.95} = 1.645  (for α=0.05, one-sided)
z_{0.80} = 0.842  (for power=0.80)

n ≈ [(1.645 + 0.842) × 1.5 / 0.08]²
  ≈ [2.487 × 18.75]²
  ≈ [46.63]²
  ≈ 2174
\`\`\`

**Answer:** Need about 2174 observations (almost 9 years of daily data!) to have 80% power.

**Key Insight:** Detecting small effects requires massive samples.

### 3.3 P-Values: Interpretation and Misinterpretation

**Correct Interpretation:**
- P-value = P(data this extreme or more | H₀ is true)
- Measures compatibility of data with H₀
- Small p-value = data unusual under H₀

**Common Misinterpretations (ALL WRONG!):**
- ❌ "P(H₀ is true | data)"
- ❌ "Probability you're wrong if you reject H₀"
- ❌ "1 - P(H₁ is true | data)"
- ❌ "Importance of the result"

**Key Points:**
1. P-value is NOT posterior probability of H₀
2. P-value depends on sample size: large n → small p even for tiny effects
3. P-value says nothing about effect size or practical importance
4. "Significant" ≠ "Important"

### Problem 3.3: Significance vs. Importance

**Question:** Strategy has mean return 0.01% (1 basis point) per day with s=2% over n=10,000 days. Test H₀: μ=0 vs H₁: μ>0.

**Solution:**

\`\`\`
t = 0.01 / (2/√10000)
  = 0.01 / 0.02
  = 0.5

Wait, this is wrong. Let me recalculate:

t = 0.01 / (2/√10000)
  = 0.01 / (2/100)
  = 0.01 / 0.02
  = 0.5

Hmm, still seems low. Let me be more careful with percentages:

Mean = 0.01% = 0.0001
s = 2% = 0.02

t = 0.0001 / (0.02/√10000)
  = 0.0001 / (0.02/100)
  = 0.0001 / 0.0002
  = 0.5

P-value ≈ 0.31 > 0.05

Actually, let me reconsider. If returns are already in percent:

t = 0.01 / (2/100) = 0.01 / 0.02 = 0.5
\`\`\`

Hmm, this gives a non-significant result. Let me try larger n:

With n=100,000 days (impractical but illustrative):
\`\`\`
t = 0.01 / (2/√100000)
  = 0.01 / 0.00632
  = 1.58

P-value ≈ 0.057 (barely not significant)
\`\`\`

With n=250,000:
\`\`\`
t = 0.01 / (2/500)
  = 0.01 / 0.004
  = 2.5

P-value ≈ 0.006 < 0.05 → Significant!
\`\`\`

**Point:** With huge sample, even tiny effect (1 bp/day) becomes "statistically significant."

**Economic significance:** 1 bp/day × 252 days ≈ 2.5% annual return. After transaction costs (say 2-3 bp per trade), profit might vanish!

**Lesson:** Always consider practical/economic significance alongside statistical significance.

---

## Section 4: Comparing Two Samples

### 4.1 Independent Two-Sample T-Test

**When:** Comparing means of two independent groups.

**Test statistic:**
\`\`\`
t = (X̄₁ - X̄₂) / SE_diff

where SE_diff = √(s₁²/n₁ + s₂²/n₂)
\`\`\`

**Degrees of freedom (Welch's approximation):**
\`\`\`
df ≈ (s₁²/n₁ + s₂²/n₂)² / [(s₁²/n₁)²/(n₁-1) + (s₂²/n₂)²/(n₂-1)]
\`\`\`

(Often approximated as min(n₁-1, n₂-1) for simplicity)

### Problem 4.1a: Comparing Two Strategies

**Question:** 
- Strategy A: n=200 days, X̄_A=0.08%, s_A=1.2%
- Strategy B: n=200 days, X̄_B=0.12%, s_B=1.5%

Test H₀: μ_A = μ_B vs H₁: μ_A ≠ μ_B at α=0.05.

**Solution:**

**Step 1: Hypotheses**
\`\`\`
H₀: μ_A = μ_B  (strategies have same mean)
H₁: μ_A ≠ μ_B  (strategies differ)
\`\`\`

**Step 2: Test statistic**
\`\`\`
SE_diff = √(s_A²/n_A + s_B²/n_B)
        = √(1.2²/200 + 1.5²/200)
        = √(0.0072 + 0.01125)
        = √0.01845
        = 0.1358%

t = (0.12 - 0.08) / 0.1358
  = 0.04 / 0.1358
  = 0.295
\`\`\`

**Step 3: P-value**
\`\`\`
df ≈ 199 (conservative)
P(|T| > 0.295) ≈ 0.77 (two-sided)
\`\`\`

**Step 4: Decision**
\`\`\`
p = 0.77 > 0.05

Fail to reject H₀
\`\`\`

**Conclusion:** No significant difference between strategies, despite Strategy B having higher mean (0.12% vs 0.08%).

**Why?** High variability (1.2-1.5% std dev) relative to difference in means (0.04%) and moderate sample size (200).

**Power analysis:** To detect 0.04% difference with 80% power:
\`\`\`
n ≈ [(1.96 + 0.84)² × (√(1.2² + 1.5²)/0.04)²]
  ≈ 7.84 × (1.92/0.04)²
  ≈ 7.84 × 2304
  ≈ 18,063 per strategy!
\`\`\`

Need ~18,000 days (70+ years) per strategy to reliably detect this small difference.

### 4.2 Paired T-Test

**When:** Comparing two measurements on the same units (e.g., same days, same stocks).

**Advantage:** Eliminates between-subject variability, increases power.

**Test statistic:**
\`\`\`
d_i = X_{1i} - X_{2i}  (paired differences)

t = d̄ / (s_d/√n)

where d̄ = mean of differences
      s_d = std dev of differences
\`\`\`

### Problem 4.2: Paired Strategy Comparison

**Question:** Test Strategy A vs B on same 100 days. Differences d_i = R_{Ai} - R_{Bi}:
- d̄ = 0.04%
- s_d = 0.8%

Test H₀: μ_d = 0 vs H₁: μ_d ≠ 0.

**Solution:**

\`\`\`
t = 0.04 / (0.8/√100)
  = 0.04 / 0.08
  = 0.5

df = 99
P(|T| > 0.5) ≈ 0.62

p = 0.62 > 0.05 → Fail to reject H₀
\`\`\`

**Compare with independent test (Problem 4.1a):**
- Independent test: needed 18,000+ observations
- Paired test: s_d=0.8% << √(s_A² + s_B²)=1.92%

**Why paired is better:** Correlation between strategies on same days reduces effective variance!

If strategies are positively correlated (both up on good days, both down on bad days), their *difference* has lower variance than each individually.

---

## Section 5: Multiple Testing

### 5.1 The Multiple Testing Problem

**Scenario:** You test k hypotheses, each at level α.

**Problem:** If all null hypotheses are true:
\`\`\`
Expected false positives = k × α
\`\`\`

**Example:** Test 100 strategies, all worthless, α=0.05:
\`\`\`
Expected false discoveries = 100 × 0.05 = 5
\`\`\`

You'll find ~5 "significant" strategies purely by chance!

### 5.2 Bonferroni Correction

**Method:** Test each hypothesis at level α/k.

**Family-wise error rate (FWER):**
\`\`\`
P(at least one Type I error) ≤ α
\`\`\`

**Conservative:** Controls FWER, but loses power for large k.

### Problem 5.2a: Multiple Strategy Testing

**Question:** Test 20 strategies. One has p=0.04. Is it significant after Bonferroni correction (α=0.05)?

**Solution:**

\`\`\`
α_Bonf = α/k = 0.05/20 = 0.0025

p = 0.04 > 0.0025

Not significant after correction!
\`\`\`

**Interpretation:** The "significant" result (p=0.04) is not surprising when testing 20 strategies. With α=0.05, we expect 20×0.05=1 false positive.

### 5.3 False Discovery Rate (FDR)

**Alternative to FWER:** Control the *proportion* of false discoveries.

**Benjamini-Hochberg procedure:**
1. Order p-values: p₍₁₎ ≤ p₍₂₎ ≤ ... ≤ p₍k₎
2. Find largest i such that p₍ᵢ₎ ≤ (i/k)α
3. Reject H₍₁₎, ..., H₍ᵢ₎

**Advantage:** More powerful than Bonferroni for large k.

### Problem 5.3: FDR Example

**Question:** Test 10 hypotheses, α=0.05. P-values: 0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205, 0.321, 0.440

Which are significant under FDR?

**Solution:**

\`\`\`
i=1: p₍₁₎=0.001 vs (1/10)×0.05=0.005 → 0.001 < 0.005 ✓
i=2: p₍₂₎=0.008 vs (2/10)×0.05=0.010 → 0.008 < 0.010 ✓
i=3: p₍₃₎=0.039 vs (3/10)×0.05=0.015 → 0.039 > 0.015 ✗
...
\`\`\`

Largest i with p₍ᵢ₎ ≤ (i/k)α is i=2.

**Reject:** First 2 hypotheses (p=0.001, 0.008)

**Bonferroni would reject:** Only first (p=0.001 < 0.05/10=0.005)

**FDR is less conservative!**

---

## Section 6: Sample Size & Power Analysis

### 6.1 Power Function

**Power = P(reject H₀ | H₁ is true)**

**Factors affecting power:**
1. **Sample size n** ↑ → Power ↑
2. **Effect size δ** ↑ → Power ↑ (easier to detect large effects)
3. **Significance level α** ↑ → Power ↑ (but more Type I errors)
4. **Variability σ** ↓ → Power ↑

### Problem 6.1: Power Calculation

**Question:** H₀: μ=0 vs H₁: μ=0.1%, σ=1.5%, α=0.05 (one-sided). What's the power with n=1000?

**Solution:**

**Under H₁:**
\`\`\`
X̄ ~ N(0.1%, 1.5%²/1000)
X̄ ~ N(0.1%, 0.0000225)
SE = 1.5%/√1000 = 0.0474%
\`\`\`

**Critical value under H₀:**
\`\`\`
Reject if X̄ > 0 + 1.645 × 0.0474% = 0.078%
\`\`\`

**Power = P(X̄ > 0.078% | μ=0.1%):**

Standardize:
\`\`\`
Z = (0.078 - 0.1) / 0.0474 = -0.464

Power = P(Z > -0.464) = Φ(0.464) ≈ 0.679
\`\`\`

**Answer:** 67.9% power

### 6.2 Sample Size Formulas

**For testing H₀: μ=μ₀ vs H₁: μ=μ₁ with power 1-β:**

**One-sided:**
\`\`\`
n = [(z_{1-α} + z_{1-β}) × σ / |μ₁-μ₀|]²
\`\`\`

**Two-sided:**
\`\`\`
n = [(z_{1-α/2} + z_{1-β}) × σ / |μ₁-μ₀|]²
\`\`\`

**Common z-values:**
- α=0.05, one-sided: z_{0.95} = 1.645
- α=0.05, two-sided: z_{0.975} = 1.96
- α=0.01, two-sided: z_{0.995} = 2.576
- β=0.20 (power=80%): z_{0.80} = 0.842
- β=0.10 (power=90%): z_{0.90} = 1.282

### Problem 6.2: Sample Size Determination

**Question:** Want 90% power to detect Sharpe ratio of 0.5 (vs 0) at α=0.05 (two-sided). How many observations?

**Solution:**

For Sharpe ratio testing:
\`\`\`
n ≈ [(z_{1-α/2} + z_{1-β}) / S]² + 3

n ≈ [(1.96 + 1.282) / 0.5]² + 3
  ≈ [3.242 / 0.5]² + 3
  ≈ [6.484]² + 3
  ≈ 42 + 3
  = 45
\`\`\`

**Answer:** Need about 45 observations (e.g., 45 months ≈ 4 years).

---

## Section 7: Trading-Specific Applications

### 7.1 Sharpe Ratio Testing

**Sharpe ratio:** S = μ/σ (return per unit risk)

**Hypothesis test:**
\`\`\`
H₀: S = 0  (no risk-adjusted return)
H₁: S > 0  (positive risk-adjusted return)
\`\`\`

**Test statistic:**
\`\`\`
t = Ŝ × √n  ~ t_{n-1} under H₀

where Ŝ = sample Sharpe ratio
\`\`\`

### Problem 7.1a: Sharpe Testing

**Question:** Strategy has Ŝ=0.8 over 120 months. Test H₀: S=0 vs H₁: S>0 at α=0.05.

**Solution:**

\`\`\`
t = 0.8 × √120 = 0.8 × 10.95 = 8.76

df = 119
Critical value: t_{0.95, 119} ≈ 1.658

t = 8.76 >> 1.658 → Reject H₀
\`\`\`

**P-value:** P(T > 8.76) < 0.0001

**Conclusion:** Highly significant evidence of positive Sharpe ratio.

### Problem 7.1b: Comparing Sharpe Ratios

**Question:** Two strategies with n=60 months each:
- Strategy A: S_A = 1.0
- Strategy B: S_B = 1.3

Test H₀: S_A = S_B vs H₁: S_A ≠ S_B.

**Solution:**

This requires specialized test (Jobson-Korkie test or bootstrap). 

**Approximate approach:**

Standard errors:
\`\`\`
SE(S_A) ≈ √((1 + S_A²/2)/n) = √((1 + 0.5)/60) ≈ 0.158
SE(S_B) ≈ √((1 + 1.3²/2)/60) = √((1 + 0.845)/60) ≈ 0.175
\`\`\`

\`\`\`
SE_diff ≈ √(SE²_A + SE²_B) = √(0.025 + 0.031) ≈ 0.236

z = (1.3 - 1.0) / 0.236 = 1.27

P-value ≈ 0.20 > 0.05 → Not significant
\`\`\`

**Conclusion:** Difference (1.3 vs 1.0) not statistically significant with 60 observations.

### 7.2 Maximum Drawdown Distribution

**Maximum drawdown:** Largest peak-to-trough decline.

**Approximate distribution (Gaussian returns):**

For horizon T with Sharpe ratio S:
\`\`\`
E[MaxDD] ≈ σ√(T) × (0.63 - 0.5×S)

where T measured in years
\`\`\`

This is complex—usually estimated via simulation.

---

## Section 8: Common Pitfalls & Best Practices

### 8.1 Data Snooping / P-Hacking

**What:** Searching through data until finding significance, then reporting only that result.

**Examples:**
- Try 100 indicators, report the one with lowest p-value
- Test many lookback periods, report best one
- Subset data until results look good

**Why it's bad:** Inflates Type I error rate dramatically.

**Solutions:**
1. **Pre-registration:** State hypotheses before seeing data
2. **Hold-out set:** Test on independent data
3. **Multiple testing correction:** Bonferroni, FDR
4. **Cross-validation:** Proper out-of-sample testing
5. **Walk-forward analysis:** Sequential out-of-sample testing

### 8.2 Look-Ahead Bias

**What:** Using information not available at decision time.

**Examples:**
- Using today's close to predict today's close
- Using adjusted prices without point-in-time adjustment
- Survivorship bias (only stocks that survived)

**Detection:** Ask "Could I have known this at the time?"

### 8.3 Overfitting

**What:** Model fits noise rather than signal.

**Indicators:**
- Excellent in-sample, poor out-of-sample
- Many parameters relative to observations
- Highly sensitive to small data changes

**Prevention:**
- Simplify models (fewer parameters)
- Regularization (L1, L2)
- Cross-validation
- Economic intuition (does it make sense?)

### 8.4 Non-Independence

**What:** Observations aren't independent (e.g., autocorrelation, overlapping periods).

**Consequence:** Standard errors underestimated → p-values too small.

**Solutions:**
- Newey-West HAC standard errors
- Block bootstrap
- Adjust degrees of freedom
- Use fewer, non-overlapping periods

---

## Section 9: Python Implementations

\`\`\`python
"""
Statistical Inference for Quantitative Trading
Comprehensive implementations of hypothesis tests, confidence intervals, and power analysis.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List

# ============================================================================
# Section 1: One-Sample Tests
# ============================================================================

def one_sample_t_test(data: np.ndarray, mu0: float = 0, 
                      alternative: str = 'two-sided') -> Dict:
    """
    One-sample t-test.
    
    H0: μ = mu0
    H1: μ ≠ mu0 (or > or <)
    
    Args:
        data: Sample data
        mu0: Hypothesized mean under null
        alternative: 'two-sided', 'greater', or 'less'
        
    Returns:
        Dictionary with test results
    """
    n = len(data)
    xbar = np.mean(data)
    s = np.std(data, ddof=1)
    se = s / np.sqrt(n)
    
    # T-statistic
    t_stat = (xbar - mu0) / se
    
    # P-value
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    elif alternative == 'greater':
        p_value = 1 - stats.t.cdf(t_stat, df=n-1)
    elif alternative == 'less':
        p_value = stats.t.cdf(t_stat, df=n-1)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    
    # Confidence interval (95%)
    t_crit = stats.t.ppf(0.975, df=n-1)
    ci_lower = xbar - t_crit * se
    ci_upper = xbar + t_crit * se
    
    return {
        'mean': xbar,
        'std': s,
        'se': se,
        't_statistic': t_stat,
        'p_value': p_value,
        'ci_95': (ci_lower, ci_upper),
        'n': n,
        'df': n-1
    }

# ============================================================================
# Section 2: Two-Sample Tests
# ============================================================================

def two_sample_t_test(data1: np.ndarray, data2: np.ndarray,
                      alternative: str = 'two-sided') -> Dict:
    """
    Independent two-sample t-test (Welch's test - unequal variances).
    
    H0: μ1 = μ2
    H1: μ1 ≠ μ2 (or > or <)
    """
    n1, n2 = len(data1), len(data2)
    xbar1, xbar2 = np.mean(data1), np.mean(data2)
    s1, s2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    
    # Standard error of difference
    se_diff = np.sqrt(s1**2/n1 + s2**2/n2)
    
    # T-statistic
    t_stat = (xbar1 - xbar2) / se_diff
    
    # Welch-Satterthwaite degrees of freedom
    df = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
    
    # P-value
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))
    elif alternative == 'greater':
        p_value = 1 - stats.t.cdf(t_stat, df=df)
    else:  # 'less'
        p_value = stats.t.cdf(t_stat, df=df)
    
    return {
        'mean1': xbar1,
        'mean2': xbar2,
        'mean_diff': xbar1 - xbar2,
        'se_diff': se_diff,
        't_statistic': t_stat,
        'df': df,
        'p_value': p_value
    }

def paired_t_test(data1: np.ndarray, data2: np.ndarray,
                  alternative: str = 'two-sided') -> Dict:
    """
    Paired t-test.
    
    H0: μ_diff = 0
    H1: μ_diff ≠ 0
    """
    differences = data1 - data2
    return one_sample_t_test(differences, mu0=0, alternative=alternative)

# ============================================================================
# Section 3: Power Analysis
# ============================================================================

def power_one_sample_t(n: int, delta: float, sigma: float,
                       alpha: float = 0.05, alternative: str = 'two-sided') -> float:
    """
    Calculate power for one-sample t-test.
    
    Args:
        n: Sample size
        delta: Effect size (μ1 - μ0)
        sigma: Standard deviation
        alpha: Significance level
        alternative: 'two-sided' or 'one-sided'
        
    Returns:
        Power (probability of rejecting H0 when H1 is true)
    """
    # Non-centrality parameter
    ncp = delta / (sigma / np.sqrt(n))
    
    # Critical value under H0
    if alternative == 'two-sided':
        t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
        # Power = P(|T| > t_crit | ncp)
        power = 1 - stats.nct.cdf(t_crit, df=n-1, nc=ncp) + stats.nct.cdf(-t_crit, df=n-1, nc=ncp)
    else:  # one-sided
        t_crit = stats.t.ppf(1 - alpha, df=n-1)
        power = 1 - stats.nct.cdf(t_crit, df=n-1, nc=ncp)
    
    return power

def sample_size_one_sample_t(delta: float, sigma: float, alpha: float = 0.05,
                             power: float = 0.8, alternative: str = 'two-sided') -> int:
    """
    Calculate required sample size for desired power.
    
    Args:
        delta: Effect size to detect
        sigma: Standard deviation
        alpha: Significance level
        power: Desired power
        alternative: 'two-sided' or 'one-sided'
        
    Returns:
        Required sample size
    """
    # Get z-values (approximate)
    if alternative == 'two-sided':
        z_alpha = stats.norm.ppf(1 - alpha/2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    
    z_beta = stats.norm.ppf(power)
    
    # Formula: n = [(z_alpha + z_beta) * sigma / delta]^2
    n = ((z_alpha + z_beta) * sigma / delta)**2
    
    return int(np.ceil(n))

# ============================================================================
# Section 4: Multiple Testing Corrections
# ============================================================================

def bonferroni_correction(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Bonferroni correction for multiple testing.
    
    Returns:
        Boolean array: True if significant after correction
    """
    k = len(p_values)
    adjusted_alpha = alpha / k
    return p_values < adjusted_alpha

def fdr_correction(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini-Hochberg False Discovery Rate correction.
    
    Returns:
        Boolean array: True if significant after FDR correction
    """
    k = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    # Find largest i where p_(i) <= (i/k) * alpha
    thresholds = np.arange(1, k+1) / k * alpha
    significant = sorted_p <= thresholds
    
    if not np.any(significant):
        return np.zeros(k, dtype=bool)
    
    max_significant_idx = np.where(significant)[0][-1]
    
    # All hypotheses up to max_significant_idx are rejected
    reject = np.zeros(k, dtype=bool)
    reject[sorted_indices[:max_significant_idx+1]] = True
    
    return reject

# ============================================================================
# Section 5: Trading-Specific Tests
# ============================================================================

def sharpe_ratio_test(returns: np.ndarray, alpha: float = 0.05) -> Dict:
    """
    Test if Sharpe ratio is significantly different from zero.
    
    H0: Sharpe ratio = 0
    H1: Sharpe ratio > 0
    """
    n = len(returns)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    
    # Sample Sharpe ratio
    sharpe = mean_ret / std_ret if std_ret > 0 else 0
    
    # Test statistic: t = SR * sqrt(n)
    t_stat = sharpe * np.sqrt(n)
    
    # P-value (one-sided)
    p_value = 1 - stats.t.cdf(t_stat, df=n-1)
    
    # Confidence interval for Sharpe ratio
    se_sharpe = np.sqrt((1 + sharpe**2/2) / n)
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    ci_lower = sharpe - t_crit * se_sharpe
    ci_upper = sharpe + t_crit * se_sharpe
    
    return {
        'sharpe_ratio': sharpe,
        'annualized_sharpe': sharpe * np.sqrt(252),  # Assuming daily returns
        't_statistic': t_stat,
        'p_value': p_value,
        'ci_95': (ci_lower, ci_upper),
        'n': n
    }

# ============================================================================
# Examples and Testing
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("STATISTICAL INFERENCE FOR QUANTITATIVE TRADING")
    print("="*70)
    
    # Example 1: One-sample t-test on strategy returns
    print("\nExample 1: Testing Strategy Profitability")
    print("-"*70)
    
    np.random.seed(42)
    # Simulate returns with small positive mean
    returns = np.random.normal(0.06, 1.2, 500) / 100  # Convert to decimal
    
    result = one_sample_t_test(returns, mu0=0, alternative='greater')
    
    print(f"Sample mean: {result['mean']*100:.4f}%")
    print(f"Sample std:  {result['std']*100:.4f}%")
    print(f"Std error:   {result['se']*100:.4f}%")
    print(f"T-statistic: {result['t_statistic']:.4f}")
    print(f"P-value:     {result['p_value']:.4f}")
    print(f"95% CI:      [{result['ci_95'][0]*100:.4f}%, {result['ci_95'][1]*100:.4f}%]")
    
    if result['p_value'] < 0.05:
        print("✓ Reject H0: Strategy appears profitable")
    else:
        print("✗ Fail to reject H0: Insufficient evidence of profitability")
    
    # Example 2: Comparing two strategies
    print("\n\nExample 2: Comparing Two Strategies")
    print("-"*70)
    
    returns_A = np.random.normal(0.08, 1.2, 200) / 100
    returns_B = np.random.normal(0.12, 1.5, 200) / 100
    
    result = two_sample_t_test(returns_A, returns_B, alternative='two-sided')
    
    print(f"Strategy A mean: {result['mean1']*100:.4f}%")
    print(f"Strategy B mean: {result['mean2']*100:.4f}%")
    print(f"Difference:      {result['mean_diff']*100:.4f}%")
    print(f"T-statistic:     {result['t_statistic']:.4f}")
    print(f"P-value:         {result['p_value']:.4f}")
    
    # Example 3: Power analysis
    print("\n\nExample 3: Power Analysis")
    print("-"*70)
    
    delta = 0.08 / 100  # Effect size: 0.08%
    sigma = 1.5 / 100   # Std dev: 1.5%
    
    for n in [100, 500, 1000, 2000, 5000]:
        pwr = power_one_sample_t(n, delta, sigma, alpha=0.05, alternative='one-sided')
        print(f"n={n:4d}: Power = {pwr:.2%}")
    
    # Example 4: Required sample size
    print("\n\nExample 4: Required Sample Size for 80% Power")
    print("-"*70)
    
    n_required = sample_size_one_sample_t(delta, sigma, alpha=0.05, power=0.8, alternative='one-sided')
    print(f"To detect {delta*100:.2f}% mean with 80% power: n = {n_required} observations")
    print(f"(Approximately {n_required/252:.1f} years of daily data)")
    
    # Example 5: Sharpe ratio test
    print("\n\nExample 5: Sharpe Ratio Test")
    print("-"*70)
    
    returns_strategy = np.random.normal(0.001, 0.02, 252)  # 1 year daily
    
    result = sharpe_ratio_test(returns_strategy, alpha=0.05)
    
    print(f"Sharpe ratio (daily):      {result['sharpe_ratio']:.4f}")
    print(f"Sharpe ratio (annualized): {result['annualized_sharpe']:.4f}")
    print(f"T-statistic:               {result['t_statistic']:.4f}")
    print(f"P-value:                   {result['p_value']:.4f}")
    print(f"95% CI:                    [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")
    
    # Example 6: Multiple testing correction
    print("\n\nExample 6: Multiple Testing Correction")
    print("-"*70)
    
    # Simulate testing 20 strategies
    p_values = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 
                         0.120, 0.205, 0.321, 0.440, 0.552, 0.631, 0.702,
                         0.751, 0.803, 0.849, 0.901, 0.945, 0.982])
    
    sig_uncorrected = p_values < 0.05
    sig_bonferroni = bonferroni_correction(p_values, alpha=0.05)
    sig_fdr = fdr_correction(p_values, alpha=0.05)
    
    print(f"Uncorrected significant: {np.sum(sig_uncorrected)} / {len(p_values)}")
    print(f"Bonferroni significant:  {np.sum(sig_bonferroni)} / {len(p_values)}")
    print(f"FDR significant:         {np.sum(sig_fdr)} / {len(p_values)}")
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
\`\`\`

---

## Summary

**Essential Skills:**
1. **Construct confidence intervals** - especially for means, Sharpe ratios
2. **Perform hypothesis tests** - one-sample, two-sample, paired
3. **Interpret p-values correctly** - what they mean and don't mean
4. **Calculate power and sample size** - understand feasibility
5. **Handle multiple testing** - Bonferroni, FDR
6. **Avoid pitfalls** - p-hacking, look-ahead bias, overfitting

**Interview Tips:**
1. Always state H₀ and H₁ explicitly
2. Check assumptions (normality, independence)
3. Consider practical vs. statistical significance
4. Acknowledge limitations (small sample, violations)
5. Discuss Type I vs Type II error tradeoffs
6. Connect to trading context

**Key Formulas to Memorize:**
- SE = σ/√n
- 95% CI: X̄ ± 1.96 × SE
- t = (X̄ - μ₀) / (s/√n)
- Sample size: n ≈ [(z_{1-α} + z_{1-β}) × σ / δ]²
- Bonferroni: α_adj = α/k

**Common α and z-values:**
- α=0.05 (two-sided): z=1.96
- α=0.05 (one-sided): z=1.645
- α=0.01 (two-sided): z=2.576
- 80% power: z=0.842
- 90% power: z=1.282

Statistical inference separates data-driven decision-making from guesswork. Master these concepts to become a rigorous quantitative trader!
`,
};
