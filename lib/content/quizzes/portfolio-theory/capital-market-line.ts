export const capitalMarketLineQuiz = {
    id: 'capital-market-line',
    title: 'Capital Market Line',
    questions: [
        {
            id: 'cml-derivation',
            text: `Derive the Capital Market Line (CML) equation starting from first principles. Given a risk-free rate of 3%, market portfolio return of 11%, and market portfolio volatility of 18%, calculate: (1) the slope of the CML and interpret its meaning, (2) the expected return and risk for a portfolio that is 150% invested in the market (50% leverage), (3) the optimal allocation to the market portfolio for an investor with utility function U = E(R) - 0.5Aσ² where A = 4, and (4) explain why the CML dominates the efficient frontier and what this means for optimal investing.`,
            type: 'discussion' as const,
            sampleAnswer: `**1. Deriving the Capital Market Line**

**Starting Principles:**

The CML represents all portfolios that combine the risk-free asset with the market portfolio (tangency portfolio).

**Portfolio Construction:**
- Weight in market portfolio: w
- Weight in risk-free asset: (1-w)

**Portfolio Return:**
\\[
R_p = wR_M + (1-w)R_f = R_f + w(R_M - R_f)
\\]

**Portfolio Risk:**
\\[
\\sigma_p = w\\sigma_M
\\]

(Risk-free asset has zero variance and zero covariance with market)

**Solving for w:**
\\[
w = \\frac{\\sigma_p}{\\sigma_M}
\\]

**Substituting back into return equation:**
\\[
R_p = R_f + \\frac{\\sigma_p}{\\sigma_M}(R_M - R_f)
\\]

**Capital Market Line Equation:**
\\[
E(R_p) = R_f + \\frac{E(R_M) - R_f}{\\sigma_M} \\cdot \\sigma_p
\\]

Or more compactly:
\\[
E(R_p) = R_f + Sharpe_M \\cdot \\sigma_p
\\]

**Geometric Interpretation:**
- Straight line starting at (0, Rf)
- Passes through market portfolio point (σ_M, R_M)
- Slope = market Sharpe ratio = (R_M - Rf) / σ_M

**Given Data:**
- Rf = 3%
- R_M = 11%
- σ_M = 18%

**CML Equation:**
\\[
E(R_p) = 3\\% + \\frac{11\\% - 3\\%}{18\\%} \\cdot \\sigma_p
\\]
\\[
E(R_p) = 3\\% + 0.444 \\cdot \\sigma_p
\\]

**Slope Calculation:**
\\[
Slope = \\frac{R_M - R_f}{\\sigma_M} = \\frac{11\\% - 3\\%}{18\\%} = \\frac{8\\%}{18\\%} = 0.444
\\]

**Interpretation of Slope (0.444):**1. **Marginal return per unit of risk:** For each 1% increase in portfolio risk, expected return increases by 0.444%

2. **Market Sharpe ratio:** The slope IS the Sharpe ratio of the market portfolio - the reward-to-risk ratio

3. **Price of risk:** In equilibrium, 1 unit of volatility is "priced" at 0.444 units of excess return

4. **Comparison benchmark:**
   - Sharpe > 0.444: Superior to market
   - Sharpe < 0.444: Inferior to market
   - Sharpe = 0.444: On the CML (optimal)

5. **Historical context:** 
   - Long-term S&P 500 Sharpe ≈ 0.40-0.50
   - 0.444 is realistic and attractive

**2. Leveraged Portfolio (150% Market Exposure)**

**Allocation:**
- w = 150% in market portfolio
- (1-w) = -50% in risk-free asset (borrowing)

**Expected Return:**
\\[
E(R_p) = 150\\% \\times 11\\% + (-50\\%) \\times 3\\%
\\]
\\[
E(R_p) = 16.5\\% - 1.5\\% = 15.0\\%
\\]

**Portfolio Risk:**
\\[
\\sigma_p = 150\\% \\times 18\\% = 27.0\\%
\\]

**Verification using CML:**
\\[
E(R_p) = 3\\% + 0.444 \\times 27\\% = 3\\% + 12\\% = 15.0\\% \\checkmark
\\]

**Sharpe Ratio:**
\\[
Sharpe = \\frac{15\\% - 3\\%}{27\\%} = \\frac{12\\%}{27\\%} = 0.444
\\]

Same as market Sharpe! (As expected on CML)

**Risk/Return Analysis:**

**Unlevered Market:**
- Return: 11%
- Risk: 18%
- Excess return: 8%

**Levered 1.5x:**
- Return: 15% (+36% higher)
- Risk: 27% (+50% higher)
- Excess return: 12% (+50% higher)

**Key Insights:**1. **Linear scaling:** Both return and risk scale proportionally with leverage (1.5x)

2. **Cost of leverage:** Borrowing at 3% costs 1.5% (50% × 3%), reducing gross return from 16.5% to 15%

3. **Amplification:** 50% leverage amplifies both gains and losses by 50%

4. **No free lunch:** Sharpe ratio unchanged - higher return exactly compensates for higher risk

5. **Practical considerations:**
   - Margin requirement: Typically need 50% equity, so max 2:1 leverage
   - Margin interest: Real borrowing costs 4-8% (higher than Rf)
   - Margin calls: If portfolio falls, forced liquidation risk
   - Volatility drag: Higher variance → lower geometric mean return

**Realistic Leverage Return:**

With 5% borrowing rate (not 3% Rf):
\\[
E(R_p) = 150\\% \\times 11\\% - 50\\% \\times 5\\% = 16.5\\% - 2.5\\% = 14.0\\%
\\]

Borrowing cost reduces return by 0.5% (from 14.5% to 14.0%).

**3. Optimal Allocation with Utility Function**

**Utility Function:**
\\[
U = E(R_p) - 0.5 A\\sigma_p^2
\\]

Where A = 4 (risk aversion coefficient)

**Portfolio characteristics (in terms of w):**
\\[
E(R_p) = R_f + w(R_M - R_f) = 3\\% + w(8\\%)
\\]
\\[
\\sigma_p^2 = w^2\\sigma_M^2 = w^2(0.18)^2 = 0.0324w^2
\\]

**Utility function:**
\\[
U = 3\\% + 8\\%w - 0.5(4)(0.0324)w^2
\\]
\\[
U = 3\\% + 8\\%w - 0.0648w^2
\\]

**Maximize utility (take derivative and set to zero):**
\\[
\\frac{dU}{dw} = 8\\% - 2(0.0648)w = 0
\\]
\\[
8\\% = 0.1296w
\\]
\\[
w^* = \\frac{8\\%}{0.1296} = \\frac{0.08}{0.1296} = 0.617 = 61.7\\%
\\]

**Optimal Portfolio:**
- **61.7% in market portfolio**
- **38.3% in risk-free asset**

**Resulting Metrics:**

**Expected Return:**
\\[
E(R_p) = 3\\% + 0.617(8\\%) = 3\\% + 4.94\\% = 7.94\\%
\\]

**Risk:**
\\[
\\sigma_p = 0.617 \\times 18\\% = 11.1\\%
\\]

**Utility:**
\\[
U = 7.94\\% - 0.5(4)(0.111)^2 = 7.94\\% - 2(0.0123) = 7.94\\% - 2.46\\% = 5.48\\%
\\]

**Sharpe Ratio:**
\\[
Sharpe = \\frac{7.94\\% - 3\\%}{11.1\\%} = \\frac{4.94\\%}{11.1\\%} = 0.445
\\]

(Slight rounding difference from 0.444)

**Interpretation:**1. **Moderate risk tolerance:** A=4 is moderate risk aversion → ~62% in risky assets is reasonable

2. **Below full market exposure:** Conservative allocation (38% cash) because investor is risk-averse

3. **Risk-return tradeoff:** Accepting lower return (7.94% vs 11%) to significantly reduce risk (11.1% vs 18%)

4. **Comparison to benchmarks:**
   - 100% market: Return 11%, Risk 18%, Utility = 11% - 2(3.24%) = 4.52%
   - 62% market: Return 7.94%, Risk 11.1%, Utility = **5.48%** (HIGHER!)
   - 0% market: Return 3%, Risk 0%, Utility = 3%

**Optimal allocation maximizes utility (5.48%) not raw return.**

**General Formula for Optimal Allocation:**

\\[
w^* = \\frac{E(R_M) - R_f}{A\\sigma_M^2}
\\]

\\[
w^* = \\frac{8\\%}{4 \\times (0.18)^2} = \\frac{0.08}{4 \\times 0.0324} = \\frac{0.08}{0.1296} = 0.617
\\]

**Sensitivity to Risk Aversion:**

| Risk Aversion (A) | Optimal w | Market % | Cash % | Return | Risk |
|------------------|-----------|----------|---------|--------|------|
| 2 (aggressive) | 123% | 123% | -23% | 12.3% | 22.1% |
| 3 | 82% | 82% | 18% | 9.6% | 14.8% |
| 4 (our case) | 62% | 62% | 38% | 7.9% | 11.1% |
| 6 | 41% | 41% | 59% | 6.3% | 7.4% |
| 8 (conservative) | 31% | 31% | 69% | 5.5% | 5.6% |

**Key Insight:** Higher risk aversion → lower allocation to risky market → more cash → lower return and risk.

**4. Why CML Dominates the Efficient Frontier**

**Dominance Theorem:**

Every point on the CML (except the market portfolio itself) **dominates** all other points on the efficient frontier with the same risk level.

**Mathematical Proof:**

For any risk level σ:

**CML portfolio return:**
\\[
R_{CML} = R_f + \\frac{R_M - R_f}{\\sigma_M} \\cdot \\sigma
\\]

**Efficient frontier portfolio return** (without risk-free asset):
\\[
R_{EF} < R_{CML}
\\]

For σ < σ_M (lower risk than market):

**Why?** 

The efficient frontier is concave (hyperbola), while CML is linear. The line connecting Rf to any point on the frontier can be extended to dominate that point.

**Graphical Proof:**

\`\`\`
Return
  ^
  | CML(linear, steeper)
        |          /
        |         /
        |        /● Market Portfolio (tangency)
        |       /
        |      /  Efficient Frontier (curved, below CML)
        |     /.──────
        |    /╱
        |   /╱
        |  /╱
        | /╱
        |/____________________> Risk
  Rf
            ```

The CML lies **above** the efficient frontier everywhere except at the market portfolio (tangency point).

**Example:**

**Target Risk: 12%**

**Option 1: Efficient Frontier Portfolio**
- Optimal diversified portfolio without risk-free asset
- Expected return: ~7.5% (from frontier optimization)
- Sharpe: (7.5% - 3%) / 12% = 0.375

**Option 2: CML Portfolio (62% market, 38% cash)**
- Expected return: 3% + 0.444 × 12% = 8.33%
- Sharpe: (8.33% - 3%) / 12% = 0.444

**CML beats EF by 0.83% at same risk!**

**Why CML Dominates:**1. **Two-fund separation:** Combining risk-free asset with market portfolio is more efficient than any single diversified portfolio

2. **Linearity advantage:** Linear combinations of Rf and market beat curved efficient frontier

3. **Market portfolio optimality:** The market portfolio is the tangency portfolio - maximum Sharpe ratio on the frontier

4. **No diversification benefit lost:** Market portfolio itself is fully diversified; adding cash doesn't reduce diversification

**Implications for Optimal Investing:**

**1. Simplicity:**
- Don't need to construct complex frontier portfolios
- Just decide one number: allocation to market portfolio
- Rest goes to risk-free asset (or borrow if aggressive)

**2. Index fund strategy:**
- Hold market index fund (S&P 500, Total Market)
- Adjust risk with cash allocation (not by picking different stocks)
- Low cost, tax efficient, simple

**3. Two-fund theorem:**
- Every investor should hold:
  - **Fund 1:** Risk-free asset (T-bills, money market)
  - **Fund 2:** Market portfolio (broad index)
- Only the **ratio** differs by risk tolerance

**4. Separation of concerns:**
- **Portfolio decision:** What's the optimal risky portfolio? (Answer: Market)
- **Allocation decision:** How much to allocate to risky assets? (Answer: Depends on risk tolerance)
- These are independent decisions!

**5. Market efficiency:**
- If markets are efficient, market portfolio is optimal
- Active management (trying to beat market) is suboptimal
- Just hold index and adjust leverage/cash allocation

**Practical Example: Three Investors**

All agree market is optimal risky portfolio (CML dominance), differ only in risk tolerance:

**Conservative (30% market):**
- 30% VTI (Total Market ETF)
- 70% BIL (T-bills ETF)
- Return: 5.4%, Risk: 5.4%

**Moderate (70% market):**
- 70% VTI
- 30% BIL
- Return: 8.6%, Risk: 12.6%

**Aggressive (120% market with leverage):**
- 120% VTI (using margin or 2x ETF)
- -20% (borrowing)
- Return: 12.8%, Risk: 21.6%

**All on CML, all optimal for their risk preferences!**

**When CML Doesn't Dominate:**1. **Market inefficiency:** If you can identify mispriced assets
2. **Constraints:** Transaction costs, taxes, restrictions prevent accessing market portfolio
3. **Leverage costs:** Borrowing rate much higher than Rf
4. **Short selling prohibited:** Can't create negative cash position
5. **Market proxy imperfect:** "Market portfolio" should be all assets (stocks, bonds, real estate, human capital) but we use S&P 500

**Conclusion:**

The CML represents the **"efficient set"** when a risk-free asset exists. It dominates the efficient frontier because combining the risk-free asset with the market portfolio (tangency portfolio) provides better risk-return tradeoffs than any other strategy. This leads to profound implications:

- **Simplicity:** Hold index + cash
- **Optimality:** CML portfolio is mean-variance optimal
- **Universality:** Same risky portfolio for all investors
- **Passive investing justification:** Market portfolio is optimal, no need for active management

The CML is the theoretical foundation for index investing and the practical reason why passive index funds have beaten most active managers over long periods.`,
        keyPoints: [
            'CML equation: E(Rp) = Rf + [(RM-Rf)/σM] × σp; linear relationship between risk and return when risk-free asset available',
            'CML slope = market Sharpe ratio = (RM-Rf)/σM = 0.444 in example; represents price of risk in equilibrium',
            'Leverage scales return and risk proportionally; 150% market gives 15% return, 27% risk, same 0.444 Sharpe ratio',
            'Optimal allocation formula: w* = (RM-Rf)/(A×σM²); higher risk aversion A → lower allocation to market',
            'CML dominates efficient frontier everywhere except tangency point; combining Rf with market beats other frontier portfolios',
            'Two-fund separation theorem: all investors hold same market portfolio, differ only in Rf/market ratio based on risk tolerance',
            'Practical implication: hold market index fund, adjust risk with cash allocation rather than picking different stocks',
            'CML provides theoretical foundation for passive index investing strategy used by Vanguard, Bogle, and billions in assets'
        ]
    },
    {
        id: 'cml-vs-sml',
        text: `Compare and contrast the Capital Market Line (CML) and the Security Market Line (SML) from the Capital Asset Pricing Model (CAPM). Explain: (1) what each line represents and what can be plotted on each, (2) why CML applies only to efficient portfolios while SML applies to all assets and portfolios, (3) the difference between total risk (σ) and systematic risk (β) and why each line uses different risk measures, and (4) provide a real-world example of an asset that lies on the SML but below the CML, explaining what this means for investment decisions.`,
            type: 'discussion' as const,
                sampleAnswer: `**1. What Each Line Represents**

**Capital Market Line (CML):**

**Equation:**
\\[
E(R_p) = R_f + \\frac{E(R_M) - R_f}{\\sigma_M} \\cdot \\sigma_p
\\]

**Risk measure:** Total risk (standard deviation, σ)

**What it represents:**
- Efficient portfolios combining risk-free asset with market portfolio
- Best possible risk-return tradeoff for **portfolios**
- **Only applies to well-diversified, efficient portfolios**

**What can be plotted:**
- ✓ Market portfolio
- ✓ Combinations of market portfolio + risk-free asset
- ✓ Leveraged market portfolio positions
- ✗ Individual stocks (typically below CML)
- ✗ Inefficient portfolios (below CML)

**Axes:**
- X-axis: Total risk (σ - standard deviation)
- Y-axis: Expected return

**Security Market Line (SML):**

**Equation:**
\\[
E(R_i) = R_f + \\beta_i[E(R_M) - R_f]
\\]

**Risk measure:** Systematic risk (beta, β)

**What it represents:**
- Equilibrium required return for **any asset or portfolio**
- Relationship between systematic risk and expected return
- **Applies universally to all assets**

**What can be plotted:**
- ✓ Individual stocks
- ✓ Bonds
- ✓ Portfolios (efficient or not)
- ✓ Derivatives
- ✓ Any asset with measurable beta

**Axes:**
- X-axis: Systematic risk (β - beta)
- Y-axis: Expected return

**Visual Comparison:**

\`\`\`
CML(plots σ): SML(plots β):

Return                      Return
    | CML | SML
    | /                       |   /
    | /● Market                 | /● Market
        | /___________> σ            |/___________ > β
  Rf    Total Risk            Rf  Systematic Risk
    ```

**Key Difference Summary:**

| Feature | CML | SML |
|---------|-----|-----|
| **Applies to** | Efficient portfolios only | All assets & portfolios |
| **Risk measure** | Total risk (σ) | Systematic risk (β) |
| **Equation** | Linear in σ | Linear in β |
| **Theoretical basis** | Portfolio theory | CAPM |
| **Use case** | Portfolio optimization | Asset pricing |
| **Market portfolio** | Tangency point | β=1 point |

**2. Why CML Only for Efficient Portfolios, SML for All Assets**

**CML: Efficient Portfolios Only**

**Reason 1: Diversification Assumption**

The CML assumes investors hold **fully diversified portfolios** where:
- Idiosyncratic (firm-specific) risk is eliminated
- Only systematic (market) risk remains
- Total risk = systematic risk (approximately)

For efficient portfolios: σ_portfolio ≈ β_portfolio × σ_market

**Reason 2: Portfolio Construction**

CML represents portfolios constructed by combining:
- Risk-free asset (σ=0, β=0)
- Market portfolio (fully diversified)

**Result:** Only efficient combinations lie on CML.

**Reason 3: Dominance**

Any portfolio **below** the CML is dominated - you can achieve better return at same risk by using CML portfolios.

Individual stocks are almost always below CML because they contain diversifiable risk.

**Example: Why Individual Stock Below CML**

**Tesla stock:**
- Expected return: 15%
- Total risk (σ): 45%
- Beta: 1.8

**Plot on CML** (using σ=45%):
- CML return at σ=45%: Rf + Sharpe_M × 45% = 3% + 0.444 × 45% = **23%**
- Tesla actual: **15%**
- Tesla is **8% below CML!**

**Why?** Tesla's 45% volatility includes:
- Systematic risk (from market exposure)
- Idiosyncratic risk (Tesla-specific events)

CML expects 23% return for 45% risk because it assumes that risk is **all systematic** (diversified portfolio). But Tesla's risk is partly diversifiable, so it doesn't earn the full CML return.

**SML: All Assets**

**Reason 1: Asset Pricing Model**

SML is derived from CAPM, which prices **any asset** based on systematic risk:

\\[
E(R_i) = R_f + \\beta_i \\lambda_M
\\]

Where λ_M = market risk premium = E(R_M) - R_f

**Reason 2: Systematic Risk Only**

Beta measures only **non-diversifiable risk** - the risk that matters in a diversified portfolio.

**Reason 3: Marginal Contribution**

When adding an asset to a diversified portfolio:
- Idiosyncratic risk diversifies away (becomes negligible)
- Only systematic risk (beta) matters for portfolio risk
- Therefore, only beta determines required return

**Example: Tesla on SML**

**Tesla:**
- Beta: 1.8
- SML required return: 3% + 1.8(11%-3%) = 3% + 1.8(8%) = **17.4%**
- Actual expected return: **15%**

**Tesla is 2.4% below SML → Overpriced!**

**All investors with diversified portfolios should avoid Tesla** (expected return too low for its systematic risk).

**3. Total Risk (σ) vs. Systematic Risk (β)**

**Total Risk (σ - Standard Deviation):**

**Definition:** Volatility of asset's returns (standard deviation)

**Components:**
\\[
Total\\ Risk = Systematic\\ Risk + Idiosyncratic\\ Risk
\\]

**Formula:**
\\[
\\sigma_i^2 = \\beta_i^2 \\sigma_M^2 + \\sigma_{\\epsilon_i}^2
\\]

Where:
- β²σ²_M = systematic variance
- σ²_ε = idiosyncratic (diversifiable) variance

**Characteristics:**
- Measures **all volatility** from all sources
- Includes company-specific events (earnings, lawsuits, product launches)
- Includes market-wide events (recessions, rate changes)
- **Cannot be diversified away completely**
- Relevant for **undiversified** investors

**Example: Individual Stock Total Risk**

Amazon (AMZN):
- σ = 40% (total volatility)
- β = 1.3 (systematic risk)
- Systematic component: 1.3 × 18% market vol ≈ 23%
- Idiosyncratic component: √(40² - 23²) ≈ 32%

**72% of Amazon's risk is idiosyncratic (company-specific)!**

**Systematic Risk (β - Beta):**

**Definition:** Sensitivity of asset returns to market returns

**Formula:**
\\[
\\beta_i = \\frac{Cov(R_i, R_M)}{Var(R_M)} = \\frac{\\rho_{i,M} \\sigma_i \\sigma_M}{\\sigma_M^2} = \\rho_{i,M} \\frac{\\sigma_i}{\\sigma_M}
\\]

**Interpretation:**
- β = 1: Moves with market (1% market change → 1% asset change)
- β > 1: Amplifies market (aggressive, high systematic risk)
- β < 1: Dampens market (defensive, low systematic risk)
- β = 0: Uncorrelated with market (no systematic risk)

**Characteristics:**
- Measures **only market-related risk**
- **Cannot be diversified away** (it's the risk remaining in diversified portfolios)
- Relevant for **diversified** investors
- Used in CAPM asset pricing

**Why CML Uses Total Risk (σ):**

**Reason 1:** CML applies to portfolios where you **choose the risk level** by adjusting allocation between risk-free asset and market.

For these portfolios, total risk = systematic risk (fully diversified).

**Reason 2:** The combination "w% market + (1-w)% risk-free" has:
- Total risk: w × σ_M
- Return: Rf + w(R_M - Rf)
- These create a linear relationship between σ and return

**Why SML Uses Systematic Risk (β):**

**Reason 1:** In equilibrium, **only systematic risk is priced**.

Investors can diversify away idiosyncratic risk for free, so they shouldn't demand extra return for bearing it.

**Reason 2:** For any asset added to a diversified portfolio:
- Marginal contribution to portfolio risk ∝ β
- Required return should compensate for this marginal risk
- Therefore, return is linear in β, not σ

**Mathematical Relationship:**

For the **market portfolio:**
- β_M = 1 (by definition)
- σ_M = 18%
- Total risk = Systematic risk (fully diversified)

For the **market portfolio, CML and SML coincide**:
- CML: E(R_M) = Rf + (Sharpe_M) × σ_M
- SML: E(R_M) = Rf + 1 × (R_M - Rf) = R_M ✓

**Numerical Example:**

**Risk-Free Asset:**
- CML: σ=0 → E(R) = Rf = 3% ✓
- SML: β=0 → E(R) = 3% + 0×8% = 3% ✓

**Market Portfolio:**
- CML: σ=18% → E(R) = 3% + 0.444×18% = 11% ✓
- SML: β=1 → E(R) = 3% + 1×8% = 11% ✓

**Individual Stock (Tesla):**
- CML: σ=45% → E(R) = 3% + 0.444×45% = 23%
  - But Tesla only offers 15% → **Far below CML**
- SML: β=1.8 → E(R) = 3% + 1.8×8% = 17.4%
  - Tesla offers 15% → **Slightly below SML (overpriced)**

**The difference (23% vs 17.4%) is due to Tesla's idiosyncratic risk not being compensated.**

**4. Real-World Example: Asset on SML but Below CML**

**Example: Johnson & Johnson (JNJ)**

**Characteristics (approximate):**
- Expected return: 9%
- Total volatility (σ): 16%
- Beta (β): 0.7
- Correlation with market: 0.8

**Analysis:**

**Plot on SML** (using β=0.7):
\\[
SML\\ Required\\ Return = R_f + \\beta(R_M - R_f)
\\]
\\[
= 3\\% + 0.7(8\\%) = 3\\% + 5.6\\% = 8.6\\%
\\]

**JNJ expected return: 9.0%**

**JNJ position on SML:** **Slightly ABOVE SML** (9.0% > 8.6%)
→ Fairly priced or slightly underpriced
→ Attractive for diversified investors

**Plot on CML** (using σ=16%):
\\[
CML\\ Expected\\ Return = R_f + Sharpe_M \\times \\sigma
\\]
\\[
= 3\\% + 0.444 \\times 16\\% = 3\\% + 7.1\\% = 10.1\\%
\\]

**JNJ expected return: 9.0%**

**JNJ position on CML:** **Below CML** (9.0% < 10.1%)
→ Underperforming for its total risk
→ Less attractive for undiversified investors

**Explanation:**

**JNJ has idiosyncratic risk that's not compensated:**

Total risk decomposition:
- Systematic component: β × σ_M = 0.7 × 18% = 12.6%
- Total risk: 16%
- Idiosyncratic risk: √(16² - 12.6²) ≈ 10%

**~35% of JNJ's risk is company-specific** (pharmaceutical trials, patent expirations, lawsuits).

**For diversified investors:** Idiosyncratic risk is irrelevant (diversifies away)
→ JNJ fairly priced on SML
→ **Good investment** for portfolios

**For undiversified investors:** Must bear idiosyncratic risk without compensation
→ JNJ below CML
→ **Poor investment** for concentrated portfolios

**Investment Decision Implications:**

**Scenario 1: Institutional Investor (Diversified)**

Portfolio: 500+ stocks, fully diversified

**Decision:** **BUY Johnson & Johnson**

**Reasoning:**
- JNJ on/above SML → fair/underpriced for systematic risk
- Idiosyncratic risk diversifies away across 500 stocks
- Contributing 0.2% position to portfolio
- Only beta matters for portfolio risk contribution

**Scenario 2: Retail Investor (Concentrated)**

Portfolio: 5-10 stocks, poorly diversified

**Decision:** **AVOID or underweight Johnson & Johnson**

**Reasoning:**
- JNJ below CML → insufficient return for total risk
- 16% volatility with only 9% return is unattractive
- Idiosyncratic risk does NOT diversify away (only 10 stocks)
- Better to hold diversified index fund (on CML)

**Scenario 3: Risk-Averse Investor**

Wants defensive stocks (low beta)

**Decision:** **Depends on implementation**

**Option A: Buy JNJ directly**
- Get 9% return, 16% total risk
- Below CML, suboptimal

**Option B: Buy market portfolio + T-bills**
- Allocate 70% to market portfolio (same β=0.7 exposure)
- Get 8.6% return, 12.6% risk (lower!)
- **On CML, optimal**

**Conclusion:** Even for defensive investors, better to lever down the market portfolio than buy low-beta stocks directly.

**Real-World Example 2: Utility Stock**

**Duke Energy (DUK):**
- Return: 7%
- Volatility: 14%
- Beta: 0.4

**SML analysis:**
- Required: 3% + 0.4(8%) = 6.2%
- Actual: 7.0%
- **Above SML → Underpriced → Buy for diversified portfolio**

**CML analysis:**
- Expected at σ=14%: 3% + 0.444(14%) = 9.2%
- Actual: 7.0%
- **Below CML → Underperforming → Avoid for concentrated portfolio**

**Practical Takeaway:**

**For diversified investors (most institutions, index funds):**
- Use SML for evaluation
- Focus on systematic risk (beta)
- Buy assets above SML (underpriced)
- Sell assets below SML (overpriced)

**For undiversified investors (retail, concentrated):**
- CML is more relevant benchmark
- Total risk matters (can't diversify)
- Better to hold market index + cash than individual stocks
- Individual stocks typically below CML due to idiosyncratic risk

**Summary of Key Insight:**

Almost all individual stocks lie **on the SML** (priced fairly for systematic risk) but **below the CML** (insufficient return for total risk). This is expected and normal - it doesn't mean stocks are bad investments. It means:

1. For diversified investors: Stocks are priced correctly (SML)
2. For concentrated investors: Holding individual stocks is suboptimal (below CML) - should hold diversified index instead

The CML tells us **diversification is powerful** - by eliminating idiosyncratic risk, diversified portfolios achieve better risk-return tradeoffs than individual securities.`,
    keyPoints: [
        'CML plots total risk (σ) vs return for efficient portfolios only; SML plots systematic risk (β) vs return for all assets',
        'CML applies to combinations of risk-free asset + market portfolio; SML applies universally to individual stocks, bonds, any asset',
        'Total risk = systematic risk + idiosyncratic risk; only systematic risk is priced in equilibrium (CAPM)',
        'Individual stocks typically lie on SML (fairly priced for beta) but below CML (insufficient return for total volatility)',
        'Beta measures only non-diversifiable market risk; sigma measures all volatility including company-specific events',
        'For diversified investors, SML is relevant (idiosyncratic risk diversifies away); for concentrated investors, CML matters',
        'Example: JNJ at 9% return, 16% vol, 0.7 beta → on/above SML (good for diversified) but below CML (poor for concentrated)',
        'Practical implication: institutional investors use SML for stock selection; retail should hold market index (on CML) not individual stocks'
    ]
    },
{
    id: 'cml-leverage',
        text: `Analyze the role of leverage in the Capital Market Line framework. A hedge fund wants to target 15% annualized return using CML strategy (market portfolio + risk-free asset combinations). Given: Rf = 3%, market return = 11%, market volatility = 18%. Calculate: (1) the required allocation to achieve 15% return and the resulting portfolio volatility, (2) the leverage ratio and amount that needs to be borrowed per $1 million invested, (3) the impact on returns if the borrowing rate is 5% instead of the risk-free rate of 3%, and (4) discuss the practical risks of leverage (margin calls, volatility drag, fat tails) and when leveraged CML strategies are appropriate vs. inappropriate.`,
            type: 'discussion' as const,
                sampleAnswer: `**1. Required Allocation for 15% Return**

**CML Equation:**
\\[
E(R_p) = R_f + w(R_M - R_f)
\\]

Where w = weight in market portfolio

**Given:**
- Target return: E(Rp) = 15%
- Rf = 3%
- R_M = 11%
- σ_M = 18%

**Solving for w:**
\\[
15\\% = 3\\% + w(11\\% - 3\\%)
\\]
\\[
15\\% = 3\\% + w(8\\%)
\\]
\\[
12\\% = 8\\%w
\\]
\\[
w = \\frac{12\\%}{8\\%} = 1.5 = 150\\%
\\]

**Allocation:**
- **150% in market portfolio** (long position)
- **-50% in risk-free asset** (short position = borrowing)

**Portfolio Volatility:**
\\[
\\sigma_p = w \\times \\sigma_M = 1.5 \\times 18\\% = 27\\%
\\]

**Verification using CML:**
\\[
E(R_p) = 3\\% + \\frac{11\\%-3\\%}{18\\%} \\times 27\\%
\\]
\\[
= 3\\% + 0.444 \\times 27\\% = 3\\% + 12\\% = 15\\% \\checkmark
\\]

**Sharpe Ratio:**
\\[
Sharpe = \\frac{15\\% - 3\\%}{27\\%} = \\frac{12\\%}{27\\%} = 0.444
\\]

Same as unlevered market (as expected on CML).

**Summary:**
- **Required allocation: 150% market**
- **Resulting volatility: 27%**
- **Leverage multiple: 1.5x**

**2. Leverage Ratio and Borrowing Amount**

**Leverage Ratio:**

\\[
Leverage\\ Ratio = \\frac{Gross\\ Assets}{Net\\ Assets} = \\frac{150\\%}{100\\%} = 1.5
\\]

**Interpretation:**
- For every $1 of equity, control $1.50 of assets
- Borrowing $0.50 for every $1 of equity
- **Debt-to-equity ratio: 50%**

**For $1,000,000 Investment:**

**Balance Sheet:**

**Assets:**
- Market portfolio position: $1,500,000

**Liabilities:**
- Borrowing: $500,000

**Equity:**
- Investor capital: $1,000,000

**Verification:** $1,500,000 - $500,000 = $1,000,000 ✓

**How to Implement:**

**Method 1: Margin Account**
- Deposit $1,000,000 cash
- Borrow $500,000 from broker (margin loan)
- Buy $1,500,000 of S&P 500 ETF (SPY)
- Broker requires minimum 50% equity ratio (Reg T)
- Our 67% equity ratio ($1M / $1.5M) > 50% → Safe margin

**Method 2: Futures**
- Buy $1,500,000 notional exposure of S&P 500 futures
- Post $150,000 margin (10% of notional)
- Keep $850,000 in T-bills
- **More capital efficient** (only $150K tied up vs $1M)
- Lower borrowing costs implicit in futures pricing

**Method 3: Leveraged ETF**
- Buy $1,000,000 of 1.5x leveraged S&P 500 ETF
- ETF internally uses swaps/futures for leverage
- **Simplest but higher fees** (0.9% vs 0.03% for SPY)
- **Volatility decay** from daily rebalancing

**Borrowing Details:**

**Annual borrowing cost** (at Rf = 3%):
\\[
Cost = Borrowed\\ Amount \\times Rate = \\$500,000 \\times 3\\% = \\$15,000
\\]

**Interest as % of equity:**
\\[
\\frac{\\$15,000}{\\$1,000,000} = 1.5\\%
\\]

**Monthly interest payment:**
\\[
\\frac{\\$15,000}{12} = \\$1,250
\\]

**3. Impact of Higher Borrowing Rate (5% vs 3%)**

**Scenario A: Borrowing at Risk-Free Rate (3%)**

**Gross return:**
\\[
Gross\\ Return = 1.5 \\times 11\\% = 16.5\\%
\\]

**Borrowing cost:**
\\[
Cost = 0.5 \\times 3\\% = 1.5\\%
\\]

**Net return:**
\\[
Net\\ Return = 16.5\\% - 1.5\\% = 15.0\\%
\\]

**Dollar returns on $1M:**
\\[
Profit = \\$1,000,000 \\times 15\\% = \\$150,000
\\]

**Scenario B: Borrowing at Market Rate (5%)**

**Gross return:**
\\[
Gross\\ Return = 1.5 \\times 11\\% = 16.5\\%
\\]

**Borrowing cost:**
\\[
Cost = 0.5 \\times 5\\% = 2.5\\%
\\]

**Net return:**
\\[
Net\\ Return = 16.5\\% - 2.5\\% = 14.0\\%
\\]

**Dollar returns on $1M:**
\\[
Profit = \\$1,000,000 \\times 14\\% = \\$140,000
\\]

**Impact Analysis:**

| Metric | @ 3% Rate | @ 5% Rate | Difference |
|--------|-----------|-----------|------------|
| Gross Return | 16.5% | 16.5% | 0% |
| Borrowing Cost | 1.5% | 2.5% | +1.0% |
| Net Return | 15.0% | 14.0% | -1.0% |
| Dollar Profit | $150,000 | $140,000 | -$10,000 |
| Sharpe Ratio | 0.444 | 0.407 | -8.3% |

**Key Insights:**1. **Each 1% increase in borrowing rate reduces net return by 0.5%** (because borrowing is 50% of equity)

2. **Formula:**
\\[
Net\\ Return = w \\times R_M - (w-1) \\times Borrow\\ Rate
\\]
\\[
= 1.5(11\\%) - 0.5(5\\%) = 16.5\\% - 2.5\\% = 14\\%
\\]

3. **Sharpe ratio decreases** because return falls but risk unchanged:
\\[
Sharpe = \\frac{14\\% - 3\\%}{27\\%} = 0.407
\\]

4. **Below CML** if borrowing rate > Rf (no longer on theoretical CML)

**Breakeven Analysis:**

**Question:** At what market return does leverage break even vs. unlever portfolio?

**Unlevered:** Return = 11%, Risk = 18%

**Levered (1.5x, 5% borrow):** Return = 1.5(R_M) - 0.5(5%) = 1.5R_M - 2.5%

**Breakeven:**
\\[
11\\% = 1.5R_M - 2.5\\%
\\]
\\[
13.5\\% = 1.5R_M
\\]
\\[
R_M = 9\\%
\\]

**If market returns > 9%, leverage wins**
**If market returns < 9%, leverage loses**

With R_M = 11%, leverage still advantageous (14% > 11%) but less so than at 3% borrow rate.

**Realistic Borrowing Costs:**

| Method | Typical Rate | Annual Cost on $500K |
|--------|--------------|---------------------|
| Broker margin | 5-8% | $25,000 - $40,000 |
| Interactive Brokers (benchmark) | 4-5% | $20,000 - $25,000 |
| Portfolio margin | 3-5% | $15,000 - $25,000 |
| Futures (implicit) | ~Rf + 0.5% | $17,500 |
| Leveraged ETF (embedded) | ~Rf + 1-2% | $20,000 - $27,500 |

**Futures or portfolio margin most cost-effective for sophisticated investors.**

**4. Practical Risks of Leverage**

**Risk 1: Margin Calls**

**How margin calls work:**

Broker requires minimum equity ratio (typically 25-30% maintenance margin after initial 50%).

**Example scenario:**

**Starting position:**
- $1.5M assets (SPY)
- $500K borrowing
- $1M equity
- Equity ratio: 67%

**Market drops 20%:**
- Assets fall to: $1.5M × 0.8 = $1.2M
- Borrowing unchanged: $500K
- Equity falls to: $1.2M - $500K = $700K
- Equity ratio: $700K / $1.2M = 58% → Still OK (> 30%)

**Market drops 40%:**
- Assets fall to: $1.5M × 0.6 = $900K
- Borrowing: $500K
- Equity: $900K - $500K = $400K
- Equity ratio: $400K / $900K = 44% → Still OK but getting tight

**Market drops 50%:**
- Assets: $1.5M × 0.5 = $750K
- Borrowing: $500K
- Equity: $750K - $500K = $250K
- Equity ratio: $250K / $750K = 33% → **Near margin call threshold!**

**Market drops 55%:**
- Assets: $1.5M × 0.45 = $675K
- Borrowing: $500K
- Equity: $675K - $500K = $175K
- Equity ratio: $175K / $675K = 26% → **MARGIN CALL!**

**Broker forces liquidation:**
- Must sell $200K+ of assets to restore 30% equity ratio
- **Forced to sell at the worst time** (during crash)
- Realize losses, lock in damage

**2008 Example:**
- S&P 500 fell 57% (peak to trough)
- 1.5x levered investors faced multiple margin calls
- Many forced to liquidate at bottom
- Some lost entire account (> 100% loss possible with leverage!)

**Risk 2: Volatility Drag**

**Problem:** Higher volatility reduces geometric returns below arithmetic returns.

**Formula:**
\\[
Geometric\\ Return \\approx Arithmetic\\ Return - \\frac{\\sigma^2}{2}
\\]

**Unlevered market:**
- Arithmetic return: 11%
- Volatility: 18%
- Volatility drag: 0.18² / 2 = 0.0162 = 1.62%
- Geometric return: 11% - 1.62% = 9.38%

**Levered 1.5x:**
- Arithmetic return: 15% (assuming 3% borrow)
- Volatility: 27%
- Volatility drag: 0.27² / 2 = 0.0365 = 3.65%
- Geometric return: 15% - 3.65% = 11.35%

**Impact:** 
- Arithmetic return increases 36% (11% → 15%)
- Geometric return increases only 21% (9.38% → 11.35%)
- **Leverage benefit partially eaten by volatility drag**

**Extreme example (2x leverage):**
- Arithmetic: 19% (2×11% - 1×3%)
- Volatility: 36%
- Drag: 0.36² / 2 = 6.48%
- Geometric: 19% - 6.48% = 12.52%

**Only 33% geometric gain for 100% leverage!**

**Risk 3: Fat Tails (Non-Normal Returns)**

**Problem:** Market returns have fatter tails than normal distribution.

**Normal distribution assumption:**
- -3σ daily move: 0.1% probability (~1 in 1000 days = once every 4 years)
- With 27% annual vol → daily vol = 27% / √252 = 1.7%
- -3σ day: -5.1% loss

**Levered portfolio:**
- -5.1% × 1.5 = **-7.65% daily loss**

**Reality - October 19, 1987 (Black Monday):**
- S&P 500: -20.5% in one day
- Levered 1.5x: **-30.75% in one day**
- 1.5x levered investor lost $307,500 on $1M investment IN ONE DAY
- Many triggered margin calls, forced liquidations

**2020 March 16 (COVID crash):**
- S&P 500: -12% in one day
- Levered 1.5x: **-18% in one day**
- $180,000 loss on $1M

**Fat tails occur 10-50x more frequently than normal distribution predicts.**

**Risk 4: Sequence Risk**

**Problem:** Timing of returns matters with leverage.

**Example: Two investors, same 10-year average market return (8%)**

**Investor A (unlucky):**
Year 1: -30% market return
- Start: $1M
- Market: $1M → $700K (-30%)
- Levered 1.5x: -30% × 1.5 = -45%
- End: $550K
- **Down 45%, need +82% to recover!**

**Investor B (lucky):**
Year 1: +30% market return
- Start: $1M
- Levered 1.5x: +30% × 1.5 = +45%
- End: $1.45M

Same market performance, but **timing determines if you survive leverage**.

**Risk 5: Correlation Spikes**

**Problem:** In crises, correlations increase, reducing diversification.

**Normal market:**
- Stock/bond correlation: 0.2
- Stock/REIT correlation: 0.6
- Diversification works

**Crisis (March 2020):**
- Stock/bond correlation: 0.7 (both fell)
- Stock/REIT correlation: 0.95 (everything fell together)
- **Diversification failed**

**Levered diversified portfolio:**
- Expected: Leverage + diversification = manageable risk
- Reality: Leverage + correlation spike = catastrophic loss
- S&P 500: -34%
- 1.5x levered: -51% (before margin call)

**When Leveraged CML Strategies Are Appropriate:**

✓ **Sophisticated Institutional Investors:**
- Professional risk management
- Access to cheap leverage (futures, swaps)
- Can weather drawdowns without forced selling
- Examples: AQR, Bridgewater use controlled leverage

✓ **Long Time Horizons:**
- 20+ years until retirement
- Can ride out multi-year drawdowns
- Sequence risk less severe over long periods
- Young investors in accumulation phase

✓ **Moderate Leverage (1.2-1.3x):**
- Lower risk than 1.5-2x
- Still meaningful return enhancement
- Less prone to margin calls
- Volatility drag manageable

✓ **Professional Leverage Implementation:**
- Futures/swaps (not margin)
- Daily monitoring
- Automatic deleveraging rules
- Hedging strategies (put options)

✓ **Low-Volatility Environments:**
- VIX < 15
- No recent market crises
- Stable macro environment
- (But this can change quickly!)

**When Leveraged CML Strategies Are Inappropriate:**

✗ **Retail Investors Without Sophistication:**
- Don't understand margin calls
- Can't monitor daily
- Emotional decision-making during drawdowns
- Most lose money attempting leverage

✗ **Short Time Horizons:**
- Nearing retirement (< 10 years)
- Need liquidity soon (< 5 years)
- Can't afford sequence risk
- Forced selling risk unacceptable

✗ **High Leverage (2x+):**
- Extreme volatility drag
- High margin call risk
- Small market moves = large losses
- Historical evidence of poor outcomes

✗ **High-Volatility Environments:**
- VIX > 25
- During/after crisis
- Correlations elevated
- Tail risks heightened

✗ **Inadequate Risk Management:**
- No stop-loss rules
- No deleverage triggers
- No hedging
- No daily monitoring

**Practical Recommendations:**

**For aggressive young investors:**
- **Maximum 1.3x leverage** (not 1.5x)
- Use futures (not margin)
- Set **automatic deleveraging at -20%** drawdown
- Consider **hedging** (buy 10% out-of-money puts)
- Only use **25-50% of portfolio** (not 100%)

**For most retail investors:**
- **AVOID leverage entirely**
- Get higher risk exposure through 100% stock allocation (not leverage)
- Use bond allocation to control risk (not deleveraging)
- Simplicity beats sophistication for most

**For institutions:**
- **1.2-1.5x** leverage acceptable with proper risk management
- Futures-based implementation
- Daily VaR monitoring
- Stress testing
- Automatic deleveraging triggers

**Historical Performance:**

**1.5x Levered S&P 500 (1928-2023):**
- Geometric return: ~10.5% vs 9.8% unlevered (+0.7%)
- Volatility: 27% vs 18% (+50%)
- Max drawdown: -85% vs -56% (1929-1932)
- **Marginal benefit NOT worth the risk for most investors**

**Conclusion:**

Leverage in CML framework is **theoretically attractive** (higher returns on CML) but **practically dangerous** (margin calls, volatility drag, fat tails, sequence risk). 

For most investors, achieving 15% returns through 1.5x leverage (150% market) is **inferior** to achieving similar risk-adjusted returns through:
1. 100% equity allocation (no leverage)
2. Factor tilts (value, momentum, quality)
3. Alternative assets (private equity, real estate)
4. Options strategies (covered calls, cash-secured puts)

**Leverage should only be used by sophisticated investors with professional risk management, long time horizons, and ability to withstand 50%+ drawdowns without forced selling.**`,
                    keyPoints: [
                        '150% market allocation required for 15% return target; results in 27% volatility and 1.5x leverage ratio',
                        'Need to borrow $500K per $1M invested; borrowing at 5% vs 3% reduces return from 15% to 14% (1% drag)',
                        'Margin calls triggered at ~55% market drawdown with 1.5x leverage; forced liquidation locks in losses',
                        'Volatility drag increases with leverage: 27% vol causes 3.65% drag vs 1.62% for unlevered, reducing geometric returns',
                        'Fat tail events (Black Monday, 2008, 2020) occur 10-50x more often than normal distribution predicts, devastating leveraged portfolios',
                        'Appropriate for: institutions with risk management, long horizons (20+ years), moderate leverage (1.2-1.3x)',
                        'Inappropriate for: retail investors, short horizons, high leverage (2x+), high volatility environments',
                        'Historical 1.5x leverage adds only 0.7% annual return vs 50% more volatility and 85% max drawdown - poor risk/reward for most'
                    ]
}
  ]
};

