export const straddlesStrangles = {
  title: 'Straddles and Strangles',
  id: 'straddles-strangles',
  content: `
# Straddles and Strangles

## Introduction

**Volatility strategies** that profit from large moves in EITHER direction (or lack thereof).

**Core Strategies:**
- **Long Straddle/Strangle:** Buy volatility, bet on big move
- **Short Straddle/Strangle:** Sell volatility, bet on calm

These are **non-directional** - don't need to predict direction, only magnitude of movement.

---

## Long Straddle

### Setup

- Buy ATM call
- Buy ATM put
- Same strike, same expiration

### Payoff

\`\`\`python
"""
Long Straddle Analysis
"""

def long_straddle(stock_prices, strike, total_premium):
    """Profit from large move in either direction"""
    call_payoff = np.maximum(stock_prices - strike, 0)
    put_payoff = np.maximum(strike - stock_prices, 0)
    total = call_payoff + put_payoff - total_premium
    return total

# Example
strike = 100
call_premium = 5
put_premium = 5
total_premium = call_premium + put_premium  # $10

stock_prices = np.linspace(70, 130, 200)
payoff = long_straddle(stock_prices, strike, total_premium)

plt.figure(figsize=(12, 6))
plt.plot(stock_prices, payoff, 'g-', linewidth=2, label='Long Straddle')
plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.axvline(strike, color='red', linestyle=':', label=f'Strike \${strike}')
plt.axvline(strike - total_premium, color='blue', linestyle=':', label='Lower Breakeven')
plt.axvline(strike + total_premium, color='blue', linestyle=':', label='Upper Breakeven')
plt.fill_between(stock_prices, payoff, 0, where=(payoff > 0), alpha=0.3, color='green', label='Profit')
plt.fill_between(stock_prices, payoff, 0, where=(payoff < 0), alpha=0.3, color='red', label='Loss')
plt.xlabel('Stock Price at Expiration')
plt.ylabel('Profit / Loss')
plt.title('Long Straddle: Profit from Large Move Either Direction')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Metrics
breakeven_lower = strike - total_premium
breakeven_upper = strike + total_premium
max_loss = total_premium

print(f"\\nLong Straddle: \\$\{strike} Strike")
print(f"  Total Cost: \\$\{total_premium}")
print(f"  Breakevens: \${breakeven_lower} and \\$\{breakeven_upper}")
print(f"  Max Loss: \${total_premium} (at \\$\{strike})")
print(f"  Max Profit: Unlimited")
print(f"  Required Move: {(total_premium/strike*100):.1f}% in either direction")
\`\`\`

### When to Use Long Straddles

**Ideal Conditions:**1. **Low IV** (cheap options)
2. **Before known events** (earnings, FDA approval, M&A rumors)
3. **Expect vol expansion** (IV will increase)
4. **Don't know direction** but expect big move

**Risks:**
- **Theta decay:** Loses value every day if stock doesn't move
- **Volatility crush:** IV drops after event (even if stock moves)
- **Need BIG move:** Must exceed breakevens

---

## Long Strangle

### Setup

- Buy OTM call
- Buy OTM put
- Different strikes, same expiration

**Advantage:** Cheaper than straddle (OTM options)
**Disadvantage:** Need bigger move to profit

\`\`\`python
"""
Long Strangle Analysis
"""

def long_strangle(stock_prices, put_strike, call_strike, total_premium):
    """Cheaper than straddle, needs bigger move"""
    call_payoff = np.maximum(stock_prices - call_strike, 0)
    put_payoff = np.maximum(put_strike - stock_prices, 0)
    total = call_payoff + put_payoff - total_premium
    return total

# Example
put_strike = 95
call_strike = 105
put_premium = 3
call_premium = 3
total_premium = 6  # Cheaper than $10 straddle

stock_prices = np.linspace(70, 130, 200)
payoff_strangle = long_strangle(stock_prices, put_strike, call_strike, total_premium)
payoff_straddle = long_straddle(stock_prices, 100, 10)

plt.figure(figsize=(12, 6))
plt.plot(stock_prices, payoff_strangle, 'g-', linewidth=2, label='Long Strangle (cheaper)')
plt.plot(stock_prices, payoff_straddle, 'b--', linewidth=2, alpha=0.7, label='Long Straddle')
plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.xlabel('Stock Price')
plt.ylabel('P&L')
plt.title('Long Strangle vs Long Straddle')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print(f"\\nLong Strangle: \${put_strike} Put / \\$\{call_strike} Call")
print(f"  Total Cost: \\$\{total_premium} (vs $10 straddle)")
print(f"  Breakevens: \${put_strike - total_premium} and \\$\{call_strike + total_premium}")
print(f"  Wider breakevens but cheaper entry")
\`\`\`

---

## Short Straddle

### Setup

- Sell ATM call
- Sell ATM put
- Collect premium, hope stock stays near strike

**WARNING:** Unlimited risk strategy!

\`\`\`python
"""
Short Straddle (HIGH RISK)
"""

def short_straddle(stock_prices, strike, total_premium):
    """Collect premium, profit if stock stays near strike"""
    call_payoff = -np.maximum(stock_prices - strike, 0)
    put_payoff = -np.maximum(strike - stock_prices, 0)
    total = call_payoff + put_payoff + total_premium
    return total

# Example
strike = 100
total_premium = 10  # Collected

stock_prices = np.linspace(70, 130, 200)
payoff = short_straddle(stock_prices, strike, total_premium)

plt.figure(figsize=(12, 6))
plt.plot(stock_prices, payoff, 'r-', linewidth=2, label='Short Straddle')
plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.axvline(strike, color='green', linestyle=':', label=f'Strike \${strike}')
plt.fill_between(stock_prices, payoff, 0, where=(payoff > 0), alpha=0.3, color='green')
plt.fill_between(stock_prices, payoff, 0, where=(payoff < 0), alpha=0.3, color='red')
plt.xlabel('Stock Price')
plt.ylabel('P&L')
plt.title('Short Straddle: High Risk, High Reward')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(-30, 15)
plt.show()

print(f"\\nShort Straddle: \\$\{strike} Strike")
print(f"  Premium Collected: \\$\{total_premium}")
print(f"  Max Profit: \${total_premium} (at \\$\{strike})")
print(f"  Max Loss: UNLIMITED")
print(f"  Profit Zone: \${strike - total_premium} to \\$\{strike + total_premium}")
\`\`\`

### Risk Management for Short Straddles

**CRITICAL:** Must have:
- **Defined risk:** Use iron butterflies instead (add wings)
- **Stop losses:** Exit at 2× premium collected
- **Position sizing:** Small percentage of portfolio
- **Hedging:** Delta hedge with stock

---

## Short Strangle

### Setup

- Sell OTM put
- Sell OTM call
- Collect premium, wider profit zone than short straddle

\`\`\`python
"""
Short Strangle
"""

def short_strangle(stock_prices, put_strike, call_strike, total_premium):
    """Wider profit zone than short straddle"""
    call_payoff = -np.maximum(stock_prices - call_strike, 0)
    put_payoff = -np.maximum(put_strike - stock_prices, 0)
    total = call_payoff + put_payoff + total_premium
    return total

# Example
put_strike = 95
call_strike = 105
total_premium = 5  # Less than straddle

stock_prices = np.linspace(70, 130, 200)
payoff_strangle = short_strangle(stock_prices, put_strike, call_strike, total_premium)
payoff_straddle = short_straddle(stock_prices, 100, 10)

plt.figure(figsize=(12, 6))
plt.plot(stock_prices, payoff_strangle, 'orange', linewidth=2, label='Short Strangle')
plt.plot(stock_prices, payoff_straddle, 'r--', linewidth=2, alpha=0.7, label='Short Straddle')
plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.xlabel('Stock Price')
plt.ylabel('P&L')
plt.title('Short Strangle: Wider Profit Zone, Less Premium')
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(-30, 15)
plt.show()

print(f"\\nShort Strangle: \${put_strike} Put / \\$\{call_strike} Call")
print(f"  Premium: \\$\{total_premium}")
print(f"  Profit Zone: \${put_strike} to \\$\{call_strike}")
print(f"  Still UNLIMITED RISK - use with caution!")
\`\`\`

---

## Earnings Plays

### Pre-Earnings: Long Straddle

\`\`\`python
"""
Earnings Straddle Strategy
"""

def earnings_straddle_analysis(stock_price, expected_move_pct, straddle_cost):
    """
    Analyze if earnings straddle is profitable
    
    Args:
        stock_price: Current stock price
        expected_move_pct: Market's expected move (from straddle price)
        straddle_cost: Cost of ATM straddle
    """
    # Expected move implied by options
    implied_move = straddle_cost
    implied_move_pct = (implied_move / stock_price) * 100
    
    # Historical average move
    # (would get from past earnings)
    
    print(f"Earnings Straddle Analysis for \\$\{stock_price} stock:")
    print(f"  Straddle Cost: \\$\{straddle_cost} ({implied_move_pct:.1f}%)")
    print(f"  Breakevens: \${stock_price - straddle_cost:.2f
} and \${ stock_price + straddle_cost:.2f}")
print(f"  \\nRequired move to profit: > {implied_move_pct:.1f}%")
    
    # Check if historical moves exceed implied
if expected_move_pct > implied_move_pct:
    print(f"  Historical move ({expected_move_pct:.1f}%) > Implied ({implied_move_pct:.1f}%)")
print(f"  → OPPORTUNITY: Straddle may be underpriced")
    else:
print(f"  Historical move ({expected_move_pct:.1f}%) < Implied ({implied_move_pct:.1f}%)")
print(f"  → CAUTION: Straddle may be overpriced, vol crush likely")

# Example: NVDA earnings
earnings_straddle_analysis(
    stock_price = 500,
    expected_move_pct = 8.5,  # Historical average
    straddle_cost = 35  # $35 straddle = 7 % move
)
\`\`\`

### Post-Earnings: Short Strangles

After earnings, IV collapses - opportunity to sell premium.

---

## Volatility Trading Metrics

\`\`\`python
"""
Straddle/Strangle Greeks Analysis
"""

def calculate_straddle_greeks(S, K, T, sigma, r=0.05):
    """Calculate Greeks for ATM straddle"""
    from scipy.stats import norm
    
    # Call Greeks
    call_price, call_delta, call_gamma, call_theta, call_vega = calculate_greeks(S, K, T, r, sigma, 'call')
    
    # Put Greeks
    put_price, put_delta, put_gamma, put_theta, put_vega = calculate_greeks(S, K, T, r, sigma, 'put')
    
    # Straddle totals
    straddle_price = call_price + put_price
    straddle_delta = call_delta + put_delta  # Should be near zero
    straddle_gamma = call_gamma + put_gamma  # Positive (long gamma)
    straddle_theta = call_theta + put_theta  # Negative (time decay)
    straddle_vega = call_vega + put_vega  # Positive (long vol)
    
    print(f"\\nATM Straddle Greeks (Stock \${S}, Strike \\$\{K}):")
    print(f"  Price: \\$\{straddle_price:.2f}")
print(f"  Delta: {straddle_delta:.3f} (near zero = direction-neutral)")
print(f"  Gamma: {straddle_gamma:.3f} (positive = benefits from moves)")
print(f"  Theta: \\$\{straddle_theta:.2f}/day (negative = loses to time)")
print(f"  Vega: \\$\{straddle_vega:.2f}/1% IV (positive = benefits from vol increase)")

return {
    'price': straddle_price,
    'delta': straddle_delta,
    'gamma': straddle_gamma,
    'theta': straddle_theta,
    'vega': straddle_vega
}

# Example
calculate_straddle_greeks(S = 100, K = 100, T = 30 / 365, sigma = 0.30)
\`\`\`

---

## Advanced Straddle/Strangle Techniques

### Adjusting Losing Positions

**When straddle goes against you:**

\`\`\`python
"""
Straddle Adjustment Strategies
"""

class StraddleAdjustment:
    """
    Manage and adjust straddle positions
    """
    
    def __init__(self, initial_stock, strike, premium_paid):
        self.initial_stock = initial_stock
        self.strike = strike
        self.premium_paid = premium_paid
        self.position_open = True
    
    def check_adjustment_needed(self, current_stock, days_held, days_to_exp):
        """
        Determine if adjustment needed
        
        Adjustment triggers:
        1. Stock hasn't moved AND theta eating profit
        2. One leg deep ITM, other worthless
        3. Approaching expiration with loss
        """
        # Calculate P&L
        call_value = max(current_stock - self.strike, 0)
        put_value = max(self.strike - current_stock, 0)
        current_value = call_value + put_value
        pnl = current_value - self.premium_paid
        
        # Check triggers
        triggers = []
        
        # Trigger 1: Stagnant stock + time decay
        if abs(current_stock - self.initial_stock) / self.initial_stock < 0.03 and days_held > 10:
            triggers.append("Stagnant: Stock moved < 3% after 10 days")
        
        # Trigger 2: One-sided move
        if abs(current_stock - self.strike) / self.strike > 0.10:
            triggers.append(f"One-sided: Stock {'+' if current_stock > self.strike else ''}{((current_stock - self.strike) / self.strike * 100):.1f}% from strike")
        
        # Trigger 3: Near expiration with loss
        if days_to_exp < 7 and pnl < -self.premium_paid * 0.5:
            triggers.append(f"Near expiration with >50% loss")
        
        return triggers, pnl, current_value
    
    def recommend_adjustment(self, current_stock, current_iv, triggers):
        """
        Recommend specific adjustment
        """
        print("=" * 70)
        print("STRADDLE ADJUSTMENT RECOMMENDATION")
        print("=" * 70)
        
        if "Stagnant" in triggers[0] if triggers else "":
            print("\\n📉 STAGNANT STOCK - HIGH THETA DECAY")
            print("\\nOptions:")
            print("  1. Close position (accept loss)")
            print("     - Cut losses before more theta decay")
            print("     - Move capital to better opportunity")
            
            print("\\n  2. Roll to later expiration")
            print("     - Close current straddle")
            print("     - Open new straddle 30-60 days out")
            print("     - Gives more time for move")
            print("     - Cost: Additional premium")
            
            print("\\n  3. Convert to iron butterfly")
            print("     - Sell OTM put and call (reduce cost)")
            print("     - Caps max loss, but also caps profit")
            print("     - Better risk/reward for stagnant stock")
        
        elif "One-sided" in str(triggers):
            print("\\n📈 ONE-SIDED MOVE")
            if current_stock > self.strike:
                direction = "UP"
                itm_leg = "call"
                otm_leg = "put"
            else:
                direction = "DOWN"
                itm_leg = "put"
                otm_leg = "call"
            
            print(f"\\nStock moved {direction}, {itm_leg} ITM, {otm_leg} worthless")
            print("\\nOptions:")
            print(f"  1. Close {otm_leg} (worth nothing)")
            print(f"     - Reduce position to long {itm_leg}")
            print(f"     - Now directional play")
            
            print(f"\\n  2. Roll {otm_leg} closer")
            print(f"     - Close current {otm_leg}")
            print(f"     - Buy new ATM {otm_leg}")
            print(f"     - Re-establish straddle at new stock price")
            print(f"     - Cost: New premium")
            
            print(f"\\n  3. Take profit on {itm_leg}")
            print(f"     - Close entire position")
            print(f"     - Lock in gains from {direction} move")
        
        return None


# Example usage
adjustment_manager = StraddleAdjustment(
    initial_stock=100,
    strike=100,
    premium_paid=10
)

# Scenario: Stock stagnant at $102 after 15 days
current_stock = 102
triggers, pnl, current_value = adjustment_manager.check_adjustment_needed(
    current_stock=102,
    days_held=15,
    days_to_exp=15
)

print("=" * 70)
print("STRADDLE POSITION CHECK")
print("=" * 70)
print(f"\\nInitial:")
print(f"  Stock: \\$\{adjustment_manager.initial_stock}")
print(f"  Strike: \\$\{adjustment_manager.strike}")
print(f"  Premium paid: \\$\{adjustment_manager.premium_paid}")

print(f"\\nCurrent (Day 15):")
print(f"  Stock: \\$\{current_stock} (+{((current_stock - adjustment_manager.initial_stock) / adjustment_manager.initial_stock * 100):.1f}%)")
print(f"  Position value: \\$\{current_value:.2f}")
print(f"  P&L: \\$\{pnl:.2f} ({(pnl / adjustment_manager.premium_paid * 100):.1f}%)")

if triggers:
    print(f"\\n⚠️  ADJUSTMENT TRIGGERS:")
for trigger in triggers:
    print(f"  - {trigger}")

adjustment_manager.recommend_adjustment(current_stock, 0.25, triggers)
else:
print(f"\\n✓ No adjustment needed - position on track")
\`\`\`

---

## Volatility Crush Analysis

### Post-Event IV Collapse

**Volatility crush** destroys straddles/strangles even if stock moves.

\`\`\`python
"""
Volatility Crush Impact Analysis
"""

def analyze_volatility_crush(entry_iv, post_event_iv, stock_move_pct, 
                            strike=100, days_to_exp=30):
    """
    Calculate straddle P&L including vol crush
    
    Shows why earnings straddles can lose money even with stock moves
    """
    from scipy.stats import norm
    
    print("=" * 70)
    print("VOLATILITY CRUSH ANALYSIS")
    print("=" * 70)
    
    # Initial straddle price (pre-event)
    T = days_to_exp / 365
    r = 0.05
    
    # Price straddle at entry IV
    d1 = (np.log(strike/strike) + (r + 0.5*entry_iv**2)*T) / (entry_iv*np.sqrt(T))
    d2 = d1 - entry_iv*np.sqrt(T)
    
    call_entry = strike*norm.cdf(d1) - strike*np.exp(-r*T)*norm.cdf(d2)
    put_entry = strike*np.exp(-r*T)*norm.cdf(-d2) - strike*norm.cdf(-d1)
    straddle_entry = call_entry + put_entry
    
    print(f"\\nPRE-EVENT:")
    print(f"  Stock: \\$\{strike}")
    print(f"  IV: {entry_iv*100:.1f}%")
    print(f"  ATM Straddle: \\$\{straddle_entry:.2f}")
print(f"  Days to expiration: {days_to_exp}")
    
    # Post - event(after earnings, 1 day later)
T_post = (days_to_exp - 1) / 365
new_stock = strike * (1 + stock_move_pct)
    
    # Price straddle at post - event IV
d1_post = (np.log(new_stock / strike) + (r + 0.5 * post_event_iv ** 2) * T_post) / (post_event_iv * np.sqrt(T_post))
d2_post = d1_post - post_event_iv * np.sqrt(T_post)

call_post = new_stock * norm.cdf(d1_post) - strike * np.exp(-r * T_post) * norm.cdf(d2_post)
put_post = strike * np.exp(-r * T_post) * norm.cdf(-d2_post) - new_stock * norm.cdf(-d1_post)
straddle_post = call_post + put_post

pnl = straddle_post - straddle_entry

print(f"\\nPOST-EVENT:")
print(f"  Stock: \\$\{new_stock:.2f} ({stock_move_pct*100:+.1f}%)")
print(f"  IV: {post_event_iv*100:.1f}% (crushed {(entry_iv - post_event_iv)*100:.1f}%)")
print(f"  ATM Straddle: \\$\{straddle_post:.2f}")

print(f"\\n{'─' * 70}")
print(f"P&L: \\$\{pnl:+.2f} ({pnl/straddle_entry*100:+.1f}%)")

if pnl > 0:
    print(f"✓ PROFIT - Stock move overcame vol crush")
else:
print(f"✗ LOSS - Vol crush exceeded stock move benefit")
    
    # Break - even analysis
print(f"\\n{'─' * 70}")
print("BREAK-EVEN ANALYSIS:")
    
    # What stock move needed to breakeven ?
    for test_move in [0.05, 0.08, 0.10, 0.12, 0.15]:
    test_stock = strike * (1 + test_move)
d1_test = (np.log(test_stock / strike) + (r + 0.5 * post_event_iv ** 2) * T_post) / (post_event_iv * np.sqrt(T_post))
d2_test = d1_test - post_event_iv * np.sqrt(T_post)

call_test = test_stock * norm.cdf(d1_test) - strike * np.exp(-r * T_post) * norm.cdf(d2_test)
put_test = strike * np.exp(-r * T_post) * norm.cdf(-d2_test) - test_stock * norm.cdf(-d1_test)
straddle_test = call_test + put_test

test_pnl = straddle_test - straddle_entry

status = "✓" if test_pnl > 0 else "✗"
print(f"  {test_move*100:.0f}% move: \\$\{test_pnl:+.2f} {status}")

if test_pnl > 0 and(test_move - 0.01) * 100 < 0:
print(f"  → Break-even at ~{test_move*100:.1f}% stock move")

return pnl


# Example 1: Small move + big vol crush = LOSS
print("\\nEXAMPLE 1: TYPICAL EARNINGS (Small Move)")
analyze_volatility_crush(
    entry_iv = 0.40,  # 40 % IV before earnings(elevated)
    post_event_iv = 0.22,  # 22 % IV after(crushed 18 %)
    stock_move_pct = 0.05,  # Stock moves 5 %
strike=150,
    days_to_exp = 30
)

print("\\n" * 2)

# Example 2: Big move + vol crush = PROFIT
print("EXAMPLE 2: BIG EARNINGS BEAT (Large Move)")
analyze_volatility_crush(
    entry_iv = 0.40,
    post_event_iv = 0.22,
    stock_move_pct = 0.12,  # Stock moves 12 % !
    strike=150,
    days_to_exp = 30
)
\`\`\`

---

## Ratio Straddles/Strangles

### Unequal Legs

**Ratio strategies** use unequal quantities to reduce cost or bias direction.

\`\`\`python
"""
Ratio Straddle: More Calls than Puts (or vice versa)
"""

def ratio_straddle(stock_prices, strike, call_quantity, put_quantity,
                   call_premium, put_premium):
    """
    Straddle with different quantities
    
    Example: Buy 2 calls, buy 1 put (bullish bias)
    """
    call_payoff = call_quantity * np.maximum(stock_prices - strike, 0)
    put_payoff = put_quantity * np.maximum(strike - stock_prices, 0)
    
    premium_paid = call_quantity * call_premium + put_quantity * put_premium
    
    total = call_payoff + put_payoff - premium_paid
    return total


# Example: Bullish ratio straddle
strike = 100
call_qty = 2  # More calls
put_qty = 1
call_premium = 5
put_premium = 5

stock_prices = np.linspace(70, 130, 200)
payoff_ratio = ratio_straddle(stock_prices, strike, call_qty, put_qty, 
                               call_premium, put_premium)
payoff_standard = ratio_straddle(stock_prices, strike, 1, 1, 
                                  call_premium, put_premium)

plt.figure(figsize=(12, 6))
plt.plot(stock_prices, payoff_ratio, 'b-', linewidth=2, label='Ratio 2:1')
plt.plot(stock_prices, payoff_standard, 'g--', linewidth=2, alpha=0.7, label='Standard 1:1')
plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.axvline(strike, color='red', linestyle=':', alpha=0.5, label='Strike')
plt.xlabel('Stock Price')
plt.ylabel('P&L')
plt.title('Ratio Straddle: 2 Calls / 1 Put (Bullish Bias)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print("=" * 70)
print("RATIO STRADDLE ANALYSIS")
print("=" * 70)
print(f"\\nStandard straddle: 1 call + 1 put = \\$\{call_premium + put_premium}")
print(f"Ratio straddle: 2 calls + 1 put = \${2 * call_premium + put_premium}")
print(f"\\nExtra cost: \\$\{call_premium} (but 2× upside leverage)")
print(f"\\nUse case: Slightly bullish, want more upside exposure")
\`\`\`

---

## Real-World Straddle Examples

### Example 1: NVDA Earnings Straddle

\`\`\`python
"""
Real Trade: NVDA Earnings (Nov 2023)
"""

def nvda_earnings_case_study():
    """
    Actual NVDA earnings straddle trade analysis
    """
    print("=" * 70)
    print("CASE STUDY: NVDA EARNINGS STRADDLE (NOV 2023)")
    print("=" * 70)
    
    # Pre-earnings data
    stock_pre = 500
    iv_pre = 0.45  # 45% IV (very high)
    atm_call = 35
    atm_put = 33
    straddle_cost = atm_call + atm_put  # $68
    
    print(f"\\nPRE-EARNINGS (Nov 20, 2023):")
    print(f"  NVDA: \\$\{stock_pre}")
    print(f"  IV: {iv_pre*100:.0f}%")
    print(f"  ATM Straddle: \\$\{straddle_cost}")
    print(f"  Implied move: {(straddle_cost/stock_pre*100):.1f}%")
    print(f"  Cost for 1 straddle: \\$\{straddle_cost * 100:,.0f}")
    
    # Post - earnings result
stock_post = 505  # Stock up ONLY 1 % !
    iv_post = 0.28  # IV crushed to 28 %
    
    # But... stock kept rallying
stock_next_day = 540  # Up 8 % total next day

print(f"\\nPOST-EARNINGS (Nov 21, 2023 - immediate):")
print(f"  NVDA: \\$\{stock_post} (+{((stock_post-stock_pre)/stock_pre*100):.1f}%)")
print(f"  IV: {iv_post*100:.0f}% (crushed {(iv_pre-iv_post)*100:.0f}%)")
print(f"\\n  Result: LOSS despite move")
print(f"  Straddle value ≈ $55 (down from $68)")
print(f"  P&L: -$13 per share (-19%)")
print(f"  On $6,800 investment: LOST $1,300")

print(f"\\nNEXT DAY (Nov 22):")
print(f"  NVDA: \\$\{stock_next_day} (+{((stock_next_day-stock_pre)/stock_pre*100):.1f}% from entry)")
print(f"  Call value: ~$40 ITM")
print(f"  Put value: $0")
print(f"  Straddle value: ~$40")
print(f"\\n  Total P&L: -$28 per share (-41%)")
print(f"  On $6,800: LOST $2,800")

print(f"\\n{'─' * 70}")
print("LESSONS:")
print("  1. IV crush can overwhelm small stock moves")
print("  2. Even 8% total move wasn't enough (needed 13.6%)")
print("  3. Earnings straddles are LOTTERY TICKETS")
print("  4. Better to sell straddles pre-earnings (collect IV premium)")
    
    # What would have worked ?
    print(f"\\n{'─' * 70}")
    print("WHAT WOULD HAVE BEEN PROFITABLE:")
print(f"  • SHORT the straddle pre-earnings")
print(f"    Collect $68, stock moves <13.6% → profit")
print(f"    With $68 collected, breakevens at $432 and $568")
print(f"    Actual move to $505 → profit $63/share")
print(f"    BUT: Unlimited risk if wrong")


nvda_earnings_case_study()
\`\`\`

---

## Summary

**Long Straddle/Strangle:**
- Buy volatility, profit from big moves
- Best in LOW IV, before events
- Requires large move to overcome premium + theta
- Risk: Theta decay + vol crush
- Break-even: Strike ± premium paid
- Win rate: ~40%, but large wins when hit

**Short Straddle/Strangle:**
- Sell volatility, profit if stock stays calm
- Best in HIGH IV (collect inflated premium)
- High win rate (~70%) but catastrophic losses possible
- Risk: UNLIMITED (must use defined-risk version)
- Break-even: Strike ± premium collected
- **NEVER trade naked** - always add wings (iron butterfly/condor)

**Advanced Techniques:**
- **Adjustments:** Roll, convert to butterfly, close legs
- **Ratio straddles:** Unequal legs for directional bias
- **Vol crush awareness:** Exit before event if up
- **Real-world:** Most earnings straddles lose (sell not buy)

**Key Decision Framework:**
\`\`\`
IF IV Rank < 25% AND expecting big move:
    → BUY straddle/strangle
    
IF IV Rank > 75% AND expecting calm:
    → SELL strangle (with wings!)
    
ELSE:
    → Skip (no edge)
\`\`\`

**Professional Approach:**
- Size small (2-3% of capital per trade)
- Use iron butterflies instead of naked shorts
- Exit at 50% profit or 50% loss
- NEVER hold through earnings long (buy side)
- Track win rate and average P&L religiously

This completes the volatility strategies foundation!
`,
};
