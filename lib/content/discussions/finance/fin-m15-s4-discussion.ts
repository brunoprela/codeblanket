export default {
  id: 'fin-m15-s4-discussion',
  title: 'Stress Testing and Scenario Analysis - Discussion Questions',
  questions: [
    {
      question:
        'Compare historical stress testing (replaying past crises) versus hypothetical stress testing (creating "what if" scenarios). What are the advantages and limitations of each approach, and how should a comprehensive stress testing framework combine both?',
      answer: `Both approaches are essential for a complete stress testing framework. Understanding when to use each is critical for effective risk management:

**Historical Stress Testing**

**Definition:** Apply shocks from actual historical crises to current portfolio.

**Method:**
\`\`\`python
def historical_stress_test(portfolio, crisis_scenario):
    """
    Apply historical crisis to current portfolio
    
    Args:
        portfolio: Current portfolio positions
        crisis_scenario: Historical market moves
        
    Returns:
        Stress loss estimate
    """
    # Example: 2008 Financial Crisis
    crisis_2008 = {
        'equities': {
            'S&P 500': -0.38,          # Down 38%
            'NASDAQ': -0.42,            # Down 42%
            'Russell 2000': -0.34,      # Down 34%
            'MSCI World': -0.42,        # Down 42%
        },
        'fixed_income': {
            'IG Corporate': -0.08,      # Down 8%
            'HY Corporate': -0.26,      # Down 26%
            'MBS': -0.15,               # Down 15%
            'EM Bonds': -0.18,          # Down 18%
        },
        'credit_spreads': {
            'IG': 300,                  # +300bp
            'HY': 1000,                 # +1000bp
        },
        'rates': {
            '10Y Treasury': -0.015,     # Down 150bp (flight to quality)
        },
        'fx': {
            'USD': 0.15,                # Dollar up 15%
            'JPY': 0.20,                # Yen up 20%
        },
        'commodities': {
            'Oil': -0.54,               # Down 54%
            'Gold': 0.06,               # Up 6%
        },
        'volatility': {
            'VIX': 4.0,                 # 4x increase (to 80)
        }
    }
    
    stressed_portfolio_value = 0
    stress_breakdown = {}
    
    for position in portfolio:
        asset_class = position.asset_class
        asset = position.asset
        current_value = position.market_value
        
        # Get historical shock
        if asset_class == 'equity':
            shock = crisis_2008['equities'].get(asset, -0.40)  # Default -40%
        elif asset_class == 'fixed_income':
            shock = crisis_2008['fixed_income'].get(asset, -0.10)
        # ... other asset classes
        
        # Apply shock
        stressed_value = current_value * (1 + shock)
        stressed_portfolio_value += stressed_value
        
        # Track breakdown
        loss = current_value - stressed_value
        stress_breakdown[position.id] = {
            'current': current_value,
            'stressed': stressed_value,
            'loss': loss,
            'shock_applied': shock
        }
    
    total_loss = portfolio.total_value - stressed_portfolio_value
    
    return {
        'scenario': '2008 Financial Crisis',
        'total_loss': total_loss,
        'loss_pct': total_loss / portfolio.total_value,
        'breakdown': stress_breakdown,
        'survived': total_loss < portfolio.capital_buffer
    }

# Example usage
portfolio = load_current_portfolio()
result = historical_stress_test(portfolio, '2008_crisis')

print(f"2008 Crisis Stress Test:")
print(f"  Portfolio Value: \${portfolio.total_value / 1e6: .1f
        }M")
print(f"  Stress Loss: \${result['total_loss']/1e6:.1f}M ({result['loss_pct']*100:.1f}%)")
print(f"  Survived: {'✓' if result['survived'] else '✗ FAILURE'}")
        \`\`\`

**Advantages of Historical Stress Testing:**

✅ **Actually Happened**
\`\`\`python
# Not hypothetical - these events occurred
# - Market can move this way (proven)
# - Correlations observed (real data)
# - Liquidity effects known (actual)

# Example: 2008 crash
# Skeptic: "Markets won't fall 40%"
# Response: "They did in 2008"
# → Hard to argue with history
\`\`\`

✅ **Regulatory Acceptance**
\`\`\`python
# Regulators require historical scenarios
# CCAR (Comprehensive Capital Analysis and Review):
# - Must test 2008-style recession
# - Must test specific historical scenarios

# European Banking Authority:
# - Adverse scenario based on historical crises
# - Sovereign debt crisis scenarios

# Easy to defend to regulators:
# "We're prepared for events like 2008"
\`\`\`

✅ **Board/Management Understanding**
\`\`\`python
# Board: "How bad could it get?"
# CRO: "In 2008, portfolio would have lost $50M"
# Board: "We remember 2008. That's concrete."

# vs

# CRO: "In a 4-sigma correlation breakdown scenario..."
# Board: "What?"

# Historical scenarios are intuitive
\`\`\`

✅ **Complete Market Picture**
\`\`\`python
# Historical crises include:
# - Primary shocks (market crashes)
# - Secondary effects (liquidity freeze)
# - Behavioral responses (panic selling)
# - Policy reactions (rate cuts, QE)
# - Time dynamics (how long to recover)

# Captures full ecosystem response
# Not just isolated shocks
\`\`\`

✅ **Correlation Structure**
\`\`\`python
# Historical data shows real correlations in stress

# 2008: Everything crashed together
equities = -0.40
credit = -0.26
real_estate = -0.35
commodities = -0.54

# Correlation in crisis: 0.95+
# Can't diversify when need it most

# Historical stress captures this perfectly
\`\`\`

**Disadvantages of Historical Stress Testing:**

❌ **"Last War" Problem**
\`\`\`python
# Always fighting the last crisis

# Post-2008: Everyone prepared for housing crash
# → Extensive mortgage stress tests
# → Strong housing risk controls

# 2020: COVID pandemic
# → No one stress tested "global lockdown"
# → Different crisis, different vulnerabilities

# History doesn't repeat exactly
# Mark Twain: "History doesn't repeat, but it rhymes"
\`\`\`

❌ **May Not Be Worst Case**
\`\`\`python
# 2008 was bad, but not worst historical

# Great Depression (1929-1932):
# - Stocks: -89% (vs -38% in 2008)
# - GDP: -30% (vs -4% in 2008)
# - Unemployment: 25% (vs 10% in 2008)

# 2008 stress test might understate risk
# Should test worse scenarios too
\`\`\`

❌ **Different Portfolio Composition**
\`\`\`python
# 2008: Heavy mortgage exposure caused crisis
# Today: Portfolio might have little mortgage exposure
#        But large crypto exposure

# 2008 shocks might not be relevant:
mortgage_shock_2008 = -0.50  # Huge
crypto_shock_2008 = 0.00     # Didn't exist!

# Current risk might be in different areas
# Historical test misses new vulnerabilities
\`\`\`

❌ **Structural Changes**
\`\`\`python
# Markets evolve

# Pre-2008:
# - Less regulation
# - Higher leverage allowed
# - Different market structure

# Post-2008:
# - Dodd-Frank
# - Lower leverage
# - Central clearing
# - Stress testing required

# Would 2008 crash be same magnitude today?
# Regulations might dampen or amplify
# Historical scenario might not repeat
\`\`\`

❌ **Missing New Risks**
\`\`\`python
# New risks that didn't exist historically:

# Cyber attacks on financial infrastructure
cyber_scenario = {
    'NYSE_outage': 5  # days
    'payment_system_compromise': True,
    'customer_data_breach': 10_000_000  # customers
}
# No historical precedent for this!

# Climate change:
climate_scenario = {
    'miami_underwater': 0.30,  # 30% flooded
    'crop_failures': -0.60,    # 60% reduction
}
# New risk, no 2008 analog

# Can't stress test what never happened
\`\`\`

---

**Hypothetical Stress Testing**

**Definition:** Create "what if" scenarios based on expert judgment, not history.

**Method:**
\`\`\`python
def hypothetical_stress_test(portfolio, scenario_definition):
    """
    Test hypothetical scenario
    
    Can be:
    - Extreme but plausible events
    - Multiple simultaneous shocks
    - Tail events beyond historical
    - Forward-looking risks
    """
    # Example: Multiple Shock Scenario
    scenario = {
        'name': 'Perfect Storm',
        'description': 'Multiple severe shocks occurring simultaneously',
        'probability': 'Low but plausible (1-2% per year)',
        
        'shocks': {
            # Shock 1: Geopolitical crisis
            'geopolitical': {
                'oil_spike': 2.0,           # Oil doubles
                'em_selloff': -0.40,        # EM down 40%
                'safe_haven_bid': 0.20,     # Treasuries rally
            },
            
            # Shock 2: Tech bubble burst
            'tech_crash': {
                'tech_stocks': -0.60,       # Tech down 60%
                'nasdaq': -0.50,            # NASDAQ down 50%
                'crypto': -0.80,            # Crypto down 80%
            },
            
            # Shock 3: Credit crisis
            'credit': {
                'hy_spreads': 1500,         # +1500bp
                'defaults': 0.15,           # 15% default rate
                'liquidity': 0.50,          # 50% haircut on illiquid
            },
            
            # Shock 4: Central bank error
            'rates': {
                'fed_hike': 0.03,           # Emergency 300bp hike
                'yield_curve': 'inverted',
            }
        },
        
        # Correlation structure
        'correlations': {
            'equity_correlation': 0.95,     # Everything falls together
            'diversification_fails': True
        }
    }
    
    # Apply multiple shocks
    stressed_value = portfolio.total_value
    
    for shock_type, shock_params in scenario['shocks'].items():
        shock_loss = calculate_shock_impact(portfolio, shock_params)
        stressed_value -= shock_loss
    
    # Add correlation effect (diversification fails)
    correlation_adjustment = 1.20  # 20% worse due to correlations
    total_loss = (portfolio.total_value - stressed_value) * correlation_adjustment
    
    return {
        'scenario': scenario['name'],
        'description': scenario['description'],
        'total_loss': total_loss,
        'loss_pct': total_loss / portfolio.total_value,
        'survived': total_loss < portfolio.capital_buffer,
        'components': {
            shock: calculate_shock_impact(portfolio, params)
            for shock, params in scenario['shocks'].items()
        }
    }

# Example
result = hypothetical_stress_test(portfolio, 'perfect_storm')
print(f"Hypothetical Stress: {result['scenario']}")
print(f"  Loss: \${result['total_loss'] / 1e6: .1f}M({ result['loss_pct']* 100: .1f } %)")
\`\`\`

**Advantages of Hypothetical Stress Testing:**

✅ **Forward-Looking**
\`\`\`python
# Can test risks that don't exist yet

# 2019: Hypothetical pandemic scenario
pandemic_scenario = {
    'global_lockdown': 6,      # months
    'gdp_impact': -0.20,       # -20% GDP
    'travel_industry': -0.80,  # -80%
}

# 2020: This actually happened!
# Firms that tested this were prepared
# Others were caught off-guard

# Hypothetical scenarios can predict future
\`\`\`

✅ **Tailored to Current Risks**
\`\`\`python
# Design scenarios for YOUR specific vulnerabilities

# If portfolio is heavy crypto:
crypto_crash_scenario = {
    'bitcoin': -0.80,
    'ethereum': -0.75,
    'defi_collapse': -0.95,
    'regulatory_ban': True
}

# If portfolio is heavy China:
china_risk_scenario = {
    'delisting_risk': True,
    'capital_controls': True,
    'property_crisis': -0.60
}

# Customize to your risks
# Historical scenarios might miss your specific exposures
\`\`\`

✅ **Test Extreme Tails**
\`\`\`python
# Can test worse than worst historical

# Worst historical: 2008 = -40% stocks
# Hypothetical: -60% stocks (Great Depression level)

# Worst historical: +300bp spreads
# Hypothetical: +1000bp spreads (truly apocalyptic)

# Tests whether you survive beyond-history events
\`\`\`

✅ **Multiple Simultaneous Shocks**
\`\`\`python
# Historical: Usually one primary shock
# 2008: Housing crash
# 2000: Dot-com bubble
# 1987: Black Monday

# Hypothetical: Test multiple shocks at once
simultaneous_shocks = {
    'pandemic': True,
    'geopolitical_war': True,
    'cyberattack': True,
    'climate_disaster': True
}

# "Perfect storm" scenarios
# Low probability but devastating
\`\`\`

✅ **Regulatory Future-Proofing**
\`\`\`python
# Test proposed regulatory changes

# Hypothetical: "What if leverage limits cut in half?"
regulatory_scenario = {
    'max_leverage': 0.50,  # Current
    'new_max_leverage': 0.25,  # Proposed
    'adjustment_period': 90,  # days to comply
}

# Impact on portfolio:
# - Must deleverage
# - Fire sale risk
# - P&L impact

# Prepare before regulation happens
\`\`\`

**Disadvantages of Hypothetical Stress Testing:**

❌ **Subjective / Arbitrary**
\`\`\`python
# Who decides the shocks?

# Risk Manager: "Let's test -40% stock crash"
# CEO: "Why not -20%? Or -60%?"
# Risk Manager: "Based on judgment..."
# CEO: "That's arbitrary"

# Hard to defend severity levels
# No "right" answer

# Unlike historical: "Because it happened in 2008"
\`\`\`

❌ **May Not Be Realistic**
\`\`\`python
# Easy to create impossible scenarios

unrealistic_scenario = {
    'stocks': -0.80,      # Down 80%
    'bonds': -0.50,       # Down 50% (impossible if stocks crash)
    'gold': -0.30,        # Down 30% (gold usually safe haven)
    'vix': 20             # Low vol (impossible in crash)
}

# Internally inconsistent
# Violates basic financial relationships
# Results meaningless

# Historical scenarios are internally consistent
# (They actually happened)
\`\`\`

❌ **Correlation Assumptions**
\`\`\`python
# How do assets correlate in hypothetical scenario?

# Risk Manager: "Stocks down 50%, bonds flat"
# Skeptic: "If stocks crash 50%, bonds would rally!"

# Correlation structure is guesswork
# Historical data doesn't apply
# → Results uncertain

# Example:
# Normal correlation: Stock-Bond = -0.2
# Stress correlation: ???
# Assumption: 0.0, but could be -0.8 or +0.5
# → Huge impact on results
\`\`\`

❌ **False Precision**
\`\`\`python
# Results look precise but aren't

hypothetical_result = {
    'loss': 47_231_892.43  # Precise to the cent!
}

# But based on:
# - Subjective shock sizes
# - Guessed correlations
# - Assumed liquidity
# - Made-up scenario

# Precision is false comfort
# Accuracy is much lower than appears
\`\`\`

❌ **Gaming Risk**
\`\`\`python
# Easy to game if incentives misaligned

# Trader wants big positions
# Designs "hypothetical" scenarios that show them safe:

trader_scenario = {
    'my_positions': -0.10,     # Mild shock
    'everyone_else': -0.50,    # Severe shock
}

# "See, my positions are safe in stress!"
# → Self-serving scenario design

# Historical scenarios harder to game
# (Can't change what happened in 2008)
\`\`\`

---

**Comprehensive Framework: Combining Both**

\`\`\`python
class ComprehensiveStressFramework:
    """
    Best practice: Use both historical and hypothetical
    """
    def __init__(self, portfolio):
        self.portfolio = portfolio
        
        # Historical scenarios
        self.historical_scenarios = {
            'Great Depression (1929)': self.stress_1929(),
            'Black Monday (1987)': self.stress_1987(),
            'Dot-com Crash (2000)': self.stress_2000(),
            'Financial Crisis (2008)': self.stress_2008(),
            'Flash Crash (2010)': self.stress_2010(),
            'COVID Crash (2020)': self.stress_covid(),
        }
        
        # Hypothetical scenarios
        self.hypothetical_scenarios = {
            # Extreme but plausible
            'Severe Recession': self.hypo_severe_recession(),
            'Geopolitical Crisis': self.hypo_geopolitical(),
            'Tech Bubble Burst': self.hypo_tech_bubble(),
            
            # Multiple shocks
            'Perfect Storm': self.hypo_perfect_storm(),
            
            # Forward-looking risks
            'Cyber Attack': self.hypo_cyber(),
            'Climate Shock': self.hypo_climate(),
            
            # Reverse stress test
            'What Breaks Us': self.reverse_stress_test(),
        }
    
    def run_comprehensive_stress_test(self):
        """
        Run all scenarios
        """
        results = {}
        
        # Historical
        print("Historical Stress Tests:")
        for name, scenario in self.historical_scenarios.items():
            result = scenario
            results[name] = result
            print(f"  {name}: Loss \${result['loss'] / 1e6: .1f}M({ result['loss_pct']* 100: .1f } %)")
        
        # Hypothetical
print("\nHypothetical Stress Tests:")
for name, scenario in self.hypothetical_scenarios.items():
    result = scenario
results[name] = result
print(f"  {name}: Loss \${result['loss']/1e6:.1f}M ({result['loss_pct']*100:.1f}%)")
        
        # Analysis
worst_historical = max(self.historical_scenarios.values(), key = lambda x: x['loss'])
worst_hypothetical = max(self.hypothetical_scenarios.values(), key = lambda x: x['loss'])

print(f"\nWorst Historical: {worst_historical['name']} (\${worst_historical['loss']/1e6:.1f}M)")
print(f"Worst Hypothetical: {worst_hypothetical['name']} (\${worst_hypothetical['loss']/1e6:.1f}M)")
        
        # Capital adequacy
capital_buffer = self.portfolio.capital_buffer

can_survive_all = all(r['loss'] < capital_buffer for r in results.values())

    if can_survive_all:
        print(f"\n✓ Portfolio survives all scenarios")
    else:
    failures = [name for name, r in results.items() if r['loss'] >= capital_buffer]
print(f"\n✗ Portfolio fails in: {failures}")

return results
    
    def reverse_stress_test(self):
"""
Hypothetical: What scenario makes us insolvent ?
    """
capital_buffer = self.portfolio.capital_buffer
        
        # Ramp up severity until failure
for severity in np.linspace(1.0, 5.0, 50):
    scenario = {
        'stocks': -0.30 * severity,
        'credit': -0.20 * severity,
        'liquidity_haircut': 0.20 * severity,
    }

loss = self.calculate_scenario_loss(scenario)

if loss >= capital_buffer:
    return {
        'name': 'Breaking Point',
        'severity_multiplier': severity,
        'scenario': scenario,
        'loss': loss,
        'loss_pct': loss / self.portfolio.total_value
    }

return { 'name': 'Unbreakable', 'loss': 0 }

# Usage
framework = ComprehensiveStressFramework(portfolio)
results = framework.run_comprehensive_stress_test()
\`\`\`

**Best Practices for Combined Framework:**

**1. Use Historical for Known Risks**
\`\`\`python
known_risks = {
    'Market crash': 'Test 2008, 1987, 2000',
    'Credit crisis': 'Test 2008',
    'Sovereign debt': 'Test European crisis 2011',
    'Pandemic': 'Test COVID 2020'
}

# If risk has historical precedent, test it
\`\`\`

**2. Use Hypothetical for New/Emerging Risks**
\`\`\`python
emerging_risks = {
    'Cyber attack': hypothetical_cyber_scenario(),
    'Climate disaster': hypothetical_climate_scenario(),
    'AI disruption': hypothetical_ai_scenario(),
    'Quantum computing breaks crypto': hypothetical_quantum_scenario()
}

# Historical can't test what never happened
\`\`\`

**3. Use Reverse Stress Test to Find Vulnerabilities**
\`\`\`python
# Don't assume scenario
# Ask: "What breaks us?"

reverse_test = find_breaking_scenario(portfolio)
# → Reveals hidden vulnerabilities
\`\`\`

**4. Validate Hypothetical Against Historical**
\`\`\`python
# Sanity check hypothetical scenarios

# If hypothetical 2008 gives different result than actual 2008:
hypo_2008_loss = hypothetical_2008_scenario()
actual_2008_loss = historical_2008_scenario()

if abs(hypo_2008_loss - actual_2008_loss) / actual_2008_loss > 0.30:
    print("⚠️ Hypothetical methodology may be off")
    print("   Calibrate hypothetical scenarios to match historical")
\`\`\`

**5. Use Both for Capital Planning**
\`\`\`python
# Conservative: Hold capital for worst of all scenarios

worst_case = max(
    max(historical_scenarios.values()),
    max(hypothetical_scenarios.values())
)

required_capital = worst_case * 1.20  # 20% buffer
\`\`\`

---

**Summary**

**Historical Stress Testing:**
- **Use for**: Known risks, regulatory compliance, board communication
- **Strengths**: Actually happened, credible, complete picture
- **Weaknesses**: Backward-looking, may miss new risks

**Hypothetical Stress Testing:**
- **Use for**: Forward-looking risks, tail events, customization
- **Strengths**: Flexible, forward-looking, tailored
- **Weaknesses**: Subjective, may be unrealistic, harder to defend

**Best Practice**: Use BOTH
- Historical: Foundation (test known crises)
- Hypothetical: Forward-looking (test emerging risks)
- Reverse: Find vulnerabilities (what breaks us?)

**The lesson from 2008 and COVID**: Firms that only tested historical scenarios were unprepared for new crises. Firms that supplemented with hypothetical scenarios (including tail events and new risks) were better positioned. A comprehensive framework uses both approaches synergistically.`,
    },
    {
      question:
        'Explain the purpose and methodology of reverse stress testing. Why is asking "what would cause us to fail?" more valuable than asking "what happens if X occurs?" Provide examples of how reverse stress testing revealed hidden vulnerabilities.',
      answer: `Reverse stress testing flips traditional stress testing on its head - and often reveals risks that forward stress testing misses entirely:

**Traditional Stress Testing vs Reverse Stress Testing**

**Traditional (Forward) Stress Testing:**
\`\`\`python
# Question: "What happens if X occurs?"

def forward_stress_test(portfolio, scenario):
    """
    Given scenario → Calculate impact
    """
    # Define scenario
    scenario = {
        'stocks': -0.40,
        'bonds': 0.10,
        'credit_spreads': 300
    }
    
    # Calculate impact
    loss = apply_scenario(portfolio, scenario)
    
    # Answer: "We lose $50M"
    return loss

# Problem: What if the scenario we chose isn't the dangerous one?
# We test what we think of, miss what we don't think of
\`\`\`

**Reverse Stress Testing:**
\`\`\`python
# Question: "What would cause us to fail?"

def reverse_stress_test(portfolio, failure_threshold):
    """
    Given failure point → Find scenarios that cause it
    
    Work backwards from failure to scenario
    """
    # Define failure
    failure_threshold = portfolio.capital_buffer
    
    # Search for scenarios that cause failure
    breaking_scenarios = []
    
    # Method 1: Systematic search
    for scenario in generate_scenarios():
        loss = apply_scenario(portfolio, scenario)
        if loss >= failure_threshold:
            breaking_scenarios.append({
                'scenario': scenario,
                'loss': loss,
                'severity': calculate_severity(scenario)
            })
    
    # Find minimum severity scenarios that break us
    critical_scenarios = sorted(breaking_scenarios, key=lambda x: x['severity'])
    
    return critical_scenarios[:5]  # Top 5 most plausible failures

# Answer: "We fail if interest rates spike 500bp AND credit spreads widen 1000bp"
# → Reveals hidden vulnerability to rate/credit combination
\`\`\`

---

**Why Reverse Stress Testing Is More Valuable**

**Reason 1: Reveals Hidden Concentrations**

\`\`\`python
# Forward test might miss concentrated risk

# Forward approach:
forward_tests = {
    'Stock crash': loss_if_stocks_fall_40(),
    'Rate spike': loss_if_rates_up_300bp(),
    'Credit crisis': loss_if_spreads_1000bp()
}
# Each test passed individually ✓

# Reverse approach:
reverse_test = find_breaking_scenario()
# → Discovers: "We fail if stocks fall 25% AND rates up 200bp"

# Hidden risk: COMBINATION of two moderate shocks
# Neither alone is fatal, together they are

# Example:
portfolio = {
    'leveraged_equity': 1_000_000_000,
    'funded_with': 'short_term_borrowing'
}

# Forward test 1: "What if stocks fall 40%?"
# Loss: $400M, survive (have $500M capital)

# Forward test 2: "What if rates up 300bp?"
# Loss: $50M (refinancing cost), survive

# Reverse test: "What breaks us?"
# → Stocks fall 30% = $300M loss
#   PLUS rates up 200bp = $100M refinancing
#   PLUS margin calls = $150M
#   Total: $550M > $500M capital
#   → FAILURE

# Forward tests missed the interaction!
\`\`\`

**Real Example: Long-Term Capital Management (1998)**

\`\`\`python
# LTCM had forward stress tests:
# - Russian default: Survived ✓
# - Market crash: Survived ✓
# - Flight to quality: Survived ✓

# What actually happened:
# Russian default (Aug 1998)
# → Flight to quality
# → Credit spreads exploded
# → Liquidity dried up
# → Forced selling
# → Margin calls
# → Positions liquidated at worst prices

# Combination of ALL events simultaneously
# Plus liquidity crisis (no buyers)

# Reverse test would have found:
"LTCM fails if:
1. Volatility spikes 3x (their short volatility positions)
2. Credit spreads widen 1000bp (their convergence trades)
3. Liquidity disappears (cannot unwind)
4. All happen simultaneously"

# That's exactly what happened
# Forward tests missed the compound scenario
\`\`\`

**Reason 2: Tests Plausibility of Survival**

\`\`\`python
# Forward tests might all show survival
# But at what cost?

# Forward approach:
for scenario in standard_scenarios:
    loss = test_scenario(scenario)
    print(f"{scenario}: Loss \${loss}M, Capital \${capital}M ✓")
    # All pass!

# Reverse approach:
breaking_point = find_minimum_breaking_scenario()

print(f"We fail if:")
print(f"  - Stocks down {breaking_point['stocks']}%")
print(f"  - Credit spreads +{breaking_point['spreads']}bp")
print(f"Plausibility: This is a {breaking_point['severity']}-sigma event")

# If breaking_point requires only 2-sigma move:
# → "We fail in routine volatility spike"
# → Not actually safe!

# If breaking_point requires 6-sigma move:
# → "We fail only in extreme tail"
# → Actually resilient

# Reverse test reveals how close to edge you are
\`\`\`

**Reason 3: Finds Non-Obvious Vulnerabilities**

\`\`\`python
# Forward tests test what you think of
# Reverse tests find what you missed

# Example: Operational dependencies

# Forward test: "Tech company loses 50% market cap"
# Loss: $100M, survive ✓

# Reverse test: "What breaks us?"
# → Discovers dependency on specific vendor
breaking_scenario = {
    'event': 'Cloud provider XYZ goes down',
    'impact': {
        'trading_system': 'offline',
        'duration': '5 days',
        'miss_margin_calls': True,
        'regulatory_breach': True,
        'customer_exodus': 0.40,
        'total_impact': 'INSOLVENCY'
    }
}

# Never tested vendor dependency in forward tests!
# Reverse test reveals single point of failure
\`\`\`

---

**Methodology for Reverse Stress Testing**

**Step 1: Define Failure**

\`\`\`python
def define_failure_criteria(firm):
    """
    What constitutes failure?
    """
    failure_criteria = {
        # Financial failure
        'capital_depletion': {
            'tier1_ratio': 0.045,  # Below minimum
            'total_capital': firm.min_capital_requirement
        },
        
        # Liquidity failure
        'liquidity_crisis': {
            'lcr': 1.0,  # Below 100%
            'cannot_meet': 'margin_call'
        },
        
        # Operational failure
        'operational': {
            'extended_outage': 5,  # days
            'data_breach': 'material',
            'regulatory_action': 'license_revoked'
        },
        
        # Reputational failure
        'reputational': {
            'client_withdrawals': 0.50,  # 50%+ AUM withdrawn
            'unable_to_fundraise': True
        }
    }
    
    return failure_criteria

# Be comprehensive - failure has many forms
\`\`\`

**Step 2: Work Backwards**

\`\`\`python
def reverse_stress_systematic(portfolio, failure_threshold):
    """
    Systematic search for breaking scenarios
    """
    # Start with current portfolio
    current_capital = portfolio.capital_buffer
    
    # How much loss causes failure?
    max_tolerable_loss = current_capital - failure_threshold
    
    print(f"Current capital: \${current_capital / 1e6: .1f
}M")
print(f"Failure point: \${failure_threshold/1e6:.1f}M")
print(f"Max tolerable loss: \${max_tolerable_loss/1e6:.1f}M")
    
    # Search for scenarios that cause this loss
breaking_scenarios = []
    
    # Grid search over market moves
for stock_move in np.arange(0, -0.80, -0.05):  # 0 % to - 80 % in 5 % steps
for spread_move in np.arange(0, 2000, 100):   # 0bp to 2000bp
for rate_move in np.arange(-0.03, 0.05, 0.005):  # - 300bp to + 500bp

scenario = {
    'stocks': stock_move,
    'credit_spreads': spread_move,
    'rates': rate_move
}
                
                # Calculate loss
loss = portfolio.calculate_loss(scenario)
                
                # Check if causes failure
if loss >= max_tolerable_loss:
                    # Calculate plausibility
sigma_level = calculate_sigma_level(scenario)

breaking_scenarios.append({
    'scenario': scenario,
    'loss': loss,
    'sigma_level': sigma_level,
    'plausibility': get_plausibility(sigma_level)
})
    
    # Sort by plausibility(most likely failures first)
breaking_scenarios.sort(key = lambda x: x['sigma_level'])

return breaking_scenarios

def calculate_sigma_level(scenario):
"""
    How many standard deviations is this scenario ?
    """
historical_vol = get_historical_volatility()
    
    # Z - score for each component
    z_stocks = scenario['stocks'] / historical_vol['stocks']
    z_spreads = scenario['credit_spreads'] / historical_vol['spreads']
z_rates = scenario['rates'] / historical_vol['rates']
    
    # Combined sigma level(roughly)
combined_sigma = np.sqrt(z_stocks ** 2 + z_spreads ** 2 + z_rates ** 2)

return combined_sigma

def get_plausibility(sigma_level):
"""
    Convert sigma to plausibility
"""
if sigma_level < 2:
    return "HIGH - Happens every few years"
    elif sigma_level < 3:
return "MEDIUM - Happens every decade"
    elif sigma_level < 4:
return "LOW - Rare but possible"
    else:
return "VERY LOW - Extreme tail"
\`\`\`

**Step 3: Analyze Critical Scenarios**

\`\`\`python
def analyze_breaking_scenarios(scenarios):
    """
    Understand what makes us vulnerable
    """
    print("Critical Failure Scenarios:")
    print("="*80)
    
    for i, scenario in enumerate(scenarios[:5]):  # Top 5
        print(f"\nScenario {i+1}: {scenario['plausibility']}")
        print(f"  Sigma Level: {scenario['sigma_level']:.2f}")
        print(f"  Loss: \${scenario['loss'] / 1e6: .1f}M")
print(f"  Market Moves:")
print(f"    Stocks: {scenario['scenario']['stocks']*100:.1f}%")
print(f"    Credit Spreads: +{scenario['scenario']['credit_spreads']:.0f}bp")
print(f"    Rates: {scenario['scenario']['rates']*100:+.1f}%")
    
    # Identify patterns
print("\n" + "=" * 80)
print("Vulnerability Pattern Analysis:")
    
    # Common factors
common_factors = identify_common_factors(scenarios)
print(f"\nCommon risk factors in breaking scenarios:")
for factor, frequency in common_factors.items():
    print(f"  - {factor}: appears in {frequency}% of failures")
    
    # Single points of failure
spof = identify_single_points_of_failure(scenarios)
if spof:
    print(f"\n⚠️ SINGLE POINTS OF FAILURE DETECTED:")
for item in spof:
    print(f"  - {item}")

# Example output:
"""
Critical Failure Scenarios:
================================================================================

Scenario 1: MEDIUM - Happens every decade
  Sigma Level: 2.8
  Loss: $520M
  Market Moves:
    Stocks: -35.0%
    Credit Spreads: +800bp
    Rates: +2.0%

Scenario 2: MEDIUM - Happens every decade
  Sigma Level: 2.9
  Loss: $515M
  Market Moves:
    Stocks: -40.0%
    Credit Spreads: +600bp
    Rates: +1.5%

================================================================================
Vulnerability Pattern Analysis:

Common risk factors in breaking scenarios:
  - Credit spread widening: appears in 95% of failures
  - Equity drawdown: appears in 85% of failures
  - Rate increases: appears in 75% of failures

⚠️ SINGLE POINTS OF FAILURE DETECTED:
  - High yield corporate bond portfolio (concentrated)
  - Funding dependent on short-term repo (liquidity risk)
  - Clearing member concentration (operational risk)
"""
\`\`\`

---

**Real-World Examples**

**Example 1: MF Global (2011)**

\`\`\`python
# MF Global: Large broker-dealer

# Forward stress tests (what they did):
mf_global_forward = {
    'European sovereign crisis': 'Survived',
    'Credit spread widening': 'Survived',
    'Repo market stress': 'Survived'
}
# All individual tests passed ✓

# What reverse test would have found:
reverse_test_result = {
    'failure_scenario': 'European sovereign crisis',
    'trigger_sequence': [
        'Italian bond prices fall',
        'Margin calls on sovereign positions',
        'Liquidity squeeze (cannot roll repo)',
        'Customer funds improperly used',
        'Discovered by regulators',
        'Customer panic withdrawal',
        'Bankruptcy'
    ],
    'vulnerability': 'Liquidity + Operational + Reputational cascade'
}

# Reverse test would have revealed:
# "We fail if European crisis triggers margin calls
#  AND we cannot roll repo funding
#  AND customers lose confidence"

# Not just market risk - operational and funding risk compound

# Lesson: Reverse test finds compound vulnerabilities
\`\`\`

**Example 2: Archegos Capital (2021)**

\`\`\`python
# Archegos: Family office with total return swaps

# Forward stress tests (presumably done):
archegos_forward = {
    'ViacomCBS down 30%': 'Survives',
    'Tech stocks down 40%': 'Survives',
    'Volatility spike': 'Survives'
}

# What actually happened (reverse scenario):
archegos_failure = {
    'initial_shock': 'ViacomCBS secondary offering → stock down 25%',
    'margin_call': '$1B+',
    'concentration': 'Highly concentrated portfolio',
    'leverage': '5-10x via swaps',
    'cascade': [
        'Cannot meet margin call',
        'Forced liquidation starts',
        'Stock prices fall further',
        'More margin calls',
        'More liquidation',
        'Death spiral',
        'Prime brokers liquidate',
        '$10B+ loss',
        'Prime brokers lose $10B+ too'
    ]
}

# Reverse test would have found:
reverse_test_archegos = {
    'question': 'What causes Archegos to fail?',
    'answer': 'Initial 25% drop in concentrated position',
    'mechanism': 'Leverage + concentration + illiquidity',
    'revelation': 'Failure requires only moderate market move!',
    'sigma_level': 2.0,  # Not even extreme!
    'action_needed': 'Reduce leverage OR diversify OR add liquidity buffer'
}

# Reverse test shows: Very fragile
# 2-sigma event causes total failure
# Forward tests missed the fragility
\`\`\`

**Example 3: Bear Stearns (2008)**

\`\`\`python
# Bear Stearns: Major investment bank

# Forward stress tests (likely done):
bear_forward = {
    'Mortgage crisis': 'Survives',
    'Market crash': 'Survives',
    'Credit crisis': 'Survives'
}

# Reverse test would have revealed:
bear_reverse = {
    'question': 'What causes Bear to fail?',
    'answer': 'Confidence loss',
    'mechanism': [
        'Rumor of liquidity problems',
        'Hedge fund clients withdraw',
        'Repo lenders refuse to roll',
        'Stock price collapses',
        'More confidence loss',
        'Bank run',
        'Failure in 72 hours'
    ],
    'critical_insight': 'Liquidity is confidence-dependent',
    'vulnerability': 'Run on bank possible even if solvent',
    'sigma_level': 1.5,  # VERY plausible!
}

# Bear collapsed in 3 days
# Not from massive losses (initially)
# From liquidity run

# Reverse test: "How do we fail?"
# Answer: "Confidence loss → funding run"
# → Should have had larger liquidity buffer
# → Should have diversified funding sources
# → Should have had contingency plans

# Forward tests missed behavioral risk
\`\`\`

---

**Implementation Framework**

\`\`\`python
class ReverseStressTestingFramework:
    """
    Comprehensive reverse stress testing
    """
    def __init__(self, portfolio):
        self.portfolio = portfolio

    def run_reverse_stress_test(self):
        """
        Complete reverse stress testing process
        """
        # Step 1: Define failure
        failure_criteria = self.define_failure()

        # Step 2: Find breaking scenarios
        breaking_scenarios = self.find_breaking_scenarios(failure_criteria)

        # Step 3: Analyze patterns
        vulnerabilities = self.analyze_vulnerabilities(breaking_scenarios)

        # Step 4: Recommend actions
        recommendations = self.generate_recommendations(vulnerabilities)

        # Step 5: Report
        self.generate_report(
            failure_criteria,
            breaking_scenarios,
            vulnerabilities,
            recommendations
        )

        return {
            'breaking_scenarios': breaking_scenarios,
            'vulnerabilities': vulnerabilities,
            'recommendations': recommendations
        }

    def find_breaking_scenarios(self, failure_criteria):
        """
        Search for scenarios that cause failure
        """
        scenarios = []

        # Market risk scenarios
        scenarios.extend(self.find_market_breaks())

        # Credit risk scenarios
        scenarios.extend(self.find_credit_breaks())

        # Liquidity risk scenarios
        scenarios.extend(self.find_liquidity_breaks())

        # Operational risk scenarios
        scenarios.extend(self.find_operational_breaks())

        # Combination scenarios
        scenarios.extend(self.find_combination_breaks())

        return sorted(scenarios, key=lambda x: x['sigma_level'])

    def analyze_vulnerabilities(self, scenarios):
        """
        Identify common vulnerabilities
        """
        vulnerabilities = {
            'concentrations': self.find_concentrations(scenarios),
            'single_points_of_failure': self.find_spof(scenarios),
            'critical_dependencies': self.find_dependencies(scenarios),
            'compounding_risks': self.find_compounding(scenarios)
        }

        return vulnerabilities

    def generate_recommendations(self, vulnerabilities):
        """
        Recommend risk mitigation
        """
        recommendations = []

        # For each vulnerability, suggest mitigation
        for vuln_type, vuln_details in vulnerabilities.items():
            for vuln in vuln_details:
                recommendation = self.create_recommendation(vuln)
                recommendations.append(recommendation)

        # Prioritize by impact and ease
        recommendations.sort(key=lambda x: x['priority'])

        return recommendations

# Usage
framework = ReverseStressTestingFramework(portfolio)
results = framework.run_reverse_stress_test()

print("Reverse Stress Test Results:")
print(f"  Found {len(results['breaking_scenarios'])} breaking scenarios")
print(f"  Identified {len(results['vulnerabilities'])} key vulnerabilities")
print(f"  Generated {len(results['recommendations'])} recommendations")
\`\`\`

---

**Key Insights from Reverse Stress Testing**

**1. Reveals Hidden Concentrations**
- Forward: Tests obvious risks
- Reverse: Finds hidden concentrations in tail

**2. Shows Compound Effects**
- Forward: Tests risks in isolation
- Reverse: Finds dangerous combinations

**3. Tests Plausibility**
- Forward: "We survive X"
- Reverse: "How extreme must X be to break us?"

**4. Finds Non-Financial Risks**
- Forward: Focuses on market risk
- Reverse: Finds operational, reputational, behavioral risks

**5. Prioritizes Risk Mitigation**
- Forward: All scenarios equally important
- Reverse: Focus on most plausible breaking scenarios

---

**Summary**

**Why Reverse Stress Testing is More Valuable:**

1. **Finds what you don't think to test**
   - Forward: Test what you think of
   - Reverse: Find what breaks you (might surprise you)

2. **Reveals compound vulnerabilities**
   - Forward: Tests risks individually
   - Reverse: Finds dangerous combinations

3. **Shows proximity to failure**
   - Forward: Binary (survive or not)
   - Reverse: "How close are we to edge?"

4. **Actionable insights**
   - Forward: "We survive scenario X"
   - Reverse: "Fix vulnerability Y to survive"

**Real-World Lessons:**
- LTCM: Compound scenario broke them
- MF Global: Liquidity + operational cascade
- Archegos: Leverage + concentration + illiquidity
- Bear Stearns: Confidence loss → run

All could have been found by asking: "What breaks us?"

**Best Practice:** Do both forward AND reverse stress testing. Forward tests known risks. Reverse tests find hidden vulnerabilities. Together they provide complete picture.`,
    },
    {
      question:
        'Explain the CCAR (Comprehensive Capital Analysis and Review) framework and DFAST (Dodd-Frank Act Stress Testing). How do regulatory stress tests differ from internal stress tests, and what are the strategic implications for banks?',
      answer: `CCAR and DFAST are the backbone of post-2008 bank regulation. Understanding them is essential for anyone in bank risk management:

**CCAR: Comprehensive Capital Analysis and Review**

**Purpose:** Ensure large banks can survive severe economic downturn while continuing to lend.

**Who Must Comply:**
\`\`\`python
# U.S. banks with $100B+ assets
required_banks = [
    'JPMorgan Chase',
    'Bank of America',
    'Citigroup',
    'Wells Fargo',
    'Goldman Sachs',
    'Morgan Stanley',
    # ... ~35 total banks
]

# Also: U.S. subsidiaries of foreign banks
\`\`\`

**Annual Cycle:**
\`\`\`python
ccar_timeline = {
    'October': 'Fed publishes scenarios',
    'January': 'Banks submit capital plans',
    'March': 'Banks submit stress test results',
    'June': 'Fed publishes results and decisions',
    'July': 'Capital actions (dividends, buybacks) can proceed if approved'
}

# Takes entire year to complete!
\`\`\`

**Three Scenarios Required:**

\`\`\`python
def ccar_scenarios():
    """
    Three macroeconomic scenarios
    """
    scenarios = {
        'baseline': {
            'description': 'Consensus economic forecast',
            'gdp_growth': 0.02,  # 2%
            'unemployment': 0.04,  # 4%
            'stock_market': 0.05,  # +5% per year
            'house_prices': 0.03,  # +3% per year
        },

        'adverse': {
            'description': 'Moderate recession',
            'gdp_growth': -0.01,  # -1%
            'unemployment': 0.07,  # 7% (spike)
            'stock_market': -0.15,  # -15%
            'house_prices': -0.08,  # -8%
            'duration_quarters': 9
        },

        'severely_adverse': {
            'description': 'Severe global recession',
            'gdp_growth': -0.04,  # -4% (2008-level)
            'unemployment': 0.10,  # 10%
            'stock_market': -0.50,  # -50% (crashes)
            'house_prices': -0.25,  # -25%
            'credit_spreads': 5.0,  # 5x wider
            'volatility': 3.0,  # VIX to 75+
            'duration_quarters': 9
        }
    }

    return scenarios

# Severely adverse is the key test
# Designed to be worse than 2008 in some dimensions
\`\`\`

**Capital Requirements:**

\`\`\`python
def ccar_capital_requirements():
    """
    Must maintain minimum ratios EVEN IN severely adverse scenario
    """
    minimum_ratios = {
        'CET1_ratio': 0.045,  # 4.5% (Common Equity Tier 1)
        'Tier1_ratio': 0.06,   # 6.0%
        'Total_capital_ratio': 0.08,  # 8.0%
        'Tier1_leverage_ratio': 0.04  # 4.0%
    }

    # Additional buffer
    capital_buffer = 0.025  # 2.5% (Capital Conservation Buffer)

    # Effective minimum (to avoid restrictions)
    effective_minimum = {
        'CET1': 0.045 + 0.025,  # 7.0%
        'Tier1': 0.06 + 0.025,   # 8.5%
    }

    return minimum_ratios, capital_buffer

# Banks must stay above minimums throughout 9-quarter stress
# If fall below → Cannot return capital to shareholders
\`\`\`

**Stress Test Components:**

\`\`\`python
class CCARStressTest:
    """
    Comprehensive stress testing process
    """
    def __init__(self, bank):
        self.bank = bank

    def run_ccar_stress_test(self, scenario):
        """
        9-quarter projection under stress
        """
        results_by_quarter = []

        # Starting point
        capital_position = self.bank.current_capital

        for quarter in range(9):
            # Apply macroeconomic scenario
            macro_state = scenario.get_quarter(quarter)

            # Calculate impacts
            impacts = {
                # Pre-provision net revenue (income before credit losses)
                'ppnr': self.calculate_ppnr(macro_state),

                # Credit losses
                'credit_losses': self.calculate_credit_losses(macro_state),

                # Market risk losses
                'trading_losses': self.calculate_trading_losses(macro_state),

                # Operational losses
                'operational_losses': self.estimate_operational_losses(),

                # Other impacts
                'tax_effects': self.calculate_taxes(),
            }

            # Net income/loss for quarter
            net_income = (
                impacts['ppnr'] -
                impacts['credit_losses'] -
                impacts['trading_losses'] -
                impacts['operational_losses'] -
                impacts['tax_effects']
            )

            # Update capital
            capital_position += net_income

            # RWA changes
            rwa = self.calculate_rwa(macro_state)

            # Capital ratios
            ratios = {
                'CET1': capital_position / rwa,
                'Tier1': capital_position / rwa,  # Simplified
                'Total': capital_position / rwa
            }

            # Check minimums
            meets_minimums = all(
                ratios[r] >= minimum
                for r, minimum in minimum_ratios.items()
            )

            results_by_quarter.append({
                'quarter': quarter,
                'capital': capital_position,
                'rwa': rwa,
                'ratios': ratios,
                'net_income': net_income,
                'meets_minimums': meets_minimums
            })

        return results_by_quarter

    def calculate_credit_losses(self, macro_state):
        """
        Project credit losses under stress

        Most important component of CCAR
        """
        # Loan portfolio
        loan_categories = {
            'residential_mortgage': self.bank.mortgages,
            'commercial_real_estate': self.bank.cre_loans,
            'credit_card': self.bank.credit_cards,
            'auto_loans': self.bank.auto_loans,
            'commercial_industrial': self.bank.ci_loans
        }

        total_losses = 0

        for category, portfolio in loan_categories.items():
            # Loss rate depends on macro conditions
            if category == 'residential_mortgage':
                loss_rate = self.mortgage_loss_rate(
                    unemployment=macro_state['unemployment'],
                    house_prices=macro_state['house_prices']
                )
            elif category == 'credit_card':
                loss_rate = self.credit_card_loss_rate(
                    unemployment=macro_state['unemployment']
                )
            # ... other categories

            # Apply loss rate
            losses = portfolio.balance * loss_rate
            total_losses += losses

        return total_losses

    def mortgage_loss_rate(self, unemployment, house_prices):
        """
        Model mortgage losses

        Based on:
        - Unemployment (ability to pay)
        - House prices (LTV / incentive to default)
        """
        # Empirical relationship
        base_loss_rate = 0.02  # 2% base

        # Unemployment effect
        unemp_effect = unemployment * 0.5  # Each % unemp adds 0.5% loss

        # House price effect
        hp_decline = min(0, house_prices)  # Only declines matter
        hp_effect = -hp_decline * 0.3  # Each % decline adds 0.3% loss

        total_loss_rate = base_loss_rate + unemp_effect + hp_effect

        return total_loss_rate

# Example
bank = LargeBank()
stress_test = CCARStressTest(bank)

severely_adverse = get_fed_scenario('severely_adverse')
results = stress_test.run_ccar_stress_test(severely_adverse)

# Check if passed
min_cet1 = min(q['ratios']['CET1'] for q in results)
passed = min_cet1 >= 0.045

print(f"CCAR Result: {'PASS' if passed else 'FAIL'}")
print(f"Minimum CET1: {min_cet1*100:.2f}%")
\`\`\`

---

**DFAST: Dodd-Frank Act Stress Testing**

**Difference from CCAR:**

\`\`\`python
comparison = {
    'CCAR': {
        'who': 'Banks $100B+',
        'authority': 'Federal Reserve',
        'focus': 'Capital planning + stress testing',
        'result': 'Can approve/reject capital plans',
        'includes': 'Qualitative review of risk management'
    },

    'DFAST': {
        'who': 'Banks $250B+ (was $10B+)',
        'authority': 'Fed + OCC + FDIC',
        'focus': 'Stress testing only',
        'result': 'Publish results, no capital plan review',
        'includes': 'Quantitative stress test only'
    }
}

# CCAR is stricter (includes qualitative review)
# DFAST is just the stress test numbers
\`\`\`

**Key Difference:**

\`\`\`python
# CCAR = DFAST + Qualitative Review + Capital Plan Review

# DFAST: "Here's what happens in stress"
dfast_result = {
    'stress_loss': 50_000_000_000,  # $50B
    'capital_ratios': {
        'start': 0.12,  # 12%
        'minimum': 0.08,  # 8%
    }
}

# CCAR: "Can you pay dividends?"
ccar_decision = {
    'dfast_result': dfast_result,
    'capital_plan': {
        'dividends': 10_000_000_000,  # $10B
        'buybacks': 15_000_000_000,   # $15B
    },
    'qualitative_review': {
        'risk_management': 'Adequate',
        'capital_planning': 'Strong',
        'governance': 'Effective'
    },
    'decision': 'APPROVED' or 'OBJECTION'
}

# Fed can object to capital plan even if bank passes stress
# → Banks must have strong processes, not just capital
\`\`\`

---

**Regulatory vs Internal Stress Tests**

**Key Differences:**

\`\`\`python
def compare_stress_tests():
    """
    Regulatory vs Internal stress testing
    """
    comparison = {
        'Scenarios': {
            'Regulatory': 'Fed-prescribed (same for all banks)',
            'Internal': 'Bank-specific (tailored to risks)'
        },

        'Frequency': {
            'Regulatory': 'Annual',
            'Internal': 'Quarterly or continuous'
        },

        'Disclosure': {
            'Regulatory': 'Public results',
            'Internal': 'Confidential'
        },

        'Consequences': {
            'Regulatory': 'Capital restrictions if fail',
            'Internal': 'Risk management decisions'
        },

        'Models': {
            'Regulatory': 'Fed uses own models',
            'Internal': 'Bank uses own models'
        },

        'Flexibility': {
            'Regulatory': 'Fixed format, comparable across banks',
            'Internal': 'Flexible, customized'
        }
    }

    return comparison
\`\`\`

**Example: Different Results**

\`\`\`python
# Same bank, same scenario, different results:

# Fed's model (regulatory CCAR):
fed_result = {
    'credit_losses': 80_000_000_000,  # $80B
    'minimum_CET1': 0.065,  # 6.5%
    'result': 'PASS (above 4.5% minimum)'
}

# Bank's internal model:
bank_internal = {
    'credit_losses': 60_000_000_000,  # $60B (less conservative)
    'minimum_CET1': 0.082,  # 8.2%
    'result': 'PASS with buffer'
}

# Why different?
differences = {
    'loss_models': 'Fed more conservative on credit cards',
    'PPNR': 'Bank projects higher net interest income',
    'trading_losses': 'Fed assumes worse market risk',
    'methodology': 'Different modeling approaches'
}

# Fed's is "stress test of last resort"
# Bank's is for internal risk management
\`\`\`

**Strategic Implications for Banks**

**Implication 1: Capital Planning Constrained**

\`\`\`python
# Banks must "pre-fund" stress losses

# Without CCAR:
bank_could_hold = {
    'minimum_capital': 100_000_000_000,  # $100B (8% of RWA)
    'actual_capital': 100_000_000_000,   # Hold exactly minimum
    'buffer': 0
}

# With CCAR:
bank_must_hold = {
    'minimum_capital': 100_000_000_000,  # $100B
    'stress_loss': 50_000_000_000,       # $50B projected loss
    'buffer_needed': 50_000_000_000,     # Must have buffer for stress
    'actual_capital': 150_000_000_000    # $150B (50% more!)
}

# CCAR requires massive capital buffers
# Cannot operate at minimum capital anymore
\`\`\`

**Implication 2: Limits Shareholder Returns**

\`\`\`python
# Banks want to return capital (dividends, buybacks)
# CCAR limits this

# Example:
bank_capital = 200_000_000_000  # $200B
net_income = 20_000_000_000     # $20B/year

# Pre-CCAR: Could return most of earnings
pre_ccar_return = 0.80 * net_income  # 80% payout
# = $16B

# Post-CCAR: Limited by stress test
post_ccar_calculation = {
    'starting_capital': 200_000_000_000,
    'stress_loss': 60_000_000_000,
    'minimum_required': 100_000_000_000,
    'excess_after_stress': 200 - 60 - 100,  # $40B
    'can_return': 40_000_000_000  # Over time
}

# But in practice:
# Fed scrutinizes capital plans
# Banks keep extra buffer to ensure approval
actual_payout_ratio = 0.40  # 40% (vs 80% pre-CCAR)

# Cost: Lower ROE, less attractive to investors
\`\`\`

**Implication 3: Business Model Impact**

\`\`\`python
# Certain businesses are capital-heavy in stress

# Trading business:
trading_stress_loss = 20_000_000_000  # $20B
trading_capital_required = 50_000_000_000  # $50B to absorb
trading_roe_stressed = 0.10  # 10%

# Mortgage business:
mortgage_stress_loss = 5_000_000_000  # $5B
mortgage_capital_required = 10_000_000_000  # $10B
mortgage_roe_stressed = 0.15  # 15%

# Decision: Shift from trading to mortgage
# CCAR makes trading less attractive

# Post-2008 trend:
# - Banks reduced trading (Volcker Rule + CCAR)
# - Increased lending (less capital-intensive in stress)
\`\`\`

**Implication 4: Competitive Effects**

\`\`\`python
# CCAR advantage for big banks

# Small bank (not subject to CCAR):
small_bank = {
    'capital': 5_000_000_000,  # $5B
    'required': 4_000_000_000,  # $4B (8% minimum)
    'buffer': 1_000_000_000,   # $1B (25% buffer)
    'roe': 0.15  # 15%
}

# Large bank (subject to CCAR):
large_bank = {
    'capital': 200_000_000_000,  # $200B
    'required': 100_000_000_000,  # $100B (8% minimum)
    'ccar_buffer': 50_000_000_000,  # $50B (stress buffer)
    'total_needed': 150_000_000_000,
    'excess': 50_000_000_000,  # $50B (only 33% buffer)
    'roe': 0.10  # 10% (lower due to more capital)
}

# Small banks have ROE advantage (less capital)
# But: Large banks have funding advantage, scale economies
# → Mixed competitive effects
\`\`\`

**Implication 5: Regulatory Relationship**

\`\`\`python
# CCAR creates ongoing regulatory engagement

annual_ccar_workload = {
    'October': 'Receive scenarios, start modeling',
    'November_December': 'Run models, analyze results',
    'January': 'Board approves capital plan, submit to Fed',
    'January_March': 'Fed questions, bank responses',
    'March': 'Submit stress test results',
    'March_June': 'Fed review, more questions',
    'June': 'Fed publishes results, decision',
    'July_September': 'Prepare for next year'
}

# Year-round process
# Hundreds of staff dedicated to CCAR
# Cost: $50-100M+ annually for large banks
\`\`\`

---

**Real-World Example: Capital Plan Objections**

\`\`\`python
# Historical CCAR objections:

objections = {
    '2014': {
        'bank': 'Citigroup',
        'reason': 'Qualitative: Deficiencies in capital planning',
        'quantitative': 'PASSED',
        'result': 'Cannot increase dividends/buybacks'
    },

    '2018': {
        'bank': 'Goldman Sachs',
        'reason': 'Qualitative: Weaknesses in capital planning',
        'quantitative': 'PASSED',
        'result': 'Conditional approval, resubmit plan'
    },

    '2019': {
        'bank': 'Deutsche Bank (U.S. subsidiary)',
        'reason': 'Qualitative: Material weaknesses',
        'quantitative': 'PASSED',
        'result': 'Cannot increase capital distributions'
    }
}

# Lesson: Passing stress test ≠ automatic approval
# Must also have strong processes
# Fed reviews:
# - Risk identification
# - Capital planning process
# - Model validation
# - Governance
# - Data quality
\`\`\`

---

**Strategic Responses by Banks**

**Response 1: Build Capital Buffers**

\`\`\`python
# Banks hold capital well above minimums

pre_ccar_capital_ratio = 0.08  # 8% (minimum)
post_ccar_capital_ratio = 0.12  # 12% (50% buffer)

# Trade-off:
# - Pro: Pass CCAR comfortably
# - Con: Lower ROE, less return to shareholders
\`\`\`

**Response 2: Adjust Business Mix**

\`\`\`python
# Exit capital-intensive businesses

business_decisions = {
    'reduce': [
        'Trading (high stress losses)',
        'Market making (volatile)',
        'Prop trading (risky)'
    ],
    'increase': [
        'Prime lending (stable)',
        'Transaction banking (low risk)',
        'Wealth management (asset-light)'
    ]
}

# CCAR shapes bank strategy
\`\`\`

**Response 3: Improve Risk Management**

\`\`\`python
# Invest in risk capabilities to pass qualitative review

investments = {
    'staff': 'Hire risk managers, modelers',
    'systems': 'Upgrade risk systems, data',
    'processes': 'Document everything',
    'governance': 'Strengthen board oversight',
    'cost': '$50-100M+ annually'
}

# But: Better risk management is good anyway
# CCAR forces discipline
\`\`\`

---

**Summary**

**CCAR / DFAST Framework:**
- **Purpose**: Ensure banks can survive severe recession
- **Scenarios**: Baseline, adverse, severely adverse (9 quarters)
- **Requirements**: Maintain minimum capital ratios throughout stress
- **Consequence**: Cannot return capital if fail

**Regulatory vs Internal:**
- **Regulatory**: Fed scenarios, public, constrains capital
- **Internal**: Bank scenarios, confidential, risk management

**Strategic Implications:**
1. **Capital**: Must hold large buffers (30-50% above minimum)
2. **Returns**: Limited dividends/buybacks (lower ROE)
3. **Business Mix**: Shift away from capital-intensive businesses
4. **Costs**: $50-100M+ annually to comply
5. **Relationships**: Year-round engagement with regulators

**Bottom Line**: CCAR fundamentally changed bank capital management. Banks must operate with large capital buffers, limiting returns but increasing stability. This is the "price" of being systemically important. Small banks have capital advantage but lack scale economies. The framework has made banking system more resilient but less profitable.`,
    },
  ],
} as const;
