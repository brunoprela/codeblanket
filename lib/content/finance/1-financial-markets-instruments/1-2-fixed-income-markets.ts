export const fixedIncomeMarkets = {
  title: 'Fixed Income Markets',
  slug: 'fixed-income-markets',
  description:
    'Master bond markets, pricing, and yield curve analysis for building fixed income systems',
  content: `
# Fixed Income Markets

## Introduction: Why Fixed Income Matters

Fixed income markets dwarf equity markets in size - the global bond market is over **$130 trillion** (vs ~$100 trillion for equities). Yet most developers focus only on stocks. Understanding bonds is critical for:

- **Building diversified portfolio systems** (60/40 stock/bond portfolios)
- **Risk management** (bonds hedge stock crashes)
- **Interest rate modeling** (affects everything in finance)
- **Corporate finance** (companies issue bonds to raise capital)
- **Government finance** (treasuries fund governments)

**What you'll learn:**
- How bonds work and why they exist
- Bond pricing mathematics
- Yield curves and term structure
- Credit risk and ratings
- Building bond analytics systems

---

## What is a Bond?

A bond is a **loan** you make to a borrower (government or corporation). In exchange:
- You receive **periodic interest payments** (coupons)
- You get your **principal back** at maturity

\`\`\`python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List
import numpy as np

@dataclass
class Bond:
    """
    Representation of a fixed-rate bond
    """
    issuer: str
    face_value: float  # Par value (usually $1,000 or $100)
    coupon_rate: float  # Annual coupon rate (e.g., 0.05 = 5%)
    maturity_date: datetime
    issue_date: datetime
    payment_frequency: int = 2  # Semi-annual = 2, Annual = 1, Quarterly = 4
    
    def get_coupon_payment(self) -> float:
        """Calculate periodic coupon payment"""
        annual_coupon = self.face_value * self.coupon_rate
        return annual_coupon / self.payment_frequency
    
    def get_cash_flows(self) -> List[dict]:
        """
        Generate all future cash flows
        Returns list of {date, amount}
        """
        cash_flows = []
        current_date = self.issue_date
        coupon_payment = self.get_coupon_payment()
        
        # Calculate payment dates
        months_between_payments = 12 // self.payment_frequency
        
        while current_date < self.maturity_date:
            current_date += timedelta(days=30 * months_between_payments)
            if current_date <= self.maturity_date:
                cash_flows.append({
                    'date': current_date,
                    'amount': coupon_payment,
                    'type': 'coupon'
                })
        
        # Final payment includes principal
        cash_flows[-1]['amount'] += self.face_value
        cash_flows[-1]['type'] = 'coupon + principal'
        
        return cash_flows
    
    def years_to_maturity(self, valuation_date: datetime = None) -> float:
        """Calculate years remaining until maturity"""
        if valuation_date is None:
            valuation_date = datetime.now()
        
        days_remaining = (self.maturity_date - valuation_date).days
        return days_remaining / 365.25

# Example: U.S. Treasury Note
treasury_note = Bond(
    issuer="U.S. Treasury",
    face_value=1000,
    coupon_rate=0.04,  # 4% annual coupon
    maturity_date=datetime(2034, 5, 15),
    issue_date=datetime(2024, 5, 15),
    payment_frequency=2  # Semi-annual
)

print(f"Bond: {treasury_note.issuer}")
print(f"Face Value: \${treasury_note.face_value:, .0f
}")
print(f"Coupon Rate: {treasury_note.coupon_rate*100}%")
print(f"Coupon Payment (semi-annual): \${treasury_note.get_coupon_payment():.2f}")
print(f"Years to Maturity: {treasury_note.years_to_maturity():.2f}")

print("\\nCash Flows:")
for i, cf in enumerate(treasury_note.get_cash_flows()[: 5], 1):
    print(f"Payment {i}: {cf['date'].strftime('%Y-%m-%d')} - \${cf['amount']:.2f} ({cf['type']})")
\`\`\`

**Key Terms:**
- **Face Value (Par)**: Amount paid at maturity (typically $1,000)
- **Coupon Rate**: Annual interest rate
- **Coupon Payment**: Periodic interest payment
- **Maturity**: When the bond expires and principal is returned
- **Yield**: The return you actually earn (different from coupon rate!)

---

## Types of Bonds

### 1. Government Bonds (Sovereigns)

**U.S. Treasury Securities:**
\`\`\`python
class TreasuryBond(Bond):
    """
    U.S. Treasury bonds are considered risk-free
    (backed by full faith and credit of U.S. government)
    """
    
    def __init__(self, **kwargs):
        super().__init__(issuer="U.S. Treasury", **kwargs)
        self.credit_rating = "AAA"
        self.default_risk = 0.0  # Assumed risk-free
    
    @classmethod
    def create_t_bill(cls, face_value: float, days_to_maturity: int) -> 'TreasuryBond':
        """
        Treasury Bills (T-Bills): < 1 year maturity
        Sold at discount, no coupon payments
        """
        return cls(
            face_value=face_value,
            coupon_rate=0.0,  # Zero-coupon
            maturity_date=datetime.now() + timedelta(days=days_to_maturity),
            issue_date=datetime.now(),
            payment_frequency=0
        )
    
    @classmethod
    def create_t_note(cls, face_value: float, years: int, coupon_rate: float) -> 'TreasuryBond':
        """
        Treasury Notes (T-Notes): 2-10 year maturity
        Pay semi-annual coupons
        """
        return cls(
            face_value=face_value,
            coupon_rate=coupon_rate,
            maturity_date=datetime.now() + timedelta(days=365*years),
            issue_date=datetime.now(),
            payment_frequency=2
        )
    
    @classmethod
    def create_t_bond(cls, face_value: float, years: int, coupon_rate: float) -> 'TreasuryBond':
        """
        Treasury Bonds (T-Bonds): 20-30 year maturity
        Pay semi-annual coupons
        """
        return cls(
            face_value=face_value,
            coupon_rate=coupon_rate,
            maturity_date=datetime.now() + timedelta(days=365*years),
            issue_date=datetime.now(),
            payment_frequency=2
        )

# Create different Treasury securities
t_bill = TreasuryBond.create_t_bill(face_value=10000, days_to_maturity=91)  # 3-month
t_note = TreasuryBond.create_t_note(face_value=1000, years=5, coupon_rate=0.04)
t_bond = TreasuryBond.create_t_bond(face_value=1000, years=30, coupon_rate=0.045)

print("Treasury Securities:")
print(f"T-Bill (91 days): \${t_bill.face_value}")
print(f"T-Note (5 year): \${t_note.face_value}, {t_note.coupon_rate*100}% coupon")
print(f"T-Bond (30 year): \${t_bond.face_value}, {t_bond.coupon_rate*100}% coupon")
\`\`\`

### 2. Corporate Bonds

\`\`\`python
class CorporateBond(Bond):
    """
    Corporate bonds have credit risk (company might default)
    Higher risk = higher yield
    """
    
    def __init__(self, credit_rating: str, **kwargs):
        super().__init__(**kwargs)
        self.credit_rating = credit_rating
        self.default_probability = self.estimate_default_probability()
    
    def estimate_default_probability(self) -> float:
        """
        Estimate default probability based on credit rating
        Based on historical default rates
        """
        default_rates = {
            'AAA': 0.0001,  # 0.01%
            'AA': 0.0005,   # 0.05%
            'A': 0.001,     # 0.1%
            'BBB': 0.005,   # 0.5% (lowest investment grade)
            'BB': 0.02,     # 2% (junk)
            'B': 0.05,      # 5%
            'CCC': 0.15,    # 15%
            'CC': 0.30,     # 30%
            'C': 0.50,      # 50%
            'D': 1.0        # Default
        }
        return default_rates.get(self.credit_rating, 0.10)
    
    def calculate_credit_spread(self, risk_free_yield: float) -> float:
        """
        Credit spread = Extra yield demanded due to default risk
        Corporate Yield = Risk-free Yield + Credit Spread
        """
        # Simplified model: spread proportional to default probability
        spread = self.default_probability * 5  # ~5x default probability
        return spread
    
    def required_yield(self, risk_free_yield: float) -> float:
        """Calculate required yield for this corporate bond"""
        credit_spread = self.calculate_credit_spread(risk_free_yield)
        return risk_free_yield + credit_spread

# Example: Apple corporate bond
apple_bond = CorporateBond(
    issuer="Apple Inc.",
    face_value=1000,
    coupon_rate=0.035,
    maturity_date=datetime(2031, 5, 15),
    issue_date=datetime(2024, 5, 15),
    payment_frequency=2,
    credit_rating="AA+"
)

risk_free_rate = 0.04  # 10-year Treasury at 4%
required_yield = apple_bond.required_yield(risk_free_rate)

print(f"Apple Corporate Bond:")
print(f"Credit Rating: {apple_bond.credit_rating}")
print(f"Default Probability: {apple_bond.default_probability*100:.3f}%")
print(f"Risk-free Rate: {risk_free_rate*100}%")
print(f"Credit Spread: {apple_bond.calculate_credit_spread(risk_free_rate)*100:.2f}%")
print(f"Required Yield: {required_yield*100:.2f}%")
\`\`\`

### 3. Municipal Bonds (Munis)

\`\`\`python
class MunicipalBond(Bond):
    """
    Municipal bonds issued by states/cities
    Key feature: TAX-EXEMPT interest for federal taxes
    """
    
    def __init__(self, tax_exempt: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.tax_exempt = tax_exempt
    
    def tax_equivalent_yield(self, nominal_yield: float, tax_rate: float) -> float:
        """
        Calculate what taxable yield would be equivalent to tax-free muni yield
        TEY = Tax-free Yield / (1 - Tax Rate)
        """
        if self.tax_exempt:
            return nominal_yield / (1 - tax_rate)
        return nominal_yield
    
    def compare_to_taxable(self, taxable_yield: float, investor_tax_rate: float) -> dict:
        """Compare muni bond to taxable alternative"""
        muni_yield = self.coupon_rate
        tey = self.tax_equivalent_yield(muni_yield, investor_tax_rate)
        
        return {
            'muni_yield': muni_yield,
            'taxable_equivalent_yield': tey,
            'taxable_alternative_yield': taxable_yield,
            'better_choice': 'Municipal' if tey > taxable_yield else 'Taxable',
            'advantage': abs(tey - taxable_yield)
        }

# Example: California muni bond
ca_muni = MunicipalBond(
    issuer="State of California",
    face_value=5000,
    coupon_rate=0.03,  # 3% tax-free
    maturity_date=datetime(2044, 1, 1),
    issue_date=datetime(2024, 1, 1),
    payment_frequency=2,
    tax_exempt=True
)

# Compare to taxable corporate bond for high-income investor
investor_tax_rate = 0.37  # 37% federal tax bracket
corporate_yield = 0.045   # 4.5% taxable

comparison = ca_muni.compare_to_taxable(corporate_yield, investor_tax_rate)

print("Municipal Bond vs Corporate Bond:")
print(f"Muni Yield: {comparison['muni_yield']*100}% (tax-free)")
print(f"Tax-Equivalent Yield: {comparison['taxable_equivalent_yield']*100:.2f}%")
print(f"Corporate Yield: {comparison['taxable_alternative_yield']*100}%")
print(f"Better Choice: {comparison['better_choice']}")
\`\`\`

### 4. High-Yield (Junk) Bonds

\`\`\`python
class HighYieldBond(CorporateBond):
    """
    Junk bonds: Below investment grade (< BBB)
    Higher risk, higher return potential
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.credit_rating in ['AAA', 'AA', 'A', 'BBB']:
            raise ValueError("High-yield bonds must be below BBB rating")
    
    def is_distressed(self) -> bool:
        """Check if bond is in distress (CCC or lower)"""
        return self.credit_rating in ['CCC', 'CC', 'C', 'D']
    
    def estimate_recovery_rate(self) -> float:
        """
        If bond defaults, what % of face value do you recover?
        Senior secured bonds recover more than subordinated
        """
        # Simplified model
        recovery_rates = {
            'BB': 0.60,   # 60% recovery
            'B': 0.50,    # 50% recovery
            'CCC': 0.30,  # 30% recovery
            'CC': 0.20,
            'C': 0.10,
            'D': 0.05
        }
        return recovery_rates.get(self.credit_rating, 0.40)

# Example: Struggling retail company
junk_bond = HighYieldBond(
    issuer="Struggling Retail Corp",
    face_value=1000,
    coupon_rate=0.10,  # 10% coupon (high!)
    maturity_date=datetime(2027, 12, 31),
    issue_date=datetime(2024, 1, 1),
    payment_frequency=2,
    credit_rating="B"
)

print(f"High-Yield Bond Analysis:")
print(f"Issuer: {junk_bond.issuer}")
print(f"Rating: {junk_bond.credit_rating} (Junk)")
print(f"Coupon: {junk_bond.coupon_rate*100}%")
print(f"Default Probability: {junk_bond.default_probability*100}%")
print(f"Expected Recovery: {junk_bond.estimate_recovery_rate()*100}%")
print(f"Is Distressed: {junk_bond.is_distressed()}")

# Expected loss from default
expected_loss = junk_bond.default_probability * (1 - junk_bond.estimate_recovery_rate())
print(f"Expected Loss: {expected_loss*100:.2f}%")
\`\`\`

---

## Bond Pricing Fundamentals

**Core Principle**: A bond is worth the **present value** of all future cash flows.

\`\`\`python
class BondPricer:
    """
    Price bonds using present value mathematics
    """
    
    @staticmethod
    def price_bond(bond: Bond, 
                   yield_to_maturity: float,
                   valuation_date: datetime = None) -> dict:
        """
        Price bond using YTM (discount rate)
        
        Price = PV of coupons + PV of principal
        
        Args:
            bond: Bond object
            yield_to_maturity: Market yield (discount rate)
            valuation_date: Date to value bond
            
        Returns:
            Dictionary with price and analytics
        """
        if valuation_date is None:
            valuation_date = datetime.now()
        
        cash_flows = bond.get_cash_flows()
        periods_per_year = bond.payment_frequency
        ytm_per_period = yield_to_maturity / periods_per_year
        
        total_pv = 0
        coupon_pv = 0
        
        for i, cf in enumerate(cash_flows, 1):
            # Days from valuation to cash flow
            days_to_cf = (cf['date'] - valuation_date).days
            periods_to_cf = days_to_cf / (365.25 / periods_per_year)
            
            # Present value
            discount_factor = (1 + ytm_per_period) ** periods_to_cf
            pv = cf['amount'] / discount_factor
            
            total_pv += pv
            
            # Track coupon PV separately
            if cf['type'] == 'coupon':
                coupon_pv += pv
        
        principal_pv = total_pv - coupon_pv
        
        # Premium/Discount/Par
        if total_pv > bond.face_value:
            price_type = "Premium"
        elif total_pv < bond.face_value:
            price_type = "Discount"
        else:
            price_type = "Par"
        
        return {
            'price': round(total_pv, 2),
            'coupon_pv': round(coupon_pv, 2),
            'principal_pv': round(principal_pv, 2),
            'face_value': bond.face_value,
            'price_type': price_type,
            'price_as_percent_of_par': round((total_pv / bond.face_value) * 100, 3),
            'yield_to_maturity': yield_to_maturity
        }
    
    @staticmethod
    def calculate_ytm(bond: Bond, 
                     market_price: float,
                     valuation_date: datetime = None) -> float:
        """
        Calculate yield to maturity given market price
        Uses Newton-Raphson method (iterative)
        
        This is the INVERSE problem: Given price, find yield
        """
        def price_difference(ytm: float) -> float:
            """Function to minimize"""
            calculated_price = BondPricer.price_bond(bond, ytm, valuation_date)['price']
            return calculated_price - market_price
        
        # Initial guess
        ytm_guess = bond.coupon_rate
        
        # Newton-Raphson iteration
        for _ in range(100):
            price_at_guess = BondPricer.price_bond(bond, ytm_guess, valuation_date)['price']
            
            # Small change in yield
            delta_ytm = 0.0001
            price_at_guess_plus = BondPricer.price_bond(
                bond, ytm_guess + delta_ytm, valuation_date
            )['price']
            
            # Derivative (slope)
            derivative = (price_at_guess_plus - price_at_guess) / delta_ytm
            
            # Newton-Raphson update
            ytm_new = ytm_guess - (price_at_guess - market_price) / derivative
            
            # Check convergence
            if abs(ytm_new - ytm_guess) < 0.000001:
                return ytm_new
            
            ytm_guess = ytm_new
        
        return ytm_guess

# Example: Price Treasury note at different yields
t_note = TreasuryBond.create_t_note(face_value=1000, years=10, coupon_rate=0.04)

print("Bond Pricing at Different Yields:\\n")

for ytm in [0.03, 0.04, 0.05]:
    result = BondPricer.price_bond(t_note, ytm)
    print(f"YTM = {ytm*100}%:")
    print(f"  Price: \${result['price']: .2f} ({ result['price_as_percent_of_par']: .2f } % of par) ")
print(f"  Type: {result['price_type']}")
print()

# Key insight: Inverse relationship between price and yield
print("KEY INSIGHT:")
print("When yield ↑ → price ↓")
print("When yield ↓ → price ↑")
print("\\nWhy? Higher discount rate → lower PV of future cash flows")

# Calculate YTM from market price
market_price = 950  # Bond trading at discount
calculated_ytm = BondPricer.calculate_ytm(t_note, market_price)
print(f"\\nIf bond trades at \${market_price}, YTM = {calculated_ytm*100:.3f}%")
\`\`\`

**Critical Relationships:**

1. **Coupon Rate vs YTM**:
   - If YTM > Coupon Rate → Bond trades at **Discount** (< par)
   - If YTM < Coupon Rate → Bond trades at **Premium** (> par)
   - If YTM = Coupon Rate → Bond trades at **Par**

2. **Price and Yield move INVERSELY**
   - Yields rise → Prices fall
   - Yields fall → Prices rise

3. **Duration and Interest Rate Risk**
   - Longer maturity → More sensitive to rate changes
   - Lower coupon → More sensitive to rate changes

---

## Yield Curve and Term Structure

The **yield curve** plots yields of bonds against their maturity. It's one of the most important charts in finance!

\`\`\`python
import matplotlib.pyplot as plt
import pandas as pd

class YieldCurve:
    """
    Model and analyze yield curves
    """
    
    def __init__(self, rates: dict[float, float]):
        """
        Args:
            rates: {maturity_years: yield}
            e.g., {0.25: 0.045, 1: 0.047, 2: 0.048, ...}
        """
        self.rates = dict(sorted(rates.items()))
        self.maturities = list(self.rates.keys())
        self.yields = list(self.rates.values())
    
    def interpolate_rate(self, maturity: float) -> float:
        """Interpolate yield for any maturity"""
        return np.interp(maturity, self.maturities, self.yields)
    
    def get_curve_shape(self) -> str:
        """Classify yield curve shape"""
        short_rate = self.rates[min(self.maturities)]
        long_rate = self.rates[max(self.maturities)]
        mid_rate = self.interpolate_rate(5.0)
        
        if long_rate > short_rate + 0.005:  # 50 bps steeper
            if mid_rate > short_rate + 0.002:
                return "Normal (Upward Sloping)"
            else:
                return "Humped"
        elif long_rate < short_rate - 0.005:
            return "Inverted (Recession Signal!)"
        else:
            return "Flat"
    
    def calculate_forward_rates(self) -> dict[tuple[float, float], float]:
        """
        Calculate implied forward rates
        Forward rate = Interest rate for future period implied by spot rates
        
        (1 + s2)^2 = (1 + s1) × (1 + f1,2)
        """
        forward_rates = {}
        
        for i in range(len(self.maturities) - 1):
            t1 = self.maturities[i]
            t2 = self.maturities[i + 1]
            s1 = self.yields[i]
            s2 = self.yields[i + 1]
            
            # Calculate forward rate
            forward = ((1 + s2) ** t2 / (1 + s1) ** t1) ** (1 / (t2 - t1)) - 1
            
            forward_rates[(t1, t2)] = forward
        
        return forward_rates
    
    def plot_curve(self):
        """Visualize the yield curve"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.maturities, [y * 100 for y in self.yields], 
                marker='o', linewidth=2, markersize=8)
        plt.xlabel('Maturity (Years)', fontsize=12)
        plt.ylabel('Yield (%)', fontsize=12)
        plt.title(f'Yield Curve - {self.get_curve_shape()}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt

# Example: Normal yield curve (typical)
normal_curve = YieldCurve({
    0.25: 0.050,  # 3-month
    0.5: 0.048,   # 6-month
    1.0: 0.045,   # 1-year
    2.0: 0.042,   # 2-year
    5.0: 0.040,   # 5-year
    10.0: 0.038,  # 10-year
    30.0: 0.042   # 30-year
})

print("Yield Curve Analysis:")
print(f"Shape: {normal_curve.get_curve_shape()}")
print(f"2-year yield: {normal_curve.interpolate_rate(2)*100:.2f}%")
print(f"7-year yield: {normal_curve.interpolate_rate(7)*100:.2f}%")

print("\\nForward Rates (implied future rates):")
forward_rates = normal_curve.calculate_forward_rates()
for (t1, t2), rate in list(forward_rates.items())[:3]:
    print(f"  {t1}y → {t2}y: {rate*100:.3f}%")

# Inverted yield curve (recession signal)
inverted_curve = YieldCurve({
    0.25: 0.052,
    0.5: 0.050,
    1.0: 0.048,
    2.0: 0.044,
    5.0: 0.038,
    10.0: 0.035,
    30.0: 0.037
})

print(f"\\nInverted Curve Shape: {inverted_curve.get_curve_shape()}")
print("⚠️  This often predicts recession within 12-18 months!")
\`\`\`

**Yield Curve Shapes and Their Meaning:**

1. **Normal (Upward Sloping)**: Healthy economy, higher rates for longer maturities
2. **Inverted**: Recession predictor! Short rates > long rates
3. **Flat**: Economic uncertainty, transition period
4. **Humped**: Short and long rates low, medium rates high

---

## Credit Ratings and Risk

\`\`\`python
class CreditRatingAnalyzer:
    """
    Analyze credit ratings and their implications
    """
    
    RATING_SCALE = {
        # Investment Grade
        'AAA': {'score': 100, 'grade': 'Investment', 'description': 'Highest quality'},
        'AA+': {'score': 95, 'grade': 'Investment', 'description': 'Very high quality'},
        'AA': {'score': 90, 'grade': 'Investment', 'description': 'Very high quality'},
        'AA-': {'score': 85, 'grade': 'Investment', 'description': 'High quality'},
        'A+': {'score': 80, 'grade': 'Investment', 'description': 'High quality'},
        'A': {'score': 75, 'grade': 'Investment', 'description': 'Strong'},
        'A-': {'score': 70, 'grade': 'Investment', 'description': 'Strong'},
        'BBB+': {'score': 65, 'grade': 'Investment', 'description': 'Adequate'},
        'BBB': {'score': 60, 'grade': 'Investment', 'description': 'Adequate'},
        'BBB-': {'score': 55, 'grade': 'Investment', 'description': 'Lowest investment grade'},
        
        # Junk / High Yield
        'BB+': {'score': 50, 'grade': 'Junk', 'description': 'Speculative'},
        'BB': {'score': 45, 'grade': 'Junk', 'description': 'Speculative'},
        'BB-': {'score': 40, 'grade': 'Junk', 'description': 'Speculative'},
        'B+': {'score': 35, 'grade': 'Junk', 'description': 'Highly speculative'},
        'B': {'score': 30, 'grade': 'Junk', 'description': 'Highly speculative'},
        'B-': {'score': 25, 'grade': 'Junk', 'description': 'Substantial risks'},
        'CCC+': {'score': 20, 'grade': 'Junk', 'description': 'Extremely speculative'},
        'CCC': {'score': 15, 'grade': 'Junk', 'description': 'Default imminent'},
        'CC': {'score': 10, 'grade': 'Junk', 'description': 'Highly vulnerable'},
        'C': {'score': 5, 'grade': 'Junk', 'description': 'Near default'},
        'D': {'score': 0, 'grade': 'Default', 'description': 'In default'}
    }
    
    @classmethod
    def get_rating_info(cls, rating: str) -> dict:
        """Get detailed information about a credit rating"""
        return cls.RATING_SCALE.get(rating, {'score': 0, 'grade': 'Unknown', 'description': 'N/A'})
    
    @classmethod
    def is_investment_grade(cls, rating: str) -> bool:
        """Check if rating is investment grade"""
        info = cls.get_rating_info(rating)
        return info['grade'] == 'Investment'
    
    @classmethod
    def compare_ratings(cls, rating1: str, rating2: str) -> dict:
        """Compare two credit ratings"""
        info1 = cls.get_rating_info(rating1)
        info2 = cls.get_rating_info(rating2)
        
        return {
            'rating1': rating1,
            'rating2': rating2,
            'better_rating': rating1 if info1['score'] > info2['score'] else rating2,
            'score_difference': abs(info1['score'] - info2['score']),
            'both_investment_grade': (info1['grade'] == 'Investment' and 
                                     info2['grade'] == 'Investment')
        }

# Examples
print("Credit Rating Analysis:\\n")

for rating in ['AAA', 'BBB-', 'BB+', 'B', 'CCC']:
    info = CreditRatingAnalyzer.get_rating_info(rating)
    print(f"{rating}: {info['description']} ({info['grade']} grade)")

print("\\n" + "="*50)
print("⚠️  BBB- is the CLIFF!")
print("Fall below BBB- = You're JUNK")
print("Many institutional investors CANNOT own junk bonds")
print("='Fallen Angel' = Investment grade → Junk")
print("="*50)
\`\`\`

---

## Real-World Example: U.S. Treasury Market

\`\`\`python
def analyze_treasury_market():
    """
    Analyze the U.S. Treasury market
    Largest, most liquid bond market in the world
    """
    
    market_overview = {
        'total_outstanding': 26_000_000_000_000,  # $26 trillion
        'daily_volume': 650_000_000_000,  # $650 billion/day
        'largest_holders': {
            'Federal Reserve': 5_000_000_000_000,
            'Foreign governments': 7_500_000_000_000,
            'U.S. mutual funds': 3_500_000_000_000,
            'U.S. banks': 1_200_000_000_000,
            'Pension funds': 2_000_000_000_000,
        },
        'benchmark_role': 'Risk-free rate for all finance'
    }
    
    print("U.S. Treasury Market Overview:\\n")
    print(f"Total Outstanding: \${market_overview['total_outstanding'] / 1e12: .1f} trillion")
print(f"Daily Trading Volume: \${market_overview['daily_volume']/1e9:.0f} billion")
print(f"Role: {market_overview['benchmark_role']}")

print("\\nTop Holders:")
for holder, amount in market_overview['largest_holders'].items():
    print(f"  {holder}: \${amount/1e12:.2f}T")

print("\\nWhy Treasuries Matter:")
print("✓ Risk-free rate (foundation of all pricing)")
print("✓ Safe haven in crises")
print("✓ Most liquid securities in the world")
print("✓ Used for collateral in derivatives")
print("✓ Fed policy tool (QE = buying treasuries)")

analyze_treasury_market()
\`\`\`

---

## Building a Bond Analytics System

\`\`\`python
class BondAnalyticsPlatform:
    """
    Production-ready bond analytics system
    """
    
    def __init__(self):
        self.bonds_database = {}
        self.yield_curve = None
    
    def add_bond(self, bond_id: str, bond: Bond):
        """Add bond to tracking database"""
        self.bonds_database[bond_id] = bond
    
    def update_yield_curve(self, curve: YieldCurve):
        """Update current yield curve"""
        self.yield_curve = curve
    
    def analyze_bond(self, bond_id: str, market_price: float = None) -> dict:
        """
        Comprehensive bond analysis
        """
        bond = self.bonds_database.get(bond_id)
        if not bond:
            return {'error': 'Bond not found'}
        
        # Price bond at current yield curve
        maturity_years = bond.years_to_maturity()
        market_yield = self.yield_curve.interpolate_rate(maturity_years)
        
        pricing = BondPricer.price_bond(bond, market_yield)
        
        # If market price provided, calculate YTM
        if market_price:
            ytm = BondPricer.calculate_ytm(bond, market_price)
            ytm_spread = ytm - market_yield
        else:
            ytm = market_yield
            ytm_spread = 0
        
        return {
            'bond_id': bond_id,
            'issuer': bond.issuer,
            'maturity_years': round(maturity_years, 2),
            'coupon_rate': bond.coupon_rate,
            'market_yield': market_yield,
            'fair_value': pricing['price'],
            'market_price': market_price,
            'ytm': ytm,
            'ytm_spread': ytm_spread,
            'is_cheap': market_price < pricing['price'] if market_price else None,
            'annual_income': bond.get_coupon_payment() * bond.payment_frequency
        }
    
    def screen_bonds(self, 
                    min_yield: float = None,
                    max_maturity: float = None,
                    min_rating_score: int = None) -> List[dict]:
        """
        Screen bonds based on criteria
        """
        results = []
        
        for bond_id, bond in self.bonds_database.items():
            # Calculate metrics
            maturity = bond.years_to_maturity()
            ytm = self.yield_curve.interpolate_rate(maturity)
            
            # Apply filters
            if min_yield and ytm < min_yield:
                continue
            if max_maturity and maturity > max_maturity:
                continue
            
            # If corporate bond, check rating
            if isinstance(bond, CorporateBond) and min_rating_score:
                rating_info = CreditRatingAnalyzer.get_rating_info(bond.credit_rating)
                if rating_info['score'] < min_rating_score:
                    continue
            
            results.append({
                'bond_id': bond_id,
                'issuer': bond.issuer,
                'ytm': ytm,
                'maturity': maturity,
                'rating': bond.credit_rating if isinstance(bond, CorporateBond) else 'AAA'
            })
        
        return sorted(results, key=lambda x: x['ytm'], reverse=True)

# Example usage
platform = BondAnalyticsPlatform()

# Add bonds
platform.add_bond('UST-10Y', TreasuryBond.create_t_note(1000, 10, 0.04))
platform.add_bond('AAPL-2031', CorporateBond(
    issuer='Apple', face_value=1000, coupon_rate=0.035,
    maturity_date=datetime(2031,1,1), issue_date=datetime(2024,1,1),
    payment_frequency=2, credit_rating='AA+'
))

# Set yield curve
platform.update_yield_curve(normal_curve)

# Analyze specific bond
analysis = platform.analyze_bond('UST-10Y', market_price=980)
print("Bond Analysis:")
print(f"Fair Value: \${analysis['fair_value']: .2f}")
print(f"Market Price: \${analysis['market_price']:.2f}")
print(f"YTM: {analysis['ytm']*100:.3f}%")
print(f"Is Cheap: {analysis['is_cheap']}")

# Screen bonds
results = platform.screen_bonds(min_yield = 0.035, max_maturity = 15)
print(f"\\nFound {len(results)} bonds matching criteria")
\`\`\`

---

## Summary

**Key Takeaways:**

1. **Bonds are loans** - You lend money, receive coupons, get principal back
2. **Government bonds** - Risk-free (Treasuries), benchmark for all rates
3. **Corporate bonds** - Credit risk, rated by agencies, higher yield
4. **Municipal bonds** - Tax-advantaged for high-income investors
5. **Bond pricing** - Present value of cash flows, inverse to yields
6. **Yield curve** - Critical indicator of economic health
7. **Credit ratings** - BBB- is the cliff (investment vs junk)

**For Engineers:**
- Fixed income is HUGE ($130T+ market)
- Every portfolio system needs bond support
- Interest rates affect EVERYTHING in finance
- Yield curve predicts recessions
- Bond math is pure present value calculations

**Next Steps:**
- Build bond pricing calculator
- Implement yield curve analyzer
- Create credit spread monitor
- Study Module 7 for deeper derivatives pricing

You now understand fixed income markets - ready to build bond analytics systems!
`,
  exercises: [
    {
      prompt:
        'Build a bond portfolio optimizer that maximizes yield while maintaining a target duration and credit quality. Include constraints for sector diversification and issue concentration limits.',
      solution:
        '// Implementation: 1) Fetch universe of bonds with prices/yields, 2) Calculate duration for each bond, 3) Set up optimization problem (maximize yield), 4) Add constraints (target duration ±0.5 years, min 80% investment grade, max 5% in single issuer, sector limits), 5) Use cvxpy or scipy.optimize, 6) Output optimal portfolio weights',
    },
    {
      prompt:
        'Create a yield curve construction system that bootstraps spot rates from bond prices, handling multiple bonds per maturity. Implement smoothing (e.g., Nelson-Siegel model) and generate forward rate curves.',
      solution:
        '// Implementation: 1) Collect bond prices and cash flows for multiple bonds, 2) Bootstrap short end (T-bills) for spot rates, 3) Iterate longer maturities solving for spot rates, 4) Fit Nelson-Siegel or cubic spline for smooth curve, 5) Calculate forward rates from spot rates, 6) Validate curve is arbitrage-free, 7) Visualize with confidence intervals',
    },
  ],
};
