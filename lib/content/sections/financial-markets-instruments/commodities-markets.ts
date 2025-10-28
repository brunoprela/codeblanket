export const commoditiesMarkets = {
  title: 'Commodities Markets',
  slug: 'commodities-markets',
  description:
    'Master physical commodity trading from oil to gold - futures, storage, and real-world delivery',
  content: `
# Commodities Markets

## Introduction: Trading Physical Goods

Commodities are **raw materials** that are essential inputs for the global economy:
- 🛢️ **Energy**: Oil, natural gas, gasoline, coal
- 🥇 **Precious Metals**: Gold, silver, platinum, palladium  
- 🔧 **Industrial Metals**: Copper, aluminum, zinc, nickel
- 🌾 **Agriculture**: Wheat, corn, soybeans, coffee, sugar, cotton
- 🥩 **Livestock**: Live cattle, lean hogs

Unlike stocks or bonds, commodities are **physical goods** that:
- Must be stored (storage costs matter)
- Can deteriorate (agricultural products spoil)
- Have quality grades (light sweet crude ≠ heavy sour crude)
- Require physical delivery (or cash settlement)

**Market Size**: ~$20 trillion annually in physical commodity trade, with futures markets providing price discovery and hedging for producers and consumers worldwide.

**What you'll learn:**
- Commodity futures mechanics
- Contango vs backwardation
- Storage and convenience yield
- Hedging for producers (airlines, farmers, miners)
- Building commodity trading systems
- The oil price crash of 2020 (negative prices!)

---

## Types of Commodities

\`\`\`python
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from typing import Optional, Literal
import numpy as np

class CommodityCategory(Enum):
    ENERGY = "Energy"
    PRECIOUS_METALS = "Precious Metals"
    INDUSTRIAL_METALS = "Industrial Metals"
    AGRICULTURE = "Agriculture"
    LIVESTOCK = "Livestock"

@dataclass
class Commodity:
    """
    Representation of a commodity
    """
    name: str
    category: CommodityCategory
    unit: str  # Barrel, ounce, bushel, etc.
    contract_size: int  # Standard futures contract size
    tick_size: float  # Minimum price move
    tick_value: float  # Dollar value of one tick
    storage_cost_annual: float  # % of value
    typical_volatility: float  # Annual volatility
    primary_exchange: str
    
    def calculate_position_value (self, 
                                 price: float,
                                 num_contracts: int = 1) -> float:
        """Calculate notional value of position"""
        return price * self.contract_size * num_contracts
    
    def calculate_storage_cost (self,
                              price: float,
                              holding_period_days: int) -> float:
        """
        Calculate storage cost for holding physical commodity
        
        Critical for understanding futures pricing!
        """
        years = holding_period_days / 365
        total_cost = price * self.storage_cost_annual * years
        return total_cost

# Define major commodities
COMMODITIES = {
    'CL': Commodity(
        name="Crude Oil (WTI)",
        category=CommodityCategory.ENERGY,
        unit="barrel",
        contract_size=1000,  # 1,000 barrels per contract
        tick_size=0.01,  # $0.01 per barrel
        tick_value=10.0,  # $10 per tick (0.01 × 1000)
        storage_cost_annual=0.02,  # 2% of value to store
        typical_volatility=0.35,  # 35% annual volatility
        primary_exchange="NYMEX"
    ),
    'GC': Commodity(
        name="Gold",
        category=CommodityCategory.PRECIOUS_METALS,
        unit="troy ounce",
        contract_size=100,  # 100 troy ounces
        tick_size=0.10,
        tick_value=10.0,
        storage_cost_annual=0.003,  # 0.3% (gold is cheap to store)
        typical_volatility=0.15,  # 15% volatility
        primary_exchange="COMEX"
    ),
    'NG': Commodity(
        name="Natural Gas",
        category=CommodityCategory.ENERGY,
        unit="MMBtu",  # Million British Thermal Units
        contract_size=10000,  # 10,000 MMBtu
        tick_size=0.001,
        tick_value=10.0,
        storage_cost_annual=0.15,  # 15% (expensive to store gas)
        typical_volatility=0.50,  # 50% volatility (very volatile!)
        primary_exchange="NYMEX"
    ),
    'ZC': Commodity(
        name="Corn",
        category=CommodityCategory.AGRICULTURE,
        unit="bushel",
        contract_size=5000,  # 5,000 bushels
        tick_size=0.0025,  # Quarter cent per bushel
        tick_value=12.50,
        storage_cost_annual=0.10,  # 10% (silos, spoilage risk)
        typical_volatility=0.25,
        primary_exchange="CBOT"
    ),
    'HG': Commodity(
        name="Copper",
        category=CommodityCategory.INDUSTRIAL_METALS,
        unit="pound",
        contract_size=25000,  # 25,000 pounds
        tick_size=0.0005,
        tick_value=12.50,
        storage_cost_annual=0.05,  # 5%
        typical_volatility=0.28,
        primary_exchange="COMEX"
    )
}

# Display commodity information
print("=== Major Commodity Futures ===\\n")

for symbol, commodity in COMMODITIES.items():
    print(f"{symbol}: {commodity.name}")
    print(f"  Category: {commodity.category.value}")
    print(f"  Contract: {commodity.contract_size} {commodity.unit}")
    print(f"  Exchange: {commodity.primary_exchange}")
    print(f"  Storage Cost: {commodity.storage_cost_annual*100:.1f}% annually")
    print(f"  Volatility: {commodity.typical_volatility*100:.0f}%\\n")

# Example: Oil futures position
oil = COMMODITIES['CL']
oil_price = 75  # $75/barrel

position_value = oil.calculate_position_value (oil_price, num_contracts=10)
print(f"Example: Long 10 WTI Crude contracts at \\$\{oil_price}/barrel")
print(f"Position Value: \\$\{position_value:,}")
print(f"(10 contracts × 1,000 barrels × \\$\{oil_price})")

# Storage cost for 6 months
storage = oil.calculate_storage_cost (oil_price, holding_period_days = 180)
print(f"\\nStorage cost for 6 months: \\$\{storage:.2f} per barrel")
print(f"Total storage: \\$\{storage * oil.contract_size * 10:,.0f} for position")
\`\`\`

**Key Differences from Financial Assets:**

| Feature | Stocks/Bonds | Commodities |
|---------|--------------|-------------|
| Storage | No cost | Significant cost |
| Perishability | Never expires | Can spoil/degrade |
| Quality | Standardized | Varies (grades) |
| Delivery | Electronic | Physical |
| Seasonality | Minimal | Major (agriculture) |
| Carry Cost | Interest rate | Storage + insurance |

---

## Commodity Futures: Contango vs Backwardation

The most important concept in commodity markets!

\`\`\`python
class FuturesCurve:
    """
    Model commodity futures curve
    
    Contango: Futures > Spot (normal, reflects storage costs)
    Backwardation: Futures < Spot (shortage, convenience yield)
    """
    
    def __init__(self, spot_price: float, commodity: Commodity):
        self.spot_price = spot_price
        self.commodity = commodity
        self.curve_points = {}
    
    def calculate_contango_curve (self,
                                 risk_free_rate: float,
                                 months_forward: list[int]) -> dict:
        """
        Calculate futures prices in contango
        
        Future Price = Spot × e^((r + s) × T)
        
        where:
        r = risk-free rate
        s = storage cost
        T = time to maturity
        """
        curve = {}
        
        for months in months_forward:
            years = months / 12
            
            # Cost of carry = interest + storage
            cost_of_carry = risk_free_rate + self.commodity.storage_cost_annual
            
            # Futures price
            futures_price = self.spot_price * np.exp (cost_of_carry * years)
            
            curve[months] = {
                'months': months,
                'futures_price': futures_price,
                'spot_price': self.spot_price,
                'premium': futures_price - self.spot_price,
                'premium_pct': ((futures_price / self.spot_price) - 1) * 100,
                'annualized_roll_cost': ((futures_price / self.spot_price) - 1) * (12 / months) * 100
            }
        
        return curve
    
    def calculate_backwardation_curve (self,
                                     convenience_yield: float,
                                     risk_free_rate: float,
                                     months_forward: list[int]) -> dict:
        """
        Calculate futures prices in backwardation
        
        Future Price = Spot × e^((r + s - c) × T)
        
        where:
        c = convenience yield (benefit of holding physical)
        
        If c > (r + s), futures < spot (backwardation)
        """
        curve = {}
        
        for months in months_forward:
            years = months / 12
            
            # Net cost of carry = interest + storage - convenience yield
            net_carry = risk_free_rate + self.commodity.storage_cost_annual - convenience_yield
            
            futures_price = self.spot_price * np.exp (net_carry * years)
            
            curve[months] = {
                'months': months,
                'futures_price': futures_price,
                'spot_price': self.spot_price,
                'discount': self.spot_price - futures_price,
                'discount_pct': ((self.spot_price / futures_price) - 1) * 100,
                'annualized_roll_yield': ((self.spot_price / futures_price) - 1) * (12 / months) * 100
            }
        
        return curve
    
    def determine_curve_shape (self, 
                             near_month_future: float,
                             far_month_future: float) -> str:
        """Determine if curve is in contango or backwardation"""
        if far_month_future > near_month_future:
            return "Contango (upward sloping)"
        elif far_month_future < near_month_future:
            return "Backwardation (downward sloping)"
        else:
            return "Flat"

# Example 1: Oil in Contango (normal market)
oil = COMMODITIES['CL']
oil_curve = FuturesCurve (spot_price=75.00, commodity=oil)

print("\\n=== Crude Oil Futures Curve ===\\n")
print("Scenario 1: CONTANGO (Normal Market)\\n")

contango = oil_curve.calculate_contango_curve(
    risk_free_rate=0.05,  # 5% interest rate
    months_forward=[1, 3, 6, 12, 24]
)

print(f"Spot Price: \\$\{oil_curve.spot_price:.2f}/barrel\\n")
print("Futures Prices:")
for months, data in contango.items():
    print(f"  {months}M: \\$\{data['futures_price']:.2f} "
          f"(+\${data['premium']:.2f}, +{data['premium_pct']:.1f}%)")

print(f"\\nInterpretation:")
print(f"• Futures trade ABOVE spot (normal)")
print(f"• Reflects storage costs + interest")
print(f"• Roll cost: ~{contango[12]['annualized_roll_cost']:.1f}% annually")
print(f"• Long-only commodity funds LOSE from rolling futures")

# Example 2: Oil in Backwardation (shortage)
print("\\n\\nScenario 2: BACKWARDATION (Shortage/High Demand)\\n")

backwardation = oil_curve.calculate_backwardation_curve(
    convenience_yield = 0.12,  # 12 % convenience yield (shortage!)
    risk_free_rate = 0.05,
    months_forward = [1, 3, 6, 12, 24]
)

print(f"Spot Price: \\$\{oil_curve.spot_price:.2f}/barrel\\n")
print("Futures Prices:")
for months, data in backwardation.items():
    print(f"  {months}M: \\$\{data['futures_price']:.2f} "
          f"(-\${data['discount']:.2f}, -{data['discount_pct']:.1f}%)")

print(f"\\nInterpretation:")
print(f"• Futures trade BELOW spot (shortage)")
print(f"• Market needs oil NOW, not later")
print(f"• Roll yield: ~{backwardation[12]['annualized_roll_yield']:.1f}% annually")
print(f"• Long-only commodity funds GAIN from rolling futures")
\`\`\`

**Why This Matters:**

**Contango (normal):**
- Futures > Spot
- Costs money to "roll" futures (sell expiring, buy next month)
- Commodity ETFs slowly lose value even if spot is flat
- Example: USO (crude oil ETF) lost 50%+ when spot was flat (2010-2014)

**Backwardation (shortage):**
- Spot > Futures
- Earn money when rolling futures
- Commodity ETFs gain even if spot is flat
- Rare, but very profitable for long positions

---

## The 2020 Oil Price Crisis: Negative Prices!

April 20, 2020: WTI crude futures went **NEGATIVE $37/barrel**. Here\'s what happened:

\`\`\`python
class OilCrisis2020:
    """
    Model the April 2020 oil price crash
    
    First time in history oil traded negative!
    """
    
    @staticmethod
    def explain_negative_prices():
        """
        Why oil went negative
        """
        timeline = {
            'background': {
                'date': 'March 2020',
                'event': 'COVID-19 pandemic',
                'impact': 'Global oil demand crashed 30%+',
                'problem': 'Production continued, storage filling up'
            },
            'storage_crisis': {
                'cushing_ok': 'Main US storage hub',
                'capacity': '76 million barrels',
                'utilization': '95% full by mid-April',
                'days_to_full': '7-10 days',
                'problem': 'Nowhere to put the oil!'
            },
            'futures_mechanics': {
                'may_contract': 'Expiring April 21, 2020',
                'delivery': 'Physical delivery at Cushing, OK',
                'holders': 'Must take delivery or roll',
                'issue': 'No storage available for delivery'
            },
            'the_crash': {
                'april_20_open': '$18/barrel',
                'april_20_low': '-$37.63/barrel',
                'total_move': '-$55/barrel in one day',
                'interpretation': 'Traders would PAY $37 to avoid taking delivery'
            },
            'lessons': [
                'Physical delivery matters (most traders never take delivery)',
                'Storage constraints can break markets',
                'Futures contracts have embedded optionality',
                'USO ETF nearly collapsed (had to restructure)',
                'Long-dated futures stayed positive (June+ contracts)',
                'Producers shut in wells (cheaper than selling at negative)'
            ]
        }
        
        return timeline
    
    @staticmethod
    def simulate_trader_dilemma(
        contract_expiration_days: int,
        storage_available: bool,
        current_price: float
    ) -> dict:
        """
        Simulate a trader's decision on expiration day
        """
        if storage_available:
            # Can take delivery
            decision = "Take delivery and store"
            cost = 0
            outcome = "Keep oil, sell later when prices recover"
        else:
            # Cannot take delivery
            if contract_expiration_days == 0:
                # Must close today at ANY price
                decision = "Panic sell at market"
                cost = abs (current_price) if current_price < 0 else 0
                outcome = f"{'Pay' if current_price < 0 else 'Receive'} \${abs (current_price)}/barrel to close"
            else:
                # Can roll to next month
                next_month_price = 20  # June futures ~$20
                roll_cost = next_month_price - current_price
                decision = "Roll to next month"
                cost = roll_cost
                outcome = f"Pay \${roll_cost}/barrel to roll"
        
        return {
            'storage_available': storage_available,
            'days_to_expiration': contract_expiration_days,
            'current_price': current_price,
            'decision': decision,
            'cost': cost,
            'outcome': outcome
        }

# Explain what happened
crisis = OilCrisis2020()
timeline = crisis.explain_negative_prices()

print("\\n=== April 2020 Oil Price Crisis ===\\n")
print(f"Background: {timeline['background']['event']}")
print(f"  Impact: {timeline['background']['impact']}")
print(f"  Problem: {timeline['background']['problem']}")

print(f"\\nStorage Crisis:")
print(f"  Location: {timeline['storage_crisis']['cushing_ok']}")
print(f"  Utilization: {timeline['storage_crisis']['utilization']}")
print(f"  Problem: {timeline['storage_crisis']['problem']}")

print(f"\\nThe Crash (April 20, 2020):")
print(f"  Open: \\$\{timeline['the_crash']['april_20_open']}/barrel")
print(f"  Low: \\$\{timeline['the_crash']['april_20_low']}/barrel")
print(f"  Interpretation: {timeline['the_crash']['interpretation']}")

print(f"\\nTrader Dilemmas:\\n")

# Scenario 1: Trader with storage
scenario1 = crisis.simulate_trader_dilemma(
    contract_expiration_days=0,
    storage_available=True,
    current_price=-37
)
print(f"Trader A (has storage):")
print(f"  Decision: {scenario1['decision']}")
print(f"  Outcome: {scenario1['outcome']}")

# Scenario 2: Trader without storage
scenario2 = crisis.simulate_trader_dilemma(
    contract_expiration_days=0,
    storage_available=False,
    current_price=-37
)
print(f"\\nTrader B (NO storage):")
print(f"  Decision: {scenario2['decision']}")
print(f"  Outcome: {scenario2['outcome']}")
print(f"  ⚠️  PAYS $37/barrel × 1,000 barrels = $37,000 to close!")

print(f"\\n📚 Lessons:")
for lesson in timeline['lessons']:
    print(f"  • {lesson}")
\`\`\`

**Impact on Commodity ETFs:**

USO (United States Oil Fund) - largest oil ETF:
- Held mostly front-month futures
- Got caught in the crash
- Had to restructure holdings to avoid negative prices
- Switched to longer-dated contracts
- Investors lost 80%+ of value (even though oil recovered!)

**Lesson**: Commodity ETFs are NOT the same as owning the commodity!

---

## Hedging with Commodity Futures

Why producers and consumers use futures:

\`\`\`python
class CommodityHedge:
    """
    Model hedging strategies for producers and consumers
    """
    
    @staticmethod
    def farmer_hedge(
        crop: str,
        expected_production: int,  # bushels
        current_futures_price: float,
        harvest_date: datetime,
        production_cost: float  # per bushel
    ) -> dict:
        """
        Farmer hedges crop price risk
        
        Problem: Planting now, selling in 6 months
        Risk: Prices might fall below production cost
        Solution: Lock in price with futures
        """
        # Expected revenue (unhedged)
        expected_revenue = expected_production * current_futures_price
        
        # Total production costs
        total_costs = expected_production * production_cost
        
        # Profit if prices stay at current level
        expected_profit = expected_revenue - total_costs
        
        # Hedge: Sell futures to lock in current price
        num_contracts = expected_production / 5000  # 5,000 bushels per contract
        
        hedged_revenue = expected_production * current_futures_price
        hedged_profit = hedged_revenue - total_costs
        
        return {
            'crop': crop,
            'production': expected_production,
            'current_futures': current_futures_price,
            'production_cost': production_cost,
            'hedge_strategy': f'Sell {num_contracts:.1f} futures contracts',
            'locked_in_revenue': hedged_revenue,
            'locked_in_profit': hedged_profit,
            'profit_margin': (hedged_profit / hedged_revenue) * 100,
            'protection': 'Profit guaranteed regardless of price moves',
            'tradeoff': 'Gives up upside if prices rise'
        }
    
    @staticmethod
    def airline_hedge(
        annual_fuel_consumption: int,  # gallons
        current_jet_fuel_price: float,
        hedge_percentage: float,  # How much to hedge
        hedge_price: float  # Strike price if using options
    ) -> dict:
        """
        Airline hedges fuel costs
        
        Problem: Fuel is 30% of costs, prices volatile
        Risk: Price spike would destroy margins
        Solution: Lock in fuel price with futures/options
        """
        total_fuel_cost_unhedged = annual_fuel_consumption * current_jet_fuel_price
        hedged_volume = annual_fuel_consumption * hedge_percentage
        unhedged_volume = annual_fuel_consumption * (1 - hedge_percentage)
        
        hedged_cost = hedged_volume * hedge_price
        
        return {
            'annual_consumption': annual_fuel_consumption,
            'current_price': current_jet_fuel_price,
            'hedge_percentage': hedge_percentage * 100,
            'hedge_strategy': f'Buy {hedge_percentage*100:.0f}% of fuel needs via futures',
            'locked_in_price': hedge_price,
            'hedged_volume': hedged_volume,
            'hedged_cost': hedged_cost,
            'unhedged_volume': unhedged_volume,
            'scenarios': {
                'price_spikes_50pct': {
                    'new_price': current_jet_fuel_price * 1.5,
                    'unhedged_cost': annual_fuel_consumption * current_jet_fuel_price * 1.5,
                    'hedged_cost': hedged_cost + (unhedged_volume * current_jet_fuel_price * 1.5),
                    'savings': (annual_fuel_consumption * current_jet_fuel_price * 1.5) - 
                              (hedged_cost + (unhedged_volume * current_jet_fuel_price * 1.5))
                }
            }
        }

# Example 1: Corn farmer hedging
farmer = CommodityHedge.farmer_hedge(
    crop="Corn",
    expected_production=50_000,  # 50,000 bushels
    current_futures_price=4.50,  # $4.50/bushel
    harvest_date=datetime.now() + timedelta (days=180),
    production_cost=3.80  # $3.80/bushel to produce
)

print("\\n=== Commodity Hedging Examples ===\\n")
print("Example 1: FARMER HEDGING CROP PRICE\\n")
print(f"Crop: {farmer['crop']}")
print(f"Production: {farmer['production']:,} bushels")
print(f"Current Futures: \\$\{farmer['current_futures']}/bushel")
print(f"Production Cost: \\$\{farmer['production_cost']}/bushel")
print(f"\\nHedge Strategy: {farmer['hedge_strategy']}")
print(f"Locked-in Revenue: \\$\{farmer['locked_in_revenue']:,.0f}")
print(f"Locked-in Profit: \\$\{farmer['locked_in_profit']:,.0f}")
print(f"Profit Margin: {farmer['profit_margin']:.1f}%")
print(f"\\n✓ {farmer['protection']}")
print(f"⚠ {farmer['tradeoff']}")

# Example 2: Airline hedging fuel
airline = CommodityHedge.airline_hedge(
    annual_fuel_consumption = 100_000_000,  # 100M gallons
    current_jet_fuel_price = 3.00,
    hedge_percentage = 0.60,  # Hedge 60 %
hedge_price=3.10  # Locked in at $3.10(slight premium)
)

print(f"\\n\\nExample 2: AIRLINE HEDGING FUEL COST\\n")
print(f"Annual Consumption: {airline['annual_consumption']/1e6:.0f}M gallons")
print(f"Current Price: \\$\{airline['current_price']}/gallon")
print(f"\\nHedge Strategy: {airline['hedge_strategy']}")
print(f"Locked-in Price: \\$\{airline['locked_in_price']}/gallon")
print(f"Hedged Volume: {airline['hedged_volume']/1e6:.0f}M gallons")

scenario = airline['scenarios']['price_spikes_50pct']
print(f"\\nScenario: Fuel prices spike 50% to \\$\{scenario['new_price']}/gallon")
print(f"  Unhedged Cost: \\$\{scenario['unhedged_cost']/1e6:.0f}M")
print(f"  Hedged Cost: \\$\{scenario['hedged_cost']/1e6:.0f}M")
print(f"  Savings: \\$\{scenario['savings']/1e6:.0f}M")
print(f"\\n✓ Hedging saved \\$\{scenario['savings']/1e6:.0f}M when fuel spiked!")
\`\`\`

**Real-World Examples:**1. **Southwest Airlines** - Famously hedged fuel in 2000s, saved billions
2. **Cargill** - Major grain trader, hedges constantly
3. **Barrick Gold** - Gold miner, hedges production

---

## Gold: The Unique Commodity

Gold is special - it's both a commodity AND a monetary asset.

\`\`\`python
class GoldMarketAnalysis:
    """
    Analyze gold's unique characteristics
    """
    
    @staticmethod
    def gold_vs_other_commodities() -> dict:
        """
        What makes gold different
        """
        return {
            'as_commodity': {
                'storage_cost': 'Very low (0.3%/year)',
                'spoilage': 'Never deteriorates',
                'industrial_use': 'Only ~10% (jewelry/electronics)',
                'convenience_yield': 'Near zero (not consumed)'
            },
            'as_monetary_asset': {
                'safe_haven': 'Rises during crises',
                'inflation_hedge': 'Preserves value',
                'no_counterparty_risk': 'Physical ownership',
                'central_bank_holdings': '35,000+ tonnes held by CBs'
            },
            'unique_features': [
                'Only commodity that is also a reserve asset',
                'Negative correlation with stocks during crashes',
                'No yield (unlike bonds), cost to hold',
                'Supply is stable (~3,000 tonnes mined annually)',
                'Demand driven by fear, not industrial use'
            ]
        }
    
    @staticmethod
    def gold_price_drivers() -> list:
        """
        What moves gold prices
        """
        return [
            {
                'driver': 'Real Interest Rates',
                'relationship': 'Inverse (gold rises when real rates fall)',
                'reason': 'Gold has no yield, so low/negative rates make it attractive'
            },
            {
                'driver': 'USD Strength',
                'relationship': 'Inverse (gold falls when USD strengthens)',
                'reason': 'Gold priced in USD, strong dollar makes gold expensive'
            },
            {
                'driver': 'Market Fear (VIX)',
                'relationship': 'Positive (gold rises with VIX)',
                'reason': 'Safe haven during market stress'
            },
            {
                'driver': 'Inflation Expectations',
                'relationship': 'Positive',
                'reason': 'Gold preserves purchasing power'
            },
            {
                'driver': 'Central Bank Buying',
                'relationship': 'Positive',
                'reason': 'Central banks accumulating reserves drives demand'
            }
        ]

# Display gold's unique characteristics
analysis = GoldMarketAnalysis()
comparison = analysis.gold_vs_other_commodities()

print("\\n=== Gold: The Unique Commodity ===\\n")
print("As a Commodity:")
for key, value in comparison['as_commodity'].items():
    print(f"  {key.replace('_', ' ').title()}: {value}")

print("\\nAs a Monetary Asset:")
for key, value in comparison['as_monetary_asset'].items():
    print(f"  {key.replace('_', ' ').title()}: {value}")

print("\\nWhat Makes Gold Unique:")
for feature in comparison['unique_features']:
    print(f"  • {feature}")

print("\\n\\nGold Price Drivers:\\n")
drivers = analysis.gold_price_drivers()
for driver in drivers:
    print(f"{driver['driver']}: {driver['relationship']}")
    print(f"  Why: {driver['reason']}\\n")
\`\`\`

**Historical Gold Performance:**
- 1980: $850/oz (inflation-adjusted: ~$3,000 today)
- 2000: $280/oz (20-year low)
- 2011: $1,920/oz (financial crisis high)
- 2020: $2,070/oz (COVID high)
- 2024: ~$2,000/oz

**Investment Vehicles:**
- Physical gold (bars, coins)
- Gold ETFs (GLD, IAU)
- Gold mining stocks (riskier, levered to gold price)
- Gold futures (100 troy oz contracts)

---

## Building a Commodity Trading System

\`\`\`python
import pandas as pd

class CommodityTradingSystem:
    """
    Production commodity trading system
    """
    
    def __init__(self):
        self.positions = {}
        self.margin_account = 100_000  # $100K
        self.trade_history = []
    
    def get_commodity_quote (self, symbol: str) -> dict:
        """
        Get commodity futures quote
        In production: API call to CME, ICE, etc.
        """
        # Simulated quotes
        quotes = {
            'CL': {'bid': 75.45, 'ask': 75.47, 'last': 75.46, 'volume': 500_000},
            'GC': {'bid': 2020.50, 'ask': 2020.80, 'last': 2020.60, 'volume': 200_000},
            'NG': {'bid': 2.85, 'ask': 2.87, 'last': 2.86, 'volume': 300_000}
        }
        
        return {
            'symbol': symbol,
            **quotes.get (symbol, {}),
            'timestamp': datetime.now()
        }
    
    def calculate_margin_requirement (self, 
                                    symbol: str,
                                    num_contracts: int) -> float:
        """
        Calculate initial margin required
        
        Commodity futures margins set by exchange
        Typically 5-15% of contract value
        """
        commodity = COMMODITIES[symbol]
        quote = self.get_commodity_quote (symbol)
        
        # Contract value
        contract_value = quote['last'] * commodity.contract_size
        
        # Margin rates (exchange-set)
        margin_rates = {
            'CL': 0.07,  # 7% for crude oil
            'GC': 0.05,  # 5% for gold
            'NG': 0.10   # 10% for natural gas (volatile!)
        }
        
        margin_per_contract = contract_value * margin_rates.get (symbol, 0.10)
        total_margin = margin_per_contract * num_contracts
        
        return total_margin
    
    def open_position (self,
                     symbol: str,
                     side: Literal['long', 'short'],
                     num_contracts: int) -> dict:
        """Open commodity futures position"""
        commodity = COMMODITIES[symbol]
        quote = self.get_commodity_quote (symbol)
        
        # Calculate margin
        required_margin = self.calculate_margin_requirement (symbol, num_contracts)
        
        if required_margin > self.margin_account:
            return {
                'status': 'REJECTED',
                'reason': 'Insufficient margin',
                'required': required_margin,
                'available': self.margin_account
            }
        
        # Execute trade
        entry_price = quote['ask'] if side == 'long' else quote['bid']
        
        position_id = f"{symbol}_{side}_{int (time.time())}"
        
        position = {
            'position_id': position_id,
            'symbol': symbol,
            'commodity_name': commodity.name,
            'side': side,
            'num_contracts': num_contracts,
            'contract_size': commodity.contract_size,
            'entry_price': entry_price,
            'entry_time': datetime.now(),
            'margin_posted': required_margin,
            'notional_value': entry_price * commodity.contract_size * num_contracts,
            'leverage': (entry_price * commodity.contract_size * num_contracts) / required_margin
        }
        
        self.positions[position_id] = position
        self.margin_account -= required_margin
        
        return {
            'status': 'FILLED',
            **position
        }
    
    def calculate_position_pnl (self, position_id: str) -> dict:
        """Calculate current P&L on position"""
        position = self.positions[position_id]
        commodity = COMMODITIES[position['symbol']]
        quote = self.get_commodity_quote (position['symbol'])
        
        current_price = quote['bid'] if position['side'] == 'long' else quote['ask']
        
        if position['side'] == 'long':
            price_change = current_price - position['entry_price']
        else:
            price_change = position['entry_price'] - current_price
        
        pnl = price_change * commodity.contract_size * position['num_contracts']
        pnl_pct = (pnl / position['margin_posted']) * 100
        
        return {
            'position_id': position_id,
            'symbol': position['symbol'],
            'entry_price': position['entry_price'],
            'current_price': current_price,
            'price_change': price_change,
            'pnl_dollars': pnl,
            'pnl_percent': pnl_pct,
            'margin_posted': position['margin_posted'],
            'new_equity': position['margin_posted'] + pnl
        }

# Example usage
system = CommodityTradingSystem()

print("\\n=== Commodity Trading System ===\\n")
print(f"Initial Margin: \\$\{system.margin_account:,.0f}")

# Trade 1: Long crude oil
oil_trade = system.open_position(
    symbol = 'CL',
    side = 'long',
    num_contracts = 5
)

print(f"\\nTrade 1: {oil_trade['status']}")
if oil_trade['status'] == 'FILLED':
    print(f"  Commodity: {oil_trade['commodity_name']}")
print(f"  Side: {oil_trade['side'].upper()}")
print(f"  Contracts: {oil_trade['num_contracts']}")
print(f"  Entry: \\$\{oil_trade['entry_price']:.2f}")
print(f"  Notional: \\$\{oil_trade['notional_value']:,.0f}")
print(f"  Margin: \\$\{oil_trade['margin_posted']:,.0f}")
print(f"  Leverage: {oil_trade['leverage']:.1f}x")

# Check P & L(simulated price move)
pnl = system.calculate_position_pnl (oil_trade['position_id'])
print(f"\\nCurrent P&L:")
print(f"  Current Price: \\$\{pnl['current_price']:.2f}")
print(f"  Price Change: \\$\{pnl['price_change']:+.2f}")
print(f"  P&L: \\$\{pnl['pnl_dollars']:,.2f}")
print(f"  Return: {pnl['pnl_percent']:+.1f}%")
\`\`\`

---

## Summary

**Key Takeaways:**1. **Physical Goods**: Commodities are real, must be stored, can spoil
2. **Futures Dominant**: Most commodity trading is futures, not spot
3. **Contango vs Backwardation**: Critical for understanding commodity returns
4. **Storage Matters**: Storage costs affect futures pricing
5. **Negative Prices Possible**: 2020 oil crisis proved it
6. **Hedging Essential**: Producers/consumers use futures to lock in prices
7. **Gold is Unique**: Both commodity and monetary asset
8. **Seasonality**: Agriculture especially seasonal

**For Engineers:**
- Need to handle physical delivery (or avoid it!)
- Storage costs in pricing models
- Exchange-specific rules (CME, ICE, etc.)
- Margin requirements vary by commodity
- Roll costs critical for long-term positions

**Next Steps:**
- Module 7: Deep dive into commodity derivatives
- Module 14: Building trading infrastructure
- Module 15: Commodity-specific risk management

You now understand commodity markets - ready to build commodity trading systems!
`,
  exercises: [
    {
      prompt:
        'Build a contango/backwardation detector that monitors futures curves for all major commodities, calculates roll yields, and alerts when curve shifts from contango to backwardation (often signals supply shortage and buying opportunity).',
      solution:
        '// Implementation: 1) Fetch futures prices for multiple maturities (1M, 3M, 6M, 12M), 2) Calculate slope of futures curve, 3) Determine contango (upward slope) vs backwardation (downward slope), 4) Calculate annualized roll yield, 5) Detect curve inversions (regime changes), 6) Alert on backwardation as potential buy signal, 7) Track historical curve shapes for pattern recognition',
    },
    {
      prompt:
        'Create a commodity hedging calculator for farmers that suggests optimal hedge ratios based on historical price volatility, production costs, and risk tolerance. Include scenario analysis showing P&L under different price outcomes.',
      solution:
        '// Implementation: 1) Input expected production, costs, current futures prices, 2) Fetch historical price volatility, 3) Calculate value-at-risk for unhedged position, 4) Optimize hedge ratio using portfolio theory (minimize downside while allowing some upside), 5) Simulate outcomes across price scenarios (-50% to +50%), 6) Display breakeven analysis, max loss protection, opportunity cost of hedging, 7) Recommend partial hedge (e.g., 60-80%) vs full hedge',
    },
  ],
};
