export const foreignExchangeMarkets = {
    title: "Foreign Exchange Markets",
    slug: "foreign-exchange-markets",
    description: "Master the world's largest financial market - $7.5 trillion daily trading volume",
    content: `
# Foreign Exchange Markets

## Introduction: The Biggest Market You've Never Seen

The foreign exchange (FX/Forex) market is the **largest financial market in the world**:
- 💰 **$7.5 TRILLION** daily trading volume (2022)
- 🌍 **24/7 global market** - never closes (except weekends)
- 🏦 **Decentralized OTC** - no central exchange
- ⚡ **Instant settlement** possible
- 🔧 **Essential for global trade** - companies need FX to do business internationally

For comparison:
- Global stock markets: ~$100B daily volume
- U.S. Treasury market: ~$650B daily volume
- FX market: **$7,500B daily volume** (10x larger!)

**What you'll learn:**
- Currency pairs and quotation conventions
- Spot vs forward FX markets
- Exchange rate determination
- Carry trade mechanics
- Building FX trading systems
- Central bank interventions and their impact

---

## Understanding Currency Pairs

In FX, you're always trading one currency against another - a **currency pair**.

\`\`\`python
from dataclasses import dataclass
from typing import Literal
from datetime import datetime
import numpy as np

@dataclass
class CurrencyPair:
    """
    Representation of a currency pair
    Always quoted as BASE/QUOTE
    """
    base_currency: str  # Currency being bought/sold
    quote_currency: str  # Currency used to quote the price
    exchange_rate: float  # How many quote currency per 1 base currency
    
    def __post_init__(self):
        self.pair = f"{self.base_currency}/{self.quote_currency}"
    
    def invert(self) -> 'CurrencyPair':
        """Invert the currency pair"""
        return CurrencyPair(
            base_currency=self.quote_currency,
            quote_currency=self.base_currency,
            exchange_rate=1 / self.exchange_rate
        )
    
    def convert_amount(self, 
                      amount: float,
                      direction: Literal['base_to_quote', 'quote_to_base']) -> float:
        """
        Convert between currencies
        
        Args:
            amount: Amount to convert
            direction: Which direction to convert
        """
        if direction == 'base_to_quote':
            # Converting base currency to quote currency
            return amount * self.exchange_rate
        else:
            # Converting quote currency to base currency
            return amount / self.exchange_rate
    
    def calculate_pip_value(self, 
                           position_size: float,
                           pip_decimal: int = 4) -> float:
        """
        Calculate value of one pip (percentage in point)
        
        Most pairs: 1 pip = 0.0001
        JPY pairs: 1 pip = 0.01
        """
        pip_size = 10 ** (-pip_decimal)
        pip_value = position_size * pip_size
        return pip_value

# Example: EUR/USD
eur_usd = CurrencyPair(
    base_currency="EUR",
    quote_currency="USD",
    exchange_rate=1.0850  # 1 EUR = 1.0850 USD
)

print("=== Currency Pair Basics ===\\n")
print(f"Pair: {eur_usd.pair}")
print(f"Rate: {eur_usd.exchange_rate}")
print(f"Meaning: 1 {eur_usd.base_currency} = {eur_usd.exchange_rate} {eur_usd.quote_currency}")

# Convert 1,000 EUR to USD
usd_amount = eur_usd.convert_amount(1000, 'base_to_quote')
print(f"\\n€1,000 = ${usd_amount:, .2f
}")

# Convert 10,000 USD to EUR
eur_amount = eur_usd.convert_amount(10000, 'quote_to_base')
print(f"$10,000 = €{eur_amount:,.2f}")

# Invert to get USD / EUR
usd_eur = eur_usd.invert()
print(f"\\nInverted pair: {usd_eur.pair}")
print(f"Rate: {usd_eur.exchange_rate:.4f}")
print(f"Meaning: 1 {usd_eur.base_currency} = {usd_eur.exchange_rate:.4f} {usd_eur.quote_currency}")

# Calculate pip value for standard lot(100,000 EUR)
pip_value = eur_usd.calculate_pip_value(100000, pip_decimal = 4)
print(f"\\nFor 100,000 EUR position:")
print(f"1 pip (0.0001) move = ${pip_value:.2f}")
print(f"If EUR/USD moves from 1.0850 to 1.0950 (+100 pips)")
print(f"Profit = 100 pips × ${pip_value:.2f} = ${pip_value * 100:,.2f}")
\`\`\`

**Key Terminology:**
- **Base Currency**: The currency being bought or sold (left side)
- **Quote Currency**: The currency used to express the price (right side)
- **Pip (Point)**: Smallest price move (usually 0.0001 for most pairs)
- **Lot**: Standard trading size (standard lot = 100,000 units)
- **Spread**: Difference between bid and ask

---

## Major, Minor, and Exotic Currency Pairs

\`\`\`python
class ForexMarketClassification:
    """
    Classification of currency pairs by liquidity and trading volume
    """
    
    MAJORS = {
        'EUR/USD': {
            'name': 'Euro Dollar',
            'daily_volume': 1_100_000_000_000,  # $1.1T
            'typical_spread_pips': 0.1,
            'characteristics': 'Most liquid pair, ~28% of all FX volume',
            'avg_daily_range': 70  # pips
        },
        'USD/JPY': {
            'name': 'Dollar Yen',
            'daily_volume': 550_000_000_000,
            'typical_spread_pips': 0.1,
            'characteristics': 'Second most liquid, carry trade favorite',
            'avg_daily_range': 50
        },
        'GBP/USD': {
            'name': 'Cable',
            'daily_volume': 330_000_000_000,
            'typical_spread_pips': 0.2,
            'characteristics': 'Nicknamed "Cable" from transatlantic telegraph cable',
            'avg_daily_range': 100
        },
        'USD/CHF': {
            'name': 'Swissie',
            'daily_volume': 180_000_000_000,
            'typical_spread_pips': 0.3,
            'characteristics': 'Safe haven currency',
            'avg_daily_range': 60
        },
        'AUD/USD': {
            'name': 'Aussie',
            'daily_volume': 210_000_000_000,
            'typical_spread_pips': 0.2,
            'characteristics': 'Commodity-linked currency',
            'avg_daily_range': 65
        },
        'USD/CAD': {
            'name': 'Loonie',
            'daily_volume': 170_000_000_000,
            'typical_spread_pips': 0.2,
            'characteristics': 'Oil-linked (Canada is major oil exporter)',
            'avg_daily_range': 70
        },
        'NZD/USD': {
            'name': 'Kiwi',
            'daily_volume': 90_000_000_000,
            'typical_spread_pips': 0.3,
            'characteristics': 'High interest rate, carry trade',
            'avg_daily_range': 60
        }
    }
    
    MINORS = {
        'EUR/GBP': {'name': 'Euro Pound', 'characteristics': 'European cross'},
        'EUR/JPY': {'name': 'Euro Yen', 'characteristics': 'Popular carry trade'},
        'GBP/JPY': {'name': 'Guppy', 'characteristics': 'Very volatile'},
        'EUR/CHF': {'name': 'Euro Swiss', 'characteristics': 'SNB intervention risk'},
        'AUD/JPY': {'name': 'Aussie Yen', 'characteristics': 'Risk-on/risk-off proxy'}
    }
    
    EXOTICS = {
        'USD/TRY': {'name': 'Dollar Turkish Lira', 'risk': 'HIGH - political instability'},
        'USD/ZAR': {'name': 'Dollar South African Rand', 'risk': 'HIGH - emerging market'},
        'USD/MXN': {'name': 'Dollar Mexican Peso', 'risk': 'MEDIUM - oil-linked'},
        'EUR/TRY': {'name': 'Euro Turkish Lira', 'risk': 'VERY HIGH'},
    }
    
    @classmethod
    def get_pair_info(cls, pair: str) -> dict:
        """Get information about a currency pair"""
        if pair in cls.MAJORS:
            return {**cls.MAJORS[pair], 'category': 'Major', 'liquidity': 'VERY HIGH'}
        elif pair in cls.MINORS:
            return {**cls.MINORS[pair], 'category': 'Minor', 'liquidity': 'HIGH'}
        elif pair in cls.EXOTICS:
            return {**cls.EXOTICS[pair], 'category': 'Exotic', 'liquidity': 'LOW'}
        return {'category': 'Unknown'}
    
    @classmethod
    def calculate_daily_turnover(cls) -> dict:
        """Calculate total daily FX turnover by category"""
        major_volume = sum(p['daily_volume'] for p in cls.MAJORS.values())
        
        return {
            'majors': major_volume,
            'majors_pct': (major_volume / 7_500_000_000_000) * 100,
            'total_market': 7_500_000_000_000,
            'interpretation': f'Majors account for {(major_volume / 7_500_000_000_000) * 100:.0f}% of all FX trading'
        }

# Display market structure
print("\\n=== Forex Market Structure ===\\n")

print("MAJOR PAIRS (Most Liquid):")
for pair, info in ForexMarketClassification.MAJORS.items():
    print(f"  {pair} ({info['name']})")
    print(f"    Volume: ${info['daily_volume'] / 1e9: .0f}B daily")
print(f"    Spread: {info['typical_spread_pips']} pips")
print(f"    {info['characteristics']}\\n")

print("MINOR PAIRS (Cross Currencies - No USD):")
for pair, info in ForexMarketClassification.MINORS.items():
    print(f"  {pair}: {info['characteristics']}")

print("\\nEXOTIC PAIRS (Emerging Markets - High Risk):")
for pair, info in ForexMarketClassification.EXOTICS.items():
    print(f"  {pair}: {info['risk']}")

turnover = ForexMarketClassification.calculate_daily_turnover()
print(f"\\nTotal Daily Turnover: ${turnover['total_market']/1e12:.1f} Trillion")
print(f"{turnover['interpretation']}")
\`\`\`

**Trading Tips:**
- **Stick to majors** for beginners - tightest spreads, most liquidity
- **Avoid exotics** unless experienced - wide spreads, high slippage, political risk
- **Watch the time**: EUR/USD most active during European/US overlap (8am-12pm ET)

---

## Spot vs Forward FX Markets

\`\`\`python
from datetime import timedelta

class FXSpotContract:
    """
    Spot FX: Immediate delivery (T+2 settlement)
    """
    
    def __init__(self, pair: CurrencyPair, trade_date: datetime):
        self.pair = pair
        self.trade_date = trade_date
        self.settlement_date = trade_date + timedelta(days=2)  # T+2
        self.spot_rate = pair.exchange_rate
    
    def execute_trade(self,
                     notional: float,
                     side: Literal['buy', 'sell']) -> dict:
        """
        Execute spot FX trade
        
        Buy = Buy base currency, sell quote currency
        Sell = Sell base currency, buy quote currency
        """
        if side == 'buy':
            base_amount = notional
            quote_amount = notional * self.spot_rate
            action = f"Buy {base_amount:,.0f} {self.pair.base_currency}"
            counteraction = f"Sell {quote_amount:,.2f} {self.pair.quote_currency}"
        else:
            base_amount = notional
            quote_amount = notional * self.spot_rate
            action = f"Sell {base_amount:,.0f} {self.pair.base_currency}"
            counteraction = f"Buy {quote_amount:,.2f} {self.pair.quote_currency}"
        
        return {
            'pair': self.pair.pair,
            'side': side,
            'action': action,
            'counteraction': counteraction,
            'spot_rate': self.spot_rate,
            'trade_date': self.trade_date,
            'settlement_date': self.settlement_date,
            'notional_base': base_amount,
            'notional_quote': quote_amount
        }

class FXForwardContract:
    """
    Forward FX: Delivery at future date
    Used for hedging FX risk
    """
    
    def __init__(self,
                 pair: CurrencyPair,
                 trade_date: datetime,
                 maturity_date: datetime,
                 spot_rate: float,
                 base_interest_rate: float,
                 quote_interest_rate: float):
        self.pair = pair
        self.trade_date = trade_date
        self.maturity_date = maturity_date
        self.spot_rate = spot_rate
        self.base_rate = base_interest_rate
        self.quote_rate = quote_interest_rate
        self.forward_rate = self.calculate_forward_rate()
    
    def calculate_forward_rate(self) -> float:
        """
        Calculate forward FX rate using interest rate parity
        
        Forward = Spot × (1 + r_quote × T) / (1 + r_base × T)
        
        Or more precisely:
        Forward = Spot × e^((r_quote - r_base) × T)
        """
        years_to_maturity = (self.maturity_date - self.trade_date).days / 365.25
        
        # Interest rate parity
        forward = self.spot_rate * np.exp(
            (self.quote_rate - self.base_rate) * years_to_maturity
        )
        
        return forward
    
    def calculate_forward_points(self) -> float:
        """
        Forward points = (Forward - Spot) × 10,000
        Market convention: quote in points
        """
        return (self.forward_rate - self.spot_rate) * 10000
    
    def is_premium_or_discount(self) -> str:
        """Determine if forward is at premium or discount to spot"""
        if self.forward_rate > self.spot_rate:
            return f"Premium ({self.pair.base_currency} forward > spot)"
        elif self.forward_rate < self.spot_rate:
            return f"Discount ({self.pair.base_currency} forward < spot)"
        else:
            return "At Par"

# Example: Spot vs Forward
eur_usd_pair = CurrencyPair("EUR", "USD", 1.0850)

# Spot trade
spot_trade = FXSpotContract(eur_usd_pair, datetime.now())
spot_result = spot_trade.execute_trade(notional=1_000_000, side='buy')

print("\\n=== Spot FX Trade ===\\n")
print(f"Pair: {spot_result['pair']}")
print(f"Trade: {spot_result['action']}")
print(f"  For: {spot_result['counteraction']}")
print(f"Spot Rate: {spot_result['spot_rate']}")
print(f"Settlement: {spot_result['settlement_date'].strftime('%Y-%m-%d')} (T+2)")

# Forward trade (6 months)
forward = FXForwardContract(
    pair=eur_usd_pair,
    trade_date=datetime.now(),
    maturity_date=datetime.now() + timedelta(days=180),
    spot_rate=1.0850,
    base_interest_rate=0.035,  # EUR interest rate 3.5%
    quote_interest_rate=0.050  # USD interest rate 5.0%
)

print("\\n=== Forward FX Contract (6 months) ===\\n")
print(f"Pair: {forward.pair.pair}")
print(f"Spot Rate: {forward.spot_rate}")
print(f"Forward Rate: {forward.forward_rate:.4f}")
print(f"Forward Points: {forward.calculate_forward_points():.1f}")
print(f"Status: {forward.is_premium_or_discount()}")
print(f"\\nInterpretation:")
print(f"  USD has higher interest rate (5% vs 3.5%)")
print(f"  → EUR trades at forward discount")
print(f"  → Reflects interest rate differential")
\`\`\`

**Interest Rate Parity**: Forward rates reflect interest rate differentials. Higher yielding currency trades at forward discount.

---

## Exchange Rate Determination

What makes currencies strengthen or weaken?

\`\`\`python
class ExchangeRateModel:
    """
    Model factors that influence exchange rates
    """
    
    @staticmethod
    def purchasing_power_parity(
        domestic_price_level: float,
        foreign_price_level: float,
        base_exchange_rate: float = 1.0
    ) -> dict:
        """
        Purchasing Power Parity (PPP)
        
        In the long run, exchange rates adjust so that identical goods
        cost the same in both countries
        
        Example: Big Mac Index
        """
        implied_exchange_rate = (domestic_price_level / foreign_price_level) * base_exchange_rate
        
        return {
            'theory': 'Purchasing Power Parity',
            'domestic_price': domestic_price_level,
            'foreign_price': foreign_price_level,
            'implied_exchange_rate': implied_exchange_rate,
            'interpretation': 'Currency should adjust until prices are equal'
        }
    
    @staticmethod
    def interest_rate_differential_effect(
        domestic_rate: float,
        foreign_rate: float,
        current_exchange_rate: float
    ) -> dict:
        """
        Interest Rate Differential
        
        Higher interest rates attract foreign capital
        → Currency appreciates
        """
        rate_differential = domestic_rate - foreign_rate
        
        # Simplified model: 1% rate diff = ~1% currency move
        expected_appreciation = rate_differential
        
        future_exchange_rate = current_exchange_rate * (1 + expected_appreciation)
        
        return {
            'theory': 'Interest Rate Differential',
            'domestic_rate': domestic_rate,
            'foreign_rate': foreign_rate,
            'rate_differential': rate_differential,
            'current_rate': current_exchange_rate,
            'expected_future_rate': future_exchange_rate,
            'interpretation': f'Domestic currency expected to {"appreciate" if expected_appreciation > 0 else "depreciate"} by {abs(expected_appreciation)*100:.1f}%'
        }
    
    @staticmethod
    def balance_of_payments_effect(
        current_account_balance: float,
        capital_account_balance: float
    ) -> dict:
        """
        Balance of Payments
        
        Current Account (trade balance)
        + Capital Account (investment flows)
        = Balance of Payments
        
        Surplus → Currency appreciates
        Deficit → Currency depreciates
        """
        total_balance = current_account_balance + capital_account_balance
        
        if total_balance > 0:
            effect = "Currency Appreciation (surplus)"
        elif total_balance < 0:
            effect = "Currency Depreciation (deficit)"
        else:
            effect = "Neutral"
        
        return {
            'theory': 'Balance of Payments',
            'current_account': current_account_balance,
            'capital_account': capital_account_balance,
            'total_balance': total_balance,
            'effect': effect,
            'interpretation': f'{"Surplus" if total_balance > 0 else "Deficit"} of ${abs(total_balance)}B'
        }

# Example: EUR/USD analysis
print("\\n=== Exchange Rate Determinants ===\\n")

# 1. PPP
ppp = ExchangeRateModel.purchasing_power_parity(
    domestic_price_level=5.50,  # Big Mac in US = $5.50
    foreign_price_level=4.80,   # Big Mac in Europe = €4.80
    base_exchange_rate=1.0
)
print("1. Purchasing Power Parity:")
print(f"   US Big Mac: ${ppp['domestic_price']}")
print(f"   EU Big Mac: €{ppp['foreign_price']}")
print(f"   Implied EUR/USD: {ppp['implied_exchange_rate']:.4f}")
print(f"   {ppp['interpretation']}")

# 2. Interest rates
rates = ExchangeRateModel.interest_rate_differential_effect(
    domestic_rate=0.05,  # US rates = 5%
    foreign_rate=0.035,  # EU rates = 3.5%
    current_exchange_rate=1.0850
)
print(f"\\n2. Interest Rate Differential:")
print(f"   US rate: {rates['domestic_rate']*100}%")
print(f"   EU rate: {rates['foreign_rate']*100}%")
print(f"   Differential: {rates['rate_differential']*100:+.1f}%")
print(f"   {rates['interpretation']}")

# 3. Balance of payments
bop = ExchangeRateModel.balance_of_payments_effect(
    current_account_balance=-50,  # US trade deficit
    capital_account_balance=80    # Capital inflows to US
)
print(f"\\n3. Balance of Payments:")
print(f"   Current Account: ${bop['current_account']}B (trade deficit)")
print(f"   Capital Account: +${bop['capital_account']}B (inflows)")
print(f"   Effect: {bop['effect']}")
\`\`\`

**Real-World Factors:**
1. **Central Bank Policy**: Fed rate hikes strengthen USD
2. **Economic Growth**: Stronger growth → stronger currency
3. **Political Stability**: Uncertainty weakens currency (Brexit → GBP crashed)
4. **Trade Balance**: Exports > Imports → currency appreciates
5. **Safe Haven Flows**: Crisis → USD, CHF, JPY strengthen

---

## The Carry Trade

One of the most popular FX strategies.

\`\`\`python
class CarryTrade:
    """
    Carry Trade: Borrow low-yield currency, invest in high-yield currency
    Profit from interest rate differential
    
    Risk: Currency moves can wipe out interest gains
    """
    
    def __init__(self,
                 funding_currency: str,
                 funding_rate: float,
                 target_currency: str,
                 target_rate: float,
                 exchange_rate: float,
                 position_size: float):
        self.funding_currency = funding_currency
        self.funding_rate = funding_rate
        self.target_currency = target_currency
        self.target_rate = target_rate
        self.exchange_rate = exchange_rate
        self.position_size = position_size
    
    def calculate_carry(self, holding_period_days: int = 365) -> dict:
        """
        Calculate carry trade returns
        
        Profit = (Target Rate - Funding Rate) × Position Size × Time
               + FX Gain/Loss
        """
        years = holding_period_days / 365
        
        # Interest income from target currency
        target_interest = self.position_size * self.target_rate * years
        
        # Interest cost on borrowed funding currency
        funding_cost = self.position_size * self.funding_rate * years
        
        # Net carry (interest differential)
        net_carry = target_interest - funding_cost
        carry_return = net_carry / self.position_size
        
        return {
            'funding_currency': self.funding_currency,
            'funding_rate': self.funding_rate,
            'target_currency': self.target_currency,
            'target_rate': self.target_rate,
            'rate_differential': self.target_rate - self.funding_rate,
            'position_size': self.position_size,
            'holding_period_days': holding_period_days,
            'target_interest_earned': target_interest,
            'funding_cost_paid': funding_cost,
            'net_carry': net_carry,
            'carry_return_pct': carry_return * 100,
            'annualized_carry': (net_carry / self.position_size) * (365 / holding_period_days) * 100
        }
    
    def simulate_scenarios(self,
                          fx_changes: list[float]) -> pd.DataFrame:
        """
        Simulate carry trade under different FX scenarios
        
        Show how currency moves can overwhelm carry income
        """
        base_carry = self.calculate_carry(365)
        scenarios = []
        
        for fx_change in fx_changes:
            # FX P&L
            fx_pnl = self.position_size * fx_change
            
            # Total return = Carry + FX
            total_pnl = base_carry['net_carry'] + fx_pnl
            total_return = (total_pnl / self.position_size) * 100
            
            scenarios.append({
                'fx_change_pct': fx_change * 100,
                'carry_income': base_carry['net_carry'],
                'fx_pnl': fx_pnl,
                'total_pnl': total_pnl,
                'total_return_pct': total_return,
                'outcome': 'Profit' if total_pnl > 0 else 'Loss'
            })
        
        return pd.DataFrame(scenarios)

# Example: Classic carry trade (borrow JPY, buy AUD)
carry = CarryTrade(
    funding_currency="JPY",
    funding_rate=0.001,  # Bank of Japan: 0.1% (near zero)
    target_currency="AUD",
    target_rate=0.045,   # Reserve Bank of Australia: 4.5%
    exchange_rate=95.50,  # AUD/JPY = 95.50
    position_size=10_000_000  # $10M AUD
)

print("\\n=== Carry Trade Example ===\\n")
print(f"Strategy: Borrow {carry.funding_currency} at {carry.funding_rate*100}%")
print(f"         Invest in {carry.target_currency} at {carry.target_rate*100}%")
print(f"         Rate Differential: {(carry.target_rate - carry.funding_rate)*100:.1f}%\\n")

result = carry.calculate_carry(365)
print(f"Annual Returns (if FX unchanged):")
print(f"  Interest Earned ({carry.target_currency}): ${result['target_interest_earned']:, .0f}")
print(f"  Interest Paid ({carry.funding_currency}): ${result['funding_cost_paid']:,.0f}")
print(f"  Net Carry: ${result['net_carry']:,.0f}")
print(f"  Carry Return: {result['carry_return_pct']:.2f}%")

# Simulate scenarios
print("\\n=== Scenario Analysis ===\\n")
fx_scenarios = [-0.10, -0.05, 0, 0.05, 0.10]  # - 10 % to + 10 %
    scenarios_df = carry.simulate_scenarios(fx_scenarios)

print(scenarios_df.to_string(index = False))

print("\\n💡 KEY INSIGHT:")
print("   Carry trade works great... until it doesn't")
print("   A 10% adverse FX move wipes out 2+ years of carry income!")
print("   This is why carry trades 'blow up' during crises")
print("   (2008: Yen carry trades unwound violently)")
\`\`\`

**Famous Carry Trade Unwinds:**
- **1998 LTCM Crisis**: Russian default → carry trades collapsed
- **2008 Financial Crisis**: Flight to safety → JPY soared 30%+
- **2015 SNB Shock**: Swiss franc unpegged → carry trades destroyed

---

## Central Bank Interventions

\`\`\`python
class CentralBankIntervention:
    """
    Model central bank FX market interventions
    
    Central banks buy/sell their currency to influence exchange rates
    """
    
    @staticmethod
    def verbal_intervention(target_currency: str, desired_direction: str) -> dict:
        """
        Verbal intervention: Talk the currency up or down
        
        'Jawboning' - often effective without spending reserves
        """
        return {
            'type': 'Verbal Intervention',
            'currency': target_currency,
            'statement': f'{target_currency} is {desired_direction}',
            'cost': 0,
            'effectiveness': 'Moderate (short-term)',
            'example': 'ECB: "Euro is too strong, hurting exports"'
        }
    
    @staticmethod
    def direct_intervention(
        currency: str,
        action: Literal['buy', 'sell'],
        amount: float,
        reserves_available: float
    ) -> dict:
        """
        Direct intervention: Actually buy/sell currency in market
        
        Expensive and often unsuccessful against market forces
        """
        if amount > reserves_available:
            return {
                'success': False,
                'reason': 'Insufficient reserves',
                'amount_requested': amount,
                'reserves_available': reserves_available
            }
        
        new_reserves = reserves_available - amount
        
        impact = {
            'type': 'Direct Intervention',
            'currency': currency,
            'action': action,
            'amount': amount,
            'reserves_spent': amount,
            'remaining_reserves': new_reserves,
            'expected_impact': f'{action.title()} pressure on {currency}',
            'duration': 'Hours to days (market often overpowers)',
            'risk': 'Can lose billions if market disagrees'
        }
        
        return impact
    
    @staticmethod
    def swiss_national_bank_2015_case_study() -> dict:
        """
        Famous example: SNB removed EUR/CHF floor in Jan 2015
        
        One of the biggest FX shocks in history
        """
        return {
            'date': '2015-01-15',
            'central_bank': 'Swiss National Bank (SNB)',
            'action': 'Removed EUR/CHF 1.20 floor',
            'background': 'SNB had defended 1.20 floor for 3+ years, costing billions',
            'immediate_impact': {
                'eur_chf_drop': '30% in minutes',
                'trading_halts': 'Most brokers halted CHF trading',
                'broker_bankruptcies': '2+ retail FX brokers went bankrupt',
                'losses': 'Estimated $100B+ in global losses'
            },
            'aftermath': [
                'EUR/CHF fell from 1.20 to 0.85',
                'Many retail traders wiped out (negative balances)',
                'Brokers sued clients for losses beyond deposits',
                'SNB lost credibility for years',
                'Massive legal battles ensued'
            ],
            'lesson': 'Central banks cannot fight markets forever'
        }

# Examples
print("\\n=== Central Bank Interventions ===\\n")

# Verbal intervention
verbal = CentralBankIntervention.verbal_intervention("EUR", "too strong")
print(f"1. {verbal['type']}")
print(f"   Currency: {verbal['currency']}")
print(f"   Statement: {verbal['statement']}")
print(f"   Cost: ${verbal['cost']}")
print(f"   Effectiveness: {verbal['effectiveness']}")

# Direct intervention
direct = CentralBankIntervention.direct_intervention(
    currency="JPY",
    action="buy",
    amount=50_000_000_000,  # $50B
    reserves_available=1_300_000_000_000  # $1.3T (Japan's reserves)
)
print(f"\\n2. {direct['type']}")
print(f"   Action: {direct['action'].title()} JPY")
print(f"   Amount: ${direct['amount'] / 1e9: .0f}B")
print(f"   Impact: {direct['expected_impact']}")
print(f"   Duration: {direct['duration']}")

# Swiss National Bank case study
snb = CentralBankIntervention.swiss_national_bank_2015_case_study()
print(f"\\n3. Case Study: {snb['central_bank']} ({snb['date']})")
print(f"   Action: {snb['action']}")
print(f"   Immediate Drop: {snb['immediate_impact']['eur_chf_drop']}")
print(f"   Losses: {snb['immediate_impact']['losses']}")
print(f"   Lesson: {snb['lesson']}")
print(f"\\n   ⚠️  This is why you should NEVER trade against central banks...")
print(f"       ...until they give up!")
\`\`\`

---

## Building an FX Trading System

\`\`\`python
import time
from collections import deque

class FXTradingSystem:
    """
    Production FX trading system
    """
    
    def __init__(self, account_currency: str = "USD"):
        self.account_currency = account_currency
        self.positions = {}
        self.account_balance = 100_000  # $100K
        self.trade_history = []
        self.price_feed = deque(maxlen=1000)
    
    def get_live_price(self, pair: str) -> dict:
        """
        Get live FX price from broker/exchange
        In reality: WebSocket feed from provider
        """
        # Simulated bid/ask spread
        mid_price = self._fetch_mid_price(pair)
        spread = self._get_typical_spread(pair)
        
        return {
            'pair': pair,
            'bid': mid_price - spread/2,
            'ask': mid_price + spread/2,
            'spread_pips': spread * 10000,
            'timestamp': datetime.now()
        }
    
    def _fetch_mid_price(self, pair: str) -> float:
        """Fetch current mid price (simulated)"""
        # In production: API call to price provider
        base_prices = {
            'EUR/USD': 1.0850,
            'GBP/USD': 1.2650,
            'USD/JPY': 150.50,
            'AUD/USD': 0.6550
        }
        return base_prices.get(pair, 1.0)
    
    def _get_typical_spread(self, pair: str) -> float:
        """Get typical spread for pair"""
        spreads = {
            'EUR/USD': 0.0001,  # 1 pip
            'GBP/USD': 0.0002,  # 2 pips
            'USD/JPY': 0.01,    # 1 pip (JPY pairs different)
            'AUD/USD': 0.0002   # 2 pips
        }
        return spreads.get(pair, 0.0003)
    
    def execute_market_order(self,
                            pair: str,
                            side: Literal['buy', 'sell'],
                            lot_size: float) -> dict:
        """
        Execute FX market order
        
        Args:
            pair: Currency pair
            side: 'buy' (long base currency) or 'sell' (short base currency)
            lot_size: Position size (1.0 = 100,000 units, 0.1 = 10,000 units)
        """
        # Get current price
        quote = self.get_live_price(pair)
        
        # Buy at ask, sell at bid
        execution_price = quote['ask'] if side == 'buy' else quote['bid']
        
        # Calculate position value
        notional = lot_size * 100_000  # Standard lot = 100,000 units
        
        # Check margin requirement (typically 1-5%)
        required_margin = notional * 0.02  # 2% margin (50:1 leverage)
        
        if required_margin > self.account_balance:
            return {
                'status': 'REJECTED',
                'reason': 'Insufficient margin',
                'required': required_margin,
                'available': self.account_balance
            }
        
        # Execute trade
        position_id = f"{pair}_{side}_{int(time.time())}"
        
        position = {
            'position_id': position_id,
            'pair': pair,
            'side': side,
            'lot_size': lot_size,
            'notional': notional,
            'entry_price': execution_price,
            'entry_time': datetime.now(),
            'margin_used': required_margin,
            'current_pnl': 0
        }
        
        self.positions[position_id] = position
        self.account_balance -= required_margin
        
        return {
            'status': 'FILLED',
            'position_id': position_id,
            **position
        }
    
    def close_position(self, position_id: str) -> dict:
        """Close existing position"""
        if position_id not in self.positions:
            return {'status': 'ERROR', 'reason': 'Position not found'}
        
        position = self.positions[position_id]
        
        # Get current price
        quote = self.get_live_price(position['pair'])
        
        # Exit at opposite side
        exit_price = quote['bid'] if position['side'] == 'buy' else quote['ask']
        
        # Calculate P&L
        if position['side'] == 'buy':
            pnl = (exit_price - position['entry_price']) * position['notional']
        else:
            pnl = (position['entry_price'] - exit_price) * position['notional']
        
        # Return margin
        self.account_balance += position['margin_used'] + pnl
        
        # Remove position
        del self.positions[position_id]
        
        return {
            'status': 'CLOSED',
            'position_id': position_id,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': (pnl / position['margin_used']) * 100,
            'new_balance': self.account_balance
        }
    
    def calculate_portfolio_metrics(self) -> dict:
        """Calculate current portfolio state"""
        total_exposure = sum(p['notional'] for p in self.positions.values())
        total_margin_used = sum(p['margin_used'] for p in self.positions.values())
        free_margin = self.account_balance
        total_equity = self.account_balance + total_margin_used
        
        # Calculate unrealized P&L
        unrealized_pnl = 0
        for position in self.positions.values():
            quote = self.get_live_price(position['pair'])
            current_price = quote['bid'] if position['side'] == 'buy' else quote['ask']
            
            if position['side'] == 'buy':
                pnl = (current_price - position['entry_price']) * position['notional']
            else:
                pnl = (position['entry_price'] - current_price) * position['notional']
            
            unrealized_pnl += pnl
        
        return {
            'account_balance': free_margin,
            'margin_used': total_margin_used,
            'total_equity': total_equity + unrealized_pnl,
            'total_exposure': total_exposure,
            'leverage': total_exposure / total_equity if total_equity > 0 else 0,
            'unrealized_pnl': unrealized_pnl,
            'num_positions': len(self.positions),
            'margin_level': (total_equity / total_margin_used * 100) if total_margin_used > 0 else float('inf')
        }

# Example usage
system = FXTradingSystem(account_currency="USD")

print("\\n=== FX Trading System ===\\n")
print(f"Initial Balance: ${system.account_balance:, .0f}")

# Open long EUR / USD position
trade1 = system.execute_market_order(
    pair = 'EUR/USD',
    side = 'buy',
    lot_size = 1.0  # 100,000 EUR
)

print(f"\\nTrade 1: {trade1['status']}")
if trade1['status'] == 'FILLED':
    print(f"  Pair: {trade1['pair']}")
print(f"  Side: {trade1['side'].upper()}")
print(f"  Size: {trade1['lot_size']} lot ({trade1['notional']:,.0f} {trade1['pair'][:3]})")
print(f"  Entry: {trade1['entry_price']:.4f}")
print(f"  Margin Used: ${trade1['margin_used']:,.0f}")

# Open short GBP / USD position
trade2 = system.execute_market_order(
    pair = 'GBP/USD',
    side = 'sell',
    lot_size = 0.5  # 50,000 GBP
)

print(f"\\nTrade 2: {trade2['status']}")
if trade2['status'] == 'FILLED':
    print(f"  Pair: {trade2['pair']}")
print(f"  Side: {trade2['side'].upper()}")
print(f"  Size: {trade2['lot_size']} lot ({trade2['notional']:,.0f} {trade2['pair'][:3]})")
print(f"  Entry: {trade2['entry_price']:.4f}")

# Check portfolio
metrics = system.calculate_portfolio_metrics()
print(f"\\nPortfolio Metrics:")
print(f"  Total Equity: ${metrics['total_equity']:,.0f}")
print(f"  Margin Used: ${metrics['margin_used']:,.0f}")
print(f"  Free Margin: ${metrics['account_balance']:,.0f}")
print(f"  Leverage: {metrics['leverage']:.1f}x")
print(f"  Unrealized P&L: ${metrics['unrealized_pnl']:,.2f}")
print(f"  Margin Level: {metrics['margin_level']:.0f}%")

# Close first position
close_result = system.close_position(trade1['position_id'])
print(f"\\nClosed Position: {close_result['position_id']}")
print(f"  Entry: {close_result['entry_price']:.4f}")
print(f"  Exit: {close_result['exit_price']:.4f}")
print(f"  P&L: ${close_result['pnl']:,.2f}")
print(f"  Return: {close_result['pnl_pct']:.2f}%")
print(f"  New Balance: ${close_result['new_balance']:,.0f}")
\`\`\`

---

## Summary

**Key Takeaways:**

1. **Largest Market**: $7.5T daily, 24/7 trading, decentralized
2. **Currency Pairs**: Always quoted BASE/QUOTE (EUR/USD, GBP/USD, etc.)
3. **Majors vs Exotics**: Stick to majors for tight spreads and liquidity
4. **Spot vs Forward**: Spot = immediate (T+2), Forward = future delivery
5. **Exchange Rates**: Driven by interest rates, trade balance, capital flows
6. **Carry Trade**: Borrow low-yield, invest high-yield (works until it doesn't)
7. **Central Banks**: Can intervene but markets usually win eventually
8. **Leverage**: Typical 50:1, amplifies gains AND losses

**For Engineers:**
- Real-time price feeds critical (WebSockets)
- Need robust risk management (margin calls happen fast)
- Latency matters for arbitrage
- Regulatory compliance varies by jurisdiction
- Most retail traders lose money (80%+)

**Next Steps:**
- Module 13: Real-time market data processing
- Module 14: Building trading infrastructure
- Module 16: Payment systems (FX settlement)

You now understand the world's largest market - ready to build FX systems!
`,
    exercises: [
        {
            prompt: "Build an FX arbitrage detector that monitors triangular arbitrage opportunities across three currency pairs (e.g., EUR/USD, USD/JPY, EUR/JPY). Account for bid-ask spreads and transaction costs. Alert when net profit > threshold.",
            solution: "// Implementation: 1) Real-time feeds for all 3 pairs, 2) Calculate implied cross rate from two legs, 3) Compare to actual cross rate, 4) Account for bid-ask spreads (buy at ask, sell at bid), 5) Calculate round-trip cost including commission, 6) Alert if arbitrage profit > 0.05% after costs, 7) Execute trades atomically if profitable"
        },
        {
            prompt: "Create a carry trade simulator that backtests the JPY carry trade strategy over 2000-2023. Include interest rate data, FX prices, calculate returns, identify unwind events, and compute risk-adjusted returns (Sharpe ratio).",
            solution: "// Implementation: 1) Fetch historical interest rates (Fed, BoJ, RBA, etc.), 2) Fetch historical FX prices, 3) For each day: calculate carry income, mark-to-market FX, compute total return, 4) Detect 'unwind' events (VIX spikes, JPY rapid appreciation), 5) Calculate drawdowns, Sharpe ratios, max loss periods, 6) Compare to buy-and-hold equity returns"
        }
    ]
};

