export const cryptocurrencyMarkets = {
  title: 'Cryptocurrency Markets',
  slug: 'cryptocurrency-markets',
  description: 'Master the 24/7 digital asset markets - from Bitcoin to DeFi',
  content: `
# Cryptocurrency Markets

## Introduction: The Digital Asset Revolution

Cryptocurrency markets represent the newest and most volatile asset class:
- 💰 **$1+ Trillion** total market capitalization (varies wildly)
- 🌍 **24/7/365 Trading** - markets never close
- 🔗 **Blockchain-based** - transparent, immutable ledgers
- ⚡ **No intermediaries** - peer-to-peer transactions
- 🎢 **Extreme volatility** - Bitcoin has dropped 80%+ three times

**What makes crypto different from traditional markets:**
- Decentralized (no central authority)
- Programmable money (smart contracts)
- Global access (anyone with internet)
- Permissionless (no account approval needed)
- Transparent (all transactions public)

**What you'll learn:**
- Bitcoin and major cryptocurrencies
- Crypto exchanges and market structure
- DeFi (Decentralized Finance) basics
- Volatility and risk management
- Building crypto trading systems
- The 2022 crypto winter (Luna, FTX collapses)

---

## Major Cryptocurrencies

\`\`\`python
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from typing import Optional, Dict
import numpy as np

class CryptoCategory(Enum):
    STORE_OF_VALUE = "Store of Value"
    SMART_CONTRACT_PLATFORM = "Smart Contract Platform"
    STABLECOIN = "Stablecoin"
    EXCHANGE_TOKEN = "Exchange Token"
    DEFI = "DeFi Token"
    MEME = "Meme Coin"

@dataclass
class Cryptocurrency:
    """Representation of a cryptocurrency"""
    symbol: str
    name: str
    category: CryptoCategory
    launch_year: int
    max_supply: Optional[int]  # Total coins that will ever exist
    current_supply: int  # Currently circulating
    consensus: str  # PoW, PoS, etc.
    use_case: str
    typical_volatility: float  # Annual volatility
    
    def calculate_market_cap (self, price: float) -> float:
        """Calculate market capitalization"""
        return price * self.current_supply
    
    def calculate_fully_diluted_value (self, price: float) -> float:
        """Calculate FDV if all coins were minted"""
        if self.max_supply:
            return price * self.max_supply
        return self.calculate_market_cap (price)
    
    def get_inflation_rate (self) -> float:
        """Calculate annual supply inflation rate"""
        if self.max_supply and self.max_supply > self.current_supply:
            remaining = self.max_supply - self.current_supply
            # Simplified annual issuance
            return (remaining / self.current_supply) * 0.05  # Rough estimate
        return 0

# Major cryptocurrencies
CRYPTOCURRENCIES = {
    'BTC': Cryptocurrency(
        symbol="BTC",
        name="Bitcoin",
        category=CryptoCategory.STORE_OF_VALUE,
        launch_year=2009,
        max_supply=21_000_000,
        current_supply=19_500_000,
        consensus="Proof of Work (PoW)",
        use_case="Digital gold, store of value, peer-to-peer payments",
        typical_volatility=0.80  # 80% annual volatility!
    ),
    'ETH': Cryptocurrency(
        symbol="ETH",
        name="Ethereum",
        category=CryptoCategory.SMART_CONTRACT_PLATFORM,
        launch_year=2015,
        max_supply=None,  # No cap (but burning mechanism)
        current_supply=120_000_000,
        consensus="Proof of Stake (PoS)",
        use_case="Smart contracts, DeFi, NFTs, decentralized applications",
        typical_volatility=0.90
    ),
    'USDT': Cryptocurrency(
        symbol="USDT",
        name="Tether",
        category=CryptoCategory.STABLECOIN,
        launch_year=2014,
        max_supply=None,
        current_supply=90_000_000_000,  # $90B
        consensus="N/A (Centralized)",
        use_case="Stable value, trading pair, payments",
        typical_volatility=0.02  # Should be 0, but depeg risk
    ),
    'BNB': Cryptocurrency(
        symbol="BNB",
        name="Binance Coin",
        category=CryptoCategory.EXCHANGE_TOKEN,
        launch_year=2017,
        max_supply=200_000_000,
        current_supply=150_000_000,
        consensus="PoS (BNB Chain)",
        use_case="Trading fee discounts, gas on BNB Chain",
        typical_volatility=0.85
    ),
    'SOL': Cryptocurrency(
        symbol="SOL",
        name="Solana",
        category=CryptoCategory.SMART_CONTRACT_PLATFORM,
        launch_year=2020,
        max_supply=None,
        current_supply=400_000_000,
        consensus="Proof of History + PoS",
        use_case="High-speed smart contracts, low fees, NFTs",
        typical_volatility=1.20  # Very volatile!
    )
}

print("=== Major Cryptocurrencies ===\\n")

for symbol, crypto in CRYPTOCURRENCIES.items():
    print(f"{symbol}: {crypto.name}")
    print(f"  Category: {crypto.category.value}")
    print(f"  Launched: {crypto.launch_year}")
    print(f"  Supply: {crypto.current_supply:,} / {crypto.max_supply:,}" if crypto.max_supply else f"  Supply: {crypto.current_supply:,} (no cap)")
    print(f"  Consensus: {crypto.consensus}")
    print(f"  Use Case: {crypto.use_case}")
    print(f"  Volatility: {crypto.typical_volatility*100:.0f}% annually\\n")

# Example: Bitcoin market cap
btc = CRYPTOCURRENCIES['BTC']
btc_price = 43000  # $43K per BTC
market_cap = btc.calculate_market_cap (btc_price)
fdv = btc.calculate_fully_diluted_value (btc_price)

print(f"Bitcoin at \\$\{btc_price:,}:")
print(f"  Market Cap: \\$\{market_cap/1e9:.0f}B")
print(f"  Fully Diluted Value: \\$\{fdv/1e9:.0f}B")
print(f"  Remaining to mine: {btc.max_supply - btc.current_supply:,} BTC")
\`\`\`

**Market Dominance:**
- Bitcoin: ~45% of total crypto market cap
- Ethereum: ~18%
- Stablecoins: ~10%
- Everything else: ~27% (thousands of coins)

---

## Crypto Exchanges and Market Structure

Unlike traditional markets, crypto has NO central exchange.

\`\`\`python
from typing import Literal, List
import time

class CryptoExchange:
    """
    Model cryptocurrency exchange
    
    Key differences from traditional exchanges:
    - 24/7 trading
    - No circuit breakers
    - Custody risk (not your keys, not your coins)
    - Hacks and bankruptcies common
    """
    
    def __init__(self, 
                 name: str,
                 exchange_type: Literal['CEX', 'DEX'],
                 jurisdiction: str):
        self.name = name
        self.type = exchange_type  # CEX = Centralized, DEX = Decentralized
        self.jurisdiction = jurisdiction
        self.order_book = {}
        self.users = {}
    
    def get_characteristics (self) -> Dict:
        """Compare CEX vs DEX"""
        if self.type == 'CEX':
            return {
                'type': 'Centralized Exchange (CEX)',
                'examples': 'Coinbase, Binance, Kraken',
                'custody': 'Exchange holds your crypto (risk!)',
                'kyc_required': True,
                'liquidity': 'High',
                'fees': 'Low (0.1-0.5%)',
                'speed': 'Fast',
                'risks': [
                    'Hacks (Mt. Gox lost 850K BTC)',
                    'Bankruptcy (FTX)',
                    'Withdrawal freezes',
                    'Regulatory shutdown'
                ],
                'benefits': [
                    'Easy to use',
                    'Fiat on/off ramps',
                    'High liquidity',
                    'Customer support'
                ]
            }
        else:  # DEX
            return {
                'type': 'Decentralized Exchange (DEX)',
                'examples': 'Uniswap, PancakeSwap, dYdX',
                'custody': 'You control your crypto (your keys)',
                'kyc_required': False,
                'liquidity': 'Variable (depends on pools)',
                'fees': 'Higher (0.3%+ plus gas)',
                'speed': 'Slower (blockchain confirmation)',
                'risks': [
                    'Smart contract bugs',
                    'Impermanent loss (liquidity providers)',
                    'Slippage on large orders',
                    'Gas fees can be high'
                ],
                'benefits': [
                    'No custody risk',
                    'Permissionless',
                    'Censorship-resistant',
                    'Transparent (on-chain)'
                ]
            }

# Major exchanges
exchanges = {
    'Coinbase': CryptoExchange('Coinbase', 'CEX', 'USA'),
    'Binance': CryptoExchange('Binance', 'CEX', 'Multiple'),
    'Uniswap': CryptoExchange('Uniswap', 'DEX', 'Decentralized')
}

print("\\n=== Crypto Exchange Comparison ===\\n")

for name, exchange in exchanges.items():
    chars = exchange.get_characteristics()
    print(f"{name} ({chars['type']}):")
    print(f"  Custody: {chars['custody']}")
    print(f"  KYC: {'Required' if chars['kyc_required'] else 'Not required'}")
    print(f"  Fees: {chars['fees']}")
    print(f"  Top Risk: {chars['risks'][0]}\\n")
\`\`\`

**Key Events:**
- **2014: Mt. Gox** - 850K BTC stolen (~$450M then, $36B today)
- **2022: FTX** - $8B+ customer funds misused, bankruptcy
- **2023: Binance** - $4B+ fine for AML violations

**Lesson**: "Not your keys, not your coins" - holding crypto on exchanges is risky!

---

## Volatility: The Defining Characteristic

Crypto is 3-5x more volatile than stocks.

\`\`\`python
class CryptoVolatilityAnalysis:
    """
    Analyze crypto's extreme volatility
    """
    
    @staticmethod
    def compare_volatility() -> Dict:
        """Compare crypto vs traditional assets"""
        return {
            'Bitcoin': {
                'annual_volatility': 0.80,  # 80%
                'max_drawdown': -0.83,  # -83% (2017-2018)
                'interpretation': 'Can drop 50%+ multiple times per year'
            },
            'Ethereum': {
                'annual_volatility': 0.90,
                'max_drawdown': -0.95,  # -95% (2018)
                'interpretation': 'Even more volatile than Bitcoin'
            },
            'S&P 500': {
                'annual_volatility': 0.18,  # 18%
                'max_drawdown': -0.57,  # -57% (2008)
                'interpretation': 'Stable compared to crypto'
            },
            'Gold': {
                'annual_volatility': 0.15,
                'max_drawdown': -0.45,
                'interpretation': 'Traditional safe haven'
            }
        }
    
    @staticmethod
    def calculate_position_sizing(
        account_size: float,
        max_risk_pct: float,
        btc_volatility: float = 0.80
    ) -> Dict:
        """
        Calculate safe position size for crypto
        
        Rule: Don't risk more than you can afford to lose
        Crypto can drop 50%+ easily
        """
        # For 80% annual vol, expect ~60% moves in a year
        expected_max_move = btc_volatility * 0.75  # 60% move
        
        # Position size that limits loss to max_risk_pct
        safe_position_size = (account_size * max_risk_pct) / expected_max_move
        
        # As % of account
        position_pct = (safe_position_size / account_size) * 100
        
        return {
            'account_size': account_size,
            'max_risk_tolerance': max_risk_pct * 100,
            'expected_max_drawdown': expected_max_move * 100,
            'safe_position_size': safe_position_size,
            'position_as_pct_of_account': position_pct,
            'recommendation': f'Never put more than {position_pct:.0f}% in crypto'
        }

# Compare volatilities
vol_comparison = CryptoVolatilityAnalysis.compare_volatility()

print("\\n=== Volatility Comparison ===\\n")

for asset, data in vol_comparison.items():
    print(f"{asset}:")
    print(f"  Annual Volatility: {data['annual_volatility']*100:.0f}%")
    print(f"  Max Drawdown: {data['max_drawdown']*100:.0f}%")
    print(f"  {data['interpretation']}\\n")

# Position sizing example
sizing = CryptoVolatilityAnalysis.calculate_position_sizing(
    account_size=100_000,
    max_risk_pct=0.10  # 10% max loss tolerance
)

print("Position Sizing for $100K Account:")
print(f"  Max Risk Tolerance: {sizing['max_risk_tolerance']:.0f}%")
print(f"  Expected Max Drawdown: {sizing['expected_max_drawdown']:.0f}%")
print(f"  Safe Position Size: \\$\{sizing['safe_position_size']:,.0f}")
print(f"  As % of Account: {sizing['position_as_pct_of_account']:.0f}%")
print(f"\\n💡 {sizing['recommendation']}")
\`\`\`

**Bitcoin\'s Historical Drawdowns:**
- 2011: -93% (peak to trough)
- 2014: -83%
- 2018: -83%
- 2022: -77%

**Pattern**: Every cycle, Bitcoin drops 75-85% from peak. Yet it has recovered every time (so far).

---

## DeFi (Decentralized Finance) Basics

DeFi = Financial services without intermediaries.

\`\`\`python
class DeFiProtocol:
    """
    Model DeFi protocols
    """
    
    @staticmethod
    def explain_defi_categories() -> Dict:
        """
        Major categories of DeFi
        """
        return {
            'DEXs (Decentralized Exchanges)': {
                'examples': 'Uniswap, SushiSwap, PancakeSwap',
                'how_it_works': 'Automated Market Makers (AMM) using liquidity pools',
                'tvl': '$20B+',
                'use_case': 'Trade tokens without centralized exchange'
            },
            'Lending Protocols': {
                'examples': 'Aave, Compound, MakerDAO',
                'how_it_works': 'Deposit crypto as collateral, borrow against it',
                'tvl': '$25B+',
                'use_case': 'Earn yield on deposits, borrow without banks'
            },
            'Stablecoins': {
                'examples': 'USDC, DAI, USDT',
                'how_it_works': 'Pegged to $1, backed by collateral or algorithms',
                'tvl': '$120B+',
                'use_case': 'Stable value for trading and payments'
            },
            'Derivatives': {
                'examples': 'GMX, dYdX, Synthetix',
                'how_it_works': 'Trade perpetuals, options on-chain',
                'tvl': '$2B+',
                'use_case': 'Leverage trading, hedging'
            },
            'Yield Farming': {
                'examples': 'Yearn, Curve, Convex',
                'how_it_works': 'Automated strategies to maximize yield',
                'tvl': '$5B+',
                'use_case': 'Passive income on crypto holdings'
            }
        }
    
    @staticmethod
    def calculate_apy_vs_apr(
        apr: float,
        compounding_frequency: int = 365
    ) -> Dict:
        """
        DeFi quotes both APR and APY
        
        APR = Annual Percentage Rate (simple interest)
        APY = Annual Percentage Yield (compound interest)
        """
        # APY = (1 + APR/n)^n - 1
        apy = (1 + apr / compounding_frequency) ** compounding_frequency - 1
        
        return {
            'apr': apr * 100,
            'apy': apy * 100,
            'compounding': f'{compounding_frequency}x per year',
            'difference': (apy - apr) * 100,
            'interpretation': f'Daily compounding increases effective yield by {((apy/apr)-1)*100:.1f}%'
        }

# Explain DeFi categories
defi = DeFiProtocol()
categories = defi.explain_defi_categories()

print("\\n=== DeFi Protocol Categories ===\\n")

for category, info in categories.items():
    print(f"{category}:")
    print(f"  Examples: {info['examples']}")
    print(f"  How it works: {info['how_it_works']}")
    print(f"  Total Value Locked: {info['tvl']}\\n")

# APY calculation example
yield_calc = defi.calculate_apy_vs_apr (apr=0.12, compounding_frequency=365)

print("Yield Farming Example:")
print(f"  APR: {yield_calc['apr']:.2f}%")
print(f"  APY: {yield_calc['apy']:.2f}% (with daily compounding)")
print(f"  Difference: {yield_calc['difference']:.2f}%")
print(f"  {yield_calc['interpretation']}")
\`\`\`

**DeFi Risks:**1. **Smart contract bugs** - Code vulnerabilities (lost $12B+ to hacks)
2. **Impermanent loss** - Liquidity providers can lose vs holding
3. **Rug pulls** - Developers abandon project with funds
4. **Regulatory risk** - Many DeFi protocols may be securities

---

## The 2022 Crypto Winter

A case study in contagion and risk.

\`\`\`python
class Crypto2022Crisis:
    """
    Model the 2022 crypto collapse
    """
    
    @staticmethod
    def timeline() -> List[Dict]:
        """
        Timeline of collapses
        """
        return [
            {
                'date': 'May 2022',
                'event': 'Luna/UST Collapse',
                'what_happened': 'Algorithmic stablecoin UST lost $1 peg, death spiral',
                'losses': '$60B+ market cap evaporated',
                'victims': 'Retail investors, Do Kwon\'s ecosystem',
                'lesson': 'Algorithmic stables are fragile, death spirals are real'
            },
            {
                'date': 'June 2022',
                'event': 'Celsius Network Bankruptcy',
                'what_happened': 'Crypto lender froze withdrawals, filed bankruptcy',
                'losses': '$25B AUM, users couldn\'t withdraw',
                'victims': 'Retail depositors seeking yield',
                'lesson': 'Crypto banks have same risks as regular banks'
            },
            {
                'date': 'July 2022',
                'event': 'Three Arrows Capital (3AC) Liquidation',
                'what_happened': 'Major crypto hedge fund went bankrupt',
                'losses': '$10B+ AUM',
                'victims': 'Institutional lenders, investors',
                'lesson': 'Leverage kills, contagion spreads'
            },
            {
                'date': 'November 2022',
                'event': 'FTX Collapse',
                'what_happened': 'Sam Bankman-Fried used customer funds for Alameda losses',
                'losses': '$8B+ customer funds missing',
                'victims': 'Millions of FTX customers',
                'lesson': 'Even "reputable" exchanges can commit fraud'
            }
        ]
    
    @staticmethod
    def calculate_contagion_spread(
        initial_failure_size: float,
        leverage_ratio: float,
        num_connected_entities: int
    ) -> Dict:
        """
        Model how crypto contagion spreads
        
        Example: 3AC fails → can't repay lenders → lenders fail
        """
        # Initial loss
        direct_loss = initial_failure_size
        
        # Connected entities with exposure
        exposure_per_entity = initial_failure_size / num_connected_entities
        
        # If entities were leveraged, losses amplify
        amplified_loss_per_entity = exposure_per_entity * leverage_ratio
        total_secondary_losses = amplified_loss_per_entity * num_connected_entities
        
        # Total system loss
        total_loss = direct_loss + total_secondary_losses
        
        return {
            'initial_failure': direct_loss,
            'num_connected_entities': num_connected_entities,
            'leverage_ratio': leverage_ratio,
            'secondary_losses': total_secondary_losses,
            'total_system_loss': total_loss,
            'amplification_factor': total_loss / direct_loss,
            'interpretation': f'\${direct_loss/1e9:.0f}B failure became \${total_loss/1e9:.0f}B crisis'
        }

# Display timeline
crisis = Crypto2022Crisis()
timeline = crisis.timeline()

print("\\n=== 2022 Crypto Winter Timeline ===\\n")

for event in timeline:
    print(f"{event['date']}: {event['event']}")
    print(f"  What: {event['what_happened']}")
    print(f"  Losses: {event['losses']}")
    print(f"  Lesson: {event['lesson']}\\n")

# Model contagion
contagion = crisis.calculate_contagion_spread(
    initial_failure_size=10_000_000_000,  # $10B (3AC)
    leverage_ratio=5,  # 5x leverage
    num_connected_entities=20  # 20 major lenders
)

print("Contagion Analysis:")
print(f"  Initial Failure: \\$\{contagion['initial_failure']/1e9:.0f}B")
print(f"  Connected Entities: {contagion['num_connected_entities']}")
print(f"  Leverage Ratio: {contagion['leverage_ratio']}x")
print(f"  Secondary Losses: \\$\{contagion['secondary_losses']/1e9:.0f}B")
print(f"  Total System Loss: \\$\{contagion['total_system_loss']/1e9:.0f}B")
print(f"  Amplification: {contagion['amplification_factor']:.1f}x")
print(f"\\n💡 {contagion['interpretation']}")
\`\`\`

**Total 2022 Crypto Losses**: ~$2 trillion in market cap destroyed

---

## Building a Crypto Trading System

\`\`\`python
import requests
from typing import Optional

class CryptoTradingSystem:
    """
    Production crypto trading system
    """
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.positions = {}
        self.account_balance = {'USD': 100000, 'BTC': 0, 'ETH': 0}
    
    def get_price (self, symbol: str) -> Optional[Dict]:
        """
        Get current crypto price
        In production: WebSocket for real-time, REST for snapshots
        """
        try:
            # Example: Coinbase API (public endpoint)
            url = f"https://api.coinbase.com/v2/prices/{symbol}-USD/spot"
            response = requests.get (url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'symbol': symbol,
                    'price': float (data['data']['amount']),
                    'currency': 'USD',
                    'timestamp': datetime.now()
                }
        except Exception as e:
            print(f"Error fetching price: {e}")
        
        return None
    
    def calculate_position_size (self,
                               symbol: str,
                               risk_per_trade: float,
                               stop_loss_pct: float) -> Dict:
        """
        Kelly Criterion-based position sizing for crypto
        
        Crypto\'s high volatility requires small position sizes
        """
        available_capital = self.account_balance['USD']
        max_loss_dollars = available_capital * risk_per_trade
        
        # Position size = Risk$ / Stop Loss%
        position_size_dollars = max_loss_dollars / stop_loss_pct
        
        # Get current price
        quote = self.get_price (symbol)
        if not quote:
            return {'error': 'Could not get price'}
        
        position_size_coins = position_size_dollars / quote['price']
        
        return {
            'symbol': symbol,
            'current_price': quote['price'],
            'available_capital': available_capital,
            'risk_per_trade': risk_per_trade * 100,
            'stop_loss_pct': stop_loss_pct * 100,
            'max_loss_dollars': max_loss_dollars,
            'position_size_dollars': position_size_dollars,
            'position_size_coins': position_size_coins,
            'position_as_pct_of_capital': (position_size_dollars / available_capital) * 100
        }
    
    def implement_dollar_cost_averaging (self,
                                       symbol: str,
                                       total_amount: float,
                                       num_purchases: int,
                                       frequency_days: int) -> List[Dict]:
        """
        DCA: Invest fixed amount at regular intervals
        
        Reduces timing risk in volatile markets
        """
        amount_per_purchase = total_amount / num_purchases
        
        schedule = []
        for i in range (num_purchases):
            purchase_date = datetime.now() + timedelta (days=frequency_days * i)
            schedule.append({
                'purchase_number': i + 1,
                'date': purchase_date.strftime('%Y-%m-%d'),
                'amount_usd': amount_per_purchase,
                'note': 'Buy regardless of price'
            })
        
        return schedule

# Example usage
system = CryptoTradingSystem (api_key="test", api_secret="test")

print("\\n=== Crypto Trading System ===\\n")

# Position sizing
sizing = system.calculate_position_size(
    symbol='BTC',
    risk_per_trade=0.02,  # 2% risk per trade
    stop_loss_pct=0.15  # 15% stop loss (crypto is volatile!)
)

if 'error' not in sizing:
    print(f"Position Sizing for {sizing['symbol']}:")
    print(f"  Current Price: \\$\{sizing['current_price']:,.0f}")
    print(f"  Risk per Trade: {sizing['risk_per_trade']:.0f}%")
    print(f"  Stop Loss: {sizing['stop_loss_pct']:.0f}%")
    print(f"  Position Size: \\$\{sizing['position_size_dollars']:,.0f}")
    print(f"  = {sizing['position_size_coins']:.4f} BTC")
    print(f"  As % of Capital: {sizing['position_as_pct_of_capital']:.1f}%")

# DCA strategy
dca = system.implement_dollar_cost_averaging(
    symbol='BTC',
    total_amount=10000,
    num_purchases=10,
    frequency_days=30  # Monthly
)

print(f"\\n\\nDollar Cost Averaging Plan:")
print(f"Total Investment: $10,000 over 10 months\\n")
for purchase in dca[:3]:
    print(f"Purchase {purchase['purchase_number']}: {purchase['date']} - \\$\{purchase['amount_usd']:,.0f}")
print("...")
\`\`\`

---

## Summary

**Key Takeaways:**1. **24/7 Markets**: Crypto never sleeps - high stress, no breaks
2. **Extreme Volatility**: 80%+ annual vol, can drop 50%+ in weeks
3. **No Safety Net**: No FDIC, no circuit breakers, no reversals
4. **Custody Risk**: Exchanges can be hacked or commit fraud
5. **DeFi Revolution**: Financial services without intermediaries
6. **Regulatory Uncertainty**: Rules still being written
7. **High Risk, High Reward**: Can 10x or go to zero

**For Engineers:**
- 24/7 monitoring required
- WebSocket feeds for real-time data
- Self-custody requires secure key management
- Smart contract risks in DeFi
- Extreme volatility requires robust risk management

**Next Steps:**
- Module 17: Blockchain & DeFi deep dive
- Module 18: ML for crypto trading
- Module 14: Building crypto infrastructure

You now understand crypto markets - ready to build crypto systems!
`,
  exercises: [
    {
      prompt:
        'Build a crypto portfolio tracker that monitors holdings across multiple wallets (MetaMask, Coinbase, cold storage), fetches real-time prices from CoinGecko/CoinMarketCap, calculates total value, and alerts on 10%+ daily moves.',
      solution:
        '// Implementation: 1) Connect to wallet APIs (Web3, exchange APIs), 2) Fetch balances for BTC, ETH, ERC-20 tokens, 3) Get prices from multiple sources, 4) Calculate total value in USD, 5) Store historical values, 6) Calculate 24h change, 7) Send alerts via email/SMS/Telegram when >10% move, 8) Display on dashboard with charts',
    },
    {
      prompt:
        'Create a DCA (Dollar Cost Averaging) automation system that buys $X of Bitcoin weekly regardless of price. Calculate historical returns if this strategy was started 1/5/10 years ago vs lump sum investing.',
      solution:
        '// Implementation: 1) Connect to exchange API (Coinbase, Binance), 2) Schedule weekly purchase using cron/scheduler, 3) Execute market buy for fixed USD amount, 4) Track cost basis and holdings, 5) Backtest: fetch historical prices, simulate weekly purchases, 6) Compare DCA vs lump sum (invested all at start), 7) Calculate returns, volatility, max drawdown for both strategies, 8) Visualize cumulative returns over time',
    },
  ],
};
