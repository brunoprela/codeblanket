export const cryptocurrencyMarketsQuiz = [
  {
    id: 'fm-1-6-q-1',
    question:
      'The 2022 crypto winter saw multiple cascading failures (Luna, Celsius, 3AC, FTX). Analyze how leverage and interconnectedness amplified losses. What risk management practices could have prevented these collapses? Compare crypto contagion to the 2008 financial crisis.',
    sampleAnswer: `**Root Causes of 2022 Crypto Contagion:**

**1. Excessive Leverage:**
- **Three Arrows Capital (3AC)**: ~5-10x leverage on illiquid positions
- **FTX/Alameda**: Used customer funds to cover trading losses
- **Celsius**: Borrowed short-term, lent long-term (maturity mismatch)
- **Problem**: When Luna collapsed (-99%), leveraged entities got margin called simultaneously

**2. Interconnectedness Without Transparency:**
\`\`\`
Luna Collapse (\$60B) 
    ↓ (3AC held $200M Luna)
3AC Bankruptcy (\$10B)
    ↓ (3AC owed money to...)
Celsius, Voyager, BlockFi (lent to 3AC)
    ↓ (can't repay depositors)
FTX "bails out" some firms
    ↓ (but FTX itself was insolvent)
FTX Collapse (\$8B customer funds missing)
\`\`\`

**3. Lack of Transparency:**
- 3AC didn't disclose their leverage or Luna exposure
- FTX hid that they were using customer deposits
- No regulatory reporting requirements
- "Trust me bro" accounting

**Risk Management That Could Have Prevented This:**

**For Lenders (Celsius, BlockFi):**
- ✅ **Due diligence on borrowers**: Verify collateral, check leverage ratios
- ✅ **Diversification**: Don't lend 20% to one entity (3AC)
- ✅ **Collateral requirements**: Overcollateralize loans (150%+)
- ✅ **Real-time monitoring**: Mark-to-market collateral daily
- ✅ **Stress testing**: Model "what if Luna goes to zero?"

**For Exchanges (FTX):**
- ✅ **Segregate customer funds**: NEVER use for trading
- ✅ **Proof of reserves**: Publish audited attestations
- ✅ **Independent custody**: Third-party custodians
- ✅ **Regulatory compliance**: Follow banking regulations

**For Investors:**
- ✅ **Self-custody**: "Not your keys, not your coins"
- ✅ **Diversify across platforms**: Don't keep everything on one exchange
- ✅ **Understand yield source**: 20% APY from where? (Celsius offered 18%!)
- ✅ **Position sizing**: Never invest more than you can afford to lose

**Comparison to 2008 Financial Crisis:**

**Similarities:**1. **Leverage**: Both had excessive leverage (banks 30x+, crypto funds 10x+)
2. **Interconnectedness**: Lehman → AIG → banks; 3AC → lenders → FTX
3. **Opacity**: CDOs were opaque; crypto lending was opaque
4. **Hubris**: "Housing never goes down" / "Luna can't fail"
5. **Contagion**: One failure triggers cascade

**Differences:**1. **Bailouts**: Governments bailed out banks (TARP); crypto had no bailout (good/bad?)
2. **Systemic risk**: 2008 threatened global economy; crypto was ~$2T isolated bubble
3. **Leverage limits**: Banks have capital requirements; crypto has none
4. **Transparency**: Banks report to regulators; crypto doesn't (yet)
5. **Speed**: 2008 played out over years; crypto collapsed in months

**Lessons for Engineering Trading Systems:**
\`\`\`python
class RiskManagement:
    def check_counterparty_risk (self, entity):
        # Don't trust, verify
        assert entity.leverage < 3  # Low leverage only
        assert entity.collateral_ratio > 1.5  # Overcollateralized
        assert entity.diversification_score > 0.8  # Not concentrated
        
    def stress_test (self, portfolio):
        # Model extreme scenarios
        scenarios = [
            'BTC drops 80%',
            'Top holding goes to zero',
            'Main exchange goes bankrupt'
        ]
        for scenario in scenarios:
            loss = self.calculate_loss (portfolio, scenario)
            assert loss < portfolio.total_value * 0.2  # Max 20% loss
\`\`\`

**Bottom Line**: Crypto\'s "move fast and break things" culture broke $100B+ in 2022. Traditional finance's boring risk management (capital requirements, stress tests, segregated accounts) exists for a reason.`,
    keyPoints: [
      '2022 cascade: Luna → 3AC → Celsius/BlockFi → FTX (\$100B+ losses)',
      'Root causes: 10x leverage, hidden interconnections, no transparency',
      'Prevention: Segregate funds, stress test, overcollateralize, diversify counterparties',
      'Similar to 2008: Leverage, contagion, opacity. Different: No bailout, faster collapse',
      'Lesson: Traditional risk management (boring but works) beats "move fast" in finance',
    ],
  },
  {
    id: 'fm-1-6-q-2',
    question:
      'Compare and contrast Centralized Exchanges (CEX) and Decentralized Exchanges (DEX). For a quantitative trading firm, which is more suitable and why? Design a hybrid system that uses both.',
    sampleAnswer: `**CEX vs DEX Comparison:**

| Aspect | CEX (Coinbase, Binance) | DEX (Uniswap, dYdX) |
|--------|-------------------------|---------------------|
| **Custody** | Exchange holds crypto (risk) | You hold crypto (self-custody) |
| **Liquidity** | High (order books) | Variable (liquidity pools) |
| **Speed** | Fast (centralized matching) | Slower (blockchain confirmation) |
| **Fees** | Low (0.1-0.5%) | Higher (0.3%+ plus gas) |
| **Slippage** | Low on liquid pairs | High on large orders |
| **KYC** | Required | Not required (anonymous) |
| **Custody Risk** | Exchange can be hacked/bankrupt | Smart contract risk |
| **Censorship** | Can freeze accounts | Permissionless |
| **API** | Fast REST/WebSocket | On-chain (slower) |
| **Leverage** | Available (up to 100x) | Limited |

**For Quantitative Trading Firms:**

**CEX is Better For:**1. **High-frequency trading**: Need low latency (<10ms)
2. **Large size**: Deep liquidity, low slippage
3. **Leverage**: Many strategies require margin
4. **Fiat on/off-ramps**: Convert to USD for risk management
5. **Multiple pairs**: Trade hundreds of pairs simultaneously
6. **API reliability**: Stable REST/WebSocket APIs

**DEX is Better For:**1. **Long-term holds**: Self-custody reduces counterparty risk
2. **New tokens**: DEXs list tokens faster than CEXs
3. **Arbitrage**: Cross-DEX price differences
4. **DeFi strategies**: Yield farming, liquidity provision
5. **Regulatory uncertainty**: No KYC, permissionless
6. **Transparency**: All transactions on-chain (auditable)

**Hybrid System Design:**

\`\`\`python
class HybridCryptoTradingSystem:
    """
    Use CEX for trading, DEX for custody and new opportunities
    """
    
    def __init__(self):
        # CEX connections
        self.cex_accounts = {
            'Coinbase': CoinbaseAPI(),
            'Binance': BinanceAPI(),
            'Kraken': KrakenAPI()
        }
        
        # DEX connections
        self.dex_protocols = {
            'Uniswap': UniswapV3(),
            'PancakeSwap': PancakeSwap(),
            'dYdX': dYdXV4()
        }
        
        # Self-custody wallets
        self.cold_storage = ColdWallet()  # Hardware wallet
        self.hot_wallet = HotWallet()  # For DEX trading
        
        # Target allocation
        self.allocation_strategy = {
            'cex_trading': 0.30,  # 30% on CEXs for active trading
            'dex_liquidity': 0.20,  # 20% providing liquidity on DEXs
            'cold_storage': 0.50   # 50% in cold storage (custody)
        }
    
    def execute_trade (self, pair, size, urgency):
        """
        Route order to best venue
        """
        if urgency == 'high' or size < 100_000:
            # CEX for speed and liquidity
            return self.cex_trade (pair, size)
        else:
            # DEX for large orders (split over time)
            return self.dex_trade_twap (pair, size, duration_hours=24)
    
    def cex_trade (self, pair, size):
        """Trade on centralized exchange"""
        # 1. Check liquidity across CEXs
        best_exchange = self.find_best_liquidity (pair)
        
        # 2. Execute with smart order routing
        return best_exchange.market_order (pair, size)
    
    def dex_trade_twap (self, pair, size, duration_hours):
        """
        TWAP on DEX to minimize slippage
        
        Break large order into small chunks over time
        """
        chunk_size = size / (duration_hours * 4)  # 4 trades per hour
        
        for i in range (int (duration_hours * 4)):
            # Execute small trade
            self.dex_protocols['Uniswap'].swap (pair, chunk_size)
            
            # Wait 15 minutes
            time.sleep(900)
    
    def rebalance_allocation (self):
        """
        Maintain target allocation
        
        Rebalance daily:
        - Trading profits on CEX → withdraw to cold storage
        - New trades needed → transfer from hot wallet to CEX
        """
        total_value = self.calculate_total_value()
        
        # Current allocation
        cex_value = sum (cex.get_balance() for cex in self.cex_accounts.values())
        dex_value = sum (dex.get_balance() for dex in self.dex_protocols.values())
        cold_value = self.cold_storage.get_balance()
        
        # Target allocation
        target_cex = total_value * self.allocation_strategy['cex_trading']
        target_cold = total_value * self.allocation_strategy['cold_storage']
        
        # Rebalance
        if cex_value > target_cex * 1.1:  # 10% buffer
            # Withdraw excess to cold storage
            excess = cex_value - target_cex
            self.withdraw_to_cold_storage (excess)
        
        if cex_value < target_cex * 0.9:
            # Transfer from hot wallet to CEX
            shortfall = target_cex - cex_value
            self.transfer_to_cex (shortfall)
    
    def arbitrage_cex_dex (self):
        """
        Arbitrage price differences between CEX and DEX
        
        Example: BTC $43K on Coinbase, $43.2K on Uniswap
        → Buy on Coinbase, sell on Uniswap, profit $200
        """
        for pair in ['BTC-USD', 'ETH-USD']:
            cex_price = self.get_cex_price (pair)
            dex_price = self.get_dex_price (pair)
            
            spread = abs (dex_price - cex_price) / cex_price
            
            if spread > 0.002:  # 0.2% spread (covers fees)
                if cex_price < dex_price:
                    # Buy CEX, sell DEX
                    self.cex_trade (pair, size=10000)
                    self.dex_protocols['Uniswap'].swap (pair, -10000)
                else:
                    # Buy DEX, sell CEX
                    self.dex_protocols['Uniswap'].swap (pair, 10000)
                    self.cex_trade (pair, size=-10000)
    
    def provide_liquidity_dex (self, pool, amount):
        """
        Earn fees by providing liquidity on DEX
        
        Uniswap V3: Concentrated liquidity in price ranges
        """
        # Example: Provide ETH-USDC liquidity
        # Earn 0.3% fee on every trade in the pool
        
        return self.dex_protocols['Uniswap'].add_liquidity(
            pool=pool,
            amount=amount,
            price_range=(1800, 2200)  # Concentrated around $2K
        )

# Usage
system = HybridCryptoTradingSystem()

# Active trading on CEX
system.execute_trade('BTC-USD', size=50000, urgency='high')

# Long-term hold in cold storage
system.cold_storage.deposit (amount=1000000)

# Earn yield on DEX
system.provide_liquidity_dex (pool='ETH-USDC', amount=100000)

# Daily rebalancing
system.rebalance_allocation()

# Arbitrage opportunities
system.arbitrage_cex_dex()
\`\`\`

**Risk Management:**1. **CEX risk**: Never keep more than 30% on exchanges (FTX taught us)
2. **DEX risk**: Audit smart contracts before using
3. **Gas risk**: ETH gas can spike to $100+ per transaction
4. **Slippage**: Use TWAP for large DEX orders

**Bottom Line**: 
- **CEX for speed and liquidity** (but withdraw daily)
- **DEX for custody and new opportunities** (but expect higher costs)
- **Hybrid approach reduces risk** while maximizing opportunities`,
    keyPoints: [
      'CEX: Fast (centralized), liquid, low fees, but custody risk (FTX!)',
      'DEX: Self-custody, permissionless, but higher fees and gas costs',
      'Quant firms: CEX for HFT/leverage, DEX for custody/new tokens',
      'Hybrid: 30% CEX trading, 20% DEX liquidity, 50% cold storage',
      'Risk management: Daily CEX withdrawals, smart contract audits, TWAP for size',
    ],
  },
  {
    id: 'fm-1-6-q-3',
    question:
      'Crypto volatility is 3-5x higher than stocks. Design a quantitative risk management framework specifically for crypto that includes position sizing, stop losses, correlation analysis, and tail risk hedging. How would you adapt a traditional equity trading strategy for crypto?',
    sampleAnswer: `**Crypto Risk Management Framework:**

**Challenge**: Crypto can drop 50%+ in weeks, correlations break down in crashes, and there are no circuit breakers.

**Framework Components:**

**1. Position Sizing (Kelly Criterion Modified for Crypto)**

\`\`\`python
class CryptoPositionSizing:
    """
    Kelly Criterion for crypto (use fractional Kelly due to volatility)
    """
    
    def kelly_position_size (self, 
                           win_rate: float,
                           avg_win: float, 
                           avg_loss: float,
                           kelly_fraction: float = 0.25) -> float:
        """
        Kelly formula: f = (p*w - q*l) / w
        
        p = win rate
        w = avg win
        q = loss rate (1-p)
        l = avg loss
        
        For crypto: Use 1/4 Kelly (too volatile for full Kelly)
        """
        q = 1 - win_rate
        kelly = (win_rate * avg_win - q * avg_loss) / avg_win
        
        # Fractional Kelly for crypto
        safe_kelly = kelly * kelly_fraction
        
        return max(0, min (safe_kelly, 0.10))  # Cap at 10% per position
    
    def volatility_adjusted_sizing (self,
                                   symbol: str,
                                   account_size: float,
                                   target_volatility: float = 0.15) -> float:
        """
        Size position inversely to volatility
        
        Higher vol → smaller position
        """
        symbol_volatility = self.get_volatility (symbol)
        
        # Scale factor
        scale = target_volatility / symbol_volatility
        
        # Base position size (e.g., 10%)
        base_size = account_size * 0.10
        
        # Adjusted size
        adjusted_size = base_size * scale
        
        return min (adjusted_size, account_size * 0.15)  # Max 15%

# Example
sizing = CryptoPositionSizing()

# Kelly sizing
position = sizing.kelly_position_size(
    win_rate=0.55,      # 55% win rate
    avg_win=0.30,       # 30% avg gain
    avg_loss=0.15,      # 15% avg loss
    kelly_fraction=0.25  # Use 1/4 Kelly
)
print(f"Kelly Position Size: {position*100:.1f}% of account")

# Volatility-adjusted
btc_position = sizing.volatility_adjusted_sizing(
    symbol='BTC',
    account_size=100000,
    target_volatility=0.15
)
print(f"BTC Position Size: \${btc_position:,.0f}")
\`\`\`

**2. Dynamic Stop Losses (Volatility-Adjusted)**

\`\`\`python
class CryptoStopLoss:
    """
    Volatility-adjusted stop losses
    """
    
    def calculate_atr_stop (self, 
                          symbol: str,
                          entry_price: float,
                          atr_multiplier: float = 2.0) -> float:
        """
        ATR-based stop loss
        
        ATR = Average True Range (measures volatility)
        Stop = Entry - (ATR * multiplier)
        
        Higher volatility → wider stops
        """
        atr = self.get_atr (symbol, period=14)
        
        stop_distance = atr * atr_multiplier
        stop_price = entry_price - stop_distance
        
        return stop_price
    
    def trailing_stop (self,
                     entry_price: float,
                     current_price: float,
                     atr: float,
                     multiplier: float = 3.0) -> float:
        """
        Trailing stop that moves up as price rises
        """
        # Initial stop
        initial_stop = entry_price - (atr * multiplier)
        
        # Trailing stop (moves up, never down)
        trailing_stop = current_price - (atr * multiplier)
        
        return max (initial_stop, trailing_stop)

# Example
stop_loss = CryptoStopLoss()

btc_entry = 43000
btc_current = 46000
btc_atr = 2000  # $2K ATR

stop = stop_loss.trailing_stop(
    entry_price=btc_entry,
    current_price=btc_current,
    atr=btc_atr,
    multiplier=3.0
)
print(f"BTC Trailing Stop: \${stop:,.0f}")
print(f"Protects \${btc_current - stop:,.0f} profit")
\`\`\`

**3. Correlation Analysis (Breaks Down in Crashes)**

\`\`\`python
class CryptoCorrelationRisk:
    """
    Monitor correlations (they go to 1 in crashes)
    """
    
    def calculate_rolling_correlation (self,
                                     asset1_returns: np.array,
                                     asset2_returns: np.array,
                                     window: int = 30) -> np.array:
        """
        Rolling correlation
        
        WARNING: In crypto crashes, all correlations → 1
        (Everything drops together)
        """
        rolling_corr = pd.Series (asset1_returns).rolling (window).corr(
            pd.Series (asset2_returns)
        )
        
        return rolling_corr
    
    def diversification_score (self, portfolio: Dict) -> float:
        """
        True diversification accounting for correlations
        
        Score = Σ(weight_i) / sqrt(Σ(weight_i * weight_j * corr_ij))
        """
        n = len (portfolio)
        weights = np.array([portfolio[asset]['weight'] for asset in portfolio])
        
        # Correlation matrix
        corr_matrix = self.get_correlation_matrix (list (portfolio.keys()))
        
        # Portfolio variance
        portfolio_var = weights.T @ corr_matrix @ weights
        
        # Diversification ratio
        weighted_avg_vol = np.sum (weights)
        portfolio_vol = np.sqrt (portfolio_var)
        
        return weighted_avg_vol / portfolio_vol  # >1 means diversification

# Example
corr = CryptoCorrelationRisk()

# In normal times
btc_eth_corr = 0.7  # Moderate correlation

# In crash (March 2020, May 2021, Nov 2022)
crash_corr = 0.95  # Everything dumps together!

print("Crypto Correlation Patterns:")
print(f"Normal: {btc_eth_corr:.2f}")
print(f"Crash: {crash_corr:.2f} (diversification fails)")
\`\`\`

**4. Tail Risk Hedging**

\`\`\`python
class CryptoTailRiskHedging:
    """
    Protect against 50%+ crashes (common in crypto)
    """
    
    def buy_put_options (self,
                       symbol: str,
                       portfolio_value: float,
                       hedge_ratio: float = 0.50) -> Dict:
        """
        Buy out-of-the-money puts
        
        Example: Long $100K BTC, buy $50K 30% OTM puts
        Cost: ~2-5% annually
        Protection: If BTC drops 50%, puts offset losses
        """
        current_price = self.get_price (symbol)
        
        # 30% OTM puts
        strike = current_price * 0.70
        
        # Hedge 50% of portfolio
        hedge_amount = portfolio_value * hedge_ratio
        
        # Buy puts
        premium = self.get_option_premium (symbol, strike, expiry='30d')
        contracts = hedge_amount / (current_price * 100)  # 100 coins per contract
        
        total_cost = premium * contracts * 100
        
        return {
            'strategy': 'Put Protection',
            'portfolio_value': portfolio_value,
            'hedge_amount': hedge_amount,
            'strike': strike,
            'premium': total_cost,
            'annual_cost_pct': (total_cost / portfolio_value) * 12,  # Monthly premium
            'protection': 'Limits downside below strike'
        }
    
    def stablecoin_allocation (self,
                             portfolio_value: float,
                             vol_target: float = 0.40) -> float:
        """
        Allocate to stablecoins as volatility buffer
        
        Higher volatility → more stablecoins
        """
        current_vol = self.get_portfolio_volatility()
        
        # If vol is 80% but we want 40%, hold 50% in stables
        stablecoin_allocation = max(0, (current_vol - vol_target) / current_vol)
        
        stablecoin_value = portfolio_value * stablecoin_allocation
        
        return stablecoin_value

# Example
hedge = CryptoTailRiskHedging()

put_hedge = hedge.buy_put_options(
    symbol='BTC',
    portfolio_value=100000,
    hedge_ratio=0.50
)

print("\\nTail Risk Hedging:")
print(f"Portfolio: \${put_hedge['portfolio_value']:,}")
print(f"Hedge: \${put_hedge['hedge_amount']:,}")
print(f"Strike: \${put_hedge['strike']:,.0f}")
print(f"Annual Cost: {put_hedge['annual_cost_pct']:.1f}%")
\`\`\`

**5. Adapting Equity Strategy to Crypto**

| Equity Strategy Component | Crypto Adaptation |
|--------------------------|-------------------|
| **Position Size**: 5-10% per stock | **Crypto**: 2-5% (higher vol) |
| **Stop Loss**: 2-3% | **Crypto**: 10-15% (volatile) |
| **Holding Period**: Weeks-months | **Crypto**: Days-weeks (faster moves) |
| **Diversification**: 20-30 stocks | **Crypto**: 5-10 coins (correlations high) |
| **Leverage**: 2x margin max | **Crypto**: No leverage (too risky) |
| **Rebalancing**: Monthly | **Crypto**: Weekly (volatility changes fast) |

**Bottom Line Framework:**
\`\`\`python
class CryptoRiskFramework:
    """Complete risk management system"""
    
    def __init__(self):
        self.max_position_size = 0.10  # 10% max per coin
        self.max_portfolio_risk = 0.02  # 2% risk per trade
        self.stop_loss_multiplier = 3.0  # 3x ATR
        self.hedge_ratio = 0.20  # Hedge 20% of portfolio
        self.stablecoin_floor = 0.25  # Always keep 25% in stables
        self.correlation_threshold = 0.80  # Reduce size if corr > 0.8
        
    def check_risk_limits (self, proposed_trade):
        # 1. Position size OK?
        assert proposed_trade.size <= self.max_position_size
        
        # 2. Portfolio risk OK?
        assert proposed_trade.risk <= self.max_portfolio_risk
        
        # 3. Stop loss in place?
        assert proposed_trade.stop_loss is not None
        
        # 4. Diversification OK?
        assert self.portfolio.diversification_score() > 1.5
        
        # 5. Tail hedges active?
        assert self.portfolio.hedge_ratio >= self.hedge_ratio
        
        return "Trade approved"
\`\`\`

**Key Insight**: Crypto\'s volatility means traditional risk rules (2% stops, 10% positions) will get you killed. Scale everything by 3-5x: wider stops, smaller positions, more hedging.`,
    keyPoints: [
      'Crypto vol 3-5x stocks: Adapt position sizing (2-5% vs 5-10%), stops (10-15% vs 2-3%)',
      'Kelly sizing: Use 1/4 Kelly for crypto (too volatile for full Kelly)',
      'ATR stops: 3x ATR for crypto vs 2x for stocks (wider stops needed)',
      'Correlations spike to 0.95 in crashes (diversification fails)',
      'Tail hedging: OTM puts (cost 2-5% annually) or stablecoin allocation (25%+ floor)',
    ],
  },
];
