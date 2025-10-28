export const orderExecutionTradingInfrastructure = {
  title: 'Order Execution & Trading Infrastructure',
  id: 'order-execution-trading-infrastructure',
  content: `
# Order Execution & Trading Infrastructure

## Introduction

Professional algorithmic trading requires robust execution infrastructure, reliable APIs, monitoring systems, and failsafe mechanisms. The difference between profitable and unprofitable trading often comes down to execution quality and infrastructure reliability.

**Key Components**:
1. **Broker API Integration**: Connect to markets
2. **Order Management System (OMS)**: Track and manage orders
3. **Position Management**: Real-time P&L tracking
4. **Risk Controls**: Pre-trade and post-trade checks
5. **Monitoring & Alerts**: System health and performance
6. **Data Infrastructure**: Real-time and historical data feeds
7. **Latency Optimization**: Speed matters

---

## Broker API Integration

### Alpaca API (Commission-Free)

\`\`\`python
"""
Complete Alpaca API integration for algorithmic trading
"""

import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import pandas as pd
import time
from typing import Dict, List, Optional

class AlpacaTrader:
    """
    Professional Alpaca API wrapper
    Handles orders, positions, data, and error recovery
    """
    
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        """
        Initialize Alpaca connection
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: Use paper trading (True) or live (False)
        """
        base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        self.paper = paper
        
        # Verify connection
        try:
            account = self.api.get_account()
            print(f"✓ Connected to Alpaca ({'paper' if paper else 'LIVE'})")
            print(f"  Account: {account.account_number}")
            print(f"  Equity: \\$\{float (account.equity):,.2f}")
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            raise
    
    # ========================================================================
    # ACCOUNT & POSITIONS
    # ========================================================================
    
    def get_account (self) -> Dict:
        """Get account information"""
        account = self.api.get_account()
        return {
            'equity': float (account.equity),
            'cash': float (account.cash),
            'buying_power': float (account.buying_power),
            'portfolio_value': float (account.portfolio_value),
            'pattern_day_trader': account.pattern_day_trader,
            'trading_blocked': account.trading_blocked,
            'account_blocked': account.account_blocked,
            'transfers_blocked': account.transfers_blocked
        }
    
    def get_positions (self) -> List[Dict]:
        """Get all open positions"""
        positions = self.api.list_positions()
        return [{
            'symbol': pos.symbol,
            'qty': int (pos.qty),
            'side': 'long' if int (pos.qty) > 0 else 'short',
            'avg_entry_price': float (pos.avg_entry_price),
            'current_price': float (pos.current_price),
            'market_value': float (pos.market_value),
            'cost_basis': float (pos.cost_basis),
            'unrealized_pl': float (pos.unrealized_pl),
            'unrealized_plpc': float (pos.unrealized_plpc),
            'unrealized_intraday_pl': float (pos.unrealized_intraday_pl),
            'unrealized_intraday_plpc': float (pos.unrealized_intraday_plpc)
        } for pos in positions]
    
    def get_position (self, symbol: str) -> Optional[Dict]:
        """Get specific position"""
        try:
            pos = self.api.get_position (symbol)
            return {
                'symbol': pos.symbol,
                'qty': int (pos.qty),
                'avg_entry_price': float (pos.avg_entry_price),
                'current_price': float (pos.current_price),
                'unrealized_pl': float (pos.unrealized_pl)
            }
        except:
            return None
    
    # ========================================================================
    # ORDER PLACEMENT
    # ========================================================================
    
    def place_market_order (self, symbol: str, qty: int, side: str = 'buy',
                          time_in_force: str = 'day') -> Dict:
        """
        Place market order
        
        Args:
            symbol: Stock symbol
            qty: Quantity (positive integer)
            side: 'buy' or 'sell'
            time_in_force: 'day', 'gtc', 'ioc', 'fok'
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force=time_in_force
            )
            return self._parse_order (order)
        except Exception as e:
            print(f"✗ Market order failed: {e}")
            return None
    
    def place_limit_order (self, symbol: str, qty: int, limit_price: float,
                         side: str = 'buy', time_in_force: str = 'day') -> Dict:
        """
        Place limit order
        
        Only executes at limit_price or better
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='limit',
                time_in_force=time_in_force,
                limit_price=limit_price
            )
            return self._parse_order (order)
        except Exception as e:
            print(f"✗ Limit order failed: {e}")
            return None
    
    def place_stop_loss (self, symbol: str, qty: int, stop_price: float,
                       side: str = 'sell', time_in_force: str = 'gtc') -> Dict:
        """
        Place stop-loss order
        
        Triggers market order when stop_price hit
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='stop',
                time_in_force=time_in_force,
                stop_price=stop_price
            )
            return self._parse_order (order)
        except Exception as e:
            print(f"✗ Stop-loss order failed: {e}")
            return None
    
    def place_bracket_order (self, symbol: str, qty: int, side: str,
                           take_profit_price: float, stop_loss_price: float) -> Dict:
        """
        Bracket order: Entry + Take Profit + Stop Loss
        
        Automatically sets profit target and stop loss
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='gtc',
                order_class='bracket',
                take_profit={'limit_price': take_profit_price},
                stop_loss={'stop_price': stop_loss_price}
            )
            return self._parse_order (order)
        except Exception as e:
            print(f"✗ Bracket order failed: {e}")
            return None
    
    # ========================================================================
    # ORDER MANAGEMENT
    # ========================================================================
    
    def get_orders (self, status: str = 'open') -> List[Dict]:
        """
        Get orders by status
        
        Args:
            status: 'open', 'closed', 'all'
        """
        orders = self.api.list_orders (status=status, limit=100)
        return [self._parse_order (order) for order in orders]
    
    def get_order (self, order_id: str) -> Dict:
        """Get specific order by ID"""
        order = self.api.get_order (order_id)
        return self._parse_order (order)
    
    def cancel_order (self, order_id: str) -> bool:
        """Cancel specific order"""
        try:
            self.api.cancel_order (order_id)
            return True
        except:
            return False
    
    def cancel_all_orders (self) -> bool:
        """Cancel all open orders"""
        try:
            self.api.cancel_all_orders()
            return True
        except:
            return False
    
    def _parse_order (self, order) -> Dict:
        """Parse order object to dict"""
        return {
            'id': order.id,
            'client_order_id': order.client_order_id,
            'symbol': order.symbol,
            'qty': int (order.qty),
            'filled_qty': int (order.filled_qty),
            'side': order.side,
            'type': order.type,
            'time_in_force': order.time_in_force,
            'limit_price': float (order.limit_price) if order.limit_price else None,
            'stop_price': float (order.stop_price) if order.stop_price else None,
            'filled_avg_price': float (order.filled_avg_price) if order.filled_avg_price else None,
            'status': order.status,
            'created_at': order.created_at,
            'updated_at': order.updated_at,
            'submitted_at': order.submitted_at,
            'filled_at': order.filled_at
        }
    
    # ========================================================================
    # MARKET DATA
    # ========================================================================
    
    def get_bars (self, symbol: str, timeframe: str = '1Day',
                start: str = None, end: str = None, limit: int = 100) -> pd.DataFrame:
        """
        Get historical bars
        
        Args:
            symbol: Stock symbol
            timeframe: '1Min', '5Min', '1Hour', '1Day'
            start: Start date (ISO format or datetime)
            end: End date
            limit: Number of bars
        """
        bars = self.api.get_bars(
            symbol,
            timeframe,
            start=start,
            end=end,
            limit=limit
        ).df
        
        return bars
    
    def get_latest_trade (self, symbol: str) -> Dict:
        """Get latest trade for symbol"""
        trade = self.api.get_latest_trade (symbol)
        return {
            'price': float (trade.price),
            'size': int (trade.size),
            'timestamp': trade.timestamp
        }
    
    def get_latest_quote (self, symbol: str) -> Dict:
        """Get latest quote (bid/ask)"""
        quote = self.api.get_latest_quote (symbol)
        return {
            'bid_price': float (quote.bid_price),
            'bid_size': int (quote.bid_size),
            'ask_price': float (quote.ask_price),
            'ask_size': int (quote.ask_size),
            'timestamp': quote.timestamp
        }
    
    # ========================================================================
    # RISK CHECKS
    # ========================================================================
    
    def check_buying_power (self, symbol: str, qty: int, price: float) -> bool:
        """Check if sufficient buying power"""
        account = self.get_account()
        required = qty * price
        return required <= account['buying_power']
    
    def check_position_limit (self, symbol: str, qty: int, max_position_size: int) -> bool:
        """Check if order exceeds position limit"""
        current_pos = self.get_position (symbol)
        current_qty = current_pos['qty'] if current_pos else 0
        new_qty = abs (current_qty + qty)
        return new_qty <= max_position_size
    
    def check_daily_loss_limit (self, max_daily_loss: float) -> bool:
        """Check if daily loss limit exceeded"""
        account = self.get_account()
        equity = account['equity']
        
        # Get starting equity (you'd cache this at market open)
        daily_pnl = equity - getattr (self, 'start_of_day_equity', equity)
        
        return daily_pnl >= -max_daily_loss


# ============================================================================
# INTERACTIVE BROKERS API
# ============================================================================

class InteractiveBrokersTrader:
    """
    Interactive Brokers integration via ib_insync
    
    More complex but supports more asset classes
    """
    
    def __init__(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 1):
        """
        Connect to IB Gateway/TWS
        
        Args:
            host: IB Gateway host
            port: 7497 (paper), 7496 (live)
            client_id: Unique client identifier
        """
        try:
            from ib_insync import IB, Stock, MarketOrder, LimitOrder
            self.ib = IB()
            self.ib.connect (host, port, clientId=client_id)
            print(f"✓ Connected to Interactive Brokers")
        except Exception as e:
            print(f"✗ IB connection failed: {e}")
            raise
    
    def place_market_order (self, symbol: str, qty: int, action: str = 'BUY'):
        """Place market order"""
        from ib_insync import Stock, MarketOrder
        
        contract = Stock (symbol, 'SMART', 'USD')
        order = MarketOrder (action, qty)
        
        trade = self.ib.placeOrder (contract, order)
        return trade
    
    def get_positions (self):
        """Get all positions"""
        positions = self.ib.positions()
        return [{
            'symbol': pos.contract.symbol,
            'position': pos.position,
            'avg_cost': pos.avgCost
        } for pos in positions]


# ============================================================================
# ORDER MANAGEMENT SYSTEM (OMS)
# ============================================================================

class OrderManagementSystem:
    """
    Complete OMS for tracking and managing orders across multiple brokers
    """
    
    def __init__(self):
        self.orders = []
        self.filled_orders = []
        self.canceled_orders = []
        self.positions = {}
    
    def submit_order (self, symbol: str, qty: int, order_type: str,
                    price: float = None, stop_price: float = None):
        """Submit order to OMS"""
        order = {
            'order_id': f"ORD_{len (self.orders)}",
            'symbol': symbol,
            'qty': qty,
            'order_type': order_type,
            'price': price,
            'stop_price': stop_price,
            'status': 'pending',
            'submitted_at': datetime.now(),
            'filled_qty': 0,
            'avg_fill_price': 0.0
        }
        
        self.orders.append (order)
        return order['order_id']
    
    def update_order_status (self, order_id: str, status: str,
                           filled_qty: int = 0, fill_price: float = 0):
        """Update order status"""
        for order in self.orders:
            if order['order_id'] == order_id:
                order['status'] = status
                order['filled_qty'] = filled_qty
                
                if filled_qty > 0:
                    # Update average fill price
                    prev_filled = order['filled_qty']
                    prev_value = prev_filled * order['avg_fill_price']
                    new_value = filled_qty * fill_price
                    order['avg_fill_price'] = (prev_value + new_value) / (prev_filled + filled_qty)
                
                if status == 'filled':
                    self.filled_orders.append (order)
                    self._update_position (order)
                elif status == 'canceled':
                    self.canceled_orders.append (order)
                
                break
    
    def _update_position (self, order):
        """Update position after fill"""
        symbol = order['symbol']
        qty = order['qty']
        price = order['avg_fill_price']
        
        if symbol not in self.positions:
            self.positions[symbol] = {
                'qty': 0,
                'avg_price': 0,
                'realized_pnl': 0
            }
        
        pos = self.positions[symbol]
        
        # Update position
        old_qty = pos['qty']
        old_avg = pos['avg_price']
        
        if old_qty * qty >= 0:  # Same direction
            # Update average price
            total_value = old_qty * old_avg + qty * price
            pos['qty'] = old_qty + qty
            pos['avg_price'] = total_value / pos['qty'] if pos['qty'] != 0 else 0
        else:  # Opposite direction (closing or flipping)
            # Realize P&L
            close_qty = min (abs (old_qty), abs (qty))
            pnl = close_qty * (price - old_avg) * np.sign (old_qty)
            pos['realized_pnl'] += pnl
            pos['qty'] = old_qty + qty
    
    def get_open_orders (self):
        """Get all open orders"""
        return [o for o in self.orders if o['status'] in ['pending', 'partial']]
    
    def get_filled_orders (self):
        """Get all filled orders"""
        return self.filled_orders
    
    def get_position (self, symbol: str):
        """Get current position"""
        return self.positions.get (symbol, {'qty': 0, 'avg_price': 0, 'realized_pnl': 0})


# ============================================================================
# REAL-TIME DATA STREAMING
# ============================================================================

class RealtimeDataStream:
    """
    WebSocket streaming for real-time market data
    """
    
    def __init__(self, api_key: str, secret_key: str):
        """Initialize Alpaca WebSocket"""
        import alpaca_trade_api as tradeapi
        self.api = tradeapi.REST(api_key, secret_key)
        
        # Callback functions
        self.on_trade = None
        self.on_quote = None
        self.on_bar = None
    
    def stream_trades (self, symbols: List[str]):
        """Stream real-time trades"""
        # Implementation depends on broker
        # Alpaca uses websocket-client
        pass
    
    def stream_quotes (self, symbols: List[str]):
        """Stream real-time quotes (bid/ask)"""
        pass


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

# Initialize trader (use your own keys!)
trader = AlpacaTrader(
    api_key="YOUR_API_KEY",
    secret_key="YOUR_SECRET_KEY",
    paper=True  # Start with paper trading!
)

# Check account
account = trader.get_account()
print(f"\\nAccount Status:")
print(f"  Equity: \\$\{account['equity']:,.2f}")
print(f"  Buying Power: \\$\{account['buying_power']:,.2f}")

# Get current positions
positions = trader.get_positions()
print(f"\\nOpen Positions: {len (positions)}")
for pos in positions:
    print(f"  {pos['symbol']}: {pos['qty']} shares @ \\$\{pos['avg_entry_price']:.2f}")
    print(f"    P&L: \\$\{pos['unrealized_pl']:.2f} ({pos['unrealized_plpc']:.2%})")

# Place orders (with risk checks)
symbol = "AAPL"
qty = 10
price = 150.00

if trader.check_buying_power (symbol, qty, price):
    order = trader.place_market_order (symbol, qty, 'buy')
    print(f"\\n✓ Order placed: {order['id']}")
else:
    print(f"\\n✗ Insufficient buying power")

# Monitor order
time.sleep(2)
order_status = trader.get_order (order['id'])
print(f"Order status: {order_status['status']}")

# Place bracket order with profit/loss targets
bracket = trader.place_bracket_order(
    symbol="MSFT",
    qty=5,
    side='buy',
    take_profit_price=160.00,  # Sell at profit
    stop_loss_price=145.00     # Sell at loss
)
\`\`\`

---

## Latency Optimization

\`\`\`python
"""
Reduce execution latency for better fills
"""

# 1. Co-location: Server in same datacenter as exchange
# 2. Direct market access: Skip retail broker routing
# 3. Compiled code: C++/Rust for speed-critical paths
# 4. Connection pooling: Reuse HTTP connections
# 5. Async I/O: Non-blocking network calls

import asyncio
import aiohttp

class LowLatencyTrader:
    """Optimized for speed"""
    
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        # Connection pooling
        connector = aiohttp.TCPConnector (limit=100, limit_per_host=30)
        self.session = aiohttp.ClientSession (connector=connector)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
    
    async def place_order_async (self, symbol, qty, side):
        """Non-blocking order placement"""
        async with self.session.post('/orders', json={
            'symbol': symbol,
            'qty': qty,
            'side': side
        }) as response:
            return await response.json()
    
    async def place_multiple_orders (self, orders):
        """Place many orders concurrently"""
        tasks = [self.place_order_async(**order) for order in orders]
        results = await asyncio.gather(*tasks)
        return results

# Usage
async def main():
    async with LowLatencyTrader() as trader:
        orders = [
            {'symbol': 'AAPL', 'qty': 100, 'side': 'buy'},
            {'symbol': 'MSFT', 'qty': 100, 'side': 'buy'},
            {'symbol': 'GOOGL', 'qty': 50, 'side': 'buy'}
        ]
        
        # Place all orders concurrently
        results = await trader.place_multiple_orders (orders)
        print(f"Placed {len (results)} orders")

# asyncio.run (main())
\`\`\`

---

## Key Takeaways

**Broker Selection**:
- **Alpaca**: Free, easy API, good for learning
- **Interactive Brokers**: Professional, all asset classes, complex API
- **TD Ameritrade**: Good API, requires approval
- **Crypto**: Coinbase, Binance, Kraken

**Infrastructure Essentials**:
1. **Redundancy**: Backup internet, power, servers
2. **Monitoring**: Alert on failures immediately  
3. **Logging**: Track every order, fill, error
4. **Kill Switch**: Ability to close all positions instantly
5. **Rate Limiting**: Respect API limits
6. **Error Handling**: Graceful degradation

**Best Practices**:
- Start with paper trading
- Test everything twice
- Monitor continuously
- Have manual override capability
- Document failure procedures
- Regular system audits

**Common Pitfalls**:
- No error handling
- Ignoring API rate limits
- No position reconciliation
- Missing market hours checks
- Insufficient logging
- No kill switch

**Remember**: Infrastructure failures can wipe out months of gains in minutes. Invest time in robust systems.
`,
};
