export const dataNormalization = {
    title: 'Data Normalization',
    id: 'data-normalization',
    content: `
# Data Normalization

## Introduction

Market data arrives from multiple exchanges with different formats, price scales, and conventions. Normalization standardizes this data into a unified format for consistent strategy execution.

**Why Normalization Matters:**
- **Multi-exchange trading**: AAPL trades on NASDAQ, NYSE, ARCA - different formats
- **Price consistency**: Some venues quote in decimals, others in ticks
- **Volume aggregation**: Combine volume across venues
- **Corporate actions**: Stock splits, dividends affect historical prices
- **Symbol mapping**: AAPL vs AAPL.O vs AAPL.US

**Real-World Examples:**
- **Consolidated tape**: NBBO (National Best Bid Offer) combines all US exchanges
- **Bloomberg**: Normalizes 300+ exchanges into single API
- **Quant firms**: Build normalization layer handling 50+ venues

This section covers multi-exchange consolidation, price adjustments, corporate actions, and production normalization pipelines.

---

## Multi-Exchange Data Consolidation

### Challenge: Different Exchange Formats

\`\`\`python
# Raw data from different exchanges

# NASDAQ ITCH format
nasdaq_quote = {
    'symbol': 'AAPL',
    'bid_price': 15024,  # Price in hundredths of cents (150.24)
    'bid_size': 500,
    'venue': 'XNAS'
}

# NYSE format
nyse_quote = {
    'symbol': 'AAPL',
    'bid': 150.24,  # Decimal price
    'bid_qty': 5,  # Round lots (5 × 100 = 500 shares)
    'exchange': 'XNYS'
}

# ARCA format  
arca_quote = {
    'ticker': 'AAPL',
    'bid_px': '150.24',  # String price
    'bid_sz': '500',  # String size
    'src': 'ARCA'
}
\`\`\`

### Unified Quote Format

\`\`\`python
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

@dataclass
class NormalizedQuote:
    """Unified quote format across all exchanges"""
    symbol: str  # Standardized symbol
    exchange: str  # Venue code (NASDAQ, NYSE, ARCA)
    timestamp: datetime  # Microsecond precision
    
    # Prices in Decimal (exact precision)
    bid_price: Decimal
    ask_price: Decimal
    
    # Sizes in shares (not lots)
    bid_size: int
    ask_size: int
    
    # Metadata
    sequence_number: Optional[int] = None
    conditions: Optional[str] = None  # Trade conditions

class DataNormalizer:
    """Normalize data from multiple exchanges"""
    
    def __init__(self):
        # Exchange-specific configurations
        self.price_scales = {
            'NASDAQ': 10000,  # Prices in 1/10000ths
            'NYSE': 1,  # Decimal prices
            'ARCA': 1
        }
        
        self.size_multipliers = {
            'NASDAQ': 1,  # Shares
            'NYSE': 100,  # Round lots
            'ARCA': 1
        }
    
    def normalize_nasdaq(self, raw: dict) -> NormalizedQuote:
        """Normalize NASDAQ ITCH data"""
        return NormalizedQuote(
            symbol=raw['symbol'],
            exchange='NASDAQ',
            timestamp=datetime.fromtimestamp(raw['timestamp'] / 1e9),
            bid_price=Decimal(raw['bid_price']) / Decimal('10000'),
            ask_price=Decimal(raw['ask_price']) / Decimal('10000'),
            bid_size=raw['bid_size'],
            ask_size=raw['ask_size'],
            sequence_number=raw.get('sequence')
        )
    
    def normalize_nyse(self, raw: dict) -> NormalizedQuote:
        """Normalize NYSE data"""
        return NormalizedQuote(
            symbol=raw['symbol'],
            exchange='NYSE',
            timestamp=datetime.fromisoformat(raw['time']),
            bid_price=Decimal(str(raw['bid'])),
            ask_price=Decimal(str(raw['ask'])),
            bid_size=raw['bid_qty'] * 100,  # Convert lots to shares
            ask_size=raw['ask_qty'] * 100,
            conditions=raw.get('conditions')
        )
    
    def normalize(self, raw: dict, exchange: str) -> NormalizedQuote:
        """Normalize data from any exchange"""
        if exchange == 'NASDAQ':
            return self.normalize_nasdaq(raw)
        elif exchange == 'NYSE':
            return self.normalize_nyse(raw)
        # Add more exchanges...
\`\`\`

---

## National Best Bid Offer (NBBO)

\`\`\`python
from typing import Dict, List
from decimal import Decimal

class NBBOCalculator:
    """Calculate National Best Bid Offer from multiple exchanges"""
    
    def __init__(self):
        # Current quotes from each exchange
        self.exchange_quotes: Dict[str, NormalizedQuote] = {}
        self.nbbo_history: List[dict] = []
    
    def update_quote(self, quote: NormalizedQuote):
        """Update quote from exchange and recalculate NBBO"""
        self.exchange_quotes[quote.exchange] = quote
        
        # Calculate new NBBO
        nbbo = self.calculate_nbbo()
        
        if nbbo:
            self.nbbo_history.append(nbbo)
        
        return nbbo
    
    def calculate_nbbo(self) -> dict:
        """Find best bid (highest) and best ask (lowest) across all exchanges"""
        if not self.exchange_quotes:
            return None
        
        # Find best bid (highest price)
        best_bid = None
        best_bid_exchange = None
        best_bid_size = 0
        
        for exchange, quote in self.exchange_quotes.items():
            if best_bid is None or quote.bid_price > best_bid:
                best_bid = quote.bid_price
                best_bid_exchange = exchange
                best_bid_size = quote.bid_size
        
        # Find best ask (lowest price)
        best_ask = None
        best_ask_exchange = None
        best_ask_size = 0
        
        for exchange, quote in self.exchange_quotes.items():
            if best_ask is None or quote.ask_price < best_ask:
                best_ask = quote.ask_price
                best_ask_exchange = exchange
                best_ask_size = quote.ask_size
        
        return {
            'symbol': list(self.exchange_quotes.values())[0].symbol,
            'bid': best_bid,
            'bid_exchange': best_bid_exchange,
            'bid_size': best_bid_size,
            'ask': best_ask,
            'ask_exchange': best_ask_exchange,
            'ask_size': best_ask_size,
            'timestamp': datetime.now()
        }
\`\`\`

---

## Corporate Actions

\`\`\`python
from datetime import date

class CorporateActionAdjuster:
    """Adjust historical prices for corporate actions"""
    
    def __init__(self):
        # Corporate action database
        self.splits = {}  # symbol -> [(date, ratio)]
        self.dividends = {}  # symbol -> [(date, amount)]
    
    def add_split(self, symbol: str, split_date: date, ratio: float):
        """Record stock split (e.g., 2:1 split, ratio=2.0)"""
        if symbol not in self.splits:
            self.splits[symbol] = []
        self.splits[symbol].append((split_date, ratio))
        self.splits[symbol].sort()  # Sort by date
    
    def adjust_price(self, symbol: str, price: Decimal, 
                    price_date: date, ref_date: date = None) -> Decimal:
        """Adjust price for splits between price_date and ref_date"""
        ref_date = ref_date or date.today()
        
        if symbol not in self.splits:
            return price
        
        # Apply all splits between price_date and ref_date
        adjusted_price = price
        for split_date, ratio in self.splits[symbol]:
            if price_date < split_date <= ref_date:
                adjusted_price = adjusted_price / Decimal(str(ratio))
        
        return adjusted_price

# Example: AAPL 4:1 split on Aug 31, 2020
adjuster = CorporateActionAdjuster()
adjuster.add_split('AAPL', date(2020, 8, 31), 4.0)

# Historical price before split
old_price = Decimal('400.00')  # Aug 1, 2020
adjusted = adjuster.adjust_price('AAPL', old_price, date(2020, 8, 1))
print(f"Adjusted price: ${adjusted}")  # $100.00 (400 / 4)
\`\`\`

---

## Production Normalization Pipeline

\`\`\`python
import asyncio

class NormalizationPipeline:
    """Production data normalization pipeline"""
    
    def __init__(self):
        self.normalizer = DataNormalizer()
        self.nbbo_calc = NBBOCalculator()
        self.adjuster = CorporateActionAdjuster()
        
        # Statistics
        self.quotes_processed = 0
        self.exchanges_seen = set()
    
    async def process_raw_quote(self, raw: dict, exchange: str):
        """Normalize and process quote"""
        # Step 1: Normalize to unified format
        norm_quote = self.normalizer.normalize(raw, exchange)
        
        # Step 2: Update NBBO
        nbbo = self.nbbo_calc.update_quote(norm_quote)
        
        # Step 3: Publish to strategies
        if nbbo:
            await self.publish_nbbo(nbbo)
        
        self.quotes_processed += 1
        self.exchanges_seen.add(exchange)
        
        return norm_quote, nbbo
    
    async def publish_nbbo(self, nbbo: dict):
        """Publish NBBO to downstream systems"""
        # Publish to Kafka, Redis, etc.
        pass
\`\`\`

---

## Best Practices

1. **Use Decimal for prices** - Avoid float rounding errors
2. **Validate after normalization** - Check for negative prices, crossed markets
3. **Track corporate actions** - Essential for accurate historical analysis
4. **Symbol mapping** - Maintain symbol→venue mapping table
5. **Timezone consistency** - Convert all timestamps to UTC
6. **Version normalization** - Track normalization logic version for reproducibility

Now you can normalize market data from any exchange into a unified format!
`,
};

