export const regulatoryLandscapeEngineers = {
  title: 'Regulatory Landscape for Engineers',
  id: 'regulatory-landscape-engineers',
  content: `
# Regulatory Landscape for Engineers

## Introduction

**Financial regulations** aren't just legal requirements—they dictate system design. As an engineer building financial systems, you need to understand regulations because they determine:

- What data you must collect and retain (7+ years for SEC)
- How you handle customer funds (segregated accounts)
- What disclosures you must show users
- What activities are prohibited (insider trading, manipulation)
- How you report to regulators (daily, monthly, ad-hoc)

This section covers regulations you'll encounter building financial systems.

---

## SEC (Securities and Exchange Commission)

### What SEC Regulates

**SEC** protects investors in securities markets (stocks, bonds, options).

**Key responsibilities**:
1. **Require disclosure**: Public companies file 10-K, 10-Q, 8-K
2. **Regulate exchanges**: Oversee NYSE, NASDAQ
3. **Oversee brokers**: Approve broker-dealers
4. **Enforce rules**: Investigate fraud, manipulation
5. **Protect investors**: Accredited investor requirements, suitability

### Key SEC Rules for Engineers

#### **Regulation SHO** (Short Selling)

**What**: Rules for short selling and naked shorting.

**Engineering impact**:
- Must **locate shares** before shorting (prove shares available to borrow)
- Track **fail-to-deliver** (FTD) situations
- Build **close-out** mechanisms (buy shares within T+5 if FTD)

\`\`\`python
"""
Reg SHO Compliance: Locate Shares Before Short
"""
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class ShareLocate:
    """Track share locate for short sales"""
    ticker: str
    quantity: int
    locate_source: str  # 'broker inventory', 'third-party', 'easy-to-borrow'
    locate_time: datetime
    expiration: datetime
    
    def is_valid (self) -> bool:
        """Check if locate is still valid"""
        return datetime.now() < self.expiration


class ShortSaleManager:
    """Manage short sales with Reg SHO compliance"""
    
    def __init__(self):
        self.locates = {}  # {ticker: [ShareLocate]}
        self.short_positions = {}  # {ticker: quantity}
        self.failed_to_deliver = {}  # {ticker: {date: quantity}}
    
    def request_locate (self, ticker: str, quantity: int) -> ShareLocate:
        """
        Request share locate before allowing short sale
        
        Reg SHO: Must have reasonable grounds to believe shares
        can be borrowed before executing short sale
        """
        # Check if shares available
        available_shares = self._check_availability (ticker)
        
        if available_shares < quantity:
            raise ValueError (f"Insufficient shares available for {ticker}. "
                           f"Requested: {quantity}, Available: {available_shares}")
        
        # Create locate (valid for 1 day)
        locate = ShareLocate(
            ticker=ticker,
            quantity=quantity,
            locate_source='broker inventory',
            locate_time=datetime.now(),
            expiration=datetime.now() + timedelta (days=1)
        )
        
        if ticker not in self.locates:
            self.locates[ticker] = []
        self.locates[ticker].append (locate)
        
        print(f"✓ Locate approved: {quantity} shares of {ticker}")
        print(f"  Valid until: {locate.expiration}")
        
        return locate
    
    def execute_short_sale (self, ticker: str, quantity: int, locate: ShareLocate):
        """Execute short sale (must have valid locate)"""
        if not locate.is_valid():
            raise ValueError("Locate expired. Must request new locate.")
        
        if locate.quantity < quantity:
            raise ValueError (f"Locate insufficient. Have: {locate.quantity}, Need: {quantity}")
        
        # Execute short
        if ticker not in self.short_positions:
            self.short_positions[ticker] = 0
        self.short_positions[ticker] += quantity
        
        print(f"✓ Short sale executed: {quantity} shares of {ticker}")
        print(f"  Total short position: {self.short_positions[ticker]}")
    
    def track_fail_to_deliver (self, ticker: str, quantity: int, trade_date: datetime):
        """
        Track FTDs (when shares not delivered by T+2)
        
        Reg SHO: Must close out FTDs within T+5 for most stocks,
        T+35 for threshold securities
        """
        if ticker not in self.failed_to_deliver:
            self.failed_to_deliver[ticker] = {}
        
        self.failed_to_deliver[ticker][trade_date] = quantity
        
        # Calculate close-out deadline
        close_out_deadline = trade_date + timedelta (days=5)  # T+5
        
        print(f"⚠ Fail to deliver recorded:")
        print(f"  Ticker: {ticker}")
        print(f"  Quantity: {quantity}")
        print(f"  Trade Date: {trade_date.date()}")
        print(f"  Close-out deadline: {close_out_deadline.date()}")
        
        return close_out_deadline
    
    def _check_availability (self, ticker: str) -> int:
        """Check how many shares available to borrow"""
        # In reality, query prime broker or stock loan department
        # For simulation, return large number for "easy-to-borrow" stocks
        easy_to_borrow = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        if ticker in easy_to_borrow:
            return 1_000_000  # 1M shares available
        else:
            return 10_000  # Limited availability


# Example: Short selling with Reg SHO compliance
manager = ShortSaleManager()

# Request locate
locate = manager.request_locate('AAPL', 1000)

# Execute short sale
manager.execute_short_sale('AAPL', 1000, locate)

# Simulate fail-to-deliver
trade_date = datetime(2024, 1, 15)
close_out = manager.track_fail_to_deliver('AAPL', 100, trade_date)
\`\`\`

#### **Regulation Best Interest (Reg BI)**

**What**: Broker-dealers must act in customer's best interest.

**Engineering impact**:
- **Suitability checks**: System must verify recommendations suitable for customer
- **Disclosure**: Must disclose conflicts of interest (like PFOF)
- **Documentation**: Must log all recommendations and rationale

\`\`\`python
"""
Reg BI: Suitability and Best Interest
"""

class CustomerProfile:
    """Customer profile for suitability analysis"""
    
    def __init__(self, age: int, income: int, net_worth: int, 
                 investment_experience: str, risk_tolerance: str,
                 investment_objectives: list):
        self.age = age
        self.income = income
        self.net_worth = net_worth
        self.investment_experience = investment_experience  # 'none', 'limited', 'good', 'extensive'
        self.risk_tolerance = risk_tolerance  # 'conservative', 'moderate', 'aggressive'
        self.investment_objectives = investment_objectives  # ['growth', 'income', 'preservation']


class SuitabilityChecker:
    """Check if investment is suitable for customer (Reg BI compliance)"""
    
    def check_suitability (self, customer: CustomerProfile, product: dict) -> dict:
        """
        Determine if product is suitable for customer
        
        Reg BI requires considering:
        1. Customer\'s investment profile
        2. Product characteristics (risk, complexity, costs)
        3. Reasonable basis suitability (product appropriate for some investors)
        4. Customer-specific suitability (appropriate for THIS investor)
        5. Quantitative suitability (not excessive trading)
        """
        warnings = []
        
        # Check risk tolerance
        product_risk = product['risk_level']  # 'low', 'medium', 'high', 'very_high'
        
        if product_risk == 'very_high' and customer.risk_tolerance != 'aggressive':
            warnings.append (f"Product risk ({product_risk}) exceeds customer tolerance ({customer.risk_tolerance})")
        
        # Check complexity vs experience
        product_complexity = product['complexity']  # 'simple', 'moderate', 'complex'
        
        if product_complexity == 'complex' and customer.investment_experience in ['none', 'limited']:
            warnings.append (f"Complex product not suitable for customer with {customer.investment_experience} experience")
        
        # Check investment objectives
        if 'preservation' in customer.investment_objectives and product_risk in ['high', 'very_high']:
            warnings.append("High-risk product conflicts with capital preservation objective")
        
        # Check age appropriateness
        if customer.age > 65 and product.get('illiquid', False):
            warnings.append("Illiquid product may not be appropriate for retiree")
        
        # Check concentration
        position_size = product.get('investment_amount', 0)
        concentration = position_size / customer.net_worth if customer.net_worth > 0 else 0
        
        if concentration > 0.10:  # >10% of net worth
            warnings.append (f"Position size ({concentration:.1%} of net worth) may be excessive")
        
        # Determine suitability
        is_suitable = len (warnings) == 0
        
        return {
            'suitable': is_suitable,
            'warnings': warnings,
            'recommendation': 'approved' if is_suitable else 'not_recommended'
        }
    
    def log_recommendation (self, customer_id: str, product: dict, 
                          suitability: dict, rationale: str):
        """
        Log recommendation for regulatory record-keeping
        
        Reg BI: Must maintain records for 6 years
        """
        record = {
            'timestamp': datetime.now(),
            'customer_id': customer_id,
            'product': product['name'],
            'suitable': suitability['suitable'],
            'warnings': suitability['warnings'],
            'rationale': rationale,
            'approved_by': 'system'  # Or human approval if required
        }
        
        # In production: Store in database with 6+ year retention
        print(f"\\n=== Suitability Record ===")
        print(f"Customer: {customer_id}")
        print(f"Product: {product['name']}")
        print(f"Suitable: {suitability['suitable']}")
        if suitability['warnings']:
            print(f"Warnings:")
            for warning in suitability['warnings']:
                print(f"  - {warning}")
        print(f"Rationale: {rationale}")
        
        return record


# Example: Check suitability
customer = CustomerProfile(
    age=35,
    income=150_000,
    net_worth=500_000,
    investment_experience='good',
    risk_tolerance='moderate',
    investment_objectives=['growth', 'income']
)

product = {
    'name': '3x Leveraged Tech ETF',
    'risk_level': 'very_high',
    'complexity': 'complex',
    'investment_amount': 10_000
}

checker = SuitabilityChecker()
suitability = checker.check_suitability (customer, product)

if not suitability['suitable']:
    print("\\n⚠ UNSUITABLE INVESTMENT")
    for warning in suitability['warnings']:
        print(f"  {warning}")
else:
    print("\\n✓ Suitable investment")

# Log for compliance
checker.log_recommendation('CUST123', product, suitability, 
                          "Customer seeking growth with moderate risk tolerance")
\`\`\`

---

## FINRA (Financial Industry Regulatory Authority)

### What FINRA Regulates

**FINRA** is a self-regulatory organization overseeing broker-dealers.

**Key rules**:
- Licensing (Series 7, 63, etc.)
- Best execution requirements
- Trading rules (pattern day trader)
- Reporting requirements

### Pattern Day Trader Rule

**Rule 4210**: If you make 4+ day trades in 5 days AND day trades >6% of total trades, you're a pattern day trader.

**Requirement**: Must maintain $25,000 minimum equity.

\`\`\`python
"""
Pattern Day Trader Detection
"""

class PatternDayTraderChecker:
    """Monitor for PDT rule violations"""
    
    def __init__(self, account_value: float):
        self.account_value = account_value
        self.trades = []  # List of trades
        self.day_trades = []  # Subset that are day trades
    
    def add_trade (self, ticker: str, buy_or_sell: str, quantity: int, 
                  trade_date: datetime):
        """Add trade and check for day trade"""
        trade = {
            'ticker': ticker,
            'side': buy_or_sell,
            'quantity': quantity,
            'date': trade_date,
            'is_day_trade': False
        }
        
        # Check if this creates a day trade
        # Day trade = buy and sell same stock same day
        if buy_or_sell == 'sell':
            # Look for buy same day
            for t in reversed (self.trades):
                if (t['ticker'] == ticker and 
                    t['side'] == 'buy' and 
                    t['date'].date() == trade_date.date()):
                    trade['is_day_trade'] = True
                    t['is_day_trade'] = True
                    self.day_trades.append (trade)
                    break
        
        self.trades.append (trade)
        
        return trade
    
    def check_pdt_status (self, lookback_days: int = 5) -> dict:
        """
        Check if account is pattern day trader
        
        PDT if:
        1. Made 4+ day trades in 5 business days
        2. Day trades > 6% of total trades
        """
        cutoff = datetime.now() - timedelta (days=lookback_days)
        
        recent_trades = [t for t in self.trades if t['date'] > cutoff]
        recent_day_trades = [t for t in self.day_trades if t['date'] > cutoff]
        
        day_trade_count = len (recent_day_trades)
        total_trades = len (recent_trades)
        day_trade_pct = day_trade_count / total_trades if total_trades > 0 else 0
        
        is_pattern_day_trader = (day_trade_count >= 4 and day_trade_pct > 0.06)
        
        # Check if account meets minimum
        meets_minimum = self.account_value >= 25_000
        
        return {
            'is_pdt': is_pattern_day_trader,
            'day_trades_in_period': day_trade_count,
            'total_trades': total_trades,
            'day_trade_percentage': day_trade_pct,
            'account_value': self.account_value,
            'meets_minimum': meets_minimum,
            'can_day_trade': meets_minimum or not is_pattern_day_trader,
            'warning': None if meets_minimum or not is_pattern_day_trader 
                      else "Pattern Day Trader with <$25K account - day trading restricted"
        }


# Example: Track day trades
checker = PatternDayTraderChecker (account_value=20_000)

# Simulate trades
today = datetime.now()

# Day trade 1
checker.add_trade('AAPL', 'buy', 100, today)
checker.add_trade('AAPL', 'sell', 100, today)

# Day trade 2
checker.add_trade('TSLA', 'buy', 50, today)
checker.add_trade('TSLA', 'sell', 50, today)

# Not a day trade (different days)
checker.add_trade('GOOGL', 'buy', 25, today)
checker.add_trade('GOOGL', 'sell', 25, today + timedelta (days=1))

# Day trades 3 & 4
checker.add_trade('MSFT', 'buy', 30, today + timedelta (days=2))
checker.add_trade('MSFT', 'sell', 30, today + timedelta (days=2))
checker.add_trade('AMZN', 'buy', 10, today + timedelta (days=3))
checker.add_trade('AMZN', 'sell', 10, today + timedelta (days=3))

# Check PDT status
status = checker.check_pdt_status()

print("\\n=== Pattern Day Trader Check ===")
print(f"Day trades (5 days): {status['day_trades_in_period']}")
print(f"Total trades: {status['total_trades']}")
print(f"Day trade %: {status['day_trade_percentage']:.1%}")
print(f"Is PDT: {status['is_pdt']}")
print(f"Account value: \\$\{status['account_value']:,.0f}")
print(f"Meets $25K min: {status['meets_minimum']}")

if status['warning']:
    print(f"\\n⚠ WARNING: {status['warning']}")
\`\`\`

---

## Know Your Customer (KYC) & Anti-Money Laundering (AML)

### KYC Requirements

**Every financial institution** must verify customer identity.

**Required information**:
- Full legal name
- Date of birth
- Address
- Social Security Number (or Tax ID)
- Government-issued ID photo

\`\`\`python
"""
KYC Verification System
"""

class KYCVerification:
    """Verify customer identity"""
    
    def __init__(self):
        self.verified_customers = {}
    
    def collect_information (self, customer_data: dict) -> dict:
        """
        Collect required KYC information
        
        Required by USA PATRIOT Act and Bank Secrecy Act
        """
        required_fields = [
            'full_name',
            'date_of_birth',
            'ssn',
            'address',
            'id_type',  # 'drivers_license', 'passport', etc.
            'id_number',
            'id_photo'
        ]
        
        missing = [field for field in required_fields if field not in customer_data]
        
        if missing:
            return {
                'status': 'incomplete',
                'missing_fields': missing
            }
        
        return {
            'status': 'collected',
            'data': customer_data
        }
    
    def verify_identity (self, customer_data: dict) -> dict:
        """
        Verify customer identity against databases
        
        Real implementation would use:
        - Credit bureaus (Experian, TransUnion)
        - Identity verification services (Jumio, Onfido)
        - Government databases (OFAC, sanctions lists)
        """
        # Simulate verification checks
        checks = {
            'ssn_valid': self._verify_ssn (customer_data['ssn']),
            'address_valid': self._verify_address (customer_data['address']),
            'id_authentic': self._verify_id (customer_data['id_photo']),
            'not_on_sanctions_list': self._check_sanctions (customer_data['full_name']),
            'age_requirement': self._verify_age (customer_data['date_of_birth'])
        }
        
        all_passed = all (checks.values())
        
        if all_passed:
            self.verified_customers[customer_data['ssn']] = {
                'verified_date': datetime.now(),
                'customer_data': customer_data
            }
        
        return {
            'verified': all_passed,
            'checks': checks,
            'result': 'approved' if all_passed else 'denied'
        }
    
    def _verify_ssn (self, ssn: str) -> bool:
        """Verify SSN format and validity"""
        # Real: Check with Social Security Administration
        return len (ssn) == 9 and ssn.isdigit()
    
    def _verify_address (self, address: str) -> bool:
        """Verify address is real"""
        # Real: Use USPS address validation API
        return len (address) > 10
    
    def _verify_id (self, id_photo: str) -> bool:
        """Verify ID is authentic"""
        # Real: Use ML to detect fake IDs (Jumio, Onfido)
        return True
    
    def _check_sanctions (self, name: str) -> bool:
        """Check against OFAC sanctions list"""
        # Real: Query OFAC SDN (Specially Designated Nationals) list
        sanctioned_names = ['John Doe Terrorist']  # Example
        return name not in sanctioned_names
    
    def _verify_age (self, dob: str) -> bool:
        """Verify customer is 18+"""
        from dateutil.parser import parse
        birth_date = parse (dob)
        age = (datetime.now() - birth_date).days / 365.25
        return age >= 18


# Example: KYC verification
kyc = KYCVerification()

customer_data = {
    'full_name': 'Alice Johnson',
    'date_of_birth': '1990-05-15',
    'ssn': '123456789',
    'address': '123 Main St, New York, NY 10001',
    'id_type': 'drivers_license',
    'id_number': 'NY123456',
    'id_photo': 'base64_encoded_photo...'
}

# Collect information
collection = kyc.collect_information (customer_data)
print(f"Collection status: {collection['status']}")

if collection['status'] == 'collected':
    # Verify identity
    verification = kyc.verify_identity (customer_data)
    
    print(f"\\n=== KYC Verification ===")
    print(f"Result: {verification['result']}")
    print(f"\\nChecks:")
    for check, passed in verification['checks'].items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
\`\`\`

### AML (Anti-Money Laundering)

**Monitor for suspicious activity**:
- Large transactions (>$10,000 → CTR filing)
- Structuring (breaking up transactions to avoid reporting)
- Unusual patterns (sudden large deposits, rapid transfers)

---

## Key Takeaways for Engineers

1. **Design for compliance from day one**: Retrofitting is expensive
2. **Maintain audit trails**: Log everything (7-year retention for SEC)
3. **Build safeguards**: Suitability checks, trading restrictions, AML monitoring
4. **Stay updated**: Regulations change frequently
5. **Work with compliance team**: Engineers implement, compliance interprets

**Next section**: Finance Terminology for Developers - glossary of terms you'll encounter.
`,
};
