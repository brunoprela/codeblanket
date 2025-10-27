export const dataFeedProtocols = {
    title: 'Data Feed Protocols (FIX, FAST, ITCH)',
    id: 'data-feed-protocols',
    content: `
# Data Feed Protocols (FIX, FAST, ITCH)

## Introduction

Financial markets operate on specialized protocols optimized for speed, reliability, and standardization. While consumer applications use HTTP and WebSocket, professional trading systems use binary protocols like FIX, FAST, and ITCH that can transmit millions of messages per second with microsecond latency. Understanding these protocols is essential for building production trading infrastructure and consuming direct exchange feeds.

**Why Financial Protocols Matter:**
- **Speed**: Binary encoding is 5-10× faster than JSON
- **Bandwidth**: Compressed messages use 70-90% less bandwidth
- **Standardization**: FIX is the lingua franca of financial messaging
- **Direct Access**: Exchange feeds use these protocols (no vendor markup)
- **Professional Requirement**: Institutional systems must speak FIX

**Protocol Usage in Production:**
- **FIX Protocol**: 90%+ of institutional order routing and execution
- **ITCH**: NASDAQ's proprietary feed (400M+ messages per day)
- **FAST**: CME, ICE futures exchanges (ultra-low latency)
- **Binary Protocols**: All major exchanges have proprietary binary formats

By the end of this section, you'll understand:
- FIX protocol structure and message types
- FAST encoding for streaming data
- NASDAQ ITCH protocol mechanics
- How to parse binary financial data
- When to use each protocol
- Python implementations for all three

---

## FIX Protocol (Financial Information eXchange)

### Overview

FIX is the **de facto standard** for electronic trading communication. Created in 1992, it defines message formats for order management, execution, trade confirmation, and market data.

**FIX Characteristics:**
- **Text-based**: Human-readable (but can be binary with FAST)
- **Tag-value pairs**: \`Tag=Value|\` format
- **Session-level**: Logon, heartbeat, sequence numbers
- **Application-level**: New Order, Execution Report, etc.
- **Versioning**: FIX 4.0, 4.2, 4.4, 5.0 SP2

**Example FIX Message:**
\`\`\`
8=FIX.4.4|9=178|35=D|49=SENDER|56=TARGET|34=1|52=20240115-10:30:00|
11=ORDER123|21=1|55=AAPL|54=1|60=20240115-10:30:00|38=100|40=2|44=150.25|10=123|
\`\`\`

(Note: | represents SOH character - Start of Header, ASCII 0x01)

**Tag Meanings:**
- `8`: Begin String (FIX.4.4)
- `9`: Body Length (178 bytes)
- `35`: Message Type (D = New Order Single)
- `49`: Sender CompID
- `56`: Target CompID
- `34`: Message Sequence Number
- `52`: Sending Time
- `11`: Client Order ID (ORDER123)
- `55`: Symbol (AAPL)
- `54`: Side (1 = Buy, 2 = Sell)
- `38`: Order Quantity (100 shares)
- `40`: Order Type (2 = Limit)
- `44`: Price (150.25)
- `10`: Checksum

### FIX Message Structure

**Standard Header:**
\`\`\`
8=FIX.4.4        | BeginString (protocol version)
9=<length>       | BodyLength (bytes from tag 35 to before tag 10)
35=<msgtype>     | MsgType (D=NewOrder, 8=ExecutionReport, etc.)
49=<senderid>    | SenderCompID
56=<targetid>    | TargetCompID
34=<seqnum>      | MsgSeqNum (for gap detection)
52=<timestamp>   | SendingTime (YYYYMMDD-HH:MM:SS)
\`\`\`

**Message Body:**
- Application-specific tags (order details, execution info, quotes, etc.)

**Standard Trailer:**
\`\`\`
10=<checksum>    | Checksum (modulo 256 sum of bytes)
\`\`\`

### Common FIX Message Types

| MsgType | Name | Direction | Purpose |
|---------|------|-----------|---------|
| A | Logon | Both | Establish session |
| 0 | Heartbeat | Both | Keep connection alive |
| D | New Order Single | Client → Broker | Submit new order |
| 8 | Execution Report | Broker → Client | Order status update |
| F | Order Cancel Request | Client → Broker | Cancel order |
| 9 | Order Cancel Reject | Broker → Client | Cancel denied |
| V | Market Data Request | Client → Vendor | Subscribe to quotes |
| W | Market Data Snapshot | Vendor → Client | Quote/trade data |
| X | Market Data Incremental | Vendor → Client | Updates |

### Python FIX Implementation

**Using simplefix Library:**

\`\`\`python
import simplefix
from datetime import datetime

# Create FIX message
msg = simplefix.FixMessage()
msg.append_pair(8, "FIX.4.4")  # BeginString
msg.append_pair(35, "D")  # MsgType = New Order Single
msg.append_pair(49, "CLIENT")  # SenderCompID
msg.append_pair(56, "BROKER")  # TargetCompID
msg.append_pair(34, 1)  # MsgSeqNum
msg.append_pair(52, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S"))  # SendingTime

# Order details
msg.append_pair(11, "ORDER123")  # ClOrdID
msg.append_pair(21, 1)  # HandlInst (1 = Automated)
msg.append_pair(55, "AAPL")  # Symbol
msg.append_pair(54, 1)  # Side (1 = Buy)
msg.append_pair(60, datetime.utcnow().strftime("%Y%m%d-%H:%M:%S"))  # TransactTime
msg.append_pair(38, 100)  # OrderQty
msg.append_pair(40, 2)  # OrdType (2 = Limit)
msg.append_pair(44, 150.25)  # Price

# Encode to bytes
encoded = msg.encode()
print(encoded)
# b'8=FIX.4.4\\x019=...\\x0135=D\\x01...10=123\\x01'
\`\`\`

**Parsing FIX Message:**

\`\`\`python
def parse_fix_message(raw_message: bytes) -> dict:
    """Parse FIX message into dictionary"""
    # Split by SOH character (\\x01)
    pairs = raw_message.decode('ascii').split('\\x01')
    
    message = {}
    for pair in pairs:
        if '=' in pair:
            tag, value = pair.split('=', 1)
            message[int(tag)] = value
    
    return message

# Usage
raw = b'8=FIX.4.4\\x019=100\\x0135=D\\x0155=AAPL\\x0154=1\\x0138=100\\x0110=123\\x01'
parsed = parse_fix_message(raw)
print(parsed)
# {8: 'FIX.4.4', 9: '100', 35: 'D', 55: 'AAPL', 54: '1', 38: '100', 10: '123'}

# Access fields
symbol = parsed[55]  # 'AAPL'
side = 'BUY' if parsed[54] == '1' else 'SELL'
quantity = int(parsed[38])  # 100
\`\`\`

**Production FIX Engine with quickfix:**

\`\`\`python
import quickfix as fix
import quickfix44 as fix44

class Application(fix.Application):
    def onCreate(self, sessionID):
        print(f"Session created: {sessionID}")
    
    def onLogon(self, sessionID):
        print(f"Logged on: {sessionID}")
    
    def onLogout(self, sessionID):
        print(f"Logged out: {sessionID}")
    
    def toAdmin(self, message, sessionID):
        """Outgoing admin message"""
        pass
    
    def fromAdmin(self, message, sessionID):
        """Incoming admin message"""
        pass
    
    def toApp(self, message, sessionID):
        """Outgoing application message"""
        print(f"Sending: {message}")
    
    def fromApp(self, message, sessionID):
        """Incoming application message"""
        print(f"Received: {message}")
        
        # Parse message type
        msgType = fix.MsgType()
        message.getHeader().getField(msgType)
        
        if msgType.getValue() == fix.MsgType_ExecutionReport:
            self.on_execution_report(message)
    
    def on_execution_report(self, message):
        """Handle execution report"""
        # Extract fields
        symbol = fix.Symbol()
        order_id = fix.ClOrdID()
        exec_type = fix.ExecType()
        order_status = fix.OrdStatus()
        
        message.getField(symbol)
        message.getField(order_id)
        message.getField(exec_type)
        message.getField(order_status)
        
        print(f"Execution: {order_id.getValue()} {symbol.getValue()} {order_status.getValue()}")

# Initialize FIX engine
settings = fix.SessionSettings("config.cfg")
storeFactory = fix.FileStoreFactory(settings)
logFactory = fix.FileLogFactory(settings)
application = Application()

initiator = fix.SocketInitiator(application, storeFactory, settings, logFactory)
initiator.start()

# Send new order
message = fix44.NewOrderSingle()
message.setField(fix.ClOrdID("ORDER123"))
message.setField(fix.Symbol("AAPL"))
message.setField(fix.Side(fix.Side_BUY))
message.setField(fix.OrderQty(100))
message.setField(fix.OrdType(fix.OrdType_LIMIT))
message.setField(fix.Price(150.25))

fix.Session.sendToTarget(message, sessionID)
\`\`\`

---

## FAST Protocol (FIX Adapted for Streaming)

### Overview

FAST is a **binary compression technique** for FIX messages, designed for market data feeds where bandwidth and latency are critical. It can achieve 70-90% compression by omitting unchanged fields and using delta encoding.

**FAST Characteristics:**
- **Binary encoding**: More compact than text FIX
- **Template-based**: Pre-defined message structures
- **Delta encoding**: Send only changed fields
- **Presence map**: Bitmap indicating which fields are present
- **Used by**: CME Group, ICE, many futures exchanges

**Compression Example:**
\`\`\`
Text FIX (100 bytes):
55=AAPL|268=1|269=0|270=150.25|271=500|...

FAST (20 bytes):
[template_id][presence_map][delta_price][delta_size]...
\`\`\`

### FAST Message Structure

**Template Definition:**
\`\`\`xml
<template name="MDIncrementalRefresh" id="81">
  <uInt32 name="MsgSeqNum" id="34"/>
  <uInt64 name="SendingTime" id="52"/>
  <sequence name="MDEntries">
    <string name="Symbol" id="55"/>
    <uInt32 name="MDUpdateAction" id="279"/>
    <decimal name="MDEntryPx" id="270"/>
    <uInt32 name="MDEntrySize" id="271"/>
  </sequence>
</template>
\`\`\`

**Binary Message:**
\`\`\`
[Template ID: 81][Presence Map: 11111000]
[MsgSeqNum][SendingTime]
[Symbol][MDUpdateAction][MDEntryPx][MDEntrySize]
\`\`\`

**Presence Map:**
- Each bit indicates if field is present (1) or absent (0)
- Allows omitting null/unchanged fields
- Saves significant bandwidth

### FAST Encoding Types

**Copy Operator:**
- Copy previous value if field absent
- Used for slowly-changing fields (symbol, exchange)

**Delta Operator:**
- Send difference from previous value
- Ideal for prices: \`delta = new_price - prev_price\`

**Increment Operator:**
- Increment previous value by 1
- Perfect for sequence numbers

**Default Operator:**
- Use default value if field absent
- Common for flags and constants

### Python FAST Implementation

**Using pyfast Library:**

\`\`\`python
from decimal import Decimal
import struct

class FASTDecoder:
    def __init__(self, templates: dict):
        self.templates = templates
        self.previous_values = {}  # For delta/copy operators
    
    def decode_message(self, data: bytes) -> dict:
        """Decode FAST binary message"""
        offset = 0
        
        # Read template ID (stop bit encoding)
        template_id, offset = self._read_uint(data, offset)
        template = self.templates[template_id]
        
        # Read presence map
        presence_map, offset = self._read_pmap(data, offset)
        
        # Decode fields according to template
        message = {'template_id': template_id}
        pmap_index = 0
        
        for field in template['fields']:
            if presence_map[pmap_index]:
                # Field is present, read it
                value, offset = self._read_field(data, offset, field)
                message[field['name']] = value
            else:
                # Field is absent, use operator default
                if field.get('operator') == 'copy':
                    message[field['name']] = self.previous_values.get(field['name'])
                elif field.get('operator') == 'delta':
                    # Read delta and apply
                    delta, offset = self._read_field(data, offset, field)
                    prev = self.previous_values.get(field['name'], 0)
                    message[field['name']] = prev + delta
            
            pmap_index += 1
        
        # Store values for next message
        self.previous_values.update(message)
        
        return message
    
    def _read_uint(self, data: bytes, offset: int) -> tuple[int, int]:
        """Read unsigned integer with stop bit encoding"""
        value = 0
        while True:
            byte = data[offset]
            value = (value << 7) | (byte & 0x7F)
            offset += 1
            if byte & 0x80:  # Stop bit set
                break
        return value, offset
    
    def _read_pmap(self, data: bytes, offset: int) -> tuple[list[bool], int]:
        """Read presence map"""
        pmap = []
        while True:
            byte = data[offset]
            for i in range(6, -1, -1):  # 7 bits
                pmap.append(bool(byte & (1 << i)))
            offset += 1
            if byte & 0x80:  # Stop bit
                break
        return pmap, offset
    
    def _read_field(self, data: bytes, offset: int, field: dict) -> tuple:
        """Read field value based on type"""
        field_type = field['type']
        
        if field_type == 'uint32':
            return self._read_uint(data, offset)
        elif field_type == 'int32':
            value, offset = self._read_uint(data, offset)
            # Convert to signed
            if value & 1:
                value = -((value + 1) >> 1)
            else:
                value = value >> 1
            return value, offset
        elif field_type == 'decimal':
            # Decimal: [exponent][mantissa]
            exponent, offset = self._read_uint(data, offset)
            mantissa, offset = self._read_uint(data, offset)
            return Decimal(mantissa) / (10 ** exponent), offset
        elif field_type == 'string':
            # String: length-prefixed
            length, offset = self._read_uint(data, offset)
            value = data[offset:offset+length].decode('ascii')
            return value, offset + length
        
        return None, offset

# Usage
templates = {
    81: {
        'name': 'MDIncrementalRefresh',
        'fields': [
            {'name': 'MsgSeqNum', 'type': 'uint32'},
            {'name': 'SendingTime', 'type': 'uint64'},
            {'name': 'Symbol', 'type': 'string'},
            {'name': 'MDEntryPx', 'type': 'decimal', 'operator': 'delta'},
            {'name': 'MDEntrySize', 'type': 'uint32', 'operator': 'delta'}
        ]
    }
}

decoder = FASTDecoder(templates)

# Decode FAST message
binary_data = b'\\x81\\xC0\\x80\\x01\\x84AAPL\\x81\\x05\\x82\\x64'  # Example
message = decoder.decode_message(binary_data)
print(message)
# {'template_id': 81, 'MsgSeqNum': 1, 'Symbol': 'AAPL', 'MDEntryPx': Decimal('150.25'), ...}
\`\`\`

---

## NASDAQ ITCH Protocol

### Overview

ITCH is NASDAQ's proprietary market data protocol delivering **every order and trade** in real-time. It's a raw, unfiltered feed showing the complete order book.

**ITCH Characteristics:**
- **Binary format**: Fixed-length messages
- **UDP multicast**: One-to-many broadcast
- **Sequence numbers**: For gap detection
- **Message types**: Add Order, Delete Order, Trade, etc.
- **Volume**: 400+ million messages per day

**ITCH Message Types:**
- `A`: Add Order
- `F`: Add Order with MPID
- `E`: Order Executed
- `C`: Order Executed with Price
- `X`: Order Cancel
- `D`: Order Delete
- `U`: Order Replace
- `P`: Trade (non-cross)
- `Q`: Cross Trade

### ITCH Message Structure

**Add Order Message (Type A):**
\`\`\`
Message Type:     1 byte  ('A')
Stock Locate:     2 bytes (uint16)
Tracking Number:  2 bytes (uint16)
Timestamp:        6 bytes (uint48, nanoseconds since midnight)
Order Ref:        8 bytes (uint64)
Buy/Sell:         1 byte  ('B' or 'S')
Shares:           4 bytes (uint32)
Stock:            8 bytes (ASCII, right-padded)
Price:            4 bytes (uint32, in 1/10000ths)

Total: 36 bytes
\`\`\`

**Example Binary:**
\`\`\`
A           // Message type
\\x00\\x01     // Stock locate = 1
\\x00\\x00     // Tracking number = 0
\\x00\\x00\\x00\\x00\\x0E\\x10  // Timestamp = 3600 nanoseconds
\\x00\\x00\\x00\\x00\\x00\\x00\\x12\\x34  // Order ref = 4660
B           // Buy order
\\x00\\x00\\x01\\xF4  // 500 shares
AAPL    // Stock (padded to 8 chars)
\\x00\\x00\\x3A\\x99  // Price = 15009 (= $150.09)
\`\`\`

### Python ITCH Parser

\`\`\`python
import struct
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ITCHMessage:
    message_type: str
    stock_locate: int
    tracking_number: int
    timestamp: int  # Nanoseconds since midnight
    data: Dict[str, Any]

class ITCHParser:
    def __init__(self):
        # Message type -> (format, parser function)
        self.parsers = {
            'A': ('>HHQ6sQcI8sI', self.parse_add_order),
            'E': ('>HHQ6sQIQ', self.parse_order_executed),
            'X': ('>HHQ6sQI', self.parse_order_cancel),
            'P': ('>HHQ6sQ8scI8sI', self.parse_trade),
            # Add more message types...
        }
    
    def parse(self, data: bytes) -> ITCHMessage:
        """Parse ITCH binary message"""
        msg_type = chr(data[0])
        
        if msg_type not in self.parsers:
            raise ValueError(f"Unknown message type: {msg_type}")
        
        format_str, parser_func = self.parsers[msg_type]
        
        # Unpack binary data
        unpacked = struct.unpack(format_str, data)
        
        # Parse with specific parser
        return parser_func(msg_type, unpacked)
    
    def parse_add_order(self, msg_type: str, fields: tuple) -> ITCHMessage:
        """Parse Add Order message"""
        stock_locate = fields[0]
        tracking_number = fields[1]
        timestamp = int.from_bytes(fields[2], 'big')
        order_ref = fields[3]
        buy_sell = fields[4].decode('ascii')
        shares = fields[5]
        stock = fields[6].decode('ascii').strip()
        price = fields[7] / 10000  # Convert to decimal
        
        return ITCHMessage(
            message_type=msg_type,
            stock_locate=stock_locate,
            tracking_number=tracking_number,
            timestamp=timestamp,
            data={
                'order_ref': order_ref,
                'side': 'BUY' if buy_sell == 'B' else 'SELL',
                'shares': shares,
                'stock': stock,
                'price': price
            }
        )
    
    def parse_order_executed(self, msg_type: str, fields: tuple) -> ITCHMessage:
        """Parse Order Executed message"""
        stock_locate = fields[0]
        tracking_number = fields[1]
        timestamp = int.from_bytes(fields[2], 'big')
        order_ref = fields[3]
        executed_shares = fields[4]
        match_number = fields[5]
        
        return ITCHMessage(
            message_type=msg_type,
            stock_locate=stock_locate,
            tracking_number=tracking_number,
            timestamp=timestamp,
            data={
                'order_ref': order_ref,
                'executed_shares': executed_shares,
                'match_number': match_number
            }
        )
    
    def parse_order_cancel(self, msg_type: str, fields: tuple) -> ITCHMessage:
        """Parse Order Cancel message"""
        stock_locate = fields[0]
        tracking_number = fields[1]
        timestamp = int.from_bytes(fields[2], 'big')
        order_ref = fields[3]
        cancelled_shares = fields[4]
        
        return ITCHMessage(
            message_type=msg_type,
            stock_locate=stock_locate,
            tracking_number=tracking_number,
            timestamp=timestamp,
            data={
                'order_ref': order_ref,
                'cancelled_shares': cancelled_shares
            }
        )
    
    def parse_trade(self, msg_type: str, fields: tuple) -> ITCHMessage:
        """Parse Trade message"""
        stock_locate = fields[0]
        tracking_number = fields[1]
        timestamp = int.from_bytes(fields[2], 'big')
        order_ref = fields[3]
        stock = fields[4].decode('ascii').strip()
        buy_sell = fields[5].decode('ascii')
        shares = fields[6]
        stock2 = fields[7].decode('ascii').strip()  # Duplicate
        price = fields[8] / 10000
        
        return ITCHMessage(
            message_type=msg_type,
            stock_locate=stock_locate,
            tracking_number=tracking_number,
            timestamp=timestamp,
            data={
                'order_ref': order_ref,
                'stock': stock,
                'side': 'BUY' if buy_sell == 'B' else 'SELL',
                'shares': shares,
                'price': price
            }
        )

# Usage
parser = ITCHParser()

# Example: Add Order message
add_order_binary = b'A\\x00\\x01\\x00\\x00\\x00\\x00\\x00\\x00\\x0E\\x10\\x00\\x00\\x00\\x00\\x00\\x00\\x12\\x34B\\x00\\x00\\x01\\xF4AAPL    \\x00\\x00\\x3A\\x99'

message = parser.parse(add_order_binary)
print(f"Type: {message.message_type}")
print(f"Stock: {message.data['stock']}")
print(f"Side: {message.data['side']}")
print(f"Shares: {message.data['shares']}")
print(f"Price: ${message.data['price']:.2f}")
# Type: A
# Stock: AAPL
# Side: BUY
# Shares: 500
# Price: $150.09
\`\`\`

### Building Order Book from ITCH

\`\`\`python
from collections import defaultdict, OrderedDict
from decimal import Decimal

class OrderBook:
    def __init__(self):
        self.orders = {}  # order_ref -> order
        self.bids = defaultdict(int)  # price -> total_size
        self.asks = defaultdict(int)
        
    def add_order(self, order_ref: int, side: str, shares: int, price: Decimal):
        """Add order to book"""
        self.orders[order_ref] = {
            'side': side,
            'shares': shares,
            'price': price
        }
        
        if side == 'BUY':
            self.bids[price] += shares
        else:
            self.asks[price] += shares
    
    def cancel_order(self, order_ref: int, cancelled_shares: int):
        """Cancel part or all of order"""
        if order_ref not in self.orders:
            return
        
        order = self.orders[order_ref]
        order['shares'] -= cancelled_shares
        
        if order['side'] == 'BUY':
            self.bids[order['price']] -= cancelled_shares
        else:
            self.asks[order['price']] -= cancelled_shares
        
        # Remove if fully cancelled
        if order['shares'] <= 0:
            del self.orders[order_ref]
    
    def execute_order(self, order_ref: int, executed_shares: int):
        """Execute part or all of order"""
        self.cancel_order(order_ref, executed_shares)
    
    def get_best_bid_ask(self) -> tuple[Decimal, Decimal]:
        """Get BBO"""
        best_bid = max(self.bids.keys()) if self.bids else Decimal('0')
        best_ask = min(self.asks.keys()) if self.asks else Decimal('999999')
        return best_bid, best_ask
    
    def get_depth(self, levels: int = 5) -> dict:
        """Get order book depth"""
        sorted_bids = sorted(self.bids.items(), reverse=True)[:levels]
        sorted_asks = sorted(self.asks.items())[:levels]
        
        return {
            'bids': [(price, size) for price, size in sorted_bids if size > 0],
            'asks': [(price, size) for price, size in sorted_asks if size > 0]
        }

# Process ITCH messages to build order book
order_book = OrderBook()
parser = ITCHParser()

def process_itch_stream(data_stream):
    """Process stream of ITCH messages"""
    for raw_message in data_stream:
        message = parser.parse(raw_message)
        
        if message.message_type == 'A':
            # Add Order
            order_book.add_order(
                message.data['order_ref'],
                message.data['side'],
                message.data['shares'],
                Decimal(str(message.data['price']))
            )
        
        elif message.message_type == 'E':
            # Order Executed
            order_book.execute_order(
                message.data['order_ref'],
                message.data['executed_shares']
            )
        
        elif message.message_type == 'X':
            # Order Cancelled
            order_book.cancel_order(
                message.data['order_ref'],
                message.data['cancelled_shares']
            )
        
        elif message.message_type == 'P':
            # Trade (may not have order ref in book)
            pass
        
        # Print BBO after each update
        best_bid, best_ask = order_book.get_best_bid_ask()
        print(f"BBO: {best_bid} x {best_ask}")
\`\`\`

---

## Protocol Comparison

| Feature | FIX | FAST | ITCH |
|---------|-----|------|------|
| **Format** | Text (tag=value) | Binary | Binary (fixed length) |
| **Use Case** | Order routing | Market data (fast) | NASDAQ raw feed |
| **Compression** | None | 70-90% | None needed |
| **Message Size** | 100-500 bytes | 20-50 bytes | 20-60 bytes |
| **Throughput** | 10K-100K msg/sec | 100K-1M msg/sec | 1M+ msg/sec |
| **Latency** | 100 μs - 1 ms | 10-100 μs | 1-10 μs |
| **Parsing Speed** | Slow (text) | Fast (binary) | Fastest (fixed) |
| **Human Readable** | Yes | No | No |
| **Standardization** | Industry standard | FIX-based | NASDAQ proprietary |

---

## When to Use Each Protocol

**Use FIX when:**
- ✅ Sending orders to brokers/exchanges
- ✅ Building OMS/EMS systems
- ✅ Interoperating with many counterparties
- ✅ Need human-readable messages (debugging)
- ✅ Speed is not critical (< 1ms acceptable)

**Use FAST when:**
- ✅ Consuming market data from exchanges
- ✅ Bandwidth is limited
- ✅ Need 100-1000× faster than text FIX
- ✅ Exchange offers FAST feeds (CME, ICE)

**Use ITCH when:**
- ✅ Need complete NASDAQ order book
- ✅ Building market-making strategies
- ✅ Require every order/trade (no conflation)
- ✅ Ultra-low latency required (< 10 μs)
- ✅ Can handle 1M+ messages per second

**Use HTTP/WebSocket when:**
- ✅ Building retail applications
- ✅ Latency > 10ms acceptable
- ✅ Want simple implementation
- ✅ Using vendor APIs (Polygon, IEX)

---

## Best Practices

1. **FIX Session Management**
   - Implement heartbeats (30-60 seconds)
   - Track sequence numbers (detect gaps)
   - Handle logon/logout gracefully
   - Store messages for replay

2. **FAST Decoding**
   - Load templates at startup
   - Cache decoded messages
   - Validate presence maps
   - Monitor template changes

3. **ITCH Processing**
   - Use UDP multicast for receive
   - Implement gap detection
   - Process messages in sequence
   - Maintain full order book state

4. **Performance**
   - Use binary protocols for production
   - Minimize memory allocations
   - Batch message processing
   - Consider C++/Rust for ultra-low latency

5. **Testing**
   - Replay historical data
   - Simulate network issues
   - Validate against vendor specs
   - Measure end-to-end latency

---

## Next Steps

Now that you understand financial protocols, you can:
1. **Build FIX order router** to send orders to brokers
2. **Consume FAST feeds** from CME/ICE for futures data
3. **Process ITCH** to reconstruct NASDAQ order book
4. **Optimize parsing** with Cython or C++ for speed
5. **Compare protocols** in your own performance tests

These protocols are the foundation of professional trading systems - mastering them opens doors to institutional finance infrastructure.
`,
};

