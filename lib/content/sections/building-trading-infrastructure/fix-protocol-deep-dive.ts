export const fixProtocolDeepDive = {
    title: 'FIX Protocol Deep Dive',
    id: 'fix-protocol-deep-dive',
    content: `
# FIX Protocol Deep Dive

## Introduction

The **Financial Information eXchange (FIX) Protocol** is the industry-standard messaging protocol for electronic trading. It's how brokers, exchanges, and trading systems communicate.

**FIX Facts:**
- Created in 1992 by Fidelity Investments and Salomon Brothers
- Used by 300+ firms worldwide
- Handles trillions of dollars in daily trading volume
- Current version: FIX 5.0 SP2 (2009), but FIX 4.2 (1998) still widely used

**Why FIX Matters:**
- **Universal language**: All major brokers and exchanges support FIX
- **Trading operations**: Order entry, execution reports, market data
- **Cross-asset**: Equities, options, futures, FX, fixed income
- **Mission-critical**: High availability, low latency required

This section builds a production-grade FIX engine from scratch.

---

## FIX Message Structure

### Basic Format

FIX messages are tag-value pairs separated by SOH (Start of Header, ASCII 0x01):

\`\`\`
8=FIX.4.2|9=100|35=D|49=SENDER|56=TARGET|34=1|52=20240115-10:30:00|...|10=123|
\`\`\`

(| represents SOH character in examples)

### Key Fields

\`\`\`python
"""
FIX Protocol Implementation
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, List
from datetime import datetime
import hashlib

class FIXTag(Enum):
    """Standard FIX tags"""
    # Header
    BeginString = 8  # FIX version
    BodyLength = 9  # Message length
    MsgType = 35  # Message type
    SenderCompID = 49  # Sender
    TargetCompID = 56  # Receiver
    MsgSeqNum = 34  # Sequence number
    SendingTime = 52  # Timestamp
    
    # Trailer
    CheckSum = 10  # Message checksum
    
    # Order Fields
    ClOrdID = 11  # Client order ID
    OrderID = 37  # Exchange order ID
    ExecID = 17  # Execution ID
    Symbol = 55  # Ticker symbol
    Side = 54  # Buy/Sell
    OrderQty = 38  # Quantity
    OrdType = 40  # Order type
    Price = 44  # Limit price
    TimeInForce = 59  # TIF
    
    # Execution Report
    OrdStatus = 39  # Order status
    ExecType = 150  # Execution type
    LeavesQty = 151  # Remaining quantity
    CumQty = 14  # Cumulative filled
    AvgPx = 6  # Average price
    LastQty = 32  # Last fill quantity
    LastPx = 31  # Last fill price
    
    # Account
    Account = 1  # Trading account
    
    # Text
    Text = 58  # Free-form text


class FIXMsgType(Enum):
    """FIX message types"""
    Heartbeat = '0'
    TestRequest = '1'
    ResendRequest = '2'
    Reject = '3'
    SequenceReset = '4'
    Logout = '5'
    Logon = 'A'
    NewOrderSingle = 'D'
    ExecutionReport = '8'
    OrderCancelRequest = 'F'
    OrderCancelReject = '9'
    OrderCancelReplaceRequest = 'G'
    MarketDataRequest = 'V'
    MarketDataSnapshot = 'W'


class FIXSide(Enum):
    """Order side"""
    Buy = '1'
    Sell = '2'
    SellShort = '5'


class FIXOrdType(Enum):
    """Order types"""
    Market = '1'
    Limit = '2'
    Stop = '3'
    StopLimit = '4'


class FIXTimeInForce(Enum):
    """Time in force"""
    Day = '0'
    GTC = '1'  # Good til cancel
    OPG = '2'  # At the open
    IOC = '3'  # Immediate or cancel
    FOK = '4'  # Fill or kill
    GTD = '6'  # Good til date


class FIXOrdStatus(Enum):
    """Order status"""
    New = '0'
    PartiallyFilled = '1'
    Filled = '2'
    DoneForDay = '3'
    Canceled = '4'
    Replaced = '5'
    PendingCancel = '6'
    Stopped = '7'
    Rejected = '8'
    Suspended = '9'
    PendingNew = 'A'
    Calculated = 'B'
    Expired = 'C'
    PendingReplace = 'E'


@dataclass
class FIXMessage:
    """
    FIX message representation
    """
    msg_type: str
    fields: Dict[int, str]
    raw_message: Optional[str] = None
    
    SOH = '\\x01'  # Start of Header character
    
    def __post_init__(self):
        """Ensure required header fields"""
        if FIXTag.BeginString.value not in self.fields:
            self.fields[FIXTag.BeginString.value] = "FIX.4.2"
        if FIXTag.MsgType.value not in self.fields:
            self.fields[FIXTag.MsgType.value] = self.msg_type
    
    def set_field(self, tag: int, value: str):
        """Set field value"""
        self.fields[tag] = str(value)
    
    def get_field(self, tag: int) -> Optional[str]:
        """Get field value"""
        return self.fields.get(tag)
    
    def to_fix_string(self) -> str:
        """
        Convert to FIX protocol string
        
        Format: tag=value|tag=value|...|
        where | is SOH character
        """
        # Build body (all fields except header and trailer)
        body_fields = {}
        for tag, value in self.fields.items():
            if tag not in [
                FIXTag.BeginString.value,
                FIXTag.BodyLength.value,
                FIXTag.CheckSum.value
            ]:
                body_fields[tag] = value
        
        # Sort by tag number (FIX requirement)
        sorted_tags = sorted(body_fields.keys())
        
        # Build body string
        body_parts = [f"{tag}={body_fields[tag]}" for tag in sorted_tags]
        body = self.SOH.join(body_parts) + self.SOH
        
        # Calculate body length
        body_length = len(body)
        
        # Build header
        header = f"8={self.fields[FIXTag.BeginString.value]}{self.SOH}"
        header += f"9={body_length}{self.SOH}"
        
        # Build complete message (without checksum)
        message_without_checksum = header + body
        
        # Calculate checksum
        checksum = self._calculate_checksum(message_without_checksum)
        
        # Complete message
        complete_message = message_without_checksum + f"10={checksum:03d}{self.SOH}"
        
        return complete_message
    
    def _calculate_checksum(self, message: str) -> int:
        """Calculate FIX checksum (sum of bytes mod 256)"""
        return sum(ord(c) for c in message) % 256
    
    @classmethod
    def from_fix_string(cls, fix_string: str) -> 'FIXMessage':
        """Parse FIX string into message"""
        fields = {}
        
        # Split by SOH
        parts = fix_string.split(cls.SOH)
        
        for part in parts:
            if '=' in part:
                tag, value = part.split('=', 1)
                fields[int(tag)] = value
        
        msg_type = fields.get(FIXTag.MsgType.value, '')
        
        return cls(
            msg_type=msg_type,
            fields=fields,
            raw_message=fix_string
        )


class FIXMessageBuilder:
    """
    Builder for creating FIX messages
    """
    
    def __init__(self, sender_comp_id: str, target_comp_id: str):
        self.sender_comp_id = sender_comp_id
        self.target_comp_id = target_comp_id
        self.msg_seq_num = 1
    
    def _create_base_message(self, msg_type: str) -> FIXMessage:
        """Create base message with header fields"""
        msg = FIXMessage(msg_type=msg_type, fields={})
        
        # Header fields
        msg.set_field(FIXTag.BeginString.value, "FIX.4.2")
        msg.set_field(FIXTag.MsgType.value, msg_type)
        msg.set_field(FIXTag.SenderCompID.value, self.sender_comp_id)
        msg.set_field(FIXTag.TargetCompID.value, self.target_comp_id)
        msg.set_field(FIXTag.MsgSeqNum.value, str(self.msg_seq_num))
        msg.set_field(
            FIXTag.SendingTime.value,
            datetime.utcnow().strftime('%Y%m%d-%H:%M:%S.%f')[:-3]
        )
        
        self.msg_seq_num += 1
        
        return msg
    
    def create_logon(self, heartbeat_interval: int = 30) -> FIXMessage:
        """Create Logon message"""
        msg = self._create_base_message(FIXMsgType.Logon.value)
        msg.set_field(98, '0')  # EncryptMethod (None)
        msg.set_field(108, str(heartbeat_interval))  # HeartBtInt
        return msg
    
    def create_heartbeat(self, test_req_id: Optional[str] = None) -> FIXMessage:
        """Create Heartbeat message"""
        msg = self._create_base_message(FIXMsgType.Heartbeat.value)
        if test_req_id:
            msg.set_field(112, test_req_id)  # TestReqID
        return msg
    
    def create_new_order(
        self,
        cl_ord_id: str,
        symbol: str,
        side: str,
        order_qty: str,
        ord_type: str,
        price: Optional[str] = None,
        time_in_force: str = FIXTimeInForce.Day.value,
        account: Optional[str] = None
    ) -> FIXMessage:
        """Create New Order Single message"""
        msg = self._create_base_message(FIXMsgType.NewOrderSingle.value)
        
        msg.set_field(FIXTag.ClOrdID.value, cl_ord_id)
        msg.set_field(FIXTag.Symbol.value, symbol)
        msg.set_field(FIXTag.Side.value, side)
        msg.set_field(FIXTag.OrderQty.value, order_qty)
        msg.set_field(FIXTag.OrdType.value, ord_type)
        msg.set_field(FIXTag.TimeInForce.value, time_in_force)
        
        if price:
            msg.set_field(FIXTag.Price.value, price)
        
        if account:
            msg.set_field(FIXTag.Account.value, account)
        
        # TransactTime (required)
        msg.set_field(60, datetime.utcnow().strftime('%Y%m%d-%H:%M:%S.%f')[:-3])
        
        return msg
    
    def create_order_cancel_request(
        self,
        orig_cl_ord_id: str,
        cl_ord_id: str,
        symbol: str,
        side: str,
        order_qty: str
    ) -> FIXMessage:
        """Create Order Cancel Request"""
        msg = self._create_base_message(FIXMsgType.OrderCancelRequest.value)
        
        msg.set_field(41, orig_cl_ord_id)  # OrigClOrdID
        msg.set_field(FIXTag.ClOrdID.value, cl_ord_id)
        msg.set_field(FIXTag.Symbol.value, symbol)
        msg.set_field(FIXTag.Side.value, side)
        msg.set_field(FIXTag.OrderQty.value, order_qty)
        msg.set_field(60, datetime.utcnow().strftime('%Y%m%d-%H:%M:%S.%f')[:-3])
        
        return msg
    
    def create_execution_report(
        self,
        order_id: str,
        exec_id: str,
        exec_type: str,
        ord_status: str,
        symbol: str,
        side: str,
        order_qty: str,
        cum_qty: str,
        leaves_qty: str,
        avg_px: str,
        last_qty: Optional[str] = None,
        last_px: Optional[str] = None
    ) -> FIXMessage:
        """Create Execution Report"""
        msg = self._create_base_message(FIXMsgType.ExecutionReport.value)
        
        msg.set_field(FIXTag.OrderID.value, order_id)
        msg.set_field(FIXTag.ExecID.value, exec_id)
        msg.set_field(FIXTag.ExecType.value, exec_type)
        msg.set_field(FIXTag.OrdStatus.value, ord_status)
        msg.set_field(FIXTag.Symbol.value, symbol)
        msg.set_field(FIXTag.Side.value, side)
        msg.set_field(FIXTag.OrderQty.value, order_qty)
        msg.set_field(FIXTag.CumQty.value, cum_qty)
        msg.set_field(FIXTag.LeavesQty.value, leaves_qty)
        msg.set_field(FIXTag.AvgPx.value, avg_px)
        
        if last_qty:
            msg.set_field(FIXTag.LastQty.value, last_qty)
        if last_px:
            msg.set_field(FIXTag.LastPx.value, last_px)
        
        msg.set_field(60, datetime.utcnow().strftime('%Y%m%d-%H:%M:%S.%f')[:-3])
        
        return msg


# Example usage
def fix_message_examples():
    """Demonstrate FIX message creation"""
    
    builder = FIXMessageBuilder(
        sender_comp_id="CLIENT1",
        target_comp_id="BROKER1"
    )
    
    print("=" * 70)
    print("FIX MESSAGE EXAMPLES")
    print("=" * 70)
    
    # 1. Logon
    print("\\n1. Logon Message:")
    logon = builder.create_logon(heartbeat_interval=30)
    fix_string = logon.to_fix_string()
    print(f"   {fix_string.replace(FIXMessage.SOH, '|')}")
    
    # 2. New Order
    print("\\n2. New Order Single (Market Buy):")
    order = builder.create_new_order(
        cl_ord_id="ORD-12345",
        symbol="AAPL",
        side=FIXSide.Buy.value,
        order_qty="100",
        ord_type=FIXOrdType.Market.value,
        account="ACC-001"
    )
    fix_string = order.to_fix_string()
    print(f"   {fix_string.replace(FIXMessage.SOH, '|')}")
    
    # 3. Limit Order
    print("\\n3. New Order Single (Limit Sell):")
    limit_order = builder.create_new_order(
        cl_ord_id="ORD-12346",
        symbol="TSLA",
        side=FIXSide.Sell.value,
        order_qty="50",
        ord_type=FIXOrdType.Limit.value,
        price="250.50",
        time_in_force=FIXTimeInForce.GTC.value
    )
    fix_string = limit_order.to_fix_string()
    print(f"   {fix_string.replace(FIXMessage.SOH, '|')}")
    
    # 4. Execution Report (Fill)
    print("\\n4. Execution Report (Filled):")
    exec_report = builder.create_execution_report(
        order_id="12345",
        exec_id="EXEC-001",
        exec_type='F',  # Trade
        ord_status=FIXOrdStatus.Filled.value,
        symbol="AAPL",
        side=FIXSide.Buy.value,
        order_qty="100",
        cum_qty="100",
        leaves_qty="0",
        avg_px="150.25",
        last_qty="100",
        last_px="150.25"
    )
    fix_string = exec_report.to_fix_string()
    print(f"   {fix_string.replace(FIXMessage.SOH, '|')}")

# fix_message_examples()
\`\`\`

---

## FIX Session Management

\`\`\`python
"""
FIX Session Manager
"""

import asyncio
import socket
from typing import Optional, Callable
from collections import deque

class FIXSession:
    """
    FIX session with sequence number management
    """
    
    def __init__(
        self,
        sender_comp_id: str,
        target_comp_id: str,
        host: str,
        port: int
    ):
        self.sender_comp_id = sender_comp_id
        self.target_comp_id = target_comp_id
        self.host = host
        self.port = port
        
        # Sequence numbers
        self.outgoing_seq_num = 1
        self.incoming_seq_num = 1
        
        # Session state
        self.is_logged_on = False
        self.heartbeat_interval = 30
        self.last_heartbeat_sent = datetime.utcnow()
        self.last_message_received = datetime.utcnow()
        
        # Socket
        self.socket: Optional[socket.socket] = None
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        
        # Message queue
        self.outbound_queue = deque()
        
        # Callbacks
        self.on_message: Optional[Callable] = None
        
        # Builder
        self.builder = FIXMessageBuilder(sender_comp_id, target_comp_id)
    
    async def connect(self):
        """Connect to FIX server"""
        self.reader, self.writer = await asyncio.open_connection(
            self.host,
            self.port
        )
        print(f"[FIXSession] Connected to {self.host}:{self.port}")
    
    async def login(self):
        """Send logon message"""
        logon_msg = self.builder.create_logon(self.heartbeat_interval)
        await self.send_message(logon_msg)
        print("[FIXSession] Logon sent")
    
    async def send_message(self, message: FIXMessage):
        """Send FIX message"""
        fix_string = message.to_fix_string()
        
        # Update sequence number
        message.set_field(FIXTag.MsgSeqNum.value, str(self.outgoing_seq_num))
        self.outgoing_seq_num += 1
        
        # Send
        self.writer.write(fix_string.encode('utf-8'))
        await self.writer.drain()
        
        print(f"[FIXSession] Sent: {message.msg_type} (seq: {self.outgoing_seq_num-1})")
    
    async def receive_message(self) -> Optional[FIXMessage]:
        """Receive FIX message"""
        try:
            # Read until SOH character
            data = await self.reader.readuntil(b'\\x01')
            
            # Check if complete message (ends with 10=xxx|)
            if b'10=' in data:
                fix_string = data.decode('utf-8')
                message = FIXMessage.from_fix_string(fix_string)
                
                # Update sequence number
                expected_seq = self.incoming_seq_num
                received_seq = int(message.get_field(FIXTag.MsgSeqNum.value))
                
                if received_seq != expected_seq:
                    print(f"[FIXSession] Sequence gap: expected {expected_seq}, got {received_seq}")
                    # Handle gap (resend request)
                
                self.incoming_seq_num = received_seq + 1
                self.last_message_received = datetime.utcnow()
                
                print(f"[FIXSession] Received: {message.msg_type} (seq: {received_seq})")
                
                return message
        
        except Exception as e:
            print(f"[FIXSession] Error receiving: {e}")
            return None
    
    async def heartbeat_monitor(self):
        """Monitor heartbeats"""
        while self.is_logged_on:
            now = datetime.utcnow()
            
            # Send heartbeat if needed
            time_since_send = (now - self.last_heartbeat_sent).total_seconds()
            if time_since_send >= self.heartbeat_interval:
                heartbeat = self.builder.create_heartbeat()
                await self.send_message(heartbeat)
                self.last_heartbeat_sent = now
            
            # Check for timeout
            time_since_receive = (now - self.last_message_received).total_seconds()
            if time_since_receive >= self.heartbeat_interval * 2:
                print("[FIXSession] Heartbeat timeout - connection lost")
                self.is_logged_on = False
                break
            
            await asyncio.sleep(1)
    
    async def disconnect(self):
        """Disconnect from FIX server"""
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        print("[FIXSession] Disconnected")
\`\`\`

---

## Summary

**FIX Protocol Essentials:**
1. **Message Structure**: Tag=Value pairs separated by SOH
2. **Session Management**: Sequence numbers, heartbeats, logon/logout
3. **Order Flow**: NewOrderSingle → ExecutionReport (acks, fills)
4. **Message Types**: 100+ defined, most common are D (order), 8 (execution)
5. **Reliability**: Sequence numbers prevent message loss

**Real-World Implementations:**
- **QuickFIX**: Open-source C++/Java/Python FIX engine
- **FIX.io**: Node.js FIX implementation
- **OnixS**: Commercial high-performance FIX engine
- **B2BITS**: Low-latency FIX for HFT

**Next Section**: Module 14.5 - Smart Order Routing (advanced routing logic)
`,
};

