export const conversationalTradingAssistants = {
    title: 'Conversational Trading Assistants',
    id: 'conversational-trading-assistants',
    content: `
# Conversational Trading Assistants

## Introduction

Trading platforms traditionally require learning complex interfaces, charts, and order entry systems. Conversational AI enables natural language interaction with trading systems-ask questions about your portfolio, market conditions, or execute trades using plain English. This democratizes trading while maintaining safety and control.

This section covers building LLM-powered trading assistants: natural language portfolio queries, conversational market analysis, voice-controlled trading with safeguards, strategy recommendation systems, alert explanation, and user-friendly trading interfaces.

### Why Conversational Interfaces for Trading

**Accessibility**: Lower barrier to entry for new traders
**Efficiency**: Faster access to information without navigating complex UIs
**Context**: AI understands intent and provides relevant information
**Education**: Explains concepts and decisions in plain language
**Safety**: Can add conversational guardrails and confirmations

---

## Natural Language Portfolio Queries

### Building Portfolio Query System

\`\`\`python
"""
Natural language portfolio query system
"""

import anthropic
from typing import Dict, List
import json

class PortfolioQueryAssistant:
    """
    Answer natural language questions about portfolio
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def query_portfolio(self, user_question: str,
                       portfolio_data: Dict) -> str:
        """
        Answer user questions about their portfolio
        
        Args:
            user_question: Natural language question
            portfolio_data: Complete portfolio data
            
        Returns:
            Natural language answer
        """
        prompt = f"""You are a helpful portfolio assistant. Answer the user's question about their portfolio clearly and concisely.

User Question: "{user_question}"

Portfolio Data:
- Total Value: \\${portfolio_data['total_value']:, .2f
}
    - Total Return: {portfolio_data['total_return']: .2f}%
        - Cash: \\${ portfolio_data['cash']:, .2f }

Holdings:
{ self._format_holdings(portfolio_data['holdings']) }

Performance:
{ self._format_performance(portfolio_data['performance']) }

Provide a clear, conversational answer.If suggesting actions, explain reasoning.
If the question requires analysis not directly in the data, make reasonable inferences but note assumptions."""

response = self.client.messages.create(
    model = self.model,
    max_tokens = 1000,
    messages = [{ "role": "user", "content": prompt }]
)

return response.content[0].text
    
    def multi_turn_conversation(self, conversation_history: List[Dict],
    portfolio_data: Dict) -> str:
"""
        Handle multi - turn conversation with context
        
        Args:
conversation_history: Previous messages
portfolio_data: Portfolio data

Returns:
            Response to latest message
"""
        # Format portfolio data
portfolio_context = f"""
Portfolio Summary:
- Total Value: \\${ portfolio_data['total_value']:, .2f }
- Holdings: { len(portfolio_data['holdings']) } positions
    - Top Positions: { ', '.join([h['ticker'] for h in portfolio_data['holdings'][: 5]]) }
- Performance: { portfolio_data['total_return']: .2f }% total return
"""

        # Build conversation with portfolio context
messages = [
    {
        "role": "user",
        "content": f"You are a portfolio assistant. Use this portfolio data to answer questions:\\n{portfolio_context}"
            },
    {
        "role": "assistant",
        "content": "I understand. I'll help you with questions about your portfolio."
    }
]
        
        # Add conversation history
messages.extend(conversation_history)

response = self.client.messages.create(
    model = self.model,
    max_tokens = 1000,
    messages = messages
)

return response.content[0].text
    
    def explain_holding(self, ticker: str, holding_data: Dict,
    market_context: Dict) -> str:
"""
        Explain a specific holding to the user

Args:
ticker: Stock ticker
holding_data: Data about the holding
market_context: Current market conditions

Returns:
Explanation
"""
prompt = f"""Explain this portfolio holding to the user in simple, conversational language.

Holding: { ticker }
Company: { holding_data.get('name') }
Shares: { holding_data.get('shares') }
Cost Basis: \\${ holding_data.get('cost_basis') }
Current Price: \\${ holding_data.get('current_price') }
Market Value: \\${ holding_data.get('market_value'):, .2f }
Return: { holding_data.get('return'): .2f }%
    Weight in Portfolio: { holding_data.get('weight'): .1f }%

        Recent Performance:
{ holding_data.get('recent_performance', 'Not available') }

Market Context:
{ market_context.get('summary', 'Market conditions normal') }

Explain:
1. What the company does(if known)
    2. How the position has performed
3. Why it might be performing this way
4. Whether the position size is appropriate
5. Any considerations for the user

Keep it conversational and avoid jargon."""

response = self.client.messages.create(
    model = self.model,
    max_tokens = 800,
    messages = [{ "role": "user", "content": prompt }]
)

return response.content[0].text
    
    def suggest_portfolio_actions(self, portfolio_data: Dict,
    user_goals: Dict,
    risk_tolerance: str) -> str:
"""
        Suggest portfolio actions in natural language

Args:
portfolio_data: Portfolio data
user_goals: User's investment goals
risk_tolerance: Risk tolerance level

Returns:
            Action suggestions
"""
prompt = f"""Provide portfolio suggestions for this investor.

Portfolio:
{ self._format_holdings(portfolio_data['holdings']) }

Investor Profile:
- Goals: { user_goals.get('primary_goal') }
- Time Horizon: { user_goals.get('time_horizon') }
- Risk Tolerance: { risk_tolerance }

Current Allocation:
{ self._format_allocation(portfolio_data.get('allocation', {})) }

Provide 3 - 5 specific, actionable suggestions in conversational language:
1. Rebalancing if needed
2. Concentration risks to address
3. Diversification opportunities
4. Position sizing adjustments
5. Risk management considerations

For each suggestion, explain:
- What to do
    - Why it's important
        - How to do it(specific actions)

Be supportive and educational."""

response = self.client.messages.create(
    model = self.model,
    max_tokens = 1500,
    messages = [{ "role": "user", "content": prompt }]
)

return response.content[0].text
    
    def _format_holdings(self, holdings: List[Dict]) -> str:
"""Format holdings for display"""
return "\\n".join([
    f"- {h['ticker']} ({h['name']}): \\${h['market_value']:,.2f} ({h['return']:+.1f}%)"
            for h in holdings
        ])
    
    def _format_performance(self, performance: Dict) -> str:
"""Format performance metrics"""
return "\\n".join([
    f"- {k.replace('_', ' ').title()}: {v}"
            for k, v in performance.items()
        ])
    
    def _format_allocation(self, allocation: Dict) -> str:
"""Format portfolio allocation"""
return "\\n".join([
    f"- {category}: {pct:.1f}%"
            for category, pct in allocation.items()
        ])

# Example usage
assistant = PortfolioQueryAssistant(api_key = "your-key")

portfolio_data = {
    'total_value': 125000,
    'total_return': 18.5,
    'cash': 5000,
    'holdings': [
        {
            'ticker': 'AAPL',
            'name': 'Apple Inc.',
            'shares': 100,
            'cost_basis': 150.00,
            'current_price': 180.50,
            'market_value': 18050,
            'return': 20.3,
            'weight': 14.4
        },
        {
            'ticker': 'MSFT',
            'name': 'Microsoft Corp.',
            'shares': 50,
            'cost_basis': 320.00,
            'current_price': 370.00,
            'market_value': 18500,
            'return': 15.6,
            'weight': 14.8
        }
    ],
    'performance': {
        'ytd_return': 15.2,
        'sharpe_ratio': 1.45,
        'max_drawdown': -8.5
    },
    'allocation': {
        'Technology': 45.0,
        'Healthcare': 20.0,
        'Financials': 15.0,
        'Consumer': 12.0,
        'Cash': 8.0
    }
}

# Example queries
questions = [
    "What's my best performing stock?",
    "Am I too concentrated in tech?",
    "How much cash do I have available?",
    "Should I sell any positions?"
]

for question in questions:
    print(f"\\nQ: {question}")
answer = assistant.query_portfolio(question, portfolio_data)
print(f"A: {answer}")
\`\`\`

---

## Voice-Controlled Trading

### Building Voice Trading Interface with Safety

\`\`\`python
"""
Voice-controlled trading with safety guardrails
"""

class VoiceTradingAssistant:
    """
    Voice-activated trading assistant with safety checks
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def parse_voice_command(self, voice_input: str,
                           portfolio_context: Dict) -> Dict:
        """
        Parse voice trading command
        
        Args:
            voice_input: Transcribed voice command
            portfolio_context: Current portfolio state
            
        Returns:
            Structured command with safety checks
        """
        prompt = f"""Parse this voice trading command into a structured action.

Voice Input: "{voice_input}"

Portfolio Context:
- Cash Available: \\${portfolio_context.get('cash', 0):, .2f}
- Buying Power: \\${ portfolio_context.get('buying_power', 0):, .2f }
- Current Holdings: { ', '.join(portfolio_context.get('tickers', [])) }

Parse the command and return JSON:
{
    {
        "intent": "BUY/SELL/CHECK_PRICE/GET_QUOTE/CHECK_POSITION/CANCEL_ORDER/OTHER",
            "ticker": "Stock ticker if mentioned",
                "quantity": "Number of shares (or null)",
                    "order_type": "MARKET/LIMIT/STOP (or null)",
                        "limit_price": "Price if limit order (or null)",
                            "action_confidence": 0.0 - 1.0,
                                "ambiguities": ["Any unclear aspects"],
                                    "clarification_needed": true / false,
                                        "clarification_questions": ["Questions to ask user"],
                                            "safety_checks": {
                                                {
                                                    "is_valid_ticker": true / false,
                                                        "sufficient_funds": true / false,
                                                            "reasonable_quantity": true / false,
                                                                "price_in_range": true / false
                                                }
        },
        "estimated_cost": "Estimated trade cost",
            "confirmation_message": "Human-friendly confirmation message"
    }
}

If the command is ambiguous or potentially unsafe, mark clarification_needed as true."""

response = self.client.messages.create(
    model = self.model,
    max_tokens = 1000,
    messages = [{ "role": "user", "content": prompt }]
)

return self._parse_json(response.content[0].text)
    
    def generate_confirmation(self, parsed_command: Dict,
    current_price: float) -> str:
"""
        Generate natural language confirmation

Args:
parsed_command: Parsed command structure
current_price: Current market price

Returns:
            Confirmation message
"""
intent = parsed_command['intent']
ticker = parsed_command.get('ticker')
quantity = parsed_command.get('quantity')
order_type = parsed_command.get('order_type', 'MARKET')

if intent == 'BUY':
    message = f"""I understand you want to buy {quantity} shares of {ticker} at {order_type.lower()} price.

Current market price: \\${ current_price: .2f }
Estimated cost: ~\\${ quantity * current_price:, .2f }

Would you like me to proceed with this order ?
    Say "Yes, execute" to confirm or "Cancel" to abort."""
        
        elif intent == 'SELL':
message = f"""I understand you want to sell {quantity} shares of {ticker} at {order_type.lower()} price.

Current market price: \\${ current_price: .2f }
Estimated proceeds: ~\\${ quantity * current_price:, .2f }

Would you like me to proceed with this order ?
    Say "Yes, execute" to confirm or "Cancel" to abort."""
        
        else:
message = parsed_command.get('confirmation_message',
    "I didn't fully understand. Could you please clarify?")

return message
    
    def validate_command_safety(self, parsed_command: Dict,
    portfolio_data: Dict,
    risk_limits: Dict) -> Dict:
"""
        Validate command against safety rules

Args:
parsed_command: Parsed command
portfolio_data: Portfolio data
risk_limits: Risk management rules

Returns:
            Validation result
"""
safety_checks = parsed_command.get('safety_checks', {})
ticker = parsed_command.get('ticker')
quantity = parsed_command.get('quantity', 0)
intent = parsed_command['intent']

issues = []
warnings = []
        
        # Check 1: Ticker validity
if not safety_checks.get('is_valid_ticker'):
issues.append(f"'{ticker}' is not a valid stock ticker")
        
        # Check 2: Sufficient funds
if intent == 'BUY' and not safety_checks.get('sufficient_funds'):
issues.append("Insufficient funds for this purchase")
        
        # Check 3: Position size limits
estimated_cost = parsed_command.get('estimated_cost', 0)
max_position_size = risk_limits.get('max_position_pct', 10)
portfolio_value = portfolio_data.get('total_value', 0)

if estimated_cost / portfolio_value * 100 > max_position_size:
    warnings.append(f"This trade would exceed maximum position size ({max_position_size}%)")
        
        # Check 4: Selling more than owned
if intent == 'SELL':
    current_position = portfolio_data.get('holdings', {}).get(ticker, {}).get('shares', 0)
if quantity > current_position:
    issues.append(f"You only own {current_position} shares of {ticker}")
        
        # Check 5: Market hours
        # In production, check if market is open
        
        # Check 6: Price reasonableness
if not safety_checks.get('price_in_range'):
warnings.append("The limit price seems unusual for this stock")

return {
    'approved': len(issues) == 0,
    'issues': issues,
    'warnings': warnings,
    'requires_override': len(warnings) > 0 and len(issues) == 0,
    'recommendation': 'PROCEED' if len(issues) == 0 else 'BLOCK'
        }
    
    def execute_confirmed_order(self, parsed_command: Dict,
        trading_api: any) -> Dict:
"""
        Execute confirmed order

Args:
parsed_command: Validated and confirmed command
trading_api: Trading API client

Returns:
            Execution result
"""
        # In production: Execute via trading API
        # For now, return mock result

return {
    'success': True,
    'order_id': 'ORD-12345',
    'message': f"Order executed: {parsed_command['intent']} {parsed_command['quantity']} {parsed_command['ticker']}",
    'confirmation_number': 'ORD-12345'
}
    
    def _parse_json(self, response_text: str) -> Dict:
"""Parse JSON from response"""
import json
        try:
if "```json" in response_text:
    json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
json_str = response_text
return json.loads(json_str)
except:
return {}

# Example usage
voice_assistant = VoiceTradingAssistant(api_key = "your-key")

# Example voice commands
voice_commands = [
    "Buy 100 shares of Apple",
    "Sell half my Microsoft position",
    "What's the price of Tesla?",
    "Place a limit order to buy NVDA at 450 dollars"
]

portfolio_context = {
    'cash': 10000,
    'buying_power': 10000,
    'tickers': ['AAPL', 'MSFT', 'GOOGL'],
    'total_value': 50000,
    'holdings': {
        'MSFT': { 'shares': 50 }
    }
}

for voice_input in voice_commands:
    print(f"\\n{'='*60}")
print(f"Voice Input: \"{voice_input}\"")
print(f"{'='*60}")
    
    # Parse command
parsed = voice_assistant.parse_voice_command(voice_input, portfolio_context)
print(f"\\nParsed Intent: {parsed.get('intent')}")
    
    # Generate confirmation
if parsed.get('ticker'):
    confirmation = voice_assistant.generate_confirmation(parsed, current_price = 180.50)
print(f"\\nConfirmation:\\n{confirmation}")
    
    # Validate safety
risk_limits = { 'max_position_pct': 10 }
validation = voice_assistant.validate_command_safety(
    parsed,
    { 'total_value': 50000, 'holdings': portfolio_context['holdings'] },
    risk_limits
)

print(f"\\nSafety Validation:")
print(f"Approved: {validation['approved']}")
if validation['issues']:
    print(f"Issues: {', '.join(validation['issues'])}")
if validation['warnings']:
    print(f"Warnings: {', '.join(validation['warnings'])}")
\`\`\`

---

## Market Analysis Chat

### Conversational Market Insights

\`\`\`python
"""
Conversational market analysis assistant
"""

class MarketAnalysisChat:
    """
    Chat interface for market analysis and education
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def answer_market_question(self, question: str,
                               market_data: Dict,
                               user_level: str = "intermediate") -> str:
        """
        Answer market-related questions conversationally
        
        Args:
            question: User's question
            market_data: Current market data
            user_level: User sophistication (novice/intermediate/advanced)
            
        Returns:
            Conversational answer
        """
        complexity_guidance = {
            'novice': 'Use simple language, explain all terms, use analogies',
            'intermediate': 'Balance technical accuracy with accessibility',
            'advanced': 'Use technical terminology, provide detailed analysis'
        }
        
        prompt = f"""Answer this market-related question conversationally.

User Question: "{question}"

User Level: {user_level}
Guidance: {complexity_guidance[user_level]}

Current Market Data:
- S&P 500: {market_data.get('sp500', 'N/A')} ({market_data.get('sp500_change', 'N/A')})
- VIX: {market_data.get('vix', 'N/A')}
- 10Y Treasury: {market_data.get('treasury_10y', 'N/A')}
- Recent News: {market_data.get('top_news', 'Not available')}

Provide a helpful, educational answer that:
1. Directly answers the question
2. Provides relevant context
3. Explains reasoning if applicable
4. Suggests where to learn more if complex
5. Uses appropriate level of detail for user level

Be conversational but accurate."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def explain_market_movement(self, ticker: str,
                                price_change: float,
                                news_context: List[str]) -> str:
        """
        Explain why a stock moved
        
        Args:
            ticker: Stock ticker
            price_change: Percentage change
            news_context: Related news
            
        Returns:
            Explanation
        """
        direction = "up" if price_change > 0 else "down"
        
        prompt = f"""Explain in simple terms why {ticker} moved {direction} {abs(price_change):.1f}% today.

Recent News:
{chr(10).join([f"- {news}" for news in news_context])}

Provide a brief, conversational explanation (2-3 sentences) that:
1. States the likely reason
2. Mentions key news if relevant
3. Adds context if needed

Keep it simple and factual."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def recommend_learning_resources(self, topic: str,
                                     user_level: str) -> str:
        """
        Recommend learning resources for trading topics
        
        Args:
            topic: Trading topic
            user_level: User sophistication level
            
        Returns:
            Recommendations
        """
        prompt = f"""Recommend learning resources for someone interested in: {topic}

User Level: {user_level}

Provide 3-5 specific recommendations including:
1. Books (classic and modern)
2. Online courses or platforms
3. Practice resources (simulators, paper trading)
4. Communities or forums
5. Specific next steps

Be specific with titles, platforms, and why each resource is helpful.
Tailor recommendations to the user's level."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text

# Example usage
market_chat = MarketAnalysisChat(api_key="your-key")

market_data = {
    'sp500': 4500,
    'sp500_change': '+0.8%',
    'vix': 16.5,
    'treasury_10y': '4.25%',
    'top_news': 'Fed holds rates steady, signals possible cut in Q2'
}

# Example questions
questions = [
    "Why did the market go up today?",
    "What does VIX mean?",
    "Should I buy stocks when VIX is high?",
    "How do interest rates affect stocks?"
]

for question in questions:
    print(f"\\nQ: {question}")
    answer = market_chat.answer_market_question(
        question, 
        market_data,
        user_level="novice"
    )
    print(f"A: {answer}")
\`\`\`

---

## Alert and Notification Explanations

### Making Alerts Actionable

\`\`\`python
"""
Generate explanatory alerts and notifications
"""

class AlertExplainer:
    """
    Generate explanatory, actionable alerts
    """
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
    
    def generate_alert_message(self, alert_type: str,
                               alert_data: Dict,
                               user_context: Dict) -> str:
        """
        Generate user-friendly alert with explanation
        
        Args:
            alert_type: Type of alert
            alert_data: Alert details
            user_context: User's portfolio and preferences
            
        Returns:
            Alert message
        """
        prompt = f"""Generate a user-friendly alert message.

Alert Type: {alert_type}
Alert Details:
{json.dumps(alert_data, indent=2)}

User Context:
{json.dumps(user_context, indent=2)}

Create an alert message that:
1. States what happened clearly
2. Explains why it matters to the user
3. Suggests what the user might want to do
4. Sets appropriate urgency
5. Is conversational and non-alarming

Format as a short message (2-4 sentences) that could be sent as notification."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def explain_stop_loss_trigger(self, ticker: str,
                                  stop_loss_price: float,
                                  current_price: float,
                                  position_data: Dict) -> str:
        """
        Explain why stop loss was triggered
        
        Args:
            ticker: Stock ticker
            stop_loss_price: Stop loss price
            current_price: Current price
            position_data: Position information
            
        Returns:
            Explanation
        """
        loss_pct = ((current_price - position_data['cost_basis']) / 
                    position_data['cost_basis'] * 100)
        
        prompt = f"""Explain to the user why their stop loss was triggered.

Position: {ticker}
Stop Loss Price: \\${stop_loss_price: .2f}
Current Price: \\${ current_price: .2f }
Original Cost: \\${ position_data['cost_basis']: .2f }
Loss: { loss_pct: .1f }%

    Explain in a supportive, educational tone:
1. What happened
2. Why we have stop losses
3. That this protects them from larger losses
4. What they can do next

Keep it brief and reassuring."""

response = self.client.messages.create(
    model = self.model,
    max_tokens = 400,
    messages = [{ "role": "user", "content": prompt }]
)

return response.content[0].text

# Example usage
alert_explainer = AlertExplainer(api_key = "your-key")

# Price alert
alert_data = {
    'ticker': 'AAPL',
    'trigger_price': 180.00,
    'current_price': 180.50,
    'change_pct': 2.5
}

user_context = {
    'owns_stock': True,
    'shares': 100,
    'cost_basis': 170.00
}

alert_message = alert_explainer.generate_alert_message(
    'PRICE_TARGET_HIT',
    alert_data,
    user_context
)

print("Alert Message:")
print(alert_message)
\`\`\`

---

## Production Chat Interface

### Complete Trading Assistant

\`\`\`python
"""
Complete conversational trading assistant
"""

class TradingChatAssistant:
    """
    Complete trading assistant with conversation memory
    """
    
    def __init__(self, api_key: str):
        self.portfolio_assistant = PortfolioQueryAssistant(api_key)
        self.voice_assistant = VoiceTradingAssistant(api_key)
        self.market_chat = MarketAnalysisChat(api_key)
        self.alert_explainer = AlertExplainer(api_key)
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20241022"
        
        # Conversation state
        self.conversation_history = []
        self.user_profile = {}
        self.portfolio_data = {}
    
    def chat(self, user_message: str) -> str:
        """
        Main chat interface
        
        Args:
            user_message: User's message
            
        Returns:
            Assistant's response
        """
        # Classify intent
        intent = self._classify_intent(user_message)
        
        # Route to appropriate handler
        if intent == 'PORTFOLIO_QUERY':
            response = self.portfolio_assistant.query_portfolio(
                user_message, self.portfolio_data
            )
        elif intent == 'TRADING_COMMAND':
            parsed = self.voice_assistant.parse_voice_command(
                user_message, self.portfolio_data
            )
            response = self.voice_assistant.generate_confirmation(
                parsed, current_price=180.50  # Would fetch real price
            )
        elif intent == 'MARKET_QUESTION':
            response = self.market_chat.answer_market_question(
                user_message,
                self._get_market_data(),
                self.user_profile.get('level', 'intermediate')
            )
        else:
            response = self._general_response(user_message)
        
        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': user_message
        })
        self.conversation_history.append({
            'role': 'assistant',
            'content': response
        })
        
        return response
    
    def _classify_intent(self, message: str) -> str:
        """Classify user intent"""
        # Simplified classification
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['buy', 'sell', 'order', 'trade']):
            return 'TRADING_COMMAND'
        elif any(word in message_lower for word in ['portfolio', 'position', 'holding', 'my ']):
            return 'PORTFOLIO_QUERY'
        elif any(word in message_lower for word in ['market', 'why', 'what happened', 'explain']):
            return 'MARKET_QUESTION'
        else:
            return 'GENERAL'
    
    def _general_response(self, message: str) -> str:
        """Handle general queries"""
        messages = self.conversation_history + [
            {'role': 'user', 'content': message}
        ]
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=messages,
            system="You are a helpful trading assistant. Be concise and practical."
        )
        
        return response.content[0].text
    
    def _get_market_data(self) -> Dict:
        """Get current market data"""
        # In production: fetch real data
        return {
            'sp500': 4500,
            'sp500_change': '+0.5%',
            'vix': 16.5
        }

# Example usage
chat_assistant = TradingChatAssistant(api_key="your-key")

# Set up user
chat_assistant.user_profile = {'level': 'intermediate'}
chat_assistant.portfolio_data = {
    'total_value': 125000,
    'holdings': [{'ticker': 'AAPL', 'shares': 100}]
}

# Chat examples
messages = [
    "What's my largest position?",
    "Why is the market up today?",
    "Buy 50 shares of Microsoft",
    "Tell me about Apple's performance"
]

for msg in messages:
    print(f"\\nUser: {msg}")
    response = chat_assistant.chat(msg)
    print(f"Assistant: {response}")
\`\`\`

---

## Best Practices

1. **Safety First**: Always require confirmation for trades
2. **Clear Language**: Avoid jargon, explain terms
3. **Context Awareness**: Remember conversation history
4. **Error Handling**: Handle ambiguous inputs gracefully
5. **Education**: Explain decisions and concepts
6. **Personalization**: Adapt to user sophistication level
7. **Transparency**: Be clear about limitations
8. **Accessibility**: Support multiple interaction modes
9. **Security**: Verify identity for sensitive operations
10. **Compliance**: Maintain audit trail of commands

---

## Summary

We covered:
- Natural language portfolio queries
- Voice-controlled trading with safety guardrails
- Conversational market analysis
- Alert explanations
- Building complete trading chat assistants
- Best practices for conversational interfaces

Next: LLM-powered backtesting and strategy development.
`,
};

