export const llmCostOptimization = {
  title: 'LLM Cost Optimization',
  id: 'llm-cost-optimization',
  content: `
# LLM Cost Optimization

## Understanding Costs

**Pricing Model**: Per token (input + output)
**Example**: Claude Sonnet: $3/M input, $15/M output

\`\`\`python
"""Cost calculation"""
def calculate_cost (input_tokens, output_tokens, model="claude-sonnet"):
    pricing = {
        "claude-sonnet": {"in": 3, "out": 15},
        "gpt-4-turbo": {"in": 10, "out": 30},
        "gpt-3.5": {"in": 0.5, "out": 1.5}
    }
    
    rates = pricing[model]
    cost = (input_tokens * rates["in"] + output_tokens * rates["out"]) / 1_000_000
    return cost

# Example: 100k calls/day
cost_per_day = calculate_cost(1000, 200, "claude-sonnet") * 100000
print(f"Daily cost: \${cost_per_day:.2f}")  # $600/day = $18k/month
\`\`\`

## Optimization Strategies

### 1. Prompt Optimization

**Shorter Prompts**: Remove unnecessary text
**Caching**: Reuse common prefixes (Claude prompt caching)
**Batch Requests**: Process multiple items together

### 2. Model Selection

**Task-Appropriate**: Use GPT-3.5 for simple tasks, GPT-4 for complex
**Cascading**: Try cheap model first, escalate if needed

\`\`\`python
"""Model cascading"""
def cascading_generation (prompt, difficulty):
    if difficulty == "easy":
        return gpt35.generate (prompt)  # $0.002/call
    elif difficulty == "medium":
        return claude_haiku.generate (prompt)  # $0.01/call
    else:
        return gpt4.generate (prompt)  # $0.05/call
\`\`\`

### 3. Caching

**Response Caching**: Store results for common queries
**Semantic Caching**: Cache similar queries
**Prompt Caching**: Claude's built-in prefix caching

### 4. Quantization

For self-hosted models:
- 4-bit: 75% memory reduction, minimal quality loss
- 8-bit: 50% reduction, negligible quality loss

## Cost Monitoring

\`\`\`python
"""Track costs"""
class CostMonitor:
    def __init__(self):
        self.daily_spend = 0
        self.daily_limit = 1000  # $1000/day
    
    def track_call (self, input_tokens, output_tokens):
        cost = calculate_cost (input_tokens, output_tokens)
        self.daily_spend += cost
        
        if self.daily_spend > self.daily_limit:
            raise Exception("Daily budget exceeded")
        
        return cost
\`\`\`

## Key Insights

- Prompt length matters more than you think
- Caching can reduce costs 70-90%
- Use appropriate model for task
- Monitor and set budgets
- Self-hosting viable at scale (>10M calls/month)
`,
};
