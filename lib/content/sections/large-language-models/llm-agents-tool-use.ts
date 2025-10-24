export const llmAgentsToolUse = {
  title: 'LLM Agents & Tool Use',
  id: 'llm-agents-tool-use',
  content: `
# LLM Agents & Tool Use

## Introduction

LLM agents extend language models with the ability to use tools, take actions, and autonomously work toward goals. Unlike simple chat, agents can browse the web, execute code, query databases, and orchestrate complex workflows. Combined with tool use (function calling), agents become powerful systems that can interact with the real world.

### Why Agents

**Autonomy**: Work toward goals without constant guidance
**Tool Access**: Use APIs, databases, calculators, web search
**Complex Tasks**: Break down and solve multi-step problems
**Adaptability**: Adjust strategy based on results
**Scalability**: Automate workflows end-to-end

---

## Function Calling Basics

### Tool Definitions

\`\`\`python
"""
Function calling with OpenAI and Anthropic
"""

import openai
import json

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol (e.g., AAPL, GOOGL)"
                    }
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# Implement tool functions
def get_stock_price(ticker):
    """Actual implementation"""
    # In reality, call an API
    prices = {"AAPL": 175.50, "GOOGL": 140.25}
    return prices.get(ticker, "Unknown")

def calculate(expression):
    """Safe math evaluation"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except:
        return "Invalid expression"

# Tool registry
tool_functions = {
    "get_stock_price": get_stock_price,
    "calculate": calculate
}

# Use tools with LLM
def chat_with_tools(user_message):
    """
    Complete function calling flow
    """
    client = openai.OpenAI()
    
    messages = [
        {"role": "user", "content": user_message}
    ]
    
    # Initial request
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    # Check if model wants to call a function
    while response.choices[0].message.tool_calls:
        # Add assistant response to messages
        messages.append(response.choices[0].message)
        
        # Execute each tool call
        for tool_call in response.choices[0].message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"Calling: {function_name}({function_args})")
            
            # Execute
            function_response = tool_functions[function_name](**function_args)
            
            # Add result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": str(function_response)
            })
        
        # Get final response
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            tools=tools
        )
    
    return response.choices[0].message.content

# Example usage
result = chat_with_tools("What's the stock price of AAPL and calculate 150 * 1.2")

# Model will:
# 1. Call get_stock_price("AAPL") → 175.50
# 2. Call calculate("150 * 1.2") → 180
# 3. Respond: "Apple's stock price is $175.50, and 150 * 1.2 equals 180."
\`\`\`

---

## ReAct Pattern

### Reasoning + Acting

\`\`\`python
"""
ReAct: Reason about what to do, then act
"""

class ReActAgent:
    """
    ReAct agent implementation
    
    Loop:
    1. Thought: Reason about current situation
    2. Action: Decide what tool to use
    3. Observation: See result
    4. Repeat until done
    """
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.max_iterations = 10
    
    def create_prompt(self, question, history):
        """
        Format ReAct prompt
        """
        tool_descriptions = "\\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in self.tools
        ])
        
        prompt = f"""Answer the following question by reasoning and using tools.

Available tools:
{tool_descriptions}

Question: {question}

You can use tools by responding with:
Action: [tool_name]
Action Input: [input]

After seeing the observation, continue reasoning.

{history}

Let's begin!

Thought:"""
        
        return prompt
    
    def parse_action(self, text):
        """
        Extract action and input from model response
        """
        import re
        
        action_match = re.search(r'Action: (.+)', text)
        input_match = re.search(r'Action Input: (.+)', text)
        
        if action_match and input_match:
            return {
                'action': action_match.group(1).strip(),
                'input': input_match.group(1).strip()
            }
        
        return None
    
    def run(self, question):
        """
        Execute ReAct loop
        """
        history = ""
        
        for i in range(self.max_iterations):
            # Generate thought + action
            prompt = self.create_prompt(question, history)
            response = self.llm.generate(prompt)
            
            print(f"\\n--- Iteration {i+1} ---")
            print(response)
            
            # Check if done
            if "Final Answer:" in response:
                final_answer = response.split("Final Answer:")[1].strip()
                return final_answer
            
            # Parse action
            action_info = self.parse_action(response)
            
            if action_info:
                # Execute tool
                tool_name = action_info['action']
                tool_input = action_info['input']
                
                observation = self.execute_tool(tool_name, tool_input)
                
                # Add to history
                history += f"{response}\\nObservation: {observation}\\n\\n"
            else:
                history += f"{response}\\n"
        
        return "Max iterations reached without final answer"
    
    def execute_tool(self, tool_name, tool_input):
        """
        Execute a tool and return result
        """
        for tool in self.tools:
            if tool['name'] == tool_name:
                return tool['function'](tool_input)
        
        return f"Unknown tool: {tool_name}"

# Example: Multi-step reasoning
tools = [
    {
        'name': 'search',
        'description': 'Search the web',
        'function': lambda q: f"Search results for '{q}': [mock results]"
    },
    {
        'name': 'calculator',
        'description': 'Perform calculations',
        'function': lambda expr: str(eval(expr))
    }
]

agent = ReActAgent(llm, tools)

question = """What is the population of France multiplied by 2?"""

answer = agent.run(question)

# Expected flow:
# Thought: I need to find France's population
# Action: search
# Action Input: population of France
# Observation: ~67 million
# 
# Thought: Now I need to multiply by 2
# Action: calculator
# Action Input: 67000000 * 2
# Observation: 134000000
#
# Thought: I have the answer
# Final Answer: 134 million
\`\`\`

---

## Agent Frameworks

### LangChain Agents

\`\`\`python
"""
LangChain for agent orchestration
"""

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define tools
def search_web(query: str) -> str:
    """Search the web"""
    return f"Results for {query}: [mock results]"

def get_weather(location: str) -> str:
    """Get weather"""
    return f"Weather in {location}: Sunny, 72°F"

tools = [
    Tool(
        name="search",
        description="Search the web for current information",
        func=search_web
    ),
    Tool(
        name="weather",
        description="Get current weather for a location",
        func=get_weather
    )
]

# Create agent
llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with access to tools."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run
result = agent_executor.invoke({
    "input": "What's the weather in Paris and what's it famous for?"
})

print(result["output"])

# Agent will:
# 1. Call weather("Paris")
# 2. Call search("Paris famous for")
# 3. Synthesize answer
\`\`\`

### AutoGPT-Style Agents

\`\`\`python
"""
Autonomous goal-oriented agents
"""

class AutonomousAgent:
    """
    Agent that works toward long-term goals
    
    Features:
    - Goal decomposition
    - Self-critique
    - Memory/context management
    - Plan adjustment
    """
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.memory = []
        self.max_iterations = 20
    
    def run(self, goal):
        """
        Work autonomously toward goal
        """
        # Decompose goal into tasks
        tasks = self.decompose_goal(goal)
        
        completed_tasks = []
        
        for task in tasks:
            print(f"\\nTask: {task}")
            
            # Execute task
            result = self.execute_task(task)
            completed_tasks.append({
                'task': task,
                'result': result
            })
            
            # Self-critique
            if not self.is_satisfactory(task, result):
                print("Result not satisfactory, retrying...")
                result = self.execute_task(task)  # Retry
            
            # Update memory
            self.memory.append({
                'task': task,
                'result': result
            })
            
            # Check if goal achieved
            if self.is_goal_achieved(goal, completed_tasks):
                print("\\nGoal achieved!")
                return self.summarize_results(completed_tasks)
        
        return "Completed all tasks"
    
    def decompose_goal(self, goal):
        """
        Break goal into subtasks
        """
        prompt = f"""Break this goal into specific actionable tasks:

Goal: {goal}

Tasks:
1."""
        
        response = self.llm.generate(prompt)
        tasks = response.strip().split('\\n')
        return [t.strip() for t in tasks if t.strip()]
    
    def execute_task(self, task):
        """
        Execute single task using available tools
        """
        # Decide which tool to use
        prompt = f"""Task: {task}

Available tools: {', '.join([t['name'] for t in self.tools])}

Which tool should be used? Respond with tool name and input.

Tool:"""
        
        response = self.llm.generate(prompt)
        
        # Execute tool (simplified)
        for tool in self.tools:
            if tool['name'].lower() in response.lower():
                return tool['function'](task)
        
        return "Task completed without tools"
    
    def is_satisfactory(self, task, result):
        """
        Self-critique: Is result good enough?
        """
        prompt = f"""Task: {task}
Result: {result}

Is this result satisfactory? Answer yes or no.

Evaluation:"""
        
        evaluation = self.llm.generate(prompt)
        return "yes" in evaluation.lower()
    
    def is_goal_achieved(self, goal, completed_tasks):
        """
        Check if original goal is achieved
        """
        summary = "\\n".join([
            f"- {t['task']}: {t['result']}"
            for t in completed_tasks
        ])
        
        prompt = f"""Goal: {goal}

Completed tasks:
{summary}

Has the goal been fully achieved? Answer yes or no.

Assessment:"""
        
        assessment = self.llm.generate(prompt)
        return "yes" in assessment.lower()
    
    def summarize_results(self, completed_tasks):
        """
        Create final summary
        """
        summary = "\\n".join([
            f"{i+1}. {t['task']}: {t['result']}"
            for i, t in enumerate(completed_tasks)
        ])
        
        return f"Completed tasks:\\n{summary}"

# Example
tools = [
    {'name': 'search', 'function': lambda x: f"Found info about {x}"},
    {'name': 'write', 'function': lambda x: f"Wrote: {x}"},
    {'name': 'analyze', 'function': lambda x: f"Analysis: {x}"}
]

agent = AutonomousAgent(llm, tools)

goal = "Research Python web frameworks and create a comparison chart"
result = agent.run(goal)
\`\`\`

---

## Memory Systems

### Agent Memory

\`\`\`python
"""
Memory for agents
"""

class AgentMemory:
    """
    Multi-tier memory for agents
    
    Types:
    - Short-term: Current conversation
    - Working: Recent task context
    - Long-term: Persistent knowledge
    - Episodic: Past experiences
    """
    
    def __init__(self, vector_db):
        self.short_term = []  # Last N messages
        self.working_memory = {}  # Current task context
        self.long_term = vector_db  # Persistent storage
        self.episodic = []  # Past episodes
    
    def add_to_short_term(self, message):
        """
        Add to short-term memory (limited size)
        """
        self.short_term.append(message)
        
        # Keep last 10 messages
        if len(self.short_term) > 10:
            # Move old to long-term
            old = self.short_term.pop(0)
            self.add_to_long_term(old)
    
    def add_to_working_memory(self, key, value):
        """
        Store current task context
        """
        self.working_memory[key] = value
    
    def add_to_long_term(self, memory):
        """
        Store in vector database for retrieval
        """
        self.long_term.add(
            documents=[memory['content']],
            metadatas=[{
                'timestamp': memory['timestamp'],
                'type': memory['type']
            }]
        )
    
    def add_episode(self, episode):
        """
        Store episodic memory (past experiences)
        """
        self.episodic.append({
            'goal': episode['goal'],
            'actions': episode['actions'],
            'outcome': episode['outcome'],
            'timestamp': episode['timestamp']
        })
    
    def retrieve_relevant_memories(self, query, k=5):
        """
        Retrieve relevant memories for current context
        """
        # Search long-term memory
        results = self.long_term.query(
            query_texts=[query],
            n_results=k
        )
        
        return results['documents'][0]
    
    def get_context(self, current_goal):
        """
        Build context from all memory types
        """
        context = {
            'short_term': self.short_term[-5:],  # Last 5 messages
            'working': self.working_memory,
            'relevant_long_term': self.retrieve_relevant_memories(current_goal),
            'similar_episodes': self.find_similar_episodes(current_goal)
        }
        
        return context
    
    def find_similar_episodes(self, goal, k=3):
        """
        Find similar past experiences
        """
        # Embed goal
        goal_emb = embedder.encode(goal)
        
        # Compare with past episodes
        similarities = []
        for episode in self.episodic:
            ep_emb = embedder.encode(episode['goal'])
            sim = cosine_similarity([goal_emb], [ep_emb])[0][0]
            similarities.append((sim, episode))
        
        # Return top-k
        similarities.sort(reverse=True)
        return [ep for _, ep in similarities[:k]]

# Example usage
from chromadb import Client

vector_db = Client().create_collection("agent_memory")
memory = AgentMemory(vector_db)

# Agent operates
memory.add_to_short_term({
    'role': 'user',
    'content': 'Find information about Python',
    'timestamp': '2024-01-01'
})

memory.add_to_working_memory('current_task', 'research Python')

# Later, retrieve context
context = memory.get_context('research Python frameworks')
print(context)
\`\`\`

---

## Multi-Agent Systems

### Agent Collaboration

\`\`\`python
"""
Multiple agents working together
"""

class MultiAgentSystem:
    """
    Coordinate multiple specialized agents
    
    Patterns:
    - Hierarchical: Manager delegates to workers
    - Sequential: Chain of agents
    - Parallel: Agents work simultaneously
    - Debate: Agents critique each other
    """
    
    def __init__(self):
        self.agents = {}
    
    def register_agent(self, name, agent):
        """
        Register specialized agent
        """
        self.agents[name] = agent
    
    def hierarchical_execution(self, task):
        """
        Manager agent delegates to specialists
        """
        # Manager breaks down task
        manager = self.agents['manager']
        subtasks = manager.decompose(task)
        
        results = []
        for subtask in subtasks:
            # Assign to appropriate specialist
            specialist = manager.select_specialist(subtask)
            result = self.agents[specialist].execute(subtask)
            results.append(result)
        
        # Manager synthesizes
        final_result = manager.synthesize(results)
        return final_result
    
    def sequential_execution(self, task, agent_chain):
        """
        Pass through chain of agents
        """
        current_output = task
        
        for agent_name in agent_chain:
            agent = self.agents[agent_name]
            current_output = agent.process(current_output)
        
        return current_output
    
    def debate_execution(self, question, num_rounds=3):
        """
        Agents debate to reach best answer
        """
        # Initial answers from each agent
        answers = {}
        for name, agent in self.agents.items():
            answers[name] = agent.answer(question)
        
        # Debate rounds
        for round in range(num_rounds):
            critiques = {}
            
            # Each agent critiques others
            for name, agent in self.agents.items():
                other_answers = {
                    k: v for k, v in answers.items() if k != name
                }
                critique = agent.critique(other_answers)
                critiques[name] = critique
            
            # Update answers based on critiques
            for name, agent in self.agents.items():
                answers[name] = agent.revise(
                    answers[name],
                    critiques
                )
        
        # Final consensus
        consensus = self.reach_consensus(answers)
        return consensus
    
    def reach_consensus(self, answers):
        """
        Combine agent answers into consensus
        """
        # Vote or synthesize
        summary = "Agent answers:\\n"
        for name, answer in answers.items():
            summary += f"- {name}: {answer}\\n"
        
        # Use manager to synthesize
        manager = self.agents.get('manager')
        if manager:
            consensus = manager.synthesize_consensus(answers)
        else:
            consensus = summary
        
        return consensus

# Example: Research team
class ResearcherAgent:
    """Finds information"""
    def execute(self, task):
        return f"Research: {task}"

class WriterAgent:
    """Writes content"""
    def execute(self, task):
        return f"Written: {task}"

class EditorAgent:
    """Reviews and improves"""
    def execute(self, task):
        return f"Edited: {task}"

# Setup
system = MultiAgentSystem()
system.register_agent('researcher', ResearcherAgent())
system.register_agent('writer', WriterAgent())
system.register_agent('editor', EditorAgent())

# Sequential execution
result = system.sequential_execution(
    "Write article about AI",
    ['researcher', 'writer', 'editor']
)

print(result)
# Output: Research → Write → Edit pipeline
\`\`\`

---

## Production Considerations

### Safety and Monitoring

\`\`\`python
"""
Production agent systems
"""

class ProductionAgent:
    """
    Production-ready agent with safety
    """
    
    def __init__(self, config):
        self.config = config
        self.action_log = []
        self.safety_checks = []
    
    def execute_with_safety(self, action):
        """
        Execute action with safety checks
        """
        # Pre-execution checks
        if not self.is_safe_action(action):
            self.log_rejection(action, "Safety check failed")
            return {"status": "rejected", "reason": "unsafe"}
        
        # Spending limit check
        if not self.check_budget(action):
            self.log_rejection(action, "Budget exceeded")
            return {"status": "rejected", "reason": "budget"}
        
        # Execute
        try:
            result = self.execute_action(action)
            self.log_action(action, result, "success")
            return result
        except Exception as e:
            self.log_action(action, str(e), "error")
            return {"status": "error", "message": str(e)}
    
    def is_safe_action(self, action):
        """
        Safety checks before execution
        """
        unsafe_patterns = [
            "delete_database",
            "drop_table",
            "rm -rf",
            "system("
        ]
        
        action_str = str(action).lower()
        return not any(pattern in action_str for pattern in unsafe_patterns)
    
    def check_budget(self, action):
        """
        Ensure action within budget
        """
        # Estimate cost
        cost = self.estimate_action_cost(action)
        
        # Check against daily budget
        daily_spent = sum(
            a['cost'] for a in self.action_log
            if a['date'] == today
        )
        
        return daily_spent + cost < self.config['daily_budget']
    
    def log_action(self, action, result, status):
        """
        Log all actions for auditing
        """
        self.action_log.append({
            'timestamp': datetime.now(),
            'action': action,
            'result': result,
            'status': status,
            'cost': self.estimate_action_cost(action)
        })
    
    def get_audit_trail(self):
        """
        Full audit trail of actions
        """
        return self.action_log

# Monitoring
class AgentMonitor:
    """
    Monitor agent performance
    """
    
    def track_metrics(self, agent):
        """
        Track key metrics
        """
        metrics = {
            'total_actions': len(agent.action_log),
            'success_rate': self.calculate_success_rate(agent),
            'avg_cost_per_action': self.calculate_avg_cost(agent),
            'safety_incidents': len([
                a for a in agent.action_log
                if a['status'] == 'rejected'
            ])
        }
        
        return metrics
    
    def alert_on_anomalies(self, metrics):
        """
        Alert if metrics anomalous
        """
        if metrics['success_rate'] < 0.8:
            self.send_alert("Low success rate")
        
        if metrics['avg_cost_per_action'] > threshold:
            self.send_alert("High costs")
\`\`\`

---

## Conclusion

LLM agents enable:

1. **Tool Use**: Access calculators, APIs, databases, web search
2. **Autonomy**: Work toward goals without constant guidance
3. **Complex Tasks**: Break down and solve multi-step problems
4. **Collaboration**: Multiple agents working together

**Key Patterns**:
- **Function Calling**: Structured tool invocation
- **ReAct**: Reasoning before acting
- **Memory**: Short-term, working, long-term, episodic
- **Multi-Agent**: Hierarchical, sequential, debate

**Frameworks**:
- LangChain: Comprehensive agent toolkit
- AutoGPT: Autonomous goal-oriented
- Custom: Build from scratch for control

**Production Best Practices**:
- Safety checks before execution
- Budget limits
- Audit logging
- Human-in-the-loop for critical actions
- Monitoring and alerting

**Costs** (per day):
- Simple agent (100 actions): $5-10
- Complex agent (1000 actions): $50-100
- Enterprise (10k actions): $500-1000

Agents are the future of AI—they transform LLMs from chatbots into autonomous systems that can actually get work done.
`,
};
