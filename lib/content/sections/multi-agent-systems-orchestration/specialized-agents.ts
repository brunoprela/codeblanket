/**
 * Specialized Agents Section
 * Module 7: Multi-Agent Systems & Orchestration
 */

export const specializedagentsSection = {
  id: 'specialized-agents',
  title: 'Specialized Agents',
  content: `# Specialized Agents

Master building agents with specific capabilities optimized for particular tasks.

## Overview: The Power of Specialization

Instead of one generalist agent, use specialized agents:

- **Better Performance**: Optimized prompts and tools for specific tasks
- **Clearer Responsibilities**: Each agent has one job
- **Easier Testing**: Test each capability independently
- **Composability**: Mix and match agents for different workflows

### Common Agent Types

**Researcher**: Gathers information  
**Planner**: Breaks down tasks  
**Coder**: Writes code  
**Reviewer**: Evaluates quality  
**Tester**: Validates functionality  
**Manager**: Coordinates others  

## Researcher Agent

Gathers information from various sources:

\`\`\`python
from typing import List, Dict, Any
import openai
from duckduckgo_search import DDGS

class ResearcherAgent:
    """Agent specialized in research and information gathering."""
    
    def __init__(self, model: str = "gpt-4"):
        self.name = "Researcher"
        self.model = model
        self.search_engine = DDGS()
    
    async def research (self, topic: str, depth: str = "normal") -> Dict[str, Any]:
        """Research a topic comprehensively."""
        if depth == "quick":
            return await self._quick_research (topic)
        elif depth == "deep":
            return await self._deep_research (topic)
        else:
            return await self._normal_research (topic)
    
    async def _quick_research (self, topic: str) -> Dict[str, Any]:
        """Quick research using LLM knowledge."""
        prompt = f"""Provide a concise research summary on: {topic}

Include:
- Key facts (5-7 points)
- Main concepts
- Current state

Be factual and cite year of information when relevant."""
        
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a research assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return {
            "topic": topic,
            "depth": "quick",
            "summary": response.choices[0].message.content,
            "sources": ["LLM knowledge base"]
        }
    
    async def _normal_research (self, topic: str) -> Dict[str, Any]:
        """Normal research with web search."""
        # 1. Web search
        search_results = self.search_engine.text (topic, max_results=5)
        
        # 2. Synthesize findings
        sources = []
        for result in search_results:
            sources.append({
                "title": result.get("title"),
                "url": result.get("href"),
                "snippet": result.get("body")
            })
        
        # 3. LLM synthesis
        sources_text = "\\n\\n".join([
            f"Source: {s['title']}\\n{s['snippet']}"
            for s in sources
        ])
        
        prompt = f"""Synthesize research on: {topic}

Web Sources:
{sources_text}

Create comprehensive research summary with:
1. Overview
2. Key Findings (bullet points)
3. Important Details
4. Current Trends"""
        
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a research synthesizer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return {
            "topic": topic,
            "depth": "normal",
            "summary": response.choices[0].message.content,
            "sources": sources
        }
    
    async def _deep_research (self, topic: str) -> Dict[str, Any]:
        """Deep research with multiple queries."""
        # Generate sub-queries
        sub_queries = await self._generate_research_questions (topic)
        
        # Research each sub-query
        sub_results = []
        for query in sub_queries:
            result = await self._normal_research (query)
            sub_results.append (result)
        
        # Synthesize all findings
        all_findings = "\\n\\n".join([
            f"Query: {r['topic']}\\n{r['summary']}"
            for r in sub_results
        ])
        
        prompt = f"""Create comprehensive research report on: {topic}

Research findings from multiple angles:
{all_findings}

Provide:
1. Executive Summary
2. Detailed Findings
3. Key Insights
4. Conclusions"""
        
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a senior researcher."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        all_sources = []
        for result in sub_results:
            all_sources.extend (result["sources"])
        
        return {
            "topic": topic,
            "depth": "deep",
            "summary": response.choices[0].message.content,
            "sub_queries": sub_queries,
            "sources": all_sources
        }
    
    async def _generate_research_questions (self, topic: str) -> List[str]:
        """Generate sub-questions for deep research."""
        prompt = f"""Generate 3-5 specific research questions for: {topic}

Questions should cover different angles (technical, historical, current state, future trends).

Format: One question per line."""
        
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a research planner."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        
        questions = response.choices[0].message.content.strip().split("\\n")
        return [q.strip() for q in questions if q.strip()]

# Usage
researcher = ResearcherAgent()

# Quick research
result = await researcher.research("quantum computing", depth="quick")
print(result["summary"])

# Deep research
result = await researcher.research("quantum computing", depth="deep")
print(result["summary"])
\`\`\`

## Planner Agent

Breaks complex tasks into steps:

\`\`\`python
from typing import List
from dataclasses import dataclass

@dataclass
class Step:
    """A step in a plan."""
    id: int
    description: str
    dependencies: List[int]
    estimated_time: str
    assigned_to: Optional[str] = None

class PlannerAgent:
    """Agent specialized in task decomposition and planning."""
    
    def __init__(self, model: str = "gpt-4"):
        self.name = "Planner"
        self.model = model
    
    async def create_plan(
        self,
        goal: str,
        available_agents: List[str]
    ) -> List[Step]:
        """Create execution plan for goal."""
        # 1. Generate steps
        steps_text = await self._generate_steps (goal)
        
        # 2. Parse into Step objects
        steps = self._parse_steps (steps_text)
        
        # 3. Assign to agents
        steps = await self._assign_steps (steps, available_agents)
        
        return steps
    
    async def _generate_steps (self, goal: str) -> str:
        """Generate step-by-step plan."""
        prompt = f"""Break down this goal into specific, actionable steps:

Goal: {goal}

Create a detailed plan with:
- Clear, specific steps
- Dependencies between steps
- Estimated time for each step

Format:
Step X: [Description]
Dependencies: [Step numbers this depends on, or "None"]
Time: [Estimate like "5 min", "1 hour"]"""
        
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert project planner."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    
    def _parse_steps (self, steps_text: str) -> List[Step]:
        """Parse plan text into Step objects."""
        steps = []
        current_step = None
        
        for line in steps_text.split("\\n"):
            line = line.strip()
            
            if line.startswith("Step"):
                # Extract step number and description
                parts = line.split(":", 1)
                step_num = len (steps) + 1
                description = parts[1].strip() if len (parts) > 1 else ""
                
                current_step = {
                    "id": step_num,
                    "description": description,
                    "dependencies": [],
                    "estimated_time": "30 min"
                }
            elif current_step and line.startswith("Dependencies:"):
                # Parse dependencies
                deps_text = line.split(":", 1)[1].strip()
                if deps_text.lower() != "none":
                    # Extract numbers
                    import re
                    deps = [int (d) for d in re.findall (r'\\d+', deps_text)]
                    current_step["dependencies"] = deps
            elif current_step and line.startswith("Time:"):
                # Parse time estimate
                time_text = line.split(":", 1)[1].strip()
                current_step["estimated_time"] = time_text
                
                # Finish this step
                steps.append(Step(**current_step))
                current_step = None
        
        # Add last step if exists
        if current_step:
            steps.append(Step(**current_step))
        
        return steps
    
    async def _assign_steps(
        self,
        steps: List[Step],
        available_agents: List[str]
    ) -> List[Step]:
        """Assign steps to appropriate agents."""
        for step in steps:
            # Use LLM to determine best agent
            prompt = f"""Which agent should handle this step?

Step: {step.description}
Available agents: {', '.join (available_agents)}

Reply with just the agent name."""
            
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You assign tasks to agents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            assigned = response.choices[0].message.content.strip()
            if assigned in available_agents:
                step.assigned_to = assigned
            else:
                step.assigned_to = available_agents[0]  # Default
        
        return steps
    
    def visualize_plan (self, steps: List[Step]) -> str:
        """Create text visualization of plan."""
        output = ["=== EXECUTION PLAN ===" ""]
        
        for step in steps:
            deps = ", ".join (str (d) for d in step.dependencies) if step.dependencies else "None"
            output.append (f"Step {step.id}: {step.description}")
            output.append (f"  Dependencies: {deps}")
            output.append (f"  Time: {step.estimated_time}")
            output.append (f"  Assigned to: {step.assigned_to or 'Unassigned'}")
            output.append("")
        
        return "\\n".join (output)

# Usage
planner = PlannerAgent()

plan = await planner.create_plan(
    goal="Build a web scraper that extracts product prices",
    available_agents=["researcher", "coder", "tester", "reviewer"]
)

print(planner.visualize_plan (plan))
\`\`\`

## Coder Agent

Generates code:

\`\`\`python
class CoderAgent:
    """Agent specialized in writing code."""
    
    def __init__(self, model: str = "gpt-4"):
        self.name = "Coder"
        self.model = model
    
    async def write_code(
        self,
        task: str,
        language: str = "python",
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Write code for task."""
        prompt = self._build_coding_prompt (task, language, context)
        
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[
                {"role": "system", "content": f"You are an expert {language} programmer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        code = response.choices[0].message.content
        
        # Extract just code if wrapped in markdown
        if "\`\`\`" in code:
            code = self._extract_code_from_markdown (code)
        
        return {
        "task": task,
        "language": language,
        "code": code,
        "explanation": self._extract_explanation (response.choices[0].message.content)
    }
    
    def _build_coding_prompt(
        self,
        task: str,
        language: str,
        context: Optional[str]
    ) -> str:
"""Build effective coding prompt."""
prompt = f"""Write {language} code for this task:

Task: { task } """

if context:
    prompt += f"""

Context:
{ context } """

prompt += """

Requirements:
- Write clean, production - ready code
    - Add type hints (if language supports)
- Include docstrings / comments
    - Add error handling
        - Follow best practices"""

return prompt
    
    def _extract_code_from_markdown (self, text: str) -> str:
        """Extract code from markdown blocks."""
        import re
        # Find content between backtick markers
        pattern = r'\`\`\`(?:\\w+)?\\n(.*?)\`\`\`'
        matches = re.findall (pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        return text
    
    def _extract_explanation (self, text: str) -> str:
        """Extract explanation text."""
        # Text before first code block
        if "\`\`\`" in text:
return text.split("\`\`\`")[0].strip()
return ""
    
    async def refactor_code(
    self,
    code: str,
    improvement: str
) -> str:
"""Refactor existing code."""
prompt = f"""Refactor this code to: {improvement}

Current code:
\`\`\`
{code}
\`\`\`

Provide improved version."""

response = await openai.ChatCompletion.acreate(
    model = self.model,
    messages = [
        { "role": "system", "content": "You are a code refactoring expert." },
        { "role": "user", "content": prompt }
    ],
    temperature = 0.2
)

return self._extract_code_from_markdown (response.choices[0].message.content)

# Usage
coder = CoderAgent()

result = await coder.write_code(
    task = "Create a function that validates email addresses",
    language = "python",
    context = "Use regex for validation. Should return True/False."
)

print("Generated Code:")
print(result["code"])
\`\`\`

## Reviewer Agent

Evaluates quality and suggests improvements:

\`\`\`python
class ReviewerAgent:
    """Agent specialized in code/content review."""
    
    def __init__(self, model: str = "gpt-4"):
        self.name = "Reviewer"
        self.model = model
    
    async def review_code (self, code: str) -> Dict[str, Any]:
        """Review code comprehensively."""
        prompt = f"""Review this code thoroughly:

\`\`\`
{ code }
\`\`\`

Evaluate:
1. Correctness: Does it work?
2. Quality: Is it well-written?
3. Security: Any vulnerabilities?
4. Performance: Any inefficiencies?
5. Style: Follows best practices?

Provide:
- Overall assessment (Excellent/Good/Needs Work/Poor)
- Specific issues found
- Suggested improvements"""
        
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a senior code reviewer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        review_text = response.choices[0].message.content
        
        return {
            "code": code,
            "assessment": self._extract_assessment (review_text),
            "issues": self._extract_issues (review_text),
            "suggestions": self._extract_suggestions (review_text),
            "full_review": review_text
        }
    
    async def review_content (self, content: str, type: str = "article") -> Dict[str, Any]:
        """Review written content."""
        prompt = f"""Review this {type}:

{content}

Evaluate:
1. Clarity: Easy to understand?
2. Accuracy: Factually correct?
3. Completeness: Covers topic well?
4. Style: Well-written?

Provide specific feedback and suggestions."""
        
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[
                {"role": "system", "content": f"You are a professional {type} reviewer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return {
            "content": content,
            "review": response.choices[0].message.content
        }
    
    def _extract_assessment (self, review: str) -> str:
        """Extract overall assessment."""
        for line in review.split("\\n"):
            if "assessment:" in line.lower():
                return line.split(":", 1)[1].strip()
        return "Unknown"
    
    def _extract_issues (self, review: str) -> List[str]:
        """Extract list of issues."""
        issues = []
        in_issues = False
        
        for line in review.split("\\n"):
            if "issues" in line.lower():
                in_issues = True
            elif in_issues and line.strip().startswith("-"):
                issues.append (line.strip()[1:].strip())
            elif in_issues and not line.strip():
                break
        
        return issues
    
    def _extract_suggestions (self, review: str) -> List[str]:
        """Extract suggestions."""
        suggestions = []
        in_suggestions = False
        
        for line in review.split("\\n"):
            if "suggestion" in line.lower():
                in_suggestions = True
            elif in_suggestions and line.strip().startswith("-"):
                suggestions.append (line.strip()[1:].strip())
        
        return suggestions

# Usage
reviewer = ReviewerAgent()

code_review = await reviewer.review_code("""
def calculate (x, y):
    return x / y
""")

print("Assessment:", code_review["assessment"])
print("Issues:", code_review["issues"])
print("Suggestions:", code_review["suggestions"])
\`\`\`

## Tester Agent

Creates and runs tests:

\`\`\`python
class TesterAgent:
    """Agent specialized in testing."""
    
    def __init__(self, model: str = "gpt-4"):
        self.name = "Tester"
        self.model = model
    
    async def generate_tests(
        self,
        code: str,
        test_framework: str = "pytest"
    ) -> str:
        """Generate test cases for code."""
        prompt = f"""Generate comprehensive tests for this code using {test_framework}:

\`\`\`
{ code }
\`\`\`

Include:
- Normal cases
- Edge cases
- Error cases
- Mock external dependencies if needed"""
        
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a testing expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        test_code = response.choices[0].message.content
        
        # Extract code from markdown
        if "\`\`\`" in test_code:
import re
            pattern = r'\`\`\`(?:\\w+)?\\n(.*?)\`\`\`'
matches = re.findall (pattern, test_code, re.DOTALL)
if matches:
    test_code = matches[0].strip()

return test_code
    
    async def run_tests(
    self,
    code: str,
    tests: str
) -> Dict[str, Any]:
"""Run tests and return results."""
        # In production, execute in sandbox
        # For now, use LLM to analyze
prompt = f"""Analyze if these tests would pass for this code:

Code:
\`\`\`
{code}
\`\`\`

Tests:
\`\`\`
{tests}
\`\`\`

Determine:
- Which tests would pass
    - Which would fail and why
        - Coverage assessment"""

response = await openai.ChatCompletion.acreate(
    model = self.model,
    messages = [
        { "role": "system", "content": "You are a test analysis expert." },
        { "role": "user", "content": prompt }
    ],
    temperature = 0.2
)

return {
    "analysis": response.choices[0].message.content,
    "tests": tests
}

# Usage
tester = TesterAgent()

tests = await tester.generate_tests("""
def add (a, b):
    return a + b
""")

print("Generated Tests:")
print(tests)
\`\`\`

## Manager Agent

Coordinates other agents:

\`\`\`python
class ManagerAgent:
    """Agent that coordinates other agents."""
    
    def __init__(self, workers: Dict[str, Any]):
        self.name = "Manager"
        self.workers = workers
    
    async def delegate_task(
        self,
        task: str
    ) -> Dict[str, Any]:
        """Delegate task to appropriate agent."""
        # Determine which agent to use
        agent_name = await self._choose_agent (task)
        agent = self.workers.get (agent_name)
        
        if not agent:
            raise ValueError (f"No agent named {agent_name}")
        
        # Execute with that agent
        result = await agent.execute (task)
        
        return {
            "task": task,
            "agent": agent_name,
            "result": result
        }
    
    async def _choose_agent (self, task: str) -> str:
        """Choose appropriate agent for task."""
        # Use heuristics or LLM
        if "research" in task.lower():
            return "researcher"
        elif "code" in task.lower() or "implement" in task.lower():
            return "coder"
        elif "test" in task.lower():
            return "tester"
        elif "review" in task.lower():
            return "reviewer"
        else:
            return "researcher"  # Default

# Usage
manager = ManagerAgent({
    "researcher": ResearcherAgent(),
    "coder": CoderAgent(),
    "tester": TesterAgent(),
    "reviewer": ReviewerAgent()
})

result = await manager.delegate_task("Research quantum computing")
print(result)
\`\`\`

## Building Custom Specialized Agents

Template for creating your own:

\`\`\`python
class CustomAgent:
    """Template for specialized agent."""
    
    def __init__(self, model: str = "gpt-4"):
        self.name = "Custom"
        self.model = model
        self.system_prompt = self._build_system_prompt()
    
    def _build_system_prompt (self) -> str:
        """Define agent's expertise and behavior."""
        return """You are a [specialist type].

Your capabilities:
- [Capability 1]
- [Capability 2]

Your approach:
- [How you work]

Your output format:
- [How you format responses]"""
    
    async def execute (self, task: str) -> Any:
        """Execute specialized task."""
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": task}
            ],
            temperature=0.3
        )
        
        return self._parse_response (response.choices[0].message.content)
    
    def _parse_response (self, response: str) -> Any:
        """Parse and structure response."""
        # Custom parsing logic
        return response
\`\`\`

## Best Practices

1. **Clear System Prompts**: Define expertise clearly
2. **Consistent Interfaces**: All agents use similar execute() pattern
3. **Specialized Tools**: Give agents domain-specific tools
4. **Error Handling**: Each agent handles its own errors
5. **Observable**: Track what each agent does

## Next Steps

You now have specialized agents. Next, learn:
- Task decomposition and planning
- Coordinating agents
- Building workflows
`,
};
