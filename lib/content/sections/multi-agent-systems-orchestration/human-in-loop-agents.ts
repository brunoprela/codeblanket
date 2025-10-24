/**
 * Human-in-the-Loop Agents Section
 * Module 7: Multi-Agent Systems & Orchestration
 */

export const humaninloopagentsSection = {
  id: 'human-in-loop-agents',
  title: 'Human-in-the-Loop Agents',
  content: `# Human-in-the-Loop Agents

Master integrating human oversight, approval, and feedback into agent systems.

## Overview: Why Human-in-the-Loop?

Agents shouldn't operate fully autonomously:

- **Critical Decisions**: Humans approve important actions
- **Quality Control**: Humans verify outputs
- **Edge Cases**: Humans handle exceptions
- **Learning**: Human feedback improves agents
- **Safety**: Humans provide oversight
- **Trust**: Users want control

### HITL Patterns

**Approval Gates**: Human approves before proceeding  
**Review & Revise**: Human provides feedback  
**Exception Handling**: Human handles edge cases  
**Progressive Automation**: Gradually reduce human involvement  
**Confidence Thresholds**: Auto-proceed if confident  

## Approval Gates

\`\`\`python
from dataclasses import dataclass
from typing import Optional, Callable, Any
from enum import Enum
import asyncio

class ApprovalStatus(Enum):
    """Status of approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"

@dataclass
class ApprovalRequest:
    """Request for human approval."""
    id: str
    agent_name: str
    action: str
    proposed_output: Any
    context: Dict[str, Any]
    status: ApprovalStatus = ApprovalStatus.PENDING
    feedback: Optional[str] = None
    timestamp: float = 0

class ApprovalGate:
    """Human approval gate for agent actions."""
    
    def __init__(self):
        self.pending_requests: Dict[str, ApprovalRequest] = {}
        self.approval_callbacks: Dict[str, asyncio.Future] = {}
    
    async def request_approval(
        self,
        agent_name: str,
        action: str,
        proposed_output: Any,
        context: Optional[Dict] = None,
        timeout: float = 300.0  # 5 minutes default
    ) -> tuple[ApprovalStatus, Optional[str]]:
        """Request human approval."""
        import time
        import uuid
        
        # Create approval request
        request_id = str(uuid.uuid4())
        request = ApprovalRequest(
            id=request_id,
            agent_name=agent_name,
            action=action,
            proposed_output=proposed_output,
            context=context or {},
            timestamp=time.time()
        )
        
        self.pending_requests[request_id] = request
        
        # Create future for response
        future = asyncio.Future()
        self.approval_callbacks[request_id] = future
        
        # Show to human
        self._display_request(request)
        
        # Wait for response with timeout
        try:
            status, feedback = await asyncio.wait_for(future, timeout=timeout)
            request.status = status
            request.feedback = feedback
            return status, feedback
        except asyncio.TimeoutError:
            request.status = ApprovalStatus.TIMEOUT
            return ApprovalStatus.TIMEOUT, "Approval timed out"
        finally:
            # Cleanup
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
            if request_id in self.approval_callbacks:
                del self.approval_callbacks[request_id]
    
    def provide_approval(
        self,
        request_id: str,
        approved: bool,
        feedback: Optional[str] = None
    ):
        """Human provides approval."""
        if request_id not in self.approval_callbacks:
            raise ValueError(f"Request {request_id} not found")
        
        future = self.approval_callbacks[request_id]
        status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        
        future.set_result((status, feedback))
    
    def _display_request(self, request: ApprovalRequest):
        """Display request to human."""
        print("\\n" + "="*60)
        print("APPROVAL REQUIRED")
        print("="*60)
        print(f"Request ID: {request.id}")
        print(f"Agent: {request.agent_name}")
        print(f"Action: {request.action}")
        print(f"\\nProposed Output:")
        print(request.proposed_output)
        print(f"\\nContext:")
        for key, value in request.context.items():
            print(f"  {key}: {value}")
        print("="*60)
        print("Waiting for approval...")

# Usage
gate = ApprovalGate()

class ApprovalAgent:
    """Agent that requests approval for actions."""
    
    def __init__(self, name: str, gate: ApprovalGate):
        self.name = name
        self.gate = gate
    
    async def execute_with_approval(self, task: str) -> Any:
        """Execute task with human approval."""
        # Generate output
        output = f"Proposed solution for: {task}"
        
        # Request approval
        print(f"[{self.name}] Generated output, requesting approval...")
        status, feedback = await self.gate.request_approval(
            agent_name=self.name,
            action="execute_task",
            proposed_output=output,
            context={"task": task},
            timeout=60.0
        )
        
        if status == ApprovalStatus.APPROVED:
            print(f"[{self.name}] ✅ Approved! Proceeding...")
            return output
        elif status == ApprovalStatus.REJECTED:
            print(f"[{self.name}] ❌ Rejected: {feedback}")
            # Could revise based on feedback
            return None
        else:
            print(f"[{self.name}] ⏱️ Timeout")
            return None

# Simulate human approval in separate task
async def human_approver():
    """Simulate human providing approval."""
    await asyncio.sleep(2)  # Human thinks
    
    # Get pending request
    if gate.pending_requests:
        request_id = list(gate.pending_requests.keys())[0]
        gate.provide_approval(request_id, approved=True, feedback="Looks good!")

# Execute both
agent = ApprovalAgent("WorkerAgent", gate)

await asyncio.gather(
    agent.execute_with_approval("Important task"),
    human_approver()
)
\`\`\`

## Confidence-Based Approval

\`\`\`python
class ConfidenceGate:
    """Approval gate based on confidence thresholds."""
    
    def __init__(
        self,
        auto_approve_threshold: float = 0.9,
        auto_reject_threshold: float = 0.3
    ):
        self.auto_approve_threshold = auto_approve_threshold
        self.auto_reject_threshold = auto_reject_threshold
        self.approval_gate = ApprovalGate()
    
    async def smart_approval(
        self,
        agent_name: str,
        action: str,
        output: Any,
        confidence: float,
        context: Optional[Dict] = None
    ) -> tuple[ApprovalStatus, Optional[str]]:
        """Request approval only if confidence is uncertain."""
        print(f"[Confidence] {confidence:.2f}")
        
        # High confidence - auto approve
        if confidence >= self.auto_approve_threshold:
            print("  ✅ Auto-approved (high confidence)")
            return ApprovalStatus.APPROVED, "Auto-approved"
        
        # Low confidence - auto reject
        elif confidence <= self.auto_reject_threshold:
            print("  ❌ Auto-rejected (low confidence)")
            return ApprovalStatus.REJECTED, "Low confidence"
        
        # Medium confidence - ask human
        else:
            print("  👤 Requesting human approval (medium confidence)")
            context = context or {}
            context['confidence'] = confidence
            
            return await self.approval_gate.request_approval(
                agent_name=agent_name,
                action=action,
                proposed_output=output,
                context=context
            )

# Usage
conf_gate = ConfidenceGate(auto_approve_threshold=0.85)

# High confidence - auto approved
status, _ = await conf_gate.smart_approval(
    "Agent", "task", "output", confidence=0.95, context={}
)

# Medium confidence - asks human
status, _ = await conf_gate.smart_approval(
    "Agent", "task", "output", confidence=0.70, context={}
)
\`\`\`

## Review & Revision Loop

\`\`\`python
class ReviewLoop:
    """Iterative review and revision with human."""
    
    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
    
    async def execute_with_review(
        self,
        generator: Callable,
        human_reviewer: Callable,
        initial_input: Any
    ) -> Dict[str, Any]:
        """Execute with human review loop."""
        iterations = []
        current_output = None
        
        for i in range(self.max_iterations):
            print(f"\\nIteration {i+1}/{self.max_iterations}")
            
            # Generate or revise
            if i == 0:
                current_output = await generator(initial_input)
            else:
                # Revise based on feedback
                current_output = await generator({
                    "input": initial_input,
                    "previous": current_output,
                    "feedback": feedback
                })
            
            print(f"Generated output: {current_output[:100]}...")
            
            # Human review
            approved, feedback = await human_reviewer(current_output)
            
            iterations.append({
                "iteration": i + 1,
                "output": current_output,
                "approved": approved,
                "feedback": feedback
            })
            
            if approved:
                print("✅ Approved!")
                return {
                    "success": True,
                    "final_output": current_output,
                    "iterations": iterations
                }
            else:
                print(f"❌ Revision needed: {feedback}")
        
        # Max iterations reached
        return {
            "success": False,
            "final_output": current_output,
            "iterations": iterations,
            "message": "Max iterations reached"
        }

# Example
async def generate_article(input: Any) -> str:
    """Generate or revise article."""
    if isinstance(input, dict):
        # Revision
        return f"Revised article incorporating: {input['feedback']}"
    else:
        # Initial generation
        return f"Initial article about {input}"

async def human_review(output: str) -> tuple[bool, str]:
    """Human reviews output."""
    # Simulate human review
    import random
    approved = random.random() > 0.5
    feedback = "" if approved else "Add more examples"
    return approved, feedback

loop = ReviewLoop(max_iterations=3)
result = await loop.execute_with_review(
    generator=generate_article,
    human_reviewer=human_review,
    initial_input="AI trends"
)

print(f"\\nSuccess: {result['success']}")
print(f"Iterations: {len(result['iterations'])}")
\`\`\`

## Exception Escalation

\`\`\`python
class EscalationSystem:
    """Escalate exceptions to humans."""
    
    def __init__(self):
        self.escalated_cases: List[Dict] = []
    
    async def handle_with_escalation(
        self,
        agent_task: Callable,
        task: Any,
        human_handler: Callable
    ) -> Any:
        """Try agent, escalate to human on failure."""
        try:
            # Try agent
            print("[Agent] Attempting task...")
            result = await agent_task(task)
            print("[Agent] ✅ Completed successfully")
            return result
        
        except Exception as e:
            # Escalate to human
            print(f"[Agent] ❌ Failed: {e}")
            print("[System] 🆘 Escalating to human...")
            
            # Record escalation
            escalation = {
                "task": task,
                "error": str(e),
                "timestamp": time.time()
            }
            self.escalated_cases.append(escalation)
            
            # Human handles
            result = await human_handler(task, error=e)
            return result

# Usage
escalation = EscalationSystem()

async def agent_task(task: str) -> str:
    """Agent attempts task."""
    if "complex" in task.lower():
        raise ValueError("Task too complex for agent")
    return f"Agent handled: {task}"

async def human_handler(task: str, error: Exception) -> str:
    """Human handles escalated case."""
    print(f"[Human] Handling: {task}")
    print(f"[Human] Original error: {error}")
    return f"Human handled: {task}"

# Simple task - agent handles
result = await escalation.handle_with_escalation(
    agent_task,
    "simple task",
    human_handler
)

# Complex task - escalates to human
result = await escalation.handle_with_escalation(
    agent_task,
    "complex task",
    human_handler
)

print(f"\\nEscalated cases: {len(escalation.escalated_cases)}")
\`\`\`

## Progressive Automation

\`\`\`python
class ProgressiveAutomation:
    """Gradually increase automation as confidence grows."""
    
    def __init__(self):
        self.task_history: Dict[str, List[bool]] = {}  # task_type -> [success/fail]
        self.automation_thresholds = {
            "low": 0.5,      # Start requiring approval
            "medium": 0.7,   # Occasional approval
            "high": 0.9      # Full automation
        }
    
    def get_automation_level(self, task_type: str) -> str:
        """Determine automation level for task type."""
        if task_type not in self.task_history:
            return "low"  # New task type - require approval
        
        history = self.task_history[task_type]
        if len(history) < 5:
            return "low"  # Not enough data
        
        # Calculate success rate
        success_rate = sum(history) / len(history)
        
        if success_rate >= self.automation_thresholds["high"]:
            return "high"
        elif success_rate >= self.automation_thresholds["medium"]:
            return "medium"
        else:
            return "low"
    
    def should_request_approval(self, task_type: str) -> bool:
        """Determine if approval needed."""
        level = self.get_automation_level(task_type)
        
        if level == "high":
            return False  # Full automation
        elif level == "medium":
            # Random sampling - 20% approval
            import random
            return random.random() < 0.2
        else:
            return True  # Always need approval
    
    def record_outcome(self, task_type: str, success: bool):
        """Record task outcome."""
        if task_type not in self.task_history:
            self.task_history[task_type] = []
        
        self.task_history[task_type].append(success)
        
        # Keep only recent history (last 20)
        self.task_history[task_type] = self.task_history[task_type][-20:]

# Usage
progressive = ProgressiveAutomation()

async def execute_with_progressive_automation(task_type: str, task: str):
    """Execute with progressive automation."""
    if progressive.should_request_approval(task_type):
        print(f"[{task_type}] Requesting approval...")
        # Request approval
        approved = True  # Simulate
    else:
        print(f"[{task_type}] Auto-proceeding (high confidence)")
        approved = True
    
    if approved:
        # Execute
        success = True  # Simulate
        progressive.record_outcome(task_type, success)
        return "Success"
    else:
        return "Rejected"

# First few times - always needs approval
for i in range(10):
    result = await execute_with_progressive_automation("email_draft", f"Task {i}")
    print(f"  Level: {progressive.get_automation_level('email_draft')}")
\`\`\`

## Real-Time Feedback

\`\`\`python
class FeedbackCollector:
    """Collect real-time human feedback."""
    
    def __init__(self):
        self.feedback_queue: asyncio.Queue = asyncio.Queue()
    
    async def stream_with_feedback(
        self,
        generator: AsyncIterator[str],
        processor: Callable[[str], Any]
    ) -> List[str]:
        """Stream output and collect feedback."""
        chunks = []
        
        # Start feedback listener
        feedback_task = asyncio.create_task(self._listen_for_feedback())
        
        try:
            async for chunk in generator:
                chunks.append(chunk)
                print(chunk, end=', flush=True)
                
                # Check for feedback
                if not self.feedback_queue.empty():
                    feedback = await self.feedback_queue.get()
                    print(f"\\n[Feedback received: {feedback}]")
                    
                    if feedback == "stop":
                        print("\\n[Stopping generation]")
                        break
                    elif feedback.startswith("revise:"):
                        # Could adjust generation
                        pass
        finally:
            feedback_task.cancel()
        
        return chunks
    
    async def _listen_for_feedback(self):
        """Listen for human feedback."""
        # In real system, this would listen to UI events
        pass
    
    def provide_feedback(self, feedback: str):
        """Human provides feedback."""
        self.feedback_queue.put_nowait(feedback)

# Usage
collector = FeedbackCollector()

async def slow_generator():
    """Simulate streaming generation."""
    words = ["The", "agent", "is", "generating", "this", "response", "slowly"]
    for word in words:
        yield word + " "
        await asyncio.sleep(0.5)

# Human can provide feedback during generation
# collector.provide_feedback("stop")  # Stop generation
# collector.provide_feedback("revise: add more detail")  # Adjust

chunks = await collector.stream_with_feedback(
    slow_generator(),
    processor=lambda x: x
)
\`\`\`

## Audit Trail

\`\`\`python
@dataclass
class HumanDecision:
    """Record of human decision."""
    timestamp: float
    decision_type: str  # "approval", "rejection", "revision"
    agent_name: str
    task: str
    decision: str
    feedback: Optional[str]
    user_id: Optional[str]

class AuditTrail:
    """Track all human decisions."""
    
    def __init__(self):
        self.decisions: List[HumanDecision] = []
    
    def record_decision(
        self,
        decision_type: str,
        agent_name: str,
        task: str,
        decision: str,
        feedback: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """Record human decision."""
        self.decisions.append(HumanDecision(
            timestamp=time.time(),
            decision_type=decision_type,
            agent_name=agent_name,
            task=task,
            decision=decision,
            feedback=feedback,
            user_id=user_id
        ))
    
    def get_decisions(
        self,
        agent_name: Optional[str] = None,
        decision_type: Optional[str] = None
    ) -> List[HumanDecision]:
        """Query decisions."""
        filtered = self.decisions
        
        if agent_name:
            filtered = [d for d in filtered if d.agent_name == agent_name]
        
        if decision_type:
            filtered = [d for d in filtered if d.decision_type == decision_type]
        
        return filtered
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get decision statistics."""
        if not self.decisions:
            return {}
        
        approvals = len([d for d in self.decisions if d.decision == "approved"])
        rejections = len([d for d in self.decisions if d.decision == "rejected"])
        
        return {
            "total_decisions": len(self.decisions),
            "approvals": approvals,
            "rejections": rejections,
            "approval_rate": approvals / len(self.decisions) if self.decisions else 0
        }

# Usage
audit = AuditTrail()

audit.record_decision(
    decision_type="approval",
    agent_name="WriterAgent",
    task="Write article",
    decision="approved",
    feedback="Excellent work",
    user_id="user123"
)

stats = audit.get_statistics()
print(f"Approval rate: {stats['approval_rate']:.1%}")
\`\`\`

## Best Practices

1. **Clear Communication**: Show agent intent clearly
2. **Reasonable Timeouts**: Don't wait forever
3. **Smart Defaults**: Auto-proceed when safe
4. **Progressive Trust**: Automate more over time
5. **Easy Feedback**: Make feedback simple
6. **Audit Everything**: Track all decisions
7. **Learn**: Use feedback to improve

## Next Steps

You now understand HITL patterns. Next, learn:
- Production deployment
- Monitoring and scaling
- Complete system integration
`,
};
