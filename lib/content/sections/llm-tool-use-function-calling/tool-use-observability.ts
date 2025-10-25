export const toolUseObservability = {
  title: 'Tool Use Observability',
  id: 'tool-use-observability',
  description:
    'Implement comprehensive monitoring, logging, and debugging for tool-using systems to ensure reliability and identify issues.',
  content: `

# Tool Use Observability

## Introduction

Production tool-using systems need comprehensive observability to:
- Debug failures and unexpected behavior
- Monitor performance and costs
- Understand usage patterns
- Optimize tool selection and parameters
- Ensure reliability and uptime
- Track user satisfaction

Without proper observability, you're flying blind. In this section, we'll build a complete observability system for tool-using agents.

## Logging Fundamentals

### Structured Logging

Use structured logging for easy parsing and analysis:

\`\`\`python
import logging
import json
from datetime import datetime
from typing import Dict, Any

class ToolLogger:
    """Structured logger for tool executions."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger (name)
        self.logger.setLevel (logging.INFO)
        
        # JSON formatter
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": %(message)s}'
        )
        
        # File handler
        handler = logging.FileHandler('tool_usage.log')
        handler.setFormatter (formatter)
        self.logger.addHandler (handler)
    
    def log_tool_call (self, 
                     tool_name: str,
                     arguments: Dict[str, Any],
                     user_id: str = None,
                     request_id: str = None):
        """Log a tool call."""
        log_data = {
            "event": "tool_call_start",
            "tool_name": tool_name,
            "arguments": arguments,
            "user_id": user_id,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        }
        self.logger.info (json.dumps (log_data))
    
    def log_tool_result (self,
                       tool_name: str,
                       result: Dict[str, Any],
                       execution_time_ms: float,
                       success: bool,
                       user_id: str = None,
                       request_id: str = None):
        """Log tool execution result."""
        log_data = {
            "event": "tool_call_complete",
            "tool_name": tool_name,
            "success": success,
            "execution_time_ms": execution_time_ms,
            "result_size_bytes": len (json.dumps (result)),
            "user_id": user_id,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        }
        self.logger.info (json.dumps (log_data))
    
    def log_tool_error (self,
                      tool_name: str,
                      error: Exception,
                      arguments: Dict[str, Any],
                      user_id: str = None,
                      request_id: str = None):
        """Log tool execution error."""
        log_data = {
            "event": "tool_call_error",
            "tool_name": tool_name,
            "error_type": type (error).__name__,
            "error_message": str (error),
            "arguments": arguments,
            "user_id": user_id,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        }
        self.logger.error (json.dumps (log_data))

# Usage
tool_logger = ToolLogger("tool_execution")

def execute_tool_with_logging (tool_name: str, arguments: Dict, 
                              user_id: str, request_id: str):
    """Execute tool with comprehensive logging."""
    tool_logger.log_tool_call (tool_name, arguments, user_id, request_id)
    
    start_time = time.time()
    
    try:
        result = registry.execute (tool_name, **arguments)
        execution_time = (time.time() - start_time) * 1000
        
        tool_logger.log_tool_result(
            tool_name, result, execution_time, True, user_id, request_id
        )
        
        return result
    
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        
        tool_logger.log_tool_error (tool_name, e, arguments, user_id, request_id)
        
        raise
\`\`\`

## Metrics Collection

### Key Metrics to Track

\`\`\`python
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List
import time

@dataclass
class ToolMetrics:
    """Metrics for tool usage."""
    # Call counts
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    
    # Timing
    total_execution_time_ms: float = 0.0
    execution_times: List[float] = field (default_factory=list)
    
    # Errors
    errors_by_type: Dict[str, int] = field (default_factory=lambda: defaultdict (int))
    
    # Cost
    total_cost_usd: float = 0.0
    
    @property
    def success_rate (self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls
    
    @property
    def average_execution_time_ms (self) -> float:
        if not self.execution_times:
            return 0.0
        return sum (self.execution_times) / len (self.execution_times)
    
    @property
    def p95_execution_time_ms (self) -> float:
        if not self.execution_times:
            return 0.0
        sorted_times = sorted (self.execution_times)
        index = int (len (sorted_times) * 0.95)
        return sorted_times[index]

class MetricsCollector:
    """Collect and aggregate tool metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, ToolMetrics] = defaultdict(ToolMetrics)
        self.global_metrics = ToolMetrics()
    
    def record_call (self, 
                   tool_name: str,
                   execution_time_ms: float,
                   success: bool,
                   error_type: str = None,
                   cost_usd: float = 0.0):
        """Record a tool call."""
        # Tool-specific metrics
        metrics = self.metrics[tool_name]
        metrics.total_calls += 1
        metrics.execution_times.append (execution_time_ms)
        metrics.total_execution_time_ms += execution_time_ms
        metrics.total_cost_usd += cost_usd
        
        if success:
            metrics.successful_calls += 1
        else:
            metrics.failed_calls += 1
            if error_type:
                metrics.errors_by_type[error_type] += 1
        
        # Global metrics
        self.global_metrics.total_calls += 1
        self.global_metrics.execution_times.append (execution_time_ms)
        self.global_metrics.total_execution_time_ms += execution_time_ms
        self.global_metrics.total_cost_usd += cost_usd
        
        if success:
            self.global_metrics.successful_calls += 1
        else:
            self.global_metrics.failed_calls += 1
    
    def get_metrics (self, tool_name: str = None) -> ToolMetrics:
        """Get metrics for a specific tool or global."""
        if tool_name:
            return self.metrics[tool_name]
        return self.global_metrics
    
    def print_report (self):
        """Print metrics report."""
        print("\\n=== Tool Usage Metrics ===\\n")
        print(f"Global Metrics:")
        print(f"  Total Calls: {self.global_metrics.total_calls}")
        print(f"  Success Rate: {self.global_metrics.success_rate:.1%}")
        print(f"  Avg Execution: {self.global_metrics.average_execution_time_ms:.2f}ms")
        print(f"  P95 Execution: {self.global_metrics.p95_execution_time_ms:.2f}ms")
        print(f"  Total Cost: \${self.global_metrics.total_cost_usd:.4f}")

print("\\nPer-Tool Metrics:")
for tool_name, metrics in sorted (self.metrics.items(),
    key = lambda x: x[1].total_calls,
    reverse = True):
    print(f"\\n{tool_name}:")
print(f"  Calls: {metrics.total_calls}")
print(f"  Success Rate: {metrics.success_rate:.1%}")
print(f"  Avg Time: {metrics.average_execution_time_ms:.2f}ms")
print(f"  Cost: \${metrics.total_cost_usd:.4f}")

if metrics.errors_by_type:
    print(f"  Errors:")
for error_type, count in metrics.errors_by_type.items():
    print(f"    {error_type}: {count}")

# Usage
metrics = MetricsCollector()

def execute_tool_with_metrics (tool_name: str, ** kwargs):
"""Execute tool and record metrics."""
start_time = time.time()

try:
result = registry.execute (tool_name, ** kwargs)
execution_time = (time.time() - start_time) * 1000
        
        # Get cost from tool metadata
tool = registry.get (tool_name)
cost = tool.estimated_cost if tool else 0.0

metrics.record_call (tool_name, execution_time, True, cost_usd = cost)

return result
    
    except Exception as e:
execution_time = (time.time() - start_time) * 1000
error_type = type (e).__name__

metrics.record_call (tool_name, execution_time, False, error_type = error_type)

raise

# Print report
metrics.print_report()
\`\`\`

## Distributed Tracing

Track tool calls across distributed systems:

\`\`\`python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger import JaegerExporter
import uuid

# Setup tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Jaeger exporter (can also use Zipkin, etc.)
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

span_processor = BatchSpanProcessor (jaeger_exporter)
trace.get_tracer_provider().add_span_processor (span_processor)

def execute_tool_with_tracing (tool_name: str, 
                              arguments: Dict[str, Any],
                              parent_span_context = None):
    """Execute tool with distributed tracing."""
    
    with tracer.start_as_current_span(
        f"tool.{tool_name}",
        context=parent_span_context
    ) as span:
        # Add attributes
        span.set_attribute("tool.name", tool_name)
        span.set_attribute("tool.arguments", json.dumps (arguments))
        
        try:
            # Execute tool
            result = registry.execute (tool_name, **arguments)
            
            # Record success
            span.set_attribute("tool.success", True)
            span.set_attribute("tool.result_size", len (json.dumps (result)))
            
            return result
        
        except Exception as e:
            # Record error
            span.set_attribute("tool.success", False)
            span.set_attribute("tool.error_type", type (e).__name__)
            span.set_attribute("tool.error_message", str (e))
            span.record_exception (e)
            
            raise

# For LLM conversation with multiple tool calls
def execute_llm_conversation_with_tracing (user_message: str):
    """Execute full LLM conversation with tracing."""
    
    request_id = str (uuid.uuid4())
    
    with tracer.start_as_current_span (f"llm.conversation.{request_id}") as conversation_span:
        conversation_span.set_attribute("user.message", user_message)
        conversation_span.set_attribute("request.id", request_id)
        
        # Initial LLM call
        with tracer.start_as_current_span("llm.call.initial") as llm_span:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": user_message}],
                functions=function_schemas
            )
            llm_span.set_attribute("llm.model", "gpt-4")
            llm_span.set_attribute("llm.tokens_used", response.usage.total_tokens)
        
        # Execute tool calls
        if response.choices[0].message.function_call:
            func_name = response.choices[0].message.function_call.name
            func_args = json.loads (response.choices[0].message.function_call.arguments)
            
            # Execute with tracing
            result = execute_tool_with_tracing(
                func_name,
                func_args,
                parent_span_context=conversation_span.get_span_context()
            )
        
        conversation_span.set_attribute("conversation.success", True)
\`\`\`

## Real-Time Monitoring Dashboard

Build a monitoring dashboard:

\`\`\`python
from flask import Flask, jsonify, render_template
from datetime import datetime, timedelta

app = Flask(__name__)

class MonitoringDashboard:
    """Real-time monitoring dashboard."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.alerts = []
    
    def get_dashboard_data (self) -> Dict[str, Any]:
        """Get data for dashboard."""
        global_metrics = self.metrics.get_metrics()
        
        # Calculate rates
        calls_per_minute = global_metrics.total_calls / 60.0  # Simplified
        errors_per_minute = global_metrics.failed_calls / 60.0
        
        # Top tools
        top_tools = sorted(
            [(name, m) for name, m in self.metrics.metrics.items()],
            key=lambda x: x[1].total_calls,
            reverse=True
        )[:10]
        
        return {
            "summary": {
                "total_calls": global_metrics.total_calls,
                "success_rate": global_metrics.success_rate,
                "avg_latency_ms": global_metrics.average_execution_time_ms,
                "p95_latency_ms": global_metrics.p95_execution_time_ms,
                "total_cost_usd": global_metrics.total_cost_usd,
                "calls_per_minute": calls_per_minute,
                "errors_per_minute": errors_per_minute
            },
            "top_tools": [
                {
                    "name": name,
                    "calls": metrics.total_calls,
                    "success_rate": metrics.success_rate,
                    "avg_latency_ms": metrics.average_execution_time_ms
                }
                for name, metrics in top_tools
            ],
            "alerts": self.alerts,
            "timestamp": datetime.now().isoformat()
        }

dashboard = MonitoringDashboard (metrics)

@app.route('/api/metrics')
def get_metrics():
    """API endpoint for metrics."""
    return jsonify (dashboard.get_dashboard_data())

@app.route('/dashboard')
def show_dashboard():
    """Show dashboard page."""
    return render_template('dashboard.html')

# Run dashboard
# app.run (port=5000)
\`\`\`

## Alerting System

Implement alerts for issues:

\`\`\`python
from enum import Enum
from dataclasses import dataclass
from typing import List, Callable

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Alert:
    """Alert definition."""
    severity: AlertSeverity
    message: str
    tool_name: str = None
    timestamp: datetime = None
    metadata: Dict[str, Any] = None

class AlertManager:
    """Manage alerts for tool usage."""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable] = []
    
    def add_alert (self, alert: Alert):
        """Add an alert."""
        alert.timestamp = datetime.now()
        self.alerts.append (alert)
        
        # Trigger handlers
        for handler in self.alert_handlers:
            handler (alert)
    
    def add_handler (self, handler: Callable):
        """Add alert handler."""
        self.alert_handlers.append (handler)
    
    def check_metrics (self, metrics: MetricsCollector):
        """Check metrics and generate alerts."""
        global_metrics = metrics.get_metrics()
        
        # Check success rate
        if global_metrics.success_rate < 0.95:
            self.add_alert(Alert(
                severity=AlertSeverity.WARNING,
                message=f"Global success rate dropped to {global_metrics.success_rate:.1%}",
                metadata={"success_rate": global_metrics.success_rate}
            ))
        
        # Check latency
        if global_metrics.p95_execution_time_ms > 5000:
            self.add_alert(Alert(
                severity=AlertSeverity.WARNING,
                message=f"P95 latency is {global_metrics.p95_execution_time_ms:.0f}ms",
                metadata={"p95_latency_ms": global_metrics.p95_execution_time_ms}
            ))
        
        # Check per-tool metrics
        for tool_name, tool_metrics in metrics.metrics.items():
            # Tool-specific success rate
            if tool_metrics.total_calls > 10 and tool_metrics.success_rate < 0.90:
                self.add_alert(Alert(
                    severity=AlertSeverity.ERROR,
                    message=f"{tool_name} success rate is {tool_metrics.success_rate:.1%}",
                    tool_name=tool_name,
                    metadata={"success_rate": tool_metrics.success_rate}
                ))
            
            # Tool-specific latency
            if tool_metrics.p95_execution_time_ms > 10000:
                self.add_alert(Alert(
                    severity=AlertSeverity.WARNING,
                    message=f"{tool_name} P95 latency is {tool_metrics.p95_execution_time_ms:.0f}ms",
                    tool_name=tool_name
                ))

# Alert handlers
def log_alert (alert: Alert):
    """Log alert."""
    logging.warning (f"[{alert.severity.value.upper()}] {alert.message}")

def send_slack_alert (alert: Alert):
    """Send alert to Slack."""
    if alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
        # Send to Slack
        slack_client.post_message(
            channel="#alerts",
            text=f"🚨 {alert.message}",
            severity=alert.severity.value
        )

# Setup alerts
alert_manager = AlertManager()
alert_manager.add_handler (log_alert)
alert_manager.add_handler (send_slack_alert)

# Periodic check
import threading

def check_metrics_periodically():
    """Check metrics every minute."""
    while True:
        alert_manager.check_metrics (metrics)
        time.sleep(60)

alert_thread = threading.Thread (target=check_metrics_periodically, daemon=True)
alert_thread.start()
\`\`\`

## LLM Decision Tracking

Track LLM decisions about tool use:

\`\`\`python
@dataclass
class LLMDecision:
    """Track LLM tool use decisions."""
    request_id: str
    user_query: str
    available_tools: List[str]
    chosen_tool: str = None
    tool_arguments: Dict[str, Any] = None
    decision_time_ms: float = 0.0
    tokens_used: int = 0
    model: str = "gpt-4"
    timestamp: datetime = None

class DecisionTracker:
    """Track LLM tool use decisions."""
    
    def __init__(self):
        self.decisions: List[LLMDecision] = []
    
    def track_decision (self,
                      request_id: str,
                      user_query: str,
                      available_tools: List[str],
                      response,
                      decision_time_ms: float):
        """Track an LLM decision."""
        decision = LLMDecision(
            request_id=request_id,
            user_query=user_query,
            available_tools=available_tools,
            decision_time_ms=decision_time_ms,
            tokens_used=response.usage.total_tokens,
            timestamp=datetime.now()
        )
        
        if response.choices[0].message.function_call:
            decision.chosen_tool = response.choices[0].message.function_call.name
            decision.tool_arguments = json.loads(
                response.choices[0].message.function_call.arguments
            )
        
        self.decisions.append (decision)
    
    def analyze_decisions (self):
        """Analyze LLM decisions."""
        if not self.decisions:
            return {}
        
        # Tool selection frequency
        tool_counts = defaultdict (int)
        for decision in self.decisions:
            if decision.chosen_tool:
                tool_counts[decision.chosen_tool] += 1
        
        # Average decision time
        avg_decision_time = sum (d.decision_time_ms for d in self.decisions) / len (self.decisions)
        
        # No-tool decision rate
        no_tool_count = sum(1 for d in self.decisions if not d.chosen_tool)
        no_tool_rate = no_tool_count / len (self.decisions)
        
        return {
            "total_decisions": len (self.decisions),
            "tool_selection_frequency": dict (tool_counts),
            "avg_decision_time_ms": avg_decision_time,
            "no_tool_decision_rate": no_tool_rate,
            "most_used_tool": max (tool_counts.items(), key=lambda x: x[1])[0] if tool_counts else None
        }

decision_tracker = DecisionTracker()
\`\`\`

## Cost Tracking

Detailed cost tracking:

\`\`\`python
class CostTracker:
    """Track costs for LLM and tool usage."""
    
    def __init__(self):
        self.costs = {
            "llm_calls": 0.0,
            "tool_calls": defaultdict (float),
            "total": 0.0
        }
        self.cost_by_user = defaultdict (float)
    
    def track_llm_cost (self, tokens: int, model: str, user_id: str = None):
        """Track LLM API cost."""
        # Cost per 1K tokens (example rates)
        rates = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002}
        }
        
        # Simplified: assume 50/50 input/output
        cost = (tokens / 1000) * (rates[model]["input"] + rates[model]["output"]) / 2
        
        self.costs["llm_calls"] += cost
        self.costs["total"] += cost
        
        if user_id:
            self.cost_by_user[user_id] += cost
    
    def track_tool_cost (self, tool_name: str, cost: float, user_id: str = None):
        """Track tool execution cost."""
        self.costs["tool_calls"][tool_name] += cost
        self.costs["total"] += cost
        
        if user_id:
            self.cost_by_user[user_id] += cost
    
    def get_report (self) -> Dict[str, Any]:
        """Get cost report."""
        return {
            "total_cost_usd": self.costs["total"],
            "llm_cost_usd": self.costs["llm_calls"],
            "tool_costs_usd": dict (self.costs["tool_calls"]),
            "cost_by_user": dict (self.cost_by_user),
            "most_expensive_tool": max(
                self.costs["tool_calls"].items(),
                key=lambda x: x[1]
            )[0] if self.costs["tool_calls"] else None
        }

cost_tracker = CostTracker()

# Track in execution
def execute_with_cost_tracking (tool_name: str, user_id: str, **kwargs):
    """Execute tool with cost tracking."""
    # Get tool cost
    tool = registry.get (tool_name)
    cost = tool.estimated_cost if tool else 0.0
    
    # Execute
    result = registry.execute (tool_name, **kwargs)
    
    # Track cost
    cost_tracker.track_tool_cost (tool_name, cost, user_id)
    
    return result
\`\`\`

## Integration Example

Complete integration of all observability features:

\`\`\`python
class ObservableToolExecutor:
    """Tool executor with full observability."""
    
    def __init__(self):
        self.logger = ToolLogger("observable_executor")
        self.metrics = MetricsCollector()
        self.alert_manager = AlertManager()
        self.decision_tracker = DecisionTracker()
        self.cost_tracker = CostTracker()
    
    def execute (self,
                tool_name: str,
                arguments: Dict[str, Any],
                user_id: str,
                request_id: str) -> Dict[str, Any]:
        """Execute tool with full observability."""
        
        # Log start
        self.logger.log_tool_call (tool_name, arguments, user_id, request_id)
        
        # Start tracing
        with tracer.start_as_current_span (f"tool.{tool_name}") as span:
            span.set_attribute("tool.name", tool_name)
            span.set_attribute("user.id", user_id)
            
            start_time = time.time()
            
            try:
                # Execute
                result = registry.execute (tool_name, **arguments)
                execution_time = (time.time() - start_time) * 1000
                
                # Get cost
                tool = registry.get (tool_name)
                cost = tool.estimated_cost if tool else 0.0
                
                # Record success
                self.metrics.record_call (tool_name, execution_time, True, cost_usd=cost)
                self.cost_tracker.track_tool_cost (tool_name, cost, user_id)
                self.logger.log_tool_result(
                    tool_name, result, execution_time, True, user_id, request_id
                )
                
                span.set_attribute("tool.success", True)
                
                return result
            
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                error_type = type (e).__name__
                
                # Record failure
                self.metrics.record_call (tool_name, execution_time, False, error_type=error_type)
                self.logger.log_tool_error (tool_name, e, arguments, user_id, request_id)
                
                span.set_attribute("tool.success", False)
                span.record_exception (e)
                
                # Check if alert needed
                tool_metrics = self.metrics.get_metrics (tool_name)
                if tool_metrics.success_rate < 0.90:
                    self.alert_manager.add_alert(Alert(
                        severity=AlertSeverity.ERROR,
                        message=f"{tool_name} failing frequently",
                        tool_name=tool_name
                    ))
                
                raise
    
    def get_health_status (self) -> Dict[str, Any]:
        """Get system health status."""
        global_metrics = self.metrics.get_metrics()
        
        return {
            "status": "healthy" if global_metrics.success_rate > 0.95 else "degraded",
            "success_rate": global_metrics.success_rate,
            "avg_latency_ms": global_metrics.average_execution_time_ms,
            "active_alerts": len (self.alert_manager.alerts),
            "total_cost_usd": self.cost_tracker.costs["total"]
        }

# Usage
executor = ObservableToolExecutor()

result = executor.execute(
    tool_name="get_weather",
    arguments={"location": "San Francisco"},
    user_id="user_123",
    request_id="req_abc"
)

# Check health
health = executor.get_health_status()
print(f"System Status: {health['status']}")
\`\`\`

## Summary

Comprehensive observability requires:
1. **Structured Logging** - JSON logs for easy parsing
2. **Metrics Collection** - Track calls, latency, success rates, costs
3. **Distributed Tracing** - Follow requests across systems
4. **Real-Time Dashboards** - Visualize metrics
5. **Alerting** - Notify on issues
6. **Decision Tracking** - Understand LLM choices
7. **Cost Tracking** - Monitor expenses
8. **Health Checks** - System status monitoring

Next, we'll explore advanced tool patterns like chaining, composition, and dynamic generation.
`,
};
