"""
Observability module for multi-agent system
Provides tracing, logging, and metrics using OpenTelemetry
"""
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
import logging
import time
from functools import wraps

# Set up OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Export traces to console
console_exporter = ConsoleSpanExporter()
span_processor = BatchSpanProcessor(console_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

logger = logging.getLogger(__name__)

def trace_agent_call(agent_name: str):
    """
    Decorator to trace agent execution
    Usage: @trace_agent_call("research_agent")
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(f"{agent_name}.execute") as span:
                # Add attributes
                span.set_attribute("agent.name", agent_name)
                span.set_attribute("agent.model", getattr(args[0], 'model', 'unknown'))
                
                # Get query from args
                if args and len(args) > 1:
                    query = str(args[1])[:100]  # First 100 chars
                    span.set_attribute("query", query)
                    span.set_attribute("query_length", len(str(args[1])))
                
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Log success
                    duration = time.time() - start_time
                    span.set_attribute("duration_ms", int(duration * 1000))
                    span.set_attribute("status", "success")
                    
                    logger.info(f"{agent_name} completed in {duration:.2f}s")
                    
                    return result
                    
                except Exception as e:
                    # Log error
                    span.set_attribute("status", "error")
                    span.set_attribute("error", str(e))
                    logger.error(f"{agent_name} failed: {e}")
                    raise
                    
        return wrapper
    return decorator

class AgentMetrics:
    """Track agent performance metrics"""
    
    def __init__(self):
        self.call_counts = {}
        self.total_latency = {}
        self.error_counts = {}
    
    def record_call(self, agent_name: str, duration: float, success: bool):
        """Record an agent call"""
        if agent_name not in self.call_counts:
            self.call_counts[agent_name] = 0
            self.total_latency[agent_name] = 0
            self.error_counts[agent_name] = 0
        
        self.call_counts[agent_name] += 1
        self.total_latency[agent_name] += duration
        
        if not success:
            self.error_counts[agent_name] += 1
    
    def get_stats(self):
        """Get performance statistics"""
        stats = {}
        for agent in self.call_counts:
            avg_latency = self.total_latency[agent] / self.call_counts[agent] if self.call_counts[agent] > 0 else 0
            error_rate = self.error_counts[agent] / self.call_counts[agent] if self.call_counts[agent] > 0 else 0
            
            stats[agent] = {
                "calls": self.call_counts[agent],
                "avg_latency_ms": int(avg_latency * 1000),
                "error_rate": f"{error_rate * 100:.1f}%"
            }
        
        return stats

# Global metrics instance
metrics = AgentMetrics()
