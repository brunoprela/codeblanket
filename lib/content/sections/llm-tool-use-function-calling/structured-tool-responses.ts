export const structuredToolResponses = {
  title: 'Structured Tool Responses',
  id: 'structured-tool-responses',
  description:
    'Design tool responses that LLMs can easily understand and act upon, with consistent formats, rich metadata, and error handling.',
  content: `

# Structured Tool Responses

## Introduction

The format and structure of tool responses significantly impacts how well LLMs can use them. A well-structured response makes it easy for the LLM to extract relevant information, handle errors, and provide helpful answers to users. A poorly structured response leads to confusion, hallucinations, and failures.

In this section, we'll learn how to design tool responses that maximize LLM understanding and usability.

## Response Format Principles

### 1. Consistent Structure

All tool responses should follow a consistent format:

\`\`\`python
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

class ResponseStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"

@dataclass
class ToolResponse:
    """Standard structure for all tool responses."""
    status: ResponseStatus
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {"status": self.status.value}
        
        if self.data is not None:
            result["data"] = self.data
        
        if self.error:
            result["error"] = self.error
        
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result

# Example usage
def get_weather(location: str) -> Dict[str, Any]:
    """Get weather with structured response."""
    try:
        # Fetch weather data
        weather_data = fetch_weather_api(location)
        
        response = ToolResponse(
            status=ResponseStatus.SUCCESS,
            data={
                "location": location,
                "temperature": weather_data["temp"],
                "conditions": weather_data["conditions"],
                "humidity": weather_data["humidity"]
            },
            metadata={
                "source": "OpenWeatherMap",
                "timestamp": datetime.now().isoformat(),
                "units": "imperial"
            }
        )
        
        return response.to_dict()
    
    except LocationNotFoundError:
        response = ToolResponse(
            status=ResponseStatus.ERROR,
            error="Location not found. Please provide a valid city name."
        )
        return response.to_dict()
\`\`\`

### 2. Self-Describing Responses

Include context that helps the LLM understand the response:

\`\`\`python
# ❌ BAD: Ambiguous response
{
    "value": 42
}

# ✅ GOOD: Self-describing response
{
    "status": "success",
    "data": {
        "count": 42,
        "description": "Number of users who signed up in the last 7 days",
        "time_period": "2024-01-08 to 2024-01-15"
    }
}
\`\`\`

### 3. Include Units and Context

\`\`\`python
# ❌ BAD: No units
{
    "temperature": 72
}

# ✅ GOOD: Clear units and context
{
    "status": "success",
    "data": {
        "temperature": {
            "value": 72,
            "unit": "fahrenheit",
            "feels_like": 68
        },
        "location": {
            "city": "San Francisco",
            "country": "USA",
            "coordinates": {"lat": 37.7749, "lon": -122.4194}
        },
        "timestamp": "2024-01-15T14:30:00Z",
        "conditions": "Sunny with light breeze"
    }
}
\`\`\`

## Success Response Patterns

### Simple Data Response

For simple queries:

\`\`\`python
def get_user_count() -> Dict[str, Any]:
    """Get total user count."""
    count = database.execute("SELECT COUNT(*) FROM users").scalar()
    
    return {
        "status": "success",
        "data": {
            "total_users": count,
            "as_of": datetime.now().isoformat()
        },
        "message": f"Found {count} total users"
    }
\`\`\`

### List Response

For queries returning multiple items:

\`\`\`python
def search_products(query: str, limit: int = 10) -> Dict[str, Any]:
    """Search products."""
    results = product_search.search(query, limit=limit)
    
    return {
        "status": "success",
        "data": {
            "query": query,
            "results": [
                {
                    "id": product.id,
                    "name": product.name,
                    "price": {
                        "amount": product.price,
                        "currency": "USD"
                    },
                    "in_stock": product.inventory > 0,
                    "url": product.url
                }
                for product in results
            ],
            "count": len(results),
            "total_matches": product_search.total_count(query)
        },
        "metadata": {
            "search_time_ms": 45,
            "limit": limit,
            "truncated": len(results) == limit
        }
    }
\`\`\`

### Action Confirmation Response

For tools that perform actions:

\`\`\`python
def send_email(to: str, subject: str, body: str) -> Dict[str, Any]:
    """Send email with confirmation."""
    try:
        message_id = email_service.send(to, subject, body)
        
        return {
            "status": "success",
            "data": {
                "action": "email_sent",
                "message_id": message_id,
                "recipient": to,
                "subject": subject,
                "sent_at": datetime.now().isoformat()
            },
            "message": f"Email successfully sent to {to}"
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to send email: {str(e)}",
            "data": {
                "action": "email_send_failed",
                "recipient": to
            }
        }
\`\`\`

## Error Response Patterns

### Clear Error Messages

Make errors understandable and actionable:

\`\`\`python
def query_database(query: str) -> Dict[str, Any]:
    """Execute database query with clear error handling."""
    
    # Validate query
    if not query.strip().upper().startswith("SELECT"):
        return {
            "status": "error",
            "error": "Only SELECT queries are allowed",
            "error_code": "INVALID_QUERY_TYPE",
            "data": {
                "provided_query": query,
                "allowed_types": ["SELECT"]
            },
            "suggestion": "Modify your query to start with SELECT"
        }
    
    try:
        results = database.execute(query).fetchall()
        
        return {
            "status": "success",
            "data": {
                "rows": [dict(row) for row in results],
                "count": len(results)
            }
        }
    
    except DatabaseSyntaxError as e:
        return {
            "status": "error",
            "error": "SQL syntax error",
            "error_code": "SYNTAX_ERROR",
            "data": {
                "query": query,
                "error_detail": str(e),
                "position": e.position if hasattr(e, 'position') else None
            },
            "suggestion": "Check your SQL syntax and try again"
        }
    
    except DatabaseTimeoutError:
        return {
            "status": "error",
            "error": "Query timeout after 30 seconds",
            "error_code": "TIMEOUT",
            "data": {
                "query": query,
                "timeout_seconds": 30
            },
            "suggestion": "Simplify your query or add WHERE clauses to reduce data"
        }
\`\`\`

### Error Types and Codes

Use standard error codes:

\`\`\`python
class ErrorCode(Enum):
    INVALID_INPUT = "INVALID_INPUT"
    NOT_FOUND = "NOT_FOUND"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    TIMEOUT = "TIMEOUT"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"

def create_error_response(
    error_code: ErrorCode,
    error_message: str,
    details: Optional[Dict] = None,
    suggestion: Optional[str] = None
) -> Dict[str, Any]:
    """Create standardized error response."""
    return {
        "status": "error",
        "error_code": error_code.value,
        "error": error_message,
        "details": details or {},
        "suggestion": suggestion,
        "timestamp": datetime.now().isoformat()
    }

# Usage
return create_error_response(
    error_code=ErrorCode.NOT_FOUND,
    error_message="User not found",
    details={"user_id": user_id},
    suggestion="Check the user ID and try again"
)
\`\`\`

## Partial Success Responses

Handle cases where operation partially succeeds:

\`\`\`python
def send_bulk_email(recipients: List[str], subject: str, body: str) -> Dict[str, Any]:
    """Send email to multiple recipients."""
    successful = []
    failed = []
    
    for recipient in recipients:
        try:
            message_id = email_service.send(recipient, subject, body)
            successful.append({
                "recipient": recipient,
                "message_id": message_id,
                "status": "sent"
            })
        except Exception as e:
            failed.append({
                "recipient": recipient,
                "error": str(e),
                "status": "failed"
            })
    
    # Determine overall status
    if failed and not successful:
        status = "error"
    elif failed and successful:
        status = "partial"
    else:
        status = "success"
    
    return {
        "status": status,
        "data": {
            "total_recipients": len(recipients),
            "successful_count": len(successful),
            "failed_count": len(failed),
            "successful": successful,
            "failed": failed
        },
        "message": f"Sent {len(successful)}/{len(recipients)} emails successfully"
    }
\`\`\`

## Rich Data Responses

### Including Relevant Context

\`\`\`python
def get_stock_price(symbol: str) -> Dict[str, Any]:
    """Get stock price with rich context."""
    data = stock_api.get_quote(symbol)
    
    return {
        "status": "success",
        "data": {
            "symbol": symbol,
            "company_name": data["name"],
            "price": {
                "current": data["price"],
                "currency": "USD",
                "change": data["price"] - data["previous_close"],
                "change_percent": ((data["price"] - data["previous_close"]) / data["previous_close"]) * 100
            },
            "market_status": data["market_status"],  # "open", "closed", "pre_market"
            "volume": data["volume"],
            "high_today": data["high"],
            "low_today": data["low"],
            "previous_close": data["previous_close"]
        },
        "metadata": {
            "source": "Alpha Vantage",
            "as_of": data["timestamp"],
            "delay_minutes": 15 if not data["is_realtime"] else 0,
            "currency": "USD"
        },
        "message": f"{symbol} is trading at \${data['price']:.2f}"
    }
\`\`\`

### Including Visualizations

For data analysis tools:

\`\`\`python
def analyze_data(data: List[float]) -> Dict[str, Any]:
    """Analyze numerical data with visualization."""
    import numpy as np
    import matplotlib.pyplot as plt
    import io
    import base64
    
    # Calculate statistics
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=20, edgecolor='black')
    plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='green', linestyle='--', label=f'Median: {median:.2f}')
    plt.legend()
    plt.title('Data Distribution')
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return {
        "status": "success",
        "data": {
            "statistics": {
                "count": len(data),
                "mean": mean,
                "median": median,
                "std_dev": std,
                "min": min(data),
                "max": max(data)
            },
            "visualization": {
                "type": "histogram",
                "format": "png",
                "data": plot_base64,
                "description": "Histogram showing data distribution with mean and median"
            }
        },
        "message": f"Analyzed {len(data)} data points. Mean: {mean:.2f}, Median: {median:.2f}"
    }
\`\`\`

## Response Metadata

Include metadata to help with caching, debugging, and optimization:

\`\`\`python
def search_web(query: str) -> Dict[str, Any]:
    """Search with comprehensive metadata."""
    start_time = time.time()
    
    results = search_api.search(query)
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    return {
        "status": "success",
        "data": {
            "query": query,
            "results": results
        },
        "metadata": {
            # Performance
            "execution_time_ms": elapsed_ms,
            "api_latency_ms": results.get("api_latency"),
            
            # Caching
            "cache_hit": False,
            "cacheable": True,
            "cache_ttl_seconds": 3600,
            
            # Cost
            "api_cost_usd": 0.001,
            
            # Source
            "source": "Google Custom Search",
            "api_version": "v1",
            
            # Timestamp
            "timestamp": datetime.now().isoformat(),
            
            # Pagination
            "page": 1,
            "per_page": 10,
            "has_more": results.get("total") > 10
        }
    }
\`\`\`

## LLM-Friendly Formatting

### Natural Language Summaries

Include human-readable summaries:

\`\`\`python
def get_user_stats(user_id: str) -> Dict[str, Any]:
    """Get user statistics with LLM-friendly summary."""
    stats = fetch_user_stats(user_id)
    
    # Generate natural language summary
    summary = f"""User {stats['name']} has been a member since {stats['joined_date']}.
They have made {stats['order_count']} orders with a total value of \${stats['total_spent']: .2f}.
Their average order value is \${ stats['avg_order_value']: .2f }.
Last activity was { stats['days_since_last_activity'] } days ago."""

return {
    "status": "success",
    "data": {
        "user_id": user_id,
        "name": stats["name"],
        "statistics": {
            "orders": {
                "total_count": stats["order_count"],
                "total_value_usd": stats["total_spent"],
                "average_value_usd": stats["avg_order_value"]
            },
            "activity": {
                "member_since": stats["joined_date"],
                "last_activity": stats["last_activity"],
                "days_since_last_activity": stats["days_since_last_activity"]
            }
        },
        "summary": summary  # LLM can use this directly
    }
}
\`\`\`

### Highlighting Key Information

Use clear markers for important data:

\`\`\`python
{
    "status": "success",
    "data": {
        "results": [...],
        "key_findings": [  # Highlighted for LLM attention
            "Temperature is 15°F above normal for this time of year",
            "High probability of rain in the next 3 hours",
            "Air quality is poor (AQI: 150)"
        ],
        "recommendations": [  # Actionable advice
            "Carry an umbrella",
            "Consider postponing outdoor activities",
            "Keep windows closed due to poor air quality"
        ]
    }
}
\`\`\`

## Pagination Responses

Handle large result sets:

\`\`\`python
def list_users(page: int = 1, per_page: int = 20) -> Dict[str, Any]:
    """List users with pagination."""
    offset = (page - 1) * per_page
    users = database.query(User).offset(offset).limit(per_page).all()
    total_count = database.query(User).count()
    total_pages = (total_count + per_page - 1) // per_page
    
    return {
        "status": "success",
        "data": {
            "users": [user.to_dict() for user in users],
            "pagination": {
                "current_page": page,
                "per_page": per_page,
                "total_pages": total_pages,
                "total_count": total_count,
                "has_previous": page > 1,
                "has_next": page < total_pages,
                "previous_page": page - 1 if page > 1 else None,
                "next_page": page + 1 if page < total_pages else None
            }
        },
        "message": f"Showing page {page} of {total_pages} ({len(users)} users)"
    }
\`\`\`

## Response Validation

Validate responses before returning:

\`\`\`python
from pydantic import BaseModel, validator
from typing import Optional, Any

class StandardResponse(BaseModel):
    """Validated response structure."""
    status: str
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @validator('status')
    def status_must_be_valid(cls, v):
        if v not in ['success', 'error', 'partial']:
            raise ValueError('status must be success, error, or partial')
        return v
    
    @validator('error')
    def error_required_if_status_error(cls, v, values):
        if values.get('status') == 'error' and not v:
            raise ValueError('error message required when status is error')
        return v
    
    @validator('data')
    def data_required_if_success(cls, v, values):
        if values.get('status') == 'success' and v is None:
            raise ValueError('data required when status is success')
        return v

def validated_tool_response(response: Dict) -> Dict:
    """Validate and return response."""
    try:
        validated = StandardResponse(**response)
        return validated.dict(exclude_none=True)
    except Exception as e:
        # Return error response if validation fails
        return {
            "status": "error",
            "error": f"Response validation failed: {str(e)}"
        }
\`\`\`

## Testing Response Formats

\`\`\`python
import pytest

def test_successful_response_format():
    """Test successful response has required fields."""
    response = get_weather("London")
    
    assert response["status"] == "success"
    assert "data" in response
    assert response["data"]["temperature"] is not None

def test_error_response_format():
    """Test error response has required fields."""
    response = get_weather("NonexistentCity123")
    
    assert response["status"] == "error"
    assert "error" in response
    assert isinstance(response["error"], str)
    assert len(response["error"]) > 0

def test_response_has_metadata():
    """Test response includes metadata."""
    response = get_weather("London")
    
    assert "metadata" in response
    assert "timestamp" in response["metadata"]
    assert "source" in response["metadata"]

def test_llm_can_parse_response():
    """Test that LLM can understand the response."""
    response = get_weather("London")
    
    # Create a prompt asking LLM to extract temperature
    prompt = f"""Given this tool response: {json.dumps(response)}
    
Extract the temperature and location."""
    
    llm_response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    # LLM should successfully extract the information
    assert "London" in llm_response.choices[0].message.content
    assert "temperature" in llm_response.choices[0].message.content.lower()
\`\`\`

## Best Practices Summary

1. **Consistent Structure**: Use the same response format across all tools
2. **Self-Describing**: Include context and descriptions
3. **Clear Units**: Always specify units for measurements
4. **Rich Metadata**: Include timestamps, sources, costs
5. **Natural Language**: Include summaries for LLM consumption
6. **Error Details**: Provide actionable error messages
7. **Validation**: Validate responses before returning
8. **Pagination**: Handle large datasets properly
9. **Highlighting**: Mark key information
10. **Testing**: Verify LLMs can parse responses

## Complete Example

\`\`\`python
@tool(description="Get weather forecast")
def get_weather_forecast(location: str, days: int = 3) -> Dict[str, Any]:
    """
    Get weather forecast with perfectly structured response.
    """
    try:
        # Validate input
        if days < 1 or days > 7:
            return create_error_response(
                ErrorCode.INVALID_INPUT,
                "Days must be between 1 and 7",
                {"provided_days": days, "valid_range": [1, 7]},
                "Please provide a number of days between 1 and 7"
            )
        
        # Fetch data
        start_time = time.time()
        forecast_data = weather_api.get_forecast(location, days)
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Build response
        return {
            "status": "success",
            "data": {
                "location": {
                    "name": forecast_data["location"]["name"],
                    "country": forecast_data["location"]["country"],
                    "coordinates": forecast_data["location"]["coords"]
                },
                "forecast": [
                    {
                        "date": day["date"],
                        "temperature": {
                            "high": day["temp_high"],
                            "low": day["temp_low"],
                            "unit": "fahrenheit"
                        },
                        "conditions": day["conditions"],
                        "precipitation_chance": day["precip_chance"],
                        "wind_speed_mph": day["wind_speed"]
                    }
                    for day in forecast_data["days"]
                ],
                "summary": f"{days}-day forecast for {forecast_data['location']['name']}: "
                          f"Temperatures ranging from {min(d['temp_low'] for d in forecast_data['days'])}°F "
                          f"to {max(d['temp_high'] for d in forecast_data['days'])}°F"
            },
            "metadata": {
                "source": "OpenWeatherMap",
                "timestamp": datetime.now().isoformat(),
                "execution_time_ms": elapsed_ms,
                "api_cost_usd": 0.001,
                "cache_ttl_seconds": 3600
            }
        }
    
    except LocationNotFoundError:
        return create_error_response(
            ErrorCode.NOT_FOUND,
            f"Location '{location}' not found",
            {"provided_location": location},
            "Try a different location name or use 'City, Country' format"
        )
    
    except Exception as e:
        logger.exception(f"Weather forecast error: {e}")
        return create_error_response(
            ErrorCode.INTERNAL_ERROR,
            "An error occurred while fetching weather data",
            {"error_type": type(e).__name__},
            "Please try again in a moment"
        )
\`\`\`

Well-structured responses make all the difference in how effectively LLMs can use your tools. Invest time in designing them properly.

Next, we'll explore observability and monitoring for tool-using systems.
`,
};
