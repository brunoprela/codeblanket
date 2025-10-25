export const apiIntegrationTools = {
  title: 'API Integration Tools',
  id: 'api-integration-tools',
  description:
    'Build tools that integrate with external APIs and services, handling authentication, rate limiting, and error conditions.',
  content: `

# API Integration Tools

## Introduction

Most real-world agentic systems need to interact with external APIs - weather services, databases, email providers, calendar systems, payment processors, and more. Building reliable API integration tools requires handling authentication, rate limiting, retries, error conditions, and data transformation.

In this section, we'll master building production-grade API integration tools that are robust, efficient, and easy to maintain.

## Basic API Integration Pattern

Every API integration tool follows a similar pattern:

\`\`\`python
import requests
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class APITool:
    """Base class for API integration tools."""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    def _make_request (self, 
                     method: str, 
                     endpoint: str, 
                     params: Optional[Dict] = None,
                     data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make an API request with error handling."""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.Timeout:
            logger.error (f"Request to {url} timed out")
            return {"error": "Request timed out"}
        
        except requests.exceptions.HTTPError as e:
            logger.error (f"HTTP error: {e}")
            return {"error": f"HTTP {response.status_code}: {response.text}"}
        
        except requests.exceptions.RequestException as e:
            logger.error (f"Request failed: {e}")
            return {"error": str (e)}
\`\`\`

## Weather API Integration

Complete example with OpenWeatherMap API:

\`\`\`python
from tools import tool, ToolCategory
from typing import Dict, Any
import os

class WeatherAPI(APITool):
    """OpenWeatherMap API integration."""
    
    def __init__(self):
        api_key = os.getenv("OPENWEATHER_API_KEY")
        super().__init__(
            api_key=api_key,
            base_url="https://api.openweathermap.org/data/2.5"
        )
    
    def get_current_weather (self, 
                           location: str, 
                           units: str = "metric") -> Dict[str, Any]:
        """
        Get current weather for a location.
        
        Args:
            location: City name, zip code, or coordinates
            units: 'metric', 'imperial', or 'standard'
        
        Returns:
            Weather data including temperature, conditions, humidity, etc.
        """
        # Handle different location formats
        params = {
            "appid": self.api_key,
            "units": units
        }
        
        # Check if location is coordinates
        if "," in location and all (part.replace(".", "").replace("-", "").isdigit() 
                                   for part in location.split(",")):
            lat, lon = location.split(",")
            params["lat"] = lat.strip()
            params["lon"] = lon.strip()
        else:
            params["q"] = location
        
        data = self._make_request("GET", "weather", params=params)
        
        if "error" in data:
            return data
        
        # Transform API response to friendly format
        return self._transform_weather_data (data, units)
    
    def _transform_weather_data (self, data: Dict, units: str) -> Dict[str, Any]:
        """Transform API response to friendly format."""
        temp_unit = "C" if units == "metric" else "F" if units == "imperial" else "K"
        
        return {
            "location": data.get("name", "Unknown"),
            "country": data.get("sys", {}).get("country", ""),
            "temperature": data.get("main", {}).get("temp"),
            "temperature_unit": temp_unit,
            "feels_like": data.get("main", {}).get("feels_like"),
            "conditions": data.get("weather", [{}])[0].get("description", ""),
            "humidity": data.get("main", {}).get("humidity"),
            "wind_speed": data.get("wind", {}).get("speed"),
            "timestamp": data.get("dt")
        }

# Create tool instance
weather_api = WeatherAPI()

@tool(
    description="Get current weather conditions for any location worldwide",
    category=ToolCategory.API_INTEGRATION,
    requires_auth=True
)
def get_weather (location: str, units: str = "metric") -> dict:
    """
    Get current weather for a location.
    
    Args:
        location: City name (e.g., 'London'), city with country ('Paris, FR'),
                 zip code ('94102'), or coordinates ('37.7749,-122.4194')
        units: Temperature unit - 'metric' (Celsius), 'imperial' (Fahrenheit), 
               or 'standard' (Kelvin). Default: metric
    
    Returns:
        Dictionary with weather data or error message
    """
    return weather_api.get_current_weather (location, units)

# Usage
result = get_weather("San Francisco", units="imperial")
print(f"Temperature: {result['temperature']}°{result['temperature_unit']}")
print(f"Conditions: {result['conditions']}")
\`\`\`

## Rate Limiting and Throttling

Handle API rate limits properly:

\`\`\`python
import time
from threading import Lock
from collections import deque
from datetime import datetime, timedelta

class RateLimiter:
    """
    Token bucket rate limiter.
    """
    def __init__(self, max_requests: int, time_window: float):
        """
        Args:
            max_requests: Maximum requests allowed in time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = Lock()
    
    def acquire (self) -> bool:
        """
        Try to acquire permission to make a request.
        Returns True if allowed, False if rate limit exceeded.
        """
        with self.lock:
            now = datetime.now()
            
            # Remove old requests outside the time window
            cutoff = now - timedelta (seconds=self.time_window)
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
            
            # Check if we can make a request
            if len (self.requests) < self.max_requests:
                self.requests.append (now)
                return True
            else:
                return False
    
    def wait_if_needed (self):
        """Wait until a request is allowed."""
        while not self.acquire():
            time.sleep(0.1)

class RateLimitedAPITool(APITool):
    """API tool with rate limiting."""
    
    def __init__(self, api_key: str, base_url: str, 
                 max_requests: int = 60, time_window: float = 60.0):
        super().__init__(api_key, base_url)
        self.rate_limiter = RateLimiter (max_requests, time_window)
    
    def _make_request (self, method: str, endpoint: str, 
                     params: Optional[Dict] = None,
                     data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make request with rate limiting."""
        # Wait if rate limit exceeded
        self.rate_limiter.wait_if_needed()
        
        # Make the actual request
        return super()._make_request (method, endpoint, params, data)
\`\`\`

## Retry Logic with Exponential Backoff

Handle transient failures with retries:

\`\`\`python
import time
import random
from functools import wraps
from typing import Callable, Any

def retry_with_backoff (max_retries: int = 3,
                       base_delay: float = 1.0,
                       max_delay: float = 60.0,
                       exponential_base: float = 2.0,
                       jitter: bool = True):
    """
    Decorator for retrying with exponential backoff.
    """
    def decorator (func: Callable) -> Callable:
        @wraps (func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range (max_retries):
                try:
                    return func(*args, **kwargs)
                
                except requests.exceptions.RequestException as e:
                    last_exception = e
                    
                    if attempt == max_retries - 1:
                        # Last attempt failed
                        raise
                    
                    # Calculate delay
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )
                    
                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay *= (0.5 + random.random())
                    
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    
                    time.sleep (delay)
            
            raise last_exception
        
        return wrapper
    return decorator

class RobustAPITool(RateLimitedAPITool):
    """API tool with rate limiting and retry logic."""
    
    @retry_with_backoff (max_retries=3, base_delay=1.0)
    def _make_request (self, method: str, endpoint: str,
                     params: Optional[Dict] = None,
                     data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make request with retry logic."""
        return super()._make_request (method, endpoint, params, data)
\`\`\`

## Authentication Patterns

### API Key Authentication

\`\`\`python
class APIKeyAuth(APITool):
    """API key in header."""
    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key, base_url)
        self.session.headers.update({
            "X-API-Key": api_key
        })
\`\`\`

### Bearer Token Authentication

\`\`\`python
class BearerTokenAuth(APITool):
    """Bearer token authentication."""
    def __init__(self, token: str, base_url: str):
        super().__init__(token, base_url)
        self.session.headers.update({
            "Authorization": f"Bearer {token}"
        })
\`\`\`

### OAuth 2.0 Authentication

\`\`\`python
from requests_oauthlib import OAuth2Session
from datetime import datetime, timedelta

class OAuth2Tool(APITool):
    """OAuth 2.0 authentication."""
    
    def __init__(self, client_id: str, client_secret: str, 
                 base_url: str, token_url: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = base_url
        self.token_url = token_url
        self.access_token = None
        self.token_expires = None
        
        self._get_access_token()
    
    def _get_access_token (self):
        """Get or refresh access token."""
        oauth = OAuth2Session (client=BackendApplicationClient(
            client_id=self.client_id
        ))
        
        token = oauth.fetch_token(
            token_url=self.token_url,
            client_id=self.client_id,
            client_secret=self.client_secret
        )
        
        self.access_token = token['access_token']
        self.token_expires = datetime.now() + timedelta(
            seconds=token.get('expires_in', 3600)
        )
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.access_token}"
        })
    
    def _make_request (self, method: str, endpoint: str,
                     params: Optional[Dict] = None,
                     data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make request, refreshing token if needed."""
        # Check if token expired
        if datetime.now() >= self.token_expires:
            self._get_access_token()
        
        return super()._make_request (method, endpoint, params, data)
\`\`\`

## Email API Integration (SendGrid)

\`\`\`python
class SendGridAPI(RobustAPITool):
    """SendGrid email API integration."""
    
    def __init__(self):
        api_key = os.getenv("SENDGRID_API_KEY")
        super().__init__(
            api_key=api_key,
            base_url="https://api.sendgrid.com/v3",
            max_requests=100,
            time_window=60.0
        )
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })

sendgrid_api = SendGridAPI()

@tool(
    description="Send an email via SendGrid",
    category=ToolCategory.COMMUNICATION,
    requires_auth=True,
    requires_approval=True
)
def send_email (to: str, subject: str, body: str, 
               from_email: str = "noreply@example.com",
               body_type: str = "text") -> dict:
    """
    Send an email.
    
    Args:
        to: Recipient email address
        subject: Email subject
        body: Email body content
        from_email: Sender email address
        body_type: 'text' or 'html'
    
    Returns:
        Success status and message ID
    """
    data = {
        "personalizations": [{
            "to": [{"email": to}]
        }],
        "from": {"email": from_email},
        "subject": subject,
        "content": [{
            "type": f"text/{body_type}",
            "value": body
        }]
    }
    
    result = sendgrid_api._make_request("POST", "mail/send", data=data)
    
    if "error" not in result:
        return {
            "success": True,
            "message": "Email sent successfully"
        }
    else:
        return {
            "success": False,
            "error": result["error"]
        }
\`\`\`

## Database Query Tool

\`\`\`python
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any

class DatabaseTool:
    """PostgreSQL database query tool."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.conn = None
    
    def _get_connection (self):
        """Get database connection with connection pooling."""
        if self.conn is None or self.conn.closed:
            self.conn = psycopg2.connect(
                self.connection_string,
                cursor_factory=RealDictCursor
            )
        return self.conn
    
    def execute_query (self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """
        Execute a read-only SQL query.
        
        Security: Only SELECT queries allowed.
        """
        # Validate query is read-only
        query_upper = query.strip().upper()
        if not query_upper.startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed")
        
        if any (keyword in query_upper for keyword in ["INSERT", "UPDATE", "DELETE", "DROP"]):
            raise ValueError("Modifying queries are not allowed")
        
        try:
            conn = self._get_connection()
            with conn.cursor() as cursor:
                cursor.execute (query, params)
                results = cursor.fetchall()
                return [dict (row) for row in results]
        
        except Exception as e:
            logger.error (f"Query failed: {e}")
            raise
    
    def close (self):
        """Close database connection."""
        if self.conn and not self.conn.closed:
            self.conn.close()

db_tool = DatabaseTool (os.getenv("DATABASE_URL"))

@tool(
    description="Query the database with SQL",
    category=ToolCategory.DATABASE,
    requires_auth=True
)
def query_database (query: str) -> dict:
    """
    Execute a read-only SQL query against the database.
    
    IMPORTANT: Only SELECT queries are allowed.
    Query timeout is 30 seconds.
    Maximum 1000 rows returned.
    
    Args:
        query: SQL SELECT query
    
    Returns:
        Query results as list of dictionaries
    """
    try:
        results = db_tool.execute_query (query)
        
        # Limit results
        if len (results) > 1000:
            results = results[:1000]
            truncated = True
        else:
            truncated = False
        
        return {
            "success": True,
            "results": results,
            "count": len (results),
            "truncated": truncated
        }
    
    except ValueError as e:
        return {
            "success": False,
            "error": str (e)
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Query failed: {str (e)}"
        }
\`\`\`

## Web Search Tool (Google Custom Search)

\`\`\`python
class GoogleSearchAPI(RobustAPITool):
    """Google Custom Search API integration."""
    
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        self.cx = os.getenv("GOOGLE_CX")  # Custom Search Engine ID
        super().__init__(
            api_key=api_key,
            base_url="https://www.googleapis.com/customsearch/v1",
            max_requests=100,
            time_window=100.0
        )

google_search_api = GoogleSearchAPI()

@tool(
    description="Search the web using Google",
    category=ToolCategory.WEB,
    requires_auth=True
)
def google_search (query: str, num_results: int = 10) -> dict:
    """
    Search the web using Google Custom Search.
    
    Args:
        query: Search query
        num_results: Number of results to return (1-10)
    
    Returns:
        List of search results with title, link, and snippet
    """
    params = {
        "key": google_search_api.api_key,
        "cx": google_search_api.cx,
        "q": query,
        "num": min (num_results, 10)
    }
    
    data = google_search_api._make_request("GET", "", params=params)
    
    if "error" in data:
        return {"success": False, "error": data["error"]}
    
    # Transform results
    results = []
    for item in data.get("items", []):
        results.append({
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("snippet"),
            "displayLink": item.get("displayLink")
        })
    
    return {
        "success": True,
        "query": query,
        "results": results,
        "count": len (results)
    }
\`\`\`

## Calendar Integration (Google Calendar)

\`\`\`python
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from datetime import datetime, timedelta

class GoogleCalendarAPI:
    """Google Calendar API integration."""
    
    def __init__(self, credentials_path: str):
        self.creds = Credentials.from_authorized_user_file (credentials_path)
        self.service = build('calendar', 'v3', credentials=self.creds)
    
    def list_events (self, time_min: str = None, max_results: int = 10) -> List[Dict]:
        """List upcoming calendar events."""
        if time_min is None:
            time_min = datetime.utcnow().isoformat() + 'Z'
        
        events_result = self.service.events().list(
            calendarId='primary',
            timeMin=time_min,
            maxResults=max_results,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        return events_result.get('items', [])
    
    def create_event (self, summary: str, start_time: str, 
                    end_time: str, description: str = "") -> Dict:
        """Create a calendar event."""
        event = {
            'summary': summary,
            'description': description,
            'start': {'dateTime': start_time, 'timeZone': 'UTC'},
            'end': {'dateTime': end_time, 'timeZone': 'UTC'},
        }
        
        event = self.service.events().insert(
            calendarId='primary',
            body=event
        ).execute()
        
        return event

calendar_api = GoogleCalendarAPI('credentials.json')

@tool(
    description="List upcoming calendar events",
    category=ToolCategory.API_INTEGRATION,
    requires_auth=True
)
def list_calendar_events (max_results: int = 10) -> dict:
    """
    List upcoming calendar events.
    
    Args:
        max_results: Maximum number of events to return (1-100)
    
    Returns:
        List of upcoming events with summary, start time, and end time
    """
    try:
        events = calendar_api.list_events (max_results=max_results)
        
        results = []
        for event in events:
            results.append({
                "summary": event.get("summary"),
                "start": event.get("start", {}).get("dateTime"),
                "end": event.get("end", {}).get("dateTime"),
                "id": event.get("id")
            })
        
        return {
            "success": True,
            "events": results,
            "count": len (results)
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str (e)
        }
\`\`\`

## Caching for API Calls

Reduce API costs and latency with caching:

\`\`\`python
import redis
import json
import hashlib
from functools import wraps

class APICache:
    """Redis-based cache for API responses."""
    
    def __init__(self, redis_url: str, default_ttl: int = 3600):
        self.redis_client = redis.from_url (redis_url)
        self.default_ttl = default_ttl
    
    def get (self, key: str) -> Optional[Any]:
        """Get cached value."""
        value = self.redis_client.get (key)
        if value:
            return json.loads (value)
        return None
    
    def set (self, key: str, value: Any, ttl: int = None):
        """Set cached value."""
        if ttl is None:
            ttl = self.default_ttl
        
        self.redis_client.setex(
            key,
            ttl,
            json.dumps (value)
        )
    
    def cache_key (self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function call."""
        key_data = f"{func_name}:{args}:{sorted (kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()

def cached_api_call (cache: APICache, ttl: int = None):
    """Decorator to cache API call results."""
    def decorator (func):
        @wraps (func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache.cache_key (func.__name__, args, kwargs)
            
            # Check cache
            cached_result = cache.get (cache_key)
            if cached_result is not None:
                logger.info (f"Cache hit for {func.__name__}")
                return cached_result
            
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            cache.set (cache_key, result, ttl)
            logger.info (f"Cached result for {func.__name__}")
            
            return result
        
        return wrapper
    return decorator

# Usage
cache = APICache (redis_url="redis://localhost:6379/0", default_ttl=3600)

@cached_api_call (cache, ttl=1800)  # Cache for 30 minutes
def get_weather_cached (location: str, units: str = "metric") -> dict:
    """Get weather with caching."""
    return weather_api.get_current_weather (location, units)
\`\`\`

## Async API Tools

For better performance with many API calls:

\`\`\`python
import aiohttp
import asyncio
from typing import List

class AsyncAPITool:
    """Async API tool base class."""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
    
    async def _make_request (self, method: str, endpoint: str,
                           params: Optional[Dict] = None,
                           data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make async API request."""
        url = f"{self.base_url}/{endpoint}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=headers,
                timeout=aiohttp.ClientTimeout (total=30)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {
                        "error": f"HTTP {response.status}: {await response.text()}"
                    }

# Parallel API calls
async def fetch_multiple_weather (locations: List[str]) -> List[Dict]:
    """Fetch weather for multiple locations in parallel."""
    weather_tool = AsyncWeatherAPI()
    
    tasks = [
        weather_tool.get_current_weather (location)
        for location in locations
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

# Usage
locations = ["San Francisco", "New York", "London", "Tokyo"]
results = asyncio.run (fetch_multiple_weather (locations))
\`\`\`

## Best Practices

1. **Error Handling**: Always handle API errors gracefully
2. **Rate Limiting**: Respect API rate limits
3. **Retries**: Implement exponential backoff for transient failures
4. **Caching**: Cache responses when appropriate
5. **Authentication**: Securely manage API keys and tokens
6. **Timeouts**: Set reasonable timeouts for all requests
7. **Logging**: Log all API calls for debugging
8. **Testing**: Mock API responses for tests
9. **Cost Tracking**: Monitor API usage and costs
10. **Documentation**: Document all tool parameters clearly

## Summary

API integration tools require:
- Robust error handling
- Rate limiting
- Retry logic
- Proper authentication
- Response transformation
- Caching for efficiency
- Async support for performance
- Security considerations

Next, we'll explore building code execution tools like ChatGPT Code Interpreter.
`,
};
