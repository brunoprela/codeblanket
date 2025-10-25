export const middlewareDevelopment = {
  title: 'Middleware Development',
  id: 'middleware-development',
  content: `
# Middleware Development

## Introduction

Django **middleware** is a framework of hooks into Django's request/response processing. It's a light, low-level plugin system for globally altering Django's input or output. Each middleware component is responsible for doing some specific function.

### What is Middleware?

Middleware is code that runs **before** the view (on the request) and **after** the view (on the response). Think of it as a series of layers that wrap your Django application:

\`\`\`
Request → [MW1 → MW2 → MW3] → View → [MW3 → MW2 → MW1] → Response
\`\`\`

### Common Use Cases

- **Authentication**: Attach user to request
- **Logging**: Log all requests/responses
- **Rate limiting**: Throttle requests
- **CORS headers**: Add cross-origin headers
- **Request timing**: Measure request duration
- **Security**: Add security headers
- **Session management**: Handle sessions
- **Custom headers**: Add/modify headers

By the end of this section, you'll understand:
- How middleware works in Django
- Creating custom middleware
- Middleware ordering and execution
- Production middleware patterns
- Common pitfalls and best practices

---

## How Middleware Works

### Middleware Execution Flow

\`\`\`python
# settings.py
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',           # 1
    'django.contrib.sessions.middleware.SessionMiddleware',    # 2
    'django.middleware.common.CommonMiddleware',               # 3
    'django.middleware.csrf.CsrfViewMiddleware',              # 4
    'django.contrib.auth.middleware.AuthenticationMiddleware', # 5
    'django.contrib.messages.middleware.MessageMiddleware',    # 6
    'django.middleware.clickjacking.XFrameOptionsMiddleware',  # 7
]

# Request flow: 1 → 2 → 3 → 4 → 5 → 6 → 7 → View
# Response flow: View → 7 → 6 → 5 → 4 → 3 → 2 → 1
\`\`\`

### Middleware Methods

Each middleware can implement these methods:

1. **\`__init__(get_response)\`**: Called once when server starts
2. **\`__call__(request)\`**: Called on every request
3. **\`process_view(request, view_func, view_args, view_kwargs)\`**: Called before view
4. **\`process_exception(request, exception)\`**: Called if view raises exception
5. **\`process_template_response(request, response)\`**: Called if response has \`render()\` method

---

## Creating Custom Middleware

### Simple Middleware (New Style - Django 1.10+)

\`\`\`python
# middleware.py
class SimpleMiddleware:
    """
    Simple middleware that adds custom header
    """
    
    def __init__(self, get_response):
        """
        One-time configuration and initialization.
        get_response: callable to get response from next middleware or view
        """
        self.get_response = get_response
        # Initialization code here (runs once at startup)
        print("SimpleMiddleware initialized")
    
    def __call__(self, request):
        """
        Code to be executed for each request before the view is called.
        """
        
        # Code that runs BEFORE the view
        print(f"Processing request: {request.path}")
        
        # Get response from next middleware or view
        response = self.get_response(request)
        
        # Code that runs AFTER the view
        response['X-Custom-Header'] = 'My Custom Value'
        print(f"Processed response for: {request.path}")
        
        return response
\`\`\`

### Registration

\`\`\`python
# settings.py
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'myapp.middleware.SimpleMiddleware',  # Add your middleware
    'django.contrib.sessions.middleware.SessionMiddleware',
    # ... other middleware
]
\`\`\`

---

## Production Middleware Examples

### 1. Request Timing Middleware

\`\`\`python
import time
import logging
from django.utils.deprecation import MiddlewareMixin

logger = logging.getLogger(__name__)

class RequestTimingMiddleware:
    """
    Measure and log request processing time
    Add timing header to response
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Start timer
        start_time = time.time()
        
        # Store start time on request for access in views
        request.start_time = start_time
        
        # Process request
        response = self.get_response(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Add header with timing
        response['X-Request-Duration'] = f'{duration:.3f}s'
        
        # Log slow requests
        if duration > 1.0:
            logger.warning(
                f'Slow request: {request.method} {request.path} took {duration:.3f}s'
            )
        
        # Log all requests
        logger.info(
            f'{request.method} {request.path} - '
            f'{response.status_code} - {duration:.3f}s'
        )
        
        return response
\`\`\`

### 2. Authentication Tracking Middleware

\`\`\`python
from django.utils import timezone
from django.contrib.auth.signals import user_logged_in

class LastActivityMiddleware:
    """
    Update user's last activity timestamp
    Useful for showing "last seen" or session management
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        response = self.get_response(request)
        
        # Update last activity for authenticated users
        if request.user.is_authenticated:
            # Use update() to avoid triggering save signals
            request.user.__class__.objects.filter(
                pk=request.user.pk
            ).update(
                last_activity=timezone.now()
            )
        
        return response

# Add field to User model
from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    last_activity = models.DateTimeField(null=True, blank=True)
    
    @property
    def is_online(self):
        """User is online if active in last 5 minutes"""
        if not self.last_activity:
            return False
        return timezone.now() - self.last_activity < timezone.timedelta(minutes=5)
\`\`\`

### 3. Rate Limiting Middleware

\`\`\`python
from django.core.cache import cache
from django.http import HttpResponse
from django.conf import settings
import hashlib

class RateLimitMiddleware:
    """
    Rate limit requests per IP address
    Uses Redis/cache backend for tracking
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        self.rate_limit = getattr(settings, 'RATE_LIMIT_PER_MINUTE', 60)
    
    def __call__(self, request):
        # Get client IP
        ip_address = self.get_client_ip(request)
        
        # Create cache key
        cache_key = f'rate_limit:{ip_address}'
        
        # Get current request count
        request_count = cache.get(cache_key, 0)
        
        # Check if rate limit exceeded
        if request_count >= self.rate_limit:
            return HttpResponse(
                'Rate limit exceeded. Please try again later.',
                status=429,
                headers={
                    'Retry-After': '60',
                    'X-RateLimit-Limit': str(self.rate_limit),
                    'X-RateLimit-Remaining': '0',
                }
            )
        
        # Increment counter
        cache.set(cache_key, request_count + 1, 60)  # Expire after 60 seconds
        
        # Process request
        response = self.get_response(request)
        
        # Add rate limit headers
        remaining = max(0, self.rate_limit - request_count - 1)
        response['X-RateLimit-Limit'] = str(self.rate_limit)
        response['X-RateLimit-Remaining'] = str(remaining)
        response['X-RateLimit-Reset'] = '60'
        
        return response
    
    def get_client_ip(self, request):
        """Get client IP address from request"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip
\`\`\`

### 4. CORS Middleware (Custom Implementation)

\`\`\`python
from django.conf import settings

class CORSMiddleware:
    """
    Add CORS headers to responses
    Production alternative to django-cors-headers package
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        self.allowed_origins = getattr(
            settings, 
            'CORS_ALLOWED_ORIGINS', 
            ['http://localhost:3000']
        )
        self.allow_credentials = getattr(settings, 'CORS_ALLOW_CREDENTIALS', True)
        self.allowed_methods = getattr(
            settings,
            'CORS_ALLOWED_METHODS',
            ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS']
        )
        self.allowed_headers = getattr(
            settings,
            'CORS_ALLOWED_HEADERS',
            ['Accept', 'Authorization', 'Content-Type', 'X-CSRFToken']
        )
    
    def __call__(self, request):
        # Get origin from request
        origin = request.META.get('HTTP_ORIGIN')
        
        # Handle preflight OPTIONS request
        if request.method == 'OPTIONS':
            response = HttpResponse()
            response.status_code = 200
        else:
            response = self.get_response(request)
        
        # Add CORS headers if origin is allowed
        if origin in self.allowed_origins or '*' in self.allowed_origins:
            response['Access-Control-Allow-Origin'] = origin
            response['Access-Control-Allow-Methods'] = ', '.join(self.allowed_methods)
            response['Access-Control-Allow-Headers'] = ', '.join(self.allowed_headers)
            response['Access-Control-Max-Age'] = '86400'  # 24 hours
            
            if self.allow_credentials:
                response['Access-Control-Allow-Credentials'] = 'true'
        
        return response
\`\`\`

### 5. Request/Response Logging Middleware

\`\`\`python
import json
import logging
from django.utils import timezone

logger = logging.getLogger(__name__)

class RequestResponseLoggingMiddleware:
    """
    Log all requests and responses for auditing
    Useful for debugging and compliance
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        self.excluded_paths = ['/health/', '/metrics/', '/static/', '/media/']
    
    def __call__(self, request):
        # Skip logging for certain paths
        if any(request.path.startswith(path) for path in self.excluded_paths):
            return self.get_response(request)
        
        # Log request
        request_log = self.log_request(request)
        
        # Process request
        response = self.get_response(request)
        
        # Log response
        self.log_response(request, response, request_log)
        
        return response
    
    def log_request(self, request):
        """Log incoming request"""
        request_data = {
            'timestamp': timezone.now().isoformat(),
            'method': request.method,
            'path': request.path,
            'query_params': dict(request.GET),
            'user': request.user.username if request.user.is_authenticated else 'anonymous',
            'ip': self.get_client_ip(request),
            'user_agent': request.META.get('HTTP_USER_AGENT', ''),
        }
        
        # Log request body for POST/PUT/PATCH
        if request.method in ['POST', 'PUT', 'PATCH']:
            try:
                if request.content_type == 'application/json':
                    request_data['body'] = json.loads(request.body)
                else:
                    request_data['body'] = dict(request.POST)
            except:
                request_data['body'] = 'Unable to parse'
        
        logger.info(f"REQUEST: {json.dumps(request_data)}")
        return request_data
    
    def log_response(self, request, response, request_log):
        """Log outgoing response"""
        duration = (timezone.now() - 
                   timezone.datetime.fromisoformat(request_log['timestamp'])).total_seconds()
        
        response_data = {
            'timestamp': timezone.now().isoformat(),
            'status_code': response.status_code,
            'duration_seconds': duration,
            'method': request.method,
            'path': request.path,
        }
        
        logger.info(f"RESPONSE: {json.dumps(response_data)}")
    
    def get_client_ip(self, request):
        """Get client IP address"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip
\`\`\`

### 6. Security Headers Middleware

\`\`\`python
class SecurityHeadersMiddleware:
    """
    Add security headers to all responses
    Complements Django's SecurityMiddleware
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        response = self.get_response(request)
        
        # Content Security Policy
        response['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.example.com; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' https://fonts.gstatic.com; "
            "connect-src 'self' https://api.example.com;"
        )
        
        # Prevent MIME type sniffing
        response['X-Content-Type-Options'] = 'nosniff'
        
        # Enable browser XSS protection
        response['X-XSS-Protection'] = '1; mode=block'
        
        # Referrer policy
        response['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # Permissions policy (formerly Feature-Policy)
        response['Permissions-Policy'] = (
            'geolocation=(), '
            'microphone=(), '
            'camera=()'
        )
        
        # Strict Transport Security (if HTTPS)
        if request.is_secure():
            response['Strict-Transport-Security'] = (
                'max-age=31536000; includeSubDomains; preload'
            )
        
        return response
\`\`\`

### 7. Database Connection Management Middleware

\`\`\`python
from django.db import connection
import logging

logger = logging.getLogger(__name__)

class DatabaseConnectionMiddleware:
    """
    Monitor and manage database connections
    Log slow queries and connection issues
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        self.slow_query_threshold = 1.0  # seconds
    
    def __call__(self, request):
        # Reset query log
        connection.queries_log.clear()
        
        # Process request
        response = self.get_response(request)
        
        # Analyze queries
        total_queries = len(connection.queries)
        total_time = sum(float(q['time']) for q in connection.queries)
        
        # Log query statistics
        if total_queries > 10:
            logger.warning(
                f'High query count: {request.path} executed {total_queries} queries '
                f'in {total_time:.3f}s'
            )
        
        # Log slow queries
        for query in connection.queries:
            query_time = float(query['time'])
            if query_time > self.slow_query_threshold:
                logger.warning(
                    f'Slow query ({query_time:.3f}s): {query["sql"][:200]}'
                )
        
        # Add query stats to response headers (dev only)
        if settings.DEBUG:
            response['X-DB-Query-Count'] = str(total_queries)
            response['X-DB-Query-Time'] = f'{total_time:.3f}s'
        
        return response
\`\`\`

### 8. Maintenance Mode Middleware

\`\`\`python
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse

class MaintenanceModeMiddleware:
    """
    Display maintenance page when MAINTENANCE_MODE is enabled
    Allow staff users to access site
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        self.maintenance_mode = getattr(settings, 'MAINTENANCE_MODE', False)
        self.allowed_ips = getattr(settings, 'MAINTENANCE_ALLOWED_IPS', [])
        self.excluded_paths = ['/admin/', '/health/']
    
    def __call__(self, request):
        # Check if maintenance mode is enabled
        if not self.maintenance_mode:
            return self.get_response(request)
        
        # Allow staff users
        if request.user.is_authenticated and request.user.is_staff:
            return self.get_response(request)
        
        # Allow certain paths
        if any(request.path.startswith(path) for path in self.excluded_paths):
            return self.get_response(request)
        
        # Allow certain IPs
        client_ip = self.get_client_ip(request)
        if client_ip in self.allowed_ips:
            return self.get_response(request)
        
        # Return maintenance page
        return HttpResponse(
            render(request, 'maintenance.html', {
                'message': 'Site is currently under maintenance. Please check back soon.'
            }),
            status=503
        )
    
    def get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip
\`\`\`

---

## Middleware with process_view

\`\`\`python
class ViewLoggingMiddleware:
    """
    Log which view function handles each request
    Useful for debugging routing issues
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        return self.get_response(request)
    
    def process_view(self, request, view_func, view_args, view_kwargs):
        """
        Called just before Django calls the view
        """
        logger.info(
            f"View: {view_func.__module__}.{view_func.__name__} "
            f"args={view_args} kwargs={view_kwargs}"
        )
        
        # Return None to continue processing
        # Return HttpResponse to short-circuit and skip view
        return None
\`\`\`

---

## Middleware with Exception Handling

\`\`\`python
import traceback
from django.http import JsonResponse
from django.conf import settings

class ExceptionHandlingMiddleware:
    """
    Catch and handle exceptions globally
    Return JSON error responses for API
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        return self.get_response(request)
    
    def process_exception(self, request, exception):
        """
        Called when a view raises an exception
        """
        # Log the exception
        logger.error(
            f'Exception in {request.path}: {str(exception)}',
            exc_info=True
        )
        
        # Return JSON response for API requests
        if request.path.startswith('/api/'):
            error_data = {
                'error': type(exception).__name__,
                'message': str(exception),
            }
            
            # Include traceback in debug mode
            if settings.DEBUG:
                error_data['traceback'] = traceback.format_exc()
            
            return JsonResponse(error_data, status=500)
        
        # Return None to let Django handle it normally
        return None
\`\`\`

---

## Middleware Ordering Best Practices

### Correct Order Matters!

\`\`\`python
MIDDLEWARE = [
    # 1. Security first (HTTPS redirect, security headers)
    'django.middleware.security.SecurityMiddleware',
    
    # 2. Session management (needed by auth)
    'django.contrib.sessions.middleware.SessionMiddleware',
    
    # 3. CORS (before CommonMiddleware for OPTIONS requests)
    'corsheaders.middleware.CorsMiddleware',
    
    # 4. Common functionality (URL normalization, etc.)
    'django.middleware.common.CommonMiddleware',
    
    # 5. CSRF protection (needs sessions)
    'django.middleware.csrf.CsrfViewMiddleware',
    
    # 6. Authentication (needs sessions and CSRF)
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    
    # 7. Messages (needs sessions and auth)
    'django.contrib.messages.middleware.MessageMiddleware',
    
    # 8. Clickjacking protection
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    
    # 9. Custom middleware that needs auth
    'myapp.middleware.LastActivityMiddleware',
    
    # 10. Rate limiting (after auth to allow user-specific limits)
    'myapp.middleware.RateLimitMiddleware',
    
    # 11. Request logging (last for accurate timing)
    'myapp.middleware.RequestTimingMiddleware',
]
\`\`\`

---

## Testing Middleware

\`\`\`python
from django.test import TestCase, RequestFactory
from django.contrib.auth.models import User
from myapp.middleware import RequestTimingMiddleware

class RequestTimingMiddlewareTest(TestCase):
    def setUp(self):
        self.factory = RequestFactory()
        self.middleware = RequestTimingMiddleware(self.get_response)
    
    def get_response(self, request):
        """Mock view function"""
        from django.http import HttpResponse
        return HttpResponse('OK')
    
    def test_adds_timing_header(self):
        """Test that middleware adds timing header"""
        request = self.factory.get('/test/')
        response = self.middleware(request)
        
        self.assertIn('X-Request-Duration', response)
        self.assertTrue(response['X-Request-Duration'].endswith('s'))
    
    def test_timing_accuracy(self):
        """Test that timing is accurate"""
        import time
        
        def slow_response(request):
            time.sleep(0.1)  # Simulate slow view
            return HttpResponse('OK')
        
        middleware = RequestTimingMiddleware(slow_response)
        request = self.factory.get('/test/')
        response = middleware(request)
        
        duration = float(response['X-Request-Duration'].rstrip('s'))
        self.assertGreater(duration, 0.1)
        self.assertLess(duration, 0.2)
\`\`\`

---

## Common Pitfalls

### ❌ Anti-Pattern 1: Blocking Operations

\`\`\`python
# ❌ BAD: Slow operation blocks every request
class BadMiddleware:
    def __call__(self, request):
        # This blocks EVERY request for 1 second!
        time.sleep(1)
        return self.get_response(request)

# ✅ GOOD: Use async operations or move to Celery
class GoodMiddleware:
    def __call__(self, request):
        # Queue slow operation
        slow_task.delay(request.user.id)
        return self.get_response(request)
\`\`\`

### ❌ Anti-Pattern 2: Modifying Request/Response Incorrectly

\`\`\`python
# ❌ BAD: Modifying request attributes that shouldn't be modified
class BadMiddleware:
    def __call__(self, request):
        request.method = 'POST'  # Don't do this!
        return self.get_response(request)

# ✅ GOOD: Add custom attributes safely
class GoodMiddleware:
    def __call__(self, request):
        request.custom_data = {'key': 'value'}
        return self.get_response(request)
\`\`\`

### ❌ Anti-Pattern 3: Not Handling Exceptions

\`\`\`python
# ❌ BAD: Exception breaks middleware chain
class BadMiddleware:
    def __call__(self, request):
        data = risky_operation()  # May raise exception
        return self.get_response(request)

# ✅ GOOD: Handle exceptions gracefully
class GoodMiddleware:
    def __call__(self, request):
        try:
            data = risky_operation()
        except Exception as e:
            logger.error(f'Error in middleware: {e}')
            data = None
        return self.get_response(request)
\`\`\`

---

## Production Deployment Checklist

**Before deploying middleware:**

1. ✅ **Test thoroughly**: Unit tests + integration tests
2. ✅ **Performance impact**: Measure overhead
3. ✅ **Exception handling**: Graceful failure
4. ✅ **Logging**: Appropriate log levels
5. ✅ **Ordering**: Correct position in MIDDLEWARE
6. ✅ **Configuration**: Environment-specific settings
7. ✅ **Monitoring**: Add metrics for critical middleware
8. ✅ **Documentation**: Document purpose and configuration

---

## Summary

**Key Takeaways:**

- **Middleware wraps your Django application** with reusable functionality
- **Execution order matters**: Request (top→bottom), Response (bottom→top)
- **Keep middleware fast**: Avoid blocking operations
- **Use for cross-cutting concerns**: Auth, logging, security, rate limiting
- **Test thoroughly**: Middleware affects every request

**Common Middleware Patterns:**
1. Request timing and monitoring
2. Authentication and authorization
3. Rate limiting and throttling
4. CORS handling
5. Security headers
6. Logging and auditing
7. Database connection management
8. Maintenance mode

**Best Practices:**
- ✅ Keep middleware simple and focused
- ✅ Handle exceptions gracefully
- ✅ Log appropriately (not too verbose)
- ✅ Use caching for expensive operations
- ✅ Test middleware in isolation
- ✅ Document configuration options
- ✅ Monitor performance impact

Middleware is powerful for implementing cross-cutting concerns, but should be used judiciously. For request-specific logic, consider decorators or view mixins instead.
`,
};
