export const middlewareDevelopmentQuiz = [
  {
    id: 1,
    question:
      'Explain Django middleware architecture, execution order, and the request/response lifecycle. How would you implement middleware for request timing, logging, and error handling? Include both function-based and class-based approaches.',
    answer: `
**Django Middleware Architecture:**

Middleware is a framework of hooks into Django\'s request/response processing. Each middleware component processes requests going in and responses going out.

**Execution Order:**

\`\`\`
Request Flow (Top to Bottom):
SecurityMiddleware → SessionMiddleware → AuthenticationMiddleware → Your View

Response Flow (Bottom to Top):
Your View → AuthenticationMiddleware → SessionMiddleware → SecurityMiddleware
\`\`\`

**Class-Based Middleware (Django 1.10+):**

\`\`\`python
class RequestTimingMiddleware:
    \"\"\"Measure request processing time\"\"\"
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Before view
        import time
        request.start_time = time.time()
        
        # Process request
        response = self.get_response (request)
        
        # After view
        duration = time.time() - request.start_time
        response['X-Request-Duration'] = f'{duration:.3f}s'
        
        return response
    
    def process_exception (self, request, exception):
        \"\"\"Handle exceptions\"\"\"
        duration = time.time() - request.start_time
        logger.error (f'Request failed after {duration:.3f}s: {exception}')
        return None  # Let other middleware handle it
\`\`\`

**Function-Based Middleware (Legacy):**

\`\`\`python
def request_timing_middleware (get_response):
    \"\"\"Function-based middleware\"\"\"
    
    def middleware (request):
        start_time = time.time()
        response = get_response (request)
        duration = time.time() - start_time
        response['X-Request-Duration'] = f'{duration:.3f}s'
        return response
    
    return middleware
\`\`\`

**Comprehensive Logging Middleware:**

\`\`\`python
import logging
import json
from django.utils.deprecation import MiddlewareMixin

logger = logging.getLogger('django.request')

class RequestLoggingMiddleware:
    \"\"\"Log all requests with detailed information\"\"\"
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Log request
        self.log_request (request)
        
        # Process request
        response = self.get_response (request)
        
        # Log response
        self.log_response (request, response)
        
        return response
    
    def log_request (self, request):
        \"\"\"Log incoming request details\"\"\"
        log_data = {
            'method': request.method,
            'path': request.path,
            'user': str (request.user) if request.user.is_authenticated else 'Anonymous',
            'ip': self.get_client_ip (request),
            'user_agent': request.META.get('HTTP_USER_AGENT', ''),
            'query_params': dict (request.GET),
        }
        
        if request.method in ['POST', 'PUT', 'PATCH']:
            log_data['body_size'] = len (request.body)
        
        logger.info (f'Request: {json.dumps (log_data)}')
    
    def log_response (self, request, response):
        \"\"\"Log response details\"\"\"
        duration = getattr (request, 'start_time', 0)
        
        log_data = {
            'method': request.method,
            'path': request.path,
            'status': response.status_code,
            'duration': duration,
            'response_size': len (response.content) if hasattr (response, 'content') else 0,
        }
        
        if response.status_code >= 400:
            logger.warning (f'Response: {json.dumps (log_data)}')
        else:
            logger.info (f'Response: {json.dumps (log_data)}')
    
    def get_client_ip (self, request):
        \"\"\"Extract client IP address\"\"\"
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            return x_forwarded_for.split(',')[0]
        return request.META.get('REMOTE_ADDR')
    
    def process_exception (self, request, exception):
        \"\"\"Log exceptions\"\"\"
        logger.error(
            f'Exception in {request.method} {request.path}: {exception}',
            exc_info=True,
            extra={'request': request}
        )
        return None
\`\`\`

**Error Handling Middleware:**

\`\`\`python
from django.http import JsonResponse
from django.core.exceptions import PermissionDenied, ValidationError
import traceback

class ErrorHandlingMiddleware:
    \"\"\"Centralized error handling\"\"\"
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        return self.get_response (request)
    
    def process_exception (self, request, exception):
        \"\"\"Handle different exception types\"\"\"
        
        # Log exception
        logger.error(
            f'Exception: {type (exception).__name__}: {exception}',
            exc_info=True,
            extra={'request': request}
        )
        
        # Return appropriate response based on exception type
        if isinstance (exception, ValidationError):
            return JsonResponse({
                'error': 'Validation Error',
                'details': exception.messages
            }, status=400)
        
        elif isinstance (exception, PermissionDenied):
            return JsonResponse({
                'error': 'Permission Denied',
                'message': str (exception)
            }, status=403)
        
        elif isinstance (exception, Http404):
            return JsonResponse({
                'error': 'Not Found',
                'message': str (exception)
            }, status=404)
        
        # Generic error handling
        if settings.DEBUG:
            return JsonResponse({
                'error': 'Internal Server Error',
                'message': str (exception),
                'traceback': traceback.format_exc()
            }, status=500)
        else:
            return JsonResponse({
                'error': 'Internal Server Error',
                'message': 'An error occurred processing your request'
            }, status=500)
\`\`\`

**Middleware Hooks:**1. **__init__(get_response)**: One-time configuration
2. **__call__(request)**: Process request and response
3. **process_view (request, view_func, view_args, view_kwargs)**: Before view is called
4. **process_exception (request, exception)**: When view raises exception
5. **process_template_response (request, response)**: After view returns TemplateResponse

**Complete Example with All Hooks:**

\`\`\`python
class ComprehensiveMiddleware:
    
    def __init__(self, get_response):
        self.get_response = get_response
        # One-time setup
    
    def __call__(self, request):
        # Before view
        request.middleware_timestamp = time.time()
        
        response = self.get_response (request)
        
        # After view
        return response
    
    def process_view (self, request, view_func, view_args, view_kwargs):
        \"\"\"Called before view executes\"\"\"
        logger.debug (f'View: {view_func.__name__}')
        return None  # Continue processing
    
    def process_exception (self, request, exception):
        \"\"\"Handle exceptions\"\"\"
        logger.error (f'Exception: {exception}')
        return None  # Let other middleware handle
    
    def process_template_response (self, request, response):
        \"\"\"Modify template responses\"\"\"
        response.context_data['middleware_processed'] = True
        return response
\`\`\`

**Registration (settings.py):**

\`\`\`python
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'myapp.middleware.RequestTimingMiddleware',  # Custom
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'myapp.middleware.RequestLoggingMiddleware',  # Custom
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'myapp.middleware.ErrorHandlingMiddleware',  # Custom
]
\`\`\`

**Best Practices:**1. ✅ Keep middleware lightweight
2. ✅ Order matters - consider dependencies
3. ✅ Use process_exception for error handling
4. ✅ Add custom headers in __call__
5. ✅ Log but don't modify request/response unnecessarily
6. ❌ Don't perform heavy computations
7. ❌ Don't make external API calls in middleware
8. ❌ Don't return HttpResponse from __call__ (unless handling errors)

Middleware is powerful for cross-cutting concerns like logging, authentication, and monitoring.
      `,
  },
  {
    question:
      'Design a rate-limiting middleware system that can handle different limits based on user type, API endpoint, and time windows. Include Redis-based implementation and handling of distributed systems.',
    answer: `
**Comprehensive Rate Limiting System:**

**1. Basic Rate Limiting Middleware:**

\`\`\`python
from django.core.cache import cache
from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page
import time

class RateLimitMiddleware:
    \"\"\"Rate limit requests by IP address\"\"\"
    
    def __init__(self, get_response):
        self.get_response = get_response
        self.rate_limit = 100  # requests
        self.time_window = 60  # seconds
    
    def __call__(self, request):
        # Get client identifier
        client_id = self.get_client_id (request)
        
        # Check rate limit
        if self.is_rate_limited (client_id):
            return JsonResponse({
                'error': 'Rate limit exceeded',
                'retry_after': self.get_retry_after (client_id)
            }, status=429)
        
        # Process request
        response = self.get_response (request)
        
        # Add rate limit headers
        self.add_rate_limit_headers (response, client_id)
        
        return response
    
    def get_client_id (self, request):
        \"\"\"Get unique client identifier\"\"\"
        # Prefer user ID for authenticated users
        if request.user.is_authenticated:
            return f'user:{request.user.id}'
        
        # Fall back to IP address
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        
        return f'ip:{ip}'
    
    def is_rate_limited (self, client_id):
        \"\"\"Check if client has exceeded rate limit\"\"\"
        cache_key = f'rate_limit:{client_id}'
        
        # Get current count
        current_count = cache.get (cache_key, 0)
        
        if current_count >= self.rate_limit:
            return True
        
        # Increment count
        if current_count == 0:
            # First request in window
            cache.set (cache_key, 1, self.time_window)
        else:
            # Increment existing count
            cache.incr (cache_key)
        
        return False
    
    def get_retry_after (self, client_id):
        \"\"\"Get seconds until rate limit resets\"\"\"
        cache_key = f'rate_limit:{client_id}'
        ttl = cache.ttl (cache_key)
        return max (ttl, 0)
    
    def add_rate_limit_headers (self, response, client_id):
        \"\"\"Add rate limit information to response headers\"\"\"
        cache_key = f'rate_limit:{client_id}'
        current_count = cache.get (cache_key, 0)
        
        response['X-RateLimit-Limit'] = str (self.rate_limit)
        response['X-RateLimit-Remaining'] = str (max (self.rate_limit - current_count, 0))
        response['X-RateLimit-Reset'] = str (int (time.time()) + self.get_retry_after (client_id))
\`\`\`

**2. Multi-Tier Rate Limiting:**

\`\`\`python
from enum import Enum

class UserTier(Enum):
    FREE = 'free'
    BASIC = 'basic'
    PREMIUM = 'premium'
    ENTERPRISE = 'enterprise'

class TieredRateLimitMiddleware:
    \"\"\"Different rate limits based on user tier\"\"\"
    
    RATE_LIMITS = {
        UserTier.FREE: {'requests': 100, 'window': 3600},  # 100/hour
        UserTier.BASIC: {'requests': 1000, 'window': 3600},  # 1000/hour
        UserTier.PREMIUM: {'requests': 10000, 'window': 3600},  # 10k/hour
        UserTier.ENTERPRISE: {'requests': 100000, 'window': 3600},  # 100k/hour
    }
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Get user tier
        tier = self.get_user_tier (request)
        limits = self.RATE_LIMITS[tier]
        
        # Check rate limit
        client_id = self.get_client_id (request)
        if self.check_rate_limit (client_id, tier, limits):
            return JsonResponse({
                'error': 'Rate limit exceeded',
                'tier': tier.value,
                'limit': limits['requests'],
                'window': limits['window'],
            }, status=429)
        
        response = self.get_response (request)
        return response
    
    def get_user_tier (self, request):
        \"\"\"Determine user's tier\"\"\"
        if not request.user.is_authenticated:
            return UserTier.FREE
        
        # Assuming User model has subscription relationship
        if hasattr (request.user, 'subscription'):
            tier_name = request.user.subscription.tier
            return UserTier (tier_name)
        
        return UserTier.FREE
    
    def check_rate_limit (self, client_id, tier, limits):
        \"\"\"Check rate limit with tier-specific limits\"\"\"
        cache_key = f'rate_limit:{tier.value}:{client_id}'
        
        current_count = cache.get (cache_key, 0)
        
        if current_count >= limits['requests']:
            return True
        
        if current_count == 0:
            cache.set (cache_key, 1, limits['window'])
        else:
            cache.incr (cache_key)
        
        return False
\`\`\`

**3. Endpoint-Specific Rate Limiting:**

\`\`\`python
class EndpointRateLimitMiddleware:
    \"\"\"Different rate limits per endpoint\"\"\"
    
    ENDPOINT_LIMITS = {
        '/api/search/': {'requests': 10, 'window': 60},  # 10/minute
        '/api/upload/': {'requests': 5, 'window': 300},  # 5 per 5 minutes
        '/api/articles/': {'requests': 100, 'window': 60},  # 100/minute
    }
    
    DEFAULT_LIMIT = {'requests': 60, 'window': 60}  # 60/minute default
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Get endpoint-specific limits
        limits = self.get_endpoint_limits (request.path)
        
        # Check rate limit
        client_id = self.get_client_id (request)
        cache_key = f'rate_limit:{request.path}:{client_id}'
        
        if self.is_rate_limited (cache_key, limits):
            return JsonResponse({
                'error': 'Rate limit exceeded for this endpoint',
                'endpoint': request.path,
                'limit': limits['requests'],
            }, status=429)
        
        return self.get_response (request)
    
    def get_endpoint_limits (self, path):
        \"\"\"Get rate limits for specific endpoint\"\"\"
        # Try exact match
        if path in self.ENDPOINT_LIMITS:
            return self.ENDPOINT_LIMITS[path]
        
        # Try prefix match
        for endpoint_pattern, limits in self.ENDPOINT_LIMITS.items():
            if path.startswith (endpoint_pattern):
                return limits
        
        return self.DEFAULT_LIMIT
\`\`\`

**4. Redis-Based Distributed Rate Limiting:**

\`\`\`python
import redis
from django.conf import settings

class RedisRateLimitMiddleware:
    \"\"\"Production-ready Redis rate limiting with sliding window\"\"\"
    
    def __init__(self, get_response):
        self.get_response = get_response
        self.redis_client = redis.StrictRedis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB
        )
    
    def __call__(self, request):
        client_id = self.get_client_id (request)
        
        # Check rate limit using sliding window
        if self.is_rate_limited_sliding_window (client_id, 100, 60):
            return JsonResponse({
                'error': 'Rate limit exceeded'
            }, status=429)
        
        return self.get_response (request)
    
    def is_rate_limited_sliding_window (self, client_id, max_requests, window_seconds):
        \"\"\"
        Sliding window rate limiting using Redis sorted sets
        More accurate than fixed window approach
        \"\"\"
        key = f'rate_limit:sliding:{client_id}'
        now = time.time()
        window_start = now - window_seconds
        
        # Use Redis pipeline for atomicity
        pipe = self.redis_client.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore (key, 0, window_start)
        
        # Count entries in current window
        pipe.zcard (key)
        
        # Add current request
        pipe.zadd (key, {str (now): now})
        
        # Set expiry
        pipe.expire (key, window_seconds + 1)
        
        results = pipe.execute()
        request_count = results[1]
        
        return request_count >= max_requests
    
    def is_rate_limited_token_bucket (self, client_id, capacity, refill_rate):
        \"\"\"
        Token bucket algorithm - allows bursts
        capacity: maximum tokens
        refill_rate: tokens per second
        \"\"\"
        key = f'rate_limit:bucket:{client_id}'
        now = time.time()
        
        # Get current bucket state
        bucket_data = self.redis_client.get (key)
        
        if bucket_data:
            bucket = json.loads (bucket_data)
            tokens = bucket['tokens']
            last_update = bucket['last_update']
            
            # Refill tokens based on time passed
            time_passed = now - last_update
            tokens = min (capacity, tokens + time_passed * refill_rate)
        else:
            tokens = capacity
        
        # Check if request can proceed
        if tokens < 1:
            return True  # Rate limited
        
        # Consume token
        tokens -= 1
        
        # Save bucket state
        self.redis_client.setex(
            key,
            3600,  # TTL
            json.dumps({'tokens': tokens, 'last_update': now})
        )
        
        return False  # Not rate limited
\`\`\`

**5. Decorator for View-Level Rate Limiting:**

\`\`\`python
from functools import wraps

def rate_limit (requests=60, window=60, key_func=None):
    \"\"\"Decorator for rate limiting specific views\"\"\"
    
    def decorator (view_func):
        @wraps (view_func)
        def wrapped_view (request, *args, **kwargs):
            # Get client identifier
            if key_func:
                client_id = key_func (request)
            else:
                client_id = request.user.id if request.user.is_authenticated else request.META.get('REMOTE_ADDR')
            
            cache_key = f'rate_limit:view:{view_func.__name__}:{client_id}'
            
            current_count = cache.get (cache_key, 0)
            
            if current_count >= requests:
                return JsonResponse({
                    'error': f'Rate limit exceeded for {view_func.__name__}'
                }, status=429)
            
            if current_count == 0:
                cache.set (cache_key, 1, window)
            else:
                cache.incr (cache_key)
            
            return view_func (request, *args, **kwargs)
        
        return wrapped_view
    return decorator

# Usage:
@rate_limit (requests=10, window=60)
def expensive_api_view (request):
    # API logic
    pass
\`\`\`

**Production Considerations:**
- ✅ Use Redis for distributed systems
- ✅ Implement sliding window for accuracy
- ✅ Different limits for different user tiers
- ✅ Different limits for different endpoints
- ✅ Add rate limit headers to responses
- ✅ Handle Redis failures gracefully
- ✅ Monitor rate limit metrics
- ✅ Consider token bucket for burst handling

This comprehensive system provides flexible, scalable rate limiting for production APIs.
      `,
  },
  {
    question:
      'Explain how to implement a tenant isolation middleware for a multi-tenant SaaS application. Include database routing, request context management, and security considerations.',
    answer: `
**Multi-Tenant Isolation Middleware:**

**Architecture Overview:**

Multi-tenancy approaches:
1. **Shared Database, Shared Schema**: tenant_id on all tables
2. **Shared Database, Separate Schema**: Different schemas per tenant
3. **Separate Database**: Each tenant has own database

**1. Request Context Middleware (Shared Schema Approach):**

\`\`\`python
from threading import local
from django.http import Http404
from django.core.cache import cache

_thread_local = local()

class TenantContext:
    \"\"\"Global tenant context\"\"\"
    
    @staticmethod
    def set_tenant (tenant):
        _thread_local.tenant = tenant
    
    @staticmethod
    def get_tenant():
        return getattr(_thread_local, 'tenant', None)
    
    @staticmethod
    def clear():
        _thread_local.tenant = None

class TenantMiddleware:
    \"\"\"Identify and set tenant for each request\"\"\"
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Identify tenant from request
        tenant = self.identify_tenant (request)
        
        if not tenant:
            return JsonResponse({'error': 'Tenant not found'}, status=404)
        
        # Set tenant in thread-local context
        TenantContext.set_tenant (tenant)
        request.tenant = tenant
        
        # Process request
        response = self.get_response (request)
        
        # Clear tenant context
        TenantContext.clear()
        
        return response
    
    def identify_tenant (self, request):
        \"\"\"Identify tenant from subdomain or header\"\"\"
        # Method 1: Subdomain (tenant.example.com)
        host = request.get_host()
        subdomain = host.split('.')[0]
        
        # Try cache first
        cache_key = f'tenant:subdomain:{subdomain}'
        tenant = cache.get (cache_key)
        
        if tenant:
            return tenant
        
        # Query database
        try:
            tenant = Tenant.objects.get (subdomain=subdomain)
            cache.set (cache_key, tenant, 3600)  # Cache for 1 hour
            return tenant
        except Tenant.DoesNotExist:
            pass
        
        # Method 2: Custom header
        tenant_id = request.headers.get('X-Tenant-ID')
        if tenant_id:
            try:
                tenant = Tenant.objects.get (id=tenant_id)
                return tenant
            except Tenant.DoesNotExist:
                pass
        
        # Method 3: API key
        api_key = request.headers.get('Authorization', '').replace('Bearer ', '')
        if api_key:
            try:
                api_key_obj = APIKey.objects.select_related('tenant').get (key=api_key)
                return api_key_obj.tenant
            except APIKey.DoesNotExist:
                pass
        
        return None
\`\`\`

**2. Tenant Models:**

\`\`\`python
from django.db import models
from django.contrib.auth.models import User

class Tenant (models.Model):
    \"\"\"Represents a tenant in the system\"\"\"
    name = models.CharField (max_length=200)
    subdomain = models.CharField (max_length=100, unique=True)
    created_at = models.DateTimeField (auto_now_add=True)
    is_active = models.BooleanField (default=True)
    
    # Subscription info
    plan = models.CharField (max_length=50, default='free')
    max_users = models.IntegerField (default=10)
    
    class Meta:
        indexes = [
            models.Index (fields=['subdomain']),
        ]

class TenantAwareModel (models.Model):
    \"\"\"Base model for all tenant-specific data\"\"\"
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE)
    
    class Meta:
        abstract = True
        
    def save (self, *args, **kwargs):
        # Auto-assign tenant if not set
        if not self.tenant_id:
            self.tenant = TenantContext.get_tenant()
        
        # Validate tenant matches context
        if self.tenant != TenantContext.get_tenant():
            raise ValueError('Tenant mismatch')
        
        super().save(*args, **kwargs)

class Article(TenantAwareModel):
    \"\"\"Example tenant-specific model\"\"\"
    title = models.CharField (max_length=200)
    content = models.TextField()
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    
    class Meta:
        indexes = [
            models.Index (fields=['tenant', 'created_at']),
        ]
\`\`\`

**3. Tenant-Aware Manager:**

\`\`\`python
from django.db import models

class TenantManager (models.Manager):
    \"\"\"Manager that automatically filters by current tenant\"\"\"
    
    def get_queryset (self):
        qs = super().get_queryset()
        tenant = TenantContext.get_tenant()
        
        if tenant:
            return qs.filter (tenant=tenant)
        
        # In admin or management commands, return all
        return qs
    
    def all_tenants (self):
        \"\"\"Explicitly get all tenants' data\"\"\"
        return super().get_queryset()

class Article(TenantAwareModel):
    # ... fields ...
    
    objects = TenantManager()
    all_objects = models.Manager()  # Unfiltered access

# Usage:
Article.objects.all()  # Only current tenant's articles
Article.all_objects.all()  # All tenants' articles
\`\`\`

**4. Database Routing (Separate Database Approach):**

\`\`\`python
class TenantDatabaseRouter:
    \"\"\"Route database operations based on current tenant\"\"\"
    
    def db_for_read (self, model, **hints):
        \"\"\"Route reads to tenant database\"\"\"
        tenant = TenantContext.get_tenant()
        
        if tenant and self.is_tenant_model (model):
            return f'tenant_{tenant.id}'
        
        return 'default'
    
    def db_for_write (self, model, **hints):
        \"\"\"Route writes to tenant database\"\"\"
        tenant = TenantContext.get_tenant()
        
        if tenant and self.is_tenant_model (model):
            return f'tenant_{tenant.id}'
        
        return 'default'
    
    def allow_relation (self, obj1, obj2, **hints):
        \"\"\"Allow relations only within same tenant\"\"\"
        if self.is_tenant_model (obj1.__class__) and self.is_tenant_model (obj2.__class__):
            return obj1.tenant_id == obj2.tenant_id
        return None
    
    def allow_migrate (self, db, app_label, model_name=None, **hints):
        \"\"\"Determine which databases get migrations\"\"\"
        if db == 'default':
            # Global models only
            return not self.is_tenant_app (app_label)
        elif db.startswith('tenant_'):
            # Tenant-specific models
            return self.is_tenant_app (app_label)
        return None
    
    def is_tenant_model (self, model):
        \"\"\"Check if model is tenant-specific\"\"\"
        return issubclass (model, TenantAwareModel)
    
    def is_tenant_app (self, app_label):
        \"\"\"Check if app is tenant-specific\"\"\"
        return app_label in ['articles', 'products']  # Your tenant apps

# settings.py
DATABASE_ROUTERS = ['core.routers.TenantDatabaseRouter']
\`\`\`

**5. Security Enforcement:**

\`\`\`python
class TenantSecurityMiddleware:
    \"\"\"Enforce tenant isolation security\"\"\"
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Ensure user belongs to tenant
        if request.user.is_authenticated:
            if not self.user_belongs_to_tenant (request.user, request.tenant):
                return JsonResponse({
                    'error': 'Access denied: User does not belong to this tenant'
                }, status=403)
        
        response = self.get_response (request)
        
        return response
    
    def user_belongs_to_tenant (self, user, tenant):
        \"\"\"Verify user belongs to tenant\"\"\"
        return user.tenant_memberships.filter (tenant=tenant).exists()

class TenantUserMembership (models.Model):
    \"\"\"Link users to tenants\"\"\"
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='tenant_memberships')
    tenant = models.ForeignKey(Tenant, on_delete=models.CASCADE)
    role = models.CharField (max_length=50)
    is_active = models.BooleanField (default=True)
    
    class Meta:
        unique_together = ['user', 'tenant']
\`\`\`

**6. API View Tenant Enforcement:**

\`\`\`python
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated

class TenantModelViewSet (viewsets.ModelViewSet):
    \"\"\"Base viewset that enforces tenant isolation\"\"\"
    permission_classes = [IsAuthenticated]
    
    def get_queryset (self):
        \"\"\"Filter queryset by current tenant\"\"\"
        qs = super().get_queryset()
        return qs.filter (tenant=self.request.tenant)
    
    def perform_create (self, serializer):
        \"\"\"Auto-assign tenant on create\"\"\"
        serializer.save (tenant=self.request.tenant)

class ArticleViewSet(TenantModelViewSet):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
\`\`\`

**7. Admin Integration:**

\`\`\`python
from django.contrib import admin

class TenantAdmin (admin.ModelAdmin):
    \"\"\"Admin that shows only current tenant's data\"\"\"
    
    def get_queryset (self, request):
        qs = super().get_queryset (request)
        
        if request.user.is_superuser:
            return qs  # Superuser sees all
        
        # Filter by user's tenants
        tenant_ids = request.user.tenant_memberships.values_list('tenant_id', flat=True)
        return qs.filter (tenant_id__in=tenant_ids)
    
    def save_model (self, request, obj, form, change):
        if not change:  # New object
            obj.tenant = TenantContext.get_tenant()
        super().save_model (request, obj, form, change)
\`\`\`

**Security Checklist:**
- ✅ Always validate tenant from request
- ✅ Filter all queries by tenant
- ✅ Verify user belongs to tenant
- ✅ Use database-level constraints
- ✅ Audit cross-tenant access attempts
- ✅ Test tenant isolation thoroughly
- ✅ Use row-level security if supported by database
- ❌ Never trust client-provided tenant ID without validation
- ❌ Never allow cross-tenant object references
- ❌ Never bypass tenant filtering in queries

This architecture provides secure, scalable multi-tenancy for SaaS applications.
      `,
  },
].map(({ id: _id, ...q }, idx) => ({
  id: `django-q-${idx + 1}`,
  question: q.question,
  sampleAnswer: String(q.answer),
  keyPoints: [],
}));
