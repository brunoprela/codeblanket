export const djangoSecurityBestPracticesQuiz = {
  title: 'Django Security Best Practices - Discussion Questions',
  questions: [
    {
      question:
        "Explain Django's built-in security features including CSRF protection, SQL injection prevention, XSS protection, and clickjacking defense. How do you configure them properly?",
      answer: `
**CSRF Protection:**

\`\`\`python
# settings.py
MIDDLEWARE = [
    'django.middleware.csrf.CsrfViewMiddleware',  # Enabled by default
]

# In templates
<form method="post">
    {% csrf_token %}
    <!-- form fields -->
</form>

# In DRF
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',  # Requires CSRF
    ]
}

# Exempt specific views (use carefully)
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def webhook_view(request):
    # External webhooks that can't send CSRF token
    pass
\`\`\`

**SQL Injection Prevention:**

\`\`\`python
# SAFE - Django ORM parameterizes queries
Article.objects.filter(author=user_input)
Article.objects.raw('SELECT * FROM articles WHERE id = %s', [user_id])

# UNSAFE - Don't do this!
Article.objects.raw(f'SELECT * FROM articles WHERE id = {user_id}')

# Safe raw SQL with params
from django.db import connection
with connection.cursor() as cursor:
    cursor.execute("SELECT * FROM articles WHERE status = %s", [status])
\`\`\`

**XSS Protection:**

\`\`\`python
# settings.py
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True

# Templates auto-escape by default
{{ user_input }}  # Safe - auto-escaped

# Mark safe only when necessary
from django.utils.safestring import mark_safe
safe_html = mark_safe(trusted_html)

# DRF escapes by default
class ArticleSerializer(serializers.ModelSerializer):
    # All text fields escaped in JSON responses
    pass
\`\`\`

**Clickjacking Defense:**

\`\`\`python
# settings.py
MIDDLEWARE = [
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

X_FRAME_OPTIONS = 'DENY'  # or 'SAMEORIGIN'

# Per-view control
from django.views.decorators.clickjacking import xframe_options_exempt

@xframe_options_exempt
def embeddable_view(request):
    pass
\`\`\`

**Content Security Policy:**

\`\`\`python
# Install django-csp
pip install django-csp

MIDDLEWARE = [
    'csp.middleware.CSPMiddleware',
]

CSP_DEFAULT_SRC = ("'self'",)
CSP_SCRIPT_SRC = ("'self'", 'cdn.example.com')
CSP_STYLE_SRC = ("'self'", "'unsafe-inline'")
\`\`\`

**HTTPS Configuration:**

\`\`\`python
SECURE_SSL_REDIRECT = True  # Redirect HTTP to HTTPS
SECURE_HSTS_SECONDS = 31536000  # 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
\`\`\`
      `,
    },
    {
      question:
        'Describe authentication security in Django including password hashing, session management, and protecting against brute force attacks. Include rate limiting strategies.',
      answer: `
**Password Security:**

\`\`\`python
# settings.py - Password hashing
PASSWORD_HASHERS = [
    'django.contrib.auth.hashers.Argon2PasswordHasher',  # Most secure
    'django.contrib.auth.hashers.PBKDF2PasswordHasher',
    'django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher',
]

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator', 'OPTIONS': {'min_length': 12}},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# Custom validator
from django.core.exceptions import ValidationError

class SpecialCharacterValidator:
    def validate(self, password, user=None):
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            raise ValidationError('Password must contain special character')
\`\`\`

**Session Security:**

\`\`\`python
# settings.py
SESSION_ENGINE = 'django.contrib.sessions.backends.cache'  # Use cache
SESSION_COOKIE_HTTPONLY = True  # No JavaScript access
SESSION_COOKIE_SECURE = True  # HTTPS only
SESSION_COOKIE_SAMESITE = 'Strict'  # CSRF protection
SESSION_COOKIE_AGE = 3600  # 1 hour
SESSION_SAVE_EVERY_REQUEST = True  # Refresh on activity

# Rotate session on login
from django.contrib.auth import login

def login_view(request):
    user = authenticate(username=username, password=password)
    if user:
        login(request, user)  # Creates new session ID
\`\`\`

**Brute Force Protection (django-ratelimit):**

\`\`\`python
pip install django-ratelimit

from django_ratelimit.decorators import ratelimit

@ratelimit(key='ip', rate='5/m', method='POST', block=True)
def login_view(request):
    # Max 5 login attempts per minute per IP
    if request.method == 'POST':
        # Handle login
        pass

# Rate limit API
from rest_framework.decorators import api_view
from django_ratelimit.decorators import ratelimit

@api_view(['POST'])
@ratelimit(key='user_or_ip', rate='10/h', method='POST')
def api_endpoint(request):
    pass
\`\`\`

**DRF Throttling:**

\`\`\`python
REST_FRAMEWORK = {
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle'
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '100/day',
        'user': '1000/day',
        'login': '5/hour',
    }
}

# Custom throttle
from rest_framework.throttling import UserRateThrottle

class LoginThrottle(UserRateThrottle):
    scope = 'login'

class LoginView(APIView):
    throttle_classes = [LoginThrottle]
\`\`\`

**Account Lockout:**

\`\`\`python
from django.core.cache import cache

def check_failed_attempts(username):
    key = f'failed_login:{username}'
    attempts = cache.get(key, 0)
    
    if attempts >= 5:
        raise ValidationError('Account locked. Try again in 30 minutes.')
    
    return attempts

def record_failed_attempt(username):
    key = f'failed_login:{username}'
    attempts = cache.get(key, 0) + 1
    cache.set(key, attempts, timeout=1800)  # 30 min

def clear_failed_attempts(username):
    key = f'failed_login:{username}'
    cache.delete(key)
\`\`\`

**Two-Factor Authentication:**

\`\`\`python
pip install django-otp

INSTALLED_APPS += ['django_otp', 'django_otp.plugins.otp_totp']
MIDDLEWARE += ['django_otp.middleware.OTPMiddleware']

from django_otp.decorators import otp_required

@otp_required
def sensitive_view(request):
    # Requires 2FA
    pass
\`\`\`
      `,
    },
    {
      question:
        'Explain secure API design in DRF including JWT security, CORS configuration, API key management, and preventing common API vulnerabilities.',
      answer: `
**JWT Security:**

\`\`\`python
from datetime import timedelta

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=15),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
    'ALGORITHM': 'HS256',
    'SIGNING_KEY': settings.SECRET_KEY,  # Use strong secret
    'AUTH_HEADER_TYPES': ('Bearer',),
    'USER_ID_FIELD': 'id',
    'USER_ID_CLAIM': 'user_id',
}

# Token blacklisting
INSTALLED_APPS += ['rest_framework_simplejwt.token_blacklist']

from rest_framework_simplejwt.tokens import RefreshToken

def logout_view(request):
    token = RefreshToken(request.data['refresh'])
    token.blacklist()
\`\`\`

**CORS Configuration:**

\`\`\`python
pip install django-cors-headers

INSTALLED_APPS += ['corsheaders']

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    # ...
]

# Production settings
CORS_ALLOWED_ORIGINS = [
    'https://app.example.com',
    'https://www.example.com',
]

CORS_ALLOW_CREDENTIALS = True

CORS_ALLOW_METHODS = [
    'DELETE',
    'GET',
    'OPTIONS',
    'PATCH',
    'POST',
    'PUT',
]

# Development only
CORS_ALLOW_ALL_ORIGINS = False  # NEVER True in production!
\`\`\`

**API Key Management:**

\`\`\`python
import secrets

class APIKey(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    key = models.CharField(max_length=64, unique=True)
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    last_used = models.DateTimeField(null=True)
    
    @classmethod
    def generate_key(cls):
        return secrets.token_urlsafe(48)
    
    def save(self, *args, **kwargs):
        if not self.key:
            self.key = self.generate_key()
        super().save(*args, **kwargs)

# Custom authentication
from rest_framework.authentication import BaseAuthentication

class APIKeyAuthentication(BaseAuthentication):
    def authenticate(self, request):
        api_key = request.META.get('HTTP_X_API_KEY')
        if not api_key:
            return None
        
        try:
            key_obj = APIKey.objects.select_related('user').get(key=api_key)
            key_obj.last_used = timezone.now()
            key_obj.save(update_fields=['last_used'])
            return (key_obj.user, None)
        except APIKey.DoesNotExist:
            raise AuthenticationFailed('Invalid API key')
\`\`\`

**Input Validation:**

\`\`\`python
from rest_framework import serializers

class ArticleSerializer(serializers.ModelSerializer):
    title = serializers.CharField(max_length=200)
    content = serializers.CharField(max_length=50000)
    
    def validate_title(self, value):
        # Sanitize input
        if '<script>' in value.lower():
            raise serializers.ValidationError('Invalid content')
        return value
    
    def validate(self, data):
        # Cross-field validation
        if len(data['content']) < 100:
            raise serializers.ValidationError('Content too short')
        return data
\`\`\`

**Mass Assignment Protection:**

\`\`\`python
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['username', 'email']  # Whitelist only safe fields
        read_only_fields = ['is_staff', 'is_superuser']  # Protect sensitive fields
\`\`\`

**API Versioning:**

\`\`\`python
REST_FRAMEWORK = {
    'DEFAULT_VERSIONING_CLASS': 'rest_framework.versioning.URLPathVersioning',
    'ALLOWED_VERSIONS': ['v1', 'v2'],
    'DEFAULT_VERSION': 'v1',
}

# URLs
urlpatterns = [
    path('api/v1/', include('api.v1.urls')),
    path('api/v2/', include('api.v2.urls')),
]
\`\`\`

**Rate Limiting Per Endpoint:**

\`\`\`python
class SensitiveViewSet(viewsets.ModelViewSet):
    throttle_classes = [UserRateThrottle]
    throttle_scope = 'sensitive'
    
REST_FRAMEWORK = {
    'DEFAULT_THROTTLE_RATES': {
        'sensitive': '10/hour',
    }
}
\`\`\`
      `,
    },
  ],
};
