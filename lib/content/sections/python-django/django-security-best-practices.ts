export const djangoSecurityBestPractices = {
  title: 'Django Security Best Practices',
  id: 'django-security-best-practices',
  content: `
# Django Security Best Practices

## Introduction

**Security** is paramount in web applications. Django provides excellent built-in security features, but developers must configure and use them correctly to protect against common vulnerabilities.

### Security Priorities

- **Data Protection**: Prevent unauthorized access
- **Input Validation**: Stop injection attacks
- **Authentication**: Verify user identity
- **Authorization**: Control access to resources
- **Encryption**: Protect sensitive data

**Real-World Consequences:**
- Equifax breach: 147M records exposed
- Capital One breach: 100M customers affected
- Target breach: 41M payment cards stolen

By the end of this section, you'll master:
- CSRF and XSS protection
- SQL injection prevention
- Secure authentication
- HTTPS and security headers
- Sensitive data handling
- Security auditing

---

## Django\'s Security Features

### SECRET_KEY Protection

\`\`\`python
# settings.py
# ❌ NEVER commit secret key to version control
SECRET_KEY = 'django-insecure-hardcoded-key'  # BAD

# ✅ Use environment variables
import os
from django.core.exceptions import ImproperlyConfigured

def get_env_variable (var_name):
    try:
        return os.environ[var_name]
    except KeyError:
        error_msg = f'Set the {var_name} environment variable'
        raise ImproperlyConfigured (error_msg)

SECRET_KEY = get_env_variable('DJANGO_SECRET_KEY')

# Or use python-decouple
from decouple import config
SECRET_KEY = config('SECRET_KEY')
\`\`\`

### DEBUG Mode

\`\`\`python
# settings.py
# ❌ NEVER enable DEBUG in production
DEBUG = False

# ✅ Use environment-based configuration
DEBUG = config('DEBUG', default=False, cast=bool)

# ✅ Proper error handling in production
if not DEBUG:
    ALLOWED_HOSTS = ['yourdomain.com', 'www.yourdomain.com']
else:
    ALLOWED_HOSTS = ['localhost', '127.0.0.1']
\`\`\`

---

## CSRF Protection

### How CSRF Works

Django automatically protects against Cross-Site Request Forgery by:
1. Generating a CSRF token for each session
2. Including token in forms
3. Validating token on POST requests

### Form Protection

\`\`\`django
<!-- In templates -->
<form method="post">
    {% csrf_token %}
    {{ form.as_p }}
    <button type="submit">Submit</button>
</form>
\`\`\`

### AJAX Protection

\`\`\`javascript
// Get CSRF token from cookie
function getCookie (name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent (cookie.substring (name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

const csrftoken = getCookie('csrftoken');

// Include in AJAX requests
fetch('/api/articles/', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrftoken,
    },
    body: JSON.stringify (data)
});
\`\`\`

### DRF CSRF

\`\`\`python
# For session-based authentication in DRF
from rest_framework.authentication import SessionAuthentication

class CsrfExemptSessionAuthentication(SessionAuthentication):
    """Session auth without CSRF for APIs"""
    
    def enforce_csrf (self, request):
        return  # Skip CSRF check

# In views
class ArticleViewSet (viewsets.ModelViewSet):
    authentication_classes = [CsrfExemptSessionAuthentication]
    # Or use TokenAuthentication, JWTAuthentication
\`\`\`

---

## XSS (Cross-Site Scripting) Protection

### Template Auto-Escaping

\`\`\`django
<!-- Django automatically escapes by default -->
<p>{{ user_input }}</p>  <!-- Safe: HTML escaped -->

<!-- Explicitly mark as safe (use carefully) -->
<p>{{ trusted_html|safe }}</p>

<!-- For JavaScript context -->
<script>
    var userName = "{{ user.name|escapejs }}";
</script>
\`\`\`

### Preventing XSS in Code

\`\`\`python
from django.utils.html import escape, format_html

# ❌ Dangerous
def dangerous_view (request):
    user_input = request.GET.get('name', '')
    html = f'<p>Hello {user_input}</p>'  # XSS vulnerable
    return HttpResponse (html)

# ✅ Safe
def safe_view (request):
    user_input = request.GET.get('name', '')
    html = format_html('<p>Hello {}</p>', user_input)  # Automatically escaped
    return HttpResponse (html)
\`\`\`

---

## SQL Injection Prevention

### Using Django ORM

\`\`\`python
# ✅ Safe: Django ORM automatically escapes
articles = Article.objects.filter (title=user_input)

# ❌ NEVER do this
from django.db import connection
cursor = connection.cursor()
cursor.execute (f"SELECT * FROM articles WHERE title = '{user_input}'")  # SQL injection!

# ✅ If you must use raw SQL, use parameters
cursor.execute("SELECT * FROM articles WHERE title = %s", [user_input])

# ✅ Or use ORM's raw() with parameters
articles = Article.objects.raw("SELECT * FROM articles WHERE title = %s", [user_input])
\`\`\`

### Safe Query Building

\`\`\`python
from django.db.models import Q

# ✅ Safe: Build complex queries with ORM
def search_articles (query):
    return Article.objects.filter(
        Q(title__icontains=query) | Q(content__icontains=query)
    )

# ❌ Dangerous
def unsafe_search (query):
    return Article.objects.extra(
        where=[f"title LIKE '%{query}%'"]  # SQL injection!
    )

# ✅ Safe with extra()
def safe_search (query):
    return Article.objects.extra(
        where=["title LIKE %s"],
        params=[f'%{query}%']
    )
\`\`\`

---

## Authentication Security

### Password Requirements

\`\`\`python
# settings.py
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
        'OPTIONS': {
            'min_length': 12,  # Minimum 12 characters
        }
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]
\`\`\`

### Secure Login

\`\`\`python
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_protect

@csrf_protect
@never_cache
def secure_login (request):
    """Secure login view"""
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate (request, username=username, password=password)
        
        if user is not None:
            login (request, user)
            return redirect('dashboard')
        else:
            # Don't reveal whether username or password was wrong
            return render (request, 'login.html', {
                'error': 'Invalid credentials'
            })
    
    return render (request, 'login.html')
\`\`\`

### Rate Limiting

\`\`\`python
# Install django-ratelimit
# pip install django-ratelimit

from django_ratelimit.decorators import ratelimit

@ratelimit (key='ip', rate='5/m', method='POST', block=True)
def login_view (request):
    """Login with rate limiting (5 attempts per minute)"""
    # Login logic
    pass

@ratelimit (key='user', rate='100/h')
def api_endpoint (request):
    """API with user-based rate limiting"""
    # API logic
    pass
\`\`\`

---

## HTTPS and Security Headers

### Force HTTPS

\`\`\`python
# settings.py
# Redirect all HTTP to HTTPS
SECURE_SSL_REDIRECT = True

# Require HTTPS for session cookies
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True

# HSTS (HTTP Strict Transport Security)
SECURE_HSTS_SECONDS = 31536000  # 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
\`\`\`

### Security Headers

\`\`\`python
# settings.py
# Prevent browsers from guessing content type
SECURE_CONTENT_TYPE_NOSNIFF = True

# Enable XSS protection in browsers
SECURE_BROWSER_XSS_FILTER = True

# Prevent clickjacking
X_FRAME_OPTIONS = 'DENY'  # Or 'SAMEORIGIN'

# Referrer policy
SECURE_REFERRER_POLICY = 'same-origin'
\`\`\`

### Custom Security Middleware

\`\`\`python
# middleware.py
class SecurityHeadersMiddleware:
    """Add custom security headers"""
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        response = self.get_response (request)
        
        # Content Security Policy
        response['Content-Security-Policy'] = "default-src 'self'"
        
        # Permissions Policy
        response['Permissions-Policy'] = "geolocation=(), microphone=()"
        
        return response

# settings.py
MIDDLEWARE = [
    'myapp.middleware.SecurityHeadersMiddleware',
    # ... other middleware
]
\`\`\`

---

## Sensitive Data Handling

### Encryption

\`\`\`python
from django.conf import settings
from cryptography.fernet import Fernet

class EncryptionService:
    """Encrypt/decrypt sensitive data"""
    
    def __init__(self):
        self.cipher = Fernet (settings.ENCRYPTION_KEY.encode())
    
    def encrypt (self, data: str) -> str:
        """Encrypt string data"""
        return self.cipher.encrypt (data.encode()).decode()
    
    def decrypt (self, encrypted_data: str) -> str:
        """Decrypt string data"""
        return self.cipher.decrypt (encrypted_data.encode()).decode()

# Usage
encryption = EncryptionService()
encrypted_ssn = encryption.encrypt('123-45-6789')

# Store encrypted_ssn in database
# Decrypt when needed
original_ssn = encryption.decrypt (encrypted_ssn)
\`\`\`

### Secure File Uploads

\`\`\`python
from django.core.exceptions import ValidationError

def validate_file_size (file):
    """Limit file size to 5MB"""
    max_size = 5 * 1024 * 1024  # 5MB
    if file.size > max_size:
        raise ValidationError('File too large (max 5MB)')

def validate_file_extension (file):
    """Only allow specific file types"""
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.pdf']
    ext = os.path.splitext (file.name)[1].lower()
    if ext not in allowed_extensions:
        raise ValidationError (f'File type not allowed. Allowed: {allowed_extensions}')

class Document (models.Model):
    file = models.FileField(
        upload_to='documents/',
        validators=[validate_file_size, validate_file_extension]
    )
\`\`\`

### Sanitize Filenames

\`\`\`python
import os
import re

def sanitize_filename (filename):
    """Remove potentially dangerous characters from filename"""
    # Remove path separators
    filename = os.path.basename (filename)
    
    # Replace spaces and special characters
    filename = re.sub (r'[^\w\s.-]', '', filename)
    filename = re.sub (r'[-\s]+', '-', filename)
    
    return filename.lower()

# Usage
safe_name = sanitize_filename (request.FILES['file'].name)
\`\`\`

---

## API Security

### JWT Authentication

\`\`\`python
# Install
# pip install djangorestframework-simplejwt

# settings.py
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ],
}

from datetime import timedelta

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta (minutes=15),
    'REFRESH_TOKEN_LIFETIME': timedelta (days=1),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
}

# urls.py
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    path('api/token/', TokenObtainPairView.as_view()),
    path('api/token/refresh/', TokenRefreshView.as_view()),
]
\`\`\`

### API Permissions

\`\`\`python
from rest_framework import permissions

class IsOwnerOrReadOnly (permissions.BasePermission):
    """Custom permission: only owner can edit"""
    
    def has_object_permission (self, request, view, obj):
        # Read permissions for any request
        if request.method in permissions.SAFE_METHODS:
            return True
        
        # Write permissions only for owner
        return obj.author == request.user

# Usage in viewset
class ArticleViewSet (viewsets.ModelViewSet):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly, IsOwnerOrReadOnly]
\`\`\`

---

## Security Auditing

### Django Security Check

\`\`\`bash
# Run Django\'s security check
python manage.py check --deploy

# Check for common security issues
python manage.py check --tag security
\`\`\`

### Dependency Scanning

\`\`\`bash
# Install safety
pip install safety

# Check for known vulnerabilities
safety check

# Generate report
safety check --full-report
\`\`\`

### Logging Security Events

\`\`\`python
# settings.py
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'security_file': {
            'level': 'WARNING',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/django/security.log',
            'maxBytes': 1024 * 1024 * 10,  # 10MB
            'backupCount': 5,
        },
    },
    'loggers': {
        'django.security': {
            'handlers': ['security_file'],
            'level': 'WARNING',
            'propagate': False,
        },
    },
}
\`\`\`

---

## Summary

**Security Checklist:**
- ✅ Keep SECRET_KEY secret
- ✅ Disable DEBUG in production
- ✅ Use HTTPS everywhere
- ✅ Enable CSRF protection
- ✅ Validate all user input
- ✅ Use ORM (prevent SQL injection)
- ✅ Escape output (prevent XSS)
- ✅ Strong password requirements
- ✅ Rate limit authentication
- ✅ Set security headers
- ✅ Encrypt sensitive data
- ✅ Validate file uploads
- ✅ Use secure session cookies
- ✅ Regular security audits
- ✅ Keep dependencies updated

**OWASP Top 10 Coverage:**1. **Injection**: Use ORM, validate input
2. **Broken Authentication**: Strong passwords, rate limiting
3. **Sensitive Data Exposure**: Encryption, HTTPS
4. **XML External Entities**: Validate XML input
5. **Broken Access Control**: Proper permissions
6. **Security Misconfiguration**: Follow checklist
7. **XSS**: Auto-escaping, CSP headers
8. **Insecure Deserialization**: Validate serialized data
9. **Using Components with Known Vulnerabilities**: Update regularly
10. **Insufficient Logging**: Log security events

Security is not optional - it's essential for production Django applications.
`,
};
