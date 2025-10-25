export const authenticationPermissionsDrf = {
  title: 'Authentication & Permissions in DRF',
  id: 'authentication-permissions-drf',
  content: `
# Authentication & Permissions in DRF

## Introduction

**Authentication** is the process of verifying who a user is, while **Permissions** determine what that authenticated user is allowed to do. DRF provides a flexible system for both.

### Authentication vs Authorization

- **Authentication**: Who are you? (401 Unauthorized)
- **Authorization/Permissions**: What can you do? (403 Forbidden)

By the end of this section, you'll master:
- Token Authentication
- JWT Authentication
- Session Authentication
- OAuth2 integration
- Permission classes
- Custom permissions
- Production security patterns

---

## Authentication Classes

### Token Authentication

\`\`\`python
# settings.py
INSTALLED_APPS = [
    'rest_framework',
    'rest_framework.authtoken',
]

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.TokenAuthentication',
    ]
}

# Migrate to create token table
python manage.py migrate

# Create tokens
from rest_framework.authtoken.models import Token
token = Token.objects.create (user=user)
print(token.key)  # 9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b

# Use token in requests
# curl -H "Authorization: Token 9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b" http://localhost:8000/api/articles/
\`\`\`

### Custom Login View

\`\`\`python
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.authtoken.models import Token
from rest_framework.response import Response

class CustomAuthToken(ObtainAuthToken):
    """Custom login endpoint returning token and user info"""
    
    def post (self, request, *args, **kwargs):
        serializer = self.serializer_class (data=request.data,
                                          context={'request': request})
        serializer.is_valid (raise_exception=True)
        user = serializer.validated_data['user']
        token, created = Token.objects.get_or_create (user=user)
        
        return Response({
            'token': token.key,
            'user_id': user.pk,
            'username': user.username,
            'email': user.email,
            'is_staff': user.is_staff,
        })

# urls.py
urlpatterns = [
    path('api/login/', CustomAuthToken.as_view()),
]
\`\`\`

### JWT Authentication (Simple JWT)

\`\`\`python
# Install
pip install djangorestframework-simplejwt

# settings.py
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ],
}

from datetime import timedelta

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta (minutes=60),
    'REFRESH_TOKEN_LIFETIME': timedelta (days=7),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
    'UPDATE_LAST_LOGIN': True,
    'ALGORITHM': 'HS256',
    'SIGNING_KEY': SECRET_KEY,
    'AUTH_HEADER_TYPES': ('Bearer',),
    'USER_ID_FIELD': 'id',
    'USER_ID_CLAIM': 'user_id',
}

# urls.py
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
    TokenVerifyView,
)

urlpatterns = [
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/token/verify/', TokenVerifyView.as_view(), name='token_verify'),
]

# Usage:
# POST /api/token/
# {"username": "john", "password": "secret"}
# Returns: {"access": "...", "refresh": "..."}

# Use access token:
# curl -H "Authorization: Bearer <access_token>" http://localhost:8000/api/articles/

# Refresh when expired:
# POST /api/token/refresh/
# {"refresh": "..."}
# Returns new access token
\`\`\`

### Custom JWT Claims

\`\`\`python
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework_simplejwt.views import TokenObtainPairView

class CustomTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token (cls, user):
        token = super().get_token (user)
        
        # Add custom claims
        token['username'] = user.username
        token['email'] = user.email
        token['is_staff'] = user.is_staff
        token['role'] = user.role
        
        return token

class CustomTokenObtainPairView(TokenObtainPairView):
    serializer_class = CustomTokenObtainPairSerializer

# urls.py
urlpatterns = [
    path('api/token/', CustomTokenObtainPairView.as_view()),
]
\`\`\`

---

## Permission Classes

### Built-in Permissions

\`\`\`python
from rest_framework import permissions

# AllowAny - no restriction
class ArticleList(APIView):
    permission_classes = [permissions.AllowAny]

# IsAuthenticated - must be logged in
class ArticleCreate(APIView):
    permission_classes = [permissions.IsAuthenticated]

# IsAuthenticatedOrReadOnly - read for all, write for authenticated
class ArticleViewSet (viewsets.ModelViewSet):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

# IsAdminUser - must be admin
class UserList(APIView):
    permission_classes = [permissions.IsAdminUser]
\`\`\`

### Object-Level Permissions

\`\`\`python
class IsAuthorOrReadOnly (permissions.BasePermission):
    """
    Custom permission: only author can edit
    """
    
    def has_object_permission (self, request, view, obj):
        # Read permissions for any request (GET, HEAD, OPTIONS)
        if request.method in permissions.SAFE_METHODS:
            return True
        
        # Write permissions only for article author
        return obj.author == request.user

class ArticleViewSet (viewsets.ModelViewSet):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly, IsAuthorOrReadOnly]
\`\`\`

### Complex Custom Permissions

\`\`\`python
class IsAuthorOrStaffOrReadOnly (permissions.BasePermission):
    """
    Custom permission with multiple conditions
    """
    
    def has_permission (self, request, view):
        """Check general permission"""
        # Anyone can read
        if request.method in permissions.SAFE_METHODS:
            return True
        
        # Must be authenticated to write
        return request.user and request.user.is_authenticated
    
    def has_object_permission (self, request, view, obj):
        """Check object-specific permission"""
        # Read permissions for everyone
        if request.method in permissions.SAFE_METHODS:
            return True
        
        # Write permissions for author or staff
        return obj.author == request.user or request.user.is_staff

class CanPublishPermission (permissions.BasePermission):
    """Only editors can publish articles"""
    
    def has_permission (self, request, view):
        if view.action == 'publish':
            return request.user.has_perm('articles.can_publish')
        return True

class ArticleViewSet (viewsets.ModelViewSet):
    permission_classes = [
        permissions.IsAuthenticated,
        IsAuthorOrStaffOrReadOnly,
        CanPublishPermission,
    ]
\`\`\`

### Dynamic Permissions

\`\`\`python
class ArticleViewSet (viewsets.ModelViewSet):
    
    def get_permissions (self):
        """Different permissions per action"""
        if self.action == 'list':
            permission_classes = [permissions.AllowAny]
        elif self.action == 'create':
            permission_classes = [permissions.IsAuthenticated]
        elif self.action in ['update', 'partial_update', 'destroy']:
            permission_classes = [permissions.IsAuthenticated, IsAuthorOrAdmin]
        elif self.action == 'publish':
            permission_classes = [permissions.IsAuthenticated, CanPublishPermission]
        else:
            permission_classes = [permissions.IsAuthenticatedOrReadOnly]
        
        return [permission() for permission in permission_classes]
\`\`\`

---

## OAuth2 Integration

### Using django-oauth-toolkit

\`\`\`python
# Install
pip install django-oauth-toolkit

# settings.py
INSTALLED_APPS = [
    'oauth2_provider',
]

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'oauth2_provider.contrib.rest_framework.OAuth2Authentication',
    ],
}

# Configure OAuth2
OAUTH2_PROVIDER = {
    'SCOPES': {
        'read': 'Read scope',
        'write': 'Write scope',
        'groups': 'Access to your groups'
    },
    'ACCESS_TOKEN_EXPIRE_SECONDS': 36000,
}

# urls.py
urlpatterns = [
    path('o/', include('oauth2_provider.urls', namespace='oauth2_provider')),
]

# Create application in admin
# Then use OAuth2 flow
\`\`\`

---

## Social Authentication

### Google OAuth2

\`\`\`python
# Install
pip install social-auth-app-django

# settings.py
INSTALLED_APPS = [
    'social_django',
]

AUTHENTICATION_BACKENDS = [
    'social_core.backends.google.GoogleOAuth2',
    'django.contrib.auth.backends.ModelBackend',
]

SOCIAL_AUTH_GOOGLE_OAUTH2_KEY = 'your-client-id'
SOCIAL_AUTH_GOOGLE_OAUTH2_SECRET = 'your-client-secret'
SOCIAL_AUTH_GOOGLE_OAUTH2_SCOPE = [
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile',
]

# urls.py
urlpatterns = [
    path('auth/', include('social_django.urls', namespace='social')),
]
\`\`\`

---

## Production Security Patterns

### API Key Authentication

\`\`\`python
from rest_framework.authentication import BaseAuthentication
from rest_framework import exceptions
from .models import APIKey

class APIKeyAuthentication(BaseAuthentication):
    """
    Custom API Key authentication
    """
    
    def authenticate (self, request):
        api_key = request.META.get('HTTP_X_API_KEY')
        
        if not api_key:
            return None
        
        try:
            key = APIKey.objects.select_related('user').get (key=api_key, is_active=True)
        except APIKey.DoesNotExist:
            raise exceptions.AuthenticationFailed('Invalid API key')
        
        # Log API key usage
        key.last_used_at = timezone.now()
        key.usage_count += 1
        key.save (update_fields=['last_used_at', 'usage_count'])
        
        return (key.user, key)

class APIKey (models.Model):
    """API Key model"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='api_keys')
    key = models.CharField (max_length=40, unique=True, db_index=True)
    name = models.CharField (max_length=100)
    is_active = models.BooleanField (default=True)
    created_at = models.DateTimeField (auto_now_add=True)
    last_used_at = models.DateTimeField (null=True, blank=True)
    usage_count = models.IntegerField (default=0)
    
    def save (self, *args, **kwargs):
        if not self.key:
            self.key = self.generate_key()
        super().save(*args, **kwargs)
    
    @staticmethod
    def generate_key():
        import secrets
        return secrets.token_urlsafe(32)
\`\`\`

### Rate Limiting by User

\`\`\`python
from rest_framework.throttling import UserRateThrottle

class BurstRateThrottle(UserRateThrottle):
    """High burst rate (100/min)"""
    rate = '100/min'

class SustainedRateThrottle(UserRateThrottle):
    """Sustained rate (10000/day)"""
    rate = '10000/day'

class ArticleViewSet (viewsets.ModelViewSet):
    throttle_classes = [BurstRateThrottle, SustainedRateThrottle]
\`\`\`

---

## Summary

**Authentication Methods:**
- Token: Simple, stateless, works everywhere
- JWT: Modern, self-contained, no database lookups
- Session: Django's default, requires cookies
- OAuth2: For third-party integrations
- Social: Google, Facebook, GitHub

**Permission Patterns:**
- Global: Set DEFAULT_PERMISSION_CLASSES
- Per-View: Set permission_classes attribute
- Per-Action: Override get_permissions()
- Object-Level: has_object_permission()

**Production Best Practices:**
- ✅ Use HTTPS in production
- ✅ Rotate secrets regularly
- ✅ Implement rate limiting
- ✅ Log authentication events
- ✅ Use short token expiry
- ✅ Implement refresh tokens
- ✅ Monitor failed auth attempts
- ✅ Use API keys for machine-to-machine
- ✅ Implement account lockout
- ✅ Add MFA for sensitive operations

Authentication and permissions are critical for API security. Choose the right authentication scheme for your use case and implement granular permissions to control access.
`,
};
