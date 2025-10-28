export const authenticationPermissionsDrfQuiz = [
  {
    id: 1,
    question:
      'Compare different DRF authentication methods (TokenAuthentication, SessionAuthentication, JWTAuthentication). Explain when to use each and how to implement custom authentication.',
    answer: `
**TokenAuthentication:**
Simple token-based auth with database-stored tokens.

\`\`\`python
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.TokenAuthentication',
    ]
}

# Generate tokens
from rest_framework.authtoken.models import Token
token = Token.objects.create (user=user)

# Client sends: Authorization: Token abc123xyz
\`\`\`

**Pros:** Simple, stateless  
**Cons:** Tokens don't expire, stored in DB

**SessionAuthentication:**
Cookie-based, uses Django sessions.

\`\`\`python
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
    ]
}
\`\`\`

**Pros:** Built-in, secure  
**Cons:** Not suitable for mobile/SPA, requires CSRF

**JWTAuthentication:**
Self-contained tokens with expiration.

\`\`\`python
pip install djangorestframework-simplejwt

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ]
}

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta (minutes=15),
    'REFRESH_TOKEN_LIFETIME': timedelta (days=1),
}
\`\`\`

**Pros:** Stateless, self-contained, expires  
**Cons:** Can't revoke before expiry

**Custom Authentication:**

\`\`\`python
from rest_framework.authentication import BaseAuthentication

class APIKeyAuthentication(BaseAuthentication):
    def authenticate (self, request):
        api_key = request.META.get('HTTP_X_API_KEY')
        if not api_key:
            return None
        
        try:
            user = User.objects.get (api_key=api_key)
            return (user, None)
        except User.DoesNotExist:
            raise AuthenticationFailed('Invalid API key')
\`\`\`

**Use Cases:**
- TokenAuth: Internal APIs, simple apps
- SessionAuth: Same-domain web apps
- JWT: Mobile apps, SPAs, microservices
- Custom: API keys, OAuth2, etc.
      `,
  },
  {
    question:
      'Explain DRF permission classes and how to implement object-level permissions. Include examples of custom permission classes for complex authorization logic.',
    answer: `
**Built-in Permissions:**

\`\`\`python
from rest_framework.permissions import (
    IsAuthenticated, IsAdminUser, AllowAny, IsAuthenticatedOrReadOnly
)

class ArticleViewSet (viewsets.ModelViewSet):
    permission_classes = [IsAuthenticatedOrReadOnly]
\`\`\`

**Object-Level Permissions:**

\`\`\`python
from rest_framework import permissions

class IsOwnerOrReadOnly (permissions.BasePermission):
    def has_object_permission (self, request, view, obj):
        # Read permissions for everyone
        if request.method in permissions.SAFE_METHODS:
            return True
        
        # Write permissions only for owner
        return obj.author == request.user
\`\`\`

**Custom Permission Class:**

\`\`\`python
class IsOwnerOrAdmin (permissions.BasePermission):
    message = "You must be the owner or admin"
    
    def has_permission (self, request, view):
        # View-level check
        return request.user and request.user.is_authenticated
    
    def has_object_permission (self, request, view, obj):
        # Object-level check
        return obj.author == request.user or request.user.is_staff

class CanPublishArticle (permissions.BasePermission):
    def has_object_permission (self, request, view, obj):
        if view.action != 'publish':
            return True
        
        # Only staff can publish
        return request.user.is_staff
\`\`\`

**Complex Authorization:**

\`\`\`python
class TeamMemberPermission (permissions.BasePermission):
    def has_object_permission (self, request, view, obj):
        # Check if user is team member
        return obj.team.members.filter (id=request.user.id).exists()

class SubscriptionPermission (permissions.BasePermission):
    def has_permission (self, request, view):
        if not request.user.is_authenticated:
            return False
        
        # Check subscription status
        return request.user.subscription.is_active
\`\`\`

**Usage:**

\`\`\`python
class ArticleViewSet (viewsets.ModelViewSet):
    permission_classes = [IsAuthenticated, IsOwnerOrReadOnly]
    
    def get_permissions (self):
        if self.action == 'destroy':
            return [IsAdminUser()]
        return super().get_permissions()
\`\`\`

**Best Practices:**
- Keep permissions focused and reusable
- Separate view-level and object-level checks
- Fail securely (deny by default)
- Cache expensive permission checks
      `,
  },
  {
    question:
      'Describe how to implement OAuth2 or social authentication with DRF. Include token refresh strategies and security considerations.',
    answer: `
**OAuth2 with django-oauth-toolkit:**

\`\`\`python
pip install django-oauth-toolkit

INSTALLED_APPS = [
    'oauth2_provider',
]

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'oauth2_provider.contrib.rest_framework.OAuth2Authentication',
    ]
}
\`\`\`

**Social Auth with dj-rest-auth:**

\`\`\`python
pip install dj-rest-auth allauth

INSTALLED_APPS = [
    'rest_framework.authtoken',
    'dj_rest_auth',
    'allauth',
    'allauth.account',
    'allauth.socialaccount',
    'allauth.socialaccount.providers.google',
]

urlpatterns = [
    path('auth/', include('dj_rest_auth.urls')),
    path('auth/registration/', include('dj_rest_auth.registration.urls')),
]
\`\`\`

**JWT Token Refresh:**

\`\`\`python
from rest_framework_simplejwt.views import (
    TokenObtainPairView, TokenRefreshView
)

urlpatterns = [
    path('token/', TokenObtainPairView.as_view()),
    path('token/refresh/', TokenRefreshView.as_view()),
]

# Client workflow:
# 1. POST /token/ -> Get access + refresh tokens
# 2. Use access token until expired
# 3. POST /token/refresh/ with refresh token -> Get new access token
\`\`\`

**Custom Token Refresh:**

\`\`\`python
from rest_framework_simplejwt.tokens import RefreshToken

class TokenRefreshSerializer (serializers.Serializer):
    refresh = serializers.CharField()
    
    def validate (self, attrs):
        refresh = RefreshToken (attrs['refresh'])
        data = {'access': str (refresh.access_token)}
        
        # Rotate refresh tokens
        if settings.SIMPLE_JWT.get('ROTATE_REFRESH_TOKENS'):
            refresh.set_jti()
            refresh.set_exp()
            data['refresh'] = str (refresh)
        
        return data
\`\`\`

**Security Considerations:**1. **Token Storage:**
   - Store refresh tokens securely (httpOnly cookies)
   - Never store in localStorage for XSS protection

2. **Token Rotation:**
   - Rotate refresh tokens on use
   - Implement token blacklisting

3. **HTTPS Only:**
   - All auth endpoints must use HTTPS
   - Set secure cookie flags

4. **Rate Limiting:**
   - Limit token refresh requests
   - Prevent brute force attacks

5. **Token Expiration:**
   - Short access token lifetime (15 min)
   - Longer refresh token (7 days)

**Production Pattern:**

\`\`\`python
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta (minutes=15),
    'REFRESH_TOKEN_LIFETIME': timedelta (days=7),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
    'ALGORITHM': 'HS256',
    'SIGNING_KEY': settings.SECRET_KEY,
    'AUTH_HEADER_TYPES': ('Bearer',),
}
\`\`\`
      `,
  },
].map(({ id: _id, ...q }, idx) => ({
  id: `django-q-${idx + 1}`,
  question: q.question,
  sampleAnswer: String(q.answer),
  keyPoints: [],
}));
