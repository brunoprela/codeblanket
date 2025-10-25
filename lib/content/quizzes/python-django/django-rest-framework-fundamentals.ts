export const djangoRestFrameworkFundamentalsQuiz = {
  title: 'Django REST Framework Fundamentals - Discussion Questions',
  questions: [
    {
      question:
        'Explain the core components of Django REST Framework and how they work together. Compare APIView, generics, and ViewSets approaches, and describe when you would use each pattern in a production API.',
      answer: `
**DRF Core Components:**

**1. Serializers**: Convert complex data to/from JSON
**2. Views**: Handle HTTP methods and business logic
**3. Routers**: Auto-generate URL patterns
**4. Parsers**: Parse request data
**5. Renderers**: Format response data
**6. Authentication**: Verify user identity
**7. Permissions**: Control access
**8. Throttling**: Rate limiting

**View Patterns Comparison:**

**APIView (Lowest Level):**
\`\`\`python
from rest_framework.views import APIView
from rest_framework.response import Response

class ArticleListView(APIView):
    """Most control, most code"""
    
    def get(self, request):
        articles = Article.objects.all()
        serializer = ArticleSerializer(articles, many=True)
        return Response(serializer.data)
    
    def post(self, request):
        serializer = ArticleSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)
\`\`\`

**When to use APIView:**
- ✅ Custom business logic
- ✅ Non-CRUD operations
- ✅ Multiple models in one view
- ✅ Complex request handling
- ❌ Simple CRUD operations
- ❌ Standard list/detail patterns

**Generic Views (Mid-Level):**
\`\`\`python
from rest_framework import generics

class ArticleListView(generics.ListCreateAPIView):
    """Less code, standard patterns"""
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    
    def get_queryset(self):
        # Custom filtering
        return Article.objects.filter(author=self.request.user)

class ArticleDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
\`\`\`

**Available Generics:**
- \`ListAPIView\`: GET list
- \`CreateAPIView\`: POST
- \`RetrieveAPIView\`: GET detail
- \`UpdateAPIView\`: PUT/PATCH
- \`DestroyAPIView\`: DELETE
- \`ListCreateAPIView\`: GET list + POST
- \`RetrieveUpdateDestroyAPIView\`: GET + PUT/PATCH + DELETE

**When to use Generics:**
- ✅ Standard CRUD operations
- ✅ Need some customization
- ✅ Want to mix and match behaviors
- ✅ Clear separation of list/detail views
- ❌ Need all CRUD in one class
- ❌ Want automatic URL routing

**ViewSets (Highest Level):**
\`\`\`python
from rest_framework import viewsets

class ArticleViewSet(viewsets.ModelViewSet):
    """Least code, most conventions"""
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    
    # Optional overrides
    def get_queryset(self):
        if self.action == 'list':
            return Article.objects.filter(status='published')
        return Article.objects.all()
    
    def perform_create(self, serializer):
        serializer.save(author=self.request.user)
    
    # Custom actions
    @action(detail=True, methods=['post'])
    def publish(self, request, pk=None):
        article = self.get_object()
        article.status = 'published'
        article.published_at = timezone.now()
        article.save()
        return Response({'status': 'published'})

# Automatic URL routing
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register(r'articles', ArticleViewSet)

# Generates:
# GET    /articles/          -> list
# POST   /articles/          -> create
# GET    /articles/{id}/     -> retrieve
# PUT    /articles/{id}/     -> update
# PATCH  /articles/{id}/     -> partial_update
# DELETE /articles/{id}/     -> destroy
# POST   /articles/{id}/publish/ -> publish (custom action)
\`\`\`

**ViewSet Types:**
- \`ViewSet\`: No default actions (define your own)
- \`GenericViewSet\`: Includes mixins but no actions by default
- \`ModelViewSet\`: Full CRUD (list, create, retrieve, update, destroy)
- \`ReadOnlyModelViewSet\`: List and retrieve only

**When to use ViewSets:**
- ✅ Standard REST resources
- ✅ Want automatic URL routing
- ✅ Need all CRUD operations
- ✅ Building consistent API
- ❌ Very custom logic per endpoint
- ❌ Non-CRUD operations
- ❌ Need fine control over URLs

**Production Decision Matrix:**

| Scenario | Recommended Pattern |
|----------|-------------------|
| Simple CRUD resource | ModelViewSet |
| CRUD with custom actions | ViewSet + custom actions |
| Read-only API | ReadOnlyModelViewSet |
| List + Create only | ListCreateAPIView |
| Custom business logic | APIView |
| Multiple models | APIView |
| Non-RESTful endpoint | APIView |

**Example - Choosing the Right Pattern:**

\`\`\`python
# Simple CRUD - Use ViewSet
class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

# Read-only public API - Use ReadOnlyModelViewSet
class PublicArticleViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Article.objects.filter(status='published')
    serializer_class = ArticleSerializer

# Custom analytics endpoint - Use APIView
class AnalyticsView(APIView):
    def get(self, request):
        stats = {
            'total_articles': Article.objects.count(),
            'published': Article.objects.filter(status='published').count(),
            'views': Article.objects.aggregate(Sum('view_count'))['view_count__sum']
        }
        return Response(stats)

# List + Create with custom validation - Use Generic
class ArticleListCreateView(generics.ListCreateAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    
    def create(self, request, *args, **kwargs):
        # Custom validation
        if Article.objects.filter(
            author=request.user,
            created_at__date=timezone.now().date()
        ).count() >= 10:
            return Response(
                {'error': 'Daily article limit reached'},
                status=400
            )
        return super().create(request, *args, **kwargs)
\`\`\`

**Best Practices:**
- ✅ Start with ViewSets for CRUD
- ✅ Use Generics for partial CRUD
- ✅ Use APIView for custom logic
- ✅ Keep ViewSets focused on one model
- ✅ Use custom actions for model-specific operations
- ✅ Use separate views for complex cross-model operations
- ❌ Don't force everything into ViewSets
- ❌ Don't put too much logic in views (use services/managers)
- ❌ Don't mix unrelated resources in one ViewSet

The right pattern depends on your specific use case, but ViewSets provide the best balance of simplicity and functionality for standard REST APIs.
      `,
    },
    {
      question:
        'Describe DRF request/response cycle, including parsing, authentication, permissions, throttling, and rendering. Explain how to customize each stage and implement middleware-like functionality.',
      answer: `
**DRF Request/Response Cycle:**

\`\`\`
1. Django URLConf → Routes to DRF View
2. Parser → Converts request body to Python data
3. Authentication → Identifies user
4. Permissions → Checks access rights
5. Throttling → Checks rate limits
6. View Logic → Business logic executes
7. Serialization → Converts Python objects to primitives
8. Renderer → Formats response (JSON/XML/etc)
9. Response → Returns to client
\`\`\`

**1. Parsing Stage:**

\`\`\`python
from rest_framework.parsers import JSONParser, MultiPartParser, FormParser

class ArticleViewSet(viewsets.ModelViewSet):
    # Default: JSONParser
    parser_classes = [JSONParser, MultiPartParser, FormParser]
    
    # Now accepts:
    # - application/json
    # - multipart/form-data
    # - application/x-www-form-urlencoded

# Custom Parser
from rest_framework.parsers import BaseParser

class PlainTextParser(BaseParser):
    """Parse plain text request bodies"""
    media_type = 'text/plain'
    
    def parse(self, stream, media_type=None, parser_context=None):
        return stream.read().decode('utf-8')

class ArticleViewSet(viewsets.ModelViewSet):
    parser_classes = [JSONParser, PlainTextParser]
    
    def create(self, request):
        # request.data contains parsed data
        if request.content_type == 'text/plain':
            # Handle plain text
            pass
        else:
            # Handle JSON
            pass
\`\`\`

**2. Authentication Stage:**

\`\`\`python
from rest_framework.authentication import (
    TokenAuthentication, SessionAuthentication, BasicAuthentication
)

class ArticleViewSet(viewsets.ModelViewSet):
    # Check multiple auth methods (first successful wins)
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    
    # request.user and request.auth are set after authentication

# Custom Authentication
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed

class APIKeyAuthentication(BaseAuthentication):
    """Authenticate via X-API-Key header"""
    
    def authenticate(self, request):
        api_key = request.META.get('HTTP_X_API_KEY')
        
        if not api_key:
            return None  # No authentication attempted
        
        try:
            user = User.objects.get(api_keys__key=api_key)
            return (user, api_key)  # (user, auth)
        except User.DoesNotExist:
            raise AuthenticationFailed('Invalid API key')

# Usage
class ArticleViewSet(viewsets.ModelViewSet):
    authentication_classes = [APIKeyAuthentication]
\`\`\`

**3. Permissions Stage:**

\`\`\`python
from rest_framework.permissions import IsAuthenticated, IsAdminUser

class ArticleViewSet(viewsets.ModelViewSet):
    # All methods require authentication
    permission_classes = [IsAuthenticated]
    
    def get_permissions(self):
        """Different permissions per action"""
        if self.action in ['list', 'retrieve']:
            # Anyone can read
            return []
        elif self.action in ['create', 'update', 'partial_update']:
            # Must be authenticated to write
            return [IsAuthenticated()]
        else:
            # Must be admin to delete
            return [IsAdminUser()]

# Custom Permission
from rest_framework.permissions import BasePermission

class IsOwnerOrReadOnly(BasePermission):
    """Object-level permission: only owner can edit"""
    
    def has_object_permission(self, request, view, obj):
        # Read permissions for any request
        if request.method in ['GET', 'HEAD', 'OPTIONS']:
            return True
        
        # Write permissions only for owner
        return obj.author == request.user

class ArticleViewSet(viewsets.ModelViewSet):
    permission_classes = [IsAuthenticated, IsOwnerOrReadOnly]
\`\`\`

**4. Throttling Stage:**

\`\`\`python
from rest_framework.throttling import UserRateThrottle, AnonRateThrottle

# settings.py
REST_FRAMEWORK = {
    'DEFAULT_THROTTLE_RATES': {
        'user': '1000/day',
        'anon': '100/day',
        'burst': '60/min',
    }
}

class ArticleViewSet(viewsets.ModelViewSet):
    throttle_classes = [UserRateThrottle, AnonRateThrottle]

# Custom Throttle
from rest_framework.throttling import SimpleRateThrottle

class BurstRateThrottle(SimpleRateThrottle):
    """Limit burst traffic"""
    scope = 'burst'
    
    def get_cache_key(self, request, view):
        if request.user.is_authenticated:
            ident = request.user.pk
        else:
            ident = self.get_ident(request)
        
        return self.cache_format % {
            'scope': self.scope,
            'ident': ident
        }

class ArticleViewSet(viewsets.ModelViewSet):
    throttle_classes = [BurstRateThrottle]
\`\`\`

**5. View Logic Stage:**

\`\`\`python
class ArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    
    def get_queryset(self):
        """Filter queryset based on request"""
        qs = super().get_queryset()
        
        # Filter by query param
        status = self.request.query_params.get('status')
        if status:
            qs = qs.filter(status=status)
        
        # Filter by user permissions
        if not self.request.user.is_staff:
            qs = qs.filter(author=self.request.user)
        
        return qs
    
    def get_serializer_context(self):
        """Add extra context to serializer"""
        context = super().get_serializer_context()
        context['user_preferences'] = self.request.user.preferences
        return context
    
    def perform_create(self, serializer):
        """Hook before saving"""
        serializer.save(
            author=self.request.user,
            ip_address=self.request.META.get('REMOTE_ADDR')
        )
\`\`\`

**6. Rendering Stage:**

\`\`\`python
from rest_framework.renderers import JSONRenderer, BrowsableAPIRenderer

class ArticleViewSet(viewsets.ModelViewSet):
    renderer_classes = [JSONRenderer, BrowsableAPIRenderer]
    
    # Client can request format:
    # /api/articles/?format=json
    # /api/articles/?format=api (browsable)

# Custom Renderer
from rest_framework.renderers import BaseRenderer

class CSVRenderer(BaseRenderer):
    """Render as CSV"""
    media_type = 'text/csv'
    format = 'csv'
    
    def render(self, data, accepted_media_type=None, renderer_context=None):
        if not data:
            return ''
        
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        
        return output.getvalue()

class ArticleViewSet(viewsets.ModelViewSet):
    renderer_classes = [JSONRenderer, CSVRenderer]
    
    # GET /api/articles/?format=csv
\`\`\`

**Middleware-Like Functionality:**

\`\`\`python
from rest_framework.views import APIView
from rest_framework.response import Response
import time

class TimingMixin:
    """Mixin to time API requests"""
    
    def initial(self, request, *args, **kwargs):
        """Called before view logic"""
        request._start_time = time.time()
        super().initial(request, *args, **kwargs)
    
    def finalize_response(self, request, response, *args, **kwargs):
        """Called after view logic"""
        response = super().finalize_response(request, response, *args, **kwargs)
        
        duration = time.time() - request._start_time
        response['X-Request-Duration'] = f'{duration:.3f}s'
        
        return response

class LoggingMixin:
    """Mixin to log API requests"""
    
    def initial(self, request, *args, **kwargs):
        logger.info(f'{request.method} {request.path} - User: {request.user}')
        super().initial(request, *args, **kwargs)

class ArticleViewSet(TimingMixin, LoggingMixin, viewsets.ModelViewSet):
    # Mixins execute in order (left to right)
    pass

# Or use a custom exception handler (global middleware)
def custom_exception_handler(exc, context):
    """Global error handler"""
    from rest_framework.views import exception_handler
    
    # Call DRF's default handler first
    response = exception_handler(exc, context)
    
    if response is not None:
        # Customize error response
        response.data['status_code'] = response.status_code
        response.data['error_type'] = exc.__class__.__name__
        
        # Log error
        logger.error(f'API Error: {exc}', exc_info=True)
    
    return response

# settings.py
REST_FRAMEWORK = {
    'EXCEPTION_HANDLER': 'myapp.utils.custom_exception_handler'
}
\`\`\`

**Complete Custom Cycle Example:**

\`\`\`python
from rest_framework import viewsets, status
from rest_framework.response import Response
from rest_framework.decorators import action

class ArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    
    # 1. Parsers
    parser_classes = [JSONParser, MultiPartParser]
    
    # 2. Authentication
    authentication_classes = [TokenAuthentication]
    
    # 3. Permissions
    permission_classes = [IsAuthenticated]
    
    # 4. Throttling
    throttle_classes = [UserRateThrottle]
    
    # 5. Renderers
    renderer_classes = [JSONRenderer]
    
    def initial(self, request, *args, **kwargs):
        """Before view logic"""
        request._custom_data = {}
        super().initial(request, *args, **kwargs)
    
    def list(self, request):
        """6. View logic"""
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)
    
    def finalize_response(self, request, response, *args, **kwargs):
        """After view logic"""
        response = super().finalize_response(request, response, *args, **kwargs)
        response['X-Total-Count'] = self.get_queryset().count()
        return response
\`\`\`

**Best Practices:**
- ✅ Use appropriate parsers for expected content types
- ✅ Implement authentication for protected endpoints
- ✅ Use object-level permissions for fine-grained access
- ✅ Apply throttling to prevent abuse
- ✅ Provide multiple renderers for flexibility
- ✅ Use mixins for reusable functionality
- ❌ Don't skip authentication for sensitive data
- ❌ Don't implement business logic in permissions
- ❌ Don't forget to handle edge cases in custom stages

Understanding this cycle allows you to customize any part of DRF request processing.
      `,
    },
    {
      question:
        'Explain DRF content negotiation, format suffixes, and API versioning strategies. Provide examples of implementing multiple API versions while maintaining backward compatibility.',
      answer: `
**Content Negotiation in DRF:**

Content negotiation is the process of selecting the best representation for a response when multiple representations are available.

**1. Accept Header Negotiation:**

\`\`\`python
# Client requests
GET /api/articles/
Accept: application/json

# Or
GET /api/articles/
Accept: application/xml

# settings.py
REST_FRAMEWORK = {
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
        'rest_framework.renderers.BrowsableAPIRenderer',
    ]
}

# DRF selects renderer based on:
# 1. Format suffix (?format=json)
# 2. Accept header
# 3. Default renderer
\`\`\`

**2. Format Suffixes:**

\`\`\`python
# urls.py
from rest_framework.urlpatterns import format_suffix_patterns

urlpatterns = [
    path('articles/', ArticleListView.as_view()),
    path('articles/<int:pk>/', ArticleDetailView.as_view()),
]

# Add format suffix support
urlpatterns = format_suffix_patterns(urlpatterns)

# Now supports:
# /api/articles.json
# /api/articles.xml
# /api/articles/?format=json

# In view
from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(['GET'])
def article_list(request, format=None):
    # format parameter is automatically added
    articles = Article.objects.all()
    serializer = ArticleSerializer(articles, many=True)
    return Response(serializer.data)
\`\`\`

**API Versioning Strategies:**

**1. URL Path Versioning (Recommended):**

\`\`\`python
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_VERSIONING_CLASS': 'rest_framework.versioning.URLPathVersioning',
    'DEFAULT_VERSION': 'v1',
    'ALLOWED_VERSIONS': ['v1', 'v2'],
    'VERSION_PARAM': 'version',
}

# urls.py
urlpatterns = [
    path('api/v1/', include('myapp.urls_v1')),
    path('api/v2/', include('myapp.urls_v2')),
]

# In view
class ArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.all()
    
    def get_serializer_class(self):
        if self.request.version == 'v1':
            return ArticleSerializerV1
        return ArticleSerializerV2
    
    def list(self, request, *args, **kwargs):
        if request.version == 'v1':
            # V1 logic
            pass
        else:
            # V2 logic
            pass
\`\`\`

**2. Accept Header Versioning:**

\`\`\`python
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_VERSIONING_CLASS': 'rest_framework.versioning.AcceptHeaderVersioning',
}

# Client request
GET /api/articles/
Accept: application/json; version=2.0

# In view
class ArticleViewSet(viewsets.ModelViewSet):
    def get_serializer_class(self):
        version = self.request.version
        if version == '1.0':
            return ArticleSerializerV1
        return ArticleSerializerV2
\`\`\`

**3. Query Parameter Versioning:**

\`\`\`python
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_VERSIONING_CLASS': 'rest_framework.versioning.QueryParameterVersioning',
}

# Client request
GET /api/articles/?version=2

# In view
def get_queryset(self):
    if self.request.version == '1':
        # V1: Return all fields
        return Article.objects.all()
    else:
        # V2: Optimize with select_related
        return Article.objects.select_related('author', 'category')
\`\`\`

**Production Versioning Strategy:**

\`\`\`python
# Version-specific serializers
# serializers/v1.py
class ArticleSerializerV1(serializers.ModelSerializer):
    class Meta:
        model = Article
        fields = ['id', 'title', 'content', 'author']

# serializers/v2.py
class ArticleSerializerV2(serializers.ModelSerializer):
    author = UserSerializer(read_only=True)
    tags = TagSerializer(many=True, read_only=True)
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'content', 'author', 'tags', 'meta']

# Version router
class VersionedArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.all()
    
    serializer_classes = {
        'v1': ArticleSerializerV1,
        'v2': ArticleSerializerV2,
    }
    
    def get_serializer_class(self):
        version = self.request.version or 'v1'
        return self.serializer_classes.get(version, ArticleSerializerV2)
    
    def get_queryset(self):
        qs = super().get_queryset()
        
        # V2 has optimization
        if self.request.version == 'v2':
            qs = qs.select_related('author').prefetch_related('tags')
        
        return qs
    
    @action(detail=False, methods=['get'])
    def trending(self, request):
        \"\"\"V2-only endpoint\"\"\"
        if request.version != 'v2':
            return Response(
                {'error': 'This endpoint is only available in API v2'},
                status=400
            )
        
        articles = self.get_queryset().filter(
            view_count__gte=1000
        ).order_by('-view_count')[:10]
        
        serializer = self.get_serializer(articles, many=True)
        return Response(serializer.data)
\`\`\`

**Backward Compatibility Patterns:**

**1. Deprecation Warnings:**

\`\`\`python
import warnings

class ArticleViewSet(viewsets.ModelViewSet):
    def list(self, request, *args, **kwargs):
        if request.version == 'v1':
            warnings.warn(
                'API v1 is deprecated. Please migrate to v2.',
                DeprecationWarning
            )
            
            # Add deprecation header
            response = super().list(request, *args, **kwargs)
            response['Warning'] = '299 - "API v1 is deprecated. Use v2."'
            response['Sunset'] = 'Sat, 31 Dec 2024 23:59:59 GMT'
            return response
        
        return super().list(request, *args, **kwargs)
\`\`\`

**2. Field Migration:**

\`\`\`python
class ArticleSerializerV2(serializers.ModelSerializer):
    # Old field (deprecated)
    author_name = serializers.SerializerMethodField()
    
    # New field
    author = UserSerializer(read_only=True)
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'author', 'author_name']  # Both included
    
    def get_author_name(self, obj):
        # Deprecated field still works
        warnings.warn('author_name is deprecated. Use author.name instead.')
        return obj.author.username

    def to_representation(self, instance):
        ret = super().to_representation(instance)
        
        # Remove deprecated field for v3+
        if self.context['request'].version >= 'v3':
            ret.pop('author_name', None)
        
        return ret
\`\`\`

**3. Endpoint Migration:**

\`\`\`python
# urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter

# V1 URLs (legacy)
router_v1 = DefaultRouter()
router_v1.register(r'articles', ArticleViewSetV1, basename='article')

# V2 URLs (current)
router_v2 = DefaultRouter()
router_v2.register(r'articles', ArticleViewSetV2, basename='article')

urlpatterns = [
    path('api/v1/', include(router_v1.urls)),
    path('api/v2/', include(router_v2.urls)),
    
    # Default to latest version
    path('api/', include(router_v2.urls)),
]

# Redirect old endpoints
from django.views.generic import RedirectView

urlpatterns += [
    path('legacy/articles/', RedirectView.as_view(
        url='/api/v2/articles/',
        permanent=False  # 302 redirect (temporary)
    )),
]
\`\`\`

**4. Gradual Migration Strategy:**

\`\`\`python
class ArticleViewSet(viewsets.ModelViewSet):
    \"\"\"
    Support multiple versions in single ViewSet
    \"\"\"
    
    def get_serializer_class(self):
        version_map = {
            'v1': ArticleSerializerV1,
            'v2': ArticleSerializerV2,
            'v3': ArticleSerializerV3,
        }
        return version_map.get(self.request.version, ArticleSerializerV3)
    
    def list(self, request, *args, **kwargs):
        # Common logic
        queryset = self.get_queryset()
        
        # Version-specific modifications
        if request.version == 'v1':
            # V1: Simple pagination
            page = self.paginate_queryset(queryset)
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        
        elif request.version == 'v2':
            # V2: Add metadata
            response = super().list(request, *args, **kwargs)
            response.data['_meta'] = {
                'version': 'v2',
                'total_count': queryset.count()
            }
            return response
        
        # V3+: Enhanced response
        return super().list(request, *args, **kwargs)
\`\`\`

**5. API Documentation Versioning:**

\`\`\`python
from drf_spectacular.utils import extend_schema, OpenApiParameter

class ArticleViewSet(viewsets.ModelViewSet):
    
    @extend_schema(
        description='List articles (V1: basic, V2: includes author details)',
        parameters=[
            OpenApiParameter(
                name='version',
                description='API version',
                required=False,
                type=str
            )
        ],
        responses={
            200: ArticleSerializerV2,
        }
    )
    def list(self, request, *args, **kwargs):
        return super().list(request, *args, **kwargs)
\`\`\`

**Best Practices:**

**Versioning:**
- ✅ Use URL path versioning for clarity
- ✅ Version major breaking changes only
- ✅ Support old versions for transition period (6-12 months)
- ✅ Document deprecation timeline
- ✅ Provide migration guides
- ✅ Use semantic versioning (v1, v2, v3)
- ❌ Don't version every minor change
- ❌ Don't support too many versions simultaneously
- ❌ Don't break old versions without warning

**Content Negotiation:**
- ✅ Support JSON as default
- ✅ Provide format suffixes for convenience
- ✅ Use Accept header for production clients
- ✅ Document available formats
- ❌ Don't assume client format preference
- ❌ Don't forget to test all formats

**Backward Compatibility:**
- ✅ Add new fields, don't remove old ones (initially)
- ✅ Make new fields optional
- ✅ Provide defaults for new required fields
- ✅ Use deprecation warnings
- ✅ Redirect old endpoints temporarily
- ❌ Don't change field types
- ❌ Don't change response structure suddenly
- ❌ Don't remove endpoints without warning

This comprehensive approach ensures smooth API evolution while maintaining client compatibility.
      `,
    },
  ],
};
