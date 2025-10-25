export const djangoRestFrameworkFundamentals = {
  title: 'Django REST Framework Fundamentals',
  id: 'django-rest-framework-fundamentals',
  content: `
# Django REST Framework Fundamentals

## Introduction

**Django REST Framework (DRF)** is a powerful and flexible toolkit for building Web APIs in Django. It's the most popular way to build REST APIs with Django, used by companies like Mozilla, Red Hat, Heroku, and Eventbrite.

### Why DRF?

- **Browsable API**: Interactive web interface for your API
- **Authentication**: Multiple authentication schemes (JWT, OAuth, Token)
- **Serialization**: Convert complex data to JSON/XML
- **Validation**: Built-in request/response validation
- **Pagination**: Out-of-box pagination support
- **Permissions**: Granular access control
- **Throttling**: Rate limiting built-in
- **Documentation**: Auto-generated API docs

By the end of this section, you'll understand:
- DRF architecture and components
- Creating APIs with APIView and ViewSets
- Serializers for data transformation
- Request and Response objects
- Authentication and permissions basics
- Production API patterns

---

## Installation and Setup

\`\`\`bash
# Install DRF
pip install djangorestframework

# For browsable API rendering
pip install markdown
pip install django-filter
\`\`\`

\`\`\`python
# settings.py
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    
    # Third-party
    'rest_framework',
    
    # Your apps
    'articles',
]

# DRF Configuration
REST_FRAMEWORK = {
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
        'rest_framework.renderers.BrowsableAPIRenderer',
    ],
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
    ],
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.BasicAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticatedOrReadOnly',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
}
\`\`\`

---

## Your First API with APIView

### Simple Function-Based View

\`\`\`python
# views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import Article
from .serializers import ArticleSerializer

@api_view(['GET', 'POST'])
def article_list(request):
    """
    List all articles or create a new article
    """
    if request.method == 'GET':
        articles = Article.objects.all()
        serializer = ArticleSerializer(articles, many=True)
        return Response(serializer.data)
    
    elif request.method == 'POST':
        serializer = ArticleSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET', 'PUT', 'DELETE'])
def article_detail(request, pk):
    """
    Retrieve, update or delete an article
    """
    try:
        article = Article.objects.get(pk=pk)
    except Article.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)
    
    if request.method == 'GET':
        serializer = ArticleSerializer(article)
        return Response(serializer.data)
    
    elif request.method == 'PUT':
        serializer = ArticleSerializer(article, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    elif request.method == 'DELETE':
        article.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
\`\`\`

### URLs Configuration

\`\`\`python
# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('articles/', views.article_list),
    path('articles/<int:pk>/', views.article_detail),
]
\`\`\`

---

## Class-Based Views (APIView)

\`\`\`python
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import Http404

class ArticleList(APIView):
    """
    List all articles or create a new article
    """
    
    def get(self, request, format=None):
        articles = Article.objects.all()
        serializer = ArticleSerializer(articles, many=True)
        return Response(serializer.data)
    
    def post(self, request, format=None):
        serializer = ArticleSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(author=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ArticleDetail(APIView):
    """
    Retrieve, update or delete an article instance
    """
    
    def get_object(self, pk):
        try:
            return Article.objects.get(pk=pk)
        except Article.DoesNotExist:
            raise Http404
    
    def get(self, request, pk, format=None):
        article = self.get_object(pk)
        serializer = ArticleSerializer(article)
        return Response(serializer.data)
    
    def put(self, request, pk, format=None):
        article = self.get_object(pk)
        serializer = ArticleSerializer(article, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def delete(self, request, pk, format=None):
        article = self.get_object(pk)
        article.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

# urls.py
urlpatterns = [
    path('articles/', ArticleList.as_view()),
    path('articles/<int:pk>/', ArticleDetail.as_view()),
]
\`\`\`

---

## Generic Views

### Using Mixins

\`\`\`python
from rest_framework import mixins, generics

class ArticleList(mixins.ListModelMixin,
                  mixins.CreateModelMixin,
                  generics.GenericAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    
    def get(self, request, *args, **kwargs):
        return self.list(request, *args, **kwargs)
    
    def post(self, request, *args, **kwargs):
        return self.create(request, *args, **kwargs)

class ArticleDetail(mixins.RetrieveModelMixin,
                    mixins.UpdateModelMixin,
                    mixins.DestroyModelMixin,
                    generics.GenericAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    
    def get(self, request, *args, **kwargs):
        return self.retrieve(request, *args, **kwargs)
    
    def put(self, request, *args, **kwargs):
        return self.update(request, *args, **kwargs)
    
    def delete(self, request, *args, **kwargs):
        return self.destroy(request, *args, **kwargs)
\`\`\`

### Using Concrete Generic Views

\`\`\`python
from rest_framework import generics

class ArticleList(generics.ListCreateAPIView):
    """List all articles or create new article"""
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    
    def perform_create(self, serializer):
        """Set author to current user"""
        serializer.save(author=self.request.user)

class ArticleDetail(generics.RetrieveUpdateDestroyAPIView):
    """Retrieve, update or delete article"""
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
\`\`\`

---

## Serializers

### Basic Model Serializer

\`\`\`python
# serializers.py
from rest_framework import serializers
from .models import Article, Category, Tag

class ArticleSerializer(serializers.ModelSerializer):
    """Serializer for Article model"""
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'slug', 'content', 'author', 
                  'category', 'tags', 'status', 'published_at', 
                  'view_count', 'created_at', 'updated_at']
        read_only_fields = ['id', 'view_count', 'created_at', 'updated_at']
\`\`\`

### Nested Serializers

\`\`\`python
class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = ['id', 'name', 'slug']

class TagSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tag
        fields = ['id', 'name', 'slug']

class ArticleDetailSerializer(serializers.ModelSerializer):
    """Serializer with nested relationships"""
    category = CategorySerializer(read_only=True)
    tags = TagSerializer(many=True, read_only=True)
    author_name = serializers.CharField(source='author.get_full_name', read_only=True)
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'slug', 'content', 'excerpt',
                  'author', 'author_name', 'category', 'tags',
                  'status', 'published_at', 'view_count',
                  'created_at', 'updated_at']
        read_only_fields = ['id', 'view_count', 'created_at', 'updated_at']
\`\`\`

---

## Request and Response

### Request Object

\`\`\`python
class ArticleList(APIView):
    def post(self, request):
        # Access data
        title = request.data.get('title')
        
        # Query params
        status_filter = request.query_params.get('status')
        
        # User (if authenticated)
        user = request.user
        
        # Headers
        auth_header = request.META.get('HTTP_AUTHORIZATION')
        
        # Content type
        content_type = request.content_type
\`\`\`

### Response Object

\`\`\`python
from rest_framework.response import Response
from rest_framework import status

class ArticleList(APIView):
    def get(self, request):
        # Simple response
        return Response({'message': 'Hello'})
        
        # With status code
        return Response(
            {'error': 'Not found'}, 
            status=status.HTTP_404_NOT_FOUND
        )
        
        # With headers
        return Response(
            data,
            headers={'X-Custom-Header': 'value'}
        )
        
        # With specific status
        return Response(serializer.data, status=status.HTTP_201_CREATED)
\`\`\`

---

## Authentication

### Token Authentication

\`\`\`python
# settings.py
INSTALLED_APPS = [
    'rest_framework.authtoken',
]

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.TokenAuthentication',
    ],
}

# Migrate to create token table
python manage.py migrate

# Create tokens for users
from rest_framework.authtoken.models import Token
token = Token.objects.create(user=user)
print(token.key)
\`\`\`

### Login View to Get Token

\`\`\`python
from rest_framework.authtoken.views import obtain_auth_token
from django.urls import path

urlpatterns = [
    path('api-token-auth/', obtain_auth_token),
]

# Usage:
# POST /api-token-auth/
# {"username": "john", "password": "secret"}
# Returns: {"token": "9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b"}

# Use token in requests:
# curl -H "Authorization: Token 9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b" http://localhost:8000/api/articles/
\`\`\`

---

## Permissions

### Basic Permissions

\`\`\`python
from rest_framework import permissions

class ArticleList(generics.ListCreateAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    # GET (list) = Anyone
    # POST (create) = Authenticated only

class ArticleDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    permission_classes = [permissions.IsAuthenticated]
    # All methods require authentication
\`\`\`

### Custom Permissions

\`\`\`python
from rest_framework import permissions

class IsAuthorOrReadOnly(permissions.BasePermission):
    """
    Custom permission to only allow authors to edit their articles
    """
    
    def has_object_permission(self, request, view, obj):
        # Read permissions for any request (GET, HEAD, OPTIONS)
        if request.method in permissions.SAFE_METHODS:
            return True
        
        # Write permissions only for article author
        return obj.author == request.user

class ArticleDetail(generics.RetrieveUpdateDestroyAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    permission_classes = [permissions.IsAuthenticated, IsAuthorOrReadOnly]
\`\`\`

---

## Pagination

### Page Number Pagination

\`\`\`python
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20
}

# Response:
{
    "count": 100,
    "next": "http://api.example.com/articles/?page=2",
    "previous": null,
    "results": [...]
}
\`\`\`

### Custom Pagination

\`\`\`python
from rest_framework.pagination import PageNumberPagination

class StandardResultsSetPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100

class ArticleList(generics.ListAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    pagination_class = StandardResultsSetPagination

# Usage:
# /api/articles/ - 20 items per page
# /api/articles/?page_size=50 - 50 items per page
# /api/articles/?page_size=200 - Max 100 items (capped)
\`\`\`

---

## Filtering and Searching

### Basic Filtering

\`\`\`python
from django_filters import rest_framework as filters

class ArticleList(generics.ListAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    filter_backends = [filters.DjangoFilterBackend]
    filterset_fields = ['status', 'category', 'author']

# Usage:
# /api/articles/?status=published
# /api/articles/?category=1&status=published
\`\`\`

### Search

\`\`\`python
from rest_framework import filters

class ArticleList(generics.ListAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    filter_backends = [filters.SearchFilter]
    search_fields = ['title', 'content', 'author__username']

# Usage:
# /api/articles/?search=django
\`\`\`

### Ordering

\`\`\`python
class ArticleList(generics.ListAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    filter_backends = [filters.OrderingFilter]
    ordering_fields = ['published_at', 'view_count', 'title']
    ordering = ['-published_at']  # Default ordering

# Usage:
# /api/articles/?ordering=-view_count
# /api/articles/?ordering=title,-published_at
\`\`\`

---

## Error Handling

### Custom Exception Handler

\`\`\`python
# exceptions.py
from rest_framework.views import exception_handler
from rest_framework.response import Response

def custom_exception_handler(exc, context):
    """Custom exception handler"""
    # Call DRF's default handler first
    response = exception_handler(exc, context)
    
    if response is not None:
        # Customize error response
        response.data = {
            'error': {
                'status_code': response.status_code,
                'message': response.data.get('detail', 'An error occurred'),
                'errors': response.data
            }
        }
    
    return response

# settings.py
REST_FRAMEWORK = {
    'EXCEPTION_HANDLER': 'myapp.exceptions.custom_exception_handler'
}
\`\`\`

---

## Production Best Practices

### 1. Use Proper HTTP Status Codes

\`\`\`python
from rest_framework import status

class ArticleList(APIView):
    def post(self, request):
        serializer = ArticleSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
\`\`\`

### 2. Optimize Queries

\`\`\`python
class ArticleList(generics.ListAPIView):
    serializer_class = ArticleSerializer
    
    def get_queryset(self):
        return Article.objects.select_related(
            'author', 'category'
        ).prefetch_related('tags')
\`\`\`

### 3. Add API Versioning

\`\`\`python
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_VERSIONING_CLASS': 'rest_framework.versioning.URLPathVersioning',
}

# urls.py
urlpatterns = [
    path('api/v1/', include('myapp.urls')),
    path('api/v2/', include('myapp.urls_v2')),
]
\`\`\`

### 4. Rate Limiting

\`\`\`python
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle'
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '100/day',
        'user': '1000/day'
    }
}
\`\`\`

---

## Summary

**DRF Key Components:**
1. **Serializers**: Convert data to/from JSON
2. **Views**: Handle HTTP methods (GET, POST, etc.)
3. **Permissions**: Control access
4. **Authentication**: Verify user identity
5. **Pagination**: Split large result sets
6. **Filtering**: Query data efficiently

**Production Checklist:**
- ✅ Use proper serializers for each endpoint
- ✅ Implement authentication (Token/JWT)
- ✅ Add permissions for access control
- ✅ Enable pagination for list endpoints
- ✅ Optimize queries (select_related/prefetch_related)
- ✅ Add rate limiting
- ✅ Use API versioning
- ✅ Document endpoints
- ✅ Handle errors gracefully
- ✅ Test all endpoints

DRF provides everything needed to build production APIs quickly while following REST best practices. The framework handles the repetitive parts so you can focus on business logic.
\`,
};
