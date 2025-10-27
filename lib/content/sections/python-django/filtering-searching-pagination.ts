export const filteringSearchingPagination = {
  title: 'Filtering, Searching & Pagination',
  id: 'filtering-searching-pagination',
  content: `
# Filtering, Searching & Pagination

## Introduction

Efficient **filtering**, **searching**, and **pagination** are essential for building usable APIs. Users need to find specific data quickly and navigate large datasets efficiently.

### Why This Matters

- **Performance**: Reduce data transfer and query time
- **Usability**: Let users find what they need
- **Scalability**: Handle millions of records
- **Best Practices**: Follow REST API conventions

By the end of this section, you'll master:
- DjangoFilterBackend for complex filtering
- SearchFilter for full-text search
- OrderingFilter for sorting
- Pagination strategies
- Query optimization
- Production patterns

---

## Filtering with django-filter

### Installation and Setup

\`\`\`bash
pip install django-filter
\`\`\`

\`\`\`python
# settings.py
INSTALLED_APPS = [
    'django_filters',
]

REST_FRAMEWORK = {
    'DEFAULT_FILTER_BACKENDS': [
        'django_filters.rest_framework.DjangoFilterBackend',
    ],
}
\`\`\`

### Basic Filtering

\`\`\`python
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import generics

class ArticleList (generics.ListAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['status', 'category', 'author']

# Usage:
# GET /api/articles/?status=published
# GET /api/articles/?category=1&status=published
# GET /api/articles/?author=5
\`\`\`

### Custom FilterSet

\`\`\`python
import django_filters
from .models import Article

class ArticleFilter (django_filters.FilterSet):
    # Exact match
    status = django_filters.CharFilter (field_name='status', lookup_expr='exact')
    
    # Case-insensitive contains
    title = django_filters.CharFilter (field_name='title', lookup_expr='icontains')
    
    # Greater than / less than
    min_views = django_filters.NumberFilter (field_name='view_count', lookup_expr='gte')
    max_views = django_filters.NumberFilter (field_name='view_count', lookup_expr='lte')
    
    # Date range
    published_after = django_filters.DateFilter (field_name='published_at', lookup_expr='gte')
    published_before = django_filters.DateFilter (field_name='published_at', lookup_expr='lte')
    
    # Boolean
    featured = django_filters.BooleanFilter (field_name='featured')
    
    # Choice filter
    status = django_filters.ChoiceFilter (choices=Article.STATUS_CHOICES)
    
    # Multiple choice (CSV)
    category = django_filters.ModelMultipleChoiceFilter(
        queryset=Category.objects.all(),
        field_name='category',
        to_field_name='slug'
    )
    
    # Range filter
    view_count_range = django_filters.RangeFilter (field_name='view_count')
    
    class Meta:
        model = Article
        fields = ['status', 'title', 'featured']

class ArticleList (generics.ListAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    filterset_class = ArticleFilter

# Usage:
# GET /api/articles/?title=django
# GET /api/articles/?min_views=1000&max_views=10000
# GET /api/articles/?published_after=2024-01-01
# GET /api/articles/?view_count_range_min=100&view_count_range_max=1000
# GET /api/articles/?category=tech,python
\`\`\`

### Custom Filter Methods

\`\`\`python
class ArticleFilter (django_filters.FilterSet):
    author_name = django_filters.CharFilter (method='filter_by_author_name')
    has_comments = django_filters.BooleanFilter (method='filter_has_comments')
    published_year = django_filters.NumberFilter (method='filter_by_year')
    
    def filter_by_author_name (self, queryset, name, value):
        """Filter by author's name (case-insensitive)"""
        return queryset.filter(
            author__username__icontains=value
        ) | queryset.filter(
            author__first_name__icontains=value
        ) | queryset.filter(
            author__last_name__icontains=value
        )
    
    def filter_has_comments (self, queryset, name, value):
        """Filter articles with/without comments"""
        if value:
            return queryset.filter (comments__isnull=False).distinct()
        return queryset.filter (comments__isnull=True)
    
    def filter_by_year (self, queryset, name, value):
        """Filter by publication year"""
        return queryset.filter (published_at__year=value)
    
    class Meta:
        model = Article
        fields = []

# Usage:
# GET /api/articles/?author_name=john
# GET /api/articles/?has_comments=true
# GET /api/articles/?published_year=2024
\`\`\`

---

## Search

### Basic Search

\`\`\`python
from rest_framework import filters

class ArticleList (generics.ListAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    filter_backends = [filters.SearchFilter]
    search_fields = ['title', 'content', 'author__username']

# Usage:
# GET /api/articles/?search=django
# Searches in title, content, and author username
\`\`\`

### Search with Different Lookup Types

\`\`\`python
class ArticleList (generics.ListAPIView):
    search_fields = [
        'title',              # icontains (default)
        '=email',             # exact match
        '^title',             # startswith
        '@content',           # full-text search (PostgreSQL)
        '$slug',              # regex
    ]

# Examples:
# =email: Exact match
# ^title: Starts with
# @content: Full-text search (requires PostgreSQL)
# $slug: Regex pattern matching
\`\`\`

### Custom Search

\`\`\`python
from rest_framework import filters

class CustomSearchFilter (filters.SearchFilter):
    """Custom search with additional logic"""
    
    def filter_queryset (self, request, queryset, view):
        search_term = request.query_params.get('search', '')
        
        if not search_term:
            return queryset
        
        # Custom search logic
        return queryset.filter(
            Q(title__icontains=search_term) |
            Q(content__icontains=search_term) |
            Q(tags__name__icontains=search_term)
        ).distinct()

class ArticleList (generics.ListAPIView):
    filter_backends = [CustomSearchFilter]
\`\`\`

---

## Ordering

### Basic Ordering

\`\`\`python
from rest_framework import filters

class ArticleList (generics.ListAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    filter_backends = [filters.OrderingFilter]
    ordering_fields = ['published_at', 'view_count', 'title']
    ordering = ['-published_at']  # Default ordering

# Usage:
# GET /api/articles/?ordering=view_count  # Ascending
# GET /api/articles/?ordering=-view_count  # Descending
# GET /api/articles/?ordering=title,-published_at  # Multiple fields
\`\`\`

### Restrict Ordering Fields

\`\`\`python
class ArticleList (generics.ListAPIView):
    ordering_fields = ['published_at', 'view_count']  # Only these allowed
    ordering = ['-published_at']
\`\`\`

---

## Combining Filters, Search, and Ordering

\`\`\`python
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import filters

class ArticleList (generics.ListAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    filter_backends = [
        DjangoFilterBackend,
        filters.SearchFilter,
        filters.OrderingFilter,
    ]
    filterset_class = ArticleFilter
    search_fields = ['title', 'content', 'author__username']
    ordering_fields = ['published_at', 'view_count', 'title']
    ordering = ['-published_at']

# Usage - combine all three:
# GET /api/articles/?status=published&search=django&ordering=-view_count
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

# Usage:
# GET /api/articles/?page=1
# GET /api/articles/?page=2
\`\`\`

### Custom Page Number Pagination

\`\`\`python
from rest_framework.pagination import PageNumberPagination

class StandardResultsSetPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'  # Allow client to set page size
    max_page_size = 100  # Maximum page size
    
    def get_paginated_response (self, data):
        """Custom response format"""
        return Response({
            'links': {
                'next': self.get_next_link(),
                'previous': self.get_previous_link()
            },
            'total': self.page.paginator.count,
            'page': self.page.number,
            'page_size': self.page.paginator.per_page,
            'results': data
        })

class ArticleList (generics.ListAPIView):
    pagination_class = StandardResultsSetPagination

# Usage:
# GET /api/articles/?page=1&page_size=50
\`\`\`

### Limit/Offset Pagination

\`\`\`python
from rest_framework.pagination import LimitOffsetPagination

class StandardLimitOffsetPagination(LimitOffsetPagination):
    default_limit = 20
    max_limit = 100

# settings.py
REST_FRAMEWORK = {
    'DEFAULT_PAGINATION_CLASS': 'myapp.pagination.StandardLimitOffsetPagination',
}

# Response:
{
    "count": 100,
    "next": "http://api.example.com/articles/?limit=20&offset=20",
    "previous": null,
    "results": [...]
}

# Usage:
# GET /api/articles/?limit=20&offset=0  # First 20
# GET /api/articles/?limit=20&offset=20  # Next 20
\`\`\`

### Cursor Pagination (Best for Large Datasets)

\`\`\`python
from rest_framework.pagination import CursorPagination

class ArticleCursorPagination(CursorPagination):
    page_size = 20
    ordering = '-created_at'  # Must have consistent ordering
    cursor_query_param = 'cursor'

class ArticleList (generics.ListAPIView):
    pagination_class = ArticleCursorPagination

# Response:
{
    "next": "cD04ODY%3D",
    "previous": null,
    "results": [...]
}

# Advantages:
# - Consistent performance regardless of pagination depth
# - No duplicate/missing results when data changes
# - Better for real-time data

# Disadvantages:
# - Can't jump to specific page
# - Must paginate sequentially
\`\`\`

### Disable Pagination for Specific View

\`\`\`python
class ArticleList (generics.ListAPIView):
    pagination_class = None  # No pagination
\`\`\`

---

## Production Patterns

### Comprehensive ViewSet with All Features

\`\`\`python
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import viewsets, filters

class ArticleViewSet (viewsets.ModelViewSet):
    """
    Production-ready ViewSet with filtering, search, ordering, and pagination
    """
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    
    # Filtering, Search, Ordering
    filter_backends = [
        DjangoFilterBackend,
        filters.SearchFilter,
        filters.OrderingFilter,
    ]
    filterset_class = ArticleFilter
    search_fields = ['title', 'content', '@description']  # @ for full-text
    ordering_fields = ['published_at', 'view_count', 'title', 'created_at']
    ordering = ['-published_at']
    
    # Pagination
    pagination_class = StandardResultsSetPagination
    
    def get_queryset (self):
        """Optimize queries"""
        queryset = Article.objects.select_related('author', 'category')
        queryset = queryset.prefetch_related('tags')
        
        # Additional filtering based on user
        if not self.request.user.is_staff:
            queryset = queryset.filter (status='published')
        
        return queryset

# Usage:
# GET /api/articles/?status=published&search=django&ordering=-view_count&page=1&page_size=20
\`\`\`

### Query Optimization

\`\`\`python
class ArticleFilter (django_filters.FilterSet):
    """Optimized filter with select_related"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Optimize queries in filter
        if self.queryset is not None:
            self.queryset = self.queryset.select_related('author', 'category')
            self.queryset = self.queryset.prefetch_related('tags')
\`\`\`

### Caching Filtered Results

\`\`\`python
from django.core.cache import cache
import hashlib

class ArticleList (generics.ListAPIView):
    
    def list (self, request, *args, **kwargs):
        # Create cache key from query params
        cache_key = self.get_cache_key()
        
        # Try cache first
        cached_response = cache.get (cache_key)
        if cached_response:
            return Response (cached_response)
        
        # Get from database
        response = super().list (request, *args, **kwargs)
        
        # Cache for 5 minutes
        cache.set (cache_key, response.data, 300)
        
        return response
    
    def get_cache_key (self):
        """Generate cache key from query params"""
        query_string = self.request.META.get('QUERY_STRING', '')
        return f'article_list_{hashlib.md5(query_string.encode()).hexdigest()}'
\`\`\`

---

## Advanced Filtering Patterns

### Related Object Filtering

\`\`\`python
class ArticleFilter (django_filters.FilterSet):
    # Filter by related object fields
    category_name = django_filters.CharFilter(
        field_name='category__name',
        lookup_expr='icontains'
    )
    
    author_username = django_filters.CharFilter(
        field_name='author__username',
        lookup_expr='exact'
    )
    
    tag_names = django_filters.CharFilter (method='filter_by_tags')
    
    def filter_by_tags (self, queryset, name, value):
        """Filter by multiple tags (comma-separated)"""
        tag_list = [tag.strip() for tag in value.split(',')]
        return queryset.filter (tags__name__in=tag_list).distinct()

# Usage:
# GET /api/articles/?category_name=technology
# GET /api/articles/?tag_names=django,python
\`\`\`

### Date Range Filters

\`\`\`python
class ArticleFilter (django_filters.FilterSet):
    published_date_range = django_filters.DateFromToRangeFilter(
        field_name='published_at'
    )
    
    # Or use DateRangeFilter for common ranges
    published = django_filters.DateRangeFilter(
        field_name='published_at',
        choices=[
            ('today', 'Today'),
            ('yesterday', 'Yesterday'),
            ('week', 'Past 7 days'),
            ('month', 'This month'),
        ]
    )

# Usage:
# GET /api/articles/?published_date_range_after=2024-01-01&published_date_range_before=2024-12-31
# GET /api/articles/?published=week
\`\`\`

---

## Summary

**Key Concepts:**1. **Filtering**: django-filter for complex field filtering
2. **Search**: SearchFilter for text search across fields
3. **Ordering**: OrderingFilter for sorting results
4. **Pagination**: Page number, limit/offset, or cursor pagination

**Production Best Practices:**
- ✅ Use DjangoFilterBackend for complex filtering
- ✅ Combine filter, search, and ordering
- ✅ Optimize queries with select_related/prefetch_related
- ✅ Use cursor pagination for large datasets
- ✅ Cache filtered results
- ✅ Limit max page size
- ✅ Add indexes on filtered fields
- ✅ Document filter options
- ✅ Validate filter inputs
- ✅ Monitor query performance

**Common Patterns:**
- Custom FilterSet classes for complex logic
- Multiple filter backends combined
- Dynamic pagination based on client needs
- Caching expensive filter operations
- Related object filtering
- Date range filters
- Full-text search with PostgreSQL

Efficient filtering, searching, and pagination are essential for building usable, scalable APIs that handle large datasets gracefully.
`,
};
