export const filteringSearchingPaginationQuiz = [
  {
    id: 1,
    question:
      'Explain django-filter integration with DRF, including FilterSet classes, custom filters, and filtering on related fields. Provide production examples.',
    answer: `
**Basic django-filter Setup:**

\`\`\`python
pip install django-filter

REST_FRAMEWORK = {
    'DEFAULT_FILTER_BACKENDS': [
        'django_filters.rest_framework.DjangoFilterBackend'
    ]
}

class ArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.all()
    filterset_fields = ['status', 'category', 'author']
    # /api/articles/?status=published&category=tech
\`\`\`

**Custom FilterSet:**

\`\`\`python
import django_filters

class ArticleFilter(django_filters.FilterSet):
    title = django_filters.CharFilter(lookup_expr='icontains')
    min_views = django_filters.NumberFilter(field_name='view_count', lookup_expr='gte')
    published_after = django_filters.DateFilter(field_name='published_at', lookup_expr='gte')
    author_name = django_filters.CharFilter(field_name='author__username', lookup_expr='icontains')
    
    class Meta:
        model = Article
        fields = ['status', 'featured']

class ArticleViewSet(viewsets.ModelViewSet):
    filterset_class = ArticleFilter
\`\`\`

**Related Field Filtering:**

\`\`\`python
class ArticleFilter(django_filters.FilterSet):
    category = django_filters.ModelChoiceFilter(queryset=Category.objects.all())
    tags = django_filters.ModelMultipleChoiceFilter(queryset=Tag.objects.all())
    has_comments = django_filters.BooleanFilter(method='filter_has_comments')
    
    def filter_has_comments(self, queryset, name, value):
        if value:
            return queryset.filter(comments__isnull=False).distinct()
        return queryset.filter(comments__isnull=True)
\`\`\`

**Production Example:**

\`\`\`python
class ArticleFilter(django_filters.FilterSet):
    search = django_filters.CharFilter(method='search_filter')
    min_rating = django_filters.NumberFilter(field_name='avg_rating', lookup_expr='gte')
    category__in = django_filters.BaseInFilter(field_name='category__slug')
    
    def search_filter(self, queryset, name, value):
        return queryset.filter(
            Q(title__icontains=value) | Q(content__icontains=value)
        )
    
    class Meta:
        model = Article
        fields = {
            'status': ['exact'],
            'published_at': ['gte', 'lte'],
            'view_count': ['gte', 'lte'],
        }
\`\`\`
      `,
  },
  {
    question:
      'Compare different DRF pagination classes (PageNumberPagination, LimitOffsetPagination, CursorPagination). When should you use each?',
    answer: `
**PageNumberPagination:**
Traditional page-based pagination.

\`\`\`python
class StandardPagination(PageNumberPagination):
    page_size = 25
    page_size_query_param = 'page_size'
    max_page_size = 100

# GET /articles/?page=2&page_size=50
\`\`\`

**Pros:** Intuitive, can jump to any page  
**Cons:** Slow for large datasets, inconsistent with real-time data

**LimitOffsetPagination:**
SQL LIMIT/OFFSET style.

\`\`\`python
class LimitOffsetPagination(pagination.LimitOffsetPagination):
    default_limit = 25
    max_limit = 100

# GET /articles/?limit=25&offset=50
\`\`\`

**Pros:** Flexible, SQL-like  
**Cons:** Same performance issues as PageNumber

**CursorPagination:**
Cursor-based for efficient large datasets.

\`\`\`python
class CursorPagination(pagination.CursorPagination):
    page_size = 25
    ordering = '-created_at'

# GET /articles/?cursor=cD0yMDIw...
\`\`\`

**Pros:** Constant performance, handles real-time data  
**Cons:** Can't jump to arbitrary pages, requires ordering

**Use Cases:**
- PageNumber: Small datasets, traditional UIs
- LimitOffset: APIs needing precise control
- Cursor: Large datasets, infinite scroll, real-time feeds

**Custom Pagination:**

\`\`\`python
class CustomPagination(PageNumberPagination):
    def get_paginated_response(self, data):
        return Response({
            'links': {
                'next': self.get_next_link(),
                'previous': self.get_previous_link()
            },
            'count': self.page.paginator.count,
            'total_pages': self.page.paginator.num_pages,
            'results': data
        })
\`\`\`
      `,
  },
  {
    question:
      'Describe DRF SearchFilter and OrderingFilter. How do you implement full-text search and optimize search performance?',
    answer: `
**SearchFilter:**

\`\`\`python
from rest_framework import filters

class ArticleViewSet(viewsets.ModelViewSet):
    filter_backends = [filters.SearchFilter]
    search_fields = ['title', 'content', '^author__username']
    # /api/articles/?search=django

# Prefixes:
# '^' - startswith
# '=' - exact match
# '@' - full-text search (PostgreSQL)
# '$' - regex
\`\`\`

**OrderingFilter:**

\`\`\`python
class ArticleViewSet(viewsets.ModelViewSet):
    filter_backends = [filters.OrderingFilter]
    ordering_fields = ['created_at', 'view_count', 'title']
    ordering = ['-created_at']  # default
    # /api/articles/?ordering=-view_count,title
\`\`\`

**Combined:**

\`\`\`python
class ArticleViewSet(viewsets.ModelViewSet):
    filter_backends = [
        DjangoFilterBackend,
        filters.SearchFilter,
        filters.OrderingFilter
    ]
    filterset_fields = ['status', 'category']
    search_fields = ['title', 'content']
    ordering_fields = ['created_at', 'view_count']
\`\`\`

**Full-Text Search (PostgreSQL):**

\`\`\`python
from django.contrib.postgres.search import SearchVector

class ArticleViewSet(viewsets.ModelViewSet):
    def get_queryset(self):
        qs = super().get_queryset()
        search = self.request.query_params.get('search')
        
        if search:
            qs = qs.annotate(
                search=SearchVector('title', 'content')
            ).filter(search=search)
        
        return qs
\`\`\`

**Search Optimization:**

\`\`\`python
# Add database index
class Article(models.Model):
    title = models.CharField(max_length=200, db_index=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['title', 'created_at']),
            GinIndex(fields=['search_vector']),  # PostgreSQL full-text
        ]
\`\`\`
      `,
  },
].map(({ id, ...q }, idx) => ({
  id: `django-q-${idx + 1}`,
  question: q.question,
  sampleAnswer: String(q.answer),
  keyPoints: [],
}));
