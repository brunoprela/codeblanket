export const customManagersQuerysetsQuiz = {
  title: 'Custom Managers & QuerySets - Discussion Questions',
  questions: [
    {
      question:
        'Explain the difference between custom managers and custom QuerySets in Django. Provide examples of when you would use each and how to combine them effectively for reusable query logic.',
      answer: `
**Custom Managers vs Custom QuerySets:**

**Custom Managers:**
- Defined at model level
- Entry point for queries
- Can change initial QuerySet
- Good for model-wide defaults

Example:
\`\`\`python
class PublishedManager(models.Manager):
    def get_queryset(self):
        return super().get_queryset().filter(status='published')

class Article(models.Model):
    objects = models.Manager()  # Default
    published = PublishedManager()  # Custom

# Usage:
Article.published.all()  # Only published articles
\`\`\`

**Custom QuerySets:**
- Chainable methods
- Reusable query logic
- Can be combined
- Better for complex filters

Example:
\`\`\`python
class ArticleQuerySet(models.QuerySet):
    def published(self):
        return self.filter(status='published')
    
    def featured(self):
        return self.filter(featured=True)
    
    def recent(self):
        from django.utils import timezone
        from datetime.timedelta import timedelta
        week_ago = timezone.now() - timedelta(days=7)
        return self.filter(published_at__gte=week_ago)

class Article(models.Model):
    objects = ArticleQuerySet.as_manager()

# Usage (chainable!):
Article.objects.published().featured().recent()
\`\`\`

**Best Practice - Combining Both:**
\`\`\`python
class ArticleQuerySet(models.QuerySet):
    def published(self):
        return self.filter(status='published')
    
    def by_author(self, author):
        return self.filter(author=author)

class ArticleManager(models.Manager.from_queryset(ArticleQuerySet)):
    def get_queryset(self):
        return super().get_queryset().select_related('author', 'category')

class Article(models.Model):
    objects = ArticleManager()

# Combines manager optimization + queryset methods
Article.objects.published().by_author(user)  # Optimized + chainable
\`\`\`

**When to Use Each:**
- **Manager only**: Simple default filtering (e.g., soft deletes)
- **QuerySet only**: Complex chainable logic
- **Both**: Performance optimizations (manager) + flexibility (queryset)

This gives both performance (manager optimization) and flexibility (queryset methods).
      `,
    },
    {
      question:
        'Describe advanced QuerySet techniques including prefetch_related with Prefetch objects, conditional expressions with Case/When, and subquery annotations. Provide production examples.',
      answer: `
**1. Advanced prefetch_related with Prefetch:**

\`\`\`python
from django.db.models import Prefetch

# Prefetch with custom queryset
Article.objects.prefetch_related(
    Prefetch(
        'comments',
        queryset=Comment.objects.filter(approved=True).select_related('user')
    )
)

# Multiple prefetches with to_attr
Article.objects.prefetch_related(
    Prefetch('comments', queryset=Comment.objects.filter(approved=True), to_attr='approved_comments'),
    Prefetch('tags', queryset=Tag.objects.filter(active=True), to_attr='active_tags')
)

# Access:
for article in articles:
    for comment in article.approved_comments:  # Filtered + cached
        print(comment.user.name)  # No additional query
\`\`\`

**2. Conditional Expressions (Case/When):**

\`\`\`python
from django.db.models import Case, When, Value, F, Q, CharField, DecimalField

# Categorize products
Product.objects.annotate(
    price_category=Case(
        When(price__lt=10, then=Value('budget')),
        When(price__lt=50, then=Value('mid-range')),
        When(price__gte=50, then=Value('premium')),
        default=Value('uncategorized'),
        output_field=CharField()
    )
)

# Conditional aggregation
Order.objects.annotate(
    premium_items=Count(
        Case(When(items__price__gt=100, then=1))
    ),
    discount_amount=Case(
        When(total__gt=500, then=F('total') * 0.20),
        When(total__gt=200, then=F('total') * 0.10),
        default=Value(0),
        output_field=DecimalField()
    )
)
\`\`\`

**3. Subquery Annotations:**

\`\`\`python
from django.db.models import Subquery, OuterRef, Exists

# Annotate with latest comment
latest_comment = Comment.objects.filter(
    article=OuterRef('pk')
).order_by('-created_at')

Article.objects.annotate(
    latest_comment_text=Subquery(latest_comment.values('text')[:1]),
    latest_comment_date=Subquery(latest_comment.values('created_at')[:1])
)

# Exists subquery for filtering
has_recent_activity = Comment.objects.filter(
    article=OuterRef('pk'),
    created_at__gte=timezone.now() - timedelta(days=7)
)

Article.objects.filter(Exists(has_recent_activity))
\`\`\`

**Production Example - E-commerce Dashboard:**

\`\`\`python
from django.db.models import (
    Count, Sum, Avg, F, Q, Case, When,
    Prefetch, Subquery, OuterRef, Exists
)

# Complex product analytics
latest_review = Review.objects.filter(
    product=OuterRef('pk')
).order_by('-created_at')

products = Product.objects.annotate(
    # Counts
    review_count=Count('reviews'),
    order_count=Count('orders'),
    
    # Aggregates
    avg_rating=Avg('reviews__rating'),
    total_revenue=Sum(F('orders__quantity') * F('price')),
    
    # Conditionals
    performance_tier=Case(
        When(order_count__gte=100, then=Value('platinum')),
        When(order_count__gte=50, then=Value('gold')),
        When(order_count__gte=10, then=Value('silver')),
        default=Value('bronze'),
        output_field=CharField()
    ),
    
    # Subqueries
    latest_review_text=Subquery(latest_review.values('text')[:1]),
    has_recent_orders=Exists(
        Order.objects.filter(
            product=OuterRef('pk'),
            created_at__gte=timezone.now() - timedelta(days=30)
        )
    )
).select_related('category').prefetch_related(
    Prefetch(
        'reviews',
        queryset=Review.objects.filter(rating__gte=4).select_related('user')[:5],
        to_attr='top_reviews'
    )
).filter(
    Q(stock__gt=0) & Q(avg_rating__gte=3.5)
).order_by('-total_revenue')

# Single optimized query with all analytics
\`\`\`

These techniques enable complex analytics without N+1 queries or post-processing in Python.
      `,
    },
    {
      question:
        'Explain how to build a generic filtering system using QuerySet methods, Q objects, and dynamic field lookups. Include handling user input safely and building reusable filter classes.',
      answer: `
**Generic Filtering System:**

**1. Basic Dynamic Filters:**

\`\`\`python
class ArticleFilter:
    def __init__(self, queryset=None):
        self.queryset = queryset or Article.objects.all()
    
    def filter_by_fields(self, **filters):
        """Apply multiple filters dynamically"""
        query = Q()
        
        for field, value in filters.items():
            if value is not None:
                query &= Q(**{field: value})
        
        return self.queryset.filter(query)
    
    def search(self, search_term, fields):
        """Search across multiple fields"""
        query = Q()
        
        for field in fields:
            query |= Q(**{f'{field}__icontains': search_term})
        
        return self.queryset.filter(query)
    
    def filter_range(self, field, min_val=None, max_val=None):
        """Filter by range"""
        query = Q()
        
        if min_val is not None:
            query &= Q(**{f'{field}__gte': min_val})
        if max_val is not None:
            query &= Q(**{f'{field}__lte': max_val})
        
        return self.queryset.filter(query)

# Usage:
filter_obj = ArticleFilter()
results = filter_obj.filter_by_fields(
    status='published',
    featured=True
).search('django', ['title', 'content'])
\`\`\`

**2. Advanced Filter Class with Validation:**

\`\`\`python
from django.core.exceptions import ValidationError
from django.db.models import Q
import operator
from functools import reduce

class SafeQueryFilter:
    # Allowed operators
    OPERATORS = {
        'exact': '',
        'iexact': '__iexact',
        'contains': '__icontains',
        'gt': '__gt',
        'gte': '__gte',
        'lt': '__lt',
        'lte': '__lte',
        'in': '__in',
        'startswith': '__istartswith',
        'endswith': '__iendswith',
    }
    
    # Whitelist of filterable fields
    ALLOWED_FIELDS = {
        'title', 'status', 'author__username',
        'category__name', 'created_at', 'view_count'
    }
    
    def __init__(self, model_class):
        self.model = model_class
        self.queryset = model_class.objects.all()
    
    def filter(self, filters: dict):
        """Safely apply filters from user input"""
        q_objects = []
        
        for filter_expr, value in filters.items():
            # Parse filter expression
            parts = filter_expr.split('__')
            
            if len(parts) >= 2:
                field = '__'.join(parts[:-1])
                operator_key = parts[-1]
            else:
                field = parts[0]
                operator_key = 'exact'
            
            # Validate field is allowed
            if field not in self.ALLOWED_FIELDS:
                raise ValidationError(f'Filtering on {field} not allowed')
            
            # Validate operator
            if operator_key not in self.OPERATORS:
                raise ValidationError(f'Operator {operator_key} not allowed')
            
            # Build Q object
            lookup = f'{field}{self.OPERATORS[operator_key]}'
            q_objects.append(Q(**{lookup: value}))
        
        # Combine with AND
        if q_objects:
            self.queryset = self.queryset.filter(
                reduce(operator.and_, q_objects)
            )
        
        return self
    
    def search(self, search_term: str, fields: list = None):
        """Full-text search across multiple fields"""
        if not search_term:
            return self
        
        search_fields = fields or ['title', 'content']
        
        # Validate fields
        for field in search_fields:
            if field not in self.ALLOWED_FIELDS:
                raise ValidationError(f'Search on {field} not allowed')
        
        # Build OR query
        q_objects = [Q(**{f'{field}__icontains': search_term}) for field in search_fields]
        self.queryset = self.queryset.filter(reduce(operator.or_, q_objects))
        
        return self
    
    def order_by(self, *fields):
        """Safe ordering"""
        safe_fields = []
        
        for field in fields:
            # Handle descending order
            if field.startswith('-'):
                actual_field = field[1:]
                prefix = '-'
            else:
                actual_field = field
                prefix = ''
            
            if actual_field in self.ALLOWED_FIELDS:
                safe_fields.append(f'{prefix}{actual_field}')
        
        if safe_fields:
            self.queryset = self.queryset.order_by(*safe_fields)
        
        return self
    
    def get_results(self):
        return self.queryset

# Usage in API view:
def article_list_api(request):
    filter_obj = SafeQueryFilter(Article)
    
    # Parse query parameters
    filters = {}
    if request.GET.get('status'):
        filters['status__exact'] = request.GET['status']
    if request.GET.get('min_views'):
        filters['view_count__gte'] = int(request.GET['min_views'])
    if request.GET.get('category'):
        filters['category__name__iexact'] = request.GET['category']
    
    # Apply filters
    try:
        results = filter_obj.filter(filters).search(
            request.GET.get('q', ''),
            ['title', 'content']
        ).order_by('-created_at').get_results()
    except ValidationError as e:
        return JsonResponse({'error': str(e)}, status=400)
    
    return JsonResponse({'results': list(results.values())})
\`\`\`

**3. Using django-filter (Production-Ready):**

\`\`\`python
import django_filters
from django_filters import rest_framework as filters

class ArticleFilter(filters.FilterSet):
    title = filters.CharFilter(lookup_expr='icontains')
    min_views = filters.NumberFilter(field_name='view_count', lookup_expr='gte')
    max_views = filters.NumberFilter(field_name='view_count', lookup_expr='lte')
    created_after = filters.DateTimeFilter(field_name='created_at', lookup_expr='gte')
    created_before = filters.DateTimeFilter(field_name='created_at', lookup_expr='lte')
    author_name = filters.CharFilter(field_name='author__username', lookup_expr='icontains')
    
    class Meta:
        model = Article
        fields = ['status', 'featured', 'category']

# In DRF ViewSet:
class ArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    filter_backends = [filters.DjangoFilterBackend]
    filterset_class = ArticleFilter

# API calls:
# /api/articles/?status=published&min_views=100&title=django
# /api/articles/?author_name=john&created_after=2024-01-01
\`\`\`

**Security Considerations:**
1. ✅ Whitelist allowed fields
2. ✅ Validate operators
3. ✅ Sanitize user input
4. ✅ Limit query complexity
5. ✅ Use parameterized queries (ORM does this)
6. ❌ Never use raw SQL with user input
7. ❌ Don't allow arbitrary field access

This approach provides flexible, safe, and reusable filtering for APIs and views.
      `,
    },
  ],
};
