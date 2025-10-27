export const customManagersQuerysets = {
  title: 'Custom Managers & QuerySets',
  id: 'custom-managers-querysets',
  content: `
# Custom Managers & QuerySets

## Introduction

Django\'s **Managers** and **QuerySets** are the core of how you interact with the database. While Django provides default managers (\`objects\`), creating custom managers and QuerySets allows you to encapsulate common query logic, making your code more reusable and maintainable.

### What are Managers?

A **Manager** is the interface through which database query operations are provided to Django models. Every model has at least one manager: \`objects\`.

\`\`\`python
# Default manager
articles = Article.objects.all()  # 'objects' is the default manager
\`\`\`

### What are QuerySets?

A **QuerySet** is a lazy collection of database objects. QuerySets can be filtered, sliced, and ordered without hitting the database until you evaluate them.

By the end of this section, you'll understand:
- Creating custom managers for reusable queries
- Building custom QuerySet methods
- Chaining custom methods
- Using managers for model creation
- Manager inheritance
- Production patterns

---

## Why Custom Managers?

### Problem: Repeated Query Logic

\`\`\`python
# ❌ BAD: Repeated logic everywhere
# In views.py
published_articles = Article.objects.filter(
    status='published',
    published_at__lte=timezone.now()
).order_by('-published_at')

# In another view
published_articles = Article.objects.filter(
    status='published',
    published_at__lte=timezone.now()
).order_by('-published_at')

# In template tag
published_articles = Article.objects.filter(
    status='published',
    published_at__lte=timezone.now()
).order_by('-published_at')

# Problem: Logic duplicated, hard to maintain
\`\`\`

### Solution: Custom Manager

\`\`\`python
# ✅ GOOD: Centralized logic
class ArticleManager (models.Manager):
    def published (self):
        return self.filter(
            status='published',
            published_at__lte=timezone.now()
        ).order_by('-published_at')

class Article (models.Model):
    # ... fields ...
    objects = ArticleManager()

# Usage (everywhere)
published_articles = Article.objects.published()
# Clean, reusable, maintainable!
\`\`\`

---

## Creating Custom Managers

### Basic Custom Manager

\`\`\`python
from django.db import models
from django.utils import timezone

class ArticleManager (models.Manager):
    """Custom manager for Article model"""
    
    def published (self):
        """Return only published articles"""
        return self.filter(
            status='published',
            published_at__lte=timezone.now()
        )
    
    def draft (self):
        """Return only draft articles"""
        return self.filter (status='draft')
    
    def by_author (self, author):
        """Return articles by specific author"""
        return self.filter (author=author)
    
    def popular (self, threshold=1000):
        """Return articles with views above threshold"""
        return self.filter (view_count__gte=threshold)

class Article (models.Model):
    title = models.CharField (max_length=200)
    content = models.TextField()
    status = models.CharField (max_length=10, default='draft')
    author = models.ForeignKey('Author', on_delete=models.CASCADE)
    view_count = models.IntegerField (default=0)
    published_at = models.DateTimeField (null=True, blank=True)
    
    # Set custom manager
    objects = ArticleManager()
    
    class Meta:
        ordering = ['-published_at']

# Usage
Article.objects.published()  # All published articles
Article.objects.draft()  # All draft articles
Article.objects.popular (threshold=5000)  # Articles with 5000+ views
Article.objects.by_author (user)  # Articles by specific author

# Chain methods
Article.objects.published().filter (category__name='Tech')
Article.objects.popular().exclude (author=excluded_user)
\`\`\`

---

## Custom QuerySets (More Powerful)

### Why QuerySets?

Custom managers can't be chained. Custom QuerySets can:

\`\`\`python
# ❌ PROBLEM: Manager methods aren't chainable
Article.objects.published().popular()  # ERROR! 'published()' returns QuerySet
                                        # QuerySet doesn't have 'popular()' method

# ✅ SOLUTION: Custom QuerySet with chainable methods
Article.objects.published().popular().by_category('Tech')
# Works! All methods chainable
\`\`\`

### Creating Custom QuerySets

\`\`\`python
from django.db import models
from django.utils import timezone

class ArticleQuerySet (models.QuerySet):
    """Custom QuerySet with chainable methods"""
    
    def published (self):
        """Return published articles"""
        return self.filter(
            status='published',
            published_at__lte=timezone.now()
        )
    
    def draft (self):
        """Return draft articles"""
        return self.filter (status='draft')
    
    def featured (self):
        """Return featured articles"""
        return self.filter (featured=True)
    
    def popular (self, threshold=1000):
        """Return articles with views above threshold"""
        return self.filter (view_count__gte=threshold)
    
    def by_category (self, category):
        """Filter by category"""
        if isinstance (category, str):
            return self.filter (category__slug=category)
        return self.filter (category=category)
    
    def by_author (self, author):
        """Filter by author"""
        return self.filter (author=author)
    
    def recent (self, days=7):
        """Return articles from last N days"""
        since = timezone.now() - timezone.timedelta (days=days)
        return self.filter (published_at__gte=since)
    
    def with_related (self):
        """Optimize with select_related/prefetch_related"""
        return self.select_related(
            'author',
            'category'
        ).prefetch_related(
            'tags',
            'comments'
        )

# Connect QuerySet to Manager
class ArticleManager (models.Manager):
    def get_queryset (self):
        return ArticleQuerySet (self.model, using=self._db)
    
    # Proxy methods to QuerySet
    def published (self):
        return self.get_queryset().published()
    
    def draft (self):
        return self.get_queryset().draft()
    
    def featured (self):
        return self.get_queryset().featured()

# Or use QuerySet.as_manager() (easier!)
class Article (models.Model):
    title = models.CharField (max_length=200)
    content = models.TextField()
    status = models.CharField (max_length=10, default='draft')
    featured = models.BooleanField (default=False)
    author = models.ForeignKey('Author', on_delete=models.CASCADE)
    category = models.ForeignKey('Category', on_delete=models.CASCADE)
    view_count = models.IntegerField (default=0)
    published_at = models.DateTimeField (null=True, blank=True)
    
    # Use QuerySet as manager
    objects = ArticleQuerySet.as_manager()

# Usage: All methods are chainable!
articles = Article.objects.published().featured().popular(5000).by_category('tech')
# SELECT * FROM article 
# WHERE status='published' 
#   AND featured=true 
#   AND view_count >= 5000 
#   AND category.slug='tech';

# Chain with Django QuerySet methods
articles = Article.objects.published().filter(
    author__name__icontains='john'
).exclude(
    tags__name='deprecated'
).recent(30).with_related()

# All methods chainable in any order!
\`\`\`

---

## Manager Inheritance and Multiple Managers

### Multiple Managers

\`\`\`python
class PublishedArticleManager (models.Manager):
    def get_queryset (self):
        return super().get_queryset().filter (status='published')

class FeaturedArticleManager (models.Manager):
    def get_queryset (self):
        return super().get_queryset().filter(
            status='published',
            featured=True
        )

class Article (models.Model):
    title = models.CharField (max_length=200)
    status = models.CharField (max_length=10, default='draft')
    featured = models.BooleanField (default=False)
    
    # Multiple managers
    objects = models.Manager()  # Default - returns all
    published = PublishedArticleManager()  # Only published
    featured = FeaturedArticleManager()  # Only featured
    
    class Meta:
        ordering = ['-published_at']

# Usage
Article.objects.all()  # All articles (including drafts)
Article.published.all()  # Only published articles
Article.featured.all()  # Only featured, published articles

# Chain with QuerySet methods
Article.published.filter (category__name='Tech')
Article.featured.order_by('-view_count')[:10]
\`\`\`

### Default Manager Behavior

\`\`\`python
# First manager defined is the default
class Article (models.Model):
    title = models.CharField (max_length=200)
    
    # This is the default (first defined)
    published = PublishedArticleManager()
    # These are additional
    objects = models.Manager()

# Important: Default manager affects related lookups
author.article_set.all()  # Uses default manager (published)
# Only returns published articles!

# Solution: Make 'objects' the default
class Article (models.Model):
    # Default (returns all)
    objects = models.Manager()
    # Additional
    published = PublishedArticleManager()

# Now author.article_set.all() returns all articles
\`\`\`

---

## Custom Model Creation Methods

\`\`\`python
class ArticleQuerySet (models.QuerySet):
    # ... query methods ...
    
    def delete (self):
        """Soft delete by setting status to 'deleted'"""
        return super().update (status='deleted', deleted_at=timezone.now())
    
    def hard_delete (self):
        """Actually delete from database"""
        return super().delete()

class ArticleManager (models.Manager):
    def get_queryset (self):
        return ArticleQuerySet (self.model, using=self._db)
    
    def create_article (self, title, content, author, **kwargs):
        """Create article with defaults"""
        article = self.create(
            title=title,
            content=content,
            author=author,
            status='draft',
            slug=slugify (title),
            **kwargs
        )
        return article
    
    def publish_article (self, article):
        """Publish a draft article"""
        article.status = 'published'
        article.published_at = timezone.now()
        article.save()
        return article
    
    def create_and_publish (self, title, content, author, **kwargs):
        """Create and immediately publish"""
        article = self.create_article (title, content, author, **kwargs)
        return self.publish_article (article)

class Article (models.Model):
    title = models.CharField (max_length=200)
    content = models.TextField()
    author = models.ForeignKey('Author', on_delete=models.CASCADE)
    status = models.CharField (max_length=10, default='draft')
    published_at = models.DateTimeField (null=True, blank=True)
    deleted_at = models.DateTimeField (null=True, blank=True)
    
    objects = ArticleManager()

# Usage
article = Article.objects.create_article(
    title="Django Guide",
    content="...",
    author=user
)

# Or create and publish
article = Article.objects.create_and_publish(
    title="Django Guide",
    content="...",
    author=user
)

# Soft delete
Article.objects.filter (status='draft').delete()  # Soft delete
Article.objects.filter (status='draft').hard_delete()  # Hard delete
\`\`\`

---

## Real-World Examples

### E-commerce Product Manager

\`\`\`python
from django.db import models
from django.db.models import Q, Count, Avg
from decimal import Decimal

class ProductQuerySet (models.QuerySet):
    """Custom QuerySet for Products"""
    
    def active (self):
        """Return active products only"""
        return self.filter (is_active=True)
    
    def in_stock (self):
        """Return products in stock"""
        return self.filter (stock_quantity__gt=0)
    
    def out_of_stock (self):
        """Return out of stock products"""
        return self.filter (stock_quantity=0)
    
    def by_category (self, category):
        """Filter by category"""
        return self.filter (category__slug=category)
    
    def by_price_range (self, min_price=None, max_price=None):
        """Filter by price range"""
        qs = self
        if min_price:
            qs = qs.filter (price__gte=min_price)
        if max_price:
            qs = qs.filter (price__lte=max_price)
        return qs
    
    def on_sale (self):
        """Return products with active sale"""
        return self.filter(
            sale_price__isnull=False,
            sale_price__lt=models.F('price')
        )
    
    def popular (self, min_orders=10):
        """Return popular products"""
        return self.annotate(
            order_count=Count('orderitem')
        ).filter(
            order_count__gte=min_orders
        ).order_by('-order_count')
    
    def top_rated (self, min_rating=4.0):
        """Return top-rated products"""
        return self.annotate(
            avg_rating=Avg('reviews__rating')
        ).filter(
            avg_rating__gte=min_rating
        ).order_by('-avg_rating')
    
    def search (self, query):
        """Full-text search"""
        return self.filter(
            Q(name__icontains=query) |
            Q(description__icontains=query) |
            Q(sku__icontains=query)
        )
    
    def with_related (self):
        """Optimize queries"""
        return self.select_related(
            'category',
            'brand'
        ).prefetch_related(
            'images',
            'reviews'
        )
    
    def apply_discount (self, discount_percent):
        """Apply discount to queryset"""
        discount_multiplier = Decimal(1) - (Decimal (discount_percent) / Decimal(100))
        return self.update(
            sale_price=models.F('price') * discount_multiplier
        )

class Product (models.Model):
    name = models.CharField (max_length=200)
    sku = models.CharField (max_length=50, unique=True)
    description = models.TextField()
    price = models.DecimalField (max_digits=10, decimal_places=2)
    sale_price = models.DecimalField (max_digits=10, decimal_places=2, null=True, blank=True)
    stock_quantity = models.IntegerField (default=0)
    is_active = models.BooleanField (default=True)
    category = models.ForeignKey('Category', on_delete=models.CASCADE)
    brand = models.ForeignKey('Brand', on_delete=models.CASCADE)
    
    objects = ProductQuerySet.as_manager()

# Usage examples
products = Product.objects.active().in_stock().by_category('electronics')

sale_products = Product.objects.on_sale().popular().with_related()

search_results = Product.objects.active().in_stock().search('laptop').by_price_range(
    min_price=500,
    max_price=2000
)

top_products = Product.objects.active().top_rated (min_rating=4.5).popular (min_orders=50)

# Apply discount to category
Product.objects.by_category('winter-clothing').apply_discount(30)
\`\`\`

### User Manager with Authentication

\`\`\`python
from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.db import models

class UserQuerySet (models.QuerySet):
    """Custom QuerySet for User"""
    
    def active (self):
        """Return active users"""
        return self.filter (is_active=True)
    
    def staff (self):
        """Return staff users"""
        return self.filter (is_staff=True)
    
    def verified (self):
        """Return verified users"""
        return self.filter (email_verified=True)
    
    def by_role (self, role):
        """Filter by role"""
        return self.filter (role=role)
    
    def registered_after (self, date):
        """Users registered after date"""
        return self.filter (date_joined__gte=date)
    
    def with_profile (self):
        """Include related profile"""
        return self.select_related('profile')

class UserManager(BaseUserManager):
    """Custom manager for User"""
    
    def get_queryset (self):
        return UserQuerySet (self.model, using=self._db)
    
    def create_user (self, email, password=None, **extra_fields):
        """Create regular user"""
        if not email:
            raise ValueError('Email is required')
        
        email = self.normalize_email (email)
        user = self.model (email=email, **extra_fields)
        user.set_password (password)
        user.save (using=self._db)
        return user
    
    def create_superuser (self, email, password=None, **extra_fields):
        """Create superuser"""
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        
        return self.create_user (email, password, **extra_fields)
    
    def active (self):
        return self.get_queryset().active()
    
    def verified (self):
        return self.get_queryset().verified()

class User(AbstractUser):
    email = models.EmailField (unique=True)
    email_verified = models.BooleanField (default=False)
    role = models.CharField (max_length=20, default='user')
    
    objects = UserManager()
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

# Usage
User.objects.create_user (email='user@example.com', password='secret')
User.objects.create_superuser (email='admin@example.com', password='secret')

active_users = User.objects.active().verified()
staff_members = User.objects.staff().with_profile()
\`\`\`

---

## Manager Methods vs QuerySet Methods

### When to Use Each

**Manager Methods:**
- Model creation (\`create_user\`, \`create_article\`)
- Factory methods that return model instances
- Methods that don't need to be chained
- Methods that work across multiple models

**QuerySet Methods:**
- Filtering operations (\`published\`, \`active\`)
- Methods that need to be chainable
- Methods that return QuerySets
- Methods that can be combined with other filters

\`\`\`python
class ArticleQuerySet (models.QuerySet):
    # QuerySet methods (chainable, return QuerySet)
    def published (self):
        return self.filter (status='published')
    
    def by_category (self, category):
        return self.filter (category=category)

class ArticleManager (models.Manager):
    def get_queryset (self):
        return ArticleQuerySet (self.model, using=self._db)
    
    # Manager methods (factory, return model instance)
    def create_article (self, title, content, author):
        return self.create(
            title=title,
            content=content,
            author=author,
            slug=slugify (title)
        )
    
    # Proxy QuerySet methods for convenience
    def published (self):
        return self.get_queryset().published()
    
    def by_category (self, category):
        return self.get_queryset().by_category (category)

class Article (models.Model):
    title = models.CharField (max_length=200)
    content = models.TextField()
    author = models.ForeignKey('Author', on_delete=models.CASCADE)
    category = models.ForeignKey('Category', on_delete=models.CASCADE)
    status = models.CharField (max_length=10, default='draft')
    
    objects = ArticleManager()

# Manager method (factory)
article = Article.objects.create_article(
    title="Django Guide",
    content="...",
    author=user
)

# QuerySet methods (chainable)
articles = Article.objects.published().by_category (tech_category)
\`\`\`

---

## Best Practices

### 1. Use QuerySet.as_manager() for Simplicity

\`\`\`python
# ✅ GOOD: Simple and clean
class ArticleQuerySet (models.QuerySet):
    def published (self):
        return self.filter (status='published')

class Article (models.Model):
    title = models.CharField (max_length=200)
    objects = ArticleQuerySet.as_manager()

# All QuerySet methods automatically available on manager
Article.objects.published()
\`\`\`

### 2. Keep Query Logic in QuerySets, Creation Logic in Managers

\`\`\`python
class ArticleQuerySet (models.QuerySet):
    # Query logic here
    def published (self):
        return self.filter (status='published')

class ArticleManager (models.Manager):
    def get_queryset (self):
        return ArticleQuerySet (self.model, using=self._db)
    
    # Creation logic here
    def create_article (self, **kwargs):
        return self.create(**kwargs)
\`\`\`

### 3. Make Methods Chainable

\`\`\`python
# ✅ GOOD: All methods return self
class ArticleQuerySet (models.QuerySet):
    def published (self):
        return self.filter (status='published')  # Returns QuerySet
    
    def popular (self):
        return self.filter (view_count__gte=1000)  # Returns QuerySet

# Chainable
Article.objects.published().popular()
\`\`\`

### 4. Document Your Custom Methods

\`\`\`python
class ArticleQuerySet (models.QuerySet):
    def published (self):
        """
        Return published articles.
        
        Returns articles with status='published' and published_at in the past.
        Ordered by published_at descending.
        
        Returns:
            QuerySet: Filtered articles
            
        Example:
            >>> Article.objects.published()
            >>> Article.objects.published().filter (category='tech')
        """
        return self.filter(
            status='published',
            published_at__lte=timezone.now()
        ).order_by('-published_at')
\`\`\`

### 5. Use with_related() for Common Optimizations

\`\`\`python
class ArticleQuerySet (models.QuerySet):
    def with_related (self):
        """Optimize with select_related and prefetch_related"""
        return self.select_related('author', 'category').prefetch_related('tags')

# Usage
articles = Article.objects.published().with_related()
# Efficient queries with no N+1 problems
\`\`\`

---

## Summary

**Key Concepts:**1. **Custom Managers**: Encapsulate query logic at the manager level
2. **Custom QuerySets**: Chainable methods for flexible queries
3. **Multiple Managers**: Different default filters for different use cases
4. **Factory Methods**: Standardized model creation in managers
5. **QuerySet.as_manager()**: Simplest way to use custom QuerySets

**Benefits:**
- ✅ **DRY**: Reusable query logic
- ✅ **Maintainable**: Changes in one place
- ✅ **Testable**: Easy to test query methods
- ✅ **Chainable**: Combine methods flexibly
- ✅ **Readable**: Self-documenting code

**Production Patterns:**
- Use QuerySet methods for filtering (chainable)
- Use Manager methods for creation (factory)
- Document all custom methods
- Optimize common queries with \`with_related()\`
- Use multiple managers for different defaults
- Make methods chainable (return QuerySet)

Custom managers and QuerySets are essential for building maintainable Django applications. They keep your views clean and your query logic centralized and reusable.
`,
};
