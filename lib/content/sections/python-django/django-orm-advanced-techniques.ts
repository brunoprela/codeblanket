export const djangoOrmAdvancedTechniques = {
  title: 'Django ORM Advanced Techniques',
  id: 'django-orm-advanced-techniques',
  content: `
# Django ORM Advanced Techniques

## Introduction

Django's **Object-Relational Mapper (ORM)** is one of its most powerful features. It allows you to interact with databases using Python code instead of writing SQL. However, to build high-performance applications, you need to master advanced ORM techniques.

### The N+1 Query Problem

The most common performance issue in Django applications:

\`\`\`python
# ❌ BAD: N+1 queries
articles = Article.objects.all()  # 1 query
for article in articles:
    print(article.author.name)  # N queries (one per article!)
# Total: 1 + N queries = 101 queries for 100 articles!

# ✅ GOOD: Single query with JOIN
articles = Article.objects.select_related('author').all()  # 1 query with JOIN
for article in articles:
    print(article.author.name)  # No additional queries!
# Total: 1 query
\`\`\`

By the end of this section, you'll understand:
- select_related vs prefetch_related
- Query optimization techniques
- Aggregation and annotation
- Raw SQL when necessary
- Database transactions
- Query debugging and profiling

---

## select_related: Forward ForeignKey and OneToOne

### What is select_related?

**select_related** performs a SQL JOIN and returns related objects in the same query. Use it for:
- Forward ForeignKey relationships
- OneToOne relationships

### Example: Articles and Authors

\`\`\`python
# models.py
class Author(models.Model):
    name = models.CharField(max_length=200)
    email = models.EmailField()
    bio = models.TextField()

class Article(models.Model):
    title = models.CharField(max_length=200)
    author = models.ForeignKey(Author, on_delete=models.CASCADE)
    content = models.TextField()
    published_at = models.DateTimeField()

# ❌ BAD: N+1 queries
articles = Article.objects.all()
for article in articles:
    print(f"{article.title} by {article.author.name}")
# Queries:
# SELECT * FROM article;  -- 1 query
# SELECT * FROM author WHERE id = 1;  -- Query 1
# SELECT * FROM author WHERE id = 2;  -- Query 2
# ... (100 more queries for 100 articles)

# ✅ GOOD: 1 query with JOIN
articles = Article.objects.select_related('author').all()
for article in articles:
    print(f"{article.title} by {article.author.name}")
# Queries:
# SELECT article.*, author.* 
# FROM article 
# INNER JOIN author ON article.author_id = author.id;
# -- Single query!
\`\`\`

### Chaining select_related

\`\`\`python
# Multiple foreign keys
class Article(models.Model):
    title = models.CharField(max_length=200)
    author = models.ForeignKey(Author, on_delete=models.CASCADE)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)

# Select multiple relations
articles = Article.objects.select_related('author', 'category').all()

# Nested relations (author -> company)
class Author(models.Model):
    name = models.CharField(max_length=200)
    company = models.ForeignKey(Company, on_delete=models.CASCADE)

# Follow nested relationships with double underscore
articles = Article.objects.select_related('author__company').all()
for article in articles:
    print(f"{article.title} by {article.author.name} at {article.author.company.name}")
# Single query with nested JOINs!
\`\`\`

### When to Use select_related

**Use select_related when:**
- Accessing ForeignKey or OneToOne relationships
- You know you'll access the related object
- The relationship is 1-to-1 or many-to-1

**Don't use select_related for:**
- ManyToMany relationships (use prefetch_related)
- Reverse ForeignKey (use prefetch_related)
- When you don't access the related object

---

## prefetch_related: Reverse ForeignKey and ManyToMany

### What is prefetch_related?

**prefetch_related** performs separate queries for related objects and "joins" them in Python. Use it for:
- ManyToMany relationships
- Reverse ForeignKey relationships (e.g., author.articles.all())

### Example: Many-to-Many Relationships

\`\`\`python
# models.py
class Article(models.Model):
    title = models.CharField(max_length=200)
    tags = models.ManyToManyField('Tag', related_name='articles')

class Tag(models.Model):
    name = models.CharField(max_length=50)

# ❌ BAD: N+1 queries
articles = Article.objects.all()
for article in articles:
    tags = article.tags.all()  # N queries!
    print(f"{article.title}: {', '.join(tag.name for tag in tags)}")

# ✅ GOOD: 2 queries total
articles = Article.objects.prefetch_related('tags').all()
for article in articles:
    tags = article.tags.all()  # No query! Already fetched
    print(f"{article.title}: {', '.join(tag.name for tag in tags)}")

# Queries executed:
# 1. SELECT * FROM article;
# 2. SELECT * FROM tag 
#    INNER JOIN article_tag ON tag.id = article_tag.tag_id
#    WHERE article_tag.article_id IN (1, 2, 3, ...);
\`\`\`

### Reverse ForeignKey Relationships

\`\`\`python
# Get all authors with their articles
authors = Author.objects.prefetch_related('articles').all()

for author in authors:
    print(f"{author.name}:")
    for article in author.articles.all():  # No additional queries!
        print(f"  - {article.title}")

# Queries:
# 1. SELECT * FROM author;
# 2. SELECT * FROM article WHERE author_id IN (1, 2, 3, ...);
\`\`\`

### Nested Prefetches

\`\`\`python
# Prefetch articles with their tags
authors = Author.objects.prefetch_related('articles__tags').all()

for author in authors:
    for article in author.articles.all():
        print(f"{article.title}: {', '.join(tag.name for tag in article.tags.all())}")

# Queries:
# 1. SELECT * FROM author;
# 2. SELECT * FROM article WHERE author_id IN (...);
# 3. SELECT * FROM tag INNER JOIN article_tag ...;
\`\`\`

### Custom Prefetch with Prefetch Object

\`\`\`python
from django.db.models import Prefetch

# Prefetch with filtering
authors = Author.objects.prefetch_related(
    Prefetch(
        'articles',
        queryset=Article.objects.filter(status='published').order_by('-published_at')
    )
).all()

# Prefetch with select_related inside
authors = Author.objects.prefetch_related(
    Prefetch(
        'articles',
        queryset=Article.objects.select_related('category')
    )
).all()

# Store prefetch result in a custom attribute
authors = Author.objects.prefetch_related(
    Prefetch(
        'articles',
        queryset=Article.objects.filter(featured=True),
        to_attr='featured_articles'  # Store in this attribute
    )
).all()

for author in authors:
    print(f"{author.name}'s featured articles:")
    for article in author.featured_articles:  # Use custom attribute
        print(f"  - {article.title}")
\`\`\`

---

## Combining select_related and prefetch_related

\`\`\`python
# Get articles with:
# - author (ForeignKey) - use select_related
# - tags (ManyToMany) - use prefetch_related
# - author's company (nested ForeignKey) - use select_related

articles = Article.objects.select_related(
    'author',
    'author__company',
    'category'
).prefetch_related(
    'tags'
).all()

# Queries:
# 1. SELECT article.*, author.*, company.*, category.*
#    FROM article
#    INNER JOIN author ON ...
#    INNER JOIN company ON ...
#    INNER JOIN category ON ...;
# 2. SELECT tag.* FROM tag INNER JOIN article_tag ...;

for article in articles:
    print(f"{article.title}")
    print(f"  Author: {article.author.name} at {article.author.company.name}")
    print(f"  Category: {article.category.name}")
    print(f"  Tags: {', '.join(tag.name for tag in article.tags.all())}")
# No additional queries!
\`\`\`

---

## Aggregation and Annotation

### Count, Sum, Avg, Min, Max

\`\`\`python
from django.db.models import Count, Sum, Avg, Min, Max, F, Q

# Count articles per author
authors = Author.objects.annotate(article_count=Count('articles'))
for author in authors:
    print(f"{author.name}: {author.article_count} articles")

# Average views per article
Article.objects.aggregate(avg_views=Avg('view_count'))
# Returns: {'avg_views': 1250.5}

# Multiple aggregations
stats = Article.objects.aggregate(
    total_articles=Count('id'),
    total_views=Sum('view_count'),
    avg_views=Avg('view_count'),
    max_views=Max('view_count'),
    min_views=Min('view_count')
)
# Returns: {
#   'total_articles': 100,
#   'total_views': 125000,
#   'avg_views': 1250.0,
#   'max_views': 50000,
#   'min_views': 10
# }
\`\`\`

### Annotation Examples

\`\`\`python
# Annotate articles with comment count
articles = Article.objects.annotate(
    comment_count=Count('comments')
).order_by('-comment_count')

# Annotate with conditional count
articles = Article.objects.annotate(
    published_comment_count=Count(
        'comments',
        filter=Q(comments__status='approved')
    )
)

# Annotate with related fields
authors = Author.objects.annotate(
    total_views=Sum('articles__view_count'),
    avg_views=Avg('articles__view_count'),
    latest_article_date=Max('articles__published_at')
)

# Complex annotation with F expressions
articles = Article.objects.annotate(
    engagement_score=F('view_count') + F('like_count') * 5 + F('comment_count') * 10
).order_by('-engagement_score')
\`\`\`

### Subqueries

\`\`\`python
from django.db.models import OuterRef, Subquery

# Get latest comment for each article
latest_comment = Comment.objects.filter(
    article=OuterRef('pk')
).order_by('-created_at')

articles = Article.objects.annotate(
    latest_comment_text=Subquery(latest_comment.values('text')[:1])
)

# Get author's most viewed article
most_viewed_article = Article.objects.filter(
    author=OuterRef('pk')
).order_by('-view_count')

authors = Author.objects.annotate(
    most_viewed_article_title=Subquery(most_viewed_article.values('title')[:1]),
    most_viewed_article_views=Subquery(most_viewed_article.values('view_count')[:1])
)
\`\`\`

---

## F Expressions and Q Objects

### F Expressions: Reference Database Fields

\`\`\`python
from django.db.models import F

# Increment view count (atomic operation)
Article.objects.filter(pk=1).update(view_count=F('view_count') + 1)

# Compare two fields
articles = Article.objects.filter(like_count__gt=F('view_count') / 10)
# Get articles where likes > 10% of views

# Use in ordering
articles = Article.objects.order_by(F('published_at').desc(nulls_last=True))

# Mathematical operations
articles = Article.objects.annotate(
    engagement_rate=(F('comment_count') + F('like_count')) * 100.0 / F('view_count')
).filter(engagement_rate__gt=5.0)
\`\`\`

### Q Objects: Complex Queries

\`\`\`python
from django.db.models import Q

# OR queries
articles = Article.objects.filter(
    Q(status='published') | Q(status='featured')
)

# AND queries (equivalent to comma-separated filters)
articles = Article.objects.filter(
    Q(status='published') & Q(view_count__gt=1000)
)

# NOT queries
articles = Article.objects.filter(
    ~Q(status='draft')
)

# Complex nested queries
articles = Article.objects.filter(
    (Q(status='published') | Q(status='featured')) &
    Q(view_count__gt=1000) &
    (Q(category__name='Tech') | Q(category__name='Science'))
)

# Dynamic query building
filters = Q()

if status:
    filters &= Q(status=status)

if min_views:
    filters &= Q(view_count__gte=min_views)

if search_term:
    filters &= Q(title__icontains=search_term) | Q(content__icontains=search_term)

articles = Article.objects.filter(filters)
\`\`\`

---

## only() and defer(): Optimize Field Selection

\`\`\`python
# only(): Fetch ONLY specified fields (reduce data transfer)
articles = Article.objects.only('id', 'title', 'published_at')
# SELECT id, title, published_at FROM article;

# Accessing other fields triggers additional queries
for article in articles:
    print(article.title)  # OK - no query
    print(article.content)  # Additional query!

# defer(): Fetch all fields EXCEPT specified ones
articles = Article.objects.defer('content', 'raw_html')
# SELECT id, title, author_id, ... FROM article;  (excludes content, raw_html)

# Accessing deferred fields triggers additional queries
for article in articles:
    print(article.title)  # OK - no query
    print(article.content)  # Additional query!

# Combine with select_related
articles = Article.objects.select_related('author').only(
    'id', 'title', 'author__name'
)
# SELECT article.id, article.title, author.name FROM ...
\`\`\`

---

## values() and values_list(): Dictionary/Tuple Results

\`\`\`python
# values(): Returns list of dictionaries
articles = Article.objects.values('id', 'title', 'author__name')
# [
#   {'id': 1, 'title': 'Django Guide', 'author__name': 'John'},
#   {'id': 2, 'title': 'Python Tips', 'author__name': 'Jane'},
# ]

# values_list(): Returns list of tuples
articles = Article.objects.values_list('id', 'title')
# [(1, 'Django Guide'), (2, 'Python Tips')]

# flat=True for single field (returns flat list)
article_titles = Article.objects.values_list('title', flat=True)
# ['Django Guide', 'Python Tips', ...]

# named=True for named tuples
articles = Article.objects.values_list('id', 'title', named=True)
# [Row(id=1, title='Django Guide'), Row(id=2, title='Python Tips')]
for article in articles:
    print(article.id, article.title)

# Performance: values/values_list avoid creating model instances
# Use when you only need specific fields for read-only operations
\`\`\`

---

## Raw SQL and Transactions

### Raw SQL Queries

\`\`\`python
# Execute raw SQL when ORM isn't enough
articles = Article.objects.raw(
    'SELECT * FROM article WHERE view_count > %s ORDER BY published_at DESC',
    [1000]
)

# Still returns model instances
for article in articles:
    print(article.title)

# Direct database cursor for complex queries
from django.db import connection

def complex_analytics():
    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT 
                a.category_id,
                c.name as category_name,
                COUNT(*) as article_count,
                SUM(a.view_count) as total_views,
                AVG(a.view_count) as avg_views
            FROM article a
            JOIN category c ON a.category_id = c.id
            WHERE a.status = 'published'
            GROUP BY a.category_id, c.name
            ORDER BY total_views DESC
        """)
        
        columns = [col[0] for col in cursor.description]
        results = [
            dict(zip(columns, row))
            for row in cursor.fetchall()
        ]
    
    return results
\`\`\`

### Database Transactions

\`\`\`python
from django.db import transaction

# Atomic decorator: All or nothing
@transaction.atomic
def create_article_with_tags(title, content, tag_names):
    """Create article and tags atomically"""
    article = Article.objects.create(
        title=title,
        content=content,
        status='published'
    )
    
    for tag_name in tag_names:
        tag, created = Tag.objects.get_or_create(name=tag_name)
        article.tags.add(tag)
    
    # If any error occurs, everything rolls back
    return article

# Atomic context manager
def transfer_authorship(article_id, new_author_id):
    try:
        with transaction.atomic():
            article = Article.objects.select_for_update().get(pk=article_id)
            old_author = article.author
            
            # Update article author
            article.author_id = new_author_id
            article.save()
            
            # Update stats
            old_author.article_count = F('article_count') - 1
            old_author.save()
            
            new_author = Author.objects.get(pk=new_author_id)
            new_author.article_count = F('article_count') + 1
            new_author.save()
            
    except Article.DoesNotExist:
        raise ValueError("Article not found")

# select_for_update(): Lock rows for update (prevent race conditions)
with transaction.atomic():
    article = Article.objects.select_for_update().get(pk=1)
    article.view_count += 1
    article.save()
# Row is locked until transaction commits
\`\`\`

### Savepoints

\`\`\`python
# Nested transactions with savepoints
def create_article_with_rollback():
    with transaction.atomic():
        article = Article.objects.create(title="Main Article")
        
        # Create savepoint
        sid = transaction.savepoint()
        
        try:
            # This might fail
            article.tags.add(invalid_tag)
        except Exception:
            # Rollback to savepoint (article still created)
            transaction.savepoint_rollback(sid)
        else:
            # Commit savepoint
            transaction.savepoint_commit(sid)
        
        return article
\`\`\`

---

## Query Optimization Techniques

### 1. Use exists() Instead of count()

\`\`\`python
# ❌ BAD: Counts all rows
if Article.objects.filter(author=user).count() > 0:
    print("User has articles")

# ✅ GOOD: Stops at first match
if Article.objects.filter(author=user).exists():
    print("User has articles")
\`\`\`

### 2. Use iterator() for Large QuerySets

\`\`\`python
# ❌ BAD: Loads all 1 million articles into memory
articles = Article.objects.all()
for article in articles:
    process(article)

# ✅ GOOD: Fetches in chunks (default 2000)
for article in Article.objects.all().iterator():
    process(article)

# Custom chunk size
for article in Article.objects.all().iterator(chunk_size=1000):
    process(article)
\`\`\`

### 3. Use bulk_create() for Bulk Inserts

\`\`\`python
# ❌ BAD: N queries
for i in range(1000):
    Article.objects.create(title=f"Article {i}")

# ✅ GOOD: Single query
articles = [
    Article(title=f"Article {i}")
    for i in range(1000)
]
Article.objects.bulk_create(articles, batch_size=1000)
\`\`\`

### 4. Use bulk_update() for Bulk Updates

\`\`\`python
# Update multiple articles efficiently
articles = Article.objects.filter(status='draft')[:100]
for article in articles:
    article.status = 'published'

# Bulk update (single query)
Article.objects.bulk_update(articles, ['status'], batch_size=100)
\`\`\`

### 5. Use update() Instead of save() for Bulk Updates

\`\`\`python
# ❌ BAD: Loads objects, updates one by one
articles = Article.objects.filter(category__name='Tech')
for article in articles:
    article.featured = True
    article.save()

# ✅ GOOD: Single UPDATE query
Article.objects.filter(category__name='Tech').update(featured=True)
\`\`\`

---

## Debugging Queries

### Log SQL Queries

\`\`\`python
# settings.py
LOGGING = {
    'version': 1,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'django.db.backends': {
            'handlers': ['console'],
            'level': 'DEBUG',
        },
    },
}

# All SQL queries will be printed to console
\`\`\`

### Using django-debug-toolbar

\`\`\`python
# Install
pip install django-debug-toolbar

# settings.py
INSTALLED_APPS = [
    ...
    'debug_toolbar',
]

MIDDLEWARE = [
    ...
    'debug_toolbar.middleware.DebugToolbarMiddleware',
]

INTERNAL_IPS = ['127.0.0.1']

# urls.py
import debug_toolbar

urlpatterns = [
    ...
    path('__debug__/', include(debug_toolbar.urls)),
]

# Shows SQL queries, execution time, and more in sidebar
\`\`\`

### Programmatic Query Inspection

\`\`\`python
from django.db import connection

# Get all executed queries
queries = connection.queries
for query in queries:
    print(f"SQL: {query['sql']}")
    print(f"Time: {query['time']}")

# Reset query log
from django.db import reset_queries
reset_queries()

# Context manager to track queries
from django.test.utils import CaptureQueriesContext

with CaptureQueriesContext(connection) as queries:
    articles = Article.objects.select_related('author').all()
    list(articles)  # Force evaluation

print(f"Number of queries: {len(queries)}")
for query in queries:
    print(query['sql'])
\`\`\`

---

## Summary

**Key ORM Optimization Techniques:**

1. **Use select_related** for ForeignKey/OneToOne (SQL JOIN)
2. **Use prefetch_related** for ManyToMany/reverse ForeignKey (separate queries)
3. **Use only/defer** to fetch only needed fields
4. **Use values/values_list** for read-only operations
5. **Use F expressions** for database-side operations
6. **Use Q objects** for complex queries
7. **Use aggregation/annotation** to compute in database
8. **Use bulk operations** for inserting/updating many objects
9. **Use exists()** instead of count() to check existence
10. **Use iterator()** for large QuerySets
11. **Use transactions** for data consistency
12. **Debug with django-debug-toolbar**

**Performance Checklist:**
- ✅ Avoid N+1 queries (use select_related/prefetch_related)
- ✅ Use database indexes on frequently queried fields
- ✅ Fetch only needed fields (only/defer/values)
- ✅ Use bulk operations for mass updates
- ✅ Use database aggregation instead of Python loops
- ✅ Profile queries with django-debug-toolbar
- ✅ Use database transactions for consistency
- ✅ Consider caching for frequently accessed data

Mastering these techniques will dramatically improve your Django application's performance and scalability.
`,
};
