export const cachingDjango = {
  title: 'Caching in Django',
  id: 'caching-django',
  content: `
# Caching in Django

## Introduction

**Caching** is crucial for building high-performance Django applications. By storing expensive computation results or frequently accessed data, you can dramatically reduce database load and improve response times.

### Why Caching Matters

- **Performance**: Reduce page load times from seconds to milliseconds
- **Scalability**: Handle 10x more traffic with same infrastructure
- **Cost Savings**: Reduce database and server resources
- **User Experience**: Faster sites = happier users

**Real Impact:**
- Instagram: Caching reduced API response times from 500ms to 50ms
- Pinterest: 80% cache hit rate handles billions of requests
- Reddit: Aggressive caching serves millions with minimal database load

By the end of this section, you'll master:
- Django cache backends (Redis, Memcached)
- Per-view caching
- Template fragment caching
- Low-level cache API
- Cache invalidation strategies
- Production patterns

---

## Cache Backends

### Redis (Recommended)

\`\`\`python
# Install
pip install django-redis

# settings.py
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'PARSER_CLASS': 'redis.connection.HiredisParser',
            'CONNECTION_POOL_CLASS_KWARGS': {
                'max_connections': 50,
                'retry_on_timeout': True,
            },
            'SERIALIZER': 'django_redis.serializers.json.JSONSerializer',
            'COMPRESSOR': 'django_redis.compressors.zlib.ZlibCompressor',
        },
        'KEY_PREFIX': 'myapp',
        'TIMEOUT': 300,  # 5 minutes default
    }
}
\`\`\`

### Memcached

\`\`\`python
# Install
pip install pylibmc  # or python-memcached

# settings.py
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.memcached.PyLibMCCache',
        'LOCATION': '127.0.0.1:11211',
    }
}
\`\`\`

### Database Cache (Development)

\`\`\`python
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.db.DatabaseCache',
        'LOCATION': 'my_cache_table',
    }
}

# Create cache table
python manage.py createcachetable
\`\`\`

### File System Cache

\`\`\`python
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.filebased.FileBasedCache',
        'LOCATION': '/var/tmp/django_cache',
    }
}
\`\`\`

---

## Per-View Caching

### Function-Based Views

\`\`\`python
from django.views.decorators.cache import cache_page

@cache_page(60 * 15)  # Cache for 15 minutes
def article_list (request):
    articles = Article.objects.all()
    return render (request, 'articles/list.html', {'articles': articles})

# Per-user caching
@cache_page(60 * 15, key_prefix='user_%s' % request.user.id)
def my_articles (request):
    articles = Article.objects.filter (author=request.user)
    return render (request, 'articles/my_list.html', {'articles': articles})
\`\`\`

### Class-Based Views

\`\`\`python
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page

@method_decorator (cache_page(60 * 15), name='dispatch')
class ArticleListView(ListView):
    model = Article
    template_name = 'articles/list.html'
\`\`\`

### Conditional Caching

\`\`\`python
from django.views.decorators.cache import cache_page

def article_detail (request, slug):
    article = get_object_or_404(Article, slug=slug)
    
    # Only cache for published articles
    if article.status == 'published':
        return cache_page(60 * 60)(article_detail_cached)(request, article)
    
    return render (request, 'articles/detail.html', {'article': article})

def article_detail_cached (request, article):
    return render (request, 'articles/detail.html', {'article': article})
\`\`\`

---

## Template Fragment Caching

### Basic Template Caching

\`\`\`django
{% load cache %}

{% cache 500 sidebar %}
    <div class="sidebar">
        <!-- Expensive sidebar content -->
        {% for item in expensive_query %}
            {{ item }}
        {% endfor %}
    </div>
{% endcache %}
\`\`\`

### Cache with Variables

\`\`\`django
{% load cache %}

<!-- Cache per user -->
{% cache 600 user_profile request.user.id %}
    <div class="user-profile">
        {{ request.user.profile }}
    </div>
{% endcache %}

<!-- Cache per article -->
{% cache 300 article_sidebar article.id article.updated_at %}
    <div class="article-sidebar">
        <h3>Related Articles</h3>
        {% for related in article.get_related %}
            {{ related.title }}
        {% endfor %}
    </div>
{% endcache %}
\`\`\`

### Multiple Cache Fragments

\`\`\`django
{% load cache %}

<div class="page">
    <!-- Cache header (changes rarely) -->
    {% cache 3600 header %}
        {% include "includes/header.html" %}
    {% endcache %}
    
    <!-- Don't cache main content (dynamic) -->
    <main>
        {{ content }}
    </main>
    
    <!-- Cache sidebar (expensive query) -->
    {% cache 600 sidebar %}
        {% include "includes/sidebar.html" %}
    {% endcache %}
    
    <!-- Cache footer -->
    {% cache 3600 footer %}
        {% include "includes/footer.html" %}
    {% endcache %}
</div>
\`\`\`

---

## Low-Level Cache API

### Basic Operations

\`\`\`python
from django.core.cache import cache

# Set cache
cache.set('my_key', 'my_value', timeout=300)  # 5 minutes

# Get cache
value = cache.get('my_key')
if value is None:
    value = expensive_computation()
    cache.set('my_key', value, timeout=300)

# Get with default
value = cache.get('my_key', 'default_value')

# Delete cache
cache.delete('my_key')

# Set multiple
cache.set_many({
    'key1': 'value1',
    'key2': 'value2',
    'key3': 'value3',
}, timeout=300)

# Get multiple
values = cache.get_many(['key1', 'key2', 'key3'])
# Returns: {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}

# Add (only if doesn't exist)
cache.add('my_key', 'value')  # Returns False if key exists

# Increment/Decrement
cache.set('counter', 0)
cache.incr('counter')  # Returns 1
cache.incr('counter', delta=5)  # Returns 6
cache.decr('counter', delta=2)  # Returns 4

# Clear all cache
cache.clear()
\`\`\`

### Cache Patterns

\`\`\`python
# Get or set pattern
def get_article_views (article_id):
    cache_key = f'article_views_{article_id}'
    views = cache.get (cache_key)
    
    if views is None:
        # Not in cache, compute and cache
        views = Article.objects.get (id=article_id).view_count
        cache.set (cache_key, views, timeout=60)
    
    return views

# Try-catch pattern
def get_cached_data (key):
    try:
        return cache.get (key)
    except Exception as e:
        logger.error (f"Cache error: {e}")
        return None

# Cache expensive query
def get_popular_articles():
    cache_key = 'popular_articles'
    articles = cache.get (cache_key)
    
    if articles is None:
        articles = list(Article.objects.filter(
            view_count__gte=1000
        ).order_by('-view_count')[:10].values('id', 'title', 'slug'))
        cache.set (cache_key, articles, timeout=600)
    
    return articles
\`\`\`

---

## Caching Strategies

### Cache-Aside (Lazy Loading)

\`\`\`python
def get_article (article_id):
    """
    1. Check cache
    2. If miss, query database
    3. Update cache
    """
    cache_key = f'article_{article_id}'
    article = cache.get (cache_key)
    
    if article is None:
        # Cache miss
        article = Article.objects.get (id=article_id)
        cache.set (cache_key, article, timeout=3600)
    
    return article
\`\`\`

### Write-Through Cache

\`\`\`python
def update_article (article_id, data):
    """
    1. Update database
    2. Update cache
    """
    article = Article.objects.get (id=article_id)
    
    for key, value in data.items():
        setattr (article, key, value)
    article.save()
    
    # Update cache
    cache_key = f'article_{article_id}'
    cache.set (cache_key, article, timeout=3600)
    
    return article
\`\`\`

### Write-Behind Cache

\`\`\`python
def increment_view_count (article_id):
    """
    1. Update cache immediately
    2. Queue database update for later
    """
    cache_key = f'article_views_{article_id}'
    cache.incr (cache_key)
    
    # Queue background task to update database
    from .tasks import update_article_views
    update_article_views.delay (article_id)
\`\`\`

---

## Cache Invalidation

### Manual Invalidation

\`\`\`python
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver

@receiver (post_save, sender=Article)
def invalidate_article_cache (sender, instance, **kwargs):
    """Invalidate cache when article is saved"""
    cache_keys = [
        f'article_{instance.id}',
        f'article_list',
        f'category_{instance.category_id}_articles',
    ]
    cache.delete_many (cache_keys)

@receiver (post_delete, sender=Article)
def invalidate_on_delete (sender, instance, **kwargs):
    """Invalidate cache when article is deleted"""
    cache.delete (f'article_{instance.id}')
\`\`\`

### Cache Versioning

\`\`\`python
def get_article_with_version (article_id):
    """Use version in cache key"""
    article = Article.objects.get (id=article_id)
    cache_key = f'article_{article_id}_v{article.updated_at.timestamp()}'
    
    cached = cache.get (cache_key)
    if cached:
        return cached
    
    # Compute and cache
    result = expensive_transformation (article)
    cache.set (cache_key, result, timeout=None)  # Never expires
    return result
\`\`\`

### Cache Tags

\`\`\`python
from django_redis import get_redis_connection

def cache_with_tags (key, value, tags, timeout=300):
    """Cache with tags for group invalidation"""
    cache.set (key, value, timeout)
    
    # Store key in tag sets
    redis_conn = get_redis_connection("default")
    for tag in tags:
        redis_conn.sadd (f'tag:{tag}', key)
        redis_conn.expire (f'tag:{tag}', timeout)

def invalidate_by_tag (tag):
    """Invalidate all keys with this tag"""
    redis_conn = get_redis_connection("default")
    keys = redis_conn.smembers (f'tag:{tag}')
    
    if keys:
        cache.delete_many (keys)
        redis_conn.delete (f'tag:{tag}')

# Usage
cache_with_tags('article_1', article_data, ['articles', 'category_tech'])
invalidate_by_tag('articles')  # Invalidates all articles
\`\`\`

---

## DRF Caching

### Cache API Responses

\`\`\`python
from rest_framework.decorators import api_view
from django.views.decorators.cache import cache_page
from django.utils.decorators import method_decorator

# Function-based view
@api_view(['GET'])
@cache_page(60 * 15)
def article_list_api (request):
    articles = Article.objects.all()
    serializer = ArticleSerializer (articles, many=True)
    return Response (serializer.data)

# Class-based view
class ArticleViewSet (viewsets.ReadOnlyModelViewSet):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    
    @method_decorator (cache_page(60 * 15))
    def list (self, request, *args, **kwargs):
        return super().list (request, *args, **kwargs)
\`\`\`

### Per-User API Caching

\`\`\`python
from rest_framework.decorators import action
from django.core.cache import cache

class ArticleViewSet (viewsets.ModelViewSet):
    
    @action (detail=False, methods=['get'])
    def popular (self, request):
        # Cache per user
        cache_key = f'popular_articles_user_{request.user.id}'
        articles = cache.get (cache_key)
        
        if articles is None:
            articles = self.get_queryset().filter(
                view_count__gte=1000
            )[:10]
            serializer = self.get_serializer (articles, many=True)
            articles = serializer.data
            cache.set (cache_key, articles, timeout=600)
        
        return Response (articles)
\`\`\`

---

## Production Patterns

### Cache Warming

\`\`\`python
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Warm up cache with frequently accessed data'
    
    def handle (self, *args, **options):
        # Warm up popular articles
        popular = Article.objects.filter(
            view_count__gte=1000
        ).order_by('-view_count')[:100]
        
        for article in popular:
            cache_key = f'article_{article.id}'
            cache.set (cache_key, article, timeout=3600)
        
        self.stdout.write('Cache warmed successfully')

# Run: python manage.py warm_cache
\`\`\`

### Cache Monitoring

\`\`\`python
from django_redis import get_redis_connection
import logging

logger = logging.getLogger(__name__)

class CacheMonitor:
    """Monitor cache hit/miss rates"""
    
    @staticmethod
    def get_stats():
        redis_conn = get_redis_connection("default")
        info = redis_conn.info()
        
        return {
            'memory_used': info['used_memory_human'],
            'keys': info['db1']['keys'] if 'db1' in info else 0,
            'hit_rate': info['keyspace_hit_rate'] if 'keyspace_hit_rate' in info else 0,
        }
    
    @staticmethod
    def log_cache_access (key, hit):
        """Log cache hit/miss"""
        if hit:
            logger.info (f'Cache HIT: {key}')
        else:
            logger.info (f'Cache MISS: {key}')
\`\`\`

### Graceful Degradation

\`\`\`python
def get_cached_data_safe (key, fallback_fn, timeout=300):
    """
    Safely get cached data with fallback
    If cache fails, call fallback function
    """
    try:
        data = cache.get (key)
        if data is not None:
            return data
    except Exception as e:
        logger.error (f'Cache error: {e}')
    
    # Cache miss or error - call fallback
    try:
        data = fallback_fn()
        try:
            cache.set (key, data, timeout)
        except Exception as e:
            logger.error (f'Cache set error: {e}')
        return data
    except Exception as e:
        logger.error (f'Fallback error: {e}')
        raise
\`\`\`

---

## Summary

**Key Caching Strategies:**1. **Per-View Caching**: Cache entire views with @cache_page
2. **Template Fragment Caching**: Cache expensive template sections
3. **Low-Level API**: Fine-grained control with cache.get/set
4. **Cache-Aside**: Check cache, query DB on miss, update cache
5. **Write-Through**: Update cache when writing to database
6. **Cache Invalidation**: Clear cache when data changes

**Production Best Practices:**
- ✅ Use Redis for production (fast, feature-rich)
- ✅ Set appropriate timeout values
- ✅ Use cache versioning or tags
- ✅ Implement cache invalidation strategy
- ✅ Monitor cache hit rates
- ✅ Have graceful degradation for cache failures
- ✅ Warm cache after deployment
- ✅ Use different caches for different purposes
- ✅ Cache at the right level (view vs fragment vs query)
- ✅ Test cache invalidation thoroughly

**Common Patterns:**
- Cache expensive database queries
- Cache API responses (especially for public data)
- Cache template fragments with expensive logic
- Cache computed/aggregated data
- Use per-user caching for personalized content
- Invalidate cache on model save/delete
- Version cache keys to avoid stale data

Caching is one of the most effective ways to improve Django application performance. Start with simple view caching, then add granular caching as needed.
`,
};
