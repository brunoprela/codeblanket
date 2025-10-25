export const cachingDjangoQuiz = [
  {
    id: 1,
    question:
      "Explain Django's caching framework including cache backends (Redis, Memcached, Database). Compare their performance characteristics and use cases.",
    answer: `
**Django Cache Backends:**

**Redis:**
\`\`\`python
pip install django-redis

CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}
\`\`\`

**Pros:** Persistent, rich data structures, pub/sub  
**Cons:** Requires separate server

**Memcached:**
\`\`\`python
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.memcached.PyMemcacheCache',
        'LOCATION': '127.0.0.1:11211',
    }
}
\`\`\`

**Pros:** Fast, simple, battle-tested  
**Cons:** No persistence, limited data types

**Database:**
\`\`\`python
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.db.DatabaseCache',
        'LOCATION': 'my_cache_table',
    }
}
\`\`\`

**Pros:** No additional infrastructure  
**Cons:** Slower, adds DB load

**Local Memory:**
\`\`\`python
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'unique-snowflake',
    }
}
\`\`\`

**Pros:** Fast for development  
**Cons:** Not shared across processes

**Performance Comparison:**
- Redis: ~10-50k ops/sec
- Memcached: ~100k ops/sec
- Database: ~1-5k ops/sec
- LocMem: Fastest, but process-local

**Use Cases:**
- Redis: Session storage, complex data, persistence needed
- Memcached: High-throughput simple caching
- Database: Small apps, no extra infrastructure
- LocMem: Development only
      `,
  },
  {
    question:
      'Describe cache invalidation strategies in Django. Include examples of cache key generation, TTL management, and cache warming.',
    answer: `
**Cache Invalidation Strategies:**

**1. Time-based (TTL):**
\`\`\`python
from django.core.cache import cache

cache.set('user:123', user_data, timeout=3600)  # 1 hour
\`\`\`

**2. Event-based (Signals):**
\`\`\`python
from django.db.models.signals import post_save
from django.dispatch import receiver

@receiver (post_save, sender=Article)
def invalidate_article_cache (sender, instance, **kwargs):
    cache_key = f'article:{instance.id}'
    cache.delete (cache_key)
    cache.delete('articles:list')  # Invalidate list too
\`\`\`

**3. Version-based:**
\`\`\`python
VERSION = 1
cache_key = f'article:{article_id}:v{VERSION}'
cache.set (cache_key, data)
# Increment VERSION to invalidate all keys
\`\`\`

**Cache Key Generation:**

\`\`\`python
def generate_cache_key (prefix, *args, **kwargs):
    key_parts = [prefix] + [str (arg) for arg in args]
    for k, v in sorted (kwargs.items()):
        key_parts.append (f'{k}:{v}')
    return ':'.join (key_parts)

# Example
key = generate_cache_key('articles', 'list', status='published', page=1)
# 'articles:list:page:1:status:published'
\`\`\`

**TTL Management:**

\`\`\`python
CACHE_TTL = {
    'short': 300,      # 5 minutes
    'medium': 3600,    # 1 hour
    'long': 86400,     # 1 day
}

cache.set('hot_articles', data, timeout=CACHE_TTL['short'])
cache.set('categories', data, timeout=CACHE_TTL['long'])
\`\`\`

**Cache Warming:**

\`\`\`python
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    def handle (self, *args, **options):
        # Warm popular articles
        popular = Article.objects.filter (featured=True)[:10]
        for article in popular:
            key = f'article:{article.id}'
            cache.set (key, article.to_dict(), timeout=3600)
        
        # Warm category lists
        for category in Category.objects.all():
            key = f'category:{category.slug}:articles'
            articles = category.articles.all()[:20]
            cache.set (key, list (articles.values()), timeout=3600)
\`\`\`

**Best Practices:**
- Short TTL for frequently changing data
- Event-based invalidation for critical data
- Versioning for coordinated cache clears
- Warm cache during off-peak hours
      `,
  },
  {
    question:
      'Explain per-view caching, template fragment caching, and low-level cache API in Django. When should you use each approach?',
    answer: `
**Per-View Caching:**

\`\`\`python
from django.views.decorators.cache import cache_page

@cache_page(60 * 15)  # 15 minutes
def article_list (request):
    articles = Article.objects.all()
    return render (request, 'articles.html', {'articles': articles})

# Per-user caching
@cache_page(60 * 15, key_prefix='user_%(user_id)s')
def dashboard (request):
    return render (request, 'dashboard.html')
\`\`\`

**Template Fragment Caching:**

\`\`\`django
{% load cache %}

{% cache 500 sidebar request.user.username %}
    <div class="sidebar">
        {% for item in sidebar_items %}
            <p>{{ item }}</p>
        {% endfor %}
    </div>
{% endcache %}
\`\`\`

**Low-Level Cache API:**

\`\`\`python
from django.core.cache import cache

# Basic operations
cache.set('key', 'value', timeout=300)
value = cache.get('key')
cache.delete('key')

# Get or set pattern
articles = cache.get('articles:featured')
if articles is None:
    articles = Article.objects.filter (featured=True)
    cache.set('articles:featured', articles, timeout=3600)

# Many keys
cache.set_many({'a': 1, 'b': 2, 'c': 3}, timeout=300)
values = cache.get_many(['a', 'b', 'c'])

# Atomic operations
cache.add('key', 'value')  # Only if doesn't exist
cache.incr('counter')
cache.decr('counter')
\`\`\`

**DRF API Caching:**

\`\`\`python
from rest_framework.decorators import action
from django.utils.decorators import method_decorator

class ArticleViewSet (viewsets.ModelViewSet):
    @method_decorator (cache_page(60 * 15))
    @action (detail=False)
    def featured (self, request):
        articles = self.get_queryset().filter (featured=True)
        serializer = self.get_serializer (articles, many=True)
        return Response (serializer.data)
\`\`\`

**When to Use Each:**

1. **Per-View Caching:**
   - Static/semi-static pages
   - Public content
   - Entire response can be cached

2. **Template Fragment:**
   - Parts of page change at different rates
   - User-specific content mixed with shared
   - Complex templates with expensive queries

3. **Low-Level API:**
   - Fine-grained control needed
   - Computed values, query results
   - Custom cache logic
   - API responses in viewsets

**Production Pattern:**

\`\`\`python
def get_articles_cached (status='published', category=None):
    key = f'articles:{status}:{category or "all"}'
    articles = cache.get (key)
    
    if articles is None:
        articles = Article.objects.filter (status=status)
        if category:
            articles = articles.filter (category=category)
        articles = list (articles.values())
        cache.set (key, articles, timeout=600)
    
    return articles
\`\`\`
      `,
  },
].map(({ id, ...q }, idx) => ({
  id: `django-q-${idx + 1}`,
  question: q.question,
  sampleAnswer: String(q.answer),
  keyPoints: [],
}));
