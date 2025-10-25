import { MultipleChoiceQuestion } from '@/lib/types';

export const CachingDjangoMultipleChoice = [
  {
    id: 1,
    question:
      'Which cache backend provides persistence across server restarts?',
    options: [
      'A) Memcached',
      'B) LocMemCache',
      'C) Redis',
      'D) FileBasedCache',
    ],
    correctAnswer: 2,
    explanation: `
**Correct Answer: C) Redis**

Redis stores data on disk and can persist across restarts.

\`\`\`python
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
    }
}
\`\`\`

Memcached and LocMemCache are memory-only and lose data on restart.
      `,
  },
  {
    question: 'How do you cache a view for 15 minutes in Django?',
    options: [
      'A) @cache_view(15)',
      'B) @cache_page(60 * 15)',
      'C) @cache(minutes=15)',
      'D) @cached(900)',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) @cache_page(60 * 15)**

\`\`\`python
from django.views.decorators.cache import cache_page

@cache_page(60 * 15)  # Timeout in seconds
def my_view(request):
    return render(request, 'template.html')
\`\`\`

cache_page takes timeout in seconds (15 min = 900 sec).
      `,
  },
  {
    question: 'What does cache.add() do differently than cache.set()?',
    options: [
      'A) cache.add() is faster',
      'B) cache.add() only sets if key does not exist',
      'C) cache.add() has no expiration',
      'D) cache.add() works with Redis only',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) cache.add() only sets if key does not exist**

\`\`\`python
cache.set('key', 'value')  # Always sets
cache.add('key', 'value')  # Only if key doesn't exist

# Useful for atomic operations
if cache.add('lock:article:123', True, timeout=60):
    # Got the lock
    process_article()
\`\`\`

add() returns True if successful, False if key exists.
      `,
  },
  {
    question: 'How do you cache a template fragment?',
    options: [
      'A) {% cache 300 fragment_name %}...{% endcache %}',
      'B) {% fragment_cache 300 %}...{% end %}',
      'C) {% cached fragment_name %}...{% endcached %}',
      'D) {% cache_fragment 300 %}...{% endcache_fragment %}',
    ],
    correctAnswer: 0,
    explanation: `
**Correct Answer: A) {% cache 300 fragment_name %}...{% endcache %}**

\`\`\`django
{% load cache %}

{% cache 500 sidebar request.user.id %}
    <div class="sidebar">
        <!-- Expensive content -->
    </div>
{% endcache %}
\`\`\`

First arg is timeout (seconds), second is unique fragment name.
      `,
  },
  {
    question: 'What is the best practice for cache key naming?',
    options: [
      'A) Use random strings',
      'B) Use descriptive, hierarchical names like "articles:list:published"',
      'C) Use single letters',
      'D) Use UUIDs',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) Use descriptive, hierarchical names like "articles:list:published"**

\`\`\`python
# Good - descriptive and structured
cache.set('user:123:profile', data)
cache.set('articles:featured:page:1', data)

# Bad - unclear
cache.set('x', data)
cache.set('data123', data)
\`\`\`

Hierarchical naming enables pattern-based invalidation and debugging.
      `,
  },
].map(({ id, ...q }, idx) => ({ id: `django-mc-${idx + 1}`, ...q }));
