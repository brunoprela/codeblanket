import { MultipleChoiceQuestion } from '@/lib/types';

export const DjangoOrmAdvancedTechniquesMultipleChoice = [
  {
    id: 1,
    question:
      'What is the primary advantage of using F() expressions in Django ORM queries?',
    options: [
      'A) They allow filtering by related model fields',
      'B) They perform operations at the database level, avoiding race conditions',
      'C) They enable complex aggregation calculations',
      'D) They automatically optimize queries with select_related',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) They perform operations at the database level, avoiding race conditions**

**Why B is correct:**
F() expressions reference model fields directly in database queries, allowing operations to be performed at the database level. This is crucial for avoiding race conditions in concurrent environments where multiple processes might update the same field simultaneously.

\`\`\`python
from django.db.models import F

# ✅ Thread-safe: Atomic database-level increment
Article.objects.filter (id=1).update (view_count=F('view_count') + 1)

# ❌ Race condition: Read-modify-write in Python
article = Article.objects.get (id=1)  # Thread A reads view_count=100
# Thread B might also read view_count=100 here
article.view_count += 1  # Thread A sets view_count=101
article.save()  # Thread B also sets view_count=101 (should be 102!)
\`\`\`

**Why A is incorrect:**
While F() expressions can be used with related field lookups, this isn't their primary advantage. Standard query lookups already support related fields (e.g., \`filter (author__name='John')\`).

**Why C is incorrect:**
Complex aggregations are typically done with aggregation functions like Sum(), Avg(), Count(), not F() expressions. F() is for field references and comparisons.

**Why D is incorrect:**
F() expressions don't automatically optimize query performance. select_related() is a separate optimization technique for reducing queries through SQL JOINs.

**Production Use Cases:**
1. **Atomic Updates**: Incrementing counters, updating timestamps
2. **Field Comparisons**: Finding records where one field > another
3. **Calculations**: Database-level math operations
4. **Avoiding Race Conditions**: Safe concurrent updates

**Example:**
\`\`\`python
# Find products where discount > markup
products = Product.objects.filter (discount__gt=F('markup'))

# Calculate profit margin in database
Product.objects.annotate(
    profit_margin=F('price') - F('cost')
).filter (profit_margin__gt=10)
\`\`\`

This database-level operation is much safer and more efficient than fetching data to Python for manipulation.
      `,
  },
  {
    question:
      "In Django\'s transaction management, what happens if an exception is raised inside an atomic() block?",
    options: [
      'A) The transaction commits and the exception is logged',
      'B) The transaction is automatically rolled back and the exception propagates',
      'C) The transaction commits only the changes made before the exception',
      'D) The exception is caught and converted to a database error',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) The transaction is automatically rolled back and the exception propagates**

**Why B is correct:**
When an exception is raised inside a \`transaction.atomic()\` block, Django automatically rolls back all database changes made within that block, then re-raises the exception for the caller to handle. This ensures database consistency.

\`\`\`python
from django.db import transaction

try:
    with transaction.atomic():
        article = Article.objects.create (title='Test')
        tag = Tag.objects.create (name='test')
        
        # This will cause rollback
        raise ValueError('Something went wrong')
        
except ValueError:
    pass  # Article and Tag were NOT saved

# Database is unchanged - both creates were rolled back
\`\`\`

**Why A is incorrect:**
The transaction never commits when an exception occurs. Django\'s atomic() ensures all-or-nothing behavior - either all operations succeed and commit, or all are rolled back.

**Why C is incorrect:**
Django doesn't support partial transactions within an atomic block. It's all-or-nothing. All changes are rolled back, not just those after the exception.

**Why D is incorrect:**
The exception isn't converted or caught - it propagates normally. Django\'s transaction management doesn't interfere with exception handling beyond the rollback.

**Production Pattern:**
\`\`\`python
from django.db import transaction

def transfer_money (from_account, to_account, amount):
    try:
        with transaction.atomic():
            # These operations are atomic
            from_account.balance -= amount
            from_account.save()
            
            to_account.balance += amount
            to_account.save()
            
            # Log the transfer
            Transfer.objects.create(
                from_account=from_account,
                to_account=to_account,
                amount=amount
            )
    except Exception as e:
        # All database changes rolled back
        logger.error (f'Transfer failed: {e}')
        raise
\`\`\`

**Savepoints (Nested Transactions):**
\`\`\`python
with transaction.atomic():  # Outer transaction
    article = Article.objects.create (title='Main')
    
    try:
        with transaction.atomic():  # Savepoint
            Tag.objects.create (name='invalid//tag')
    except ValidationError:
        pass  # Only savepoint rolled back
    
    # Article still saved
\`\`\`

**Best Practices:**
- ✅ Use atomic() for multi-model operations
- ✅ Keep transactions short
- ✅ Don't call external APIs inside transactions
- ✅ Use savepoints for partial rollbacks
- ❌ Don't mix external side effects with database transactions

Understanding transaction behavior is critical for data integrity in production applications.
      `,
  },
  {
    question:
      'Which approach is most efficient for retrieving articles with their comments, where comments also need their associated users?',
    options: [
      'A) Article.objects.all().prefetch_related("comments", "comments__user")',
      'B) Article.objects.select_related("comments").prefetch_related("user")',
      'C) Article.objects.prefetch_related(Prefetch("comments", queryset=Comment.objects.select_related("user")))',
      'D) Article.objects.all().annotate (comment_count=Count("comments"))',
    ],
    correctAnswer: 2,
    explanation: `
**Correct Answer: C) Article.objects.prefetch_related(Prefetch("comments", queryset=Comment.objects.select_related("user")))**

**Why C is correct:**
This approach uses a \`Prefetch\` object to optimize the nested relationship. It fetches articles in one query, comments in a second query, and eagerly loads users via JOIN in that second query. This is the most efficient pattern for nested relationships mixing "to-many" and "to-one" patterns.

\`\`\`python
from django.db.models import Prefetch

articles = Article.objects.prefetch_related(
    Prefetch(
        'comments',
        queryset=Comment.objects.select_related('user')
    )
)

# Execution:
# Query 1: SELECT * FROM articles
# Query 2: SELECT comments.*, users.* FROM comments 
#          JOIN users ON comments.user_id = users.id
#          WHERE comments.article_id IN (1,2,3,...)

# Total: 2 queries, fully optimized

for article in articles:
    for comment in article.comments.all():
        print(comment.user.name)  # No additional queries!
\`\`\`

**Why A is incorrect:**
While this approach works, it's less efficient. \`prefetch_related("comments__user")\` will make 3 separate queries: one for articles, one for comments, and one for all users. Option C combines the comment and user queries into one with a JOIN.

\`\`\`python
# Makes 3 queries instead of 2
Article.objects.prefetch_related("comments", "comments__user")
# Query 1: SELECT * FROM articles
# Query 2: SELECT * FROM comments WHERE article_id IN (...)
# Query 3: SELECT * FROM users WHERE id IN (...)
\`\`\`

**Why B is incorrect:**
\`select_related("comments")\` is invalid because comments is a reverse ForeignKey (one-to-many), not a forward ForeignKey. select_related() only works for "to-one" relationships. This would raise an error or fail to optimize properly.

**Why D is incorrect:**
This only counts comments but doesn't actually retrieve them or their users. It\'s useful for displaying counts but doesn't solve the nested relationship optimization problem.

**Performance Comparison (100 articles, 500 comments):**
- No optimization: 1 + 500 + 500 = 1001 queries
- Option A: 3 queries
- Option C: 2 queries (best)
- Option D: 1 query (but no actual comments/users retrieved)

**Production Pattern:**
\`\`\`python
class ArticleViewSet (viewsets.ModelViewSet):
    def get_queryset (self):
        if self.action == 'retrieve':
            # Detail view: optimize nested relationships
            return Article.objects.prefetch_related(
                Prefetch(
                    'comments',
                    queryset=Comment.objects.select_related('user').filter(
                        approved=True
                    ).order_by('-created_at')[:10]
                )
            )
        
        return Article.objects.all()
\`\`\`

**Advanced Optimization:**
\`\`\`python
# Multiple nested relationships
Article.objects.prefetch_related(
    Prefetch('comments', queryset=Comment.objects.select_related('user')),
    Prefetch('images', queryset=Image.objects.order_by('order')),
    'tags'  # Simple prefetch for tags
)
\`\`\`

This pattern is essential for building performant APIs that return nested data structures efficiently.
      `,
  },
  {
    question:
      'When implementing a multi-database setup with read replicas in Django, which database router method determines where write operations are sent?',
    options: [
      'A) db_for_read()',
      'B) db_for_write()',
      'C) allow_migrate()',
      'D) allow_relation()',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) db_for_write()**

**Why B is correct:**
The \`db_for_write()\` method in a database router determines which database should be used for write operations (INSERT, UPDATE, DELETE). This is crucial for routing writes to the primary database while potentially routing reads to replicas.

\`\`\`python
class PrimaryReplicaRouter:
    def db_for_write (self, model, **hints):
        """All writes go to primary database"""
        return 'default'  # Primary database
    
    def db_for_read (self, model, **hints):
        """Reads can go to replicas"""
        import random
        return random.choice(['replica1', 'replica2'])

# settings.py
DATABASE_ROUTERS = ['myapp.routers.PrimaryReplicaRouter']

# Usage:
article = Article.objects.create (title='Test')  # Uses 'default' (primary)
articles = Article.objects.all()  # Uses random replica
\`\`\`

**Why A is incorrect:**
\`db_for_read()\` determines where READ operations (SELECT queries) are sent, not writes. This is used to route queries to read replicas for load distribution.

**Why C is incorrect:**
\`allow_migrate()\` controls whether migrations should be run on a particular database. It doesn't affect regular read/write routing. It\'s used to determine which databases get which migrations applied.

\`\`\`python
def allow_migrate (self, db, app_label, model_name=None, **hints):
    """Only run migrations on primary database"""
    return db == 'default'
\`\`\`

**Why D is incorrect:**
\`allow_relation()\` determines whether relationships between objects from different databases are allowed. It doesn't control where operations are sent.

\`\`\`python
def allow_relation (self, obj1, obj2, **hints):
    """Allow relations if both objects in same database"""
    return obj1._state.db == obj2._state.db
\`\`\`

**Production Router Pattern:**
\`\`\`python
import threading
import time

_thread_local = threading.local()

class StickyPrimaryRouter:
    """Route reads to replicas, but use primary after writes"""
    
    def db_for_write (self, model, **hints):
        # All writes to primary
        self._mark_write()
        return 'default'
    
    def db_for_read (self, model, **hints):
        # Use primary for short time after write (avoid replication lag)
        if self._recent_write():
            return 'default'
        
        # Otherwise use replica
        import random
        return random.choice(['replica1', 'replica2'])
    
    def _mark_write (self):
        _thread_local.last_write = time.time()
    
    def _recent_write (self):
        last_write = getattr(_thread_local, 'last_write', 0)
        return (time.time() - last_write) < 5  # 5 second sticky window
    
    def allow_migrate (self, db, app_label, model_name=None, **hints):
        # Only migrate primary
        return db == 'default'
\`\`\`

**Complete Database Configuration:**
\`\`\`python
# settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'primary_db',
        'HOST': 'primary.example.com',
    },
    'replica1': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'primary_db',
        'HOST': 'replica1.example.com',
        'OPTIONS': {'options': '-c default_transaction_read_only=on'},
    },
    'replica2': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'primary_db',
        'HOST': 'replica2.example.com',
        'OPTIONS': {'options': '-c default_transaction_read_only=on'},
    },
}

DATABASE_ROUTERS = ['myapp.routers.StickyPrimaryRouter']
\`\`\`

**Best Practices:**
- ✅ Always route writes to primary
- ✅ Use "sticky" reads after writes (avoid replication lag)
- ✅ Monitor replication lag
- ✅ Only run migrations on primary
- ✅ Use read-only database users for replicas
- ❌ Don't assume immediate consistency
- ❌ Don't forget to handle replication lag

**Manual Database Selection:**
\`\`\`python
# Force specific database
Article.objects.using('default').create (title='Test')
articles = Article.objects.using('replica1').all()
\`\`\`

Understanding database routing is essential for scaling Django applications horizontally with read replicas.
      `,
  },
  {
    question:
      'What is the correct way to perform a conditional update in Django that only updates records meeting specific criteria?',
    options: [
      'A) Article.objects.filter (status="draft").save()',
      'B) Article.objects.update (status="published")',
      'C) Article.objects.filter (status="draft").update (status="published")',
      'D) for article in Article.objects.filter (status="draft"): article.status="published"; article.save()',
    ],
    correctAnswer: 2,
    explanation: `
**Correct Answer: C) Article.objects.filter (status="draft").update (status="published")**

**Why C is correct:**
Chaining \`filter()\` with \`update()\` performs a single SQL UPDATE query that only affects records matching the filter criteria. This is the most efficient approach for bulk updates.

\`\`\`python
# ✅ Efficient: Single UPDATE query
updated_count = Article.objects.filter (status='draft').update(
    status='published',
    published_at=timezone.now()
)

# SQL Generated:
# UPDATE articles 
# SET status = 'published', published_at = NOW()
# WHERE status = 'draft'

print(f'Updated {updated_count} articles')
\`\`\`

**Why A is incorrect:**
QuerySets don't have a \`save()\` method. \`save()\` is a model instance method, not a QuerySet method. This would raise an AttributeError.

**Why B is incorrect:**
\`Article.objects.update()\` would update ALL articles in the database, not just drafts. Without a filter, it affects every record, which is dangerous and rarely what you want.

\`\`\`python
# ❌ Dangerous: Updates ALL articles
Article.objects.update (status='published')  # Updates every single article!
\`\`\`

**Why D is incorrect:**
While this works functionally, it's extremely inefficient. It creates N separate UPDATE queries (one per article), loads all objects into memory, and doesn't leverage database-level optimizations. For 1000 articles, this creates 1000 queries vs. 1 query with option C.

\`\`\`python
# ❌ Inefficient: N queries + loads all into memory
for article in Article.objects.filter (status='draft'):
    article.status = 'published'
    article.save()  # Separate UPDATE query for EACH article
\`\`\`

**Performance Comparison (1000 draft articles):**
- Option C: 1 UPDATE query, ~10ms
- Option D: 1 SELECT + 1000 UPDATE queries, ~5000ms
- **Result: 500x faster!**

**Advanced Update Patterns:**
\`\`\`python
from django.db.models import F
from django.utils import timezone

# Conditional update with F() expressions
Article.objects.filter(
    status='draft',
    created_at__lt=timezone.now() - timedelta (days=30)
).update(
    status='archived',
    view_count=F('view_count') + 1  # Atomic increment
)

# Update with case/when for conditional logic
from django.db.models import Case, When, Value

Article.objects.update(
    priority=Case(
        When (view_count__gte=1000, then=Value('high')),
        When (view_count__gte=100, then=Value('medium')),
        default=Value('low')
    )
)
\`\`\`

**Important Notes:**
1. **Signals**: \`update()\` does NOT trigger model signals (pre_save, post_save)
2. **Validation**: \`update()\` bypasses model validation
3. **save() override**: Custom save() logic is not executed
4. **Performance**: Much faster for bulk operations

**When to Use Each Approach:**

**Use update() when:**
- ✅ Bulk updating many records
- ✅ Simple field updates
- ✅ Performance is critical
- ✅ Signals not needed

**Use save() when:**
- ✅ Updating single instances
- ✅ Need signals to fire
- ✅ Need custom save() logic
- ✅ Need validation

**Production Pattern:**
\`\`\`python
def publish_drafts_batch():
    """Publish old draft articles in batch"""
    threshold = timezone.now() - timedelta (days=7)
    
    # Single efficient query
    count = Article.objects.filter(
        status='draft',
        created_at__lt=threshold
    ).update(
        status='published',
        published_at=timezone.now()
    )
    
    logger.info (f'Published {count} draft articles')
    return count
\`\`\`

Understanding the difference between \`update()\` and iterating with \`save()\` is crucial for building performant Django applications, especially when dealing with large datasets.
      `,
  },
].map(({ id, ...q }, idx) => ({ id: `django-mc-${idx + 1}`, ...q }));
