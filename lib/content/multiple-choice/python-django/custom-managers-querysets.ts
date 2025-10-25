import { MultipleChoiceQuestion } from '@/lib/types';

export const CustomManagersQuerysetsMultipleChoice = [
  {
    id: 1,
    question:
      'What is the primary advantage of using custom QuerySets over custom Managers in Django?',
    options: [
      'A) Custom QuerySets are faster than custom Managers',
      'B) Custom QuerySets allow method chaining while Managers do not',
      'C) Custom QuerySets can filter data while Managers cannot',
      'D) Custom QuerySets are required for all custom queries',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) Custom QuerySets allow method chaining while Managers do not**

**Why B is correct:**
Custom QuerySets enable method chaining, allowing you to build complex queries by calling multiple custom methods in sequence. Each method returns a QuerySet, so you can chain calls together. Custom Managers, by default, return QuerySets only for the initial query.

\`\`\`python
class ArticleQuerySet(models.QuerySet):
    def published(self):
        return self.filter(status='published')
    
    def featured(self):
        return self.filter(featured=True)
    
    def recent(self):
        return self.filter(created_at__gte=timezone.now() - timedelta(days=7))

class Article(models.Model):
    objects = ArticleQuerySet.as_manager()

# ✅ Method chaining works beautifully
Article.objects.published().featured().recent()
# Returns published, featured articles from the last 7 days

# Can continue chaining with standard QuerySet methods
Article.objects.published().featured().order_by('-created_at')[:10]
\`\`\`

**Why A is incorrect:**
Performance is the same between custom QuerySets and Managers. Both ultimately generate SQL queries. The difference is in API design and chainability, not speed.

**Why C is incorrect:**
Both custom QuerySets and Managers can filter data. Managers have access to all QuerySet methods. The difference is in how they're composed and chained.

**Why D is incorrect:**
Custom QuerySets are not required for custom queries. You can use standard QuerySet methods, custom Managers, or even raw SQL. Custom QuerySets are a design choice for cleaner, more maintainable code.

**Production Best Practice:**
\`\`\`python
# Combine both for optimal API
class ArticleQuerySet(models.QuerySet):
    def published(self):
        return self.filter(status='published')
    
    def by_author(self, author):
        return self.filter(author=author)

class ArticleManager(models.Manager.from_queryset(ArticleQuerySet)):
    def get_queryset(self):
        # Manager adds optimization
        return super().get_queryset().select_related('author', 'category')

class Article(models.Model):
    objects = ArticleManager()

# Combines: optimization (from Manager) + chaining (from QuerySet)
Article.objects.published().by_author(user).order_by('-created_at')
\`\`\`

This pattern provides both performance optimization and flexible, chainable query building.
      `,
  },
  {
    question:
      'When using prefetch_related with a Prefetch object, what is the main benefit over simple prefetch_related("relation_name")?',
    options: [
      'A) Prefetch objects execute faster than simple prefetch_related',
      'B) Prefetch objects allow you to customize the queryset for the related objects',
      'C) Prefetch objects work with ForeignKey while simple prefetch does not',
      'D) Prefetch objects automatically cache results in Redis',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) Prefetch objects allow you to customize the queryset for the related objects**

**Why B is correct:**
Prefetch objects let you customize the queryset used to fetch related objects, including filtering, ordering, and applying select_related for nested relationships. This provides fine-grained control over what data is fetched and how.

\`\`\`python
from django.db.models import Prefetch

# Simple prefetch - fetches all comments
articles = Article.objects.prefetch_related('comments')

# ✅ Prefetch object - customize the related queryset
articles = Article.objects.prefetch_related(
    Prefetch(
        'comments',
        queryset=Comment.objects.filter(
            approved=True
        ).select_related('user').order_by('-created_at')[:10]
    )
)

# Now you get:
# - Only approved comments
# - User data pre-fetched (select_related)
# - Ordered by newest first
# - Limited to 10 per article
\`\`\`

**Why A is incorrect:**
Performance is similar or identical. In fact, Prefetch objects might be slightly slower if you add complex filtering/ordering, but the benefit is in data control, not raw speed.

**Why C is incorrect:**
Both simple prefetch_related and Prefetch objects work with the same relationship types (ManyToMany, reverse ForeignKey). Neither works with forward ForeignKey (use select_related for that).

**Why D is incorrect:**
Prefetch objects don't automatically cache in Redis or any external cache. They optimize database queries but don't involve caching layers unless you explicitly implement that.

**Advanced Pattern - Nested Optimization:**
\`\`\`python
# Optimize articles with comments (and comment authors)
articles = Article.objects.prefetch_related(
    Prefetch(
        'comments',
        queryset=Comment.objects.select_related('user').filter(
            approved=True
        ),
        to_attr='approved_comments'  # Store in custom attribute
    )
)

for article in articles:
    for comment in article.approved_comments:  # Use custom attribute
        print(comment.user.name)  # No additional queries!
\`\`\`

**Performance Comparison:**
\`\`\`python
# Without Prefetch: 1 + N + N*M queries
articles = Article.objects.all()
for article in articles:
    for comment in article.comments.filter(approved=True):
        print(comment.user.name)

# With simple prefetch: 1 + 1 + M queries
articles = Article.objects.prefetch_related('comments')
for article in articles:
    for comment in article.comments.filter(approved=True):
        print(comment.user.name)

# With Prefetch object: 1 + 1 queries (optimal!)
articles = Article.objects.prefetch_related(
    Prefetch('comments', queryset=Comment.objects.filter(approved=True).select_related('user'))
)
for article in articles:
    for comment in article.comments.all():  # Already filtered!
        print(comment.user.name)  # Already prefetched!
\`\`\`

**Production Use Case:**
\`\`\`python
class ArticleViewSet(viewsets.ModelViewSet):
    def get_queryset(self):
        if self.action == 'retrieve':
            return Article.objects.prefetch_related(
                Prefetch(
                    'comments',
                    queryset=Comment.objects.filter(
                        approved=True
                    ).select_related('user')[:5],
                    to_attr='recent_comments'
                ),
                Prefetch(
                    'images',
                    queryset=Image.objects.order_by('order')
                )
            )
        return Article.objects.all()
\`\`\`

Prefetch objects are essential for optimizing complex nested relationships in production APIs.
      `,
  },
  {
    question:
      'In Django, which approach correctly implements a soft delete pattern using a custom Manager?',
    options: [
      'A) Override the delete() method in the Model',
      'B) Create a custom Manager that filters out deleted objects by default',
      'C) Use a pre_delete signal to prevent deletion',
      'D) Add a database constraint to prevent DELETE operations',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) Create a custom Manager that filters out deleted objects by default**

**Why B is correct:**
A custom Manager that filters the default queryset is the Django-idiomatic way to implement soft deletes. This ensures that "deleted" objects are hidden from all standard queries by default, while still allowing access when needed.

\`\`\`python
class SoftDeleteQuerySet(models.QuerySet):
    def delete(self):
        """Soft delete for QuerySets"""
        return self.update(deleted_at=timezone.now())
    
    def hard_delete(self):
        """Actually delete from database"""
        return super().delete()
    
    def alive(self):
        """Only non-deleted objects"""
        return self.filter(deleted_at__isnull=True)
    
    def dead(self):
        """Only deleted objects"""
        return self.filter(deleted_at__isnull=False)

class SoftDeleteManager(models.Manager):
    def get_queryset(self):
        """Default queryset excludes deleted objects"""
        return SoftDeleteQuerySet(self.model, using=self._db).alive()
    
    def all_with_deleted(self):
        """Get all objects including deleted"""
        return SoftDeleteQuerySet(self.model, using=self._db)

class Article(models.Model):
    title = models.CharField(max_length=200)
    deleted_at = models.DateTimeField(null=True, blank=True)
    
    objects = SoftDeleteManager()
    all_objects = models.Manager()  # Bypass soft delete filter
    
    def delete(self, using=None, keep_parents=False):
        """Soft delete for model instances"""
        self.deleted_at = timezone.now()
        self.save()

# Usage:
article = Article.objects.get(id=1)
article.delete()  # Soft delete

Article.objects.all()  # Excludes deleted articles
Article.all_objects.all()  # Includes deleted articles
Article.objects.all_with_deleted()  # Includes deleted articles
\`\`\`

**Why A is incorrect:**
While overriding delete() in the Model is part of the solution, it's not sufficient alone. You also need a custom Manager to hide deleted objects from queries. Option A addresses half the problem.

**Why C is incorrect:**
Using signals to prevent deletion is not the soft delete pattern. Signals run after the operation starts and can't cleanly prevent it. Also, this wouldn't mark objects as deleted, just prevent deletion entirely.

**Why D is incorrect:**
Database constraints prevent deletion at the database level, but that's not soft deletion - that's hard prevention. Soft delete means marking as deleted while keeping the data.

**Complete Production Pattern:**
\`\`\`python
from django.db import models
from django.utils import timezone

class SoftDeleteQuerySet(models.QuerySet):
    def delete(self):
        return self.update(deleted_at=timezone.now(), deleted_by=self._get_current_user())
    
    def hard_delete(self):
        return super().delete()
    
    def alive(self):
        return self.filter(deleted_at__isnull=True)
    
    def dead(self):
        return self.filter(deleted_at__isnull=False)
    
    def restore(self):
        """Restore soft-deleted objects"""
        return self.update(deleted_at=None, deleted_by=None)

class SoftDeleteManager(models.Manager.from_queryset(SoftDeleteQuerySet)):
    def get_queryset(self):
        return super().get_queryset().alive()

class SoftDeleteModel(models.Model):
    """Abstract base model for soft deletion"""
    deleted_at = models.DateTimeField(null=True, blank=True, db_index=True)
    deleted_by = models.ForeignKey(
        'auth.User',
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name='+'
    )
    
    objects = SoftDeleteManager()
    all_objects = models.Manager()
    
    class Meta:
        abstract = True
    
    def delete(self, using=None, keep_parents=False, hard=False):
        if hard:
            return super().delete(using=using, keep_parents=keep_parents)
        else:
            self.deleted_at = timezone.now()
            self.save()
    
    def restore(self):
        self.deleted_at = None
        self.deleted_by = None
        self.save()

# Usage in models:
class Article(SoftDeleteModel):
    title = models.CharField(max_length=200)
    content = models.TextField()

# In views:
article.delete()  # Soft delete
article.restore()  # Restore
article.delete(hard=True)  # Hard delete

Article.objects.all()  # Only non-deleted
Article.all_objects.all()  # Including deleted
Article.objects.dead()  # Only deleted
\`\`\`

**Benefits:**
- ✅ Data recovery possible
- ✅ Audit trail (when/who deleted)
- ✅ Maintain referential integrity
- ✅ Analyze deleted data
- ✅ Comply with data retention policies

**Considerations:**
- ❌ Unique constraints must account for deleted objects
- ❌ Backup/restore strategies need adjustment
- ❌ Database size grows over time
- ❌ Need periodic cleanup of old deleted records

This pattern is widely used in production for critical data that shouldn't be permanently deleted immediately.
      `,
  },
  {
    question:
      'What is the most efficient way to check if a QuerySet has any results without loading them into memory?',
    options: [
      'A) if len(queryset) > 0:',
      'B) if queryset.count() > 0:',
      'C) if queryset.exists():',
      'D) if list(queryset):',
    ],
    correctAnswer: 2,
    explanation: `
**Correct Answer: C) if queryset.exists():**

**Why C is correct:**
\`exists()\` is the most efficient way to check for existence. It generates a highly optimized SQL query that only checks if at least one row exists, without fetching any data or counting all rows.

\`\`\`python
# ✅ Most efficient
if Article.objects.filter(status='published').exists():
    print("Published articles exist")

# SQL Generated:
# SELECT 1 FROM articles WHERE status = 'published' LIMIT 1
# Stops as soon as it finds one row
\`\`\`

**Why A is incorrect:**
\`len(queryset)\` forces Django to execute the query and load ALL results into Python memory, then count them. This is extremely inefficient, especially for large querysets.

\`\`\`python
# ❌ Inefficient: Loads all 10,000 articles into memory
if len(Article.objects.all()) > 0:
    pass

# Fetches and loads all data just to check if any exists!
\`\`\`

**Why B is incorrect:**
\`count()\` performs a SQL COUNT(*) query which counts ALL matching rows, even though we only need to know if ANY exist. For large tables, this is much slower than exists().

\`\`\`python
# ❌ Less efficient: Counts all rows
if Article.objects.filter(status='published').count() > 0:
    pass

# SQL: SELECT COUNT(*) FROM articles WHERE status = 'published'
# Has to scan all matching rows to count them
\`\`\`

**Why D is incorrect:**
\`list(queryset)\` converts the entire queryset to a Python list, loading all data into memory. This is the worst option for checking existence.

**Performance Comparison (1 million articles):**
- exists(): ~1ms (stops at first match)
- count() > 0: ~100ms (counts all rows)
- len() > 0: ~5000ms (loads all into memory)
- list(): ~5000ms (loads all into memory)

**Production Patterns:**

\`\`\`python
# ✅ Check existence before querying
if Article.objects.filter(author=user).exists():
    articles = Article.objects.filter(author=user).select_related('category')
else:
    articles = []

# ✅ Conditional logic
has_articles = Article.objects.filter(status='published').exists()
has_drafts = Article.objects.filter(status='draft').exists()

if has_articles and not has_drafts:
    # All articles are published
    pass

# ✅ In templates (via context processor)
context = {
    'has_notifications': Notification.objects.filter(
        user=request.user, 
        read=False
    ).exists()
}
\`\`\`

**When to Use Each:**

**Use exists():**
- ✅ Checking if any records match
- ✅ Boolean conditions
- ✅ Large datasets
- ✅ Performance-critical code

**Use count():**
- ✅ Need actual number
- ✅ Displaying counts to users
- ✅ Pagination total count
- ❌ Just checking if > 0

**Use len():**
- ✅ Already have QuerySet in memory
- ✅ Small, cached datasets
- ❌ Database queries
- ❌ Large datasets

**Advanced Pattern - Multiple Checks:**
\`\`\`python
from django.db.models import Exists, OuterRef

# Check multiple existence conditions in one query
Article.objects.annotate(
    has_comments=Exists(
        Comment.objects.filter(article=OuterRef('pk'))
    ),
    has_likes=Exists(
        Like.objects.filter(article=OuterRef('pk'))
    )
).filter(has_comments=True, has_likes=True)
\`\`\`

**API Example:**
\`\`\`python
@api_view(['GET'])
def check_username(request):
    """Check if username is available"""
    username = request.query_params.get('username')
    
    # ✅ Efficient existence check
    is_taken = User.objects.filter(username=username).exists()
    
    return Response({
        'available': not is_taken
    })
\`\`\`

Using \`exists()\` correctly can improve query performance by 100x or more in large databases!
      `,
  },
  {
    question:
      'In a custom QuerySet, what does returning "self" from a method allow you to do?',
    options: [
      'A) Cache the QuerySet results in memory',
      'B) Chain additional QuerySet methods after calling your custom method',
      'C) Execute the query immediately without lazy evaluation',
      'D) Return the current model instance instead of a QuerySet',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) Chain additional QuerySet methods after calling your custom method**

**Why B is correct:**
Returning \`self\` (or more commonly, returning the modified QuerySet) allows method chaining - calling multiple methods in sequence. This is the core pattern that makes Django's QuerySet API so elegant and powerful.

\`\`\`python
class ArticleQuerySet(models.QuerySet):
    def published(self):
        """Filter to published articles"""
        return self.filter(status='published')  # Returns QuerySet
    
    def featured(self):
        """Filter to featured articles"""
        return self.filter(featured=True)  # Returns QuerySet
    
    def recent(self):
        """Filter to recent articles"""
        from datetime import timedelta
        week_ago = timezone.now() - timedelta(days=7)
        return self.filter(created_at__gte=week_ago)  # Returns QuerySet

class Article(models.Model):
    objects = ArticleQuerySet.as_manager()

# ✅ Method chaining works because each method returns a QuerySet
articles = (Article.objects
    .published()      # Returns QuerySet
    .featured()       # Returns QuerySet
    .recent()         # Returns QuerySet
    .order_by('-created_at')  # Standard QuerySet method
    [:10])           # Slicing

# This is only possible because each method returns a QuerySet (self)
\`\`\`

**Why A is incorrect:**
Returning \`self\` doesn't cache results. QuerySets are lazy and only execute when evaluated (iteration, slicing, etc.), regardless of what methods return.

**Why C is incorrect:**
Returning \`self\` maintains lazy evaluation - it doesn't execute the query. The query only executes when the QuerySet is evaluated (e.g., iteration, list(), etc.).

**Why D is incorrect:**
Custom QuerySet methods should return QuerySets, not model instances. To return a single instance, use \`get()\` or similar methods that explicitly return model objects.

**Bad Pattern - Breaking the Chain:**
\`\`\`python
class ArticleQuerySet(models.QuerySet):
    def published(self):
        """❌ BAD: Returns list, breaks chaining"""
        return list(self.filter(status='published'))

# This breaks method chaining:
articles = Article.objects.published().featured()  # ❌ AttributeError!
# 'list' object has no attribute 'featured'
\`\`\`

**Good Pattern - Maintaining Chainability:**
\`\`\`python
class ArticleQuerySet(models.QuerySet):
    def published(self):
        """✅ GOOD: Returns QuerySet"""
        return self.filter(status='published')
    
    def by_category(self, category):
        """✅ GOOD: Returns QuerySet"""
        return self.filter(category=category)
    
    def with_author_details(self):
        """✅ GOOD: Returns QuerySet with optimization"""
        return self.select_related('author', 'author__profile')

# All methods return QuerySets, enabling full chaining
Article.objects.published().by_category('tech').with_author_details().order_by('-views')
\`\`\`

**When to Return Something Other Than QuerySet:**
\`\`\`python
class ArticleQuerySet(models.QuerySet):
    def published(self):
        return self.filter(status='published')  # Returns QuerySet
    
    def get_total_views(self):
        """Returns integer, not QuerySet - terminates chain"""
        return self.aggregate(Sum('view_count'))['view_count__sum'] or 0
    
    def export_to_csv(self):
        """Returns string, not QuerySet - terminates chain"""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Title', 'Author', 'Views'])
        
        for article in self:
            writer.writerow([article.title, article.author, article.view_count])
        
        return output.getvalue()

# Chaining works until terminal method
total_views = Article.objects.published().featured().get_total_views()  # ✅ Works
csv_data = Article.objects.published().export_to_csv()  # ✅ Works
articles = Article.objects.published().get_total_views().featured()  # ❌ Breaks - int has no featured()
\`\`\`

**Best Practices:**
- ✅ Filter methods should return QuerySet
- ✅ Aggregate methods can return values
- ✅ Export methods can return formatted data
- ✅ Keep QuerySet methods chainable
- ✅ Document when methods terminate the chain

**Production Example:**
\`\`\`python
class ArticleQuerySet(models.QuerySet):
    # Chainable filters
    def published(self):
        return self.filter(status='published')
    
    def in_language(self, lang):
        return self.filter(language=lang)
    
    def for_user(self, user):
        if user.is_staff:
            return self  # All articles
        return self.filter(author=user)  # Only user's articles
    
    # Terminal methods (return non-QuerySet)
    def stats(self):
        from django.db.models import Count, Avg, Sum
        return self.aggregate(
            total=Count('id'),
            avg_views=Avg('view_count'),
            total_views=Sum('view_count')
        )

# Usage:
articles = Article.objects.published().in_language('en').for_user(user)  # ✅ Chaining
stats = Article.objects.published().in_language('en').stats()  # ✅ Terminal method
\`\`\`

Understanding QuerySet chaining is fundamental to writing clean, efficient Django code.
      `,
  },
].map(({ id, ...q }, idx) => ({ id: `django-mc-${idx + 1}`, ...q }));
