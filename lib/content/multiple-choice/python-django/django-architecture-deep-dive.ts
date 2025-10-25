import { MultipleChoiceQuestion } from '@/lib/types';

export const DjangoArchitectureDeepDiveMultipleChoice = [
  {
    id: 1,
    question:
      "In Django\'s MVT architecture, what is the primary role of the View component?",
    options: [
      'A) Render HTML templates and handle presentation logic',
      'B) Define database schema and handle data persistence',
      'C) Process HTTP requests and contain business logic',
      'D) Configure URL routing and middleware execution',
    ],
    correctAnswer: 2,
    explanation: `
**Correct Answer: C) Process HTTP requests and contain business logic**

**Why C is correct:**
In Django\'s MVT (Model-View-Template) architecture, the View is responsible for processing HTTP requests and containing the business logic. Views receive requests, interact with models to retrieve/modify data, and return responses (often rendering templates). This is equivalent to the Controller in traditional MVC architecture.

\`\`\`python
def article_list (request):
    # View contains business logic
    articles = Article.objects.filter (status='published')
    return render (request, 'articles/list.html', {'articles': articles})
\`\`\`

**Why A is incorrect:**
Rendering HTML and presentation logic is the responsibility of the Template component (the "T" in MVT), not the View. Templates handle how data is displayed to users.

**Why B is incorrect:**
Defining database schema and data persistence is the Model component's responsibility. Models define the structure of data and provide an ORM interface.

**Why D is incorrect:**
While views are connected to URLs, URL routing itself is configured in urls.py files, and middleware is a separate component that processes requests before they reach views.

**Production Note:**
Understanding the distinction between Views (business logic) and Templates (presentation) is crucial for proper separation of concerns in Django applications. This separation makes code more maintainable and testable.
      `,
  },
  {
    question:
      'Which Django ORM method would you use to optimize a query that retrieves articles with their related authors and categories, avoiding N+1 query problems?',
    options: [
      'A) Article.objects.all().filter (author__isnull=False)',
      'B) Article.objects.select_related("author", "category")',
      'C) Article.objects.prefetch_related("author", "category")',
      'D) Article.objects.raw("SELECT * FROM articles JOIN authors")',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) Article.objects.select_related("author", "category")**

**Why B is correct:**
\`select_related()\` is the correct method for optimizing queries with ForeignKey and OneToOneField relationships. It performs a SQL JOIN operation, fetching all related data in a single database query. For articles with their related authors and categories (both ForeignKey relationships), this is the most efficient approach.

\`\`\`python
# Optimized: 1 query with JOINs
articles = Article.objects.select_related('author', 'category').all()
for article in articles:
    print(article.author.name)  # No additional query
    print(article.category.name)  # No additional query
\`\`\`

**Why A is incorrect:**
\`filter (author__isnull=False)\` simply filters articles that have authors but doesn't optimize the query. Each access to \`article.author\` would still trigger a separate database query (N+1 problem).

**Why C is incorrect:**
\`prefetch_related()\` is for ManyToManyField and reverse ForeignKey relationships, not for forward ForeignKey relationships like author and category. It uses separate queries and performs the join in Python, which is less efficient for this case.

**Why D is incorrect:**
Using raw SQL bypasses Django\'s ORM benefits (query optimization, database portability, security) and is harder to maintain. While it might work, it's not the recommended approach and doesn't provide the same safety and convenience as \`select_related()\`.

**Production Note:**
Always use \`select_related()\` for "to-one" relationships (ForeignKey, OneToOneField) and \`prefetch_related()\` for "to-many" relationships (ManyToManyField, reverse ForeignKey). This can reduce database queries from hundreds to just a few, dramatically improving performance.

**Performance Comparison:**
- Without optimization: 1 + N + N queries (1 for articles, N for authors, N for categories)
- With select_related(): 1 query with JOINs
- Result: ~100x faster for 100 articles!
      `,
  },
  {
    question:
      "In Django\'s request/response cycle, at what stage does middleware execute its request processing methods?",
    options: [
      'A) After the view function executes but before the template is rendered',
      'B) Before the URL resolver determines which view to call',
      'C) After the template is rendered but before the response is sent to the client',
      'D) Only when an exception occurs during view execution',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) Before the URL resolver determines which view to call**

**Why B is correct:**
Django middleware's request processing methods (\`process_request\` in legacy middleware, or the code before \`get_response()\` in modern middleware) execute early in the request cycle, before URL resolution and view execution. This allows middleware to modify the request, perform authentication, or short-circuit the request entirely before reaching the view.

**Request/Response Flow:**
\`\`\`
1. HTTP Request arrives
2. Middleware (Request Phase) ← You are here
3. URL Resolver matches URL pattern
4. View function executes
5. Middleware (Response Phase)
6. HTTP Response sent to client
\`\`\`

**Example:**
\`\`\`python
class AuthenticationMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Executed BEFORE URL resolution and view
        request.user = authenticate (request)
        
        # Call next middleware/view
        response = self.get_response (request)
        
        # Executed AFTER view but before sending response
        return response
\`\`\`

**Why A is incorrect:**
Middleware request processing happens before the view executes, not after. Response processing happens after the view, but this question asks specifically about request processing methods.

**Why C is incorrect:**
This describes response processing, not request processing. Response middleware runs after the view and template rendering, but request middleware runs before URL resolution.

**Why D is incorrect:**
While middleware can have exception handling methods (\`process_exception\`), request processing middleware runs on every request regardless of whether exceptions occur. Exception handling is a separate, optional middleware hook.

**Production Note:**
Understanding middleware execution order is crucial for security and performance. Authentication middleware must run before views that require authentication, and caching middleware should run early to potentially bypass the entire request chain. Middleware order in \`MIDDLEWARE\` setting matters!

**Common Middleware Order:**
\`\`\`python
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',  # First: security
    'django.middleware.cache.UpdateCacheMiddleware',  # Early: cache writes
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',  # Auth needed by views
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'django.middleware.cache.FetchFromCacheMiddleware',  # Last: cache reads
]
\`\`\`
      `,
  },
  {
    question:
      'What is the most efficient way to count related objects in Django ORM without loading them into memory?',
    options: [
      'A) len (article.comments.all())',
      'B) article.comments.count()',
      'C) Article.objects.annotate (comment_count=Count("comments"))',
      'D) len([c for c in article.comments.iterator()])',
    ],
    correctAnswer: 2,
    explanation: `
**Correct Answer: C) Article.objects.annotate (comment_count=Count("comments"))**

**Why C is correct:**
Using \`annotate()\` with \`Count()\` performs the counting at the database level as part of the main query, which is the most efficient approach, especially when retrieving multiple articles. The count is calculated using SQL's COUNT() function and returned as part of the queryset.

\`\`\`python
from django.db.models import Count

articles = Article.objects.annotate(
    comment_count=Count('comments')
)

for article in articles:
    print(article.comment_count)  # No additional query, value already present
\`\`\`

**SQL Generated:**
\`\`\`sql
SELECT article.*, COUNT(comments.id) as comment_count
FROM articles
LEFT JOIN comments ON articles.id = comments.article_id
GROUP BY article.id
\`\`\`

**Why A is incorrect:**
\`len (article.comments.all())\` fetches ALL comment objects from the database into Python memory and then counts them. This is extremely inefficient for large datasets.

\`\`\`python
# ❌ Bad: Fetches all comments into memory
count = len (article.comments.all())  # Loads 1000 comment objects just to count them
\`\`\`

**Why B is incorrect:**
While \`article.comments.count()\` does perform database-level counting (much better than option A), it requires a separate query for each article. If you're displaying a list of articles with comment counts, this creates an N+1 query problem (1 query for articles + N queries for counts).

\`\`\`python
# Works but creates N+1 queries
for article in Article.objects.all():  # 1 query
    count = article.comments.count()  # N queries (one per article)
\`\`\`

**Why D is incorrect:**
\`iterator()\` helps with memory efficiency for large querysets by not caching results, but \`len()\` still needs to iterate through all comments in Python. This is better than option A for memory but still loads data unnecessarily and doesn't use database-level counting.

**Production Note:**
Always prefer database-level aggregations using \`annotate()\` when displaying lists of objects with related counts. This single query approach scales much better than per-object counting.

**Performance Comparison (1000 articles):**
- Option A: ~1001 queries, loads all comments into memory
- Option B: 1001 queries (N+1 problem)
- Option C: 1 query with GROUP BY
- Option D: ~1001 queries, memory efficient but slow

**Best Practice:**
\`\`\`python
# For list views with counts
class ArticleViewSet (viewsets.ModelViewSet):
    def get_queryset (self):
        return Article.objects.annotate(
            comment_count=Count('comments'),
            like_count=Count('likes'),
            view_count_sum=Sum('views__count')
        )

# Serializer can use annotated fields
class ArticleSerializer (serializers.ModelSerializer):
    comment_count = serializers.IntegerField (read_only=True)
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'comment_count']
\`\`\`

This approach ensures optimal performance even with thousands of articles and millions of comments.
      `,
  },
  {
    question:
      'When using Django signals, what is the recommended approach to ensure side effects (like sending emails) only occur after a database transaction commits successfully?',
    options: [
      'A) Use post_save signal without any special handling',
      'B) Wrap the signal handler in a try/except block',
      'C) Use transaction.on_commit() inside the signal handler',
      'D) Use pre_save signal instead of post_save',
    ],
    correctAnswer: 2,
    explanation: `
**Correct Answer: C) Use transaction.on_commit() inside the signal handler**

**Why C is correct:**
\`transaction.on_commit()\` ensures that the callback only executes after the database transaction commits successfully. This prevents sending emails or performing other side effects for objects that might be rolled back.

\`\`\`python
from django.db import transaction
from django.db.models.signals import post_save
from django.dispatch import receiver

@receiver (post_save, sender=Order)
def send_order_confirmation (sender, instance, created, **kwargs):
    if created:
        # ✅ Correct: Only send email after transaction commits
        transaction.on_commit(
            lambda: send_confirmation_email (instance.id)
        )

# If the transaction rolls back, email is never sent
\`\`\`

**Why A is incorrect:**
Without \`transaction.on_commit()\`, the signal fires immediately after the model save, even if the transaction hasn't committed yet. If the transaction later rolls back due to an error, the email would have already been sent for a non-existent order.

\`\`\`python
# ❌ Problem: Email sent even if transaction rolls back
@receiver (post_save, sender=Order)
def send_email (sender, instance, created, **kwargs):
    if created:
        send_confirmation_email (instance.id)  # Sent immediately!

# Later in the view:
try:
    with transaction.atomic():
        order = Order.objects.create(...)  # Signal fires here
        process_payment (order)  # This might fail and rollback
except:
    pass  # Transaction rolled back, but email was already sent!
\`\`\`

**Why B is incorrect:**
A try/except block handles exceptions within the signal handler itself but doesn't solve the transaction timing problem. The signal still fires before the transaction commits, so side effects occur prematurely.

**Why D is incorrect:**
\`pre_save\` fires before the database save occurs, which is even earlier in the process. This doesn't solve the transaction commit problem and would make it worse, as the object isn't even saved yet.

**Production Note:**
This is a critical issue in production systems. Without \`transaction.on_commit()\`, you might:
- Send confirmation emails for failed orders
- Trigger webhooks for uncommitted data
- Create external records that don't match your database
- Notify users of changes that never persisted

**Best Practice Pattern:**
\`\`\`python
@receiver (post_save, sender=Order)
def handle_order_creation (sender, instance, created, **kwargs):
    if created:
        # Multiple side effects, all deferred until commit
        transaction.on_commit (lambda: send_confirmation_email (instance.id))
        transaction.on_commit (lambda: update_inventory (instance.id))
        transaction.on_commit (lambda: trigger_analytics_event (instance.id))

# Alternative: Celery tasks with on_commit
@receiver (post_save, sender=Order)
def queue_order_tasks (sender, instance, created, **kwargs):
    if created:
        transaction.on_commit(
            lambda: process_order.delay (instance.id)
        )
\`\`\`

**When to Use:**
- ✅ Sending emails
- ✅ Calling external APIs
- ✅ Triggering webhooks
- ✅ Creating external records
- ✅ Sending push notifications
- ❌ Not needed for simple database operations within the same transaction

This pattern ensures data consistency and prevents embarrassing production bugs where users receive notifications for actions that never actually happened.
      `,
  },
].map(({ id, ...q }, idx) => ({ id: `django-mc-${idx + 1}`, ...q }));
