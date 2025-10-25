export const signalsHooks = {
  title: 'Signals & Hooks',
  id: 'signals-hooks',
  content: `
# Signals & Hooks

## Introduction

Django **signals** allow decoupled applications to get notified when actions occur elsewhere in the framework. They provide a way to execute code automatically when certain events happen, without tight coupling between components.

### What are Signals?

Signals are Django's implementation of the **Observer pattern**:
- **Sender**: The component that triggers the signal
- **Signal**: The event being broadcast
- **Receiver**: The function that gets called when the signal is sent

### Common Use Cases

- Send email when user registers
- Create user profile when user is created
- Update cache when model is saved
- Log model changes for auditing
- Trigger background tasks
- Invalidate caches
- Create notifications

By the end of this section, you'll understand:
- Built-in Django signals
- Creating custom signals
- Signal receivers and decorators
- Signal best practices and pitfalls
- When to use (and not use) signals
- Production patterns

---

## Built-in Model Signals

### pre_save and post_save

\`\`\`python
from django.db.models.signals import pre_save, post_save
from django.dispatch import receiver
from django.contrib.auth.models import User
from .models import Article, UserProfile

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """
    Create UserProfile when User is created
    """
    if created:
        UserProfile.objects.create(user=instance)
        print(f"Profile created for {instance.username}")

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    """
    Save UserProfile when User is saved
    """
    if hasattr(instance, 'profile'):
        instance.profile.save()

@receiver(pre_save, sender=Article)
def generate_slug(sender, instance, **kwargs):
    """
    Auto-generate slug before saving if not provided
    """
    if not instance.slug:
        from django.utils.text import slugify
        instance.slug = slugify(instance.title)
        
        # Handle duplicates
        if Article.objects.filter(slug=instance.slug).exists():
            instance.slug = f"{instance.slug}-{instance.id or 'new'}"
\`\`\`

### pre_delete and post_delete

\`\`\`python
from django.db.models.signals import pre_delete, post_delete
import os

@receiver(post_delete, sender=Article)
def delete_article_files(sender, instance, **kwargs):
    """
    Delete associated files when article is deleted
    """
    if instance.image:
        if os.path.isfile(instance.image.path):
            os.remove(instance.image.path)

@receiver(pre_delete, sender=User)
def log_user_deletion(sender, instance, **kwargs):
    """
    Log before user is deleted (for auditing)
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"User {instance.username} is being deleted")
\`\`\`

### m2m_changed

\`\`\`python
from django.db.models.signals import m2m_changed

@receiver(m2m_changed, sender=Article.tags.through)
def article_tags_changed(sender, instance, action, **kwargs):
    """
    Called when tags are added/removed from article
    """
    if action == "post_add":
        print(f"Tags added to {instance.title}")
        # Invalidate cache, trigger reindex, etc.
    
    elif action == "post_remove":
        print(f"Tags removed from {instance.title}")
    
    elif action == "post_clear":
        print(f"All tags cleared from {instance.title}")
\`\`\`

---

## Request/Response Signals

\`\`\`python
from django.core.signals import request_started, request_finished
from django.dispatch import receiver
import time

# Store request start time
_request_start_times = {}

@receiver(request_started)
def on_request_started(sender, environ, **kwargs):
    """Called when request starts"""
    request_id = id(environ)
    _request_start_times[request_id] = time.time()

@receiver(request_finished)
def on_request_finished(sender, **kwargs):
    """Called when request finishes"""
    # Log request duration, cleanup, etc.
    pass
\`\`\`

---

## Custom Signals

\`\`\`python
from django.dispatch import Signal, receiver

# Define custom signals
article_published = Signal()  # No arguments
article_viewed = Signal()     # Can pass providing_args (deprecated in Django 3.1+)
payment_completed = Signal()

# Send signal
class Article(models.Model):
    title = models.CharField(max_length=200)
    status = models.CharField(max_length=10)
    
    def publish(self):
        """Publish article and send signal"""
        self.status = 'published'
        self.published_at = timezone.now()
        self.save()
        
        # Send signal
        article_published.send(
            sender=self.__class__,
            instance=self,
            published_at=self.published_at
        )

# Receive signal
@receiver(article_published)
def on_article_published(sender, instance, **kwargs):
    """Handle article publication"""
    # Send email to subscribers
    send_email_to_subscribers(instance)
    
    # Create notification
    Notification.objects.create(
        type='article_published',
        title=f"New article: {instance.title}",
        article=instance
    )
    
    # Trigger Celery task
    from .tasks import generate_social_media_posts
    generate_social_media_posts.delay(instance.id)
\`\`\`

---

## Signal Receiver Patterns

### 1. Decorator Pattern

\`\`\`python
@receiver(post_save, sender=User)
def create_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)
\`\`\`

### 2. connect() Method

\`\`\`python
def create_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)

post_save.connect(create_profile, sender=User)
\`\`\`

### 3. Multiple Receivers

\`\`\`python
@receiver(post_save, sender=Article)
@receiver(post_save, sender=BlogPost)
@receiver(post_save, sender=News)
def invalidate_cache(sender, instance, **kwargs):
    """Invalidate cache for multiple models"""
    cache_key = f"{sender.__name__}_{instance.pk}"
    cache.delete(cache_key)
\`\`\`

### 4. Conditional Receivers

\`\`\`python
@receiver(post_save, sender=Article)
def on_article_save(sender, instance, created, **kwargs):
    """Different logic for create vs update"""
    if created:
        # Logic for new articles
        send_notification("New article created")
    else:
        # Logic for updates
        if instance.status_changed:
            send_notification("Article status changed")
\`\`\`

---

## Signal Best Practices

### 1. Keep Receivers Fast

\`\`\`python
# ❌ BAD: Slow synchronous operation in signal
@receiver(post_save, sender=Article)
def slow_receiver(sender, instance, **kwargs):
    # Slow operation blocks the save() call!
    send_email_to_1000_subscribers(instance)  # Takes 10 seconds
    generate_thumbnails(instance)  # Takes 5 seconds

# ✅ GOOD: Use Celery for slow operations
@receiver(post_save, sender=Article)
def fast_receiver(sender, instance, **kwargs):
    # Queue background tasks
    from .tasks import send_emails, generate_thumbnails
    send_emails.delay(instance.id)
    generate_thumbnails.delay(instance.id)
\`\`\`

### 2. Avoid Signal Cascades

\`\`\`python
# ❌ BAD: Signal triggers save, which triggers another signal
@receiver(post_save, sender=Article)
def update_related(sender, instance, **kwargs):
    instance.category.article_count += 1
    instance.category.save()  # Triggers post_save for Category!

# ✅ GOOD: Use update() to avoid triggering signals
@receiver(post_save, sender=Article)
def update_related(sender, instance, **kwargs):
    Category.objects.filter(pk=instance.category_id).update(
        article_count=F('article_count') + 1
    )
\`\`\`

### 3. Use dispatch_uid to Prevent Duplicates

\`\`\`python
@receiver(post_save, sender=User, dispatch_uid="create_user_profile_signal")
def create_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)

# dispatch_uid ensures this receiver is only registered once
\`\`\`

### 4. Register Signals in AppConfig.ready()

\`\`\`python
# apps.py
from django.apps import AppConfig

class ArticlesConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'articles'
    
    def ready(self):
        # Import signals
        import articles.signals  # noqa
\`\`\`

\`\`\`python
# signals.py (separate file)
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Article

@receiver(post_save, sender=Article)
def article_saved(sender, instance, **kwargs):
    pass
\`\`\`

---

## When NOT to Use Signals

### Anti-Pattern 1: Business Logic in Signals

\`\`\`python
# ❌ BAD: Core business logic in signal
@receiver(post_save, sender=Order)
def process_order(sender, instance, created, **kwargs):
    if created:
        charge_customer(instance)
        update_inventory(instance)
        send_confirmation_email(instance)

# ✅ GOOD: Explicit business logic in view/service
class OrderService:
    def create_order(self, data):
        order = Order.objects.create(**data)
        self.charge_customer(order)
        self.update_inventory(order)
        self.send_confirmation_email(order)
        return order
\`\`\`

### Anti-Pattern 2: Modifying Sender in post_save

\`\`\`python
# ❌ BAD: Modifying instance in post_save causes infinite loop!
@receiver(post_save, sender=Article)
def update_article(sender, instance, **kwargs):
    instance.view_count += 1
    instance.save()  # Infinite loop!

# ✅ GOOD: Use pre_save or update()
@receiver(pre_save, sender=Article)
def update_article(sender, instance, **kwargs):
    # Modify before save (no infinite loop)
    instance.slug = slugify(instance.title)

# OR use update() in post_save
@receiver(post_save, sender=Article)
def update_article(sender, instance, **kwargs):
    Article.objects.filter(pk=instance.pk).update(
        view_count=F('view_count') + 1
    )
\`\`\`

### Anti-Pattern 3: Complex Dependencies

\`\`\`python
# ❌ BAD: Complex signal dependencies
@receiver(post_save, sender=Article)
def on_article_save(sender, instance, **kwargs):
    update_category_stats(instance)
    update_author_stats(instance)
    invalidate_cache(instance)
    trigger_reindex(instance)
    send_notifications(instance)
# Hard to debug, order matters, hidden behavior

# ✅ GOOD: Explicit service layer
class ArticleService:
    def save_article(self, article):
        article.save()
        self.update_category_stats(article)
        self.update_author_stats(article)
        self.invalidate_cache(article)
        self.trigger_reindex(article)
        self.send_notifications(article)
# Clear, testable, explicit
\`\`\`

---

## Production Patterns

### Pattern 1: Audit Trail

\`\`\`python
from django.contrib.contenttypes.models import ContentType
import json

class AuditLog(models.Model):
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    object_id = models.PositiveIntegerField()
    action = models.CharField(max_length=10)  # create, update, delete
    changes = models.JSONField()
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)

@receiver(post_save)
def log_model_save(sender, instance, created, **kwargs):
    """Log all model saves"""
    if sender._meta.app_label == 'audit':
        return  # Don't log audit logs
    
    AuditLog.objects.create(
        content_type=ContentType.objects.get_for_model(sender),
        object_id=instance.pk,
        action='create' if created else 'update',
        changes=instance.__dict__
    )

@receiver(post_delete)
def log_model_delete(sender, instance, **kwargs):
    """Log all model deletions"""
    if sender._meta.app_label == 'audit':
        return
    
    AuditLog.objects.create(
        content_type=ContentType.objects.get_for_model(sender),
        object_id=instance.pk,
        action='delete',
        changes=instance.__dict__
    )
\`\`\`

### Pattern 2: Cache Invalidation

\`\`\`python
from django.core.cache import cache

CACHE_MODELS = ['Article', 'Category', 'Tag']

@receiver(post_save)
@receiver(post_delete)
def invalidate_model_cache(sender, instance, **kwargs):
    """Automatically invalidate cache for specific models"""
    if sender.__name__ in CACHE_MODELS:
        cache_key = f"{sender.__name__}_{instance.pk}"
        cache.delete(cache_key)
        
        # Invalidate list cache
        cache.delete(f"{sender.__name__}_list")
\`\`\`

### Pattern 3: Elasticsearch Reindexing

\`\`\`python
@receiver(post_save, sender=Article)
def reindex_article(sender, instance, **kwargs):
    """Reindex article in Elasticsearch"""
    from .search import index_article
    index_article.delay(instance.pk)  # Celery task

@receiver(post_delete, sender=Article)
def remove_from_index(sender, instance, **kwargs):
    """Remove from Elasticsearch"""
    from .search import delete_article
    delete_article.delay(instance.pk)
\`\`\`

### Pattern 4: Notifications

\`\`\`python
article_published = Signal()

@receiver(article_published)
def notify_subscribers(sender, instance, **kwargs):
    """Notify subscribers when article is published"""
    subscribers = instance.author.followers.all()
    
    notifications = [
        Notification(
            user=subscriber,
            type='article_published',
            title=f"{instance.author.name} published: {instance.title}",
            article=instance
        )
        for subscriber in subscribers
    ]
    
    Notification.objects.bulk_create(notifications)
    
    # Send push notifications
    from .tasks import send_push_notifications
    send_push_notifications.delay([s.id for s in subscribers], instance.id)
\`\`\`

---

## Testing Signals

\`\`\`python
from django.test import TestCase
from django.contrib.auth.models import User
from .models import UserProfile

class SignalTests(TestCase):
    def test_profile_created_on_user_creation(self):
        """Test that profile is created when user is created"""
        user = User.objects.create_user(
            username='testuser',
            email='test@example.com',
            password='testpass'
        )
        
        # Profile should be created by signal
        self.assertTrue(UserProfile.objects.filter(user=user).exists())
        self.assertEqual(user.profile.user, user)
    
    def test_signal_not_triggered_on_update(self):
        """Test that profile creation signal only triggers on create"""
        user = User.objects.create_user(username='test', password='pass')
        profile_count = UserProfile.objects.count()
        
        user.email = 'newemail@example.com'
        user.save()
        
        # Profile count should not increase
        self.assertEqual(UserProfile.objects.count(), profile_count)
\`\`\`

---

## Summary

**When to Use Signals:**
✅ Side effects (logging, auditing, notifications)
✅ Decoupled apps (plugin architecture)
✅ Framework integration (third-party apps)
✅ Cross-cutting concerns (caching, indexing)

**When NOT to Use Signals:**
❌ Core business logic (use services)
❌ Complex dependencies (use explicit calls)
❌ Slow operations (use Celery tasks instead)
❌ Modifying sender (causes infinite loops)

**Best Practices:**
1. Keep receivers fast (use Celery for slow operations)
2. Avoid signal cascades (use update() to skip signals)
3. Use dispatch_uid to prevent duplicates
4. Register signals in AppConfig.ready()
5. Test signal behavior thoroughly
6. Document signal side effects
7. Use custom signals for domain events

**Common Pitfalls:**
- ⚠️ Infinite loops (modifying sender in post_save)
- ⚠️ Hidden behavior (hard to debug)
- ⚠️ Performance issues (slow synchronous operations)
- ⚠️ Import order issues (register in AppConfig.ready())

Signals are powerful but should be used judiciously. For most business logic, explicit service layers are clearer and more maintainable than signals.
`,
};
