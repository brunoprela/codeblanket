export const signalsHooksQuiz = [
  {
    id: 1,
    question:
      'Explain Django signals, their use cases, and potential pitfalls. Compare signals with other approaches like overriding save() methods or using custom managers. When should you use signals vs alternatives?',
    answer: `
**Django Signals Overview:**

Signals allow decoupled applications to get notified when actions occur elsewhere in the application. They implement the observer pattern.

**Built-in Signals:**
1. **pre_save / post_save**: Before/after model save
2. **pre_delete / post_delete**: Before/after model delete
3. **m2m_changed**: When ManyToMany relationships change
4. **request_started / request_finished**: Request lifecycle
5. **post_migrate**: After migrations run

**Example Use Cases:**

\`\`\`python
from django.db.models.signals import post_save, pre_delete
from django.dispatch import receiver
from django.contrib.auth.models import User

# 1. Create related objects automatically
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)

# 2. Clear cache when model changes
@receiver(post_save, sender=Article)
def invalidate_article_cache(sender, instance, **kwargs):
    cache.delete(f'article_{instance.id}')

# 3. Send notifications
@receiver(post_save, sender=Comment)
def notify_author(sender, instance, created, **kwargs):
    if created:
        send_email_notification(instance.article.author, instance)

# 4. Log changes
@receiver(pre_delete, sender=Article)
def log_deletion(sender, instance, **kwargs):
    AuditLog.objects.create(
        action='DELETE',
        model='Article',
        object_id=instance.id
    )
\`\`\`

**Signals vs save() Override:**

\`\`\`python
# Option 1: Override save()
class Article(models.Model):
    def save(self, *args, **kwargs):
        # Tight coupling - logic in model
        super().save(*args, **kwargs)
        cache.delete(f'article_{self.id}')
        send_notification(self.author)

# Option 2: Signals
class Article(models.Model):
    pass  # Clean model

@receiver(post_save, sender=Article)
def handle_article_save(sender, instance, **kwargs):
    # Loose coupling - logic separated
    cache.delete(f'article_{instance.id}')
    send_notification(instance.author)
\`\`\`

**When to Use Signals:**
✅ **Good Use Cases:**
- Cross-app communication (loose coupling)
- Side effects that don't belong in model
- Multiple handlers for same event
- Plugin/extension architecture
- Audit logging
- Cache invalidation across modules

❌ **Avoid Signals For:**
- Simple model logic (use save() override)
- Performance-critical paths (signals add overhead)
- Direct model relationships (use ForeignKey)
- Complex business logic (use service layer)

**Potential Pitfalls:**

**1. Transaction Issues:**
\`\`\`python
# ❌ Problem: Signal fires before transaction commits
@receiver(post_save, sender=Order)
def send_confirmation_email(sender, instance, created, **kwargs):
    if created:
        # Email sent even if transaction rolls back!
        send_email(instance.user, 'Order confirmed')

# ✅ Solution: Use transaction.on_commit()
from django.db import transaction

@receiver(post_save, sender=Order)
def send_confirmation_email(sender, instance, created, **kwargs):
    if created:
        transaction.on_commit(lambda: send_email(instance.user, 'Order confirmed'))
\`\`\`

**2. Infinite Loops:**
\`\`\`python
# ❌ Problem: Signal triggers itself
@receiver(post_save, sender=Article)
def update_view_count(sender, instance, **kwargs):
    instance.view_count += 1
    instance.save()  # Triggers post_save again!

# ✅ Solution: Use update() or flag
@receiver(post_save, sender=Article)
def update_view_count(sender, instance, created, **kwargs):
    if not created:
        Article.objects.filter(id=instance.id).update(
            view_count=F('view_count') + 1
        )  # update() doesn't trigger signals
\`\`\`

**3. Performance Impact:**
\`\`\`python
# ❌ Problem: Slow signals block save
@receiver(post_save, sender=Article)
def expensive_processing(sender, instance, **kwargs):
    # Expensive operation blocks save
    generate_thumbnails(instance)
    update_search_index(instance)

# ✅ Solution: Use async tasks
@receiver(post_save, sender=Article)
def queue_processing(sender, instance, created, **kwargs):
    if created:
        process_article.delay(instance.id)  # Celery task
\`\`\`

**4. Testing Difficulty:**
\`\`\`python
# Signals make testing harder
# Need to ensure signals are connected/disconnected properly

from django.test import TestCase, override_settings

class ArticleTest(TestCase):
    @override_settings(DISABLE_SIGNALS=True)
    def test_without_signals(self):
        # Test without side effects
        article = Article.objects.create(...)
\`\`\`

**Best Practices:**

1. ✅ Use transaction.on_commit() for side effects
2. ✅ Keep signal handlers lightweight
3. ✅ Move expensive operations to background tasks
4. ✅ Document all signals in your codebase
5. ✅ Use sender parameter to avoid unnecessary calls
6. ✅ Be explicit about signal registration
7. ❌ Don't modify the sender instance in post_save
8. ❌ Don't use signals for core business logic
9. ❌ Don't create circular dependencies

**When to Use Alternatives:**

**Use save() override when:**
- Logic is core to model behavior
- Always needs to happen
- Performance is critical

**Use custom managers when:**
- Creating objects with default behavior
- Reusable query logic needed

**Use service layer when:**
- Complex business workflows
- Multiple models involved
- Need transactional control

**Use signals when:**
- Loose coupling required
- Plugin architecture needed
- Cross-app communication
- Optional side effects

Signals are powerful but should be used judiciously. When in doubt, prefer explicit code (save() override or service layer) over "magic" (signals).
      `,
  },
  {
    question:
      'Describe how to implement a robust audit logging system using Django signals. Include handling of ManyToMany fields, tracking field changes, and performance considerations.',
    answer: `
**Comprehensive Audit Logging with Signals:**

**1. Basic Audit Log Model:**

\`\`\`python
from django.db import models
from django.contrib.auth.models import User
from django.contrib.contenttypes.models import ContentType
from django.contrib.contenttypes.fields import GenericForeignKey
import json

class AuditLog(models.Model):
    ACTION_CHOICES = [
        ('CREATE', 'Create'),
        ('UPDATE', 'Update'),
        ('DELETE', 'Delete'),
        ('M2M_ADD', 'M2M Add'),
        ('M2M_REMOVE', 'M2M Remove'),
    ]
    
    # Who
    user = models.ForeignKey(User, null=True, on_delete=models.SET_NULL)
    
    # What
    action = models.CharField(max_length=20, choices=ACTION_CHOICES)
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    object_id = models.PositiveIntegerField()
    content_object = GenericForeignKey('content_type', 'object_id')
    
    # Details
    object_repr = models.CharField(max_length=200)
    changes = models.JSONField(null=True, blank=True)
    
    # When
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # Context
    ip_address = models.GenericIPAddressField(null=True)
    user_agent = models.TextField(null=True)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['content_type', 'object_id']),
            models.Index(fields=['-timestamp']),
        ]
\`\`\`

**2. Track Field Changes:**

\`\`\`python
from django.db.models.signals import pre_save, post_save, post_delete, m2m_changed
from django.dispatch import receiver
from threading import local

# Thread-local storage for request context
_thread_locals = local()

def get_current_user():
    return getattr(_thread_locals, 'user', None)

def set_current_user(user):
    _thread_locals.user = user

# Middleware to capture user
class AuditMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        set_current_user(request.user if request.user.is_authenticated else None)
        response = self.get_response(request)
        set_current_user(None)
        return response

# Signal handlers
@receiver(pre_save)
def capture_pre_save_state(sender, instance, **kwargs):
    """Store original state before save"""
    if instance.pk:
        try:
            original = sender.objects.get(pk=instance.pk)
            instance._original_state = {
                field.name: getattr(original, field.name)
                for field in instance._meta.fields
            }
        except sender.DoesNotExist:
            instance._original_state = None
    else:
        instance._original_state = None

@receiver(post_save)
def log_model_change(sender, instance, created, **kwargs):
    """Log create/update"""
    # Skip audit log model itself
    if sender == AuditLog:
        return
    
    # Skip abstract models
    if instance._meta.abstract:
        return
    
    action = 'CREATE' if created else 'UPDATE'
    
    # Calculate changes for updates
    changes = None
    if not created and hasattr(instance, '_original_state') and instance._original_state:
        changes = {}
        for field in instance._meta.fields:
            if field.name in ['id', 'created_at', 'updated_at']:
                continue
            
            old_value = instance._original_state.get(field.name)
            new_value = getattr(instance, field.name)
            
            if old_value != new_value:
                changes[field.name] = {
                    'old': str(old_value),
                    'new': str(new_value),
                }
    
    # Create audit log
    AuditLog.objects.create(
        user=get_current_user(),
        action=action,
        content_type=ContentType.objects.get_for_model(sender),
        object_id=instance.pk,
        object_repr=str(instance),
        changes=changes,
    )

@receiver(post_delete)
def log_model_deletion(sender, instance, **kwargs):
    """Log deletion"""
    if sender == AuditLog:
        return
    
    AuditLog.objects.create(
        user=get_current_user(),
        action='DELETE',
        content_type=ContentType.objects.get_for_model(sender),
        object_id=instance.pk,
        object_repr=str(instance),
    )
\`\`\`

**3. Handle ManyToMany Fields:**

\`\`\`python
@receiver(m2m_changed)
def log_m2m_changes(sender, instance, action, pk_set, **kwargs):
    """Log M2M field changes"""
    if action in ['post_add', 'post_remove', 'post_clear']:
        audit_action = {
            'post_add': 'M2M_ADD',
            'post_remove': 'M2M_REMOVE',
            'post_clear': 'M2M_CLEAR',
        }[action]
        
        changes = {
            'field': sender._meta.label,
            'action': action,
            'pks': list(pk_set) if pk_set else [],
        }
        
        AuditLog.objects.create(
            user=get_current_user(),
            action=audit_action,
            content_type=ContentType.objects.get_for_model(instance),
            object_id=instance.pk,
            object_repr=str(instance),
            changes=changes,
        )
\`\`\`

**4. Performance Optimizations:**

\`\`\`python
# Selective auditing with decorator
def audited(func):
    """Decorator to enable auditing for specific operations"""
    def wrapper(*args, **kwargs):
        _thread_locals.enable_audit = True
        result = func(*args, **kwargs)
        _thread_locals.enable_audit = False
        return result
    return wrapper

@receiver(post_save)
def conditional_audit_log(sender, instance, created, **kwargs):
    """Only audit if enabled"""
    if not getattr(_thread_locals, 'enable_audit', True):
        return
    
    # Audit logic...

# Bulk operations without auditing
@transaction.atomic
def bulk_update_without_audit():
    _thread_locals.enable_audit = False
    Article.objects.filter(status='draft').update(status='archived')
    _thread_locals.enable_audit = True

# Async audit logging
@receiver(post_save)
def async_audit_log(sender, instance, created, **kwargs):
    """Queue audit logging as background task"""
    from .tasks import create_audit_log
    
    create_audit_log.delay(
        model=sender._meta.label,
        object_id=instance.pk,
        action='CREATE' if created else 'UPDATE',
        user_id=get_current_user().id if get_current_user() else None,
    )
\`\`\`

**5. Querying Audit Logs:**

\`\`\`python
class AuditLogQuerySet(models.QuerySet):
    def for_object(self, obj):
        """Get all logs for an object"""
        content_type = ContentType.objects.get_for_model(obj)
        return self.filter(content_type=content_type, object_id=obj.pk)
    
    def for_user(self, user):
        """Get all logs by a user"""
        return self.filter(user=user)
    
    def for_model(self, model):
        """Get all logs for a model"""
        content_type = ContentType.objects.get_for_model(model)
        return self.filter(content_type=content_type)
    
    def creates(self):
        return self.filter(action='CREATE')
    
    def updates(self):
        return self.filter(action='UPDATE')
    
    def deletes(self):
        return self.filter(action='DELETE')

class AuditLog(models.Model):
    # ... fields ...
    
    objects = AuditLogQuerySet.as_manager()

# Usage:
article_history = AuditLog.objects.for_object(article)
user_actions = AuditLog.objects.for_user(user).creates()
\`\`\`

**6. Production Considerations:**

**Performance:**
- ✅ Use database indexes on commonly queried fields
- ✅ Consider partitioning audit tables by date
- ✅ Archive old audit logs to separate storage
- ✅ Use async tasks for expensive audit operations
- ✅ Batch audit log creation when possible

**Storage:**
- ✅ Implement retention policies
- ✅ Compress old audit logs
- ✅ Move to cheaper storage after time period

**Privacy/Compliance:**
- ✅ Don't log sensitive fields (passwords, PII)
- ✅ Implement GDPR-compliant deletion
- ✅ Encrypt audit logs at rest
- ✅ Control access to audit logs

This comprehensive audit system provides full change tracking while maintaining performance and compliance.
      `,
  },
  {
    question:
      'Explain how to build a plugin system using Django signals. Describe the architecture, registration mechanism, and how third-party apps can hook into your application without modifying core code.',
    answer: `
**Plugin Architecture with Django Signals:**

**1. Core Plugin System:**

\`\`\`python
# core/plugins.py
from django.dispatch import Signal
from typing import Dict, Callable, List

# Define plugin signals
plugin_pre_article_publish = Signal()  # providing_args=['article', 'request']
plugin_post_article_publish = Signal()  # providing_args=['article', 'request', 'result']
plugin_article_list_filter = Signal()  # providing_args=['queryset', 'request']
plugin_dashboard_widget = Signal()  # providing_args=['request']

class PluginRegistry:
    """Central registry for all plugins"""
    
    _plugins: Dict[str, Dict] = {}
    
    @classmethod
    def register(cls, name: str, plugin_class):
        """Register a plugin"""
        if name in cls._plugins:
            raise ValueError(f'Plugin {name} already registered')
        
        plugin_instance = plugin_class()
        cls._plugins[name] = {
            'name': name,
            'instance': plugin_instance,
            'enabled': True,
        }
        
        # Auto-connect signals if plugin has handlers
        if hasattr(plugin_instance, 'connect_signals'):
            plugin_instance.connect_signals()
    
    @classmethod
    def unregister(cls, name: str):
        """Unregister a plugin"""
        if name in cls._plugins:
            plugin = cls._plugins[name]['instance']
            if hasattr(plugin, 'disconnect_signals'):
                plugin.disconnect_signals()
            del cls._plugins[name]
    
    @classmethod
    def get_plugin(cls, name: str):
        """Get a plugin by name"""
        return cls._plugins.get(name, {}).get('instance')
    
    @classmethod
    def list_plugins(cls) -> List[Dict]:
        """List all registered plugins"""
        return [
            {
                'name': info['name'],
                'enabled': info['enabled'],
                'plugin': info['instance'],
            }
            for info in cls._plugins.values()
        ]
\`\`\`

**2. Base Plugin Class:**

\`\`\`python
# core/plugins.py
from abc import ABC, abstractmethod

class BasePlugin(ABC):
    """Base class for all plugins"""
    
    name = None
    version = '1.0.0'
    author = None
    description = None
    
    def __init__(self):
        self.receivers = []
    
    @abstractmethod
    def connect_signals(self):
        """Connect plugin to signals - implemented by subclasses"""
        pass
    
    def disconnect_signals(self):
        """Disconnect all signal receivers"""
        for receiver_func in self.receivers:
            # Disconnect from all signals
            for signal in [plugin_pre_article_publish, plugin_post_article_publish]:
                signal.disconnect(receiver_func)
    
    def register_receiver(self, signal, handler):
        """Helper to register and track signal receivers"""
        signal.connect(handler, weak=False)
        self.receivers.append(handler)
\`\`\`

**3. Example Plugins:**

\`\`\`python
# plugins/spam_checker.py
from core.plugins import BasePlugin, plugin_pre_article_publish, PluginRegistry

class SpamCheckerPlugin(BasePlugin):
    name = 'spam_checker'
    description = 'Check articles for spam before publishing'
    
    def connect_signals(self):
        self.register_receiver(
            plugin_pre_article_publish,
            self.check_spam
        )
    
    def check_spam(self, sender, article, request, **kwargs):
        """Check if article contains spam"""
        spam_keywords = ['spam', 'click here', 'buy now']
        content_lower = article.content.lower()
        
        for keyword in spam_keywords:
            if keyword in content_lower:
                raise ValidationError(f'Spam detected: contains "{keyword}"')

# Register plugin
PluginRegistry.register('spam_checker', SpamCheckerPlugin)

# plugins/auto_tagger.py
class AutoTaggerPlugin(BasePlugin):
    name = 'auto_tagger'
    description = 'Automatically add tags to articles'
    
    def connect_signals(self):
        self.register_receiver(
            plugin_post_article_publish,
            self.add_auto_tags
        )
    
    def add_auto_tags(self, sender, article, request, result, **kwargs):
        """Add tags based on content"""
        keywords = self.extract_keywords(article.content)
        
        for keyword in keywords:
            tag, created = Tag.objects.get_or_create(name=keyword)
            article.tags.add(tag)
    
    def extract_keywords(self, text):
        # Simple keyword extraction
        return ['django', 'python']  # Placeholder

PluginRegistry.register('auto_tagger', AutoTaggerPlugin)

# plugins/dashboard_analytics.py
class AnalyticsWidgetPlugin(BasePlugin):
    name = 'analytics_widget'
    description = 'Add analytics widget to dashboard'
    
    def connect_signals(self):
        self.register_receiver(
            plugin_dashboard_widget,
            self.render_widget
        )
    
    def render_widget(self, sender, request, **kwargs):
        """Return widget HTML"""
        return {
            'html': '<div class="analytics-widget">Analytics Data</div>',
            'position': 'top-right',
        }

PluginRegistry.register('analytics_widget', AnalyticsWidgetPlugin)
\`\`\`

**4. Core Application Integration:**

\`\`\`python
# articles/views.py
from core.plugins import plugin_pre_article_publish, plugin_post_article_publish

def publish_article(request, article_id):
    article = get_object_or_404(Article, id=article_id)
    
    # Send pre-publish signal (plugins can modify or reject)
    responses = plugin_pre_article_publish.send(
        sender=publish_article,
        article=article,
        request=request
    )
    
    # Check if any plugin rejected publication
    for receiver, response in responses:
        if isinstance(response, Exception):
            return JsonResponse({'error': str(response)}, status=400)
    
    # Publish article
    article.status = 'published'
    article.published_at = timezone.now()
    article.save()
    
    # Send post-publish signal
    plugin_post_article_publish.send(
        sender=publish_article,
        article=article,
        request=request,
        result={'success': True}
    )
    
    return JsonResponse({'status': 'published'})

# Dashboard view
def dashboard(request):
    # Collect widgets from plugins
    widgets = []
    responses = plugin_dashboard_widget.send(sender=dashboard, request=request)
    
    for receiver, response in responses:
        if isinstance(response, dict) and 'html' in response:
            widgets.append(response)
    
    return render(request, 'dashboard.html', {'widgets': widgets})
\`\`\`

**5. Plugin Configuration:**

\`\`\`python
# settings.py
INSTALLED_PLUGINS = [
    'plugins.spam_checker.SpamCheckerPlugin',
    'plugins.auto_tagger.AutoTaggerPlugin',
    'plugins.analytics_widget.AnalyticsWidgetPlugin',
]

# Auto-discover and register plugins
from django.utils.module_loading import import_string
from core.plugins import PluginRegistry

for plugin_path in settings.INSTALLED_PLUGINS:
    try:
        plugin_class = import_string(plugin_path)
        plugin_name = plugin_class.name or plugin_path.split('.')[-1]
        PluginRegistry.register(plugin_name, plugin_class)
    except Exception as e:
        logger.error(f'Failed to load plugin {plugin_path}: {e}')
\`\`\`

**6. Plugin Management API:**

\`\`\`python
# api/views.py
from rest_framework.decorators import api_view
from core.plugins import PluginRegistry

@api_view(['GET'])
def list_plugins(request):
    """List all registered plugins"""
    plugins = PluginRegistry.list_plugins()
    return Response([
        {
            'name': p['name'],
            'enabled': p['enabled'],
            'version': p['plugin'].version,
            'description': p['plugin'].description,
        }
        for p in plugins
    ])

@api_view(['POST'])
def toggle_plugin(request, plugin_name):
    """Enable/disable a plugin"""
    plugin = PluginRegistry.get_plugin(plugin_name)
    
    if not plugin:
        return Response({'error': 'Plugin not found'}, status=404)
    
    enabled = request.data.get('enabled', True)
    
    if enabled:
        plugin.connect_signals()
    else:
        plugin.disconnect_signals()
    
    return Response({'status': 'ok', 'enabled': enabled})
\`\`\`

**7. Third-Party Plugin Example:**

\`\`\`python
# third_party_plugin/seo_optimizer.py
\"\"\"
Third-party plugin that doesn't modify core code
\"\"\"
from myapp.core.plugins import BasePlugin, plugin_post_article_publish, PluginRegistry

class SEOOptimizerPlugin(BasePlugin):
    name = 'seo_optimizer'
    version = '2.0.1'
    author = 'Third Party Dev'
    description = 'Optimize articles for SEO'
    
    def connect_signals(self):
        self.register_receiver(
            plugin_post_article_publish,
            self.optimize_seo
        )
    
    def optimize_seo(self, sender, article, **kwargs):
        # SEO optimization logic
        article.meta_description = self.generate_meta_description(article)
        article.meta_keywords = self.extract_keywords(article)
        article.save(update_fields=['meta_description', 'meta_keywords'])
    
    def generate_meta_description(self, article):
        # Generate optimized meta description
        return article.content[:160]
    
    def extract_keywords(self, article):
        # Extract SEO keywords
        return 'keyword1, keyword2'

# Plugin auto-registers when imported
PluginRegistry.register('seo_optimizer', SEOOptimizerPlugin)
\`\`\`

**Benefits of Signal-Based Plugin System:**
- ✅ No core code modification required
- ✅ Plugins can be enabled/disabled dynamically
- ✅ Clean separation of concerns
- ✅ Easy to distribute third-party plugins
- ✅ Plugins can intercept and modify behavior
- ✅ Multiple plugins can handle same event
- ✅ Plugin isolation (one plugin failure doesn't break others)

This architecture enables a flexible, extensible application where functionality can be added without touching core code.
      `,
  },
].map(({ id, ...q }, idx) => ({
  id: `django-q-${idx + 1}`,
  question: q.question,
  sampleAnswer: String(q.answer),
  keyPoints: [],
}));
