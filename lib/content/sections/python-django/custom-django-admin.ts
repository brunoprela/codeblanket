export const customDjangoAdmin = {
  title: 'Custom Django Admin',
  id: 'custom-django-admin',
  content: `
# Custom Django Admin

## Introduction

Django\'s **admin interface** is one of its most powerful features - a production-ready interface for managing your data that comes for free. Out of the box, it's functional but basic. Customizing the admin transforms it into a powerful management tool tailored to your application's needs.

### Why Customize the Admin?

- **Efficiency**: Add custom actions for bulk operations
- **Usability**: Improve UI with filters, search, and better layouts
- **Business Logic**: Add custom validation and workflows
- **Reporting**: Add statistics and dashboards
- **User Experience**: Make it intuitive for non-technical staff

By the end of this section, you'll understand:
- Customizing list views and forms
- Creating custom actions
- Adding filters and search
- Inline editing
- Custom admin views
- Permissions and security
- Production admin patterns

---

## Basic Admin Registration

### Simple Registration

\`\`\`python
# admin.py
from django.contrib import admin
from .models import Article

# Simple registration
admin.site.register(Article)
\`\`\`

### Custom Admin Class

\`\`\`python
from django.contrib import admin
from .models import Article, Category, Tag, Comment

@admin.register(Article)
class ArticleAdmin (admin.ModelAdmin):
    """Custom admin for Article model"""
    
    # List view configuration
    list_display = ['title', 'author', 'status', 'published_at', 'view_count']
    list_filter = ['status', 'created_at', 'published_at', 'category']
    search_fields = ['title', 'content', 'author__username']
    date_hierarchy = 'published_at'
    ordering = ['-published_at']
    
    # Form configuration
    fields = ['title', 'slug', 'author', 'category', 'tags', 
              'content', 'excerpt', 'featured', 'status']
    readonly_fields = ['created_at', 'updated_at', 'view_count']
    
    # Pagination
    list_per_page = 50
    
    # Actions
    actions = ['make_published', 'make_draft']
    
    def make_published (self, request, queryset):
        """Bulk publish articles"""
        updated = queryset.update (status='published')
        self.message_user (request, f'{updated} articles published.')
    make_published.short_description = "Publish selected articles"
    
    def make_draft (self, request, queryset):
        """Bulk draft articles"""
        updated = queryset.update (status='draft')
        self.message_user (request, f'{updated} articles marked as draft.')
    make_draft.short_description = "Mark as draft"
\`\`\`

---

## List Display Customization

### Custom Display Methods

\`\`\`python
from django.utils.html import format_html
from django.urls import reverse
from django.utils import timezone

@admin.register(Article)
class ArticleAdmin (admin.ModelAdmin):
    list_display = [
        'title_with_link',
        'author_name',
        'colored_status',
        'published_date',
        'view_count_badge',
        'has_comments',
        'thumbnail_preview'
    ]
    
    def title_with_link (self, obj):
        """Display title as clickable link to public page"""
        if obj.status == 'published':
            url = obj.get_absolute_url()
            return format_html(
                '<a href="{}" target="_blank">{}</a>',
                url, obj.title
            )
        return obj.title
    title_with_link.short_description = 'Title'
    title_with_link.admin_order_field = 'title'
    
    def author_name (self, obj):
        """Display author with link to their profile"""
        url = reverse('admin:auth_user_change', args=[obj.author.id])
        return format_html('<a href="{}">{}</a>', url, obj.author.get_full_name())
    author_name.short_description = 'Author'
    author_name.admin_order_field = 'author__username'
    
    def colored_status (self, obj):
        """Display status with color coding"""
        colors = {
            'draft': 'gray',
            'published': 'green',
            'archived': 'red',
        }
        color = colors.get (obj.status, 'black')
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            color, obj.get_status_display()
        )
    colored_status.short_description = 'Status'
    colored_status.admin_order_field = 'status'
    
    def published_date (self, obj):
        """Display published date or 'Not published'"""
        if obj.published_at:
            return obj.published_at.strftime('%Y-%m-%d %H:%M')
        return format_html('<em>Not published</em>')
    published_date.short_description = 'Published'
    published_date.admin_order_field = 'published_at'
    
    def view_count_badge (self, obj):
        """Display view count with badge"""
        if obj.view_count > 10000:
            color = 'red'
        elif obj.view_count > 1000:
            color = 'orange'
        else:
            color = 'blue'
        
        return format_html(
            '<span style="background-color: {}; color: white; '
            'padding: 3px 8px; border-radius: 3px;">{}</span>',
            color, obj.view_count
        )
    view_count_badge.short_description = 'Views'
    view_count_badge.admin_order_field = 'view_count'
    
    def has_comments (self, obj):
        """Display checkmark if article has comments"""
        count = obj.comments.count()
        if count > 0:
            return format_html('✓ ({})', count)
        return '✗'
    has_comments.short_description = 'Comments'
    has_comments.boolean = True
    
    def thumbnail_preview (self, obj):
        """Display thumbnail image"""
        if obj.image:
            return format_html(
                '<img src="{}" width="50" height="50" style="object-fit: cover;" />',
                obj.image.url
            )
        return '-'
    thumbnail_preview.short_description = 'Image'
\`\`\`

---

## Advanced Filters

### Custom List Filters

\`\`\`python
from django.contrib.admin import SimpleListFilter
from django.db.models import Count

class ViewCountFilter(SimpleListFilter):
    """Filter by view count ranges"""
    title = 'view count'
    parameter_name = 'views'
    
    def lookups (self, request, model_admin):
        return (
            ('low', 'Low (< 100)'),
            ('medium', 'Medium (100-1000)'),
            ('high', 'High (1000-10000)'),
            ('viral', 'Viral (> 10000)'),
        )
    
    def queryset (self, request, queryset):
        if self.value() == 'low':
            return queryset.filter (view_count__lt=100)
        if self.value() == 'medium':
            return queryset.filter (view_count__gte=100, view_count__lt=1000)
        if self.value() == 'high':
            return queryset.filter (view_count__gte=1000, view_count__lt=10000)
        if self.value() == 'viral':
            return queryset.filter (view_count__gte=10000)

class PublishStatusFilter(SimpleListFilter):
    """Filter by publication status"""
    title = 'publication status'
    parameter_name = 'pub_status'
    
    def lookups (self, request, model_admin):
        return (
            ('published', 'Published'),
            ('scheduled', 'Scheduled'),
            ('overdue', 'Overdue for Publication'),
            ('never', 'Never Published'),
        )
    
    def queryset (self, request, queryset):
        now = timezone.now()
        if self.value() == 'published':
            return queryset.filter (status='published', published_at__lte=now)
        if self.value() == 'scheduled':
            return queryset.filter (status='published', published_at__gt=now)
        if self.value() == 'overdue':
            return queryset.filter (status='draft', created_at__lt=now - timezone.timedelta (days=7))
        if self.value() == 'never':
            return queryset.filter (published_at__isnull=True)

class CommentCountFilter(SimpleListFilter):
    """Filter by comment count"""
    title = 'comments'
    parameter_name = 'comments'
    
    def lookups (self, request, model_admin):
        return (
            ('none', 'No comments'),
            ('some', 'Has comments'),
            ('popular', 'Many comments (10+)'),
        )
    
    def queryset (self, request, queryset):
        if self.value() == 'none':
            return queryset.annotate (comment_count=Count('comments')).filter (comment_count=0)
        if self.value() == 'some':
            return queryset.annotate (comment_count=Count('comments')).filter (comment_count__gt=0)
        if self.value() == 'popular':
            return queryset.annotate (comment_count=Count('comments')).filter (comment_count__gte=10)

@admin.register(Article)
class ArticleAdmin (admin.ModelAdmin):
    list_filter = [
        ViewCountFilter,
        PublishStatusFilter,
        CommentCountFilter,
        'status',
        'category',
        'featured',
        ('published_at', admin.DateFieldListFilter),
    ]
\`\`\`

---

## Inline Editing

### Tabular Inlines

\`\`\`python
from django.contrib import admin

class CommentInline (admin.TabularInline):
    """Inline editor for comments"""
    model = Comment
    extra = 0  # Don't show empty forms
    fields = ['author', 'text', 'status', 'created_at']
    readonly_fields = ['created_at']
    can_delete = True

class ImageInline (admin.TabularInline):
    """Inline editor for article images"""
    model = ArticleImage
    extra = 1
    fields = ['image', 'caption', 'order']
    
@admin.register(Article)
class ArticleAdmin (admin.ModelAdmin):
    inlines = [ImageInline, CommentInline]
\`\`\`

### Stacked Inlines

\`\`\`python
class ArticleMetadataInline (admin.StackedInline):
    """Stacked inline for detailed metadata"""
    model = ArticleMetadata
    can_delete = False
    fields = [
        ('meta_title', 'meta_description'),
        ('og_title', 'og_description', 'og_image'),
        ('twitter_title', 'twitter_description'),
        'canonical_url',
        'schema_markup',
    ]
    
@admin.register(Article)
class ArticleAdmin (admin.ModelAdmin):
    inlines = [ArticleMetadataInline, ImageInline, CommentInline]
\`\`\`

### Nested Inlines (with django-nested-admin)

\`\`\`python
import nested_admin

class ReplyInline (nested_admin.NestedTabularInline):
    """Inline for comment replies"""
    model = CommentReply
    extra = 0
    fields = ['author', 'text', 'created_at']
    readonly_fields = ['created_at']

class CommentInline (nested_admin.NestedTabularInline):
    """Inline for comments with nested replies"""
    model = Comment
    extra = 0
    inlines = [ReplyInline]
    fields = ['author', 'text', 'status', 'created_at']
    readonly_fields = ['created_at']

@admin.register(Article)
class ArticleAdmin (nested_admin.NestedModelAdmin):
    inlines = [CommentInline]
\`\`\`

---

## Custom Actions

### Bulk Actions

\`\`\`python
from django.contrib import messages
from django.utils import timezone

@admin.register(Article)
class ArticleAdmin (admin.ModelAdmin):
    actions = [
        'publish_articles',
        'unpublish_articles',
        'feature_articles',
        'send_notification',
        'export_to_csv',
        'duplicate_articles',
    ]
    
    def publish_articles (self, request, queryset):
        """Publish selected articles"""
        now = timezone.now()
        updated = queryset.filter (status='draft').update(
            status='published',
            published_at=now
        )
        
        if updated:
            self.message_user(
                request,
                f'{updated} articles published successfully.',
                messages.SUCCESS
            )
        else:
            self.message_user(
                request,
                'No draft articles selected.',
                messages.WARNING
            )
    publish_articles.short_description = "Publish selected articles"
    
    def unpublish_articles (self, request, queryset):
        """Unpublish selected articles"""
        updated = queryset.filter (status='published').update (status='draft')
        self.message_user (request, f'{updated} articles unpublished.')
    unpublish_articles.short_description = "Unpublish selected articles"
    
    def feature_articles (self, request, queryset):
        """Feature selected articles"""
        updated = queryset.update (featured=True)
        self.message_user (request, f'{updated} articles featured.')
    feature_articles.short_description = "Feature selected articles"
    
    def send_notification (self, request, queryset):
        """Send notification to subscribers"""
        from .tasks import send_article_notification
        
        for article in queryset:
            send_article_notification.delay (article.id)
        
        self.message_user(
            request,
            f'Notifications queued for {queryset.count()} articles.',
            messages.SUCCESS
        )
    send_notification.short_description = "Send notification to subscribers"
    
    def export_to_csv (self, request, queryset):
        """Export selected articles to CSV"""
        import csv
        from django.http import HttpResponse
        
        response = HttpResponse (content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="articles.csv"'
        
        writer = csv.writer (response)
        writer.writerow(['ID', 'Title', 'Author', 'Status', 'Published', 'Views'])
        
        for article in queryset:
            writer.writerow([
                article.id,
                article.title,
                article.author.username,
                article.status,
                article.published_at,
                article.view_count,
            ])
        
        return response
    export_to_csv.short_description = "Export to CSV"
    
    def duplicate_articles (self, request, queryset):
        """Duplicate selected articles"""
        count = 0
        for article in queryset:
            article.pk = None
            article.title = f"{article.title} (Copy)"
            article.slug = f"{article.slug}-copy"
            article.status = 'draft'
            article.published_at = None
            article.save()
            count += 1
        
        self.message_user (request, f'{count} articles duplicated.')
    duplicate_articles.short_description = "Duplicate selected articles"
\`\`\`

---

## Fieldsets and Form Customization

### Organized Fieldsets

\`\`\`python
@admin.register(Article)
class ArticleAdmin (admin.ModelAdmin):
    fieldsets = (
        ('Basic Information', {
            'fields': ('title', 'slug', 'author', 'category')
        }),
        ('Content', {
            'fields': ('content', 'excerpt', 'image'),
            'classes': ('wide',),
        }),
        ('Metadata', {
            'fields': ('tags', 'meta_description', 'keywords'),
        }),
        ('Publishing', {
            'fields': ('status', 'featured', 'published_at'),
            'classes': ('collapse',),
        }),
        ('Statistics', {
            'fields': ('view_count', 'created_at', 'updated_at'),
            'classes': ('collapse',),
        }),
    )
    
    readonly_fields = ['created_at', 'updated_at', 'view_count']
\`\`\`

### Custom Form Widgets

\`\`\`python
from django import forms
from django.contrib import admin

class ArticleAdminForm (forms.ModelForm):
    """Custom form with specialized widgets"""
    
    class Meta:
        model = Article
        fields = '__all__'
        widgets = {
            'content': forms.Textarea (attrs={'rows': 20, 'cols': 80}),
            'excerpt': forms.Textarea (attrs={'rows': 3, 'cols': 80}),
            'meta_description': forms.TextInput (attrs={
                'size': 80,
                'placeholder': 'SEO description (160 characters max)'
            }),
        }
    
    def clean_slug (self):
        """Validate slug is unique"""
        slug = self.cleaned_data['slug']
        if Article.objects.filter (slug=slug).exclude (pk=self.instance.pk).exists():
            raise forms.ValidationError('This slug is already in use.')
        return slug
    
    def clean (self):
        """Custom validation"""
        cleaned_data = super().clean()
        status = cleaned_data.get('status')
        published_at = cleaned_data.get('published_at')
        
        if status == 'published' and not published_at:
            raise forms.ValidationError(
                'Published articles must have a publication date.'
            )
        
        return cleaned_data

@admin.register(Article)
class ArticleAdmin (admin.ModelAdmin):
    form = ArticleAdminForm
\`\`\`

---

## Custom Admin Views

### Adding Custom URLs and Views

\`\`\`python
from django.urls import path
from django.shortcuts import render
from django.db.models import Count, Sum
from django.utils import timezone

@admin.register(Article)
class ArticleAdmin (admin.ModelAdmin):
    
    def get_urls (self):
        """Add custom URLs to admin"""
        urls = super().get_urls()
        custom_urls = [
            path('statistics/', self.admin_site.admin_view (self.statistics_view), name='article_statistics'),
            path('export/', self.admin_site.admin_view (self.export_view), name='article_export'),
        ]
        return custom_urls + urls
    
    def statistics_view (self, request):
        """Custom statistics view"""
        # Calculate statistics
        total_articles = Article.objects.count()
        published = Article.objects.filter (status='published').count()
        draft = Article.objects.filter (status='draft').count()
        total_views = Article.objects.aggregate(Sum('view_count'))['view_count__sum'] or 0
        
        # Top authors
        top_authors = Article.objects.values(
            'author__username'
        ).annotate(
            article_count=Count('id'),
            total_views=Sum('view_count')
        ).order_by('-article_count')[:10]
        
        # Monthly statistics
        now = timezone.now()
        monthly_stats = []
        for i in range(6):
            month_start = now - timezone.timedelta (days=30 * i)
            month_end = now - timezone.timedelta (days=30 * (i - 1))
            count = Article.objects.filter(
                published_at__gte=month_start,
                published_at__lt=month_end
            ).count()
            monthly_stats.append({
                'month': month_start.strftime('%B %Y'),
                'count': count
            })
        
        context = {
            'title': 'Article Statistics',
            'total_articles': total_articles,
            'published': published,
            'draft': draft,
            'total_views': total_views,
            'top_authors': top_authors,
            'monthly_stats': monthly_stats,
            'opts': self.model._meta,
        }
        
        return render (request, 'admin/article_statistics.html', context)
    
    def export_view (self, request):
        """Custom export view"""
        # Render export form/options
        context = {
            'title': 'Export Articles',
            'opts': self.model._meta,
        }
        return render (request, 'admin/article_export.html', context)
\`\`\`

### Adding Buttons to Change List

\`\`\`python
@admin.register(Article)
class ArticleAdmin (admin.ModelAdmin):
    
    change_list_template = 'admin/article_changelist.html'
    
    def changelist_view (self, request, extra_context=None):
        """Add extra context to change list"""
        extra_context = extra_context or {}
        extra_context['statistics_url'] = '/admin/articles/article/statistics/'
        return super().changelist_view (request, extra_context)
\`\`\`

\`\`\`django
{# templates/admin/article_changelist.html #}
{% extends "admin/change_list.html" %}

{% block object-tools-items %}
  {{ block.super }}
  <li>
    <a href="{% url 'admin:article_statistics' %}" class="button">
      View Statistics
    </a>
  </li>
  <li>
    <a href="{% url 'admin:article_export' %}" class="button">
      Export Articles
    </a>
  </li>
{% endblock %}
\`\`\`

---

## Permissions and Security

### Custom Permissions

\`\`\`python
class Article (models.Model):
    # ... fields ...
    
    class Meta:
        permissions = [
            ('can_publish', 'Can publish articles'),
            ('can_feature', 'Can feature articles'),
            ('can_view_statistics', 'Can view article statistics'),
        ]

@admin.register(Article)
class ArticleAdmin (admin.ModelAdmin):
    
    def has_publish_permission (self, request):
        """Check if user can publish articles"""
        return request.user.has_perm('articles.can_publish')
    
    def get_actions (self, request):
        """Filter actions based on permissions"""
        actions = super().get_actions (request)
        
        if not self.has_publish_permission (request):
            if 'publish_articles' in actions:
                del actions['publish_articles']
        
        return actions
    
    def get_queryset (self, request):
        """Filter queryset based on user permissions"""
        qs = super().get_queryset (request)
        
        if request.user.is_superuser:
            return qs
        
        # Regular staff only sees their own articles
        return qs.filter (author=request.user)
    
    def save_model (self, request, obj, form, change):
        """Auto-set author to current user for new articles"""
        if not change:  # New object
            obj.author = request.user
        super().save_model (request, obj, form, change)
\`\`\`

---

## Admin Site Customization

### Custom Admin Site

\`\`\`python
# admin.py
from django.contrib import admin
from django.contrib.admin import AdminSite

class MyAdminSite(AdminSite):
    site_header = 'My Blog Administration'
    site_title = 'Blog Admin'
    index_title = 'Welcome to Blog Administration'
    
    def index (self, request, extra_context=None):
        """Customize admin index page"""
        extra_context = extra_context or {}
        
        # Add custom statistics
        extra_context['total_articles'] = Article.objects.count()
        extra_context['published_articles'] = Article.objects.filter(
            status='published'
        ).count()
        extra_context['total_users'] = User.objects.count()
        
        return super().index (request, extra_context)

# Create custom admin site instance
admin_site = MyAdminSite (name='myadmin')

# Register models with custom site
admin_site.register(Article, ArticleAdmin)
admin_site.register(User, UserAdmin)

# urls.py
from django.urls import path
from .admin import admin_site

urlpatterns = [
    path('admin/', admin_site.urls),
]
\`\`\`

---

## Autocomplete Fields

\`\`\`python
@admin.register(Author)
class AuthorAdmin (admin.ModelAdmin):
    search_fields = ['username', 'email', 'first_name', 'last_name']

@admin.register(Category)
class CategoryAdmin (admin.ModelAdmin):
    search_fields = ['name']

@admin.register(Tag)
class TagAdmin (admin.ModelAdmin):
    search_fields = ['name']

@admin.register(Article)
class ArticleAdmin (admin.ModelAdmin):
    autocomplete_fields = ['author', 'category', 'tags']
    # Instead of dropdown, shows searchable autocomplete
\`\`\`

---

## Production Best Practices

### 1. Performance Optimization

\`\`\`python
@admin.register(Article)
class ArticleAdmin (admin.ModelAdmin):
    
    def get_queryset (self, request):
        """Optimize queries with select_related/prefetch_related"""
        qs = super().get_queryset (request)
        return qs.select_related('author', 'category').prefetch_related('tags')
    
    list_select_related = ['author', 'category']
    # Django automatically does select_related for these
\`\`\`

### 2. Secure Sensitive Data

\`\`\`python
@admin.register(User)
class UserAdmin (admin.ModelAdmin):
    list_display = ['username', 'email', 'is_staff']
    
    # Don't show passwords in admin
    exclude = ['password']
    
    # Or make read-only
    readonly_fields = ['password', 'last_login', 'date_joined']
\`\`\`

### 3. Add Help Text

\`\`\`python
@admin.register(Article)
class ArticleAdmin (admin.ModelAdmin):
    fieldsets = (
        ('Content', {
            'fields': ('title', 'content'),
            'description': 'Enter the article title and main content here.'
        }),
        ('SEO', {
            'fields': ('meta_description', 'keywords'),
            'description': 'Optimize for search engines (160 char max for description).'
        }),
    )
\`\`\`

---

## Summary

**Key Admin Customizations:**1. **List Display**: Custom columns, formatting, links
2. **Filters**: Custom filters for better searching
3. **Actions**: Bulk operations for efficiency
4. **Inlines**: Edit related objects on same page
5. **Fieldsets**: Organized, collapsible sections
6. **Custom Views**: Statistics, reports, exports
7. **Permissions**: Fine-grained access control
8. **Performance**: Query optimization

**Production Checklist:**
- ✅ Optimize queries (select_related/prefetch_related)
- ✅ Add meaningful list_display columns
- ✅ Implement useful filters and search
- ✅ Create bulk actions for common tasks
- ✅ Use readonly_fields for calculated values
- ✅ Add custom permissions for sensitive actions
- ✅ Provide clear help text
- ✅ Test admin with different user roles
- ✅ Monitor admin performance

**Common Patterns:**
- Status badges with colors
- Thumbnail previews
- Clickable external links
- Inline editing for related objects
- CSV/Excel export actions
- Custom statistics dashboards
- Autocomplete for foreign keys
- Bulk publish/unpublish workflows

Django admin is incredibly powerful when customized properly. It can serve as your application's primary management interface, saving countless hours of development time while providing a professional interface for non-technical users.
`,
};
