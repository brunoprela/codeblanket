export const customDjangoAdminQuiz = [
  {
    id: 1,
    question:
      'Explain how to customize Django Admin with custom actions, filters, and inline editing. Provide examples of advanced admin customization including custom views, dashboard widgets, and permission-based UI modifications.',
    answer: `
**Django Admin Customization:**

**1. ModelAdmin Basics:**

\`\`\`python
from django.contrib import admin
from .models import Article

@admin.register(Article)
class ArticleAdmin (admin.ModelAdmin):
    # List display
    list_display = ['title', 'author', 'status', 'published_at', 'view_count']
    list_display_links = ['title']
    
    # Filters
    list_filter = ['status', 'published_at', 'category']
    
    # Search
    search_fields = ['title', 'content', 'author__username']
    
    # Ordering
    ordering = ['-published_at']
    
    # Read-only fields
    readonly_fields = ['view_count', 'created_at', 'updated_at']
    
    # Fieldsets for organization
    fieldsets = (
        ('Basic Information', {
            'fields': ('title', 'slug', 'author', 'category')
        }),
        ('Content', {
            'fields': ('content', 'excerpt'),
            'classes': ('wide',),
        }),
        ('Metadata', {
            'fields': ('status', 'featured', 'published_at'),
        }),
        ('Statistics', {
            'fields': ('view_count', 'created_at', 'updated_at'),
            'classes': ('collapse',),
        }),
    )
    
    # Prepopulate slug from title
    prepopulated_fields = {'slug': ('title',)}
    
    # Date hierarchy
    date_hierarchy = 'published_at'
    
    # Pagination
    list_per_page = 25
\`\`\`

**2. Custom Actions:**

\`\`\`python
@admin.register(Article)
class ArticleAdmin (admin.ModelAdmin):
    actions = ['make_published', 'make_draft', 'export_to_csv']
    
    @admin.action (description='Mark selected articles as published')
    def make_published (self, request, queryset):
        updated = queryset.update (status='published', published_at=timezone.now())
        self.message_user (request, f'{updated} articles marked as published.', messages.SUCCESS)
    
    @admin.action (description='Mark selected articles as draft')
    def make_draft (self, request, queryset):
        updated = queryset.update (status='draft')
        self.message_user (request, f'{updated} articles marked as draft.', messages.INFO)
    
    @admin.action (description='Export selected articles to CSV')
    def export_to_csv (self, request, queryset):
        import csv
        from django.http import HttpResponse
        
        response = HttpResponse (content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="articles.csv"'
        
        writer = csv.writer (response)
        writer.writerow(['Title', 'Author', 'Status', 'Published Date'])
        
        for article in queryset:
            writer.writerow([
                article.title,
                article.author.username,
                article.status,
                article.published_at
            ])
        
        return response
\`\`\`

**3. Custom Filters:**

\`\`\`python
from django.contrib.admin import SimpleListFilter

class PublishedDateFilter(SimpleListFilter):
    title = 'published date'
    parameter_name = 'published'
    
    def lookups (self, request, model_admin):
        return (
            ('today', 'Today'),
            ('week', 'This week'),
            ('month', 'This month'),
            ('year', 'This year'),
        )
    
    def queryset (self, request, queryset):
        from datetime import timedelta
        from django.utils import timezone
        
        now = timezone.now()
        
        if self.value() == 'today':
            return queryset.filter (published_at__date=now.date())
        elif self.value() == 'week':
            week_ago = now - timedelta (days=7)
            return queryset.filter (published_at__gte=week_ago)
        elif self.value() == 'month':
            month_ago = now - timedelta (days=30)
            return queryset.filter (published_at__gte=month_ago)
        elif self.value() == 'year':
            year_ago = now - timedelta (days=365)
            return queryset.filter (published_at__gte=year_ago)

class ViewCountFilter(SimpleListFilter):
    title = 'view count'
    parameter_name = 'views'
    
    def lookups (self, request, model_admin):
        return (
            ('low', 'Low (< 100)'),
            ('medium', 'Medium (100-1000)'),
            ('high', 'High (> 1000)'),
        )
    
    def queryset (self, request, queryset):
        if self.value() == 'low':
            return queryset.filter (view_count__lt=100)
        elif self.value() == 'medium':
            return queryset.filter (view_count__gte=100, view_count__lte=1000)
        elif self.value() == 'high':
            return queryset.filter (view_count__gt=1000)

@admin.register(Article)
class ArticleAdmin (admin.ModelAdmin):
    list_filter = [PublishedDateFilter, ViewCountFilter, 'status', 'category']
\`\`\`

**4. Inline Editing:**

\`\`\`python
class CommentInline (admin.TabularInline):
    model = Comment
    extra = 1
    fields = ['user', 'text', 'approved']
    readonly_fields = ['user', 'created_at']
    
    def get_queryset (self, request):
        # Show only recent comments by default
        qs = super().get_queryset (request)
        return qs.order_by('-created_at')[:10]

class TagInline (admin.TabularInline):
    model = Article.tags.through
    extra = 1

class ImageInline (admin.StackedInline):
    model = ArticleImage
    extra = 0
    fields = ['image', 'caption', 'order']
    
    # Custom widget for image preview
    readonly_fields = ['image_preview']
    
    def image_preview (self, obj):
        if obj.image:
            return format_html('<img src="{}" style="max-height: 200px;" />', obj.image.url)
        return "No image"
    image_preview.short_description = 'Preview'

@admin.register(Article)
class ArticleAdmin (admin.ModelAdmin):
    inlines = [ImageInline, TagInline, CommentInline]
\`\`\`

**5. Custom Display Methods:**

\`\`\`python
@admin.register(Article)
class ArticleAdmin (admin.ModelAdmin):
    list_display = ['title', 'author_link', 'status_badge', 'preview_button', 'view_stats']
    
    @admin.display (description='Author', ordering='author__username')
    def author_link (self, obj):
        url = reverse('admin:auth_user_change', args=[obj.author.id])
        return format_html('<a href="{}">{}</a>', url, obj.author.username)
    
    @admin.display (description='Status')
    def status_badge (self, obj):
        colors = {
            'published': 'green',
            'draft': 'orange',
            'archived': 'gray',
        }
        color = colors.get (obj.status, 'gray')
        return format_html(
            '<span style="background: {}; color: white; padding: 3px 10px; border-radius: 3px;">{}</span>',
            color, obj.get_status_display()
        )
    
    @admin.display (description='Preview')
    def preview_button (self, obj):
        if obj.status == 'published':
            url = obj.get_absolute_url()
            return format_html('<a href="{}" target="_blank">View</a>', url)
        return '-'
    
    @admin.display (description='Views')
    def view_stats (self, obj):
        return format_html(
            '<strong>{}</strong> views<br><small>{} today</small>',
            obj.view_count,
            obj.get_today_views()  # Custom model method
        )
\`\`\`

**6. Custom Admin Views:**

\`\`\`python
from django.urls import path
from django.shortcuts import render
from django.db.models import Count, Sum
from django.utils import timezone

class ArticleAdmin (admin.ModelAdmin):
    
    def get_urls (self):
        urls = super().get_urls()
        custom_urls = [
            path('analytics/', self.admin_site.admin_view (self.analytics_view), name='article_analytics'),
            path('bulk-import/', self.admin_site.admin_view (self.bulk_import_view), name='article_bulk_import'),
        ]
        return custom_urls + urls
    
    def analytics_view (self, request):
        \"\"\"Custom analytics dashboard\"\"\"
        # Gather statistics
        today = timezone.now().date()
        
        stats = {
            'total_articles': Article.objects.count(),
            'published': Article.objects.filter (status='published').count(),
            'drafts': Article.objects.filter (status='draft').count(),
            'total_views': Article.objects.aggregate(Sum('view_count'))['view_count__sum'] or 0,
            'today_published': Article.objects.filter (published_at__date=today).count(),
        }
        
        # Top authors
        top_authors = Article.objects.values('author__username').annotate(
            article_count=Count('id'),
            total_views=Sum('view_count')
        ).order_by('-article_count')[:10]
        
        context = {
            'title': 'Article Analytics',
            'stats': stats,
            'top_authors': top_authors,
        }
        
        return render (request, 'admin/article_analytics.html', context)
    
    def bulk_import_view (self, request):
        \"\"\"Bulk import articles from CSV\"\"\"
        if request.method == 'POST' and request.FILES.get('csv_file'):
            import csv
            csv_file = request.FILES['csv_file']
            
            decoded_file = csv_file.read().decode('utf-8').splitlines()
            reader = csv.DictReader (decoded_file)
            
            imported = 0
            for row in reader:
                Article.objects.create(
                    title=row['title'],
                    content=row['content'],
                    author_id=row['author_id'],
                    status='draft'
                )
                imported += 1
            
            self.message_user (request, f'Imported {imported} articles.', messages.SUCCESS)
            return redirect('admin:articles_article_changelist')
        
        return render (request, 'admin/article_bulk_import.html', {
            'title': 'Bulk Import Articles'
        })
\`\`\`

**7. Permission-Based Customization:**

\`\`\`python
@admin.register(Article)
class ArticleAdmin (admin.ModelAdmin):
    
    def get_queryset (self, request):
        \"\"\"Filter queryset based on user permissions\"\"\"
        qs = super().get_queryset (request)
        
        if request.user.is_superuser:
            return qs
        
        # Regular staff only see their own articles
        return qs.filter (author=request.user)
    
    def has_delete_permission (self, request, obj=None):
        \"\"\"Only superusers can delete\"\"\"
        if request.user.is_superuser:
            return True
        
        # Authors can delete their own drafts
        if obj and obj.author == request.user and obj.status == 'draft':
            return True
        
        return False
    
    def get_readonly_fields (self, request, obj=None):
        \"\"\"Different read-only fields based on user\"\"\"
        if request.user.is_superuser:
            return ['created_at', 'updated_at']
        
        # Regular staff can't change status
        return ['status', 'featured', 'created_at', 'updated_at']
    
    def get_list_display (self, request):
        \"\"\"Different columns for different users\"\"\"
        if request.user.is_superuser:
            return ['title', 'author', 'status', 'view_count', 'published_at']
        
        return ['title', 'status', 'published_at']
    
    def formfield_for_foreignkey (self, db_field, request, **kwargs):
        \"\"\"Auto-set author to current user for new articles\"\"\"
        if db_field.name == 'author':
            if not request.user.is_superuser:
                kwargs['initial'] = request.user.id
                kwargs['disabled'] = True
        
        return super().formfield_for_foreignkey (db_field, request, **kwargs)
\`\`\`

**8. Custom Dashboard Widget:**

\`\`\`python
# templates/admin/index.html (override)
{% extends "admin/index.html" %}
{% block content %}
<div class="dashboard-stats">
    <div class="stat-card">
        <h3>Articles Today</h3>
        <p class="stat-number">{{ articles_today }}</p>
    </div>
    <div class="stat-card">
        <h3>Total Views</h3>
        <p class="stat-number">{{ total_views }}</p>
    </div>
</div>
{{ block.super }}
{% endblock %}

# admin.py
class MyAdminSite (admin.AdminSite):
    site_header = 'My CMS Administration'
    site_title = 'My CMS Admin'
    index_title = 'Welcome to My CMS'
    
    def index (self, request, extra_context=None):
        extra_context = extra_context or {}
        
        # Add custom dashboard data
        extra_context['articles_today'] = Article.objects.filter(
            published_at__date=timezone.now().date()
        ).count()
        
        extra_context['total_views'] = Article.objects.aggregate(
            Sum('view_count')
        )['view_count__sum'] or 0
        
        return super().index (request, extra_context)

# Use custom admin site
admin_site = MyAdminSite (name='myadmin')
admin_site.register(Article, ArticleAdmin)
\`\`\`

**Best Practices:**
- ✅ Use list_display for important fields
- ✅ Add custom actions for bulk operations
- ✅ Create custom filters for common queries
- ✅ Use inlines for related objects
- ✅ Add custom views for complex operations
- ✅ Implement permission-based customization
- ✅ Optimize queries with select_related/prefetch_related
- ✅ Use readonly_fields for display-only data
- ✅ Add search_fields for searchability
- ❌ Don't expose sensitive data in list_display
- ❌ Don't make admin too complex (consider building custom UI)

Django Admin is extremely customizable and can handle sophisticated requirements with proper configuration.
      `,
  },
  {
    question:
      'Describe how to build a custom admin dashboard with charts, real-time statistics, and interactive widgets. Include integration with Chart.js or similar libraries and WebSocket updates.',
    answer: `
**Custom Admin Dashboard Implementation:**

**1. Django Admin Site Override:**

\`\`\`python
# admin/sites.py
from django.contrib.admin import AdminSite
from django.urls import path
from django.shortcuts import render
from django.db.models import Count, Sum, Avg
from django.utils import timezone
from datetime import timedelta

class DashboardAdminSite(AdminSite):
    site_header = 'Dashboard Administration'
    site_title = 'Dashboard Admin'
    index_title = 'Analytics Dashboard'
    
    def get_urls (self):
        urls = super().get_urls()
        custom_urls = [
            path('dashboard/', self.admin_view (self.dashboard_view), name='dashboard'),
            path('api/stats/', self.admin_view (self.stats_api), name='stats_api'),
        ]
        return custom_urls + urls
    
    def dashboard_view (self, request):
        \"\"\"Render custom dashboard\"\"\"
        context = {
            **self.each_context (request),
            'title': 'Dashboard',
            'stats': self.get_dashboard_stats(),
        }
        return render (request, 'admin/dashboard.html', context)
    
    def stats_api (self, request):
        \"\"\"API endpoint for real-time stats\"\"\"
        from django.http import JsonResponse
        return JsonResponse (self.get_dashboard_stats())
    
    def get_dashboard_stats (self):
        \"\"\"Gather dashboard statistics\"\"\"
        now = timezone.now()
        today = now.date()
        week_ago = now - timedelta (days=7)
        month_ago = now - timedelta (days=30)
        
        # Basic counts
        stats = {
            'total_articles': Article.objects.count(),
            'published_articles': Article.objects.filter (status='published').count(),
            'total_users': User.objects.count(),
            'active_users': User.objects.filter (last_login__gte=week_ago).count(),
            
            # Today\'s activity
            'articles_today': Article.objects.filter (created_at__date=today).count(),
            'comments_today': Comment.objects.filter (created_at__date=today).count(),
            
            # View statistics
            'total_views': Article.objects.aggregate(Sum('view_count'))['view_count__sum'] or 0,
            'avg_views': Article.objects.aggregate(Avg('view_count'))['view_count__avg'] or 0,
        }
        
        # Articles per day (last 30 days)
        articles_per_day = []
        for i in range(30):
            date = today - timedelta (days=i)
            count = Article.objects.filter (created_at__date=date).count()
            articles_per_day.append({
                'date': date.isoformat(),
                'count': count
            })
        stats['articles_per_day'] = list (reversed (articles_per_day))
        
        # Top authors
        stats['top_authors'] = list(
            Article.objects.values('author__username').annotate(
                article_count=Count('id'),
                total_views=Sum('view_count')
            ).order_by('-article_count')[:5]
        )
        
        # Category distribution
        stats['category_distribution'] = list(
            Article.objects.values('category__name').annotate(
                count=Count('id')
            ).order_by('-count')
        )
        
        return stats

# Use custom admin site
dashboard_admin = DashboardAdminSite (name='dashboard_admin')
\`\`\`

**2. Dashboard Template with Chart.js:**

\`\`\`html
<!-- templates/admin/dashboard.html -->
{% extends "admin/base_site.html" %}
{% load static %}

{% block extrastyle %}
<link rel="stylesheet" href="{% static 'admin/css/dashboard.css' %}">
{% endblock %}

{% block content %}
<div class="dashboard-container">
    <!-- Stats Cards -->
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-icon">📝</div>
            <div class="stat-content">
                <h3>Total Articles</h3>
                <p class="stat-number" id="total-articles">{{ stats.total_articles }}</p>
                <span class="stat-change">+{{ stats.articles_today }} today</span>
            </div>
        </div>
        
        <div class="stat-card">
            <div class="stat-icon">👥</div>
            <div class="stat-content">
                <h3>Active Users</h3>
                <p class="stat-number" id="active-users">{{ stats.active_users }}</p>
                <span class="stat-change">{{ stats.total_users }} total</span>
            </div>
        </div>
        
        <div class="stat-card">
            <div class="stat-icon">👁</div>
            <div class="stat-content">
                <h3>Total Views</h3>
                <p class="stat-number" id="total-views">{{ stats.total_views|floatformat:0 }}</p>
                <span class="stat-change">{{ stats.avg_views|floatformat:0 }} avg</span>
            </div>
        </div>
        
        <div class="stat-card">
            <div class="stat-icon">💬</div>
            <div class="stat-content">
                <h3>Comments Today</h3>
                <p class="stat-number" id="comments-today">{{ stats.comments_today }}</p>
            </div>
        </div>
    </div>
    
    <!-- Charts Row -->
    <div class="charts-grid">
        <!-- Articles Timeline Chart -->
        <div class="chart-card">
            <h3>Articles Published (Last 30 Days)</h3>
            <canvas id="articles-chart"></canvas>
        </div>
        
        <!-- Category Distribution Chart -->
        <div class="chart-card">
            <h3>Articles by Category</h3>
            <canvas id="category-chart"></canvas>
        </div>
    </div>
    
    <!-- Top Authors Table -->
    <div class="table-card">
        <h3>Top Authors</h3>
        <table class="dashboard-table">
            <thead>
                <tr>
                    <th>Author</th>
                    <th>Articles</th>
                    <th>Total Views</th>
                </tr>
            </thead>
            <tbody id="top-authors">
                {% for author in stats.top_authors %}
                <tr>
                    <td>{{ author.author__username }}</td>
                    <td>{{ author.article_count }}</td>
                    <td>{{ author.total_views }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script src="{% static 'admin/js/dashboard.js' %}"></script>
<script>
    // Initialize with server data
    const dashboardData = {{ stats|safe }};
    initializeDashboard (dashboardData);
</script>
{% endblock %}
\`\`\`

**3. Dashboard JavaScript:**

\`\`\`javascript
// static/admin/js/dashboard.js

let articlesChart = null;
let categoryChart = null;

function initializeDashboard (data) {
    // Create articles timeline chart
    const articlesCtx = document.getElementById('articles-chart').getContext('2d');
    articlesChart = new Chart (articlesCtx, {
        type: 'line',
        data: {
            labels: data.articles_per_day.map (d => d.date),
            datasets: [{
                label: 'Articles Published',
                data: data.articles_per_day.map (d => d.count),
                borderColor: '#4CAF50',
                backgroundColor: 'rgba(76, 175, 80, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        precision: 0
                    }
                }
            }
        }
    });
    
    // Create category distribution chart
    const categoryCtx = document.getElementById('category-chart').getContext('2d');
    categoryChart = new Chart (categoryCtx, {
        type: 'doughnut',
        data: {
            labels: data.category_distribution.map (c => c.category__name),
            datasets: [{
                data: data.category_distribution.map (c => c.count),
                backgroundColor: [
                    '#FF6384',
                    '#36A2EB',
                    '#FFCE56',
                    '#4BC0C0',
                    '#9966FF',
                    '#FF9F40'
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right'
                }
            }
        }
    });
    
    // Start real-time updates
    startRealTimeUpdates();
}

function startRealTimeUpdates() {
    // Poll for updates every 30 seconds
    setInterval (updateDashboard, 30000);
    
    // Or use WebSocket for real-time updates
    if (window.WebSocket) {
        connectWebSocket();
    }
}

function updateDashboard() {
    fetch('/admin/api/stats/')
        .then (response => response.json())
        .then (data => {
            // Update stat cards
            document.getElementById('total-articles').textContent = data.total_articles;
            document.getElementById('active-users').textContent = data.active_users;
            document.getElementById('total-views').textContent = Math.round (data.total_views);
            document.getElementById('comments-today').textContent = data.comments_today;
            
            // Update charts
            updateCharts (data);
        })
        .catch (error => console.error('Error updating dashboard:', error));
}

function updateCharts (data) {
    // Update articles chart
    articlesChart.data.labels = data.articles_per_day.map (d => d.date);
    articlesChart.data.datasets[0].data = data.articles_per_day.map (d => d.count);
    articlesChart.update('none');  // No animation for updates
    
    // Update category chart
    categoryChart.data.labels = data.category_distribution.map (c => c.category__name);
    categoryChart.data.datasets[0].data = data.category_distribution.map (c => c.count);
    categoryChart.update('none');
}

function connectWebSocket() {
    const ws = new WebSocket('ws://localhost:8000/ws/dashboard/');
    
    ws.onmessage = function (event) {
        const data = JSON.parse (event.data);
        
        // Update specific stat
        if (data.type === 'stat_update') {
            document.getElementById (data.stat_id).textContent = data.value;
        }
        
        // Full dashboard update
        if (data.type === 'full_update') {
            updateDashboard();
        }
    };
    
    ws.onclose = function() {
        // Reconnect after 5 seconds
        setTimeout (connectWebSocket, 5000);
    };
}
\`\`\`

**4. WebSocket Consumer for Real-Time Updates:**

\`\`\`python
# consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async

class DashboardConsumer(AsyncWebsocketConsumer):
    async def connect (self):
        await self.channel_layer.group_add('dashboard', self.channel_name)
        await self.accept()
    
    async def disconnect (self, close_code):
        await self.channel_layer.group_discard('dashboard', self.channel_name)
    
    async def dashboard_update (self, event):
        # Send update to WebSocket
        await self.send (text_data=json.dumps({
            'type': 'stat_update',
            'stat_id': event['stat_id'],
            'value': event['value']
        }))

# signals.py - Trigger WebSocket updates
from django.db.models.signals import post_save
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

@receiver (post_save, sender=Article)
def notify_dashboard (sender, instance, created, **kwargs):
    if created:
        channel_layer = get_channel_layer()
        async_to_sync (channel_layer.group_send)(
            'dashboard',
            {
                'type': 'dashboard_update',
                'stat_id': 'total-articles',
                'value': Article.objects.count()
            }
        )
\`\`\`

**5. Dashboard CSS:**

\`\`\`css
/* static/admin/css/dashboard.css */
.dashboard-container {
    padding: 20px;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat (auto-fit, minmax(250px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}

.stat-card {
    background: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    display: flex;
    align-items: center;
}

.stat-icon {
    font-size: 48px;
    margin-right: 20px;
}

.stat-content h3 {
    margin: 0;
    color: #666;
    font-size: 14px;
    font-weight: normal;
}

.stat-number {
    font-size: 32px;
    font-weight: bold;
    margin: 5px 0;
    color: #333;
}

.stat-change {
    font-size: 12px;
    color: #4CAF50;
}

.charts-grid {
    display: grid;
    grid-template-columns: repeat (auto-fit, minmax(400px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}

.chart-card {
    background: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    height: 400px;
}

.table-card {
    background: white;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.dashboard-table {
    width: 100%;
    border-collapse: collapse;
}

.dashboard-table th,
.dashboard-table td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid #ddd;
}

.dashboard-table thead th {
    background: #f5f5f5;
    font-weight: 600;
}
\`\`\`

**Features:**
- ✅ Real-time statistics with WebSocket
- ✅ Interactive charts (Chart.js)
- ✅ Responsive grid layout
- ✅ REST API for data
- ✅ Periodic polling fallback
- ✅ Clean, modern UI
- ✅ Permission-based access

This creates a professional, real-time admin dashboard with rich data visualization.
      `,
  },
  {
    question:
      'Explain how to implement advanced admin features like bulk editing with formsets, custom change views with multiple tabs, and admin action confirmations with custom forms.',
    answer: `
**Advanced Django Admin Features:**

**1. Bulk Editing with Formsets:**

\`\`\`python
from django.forms import modelformset_factory
from django.shortcuts import render, redirect
from django.contrib import messages

@admin.register(Article)
class ArticleAdmin (admin.ModelAdmin):
    actions = ['bulk_edit_selected']
    
    @admin.action (description='Bulk edit selected articles')
    def bulk_edit_selected (self, request, queryset):
        # Create formset for selected articles
        ArticleFormSet = modelformset_factory(
            Article,
            fields=['title', 'status', 'category', 'featured'],
            extra=0
        )
        
        if request.method == 'POST':
            formset = ArticleFormSet(
                request.POST,
                queryset=queryset
            )
            
            if formset.is_valid():
                instances = formset.save()
                self.message_user(
                    request,
                    f'Successfully updated {len (instances)} articles.',
                    messages.SUCCESS
                )
                return redirect('admin:articles_article_changelist')
        else:
            formset = ArticleFormSet (queryset=queryset)
        
        context = {
            'formset': formset,
            'articles': queryset,
            'opts': self.model._meta,
            'title': 'Bulk Edit Articles',
        }
        
        return render (request, 'admin/bulk_edit.html', context)
\`\`\`

**Bulk Edit Template:**

\`\`\`html
<!-- templates/admin/bulk_edit.html -->
{% extends "admin/base_site.html" %}

{% block content %}
<h1>Bulk Edit {{ articles.count }} Articles</h1>

<form method="post">
    {% csrf_token %}
    {{ formset.management_form }}
    
    <table class="bulk-edit-table">
        <thead>
            <tr>
                <th>Title</th>
                <th>Status</th>
                <th>Category</th>
                <th>Featured</th>
            </tr>
        </thead>
        <tbody>
            {% for form in formset %}
            <tr>
                {{ form.id }}
                <td>{{ form.title }}</td>
                <td>{{ form.status }}</td>
                <td>{{ form.category }}</td>
                <td>{{ form.featured }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    
    <div class="submit-row">
        <input type="submit" value="Save Changes" class="default" />
        <a href="{% url 'admin:articles_article_changelist' %}" class="button cancel-link">Cancel</a>
    </div>
</form>
{% endblock %}
\`\`\`

**2. Multi-Tab Change View:**

\`\`\`python
@admin.register(Article)
class ArticleAdmin (admin.ModelAdmin):
    
    change_form_template = 'admin/article_change_form.html'
    
    def get_fieldsets (self, request, obj=None):
        \"\"\"Organize fields into tabs\"\"\"
        return [
            ('Basic Info', {
                'fields': ('title', 'slug', 'author', 'category'),
                'classes': ['tab-content', 'tab-basic'],
            }),
            ('Content', {
                'fields': ('content', 'excerpt'),
                'classes': ['tab-content', 'tab-content-section'],
            }),
            ('SEO', {
                'fields': ('meta_title', 'meta_description', 'meta_keywords'),
                'classes': ['tab-content', 'tab-seo'],
            }),
            ('Publishing', {
                'fields': ('status', 'featured', 'published_at'),
                'classes': ['tab-content', 'tab-publishing'],
            }),
            ('Media', {
                'fields': ('featured_image', 'gallery_images'),
                'classes': ['tab-content', 'tab-media'],
            }),
            ('Advanced', {
                'fields': ('allow_comments', 'template_override', 'custom_css'),
                'classes': ['tab-content', 'tab-advanced', 'collapse'],
            }),
        ]
    
    class Media:
        css = {
            'all': ('admin/css/tabs.css',)
        }
        js = ('admin/js/tabs.js',)
\`\`\`

**Tabbed Form Template:**

\`\`\`html
<!-- templates/admin/article_change_form.html -->
{% extends "admin/change_form.html" %}

{% block form_top %}
<div class="tabs-container">
    <ul class="tabs">
        <li class="tab active" data-tab="basic">
            <a href="#basic">Basic Info</a>
        </li>
        <li class="tab" data-tab="content">
            <a href="#content">Content</a>
        </li>
        <li class="tab" data-tab="seo">
            <a href="#seo">SEO</a>
        </li>
        <li class="tab" data-tab="publishing">
            <a href="#publishing">Publishing</a>
        </li>
        <li class="tab" data-tab="media">
            <a href="#media">Media</a>
        </li>
        <li class="tab" data-tab="advanced">
            <a href="#advanced">Advanced</a>
        </li>
    </ul>
</div>
{% endblock %}
\`\`\`

**Tab JavaScript:**

\`\`\`javascript
// static/admin/js/tabs.js
(function($) {
    $(document).ready (function() {
        // Hide all tabs except first
        $('.tab-content:not(.tab-basic)').hide();
        
        // Tab click handler
        $('.tabs .tab').click (function (e) {
            e.preventDefault();
            
            // Update active tab
            $('.tabs .tab').removeClass('active');
            $(this).addClass('active');
            
            // Show corresponding content
            const tabName = $(this).data('tab');
            $('.tab-content').hide();
            $(\`.tab - \${ tabName }\`).show();
        });
    });
})(django.jQuery);
\`\`\`

**3. Admin Action with Confirmation Form:**

\`\`\`python
from django import forms

class BulkStatusChangeForm (forms.Form):
    \"\"\"Form for bulk status change action\"\"\"
    status = forms.ChoiceField(
        choices=Article.STATUS_CHOICES,
        required=True,
        help_text='Select new status for selected articles'
    )
    send_notification = forms.BooleanField(
        required=False,
        initial=True,
        help_text='Send notification to authors'
    )
    notification_message = forms.CharField(
        widget=forms.Textarea,
        required=False,
        help_text='Optional message to include in notification'
    )

@admin.register(Article)
class ArticleAdmin (admin.ModelAdmin):
    actions = ['change_status_with_confirmation']
    
    @admin.action (description='Change status with confirmation')
    def change_status_with_confirmation (self, request, queryset):
        form = None
        
        if 'apply' in request.POST:
            # Form was submitted
            form = BulkStatusChangeForm (request.POST)
            
            if form.is_valid():
                status = form.cleaned_data['status']
                send_notification = form.cleaned_data['send_notification']
                message = form.cleaned_data['notification_message']
                
                # Update articles
                count = queryset.update (status=status)
                
                # Send notifications if requested
                if send_notification:
                    for article in queryset:
                        send_status_change_notification(
                            article,
                            status,
                            message
                        )
                
                self.message_user(
                    request,
                    f'Successfully changed status of {count} articles to {status}.',
                    messages.SUCCESS
                )
                return
        
        if not form:
            form = BulkStatusChangeForm()
        
        context = {
            'form': form,
            'queryset': queryset,
            'action': 'change_status_with_confirmation',
            'action_checkbox_name': admin.helpers.ACTION_CHECKBOX_NAME,
            'opts': self.model._meta,
            'title': 'Change Status with Confirmation',
        }
        
        return render (request, 'admin/bulk_status_change.html', context)
\`\`\`

**Action Confirmation Template:**

\`\`\`html
<!-- templates/admin/bulk_status_change.html -->
{% extends "admin/base_site.html" %}

{% block content %}
<h1>Change Status: {{ queryset.count }} Articles</h1>

<form method="post">
    {% csrf_token %}
    
    <p>You are about to change the status of the following articles:</p>
    
    <ul class="article-list">
        {% for article in queryset %}
        <li>{{ article.title }} (currently: {{ article.get_status_display }})</li>
        {% endfor %}
    </ul>
    
    {{ form.as_p }}
    
    <!-- Hidden inputs for action -->
    <input type="hidden" name="action" value="change_status_with_confirmation" />
    {% for obj in queryset %}
    <input type="hidden" name="{{ action_checkbox_name }}" value="{{ obj.pk }}" />
    {% endfor %}
    
    <div class="submit-row">
        <input type="submit" name="apply" value="Apply Changes" class="default" />
        <a href="{% url 'admin:articles_article_changelist' %}" class="button cancel-link">Cancel</a>
    </div>
</form>
{% endblock %}
\`\`\`

**4. Custom Widgets in Admin:**

\`\`\`python
from django import forms
from django.contrib.admin import widgets

class RichTextWidget (forms.Textarea):
    \"\"\"CKEditor widget for rich text editing\"\"\"
    class Media:
        js = (
            'https://cdn.ckeditor.com/4.16.2/standard/ckeditor.js',
            'admin/js/ckeditor-init.js',
        )

class ArticleAdminForm (forms.ModelForm):
    \"\"\"Custom form with rich widgets\"\"\"
    
    content = forms.CharField(
        widget=RichTextWidget (attrs={'rows': 20})
    )
    
    tags = forms.ModelMultipleChoiceField(
        queryset=Tag.objects.all(),
        widget=admin.widgets.FilteredSelectMultiple('Tags', False),
        required=False
    )
    
    class Meta:
        model = Article
        fields = '__all__'

@admin.register(Article)
class ArticleAdmin (admin.ModelAdmin):
    form = ArticleAdminForm
\`\`\`

**5. Nested Inlines (Admin Inline within Inline):**

\`\`\`python
from nested_admin import NestedModelAdmin, NestedTabularInline

class ImageVariantInline(NestedTabularInline):
    model = ImageVariant
    extra = 0
    fields = ['size', 'width', 'height', 'file']

class ArticleImageInline(NestedTabularInline):
    model = ArticleImage
    extra = 1
    fields = ['image', 'caption', 'order']
    inlines = [ImageVariantInline]

@admin.register(Article)
class ArticleAdmin(NestedModelAdmin):
    inlines = [ArticleImageInline]
\`\`\`

**Production Tips:**
- ✅ Use formsets for bulk operations
- ✅ Implement tabs for complex forms
- ✅ Add confirmation steps for dangerous actions
- ✅ Use custom widgets for better UX
- ✅ Validate forms thoroughly
- ✅ Provide clear feedback messages
- ✅ Handle errors gracefully
- ✅ Test admin actions extensively

These advanced features create a powerful, user-friendly admin interface for content management.
      `,
  },
].map(({ id, ...q }, idx) => ({
  id: `django-q-${idx + 1}`,
  question: q.question,
  sampleAnswer: String(q.answer),
  keyPoints: [],
}));
