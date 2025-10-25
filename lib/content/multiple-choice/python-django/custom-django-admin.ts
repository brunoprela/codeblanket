import { MultipleChoiceQuestion } from '@/lib/types';

export const CustomDjangoAdminMultipleChoice = [
  {
    id: 1,
    question: 'What is the purpose of list_display in Django ModelAdmin?',
    options: [
      'A) To control which fields appear in the change form',
      'B) To specify which columns appear in the list view',
      'C) To define the ordering of objects',
      'D) To filter the queryset displayed',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) To specify which columns appear in the list view**

\`list_display\` controls which fields/methods are shown as columns in the admin list view.

\`\`\`python
class ArticleAdmin(admin.ModelAdmin):
    list_display = ['title', 'author', 'status', 'published_at']
\`\`\`

You can also include custom methods and properties for computed columns.
      `,
  },
  {
    question:
      'How do you make a custom admin action available only to superusers?',
    options: [
      'A) Add @admin.superuser_only decorator',
      'B) Check request.user.is_superuser in the action method',
      'C) Override get_actions() to filter based on permissions',
      'D) Set action.superuser_required = True',
    ],
    correctAnswer: 2,
    explanation: `
**Correct Answer: C) Override get_actions() to filter based on permissions**

\`\`\`python
class ArticleAdmin(admin.ModelAdmin):
    actions = ['delete_selected', 'publish_articles']
    
    def get_actions(self, request):
        actions = super().get_actions(request)
        if not request.user.is_superuser:
            actions.pop('delete_selected', None)
        return actions
\`\`\`

This gives you full control over which actions are available to which users.
      `,
  },
  {
    question: 'What does prepopulated_fields do in Django admin?',
    options: [
      'A) Sets default values for new objects',
      'B) Auto-generates slug from other fields using JavaScript',
      'C) Pre-loads related objects to avoid N+1 queries',
      'D) Fills in fields based on user permissions',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) Auto-generates slug from other fields using JavaScript**

\`\`\`python
class ArticleAdmin(admin.ModelAdmin):
    prepopulated_fields = {'slug': ('title',)}
\`\`\`

As you type in the title field, the slug field is automatically populated (using JavaScript in the browser). This only works for SlugField.
      `,
  },
  {
    question: 'How can you add a custom view to Django admin?',
    options: [
      'A) Add methods to ModelAdmin with @admin.view decorator',
      'B) Override get_urls() to add custom URL patterns',
      'C) Create a custom AdminSite subclass',
      'D) Use admin.site.register_view()',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) Override get_urls() to add custom URL patterns**

\`\`\`python
class ArticleAdmin(admin.ModelAdmin):
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('stats/', self.admin_site.admin_view(self.stats_view)),
        ]
        return custom_urls + urls
    
    def stats_view(self, request):
        # Your custom view logic
        return render(request, 'admin/stats.html', context)
\`\`\`

This allows you to add completely custom views within the admin interface.
      `,
  },
  {
    question:
      'What is the correct way to optimize admin list queries with related objects?',
    options: [
      'A) Use list_display with ForeignKey fields',
      'B) Override get_queryset() with select_related()',
      'C) Enable admin.site.OPTIMIZE_QUERIES',
      'D) Add prefetch=True to list_display',
    ],
    correctAnswer: 1,
    explanation: `
**Correct Answer: B) Override get_queryset() with select_related()**

\`\`\`python
class ArticleAdmin(admin.ModelAdmin):
    list_display = ['title', 'author', 'category']
    
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.select_related('author', 'category')
\`\`\`

This prevents N+1 queries when displaying related fields in the list view, dramatically improving performance.
      `,
  },
].map(({ id, ...q }, idx) => ({ id: `django-mc-${idx + 1}`, ...q }));
