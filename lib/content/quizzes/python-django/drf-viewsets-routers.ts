export const drfViewsetsRoutersQuiz = {
  title: 'DRF ViewSets & Routers - Discussion Questions',
  questions: [
    {
      question:
        'Explain the different types of ViewSets in DRF (ViewSet, GenericViewSet, ModelViewSet, ReadOnlyModelViewSet) and when to use each. Include examples of custom actions and how routers generate URLs.',
      answer: `
**ViewSet Types:**

**1. ViewSet (Base):**
No default actions - define your own methods.

\`\`\`python
class CustomViewSet(viewsets.ViewSet):
    def list(self, request):
        return Response([...])
    
    def create(self, request):
        return Response({...})
\`\`\`

**2. GenericViewSet:**
Includes mixin functionality but no actions by default.

\`\`\`python
from rest_framework import mixins

class ArticleViewSet(mixins.ListModelMixin, 
                      mixins.CreateModelMixin,
                      viewsets.GenericViewSet):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    # Only list and create, no retrieve/update/delete
\`\`\`

**3. ModelViewSet:**
Full CRUD - list, create, retrieve, update, partial_update, destroy.

\`\`\`python
class ArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    # All operations included
\`\`\`

**4. ReadOnlyModelViewSet:**
Only list and retrieve - no modifications.

\`\`\`python
class PublicArticleViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Article.objects.filter(status='published')
    serializer_class = ArticleSerializer
\`\`\`

**Custom Actions:**

\`\`\`python
class ArticleViewSet(viewsets.ModelViewSet):
    @action(detail=True, methods=['post'])
    def publish(self, request, pk=None):
        article = self.get_object()
        article.status = 'published'
        article.save()
        return Response({'status': 'published'})
    
    @action(detail=False, methods=['get'])
    def featured(self, request):
        articles = self.get_queryset().filter(featured=True)
        serializer = self.get_serializer(articles, many=True)
        return Response(serializer.data)
\`\`\`

**Router URL Generation:**

\`\`\`python
router = DefaultRouter()
router.register(r'articles', ArticleViewSet)

# Generates:
# GET    /articles/              -> list
# POST   /articles/              -> create
# GET    /articles/{id}/         -> retrieve
# PUT    /articles/{id}/         -> update
# PATCH  /articles/{id}/         -> partial_update
# DELETE /articles/{id}/         -> destroy
# POST   /articles/{id}/publish/ -> publish (custom)
# GET    /articles/featured/     -> featured (custom)
\`\`\`

**Use Cases:**
- ModelViewSet: Standard CRUD resources
- ReadOnlyModelViewSet: Public read-only APIs
- Custom mixins: Partial CRUD (e.g., list + create only)
- ViewSet: Fully custom non-CRUD endpoints
      `,
    },
    {
      question:
        'Describe how to implement action-specific serializers, permissions, and pagination in DRF ViewSets. Provide examples of customizing behavior per action.',
      answer: `
**Action-Specific Customization:**

**1. Different Serializers Per Action:**

\`\`\`python
class ArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.all()
    
    def get_serializer_class(self):
        if self.action == 'list':
            return ArticleListSerializer  # Minimal fields
        elif self.action in ['create', 'update']:
            return ArticleWriteSerializer  # Write fields
        return ArticleDetailSerializer  # Full details
\`\`\`

**2. Different Permissions Per Action:**

\`\`\`python
class ArticleViewSet(viewsets.ModelViewSet):
    def get_permissions(self):
        if self.action in ['list', 'retrieve']:
            return []  # Public read
        elif self.action == 'destroy':
            return [IsAdminUser()]  # Admin only delete
        return [IsAuthenticated()]  # Auth for create/update
\`\`\`

**3. Different QuerySets Per Action:**

\`\`\`python
class ArticleViewSet(viewsets.ModelViewSet):
    def get_queryset(self):
        if self.action == 'list':
            return Article.objects.filter(status='published')
        return Article.objects.all()
\`\`\`

**4. Action-Specific Pagination:**

\`\`\`python
class ArticleViewSet(viewsets.ModelViewSet):
    def get_paginator(self):
        if self.action == 'list':
            return LargeResultsSetPagination()
        return StandardResultsSetPagination()
\`\`\`

**Complete Example:**

\`\`\`python
class ArticleViewSet(viewsets.ModelViewSet):
    def get_queryset(self):
        qs = Article.objects.all()
        
        if self.action == 'list':
            # Optimize list view
            qs = qs.select_related('author').only('id', 'title', 'author')
        elif self.action == 'retrieve':
            # Full optimization for detail
            qs = qs.select_related('author').prefetch_related('tags')
        
        return qs
    
    def get_serializer_class(self):
        if self.action == 'list':
            return ArticleListSerializer
        elif self.action in ['create', 'update', 'partial_update']:
            return ArticleWriteSerializer
        return ArticleDetailSerializer
    
    def get_permissions(self):
        if self.action in ['list', 'retrieve']:
            return [AllowAny()]
        elif self.action == 'destroy':
            return [IsAdminUser()]
        return [IsAuthenticated(), IsOwnerOrReadOnly()]
    
    @action(detail=True, methods=['post'], 
            permission_classes=[IsAuthenticated])
    def like(self, request, pk=None):
        article = self.get_object()
        article.likes.add(request.user)
        return Response({'status': 'liked'})
\`\`\`

This pattern provides maximum flexibility while keeping code organized.
      `,
    },
    {
      question:
        'Explain DRF routers (SimpleRouter vs DefaultRouter) and how to customize URL patterns. Include nested routing and custom route configurations.',
      answer: `
**Router Comparison:**

**SimpleRouter:**
Basic URL patterns without API root view.

**DefaultRouter:**
Includes browsable API root and .json suffix support.

\`\`\`python
from rest_framework.routers import DefaultRouter, SimpleRouter

# SimpleRouter generates:
# /articles/
# /articles/{id}/

# DefaultRouter generates:
# / (API root)
# /articles/
# /articles/{id}/
# /articles.json (format suffix)
\`\`\`

**Custom URL Patterns:**

\`\`\`python
router = DefaultRouter()
router.register(r'articles', ArticleViewSet, basename='article')

# Customize trailing slash
router = DefaultRouter(trailing_slash=False)

# Custom actions generate URLs automatically
class ArticleViewSet(viewsets.ModelViewSet):
    @action(detail=True, methods=['post'], url_path='publish-now')
    def publish(self, request, pk=None):
        # URL: /articles/{id}/publish-now/
        pass
\`\`\`

**Nested Routers:**

\`\`\`python
from rest_framework_nested import routers

router = routers.DefaultRouter()
router.register(r'articles', ArticleViewSet)

# Nest comments under articles
articles_router = routers.NestedDefaultRouter(
    router, r'articles', lookup='article'
)
articles_router.register(
    r'comments', CommentViewSet, basename='article-comments'
)

# Generates:
# /articles/{article_id}/comments/
# /articles/{article_id}/comments/{id}/
\`\`\`

**Manual URL Configuration:**

\`\`\`python
urlpatterns = [
    path('custom/', CustomViewSet.as_view({
        'get': 'list',
        'post': 'create'
    })),
]
\`\`\`

**Best Practices:**
- Use DefaultRouter for full-featured APIs
- SimpleRouter for minimal overhead
- Nested routers for hierarchical resources
- Custom URL patterns for non-standard endpoints
      `,
    },
  ],
};
