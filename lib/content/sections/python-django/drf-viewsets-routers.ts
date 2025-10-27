export const drfViewsetsRouters = {
  title: 'DRF ViewSets & Routers',
  id: 'drf-viewsets-routers',
  content: `
# DRF ViewSets & Routers

## Introduction

**ViewSets** in Django REST Framework allow you to combine the logic for a set of related views into a single class. **Routers** automatically determine the URL conf for your ViewSets, drastically reducing the amount of boilerplate code you need to write.

### Why ViewSets and Routers?

- **Less Code**: Combine list, create, retrieve, update, delete in one class
- **Automatic URLs**: Routers generate URL patterns automatically
- **Consistency**: Standardized API structure
- **Flexibility**: Override specific actions as needed
- **DRY Principle**: Reusable view logic

By the end of this section, you'll master:
- ViewSet types and when to use each
- Router configuration
- Custom actions with @action decorator
- Nested routes
- ViewSet methods and hooks
- Production patterns

---

## ViewSet Basics

### From APIView to ViewSet

Compare the evolution:

\`\`\`python
# 1. APIView (most verbose)
class ArticleList(APIView):
    def get (self, request):
        articles = Article.objects.all()
        serializer = ArticleSerializer (articles, many=True)
        return Response (serializer.data)
    
    def post (self, request):
        serializer = ArticleSerializer (data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response (serializer.data, status=201)
        return Response (serializer.errors, status=400)

class ArticleDetail(APIView):
    def get (self, request, pk):
        article = get_object_or_404(Article, pk=pk)
        serializer = ArticleSerializer (article)
        return Response (serializer.data)
    
    def put (self, request, pk):
        article = get_object_or_404(Article, pk=pk)
        serializer = ArticleSerializer (article, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response (serializer.data)
        return Response (serializer.errors, status=400)
    
    def delete (self, request, pk):
        article = get_object_or_404(Article, pk=pk)
        article.delete()
        return Response (status=204)

# 2. Generic Views (better)
class ArticleList (generics.ListCreateAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer

class ArticleDetail (generics.RetrieveUpdateDestroyAPIView):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer

# 3. ViewSet (best - single class!)
class ArticleViewSet (viewsets.ModelViewSet):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    # Handles list, create, retrieve, update, partial_update, destroy
\`\`\`

---

## ModelViewSet

### Complete CRUD ViewSet

\`\`\`python
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticatedOrReadOnly
from .models import Article
from .serializers import ArticleSerializer, ArticleDetailSerializer

class ArticleViewSet (viewsets.ModelViewSet):
    """
    ViewSet for Article model with full CRUD operations
    
    Provides:
    - list: GET /articles/
    - create: POST /articles/
    - retrieve: GET /articles/{id}/
    - update: PUT /articles/{id}/
    - partial_update: PATCH /articles/{id}/
    - destroy: DELETE /articles/{id}/
    """
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    permission_classes = [IsAuthenticatedOrReadOnly]
    
    def get_queryset (self):
        """Optimize queries and filter results"""
        queryset = Article.objects.select_related('author', 'category')
        queryset = queryset.prefetch_related('tags')
        
        # Filter by status query param
        status = self.request.query_params.get('status')
        if status:
            queryset = queryset.filter (status=status)
        
        return queryset
    
    def get_serializer_class (self):
        """Use different serializer for detail vs list"""
        if self.action == 'retrieve':
            return ArticleDetailSerializer
        return ArticleSerializer
    
    def perform_create (self, serializer):
        """Set author to current user on creation"""
        serializer.save (author=self.request.user)
    
    def perform_update (self, serializer):
        """Log updates"""
        serializer.save()
        # Log the update
        logger.info (f"Article {serializer.instance.id} updated by {self.request.user}")
    
    def perform_destroy (self, instance):
        """Soft delete instead of hard delete"""
        instance.status = 'deleted'
        instance.save()
        # Or use: instance.delete() for hard delete
\`\`\`

### URLs with Router

\`\`\`python
# urls.py
from rest_framework.routers import DefaultRouter
from .views import ArticleViewSet

router = DefaultRouter()
router.register (r'articles', ArticleViewSet, basename='article')

urlpatterns = router.urls

# Generated URLs:
# GET    /articles/           -> list
# POST   /articles/           -> create
# GET    /articles/{id}/      -> retrieve
# PUT    /articles/{id}/      -> update
# PATCH  /articles/{id}/      -> partial_update
# DELETE /articles/{id}/      -> destroy
\`\`\`

---

## ReadOnlyModelViewSet

For read-only APIs:

\`\`\`python
from rest_framework import viewsets

class CategoryViewSet (viewsets.ReadOnlyModelViewSet):
    """
    Read-only ViewSet - only list and retrieve
    
    Provides:
    - list: GET /categories/
    - retrieve: GET /categories/{id}/
    """
    queryset = Category.objects.all()
    serializer_class = CategorySerializer
    lookup_field = 'slug'  # Use slug instead of pk

# Generated URLs:
# GET /categories/
# GET /categories/{slug}/
\`\`\`

---

## Custom Actions with @action

### Adding Custom Endpoints

\`\`\`python
from rest_framework.decorators import action
from rest_framework.response import Response
from django.utils import timezone

class ArticleViewSet (viewsets.ModelViewSet):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
    
    @action (detail=True, methods=['post'])
    def publish (self, request, pk=None):
        """
        Publish an article
        POST /articles/{id}/publish/
        """
        article = self.get_object()
        article.status = 'published'
        article.published_at = timezone.now()
        article.save()
        
        serializer = self.get_serializer (article)
        return Response (serializer.data)
    
    @action (detail=True, methods=['post'])
    def unpublish (self, request, pk=None):
        """
        Unpublish an article
        POST /articles/{id}/unpublish/
        """
        article = self.get_object()
        article.status = 'draft'
        article.save()
        
        return Response({'status': 'article unpublished'})
    
    @action (detail=False, methods=['get'])
    def featured (self, request):
        """
        Get featured articles
        GET /articles/featured/
        """
        featured = self.get_queryset().filter (featured=True)
        
        page = self.paginate_queryset (featured)
        if page is not None:
            serializer = self.get_serializer (page, many=True)
            return self.get_paginated_response (serializer.data)
        
        serializer = self.get_serializer (featured, many=True)
        return Response (serializer.data)
    
    @action (detail=False, methods=['get'])
    def popular (self, request):
        """
        Get popular articles
        GET /articles/popular/
        """
        popular = self.get_queryset().filter(
            view_count__gte=1000
        ).order_by('-view_count')[:10]
        
        serializer = self.get_serializer (popular, many=True)
        return Response (serializer.data)
    
    @action (detail=True, methods=['post'], permission_classes=[IsAuthenticated])
    def like (self, request, pk=None):
        """
        Like an article
        POST /articles/{id}/like/
        """
        article = self.get_object()
        
        # Toggle like
        if article.likes.filter (user=request.user).exists():
            article.likes.filter (user=request.user).delete()
            return Response({'status': 'unliked'})
        else:
            Like.objects.create (article=article, user=request.user)
            return Response({'status': 'liked'})
    
    @action (detail=True, methods=['get'])
    def comments (self, request, pk=None):
        """
        Get article comments
        GET /articles/{id}/comments/
        """
        article = self.get_object()
        comments = article.comments.all()
        
        serializer = CommentSerializer (comments, many=True)
        return Response (serializer.data)
    
    @action (detail=False, methods=['get'], url_path='by-category/(? P<category_slug>[^/.]+)')
    def by_category (self, request, category_slug=None):
        """
        Get articles by category
        GET /articles/by-category/{slug}/
        """
        articles = self.get_queryset().filter (category__slug=category_slug)
        
        page = self.paginate_queryset (articles)
        if page is not None:
            serializer = self.get_serializer (page, many=True)
            return self.get_paginated_response (serializer.data)
        
        serializer = self.get_serializer (articles, many=True)
        return Response (serializer.data)
\`\`\`

### Action Parameters

\`\`\`python
@action(
    detail=True,              # detail=True: /articles/{id}/action/
                              # detail=False: /articles/action/
    methods=['get', 'post'],  # HTTP methods
    url_path='custom-path',   # Custom URL path
    url_name='custom-name',   # URL name for reverse()
    permission_classes=[IsAuthenticated],  # Custom permissions
    serializer_class=CustomSerializer,     # Custom serializer
)
def custom_action (self, request, pk=None):
    pass
\`\`\`

---

## Routers

### DefaultRouter

\`\`\`python
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register (r'articles', ArticleViewSet)
router.register (r'categories', CategoryViewSet)
router.register (r'tags', TagViewSet)

# Include in main urls
urlpatterns = [
    path('api/', include (router.urls)),
]

# Generated URLs:
# /api/articles/
# /api/articles/{pk}/
# /api/articles/{pk}/publish/  (custom action)
# /api/categories/
# /api/tags/
# ...plus API root at /api/
\`\`\`

### SimpleRouter

\`\`\`python
from rest_framework.routers import SimpleRouter

router = SimpleRouter()
router.register (r'articles', ArticleViewSet)

# Like DefaultRouter but without API root view
\`\`\`

### Custom Router

\`\`\`python
from rest_framework.routers import DefaultRouter

class CustomRouter(DefaultRouter):
    """
    Custom router with additional routes
    """
    
    def get_default_basename (self, viewset):
        """
        Custom basename logic
        """
        return viewset.queryset.model._meta.object_name.lower()

router = CustomRouter()
\`\`\`

---

## ViewSet Methods and Hooks

### Available Methods

\`\`\`python
class ArticleViewSet (viewsets.ModelViewSet):
    
    def list (self, request):
        """GET /articles/ - Override list"""
        queryset = self.get_queryset()
        serializer = self.get_serializer (queryset, many=True)
        return Response (serializer.data)
    
    def create (self, request):
        """POST /articles/ - Override create"""
        serializer = self.get_serializer (data=request.data)
        serializer.is_valid (raise_exception=True)
        self.perform_create (serializer)
        headers = self.get_success_headers (serializer.data)
        return Response (serializer.data, status=201, headers=headers)
    
    def retrieve (self, request, pk=None):
        """GET /articles/{pk}/ - Override retrieve"""
        instance = self.get_object()
        
        # Increment view count
        instance.view_count += 1
        instance.save (update_fields=['view_count'])
        
        serializer = self.get_serializer (instance)
        return Response (serializer.data)
    
    def update (self, request, pk=None):
        """PUT /articles/{pk}/ - Override update"""
        instance = self.get_object()
        serializer = self.get_serializer (instance, data=request.data)
        serializer.is_valid (raise_exception=True)
        self.perform_update (serializer)
        return Response (serializer.data)
    
    def partial_update (self, request, pk=None):
        """PATCH /articles/{pk}/ - Override partial update"""
        instance = self.get_object()
        serializer = self.get_serializer (instance, data=request.data, partial=True)
        serializer.is_valid (raise_exception=True)
        self.perform_update (serializer)
        return Response (serializer.data)
    
    def destroy (self, request, pk=None):
        """DELETE /articles/{pk}/ - Override destroy"""
        instance = self.get_object()
        self.perform_destroy (instance)
        return Response (status=204)
\`\`\`

### Hook Methods (Preferred)

\`\`\`python
class ArticleViewSet (viewsets.ModelViewSet):
    
    def get_queryset (self):
        """Customize queryset - called for every request"""
        queryset = Article.objects.all()
        
        # Filter by user
        if not self.request.user.is_staff:
            queryset = queryset.filter (author=self.request.user)
        
        # Apply query params
        status = self.request.query_params.get('status')
        if status:
            queryset = queryset.filter (status=status)
        
        return queryset
    
    def get_serializer_class (self):
        """Different serializers for different actions"""
        if self.action == 'list':
            return ArticleListSerializer
        elif self.action == 'retrieve':
            return ArticleDetailSerializer
        elif self.action in ['create', 'update']:
            return ArticleWriteSerializer
        return ArticleSerializer
    
    def get_permissions (self):
        """Different permissions for different actions"""
        if self.action in ['create', 'update', 'partial_update', 'destroy']:
            permission_classes = [IsAuthenticated]
        else:
            permission_classes = [AllowAny]
        return [permission() for permission in permission_classes]
    
    def perform_create (self, serializer):
        """Called during create() - customize object creation"""
        serializer.save(
            author=self.request.user,
            created_ip=self.request.META.get('REMOTE_ADDR')
        )
    
    def perform_update (self, serializer):
        """Called during update() - customize object update"""
        serializer.save (updated_by=self.request.user)
    
    def perform_destroy (self, instance):
        """Called during destroy() - customize deletion"""
        # Soft delete
        instance.deleted_at = timezone.now()
        instance.deleted_by = self.request.user
        instance.save()
\`\`\`

---

## Nested Routes

### Manual Nested Routes

\`\`\`python
class ArticleCommentViewSet (viewsets.ModelViewSet):
    """
    ViewSet for comments nested under articles
    /articles/{article_pk}/comments/
    """
    serializer_class = CommentSerializer
    
    def get_queryset (self):
        """Filter comments by article"""
        article_pk = self.kwargs['article_pk']
        return Comment.objects.filter (article_id=article_pk)
    
    def perform_create (self, serializer):
        """Auto-set article from URL"""
        article_pk = self.kwargs['article_pk']
        article = Article.objects.get (pk=article_pk)
        serializer.save (article=article, author=self.request.user)

# urls.py
from rest_framework_nested import routers

router = routers.DefaultRouter()
router.register (r'articles', ArticleViewSet)

articles_router = routers.NestedDefaultRouter (router, r'articles', lookup='article')
articles_router.register (r'comments', ArticleCommentViewSet, basename='article-comments')

urlpatterns = [
    path('api/', include (router.urls)),
    path('api/', include (articles_router.urls)),
]

# Generated URLs:
# GET    /api/articles/
# GET    /api/articles/{id}/
# GET    /api/articles/{id}/comments/
# POST   /api/articles/{id}/comments/
# GET    /api/articles/{id}/comments/{comment_id}/
# PUT    /api/articles/{id}/comments/{comment_id}/
# DELETE /api/articles/{id}/comments/{comment_id}/
\`\`\`

---

## Custom ViewSets

### Combining Mixins

\`\`\`python
from rest_framework import mixins, viewsets

class CreateListRetrieveViewSet (mixins.CreateModelMixin,
                                 mixins.ListModelMixin,
                                 mixins.RetrieveModelMixin,
                                 viewsets.GenericViewSet):
    """
    ViewSet that only allows create, list, and retrieve
    No update or delete
    """
    pass

class ArticleViewSet(CreateListRetrieveViewSet):
    queryset = Article.objects.all()
    serializer_class = ArticleSerializer
\`\`\`

### Custom ViewSet from Scratch

\`\`\`python
from rest_framework.viewsets import ViewSet
from rest_framework.response import Response
from rest_framework import status

class ArticleViewSet(ViewSet):
    """
    Fully custom ViewSet without using mixins
    """
    
    def list (self, request):
        articles = Article.objects.all()
        serializer = ArticleSerializer (articles, many=True)
        return Response (serializer.data)
    
    def create (self, request):
        serializer = ArticleSerializer (data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response (serializer.data, status=status.HTTP_201_CREATED)
        return Response (serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def retrieve (self, request, pk=None):
        article = get_object_or_404(Article, pk=pk)
        serializer = ArticleSerializer (article)
        return Response (serializer.data)
    
    def update (self, request, pk=None):
        article = get_object_or_404(Article, pk=pk)
        serializer = ArticleSerializer (article, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response (serializer.data)
        return Response (serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def destroy (self, request, pk=None):
        article = get_object_or_404(Article, pk=pk)
        article.delete()
        return Response (status=status.HTTP_204_NO_CONTENT)
\`\`\`

---

## Production Patterns

### Comprehensive ViewSet Example

\`\`\`python
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticatedOrReadOnly
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import filters

class ArticleViewSet (viewsets.ModelViewSet):
    """
    Production-ready Article ViewSet with all features
    """
    queryset = Article.objects.all()
    permission_classes = [IsAuthenticatedOrReadOnly]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['status', 'category', 'author']
    search_fields = ['title', 'content']
    ordering_fields = ['created_at', 'published_at', 'view_count']
    ordering = ['-published_at']
    
    def get_queryset (self):
        """Optimize queries"""
        queryset = Article.objects.select_related('author', 'category')
        queryset = queryset.prefetch_related('tags', 'comments')
        
        # Filter by user role
        if not self.request.user.is_staff:
            queryset = queryset.filter (status='published')
        
        return queryset
    
    def get_serializer_class (self):
        """Different serializers for different actions"""
        if self.action == 'list':
            return ArticleListSerializer
        elif self.action == 'retrieve':
            return ArticleDetailSerializer
        elif self.action in ['create', 'update', 'partial_update']:
            return ArticleWriteSerializer
        return ArticleSerializer
    
    def get_permissions (self):
        """Custom permissions per action"""
        if self.action in ['update', 'partial_update', 'destroy']:
            permission_classes = [IsAuthenticated, IsAuthorOrAdmin]
        elif self.action == 'create':
            permission_classes = [IsAuthenticated]
        else:
            permission_classes = [AllowAny]
        return [permission() for permission in permission_classes]
    
    def perform_create (self, serializer):
        """Auto-set author and log creation"""
        article = serializer.save (author=self.request.user)
        logger.info (f"Article {article.id} created by {self.request.user}")
    
    def perform_update (self, serializer):
        """Log updates"""
        article = serializer.save()
        logger.info (f"Article {article.id} updated by {self.request.user}")
    
    @action (detail=True, methods=['post'], permission_classes=[IsAuthenticated])
    def publish (self, request, pk=None):
        """Publish article"""
        article = self.get_object()
        
        if article.author != request.user and not request.user.is_staff:
            return Response(
                {'error': 'Only the author can publish this article'},
                status=status.HTTP_403_FORBIDDEN
            )
        
        article.status = 'published'
        article.published_at = timezone.now()
        article.save()
        
        # Trigger notification task
        from .tasks import notify_subscribers
        notify_subscribers.delay (article.id)
        
        serializer = self.get_serializer (article)
        return Response (serializer.data)
    
    @action (detail=False, methods=['get'])
    def stats (self, request):
        """Get article statistics"""
        total = self.get_queryset().count()
        published = self.get_queryset().filter (status='published').count()
        draft = self.get_queryset().filter (status='draft').count()
        
        return Response({
            'total': total,
            'published': published,
            'draft': draft,
            'total_views': self.get_queryset().aggregate(Sum('view_count'))['view_count__sum'] or 0
        })
\`\`\`

---

## Summary

**ViewSet Benefits:**
- ✅ **Less boilerplate**: Combine related views in one class
- ✅ **Automatic URLs**: Routers generate URL patterns
- ✅ **Consistency**: Standardized API structure
- ✅ **Flexibility**: Easy to customize with actions and hooks
- ✅ **Maintainability**: Single place for related logic

**Key Concepts:**1. **ModelViewSet**: Full CRUD operations
2. **ReadOnlyModelViewSet**: List and retrieve only
3. **@action**: Custom endpoints
4. **Routers**: Automatic URL generation
5. **Hook methods**: get_queryset, get_serializer_class, perform_create
6. **Custom ViewSets**: Mix and match mixins

**Production Checklist:**
- ✅ Optimize queries in get_queryset()
- ✅ Use different serializers per action
- ✅ Implement proper permissions
- ✅ Add custom actions for business logic
- ✅ Use filters, search, and ordering
- ✅ Handle errors gracefully
- ✅ Log important operations
- ✅ Add pagination
- ✅ Document custom actions
- ✅ Test all endpoints

ViewSets and Routers are the recommended way to build DRF APIs - they reduce code while maintaining flexibility and following best practices.
`,
};
