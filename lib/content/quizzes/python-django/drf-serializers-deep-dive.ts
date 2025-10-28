export const drfSerializersDeepDiveQuiz = [
  {
    id: 1,
    question:
      'Explain nested serializers in DRF, including writable nested serializers, handling create/update operations, and performance optimization with select_related/prefetch_related. Provide production examples with complex relationships.',
    answer: `
**Nested Serializers in DRF:**

Nested serializers allow you to represent related objects within a parent serializer, creating hierarchical JSON structures.

**1. Read-Only Nested Serializers (Simple):**

\`\`\`python
class AuthorSerializer (serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email']

class ArticleSerializer (serializers.ModelSerializer):
    author = AuthorSerializer (read_only=True)  # Nested
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'content', 'author']

# Output:
{
    "id": 1,
    "title": "Django Tips",
    "content": "...",
    "author": {
        "id": 5,
        "username": "john",
        "email": "john@example.com"
    }
}
\`\`\`

**2. Writable Nested Serializers:**

\`\`\`python
class CommentSerializer (serializers.ModelSerializer):
    class Meta:
        model = Comment
        fields = ['id', 'text', 'created_at']

class ArticleSerializer (serializers.ModelSerializer):
    comments = CommentSerializer (many=True)  # Writable nested
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'content', 'comments']
    
    def create (self, validated_data):
        """Handle nested creation"""
        comments_data = validated_data.pop('comments')
        article = Article.objects.create(**validated_data)
        
        for comment_data in comments_data:
            Comment.objects.create (article=article, **comment_data)
        
        return article
    
    def update (self, instance, validated_data):
        """Handle nested updates"""
        comments_data = validated_data.pop('comments', None)
        
        # Update article fields
        instance.title = validated_data.get('title', instance.title)
        instance.content = validated_data.get('content', instance.content)
        instance.save()
        
        if comments_data is not None:
            # Clear existing comments
            instance.comments.all().delete()
            
            # Create new comments
            for comment_data in comments_data:
                Comment.objects.create (article=instance, **comment_data)
        
        return instance

# POST /api/articles/
{
    "title": "New Article",
    "content": "Content here",
    "comments": [
        {"text": "Great article!"},
        {"text": "Very informative"}
    ]
}
\`\`\`

**3. Advanced Writable Nested Serializers:**

\`\`\`python
class TagSerializer (serializers.ModelSerializer):
    class Meta:
        model = Tag
        fields = ['id', 'name']

class ImageSerializer (serializers.ModelSerializer):
    class Meta:
        model = ArticleImage
        fields = ['id', 'image', 'caption']

class ArticleSerializer (serializers.ModelSerializer):
    tags = TagSerializer (many=True)
    images = ImageSerializer (many=True)
    author = serializers.PrimaryKeyRelatedField (read_only=True)
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'content', 'author', 'tags', 'images']
    
    def create (self, validated_data):
        """Handle multiple nested relationships"""
        tags_data = validated_data.pop('tags')
        images_data = validated_data.pop('images')
        
        # Create article
        article = Article.objects.create(**validated_data)
        
        # Handle tags (get_or_create for reusable tags)
        for tag_data in tags_data:
            tag, created = Tag.objects.get_or_create(
                name=tag_data['name'],
                defaults=tag_data
            )
            article.tags.add (tag)
        
        # Handle images (always create new)
        for image_data in images_data:
            ArticleImage.objects.create (article=article, **image_data)
        
        return article
    
    def update (self, instance, validated_data):
        """Handle updates with nested relationships"""
        tags_data = validated_data.pop('tags', None)
        images_data = validated_data.pop('images', None)
        
        # Update article fields
        for attr, value in validated_data.items():
            setattr (instance, attr, value)
        instance.save()
        
        # Update tags
        if tags_data is not None:
            instance.tags.clear()
            for tag_data in tags_data:
                tag, created = Tag.objects.get_or_create(
                    name=tag_data['name'],
                    defaults=tag_data
                )
                instance.tags.add (tag)
        
        # Update images
        if images_data is not None:
            # Get existing image IDs
            existing_ids = {img.get('id') for img in images_data if img.get('id')}
            
            # Delete images not in the update
            instance.images.exclude (id__in=existing_ids).delete()
            
            for image_data in images_data:
                image_id = image_data.get('id')
                if image_id:
                    # Update existing
                    ArticleImage.objects.filter (id=image_id).update(**image_data)
                else:
                    # Create new
                    ArticleImage.objects.create (article=instance, **image_data)
        
        return instance
\`\`\`

**4. Performance Optimization:**

\`\`\`python
class ArticleSerializer (serializers.ModelSerializer):
    author = AuthorSerializer (read_only=True)
    category = CategorySerializer (read_only=True)
    tags = TagSerializer (many=True, read_only=True)
    comments = CommentSerializer (many=True, read_only=True)
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'author', 'category', 'tags', 'comments']

# ❌ N+1 Query Problem
articles = Article.objects.all()
serializer = ArticleSerializer (articles, many=True)
# Queries: 1 for articles + N for authors + N for categories + N for tags + N for comments

# ✅ Optimized with select_related/prefetch_related
articles = Article.objects.select_related(
    'author',
    'category'
).prefetch_related(
    'tags',
    'comments'
)
serializer = ArticleSerializer (articles, many=True)
# Queries: 1 for articles + 1 JOIN for author/category + 1 for tags + 1 for comments = 4 queries total

# ViewSet optimization
class ArticleViewSet (viewsets.ModelViewSet):
    serializer_class = ArticleSerializer
    
    def get_queryset (self):
        """Optimize queryset based on action"""
        qs = Article.objects.all()
        
        if self.action == 'list':
            # List: Basic optimization
            qs = qs.select_related('author', 'category')
        elif self.action == 'retrieve':
            # Detail: Full optimization
            qs = qs.select_related('author', 'category').prefetch_related(
                'tags',
                Prefetch('comments', queryset=Comment.objects.select_related('user'))
            )
        
        return qs
\`\`\`

**5. Depth Control:**

\`\`\`python
# Automatic nested serialization with depth
class ArticleSerializer (serializers.ModelSerializer):
    class Meta:
        model = Article
        fields = '__all__'
        depth = 2  # Automatically nest 2 levels deep

# Output includes:
# Article -> Author (depth 1) -> Profile (depth 2)

# ⚠️ Warning: depth creates read-only nested serializers
# Better to explicitly define nested serializers for control
\`\`\`

**6. Conditional Nested Serializers:**

\`\`\`python
class ArticleSerializer (serializers.ModelSerializer):
    author = serializers.SerializerMethodField()
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'author']
    
    def get_author (self, obj):
        """Conditionally nest author based on context"""
        request = self.context.get('request')
        
        # Full details for authenticated users
        if request and request.user.is_authenticated:
            return AuthorDetailSerializer (obj.author).data
        
        # Minimal info for anonymous users
        return {'id': obj.author.id, 'username': obj.author.username}
\`\`\`

**7. Partial Updates with Nested Serializers:**

\`\`\`python
class ArticleSerializer (serializers.ModelSerializer):
    tags = TagSerializer (many=True, required=False)
    images = ImageSerializer (many=True, required=False)
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'content', 'tags', 'images']
    
    def update (self, instance, validated_data):
        """Support partial updates"""
        # Only update provided fields
        for attr in ['title', 'content']:
            if attr in validated_data:
                setattr (instance, attr, validated_data[attr])
        
        instance.save()
        
        # Only update tags if provided
        if 'tags' in validated_data:
            tags_data = validated_data['tags']
            instance.tags.clear()
            for tag_data in tags_data:
                tag, _ = Tag.objects.get_or_create(**tag_data)
                instance.tags.add (tag)
        
        # Only update images if provided
        if 'images' in validated_data:
            # Handle images update
            pass
        
        return instance

# PATCH /api/articles/1/
{
    "title": "Updated Title"  # Only update title, leave tags/images unchanged
}
\`\`\`

**Production Best Practices:**

**Performance:**
- ✅ Always use select_related/prefetch_related with nested serializers
- ✅ Optimize based on action (list vs retrieve)
- ✅ Use Prefetch objects for complex nested queries
- ✅ Consider using SerializerMethodField for computed nested data
- ❌ Don't use depth parameter in production (too inflexible)
- ❌ Don't forget to optimize nested queries (N+1 problem)

**Writable Nested Serializers:**
- ✅ Explicitly handle create() and update()
- ✅ Use transactions for multi-model creates
- ✅ Use get_or_create for reusable nested objects (tags, categories)
- ✅ Validate nested data thoroughly
- ✅ Handle partial updates correctly
- ❌ Don't silently ignore nested data
- ❌ Don't allow deep nesting (max 2-3 levels)

**API Design:**
- ✅ Use nested serializers for GET (read)
- ✅ Use IDs for POST/PUT/PATCH (write)
- ✅ Provide separate endpoints for managing relationships
- ✅ Document nested structure clearly
- ❌ Don't make deeply nested POST requests required
- ❌ Don't expose sensitive nested data

**Example Production Pattern:**

\`\`\`python
# Read: Nested serializers
class ArticleDetailSerializer (serializers.ModelSerializer):
    author = AuthorSerializer (read_only=True)
    tags = TagSerializer (many=True, read_only=True)
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'content', 'author', 'tags']

# Write: ID-based
class ArticleCreateUpdateSerializer (serializers.ModelSerializer):
    author_id = serializers.IntegerField (write_only=True)
    tag_ids = serializers.ListField(
        child=serializers.IntegerField(),
        write_only=True
    )
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'content', 'author_id', 'tag_ids']
    
    def create (self, validated_data):
        tag_ids = validated_data.pop('tag_ids', [])
        article = Article.objects.create(**validated_data)
        article.tags.set (tag_ids)
        return article

# ViewSet
class ArticleViewSet (viewsets.ModelViewSet):
    queryset = Article.objects.all()
    
    def get_serializer_class (self):
        if self.action in ['create', 'update', 'partial_update']:
            return ArticleCreateUpdateSerializer
        return ArticleDetailSerializer
    
    def get_queryset (self):
        if self.action == 'retrieve':
            return self.queryset.select_related('author').prefetch_related('tags')
        return self.queryset.all()
\`\`\`

This approach provides optimal performance, flexibility, and maintainability for production APIs with nested relationships.
      `,
  },
  {
    question:
      'Describe custom serializer fields, SerializerMethodField, and field-level validation in DRF. Explain how to implement complex validation logic, custom representations, and field-level permissions.',
    answer: `
**Custom Serializer Fields in DRF:**

**1. SerializerMethodField (Read-Only):**

\`\`\`python
class ArticleSerializer (serializers.ModelSerializer):
    # Computed field
    word_count = serializers.SerializerMethodField()
    reading_time = serializers.SerializerMethodField()
    author_name = serializers.SerializerMethodField()
    is_owner = serializers.SerializerMethodField()
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'content', 'word_count', 'reading_time', 'author_name', 'is_owner']
    
    def get_word_count (self, obj):
        """Calculate word count from content"""
        return len (obj.content.split())
    
    def get_reading_time (self, obj):
        """Estimate reading time (250 words per minute)"""
        word_count = len (obj.content.split())
        minutes = max(1, word_count // 250)
        return f"{minutes} min read"
    
    def get_author_name (self, obj):
        """Get author's full name"""
        return f"{obj.author.first_name} {obj.author.last_name}"
    
    def get_is_owner (self, obj):
        """Check if current user is owner"""
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            return obj.author == request.user
        return False

# Output:
{
    "id": 1,
    "title": "My Article",
    "content": "...",
    "word_count": 523,
    "reading_time": "2 min read",
    "author_name": "John Doe",
    "is_owner": true
}
\`\`\`

**2. Custom Field Classes:**

\`\`\`python
from rest_framework import serializers

class LowercaseCharField (serializers.CharField):
    """Custom field that converts to lowercase"""
    
    def to_internal_value (self, data):
        """Convert input to lowercase"""
        data = super().to_internal_value (data)
        return data.lower()
    
    def to_representation (self, value):
        """Convert output to lowercase"""
        return value.lower() if value else value

class Base64ImageField (serializers.ImageField):
    """Custom field for base64 image upload"""
    
    def to_internal_value (self, data):
        import base64
        import io
        from django.core.files.uploadedfile import InMemoryUploadedFile
        
        if isinstance (data, str) and data.startswith('data:image'):
            # Parse base64 string
            format, imgstr = data.split(';base64,')
            ext = format.split('/')[-1]
            
            # Decode
            decoded = base64.b64decode (imgstr)
            
            # Create file
            file = InMemoryUploadedFile(
                io.BytesIO(decoded),
                field_name='image',
                name=f'image.{ext}',
                content_type=f'image/{ext}',
                size=len (decoded),
                charset=None
            )
            
            return super().to_internal_value (file)
        
        return super().to_internal_value (data)

class ColorField (serializers.Field):
    """Custom field for hex color codes"""
    
    def to_representation (self, value):
        """Convert to hex string"""
        return value
    
    def to_internal_value (self, data):
        """Validate hex color code"""
        if not isinstance (data, str):
            raise serializers.ValidationError('Color must be a string')
        
        if not data.startswith('#'):
            raise serializers.ValidationError('Color must start with #')
        
        if len (data) not in [4, 7]:  # #RGB or #RRGGBB
            raise serializers.ValidationError('Invalid color format')
        
        try:
            int (data[1:], 16)  # Validate hex
        except ValueError:
            raise serializers.ValidationError('Invalid hex color')
        
        return data

# Usage
class ProductSerializer (serializers.ModelSerializer):
    sku = LowercaseCharField (max_length=50)
    image = Base64ImageField()
    color = ColorField()
    
    class Meta:
        model = Product
        fields = ['id', 'name', 'sku', 'image', 'color']
\`\`\`

**3. Field-Level Validation:**

\`\`\`python
class ArticleSerializer (serializers.ModelSerializer):
    class Meta:
        model = Article
        fields = ['id', 'title', 'slug', 'content', 'published_at']
    
    def validate_title (self, value):
        """Validate title field"""
        # Check length
        if len (value) < 10:
            raise serializers.ValidationError('Title must be at least 10 characters')
        
        # Check for spam keywords
        spam_keywords = ['spam', 'click here', 'buy now']
        if any (keyword in value.lower() for keyword in spam_keywords):
            raise serializers.ValidationError('Title contains spam keywords')
        
        # Check uniqueness (excluding current instance)
        instance = getattr (self, 'instance', None)
        if Article.objects.exclude (pk=instance.pk if instance else None).filter (title=value).exists():
            raise serializers.ValidationError('Article with this title already exists')
        
        return value
    
    def validate_slug (self, value):
        """Validate slug format"""
        import re
        if not re.match (r'^[a-z0-9-]+$', value):
            raise serializers.ValidationError('Slug must contain only lowercase letters, numbers, and hyphens')
        return value
    
    def validate_content (self, value):
        """Validate content"""
        if len (value) < 100:
            raise serializers.ValidationError('Content must be at least 100 characters')
        
        # Check for minimum number of paragraphs
        paragraphs = [p for p in value.split('\n\n') if p.strip()]
        if len (paragraphs) < 3:
            raise serializers.ValidationError('Content must have at least 3 paragraphs')
        
        return value
    
    def validate_published_at (self, value):
        """Validate publication date"""
        from django.utils import timezone
        
        if value and value < timezone.now():
            raise serializers.ValidationError('Publication date cannot be in the past')
        
        return value
\`\`\`

**4. Object-Level Validation:**

\`\`\`python
class ArticleSerializer (serializers.ModelSerializer):
    class Meta:
        model = Article
        fields = ['id', 'title', 'status', 'published_at']
    
    def validate (self, attrs):
        """Validate across multiple fields"""
        status = attrs.get('status')
        published_at = attrs.get('published_at')
        
        # If publishing, must have publication date
        if status == 'published' and not published_at:
            raise serializers.ValidationError({
                'published_at': 'Publication date required for published articles'
            })
        
        # If not published, shouldn't have publication date
        if status == 'draft' and published_at:
            raise serializers.ValidationError({
                'published_at': 'Draft articles cannot have publication date'
            })
        
        # Check user permissions
        request = self.context.get('request')
        if request and status == 'published':
            if not request.user.has_perm('articles.publish_article'):
                raise serializers.ValidationError({
                    'status': 'You do not have permission to publish articles'
                })
        
        return attrs
\`\`\`

**5. Complex Custom Validation:**

\`\`\`python
class OrderSerializer (serializers.ModelSerializer):
    items = OrderItemSerializer (many=True)
    
    class Meta:
        model = Order
        fields = ['id', 'customer', 'items', 'total', 'discount_code']
    
    def validate_items (self, value):
        """Validate order items"""
        if not value:
            raise serializers.ValidationError('Order must have at least one item')
        
        if len (value) > 50:
            raise serializers.ValidationError('Order cannot have more than 50 items')
        
        # Check stock availability
        for item in value:
            product = item['product']
            quantity = item['quantity']
            
            if product.stock < quantity:
                raise serializers.ValidationError(
                    f'Insufficient stock for {product.name}. Available: {product.stock}'
                )
        
        return value
    
    def validate_discount_code (self, value):
        """Validate discount code"""
        if not value:
            return value
        
        try:
            discount = DiscountCode.objects.get (code=value)
        except DiscountCode.DoesNotExist:
            raise serializers.ValidationError('Invalid discount code')
        
        # Check if expired
        if discount.is_expired():
            raise serializers.ValidationError('This discount code has expired')
        
        # Check usage limit
        if discount.usage_count >= discount.max_uses:
            raise serializers.ValidationError('This discount code has reached its usage limit')
        
        return value
    
    def validate (self, attrs):
        """Cross-field validation"""
        items = attrs.get('items', [])
        discount_code = attrs.get('discount_code')
        
        # Calculate total
        subtotal = sum (item['quantity'] * item['product'].price for item in items)
        
        # Apply discount
        if discount_code:
            discount_obj = DiscountCode.objects.get (code=discount_code)
            
            # Check minimum order amount
            if subtotal < discount_obj.minimum_order:
                raise serializers.ValidationError({
                    'discount_code': f'Minimum order amount for this discount is $''{discount_obj.minimum_order}'
                })
            
            discount_amount = discount_obj.calculate_discount (subtotal)
            total = subtotal - discount_amount
        else:
            total = subtotal
        
        attrs['calculated_total'] = total
        return attrs
\`\`\`

**6. Field-Level Permissions:**

\`\`\`python
class ArticleSerializer (serializers.ModelSerializer):
    class Meta:
        model = Article
        fields = ['id', 'title', 'content', 'status', 'featured', 'internal_notes']
        read_only_fields = ['featured']  # Only admins can set featured
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        request = self.context.get('request')
        
        # Hide internal notes from non-staff
        if request and not request.user.is_staff:
            self.fields.pop('internal_notes', None)
        
        # Only admins can change status to 'published'
        if request and not request.user.has_perm('articles.publish_article'):
            if 'status' in self.fields:
                self.fields['status'].read_only = True
    
    def validate_featured (self, value):
        """Only admins can set featured"""
        request = self.context.get('request')
        if request and not request.user.is_staff:
            raise serializers.ValidationError('Only staff can set featured status')
        return value
\`\`\`

**7. Dynamic Fields:**

\`\`\`python
class DynamicFieldsSerializer (serializers.ModelSerializer):
    """Serializer that supports dynamic field selection"""
    
    def __init__(self, *args, **kwargs):
        # Get fields from context
        fields = self.context.get('fields')
        
        super().__init__(*args, **kwargs)
        
        if fields:
            # Remove fields not in the requested set
            allowed = set (fields.split(','))
            existing = set (self.fields.keys())
            for field_name in existing - allowed:
                self.fields.pop (field_name)

class ArticleSerializer(DynamicFieldsSerializer):
    class Meta:
        model = Article
        fields = ['id', 'title', 'content', 'author', 'published_at']

# Usage in ViewSet
class ArticleViewSet (viewsets.ModelViewSet):
    serializer_class = ArticleSerializer
    
    def get_serializer_context (self):
        context = super().get_serializer_context()
        # Pass requested fields
        context['fields'] = self.request.query_params.get('fields')
        return context

# API call:
# GET /api/articles/?fields=id,title,author
# Returns only id, title, and author fields
\`\`\`

**Production Best Practices:**

**Performance:**
- ✅ Use SerializerMethodField for computed data
- ✅ Cache expensive SerializerMethodField calculations
- ✅ Keep validation logic fast
- ❌ Don't perform database queries in SerializerMethodField (use prefetch)
- ❌ Don't make external API calls in validation

**Validation:**
- ✅ Validate at field level when possible
- ✅ Use object-level validation for cross-field logic
- ✅ Provide clear, specific error messages
- ✅ Return field-specific errors
- ✅ Validate business rules thoroughly
- ❌ Don't mix validation with business logic
- ❌ Don't perform side effects in validation
- ❌ Don't validate in multiple places

**Security:**
- ✅ Validate user permissions in validation
- ✅ Hide sensitive fields based on user
- ✅ Sanitize input data
- ✅ Implement rate limiting for expensive validations
- ❌ Don't expose sensitive data in error messages
- ❌ Don't trust client-side validation alone

This comprehensive approach ensures robust, secure, and performant DRF serializers in production.
      `,
  },
  {
    question:
      'Explain serializer performance optimization techniques, including queryset optimization, caching, and bulk operations. Provide examples of optimizing serializers for large datasets and high-traffic APIs.',
    answer: `
**Serializer Performance Optimization:**

**1. QuerySet Optimization:**

\`\`\`python
# ❌ N+1 Query Problem
class ArticleSerializer (serializers.ModelSerializer):
    author_name = serializers.CharField (source='author.username', read_only=True)
    category_name = serializers.CharField (source='category.name', read_only=True)
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'author_name', 'category_name']

# Without optimization: 1 query for articles + N for authors + N for categories
articles = Article.objects.all()
serializer = ArticleSerializer (articles, many=True)

# ✅ Optimized with select_related
class ArticleViewSet (viewsets.ModelViewSet):
    serializer_class = ArticleSerializer
    
    def get_queryset (self):
        return Article.objects.select_related('author', 'category')

# Now: 1 query with JOINs (3 queries → 1 query)

# ✅ Optimize Many-to-Many and Reverse ForeignKey
class ArticleSerializer (serializers.ModelSerializer):
    tags = TagSerializer (many=True, read_only=True)
    comments_count = serializers.IntegerField (read_only=True)
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'tags', 'comments_count']

class ArticleViewSet (viewsets.ModelViewSet):
    serializer_class = ArticleSerializer
    
    def get_queryset (self):
        from django.db.models import Count
        
        return Article.objects.prefetch_related('tags').annotate(
            comments_count=Count('comments')
        )

# Efficient: 1 query for articles + 1 for tags + 0 for count (already annotated)
\`\`\`

**2. Only/Defer for Field Selection:**

\`\`\`python
class ArticleListSerializer (serializers.ModelSerializer):
    """Light serializer for list views"""
    class Meta:
        model = Article
        fields = ['id', 'title', 'author', 'published_at']

class ArticleDetailSerializer (serializers.ModelSerializer):
    """Full serializer for detail views"""
    class Meta:
        model = Article
        fields = '__all__'

class ArticleViewSet (viewsets.ModelViewSet):
    queryset = Article.objects.all()
    
    def get_serializer_class (self):
        if self.action == 'list':
            return ArticleListSerializer
        return ArticleDetailSerializer
    
    def get_queryset (self):
        qs = super().get_queryset()
        
        if self.action == 'list':
            # Only fetch needed fields
            qs = qs.only('id', 'title', 'author_id', 'published_at')
        
        return qs

# or defer expensive fields
qs = Article.objects.defer('content', 'metadata')  # Skip large text fields
\`\`\`

**3. Caching Serialized Data:**

\`\`\`python
from django.core.cache import cache
from django.utils.encoding import force_str
from rest_framework.response import Response
import hashlib
import json

class CachedSerializerMixin:
    """Mixin to cache serialized data"""
    cache_timeout = 300  # 5 minutes
    
    def get_cache_key (self, instance):
        """Generate cache key for instance"""
        model_name = instance.__class__.__name__
        return f'serializer_{model_name}_{instance.pk}_{instance.updated_at.timestamp()}'
    
    def to_representation (self, instance):
        """Cache serialized representation"""
        cache_key = self.get_cache_key (instance)
        cached_data = cache.get (cache_key)
        
        if cached_data:
            return cached_data
        
        data = super().to_representation (instance)
        cache.set (cache_key, data, self.cache_timeout)
        
        return data

class ArticleSerializer(CachedSerializerMixin, serializers.ModelSerializer):
    class Meta:
        model = Article
        fields = ['id', 'title', 'content', 'author']

# View-level caching
from django.views.decorators.cache import cache_page
from django.utils.decorators import method_decorator

class ArticleViewSet (viewsets.ModelViewSet):
    
    @method_decorator (cache_page(60 * 15))  # Cache for 15 minutes
    def list (self, request, *args, **kwargs):
        return super().list (request, *args, **kwargs)
\`\`\`

**4. Bulk Operations:**

\`\`\`python
class BulkArticleSerializer (serializers.ListSerializer):
    """Optimized bulk serializer"""
    
    def create (self, validated_data):
        """Bulk create articles"""
        articles = [Article(**item) for item in validated_data]
        return Article.objects.bulk_create (articles)
    
    def update (self, instance, validated_data):
        """Bulk update articles"""
        # Map instances by ID
        article_mapping = {article.id: article for article in instance}
        
        # Update attributes
        updated_articles = []
        for item in validated_data:
            article = article_mapping.get (item['id'])
            if article:
                for attr, value in item.items():
                    setattr (article, attr, value)
                updated_articles.append (article)
        
        # Bulk update
        Article.objects.bulk_update(
            updated_articles,
            ['title', 'content', 'status']
        )
        
        return updated_articles

class ArticleSerializer (serializers.ModelSerializer):
    class Meta:
        model = Article
        fields = ['id', 'title', 'content', 'status']
        list_serializer_class = BulkArticleSerializer

# Usage
data = [
    {'title': 'Article 1', 'content': '...'},
    {'title': 'Article 2', 'content': '...'},
    # ... hundreds more
]

serializer = ArticleSerializer (data=data, many=True)
if serializer.is_valid():
    serializer.save()  # Bulk creates all articles
\`\`\`

**5. Pagination for Large Datasets:**

\`\`\`python
from rest_framework.pagination import CursorPagination

class ArticleCursorPagination(CursorPagination):
    """Efficient pagination for large datasets"""
    page_size = 100
    ordering = '-created_at'

class ArticleViewSet (viewsets.ModelViewSet):
    pagination_class = ArticleCursorPagination
    
    def get_queryset (self):
        # Optimize with only needed fields
        return Article.objects.only(
            'id', 'title', 'created_at'
        ).order_by('-created_at')

# Cursor pagination is faster than offset pagination for large datasets
# No COUNT(*) query, constant performance regardless of page number
\`\`\`

**6. Conditional Serialization:**

\`\`\`python
class ArticleSerializer (serializers.ModelSerializer):
    # Expensive computed fields
    related_articles = serializers.SerializerMethodField()
    stats = serializers.SerializerMethodField()
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'related_articles', 'stats']
    
    def __init__(self, *args, **kwargs):
        # Remove expensive fields if not requested
        include_related = self.context.get('include_related', False)
        include_stats = self.context.get('include_stats', False)
        
        super().__init__(*args, **kwargs)
        
        if not include_related:
            self.fields.pop('related_articles', None)
        if not include_stats:
            self.fields.pop('stats', None)
    
    def get_related_articles (self, obj):
        # Expensive operation
        related = obj.get_related_articles()
        return MinimalArticleSerializer (related, many=True).data
    
    def get_stats (self, obj):
        # Expensive calculation
        return obj.calculate_stats()

# ViewSet
class ArticleViewSet (viewsets.ModelViewSet):
    def get_serializer_context (self):
        context = super().get_serializer_context()
        # Allow client to request optional fields
        context['include_related'] = self.request.query_params.get('include_related') == 'true'
        context['include_stats'] = self.request.query_params.get('include_stats') == 'true'
        return context

# API call:
# GET /api/articles/  # Fast, no expensive fields
# GET /api/articles/?include_related=true  # Include related articles
\`\`\`

**7. Database-Level Aggregation:**

\`\`\`python
from django.db.models import Count, Avg, Sum, F

class ArticleSerializer (serializers.ModelSerializer):
    comment_count = serializers.IntegerField (read_only=True)
    avg_rating = serializers.FloatField (read_only=True)
    total_views = serializers.IntegerField (read_only=True)
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'comment_count', 'avg_rating', 'total_views']

class ArticleViewSet (viewsets.ModelViewSet):
    def get_queryset (self):
        # Calculate aggregations in database, not Python
        return Article.objects.annotate(
            comment_count=Count('comments'),
            avg_rating=Avg('ratings__value'),
            total_views=Sum('views__count')
        )

# Much faster than calculating in SerializerMethodField
\`\`\`

**8. Async Serialization (Django 4.1+):**

\`\`\`python
import asyncio
from asgiref.sync import sync_to_async

class AsyncArticleSerializer (serializers.ModelSerializer):
    related_data = serializers.SerializerMethodField()
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'related_data']
    
    async def aget_related_data (self, obj):
        """Async method for expensive operation"""
        # Perform async operation
        result = await some_async_operation (obj)
        return result
    
    def get_related_data (self, obj):
        """Sync wrapper for async method"""
        return asyncio.run (self.aget_related_data (obj))
\`\`\`

**9. Monitoring and Profiling:**

\`\`\`python
import time
import logging

logger = logging.getLogger(__name__)

class ProfilingSerializerMixin:
    """Mixin to profile serializer performance"""
    
    def to_representation (self, instance):
        start = time.time()
        data = super().to_representation (instance)
        duration = time.time() - start
        
        if duration > 0.1:  # Log if > 100ms
            logger.warning(
                f'Slow serialization: {self.__class__.__name__} '
                f'took {duration:.3f}s for {instance}'
            )
        
        return data

class ArticleSerializer(ProfilingSerializerMixin, serializers.ModelSerializer):
    class Meta:
        model = Article
        fields = '__all__'
\`\`\`

**Production Best Practices:**

**QuerySet Optimization:**
- ✅ Always use select_related for ForeignKey/OneToOne
- ✅ Always use prefetch_related for ManyToMany/reverse FK
- ✅ Use only()/defer() for large models
- ✅ Annotate counts/aggregations in database
- ❌ Don't access related objects without prefetching
- ❌ Don't calculate aggregations in Python

**Caching:**
- ✅ Cache expensive serialized data
- ✅ Use cache versioning (updated_at in key)
- ✅ Cache at appropriate level (view > serializer > field)
- ❌ Don't cache without invalidation strategy
- ❌ Don't cache user-specific data globally

**Serializer Design:**
- ✅ Use different serializers for list vs detail
- ✅ Make expensive fields optional
- ✅ Use bulk operations for multiple objects
- ✅ Profile and monitor serializer performance
- ❌ Don't include all fields by default
- ❌ Don't perform expensive operations in SerializerMethodField

**Pagination:**
- ✅ Always paginate list endpoints
- ✅ Use cursor pagination for large datasets
- ✅ Keep page size reasonable (50-100)
- ❌ Don't allow unlimited page sizes
- ❌ Don't use offset pagination for millions of records

**Benchmarks (1000 articles):**
- No optimization: ~5000ms, 2001 queries
- select_related: ~500ms, 1 query
- + prefetch_related: ~600ms, 2 queries
- + only(): ~400ms, 2 queries
- + caching: ~50ms, 0 queries (cache hit)

These optimizations can improve API performance by 100x or more!
      `,
  },
].map(({ id: _id, ...q }, idx) => ({
  id: `django-q-${idx + 1}`,
  question: q.question,
  sampleAnswer: String(q.answer),
  keyPoints: [],
}));
