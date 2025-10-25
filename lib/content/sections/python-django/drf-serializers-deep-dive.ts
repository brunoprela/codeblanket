export const drfSerializersDeepDive = {
  title: 'DRF Serializers Deep Dive',
  id: 'drf-serializers-deep-dive',
  content: `
# DRF Serializers Deep Dive

## Introduction

**Serializers** in Django REST Framework allow complex data such as querysets and model instances to be converted to native Python datatypes that can then be easily rendered into JSON, XML or other content types. Serializers also provide deserialization, allowing parsed data to be converted back into complex types, after first validating the incoming data.

### Why Serializers?

- **Data Transformation**: Convert Django models to JSON
- **Validation**: Built-in field and object-level validation
- **Nested Relationships**: Handle complex object graphs
- **Read/Write Control**: Separate read and write representations
- **Custom Fields**: Create reusable custom serializer fields

By the end of this section, you'll master:
- ModelSerializer vs Serializer
- Field types and options
- Validation techniques
- Nested serializers
- SerializerMethodField
- Custom serializers
- Performance optimization

---

## ModelSerializer Basics

### Simple ModelSerializer

\`\`\`python
from rest_framework import serializers
from .models import Article

class ArticleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Article
        fields = '__all__'  # All fields
        
# Or specify fields explicitly
class ArticleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Article
        fields = ['id', 'title', 'content', 'author', 'created_at']
        read_only_fields = ['id', 'created_at']
\`\`\`

### Field Selection

\`\`\`python
class ArticleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Article
        # Include specific fields
        fields = ['id', 'title', 'content', 'author']
        
        # Exclude specific fields
        exclude = ['deleted_at', 'internal_notes']
        
        # Read-only fields (GET only)
        read_only_fields = ['id', 'view_count', 'created_at', 'updated_at']
\`\`\`

---

## Field Types and Options

### Common Field Types

\`\`\`python
class ArticleSerializer(serializers.ModelSerializer):
    # Explicitly define fields with options
    title = serializers.CharField(max_length=200, required=True)
    slug = serializers.SlugField(max_length=200, allow_blank=False)
    content = serializers.CharField(style={'base_template': 'textarea.html'})
    excerpt = serializers.CharField(max_length=500, allow_blank=True)
    
    status = serializers.ChoiceField(
        choices=['draft', 'published', 'archived'],
        default='draft'
    )
    
    view_count = serializers.IntegerField(min_value=0, read_only=True)
    rating = serializers.FloatField(min_value=0.0, max_value=5.0)
    
    published_at = serializers.DateTimeField(required=False, allow_null=True)
    featured = serializers.BooleanField(default=False)
    
    # URL field with validation
    source_url = serializers.URLField(required=False, allow_blank=True)
    
    # Email field
    contact_email = serializers.EmailField(required=False)
    
    class Meta:
        model = Article
        fields = '__all__'
\`\`\`

### Field Options

\`\`\`python
serializers.CharField(
    max_length=200,           # Maximum length
    min_length=10,            # Minimum length
    allow_blank=False,        # Allow empty string
    trim_whitespace=True,     # Strip whitespace
    required=True,            # Field is required
    default='',               # Default value
    initial='',               # Initial value for HTML forms
    read_only=False,          # Read-only field
    write_only=False,         # Write-only field (not in output)
    allow_null=False,         # Allow null values
    validators=[],            # Custom validators
    error_messages={},        # Custom error messages
    help_text='',            # Help text
    style={},                 # Widget style
)
\`\`\`

---

## Source and Field Mapping

### Using source Parameter

\`\`\`python
class ArticleSerializer(serializers.ModelSerializer):
    # Map to different model field
    heading = serializers.CharField(source='title')
    
    # Access related object
    author_name = serializers.CharField(source='author.username')
    author_email = serializers.EmailField(source='author.email')
    
    # Call method
    full_name = serializers.CharField(source='author.get_full_name')
    
    # Access property
    is_published = serializers.BooleanField(source='is_published')
    
    class Meta:
        model = Article
        fields = ['heading', 'author_name', 'author_email', 'full_name', 'is_published']
\`\`\`

---

## SerializerMethodField

### Custom Computed Fields

\`\`\`python
class ArticleSerializer(serializers.ModelSerializer):
    # Custom read-only field computed by method
    reading_time = serializers.SerializerMethodField()
    comment_count = serializers.SerializerMethodField()
    is_popular = serializers.SerializerMethodField()
    author_info = serializers.SerializerMethodField()
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'reading_time', 'comment_count', 
                  'is_popular', 'author_info']
    
    def get_reading_time(self, obj):
        """Calculate reading time in minutes"""
        words = len(obj.content.split())
        return max(1, words // 200)  # ~200 words per minute
    
    def get_comment_count(self, obj):
        """Get comment count"""
        return obj.comments.count()
    
    def get_is_popular(self, obj):
        """Check if article is popular"""
        return obj.view_count > 10000
    
    def get_author_info(self, obj):
        """Return author information"""
        return {
            'id': obj.author.id,
            'username': obj.author.username,
            'full_name': obj.author.get_full_name(),
            'email': obj.author.email,
        }
\`\`\`

### Accessing Request Context

\`\`\`python
class ArticleSerializer(serializers.ModelSerializer):
    is_bookmarked = serializers.SerializerMethodField()
    can_edit = serializers.SerializerMethodField()
    
    def get_is_bookmarked(self, obj):
        """Check if current user bookmarked this article"""
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            return obj.bookmarks.filter(user=request.user).exists()
        return False
    
    def get_can_edit(self, obj):
        """Check if current user can edit"""
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            return obj.author == request.user or request.user.is_staff
        return False

# Pass context in view
serializer = ArticleSerializer(article, context={'request': request})
\`\`\`

---

## Nested Serializers

### Nested Read Operations

\`\`\`python
class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = ['id', 'name', 'slug']

class TagSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tag
        fields = ['id', 'name', 'slug']

class AuthorSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name']

class ArticleDetailSerializer(serializers.ModelSerializer):
    # Nested serializers for related objects
    category = CategorySerializer(read_only=True)
    tags = TagSerializer(many=True, read_only=True)
    author = AuthorSerializer(read_only=True)
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'content', 'author', 'category', 
                  'tags', 'created_at']

# Output:
{
    "id": 1,
    "title": "Django Guide",
    "content": "...",
    "author": {
        "id": 5,
        "username": "john",
        "email": "john@example.com",
        "first_name": "John",
        "last_name": "Doe"
    },
    "category": {
        "id": 2,
        "name": "Technology",
        "slug": "technology"
    },
    "tags": [
        {"id": 1, "name": "Django", "slug": "django"},
        {"id": 2, "name": "Python", "slug": "python"}
    ],
    "created_at": "2024-01-01T10:00:00Z"
}
\`\`\`

### Nested Write Operations

\`\`\`python
class ArticleSerializer(serializers.ModelSerializer):
    # Writable nested serializer
    category = CategorySerializer()
    tags = TagSerializer(many=True)
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'content', 'category', 'tags']
    
    def create(self, validated_data):
        """Handle nested writes on creation"""
        # Extract nested data
        category_data = validated_data.pop('category')
        tags_data = validated_data.pop('tags')
        
        # Create category
        category, _ = Category.objects.get_or_create(**category_data)
        
        # Create article
        article = Article.objects.create(category=category, **validated_data)
        
        # Create tags
        for tag_data in tags_data:
            tag, _ = Tag.objects.get_or_create(**tag_data)
            article.tags.add(tag)
        
        return article
    
    def update(self, instance, validated_data):
        """Handle nested writes on update"""
        # Extract nested data
        category_data = validated_data.pop('category', None)
        tags_data = validated_data.pop('tags', None)
        
        # Update basic fields
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        
        # Update category
        if category_data:
            category, _ = Category.objects.get_or_create(**category_data)
            instance.category = category
        
        # Update tags
        if tags_data is not None:
            instance.tags.clear()
            for tag_data in tags_data:
                tag, _ = Tag.objects.get_or_create(**tag_data)
                instance.tags.add(tag)
        
        instance.save()
        return instance
\`\`\`

### Using PrimaryKeyRelatedField

\`\`\`python
class ArticleSerializer(serializers.ModelSerializer):
    # Write with IDs, read with nested data
    category_id = serializers.PrimaryKeyRelatedField(
        queryset=Category.objects.all(),
        source='category',
        write_only=True
    )
    category = CategorySerializer(read_only=True)
    
    tag_ids = serializers.PrimaryKeyRelatedField(
        queryset=Tag.objects.all(),
        source='tags',
        many=True,
        write_only=True
    )
    tags = TagSerializer(many=True, read_only=True)
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'category_id', 'category', 'tag_ids', 'tags']

# Write request:
{
    "title": "New Article",
    "category_id": 2,
    "tag_ids": [1, 3, 5]
}

# Read response:
{
    "id": 10,
    "title": "New Article",
    "category": {"id": 2, "name": "Tech", "slug": "tech"},
    "tags": [
        {"id": 1, "name": "Django", "slug": "django"},
        {"id": 3, "name": "Python", "slug": "python"},
        {"id": 5, "name": "API", "slug": "api"}
    ]
}
\`\`\`

---

## Validation

### Field-Level Validation

\`\`\`python
class ArticleSerializer(serializers.ModelSerializer):
    
    def validate_title(self, value):
        """Validate title field"""
        if len(value) < 10:
            raise serializers.ValidationError(
                "Title must be at least 10 characters long."
            )
        
        # Check for profanity (example)
        banned_words = ['spam', 'fake']
        if any(word in value.lower() for word in banned_words):
            raise serializers.ValidationError(
                "Title contains inappropriate content."
            )
        
        return value
    
    def validate_slug(self, value):
        """Validate slug is unique"""
        if self.instance:  # Update
            if Article.objects.filter(slug=value).exclude(pk=self.instance.pk).exists():
                raise serializers.ValidationError("This slug is already in use.")
        else:  # Create
            if Article.objects.filter(slug=value).exists():
                raise serializers.ValidationError("This slug is already in use.")
        
        return value
    
    def validate_published_at(self, value):
        """Validate publication date"""
        from django.utils import timezone
        if value and value > timezone.now() + timezone.timedelta(days=365):
            raise serializers.ValidationError(
                "Publication date cannot be more than 1 year in the future."
            )
        return value
\`\`\`

### Object-Level Validation

\`\`\`python
class ArticleSerializer(serializers.ModelSerializer):
    
    def validate(self, data):
        """Validate across multiple fields"""
        
        # Check status and published_at consistency
        if data.get('status') == 'published' and not data.get('published_at'):
            raise serializers.ValidationError(
                "Published articles must have a publication date."
            )
        
        # Check content and excerpt relationship
        content = data.get('content', '')
        excerpt = data.get('excerpt', '')
        if excerpt and len(excerpt) > len(content):
            raise serializers.ValidationError(
                "Excerpt cannot be longer than content."
            )
        
        # Validate featured articles have image
        if data.get('featured') and not data.get('image'):
            raise serializers.ValidationError(
                "Featured articles must have an image."
            )
        
        return data
\`\`\`

### Custom Validators

\`\`\`python
from rest_framework.validators import UniqueValidator, UniqueTogetherValidator

def validate_word_count(value):
    """Custom validator function"""
    words = len(value.split())
    if words < 100:
        raise serializers.ValidationError(
            f"Content must be at least 100 words (current: {words} words)."
        )

class ArticleSerializer(serializers.ModelSerializer):
    slug = serializers.SlugField(
        validators=[UniqueValidator(queryset=Article.objects.all())]
    )
    
    content = serializers.CharField(
        validators=[validate_word_count]
    )
    
    class Meta:
        model = Article
        fields = '__all__'
        validators = [
            UniqueTogetherValidator(
                queryset=Article.objects.all(),
                fields=['author', 'slug'],
                message="You already have an article with this slug."
            )
        ]
\`\`\`

---

## Different Serializers for Read and Write

### Separate Serializers

\`\`\`python
# Read serializer with nested objects
class ArticleDetailSerializer(serializers.ModelSerializer):
    author = AuthorSerializer(read_only=True)
    category = CategorySerializer(read_only=True)
    tags = TagSerializer(many=True, read_only=True)
    reading_time = serializers.SerializerMethodField()
    
    class Meta:
        model = Article
        fields = ['id', 'title', 'content', 'author', 'category', 
                  'tags', 'reading_time', 'created_at']
    
    def get_reading_time(self, obj):
        return len(obj.content.split()) // 200

# Write serializer with IDs
class ArticleWriteSerializer(serializers.ModelSerializer):
    category_id = serializers.PrimaryKeyRelatedField(
        queryset=Category.objects.all(),
        source='category'
    )
    tag_ids = serializers.PrimaryKeyRelatedField(
        queryset=Tag.objects.all(),
        source='tags',
        many=True
    )
    
    class Meta:
        model = Article
        fields = ['title', 'content', 'category_id', 'tag_ids']

# Use in view
class ArticleViewSet(viewsets.ModelViewSet):
    queryset = Article.objects.all()
    
    def get_serializer_class(self):
        if self.action in ['create', 'update', 'partial_update']:
            return ArticleWriteSerializer
        return ArticleDetailSerializer
\`\`\`

---

## Custom Serializer Fields

### Creating Custom Fields

\`\`\`python
from rest_framework import serializers

class ColorField(serializers.Field):
    """
    Custom field for hex color codes
    """
    
    def to_representation(self, value):
        """Convert internal to JSON"""
        return value  # Return hex string as-is
    
    def to_internal_value(self, data):
        """Convert JSON to internal"""
        if not isinstance(data, str):
            raise serializers.ValidationError("Color must be a string.")
        
        if not data.startswith('#'):
            raise serializers.ValidationError("Color must start with #.")
        
        if len(data) != 7:
            raise serializers.ValidationError("Color must be in #RRGGBB format.")
        
        try:
            int(data[1:], 16)  # Validate hex
        except ValueError:
            raise serializers.ValidationError("Invalid hex color code.")
        
        return data

class CategorySerializer(serializers.ModelSerializer):
    color = ColorField()
    
    class Meta:
        model = Category
        fields = ['id', 'name', 'color']
\`\`\`

### ListField with Validation

\`\`\`python
class ArticleSerializer(serializers.ModelSerializer):
    keywords = serializers.ListField(
        child=serializers.CharField(max_length=50),
        allow_empty=False,
        max_length=10  # Max 10 keywords
    )
    
    coordinates = serializers.ListField(
        child=serializers.FloatField(),
        min_length=2,
        max_length=2  # [lat, lng]
    )
\`\`\`

### JSONField

\`\`\`python
class ArticleSerializer(serializers.ModelSerializer):
    metadata = serializers.JSONField()
    
    def validate_metadata(self, value):
        """Validate JSON structure"""
        required_keys = ['version', 'source']
        for key in required_keys:
            if key not in value:
                raise serializers.ValidationError(
                    f"Metadata must contain '{key}' field."
                )
        return value
\`\`\`

---

## Performance Optimization

### to_representation Override

\`\`\`python
class ArticleSerializer(serializers.ModelSerializer):
    
    def to_representation(self, instance):
        """Customize output representation"""
        representation = super().to_representation(instance)
        
        # Remove null values
        representation = {
            key: value for key, value in representation.items()
            if value is not None
        }
        
        # Add computed fields conditionally
        request = self.context.get('request')
        if request and 'include_stats' in request.query_params:
            representation['statistics'] = {
                'views': instance.view_count,
                'comments': instance.comments.count(),
                'likes': instance.likes.count(),
            }
        
        return representation
\`\`\`

### Optimizing Nested Queries

\`\`\`python
class ArticleSerializer(serializers.ModelSerializer):
    author = AuthorSerializer()
    
    @classmethod
    def setup_eager_loading(cls, queryset):
        """Optimize queries with select_related/prefetch_related"""
        queryset = queryset.select_related('author', 'category')
        queryset = queryset.prefetch_related('tags', 'comments')
        return queryset

# Use in view
class ArticleListView(generics.ListAPIView):
    serializer_class = ArticleSerializer
    
    def get_queryset(self):
        queryset = Article.objects.all()
        return ArticleSerializer.setup_eager_loading(queryset)
\`\`\`

---

## Dynamic Fields

### Conditionally Include Fields

\`\`\`python
class DynamicFieldsModelSerializer(serializers.ModelSerializer):
    """
    A ModelSerializer that takes an additional \`fields\` argument to
    dynamically include/exclude fields.
    """
    
    def __init__(self, *args, **kwargs):
        # Get fields from kwargs or context
        fields = kwargs.pop('fields', None)
        if fields is None:
            request = kwargs.get('context', {}).get('request')
            if request:
                fields = request.query_params.get('fields')
                if fields:
                    fields = fields.split(',')
        
        super().__init__(*args, **kwargs)
        
        if fields is not None:
            # Drop any fields not specified
            allowed = set(fields)
            existing = set(self.fields)
            for field_name in existing - allowed:
                self.fields.pop(field_name)

class ArticleSerializer(DynamicFieldsModelSerializer):
    class Meta:
        model = Article
        fields = ['id', 'title', 'content', 'author', 'created_at', 
                  'view_count', 'comment_count']

# Usage:
# GET /api/articles/?fields=id,title,author
# Returns only id, title, and author fields
\`\`\`

---

## Summary

**Key Serializer Concepts:**

1. **ModelSerializer**: Automatic serialization from Django models
2. **SerializerMethodField**: Custom computed fields
3. **Nested Serializers**: Handle related objects
4. **Validation**: Field-level and object-level validation
5. **source Parameter**: Map fields to different model attributes
6. **read_only/write_only**: Control field direction
7. **Custom Fields**: Create reusable field types
8. **Performance**: Optimize queries for nested serializers

**Production Best Practices:**
- ✅ Use separate serializers for read/write when complex
- ✅ Optimize queries with select_related/prefetch_related
- ✅ Validate all input data comprehensively
- ✅ Use SerializerMethodField for computed values
- ✅ Implement dynamic field inclusion for flexibility
- ✅ Handle nested writes carefully
- ✅ Add helpful error messages
- ✅ Document expected formats
- ✅ Use appropriate field types
- ✅ Consider response size (pagination, field selection)

**Common Patterns:**
- Different serializers for list vs detail views
- Nested objects for reads, IDs for writes
- SerializerMethodField for computed/aggregated data
- Dynamic field inclusion via query params
- Custom validation for business logic
- Context-aware serialization (user permissions, etc.)

Serializers are the heart of DRF - mastering them is essential for building robust, efficient APIs.
\`,
};
