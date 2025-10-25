export const advancedFiltering = {
  title: 'Advanced Filtering & Expressions',
  id: 'advanced-filtering',
  content: `
# Advanced Filtering & Expressions

## Introduction

Beyond basic WHERE clauses, SQLAlchemy supports sophisticated filtering techniques: JSON querying, array operations, full-text search, pattern matching, custom operators, and database-specific features. These advanced techniques are essential for building modern applications with complex data structures.

In this section, you'll master:
- JSON and JSONB column querying (PostgreSQL)
- Array operations and containment queries
- Full-text search implementation
- Pattern matching and similarity search
- Custom SQL operators and functions
- Hybrid properties for computed columns
- Database-specific features
- Type casting and coercion
- Complex boolean logic

### Why Advanced Filtering Matters

Modern applications store semi-structured data (JSON), arrays, and require advanced search capabilities. Understanding these techniques separates basic CRUD apps from sophisticated production systems.

---

## JSON Column Querying

### JSON vs JSONB (PostgreSQL)

\`\`\`python
"""
JSON Column Types
"""

from sqlalchemy import JSON
from sqlalchemy.dialects.postgresql import JSONB

class Product(Base):
    __tablename__ = 'products'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    
    # JSON: Stores as text, slower queries
    config_json: Mapped[dict] = mapped_column(JSON)
    
    # JSONB: Binary format, faster queries, supports indexing
    metadata: Mapped[dict] = mapped_column(JSONB)  # PREFERRED

# Why JSONB?
# ✅ Binary format (faster processing)
# ✅ Supports GIN indexes (fast searches)
# ✅ Deduplicates keys
# ✅ Orders keys consistently
# ❌ Slightly slower writes (binary conversion)

# JSON advantages:
# ✅ Preserves whitespace and key order
# ✅ Faster writes (no conversion)
# ❌ No indexing support
# ❌ Slower queries

# Rule: Use JSONB for querying, JSON only if you need exact text preservation
\`\`\`

### Basic JSON Operations

\`\`\`python
"""
JSON Operators and Extraction
"""

# Operator reference:
# -> extract JSON object field (returns JSON)
# ->> extract JSON object field as text
# #> extract nested path (returns JSON)
# #>> extract nested path as text
# @> contains JSON object
# <@ is contained by
# ? key exists
# ?| any key exists
# ?& all keys exist

# Extract top-level field as JSON
stmt = select(Product.metadata['brand'])
# SQL: SELECT metadata->'brand' FROM products

# Extract as text (for comparison)
stmt = select(Product).where(
    Product.metadata['brand'].astext == 'Apple'
)
# SQL: SELECT * FROM products WHERE metadata->>'brand' = 'Apple'

# Extract nested field
stmt = select(Product).where(
    Product.metadata['specs']['cpu'].astext == 'M1'
)
# SQL: SELECT * FROM products WHERE metadata->'specs'->>'cpu' = 'M1'

# Alternative nested syntax (cleaner)
stmt = select(Product).where(
    Product.metadata['specs', 'cpu'].astext == 'M1'
)
\`\`\`

### JSON Containment and Existence

\`\`\`python
"""
JSON Containment Queries (@> operator)
"""

# Contains exact object
stmt = select(Product).where(
    Product.metadata.contains({'brand': 'Apple'})
)
# SQL: SELECT * FROM products WHERE metadata @> '{"brand": "Apple"}'

# Contains nested object
stmt = select(Product).where(
    Product.metadata.contains({'specs': {'cpu': 'M1'}})
)

# Is contained by (inverse)
stmt = select(Product).where(
    Product.metadata.contained_by({'brand': 'Apple', 'year': 2023, 'extra': 'data'})
)
# Matches if product metadata is subset of provided JSON

# Key exists
stmt = select(Product).where(
    Product.metadata.has_key('warranty')
)
# SQL: SELECT * FROM products WHERE metadata ? 'warranty'

# Any of these keys exist
stmt = select(Product).where(
    Product.metadata.has_any(['warranty', 'guarantee'])
)
# SQL: SELECT * FROM products WHERE metadata ?| ARRAY['warranty', 'guarantee']

# All of these keys exist
stmt = select(Product).where(
    Product.metadata.has_all(['brand', 'price', 'sku'])
)
# SQL: SELECT * FROM products WHERE metadata ?& ARRAY['brand', 'price', 'sku']
\`\`\`

### JSON Array Operations

\`\`\`python
"""
Querying JSON Arrays
"""

class Post(Base):
    __tablename__ = 'posts'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    data: Mapped[dict] = mapped_column(JSONB)
    # Example: {"title": "...", "tags": ["python", "sql"], "stats": {"views": 100}}

# Array contains element
stmt = select(Post).where(
    Post.data['tags'].contains(['python'])
)
# SQL: SELECT * FROM posts WHERE data->'tags' @> '["python"]'

# Array contains any of these elements
stmt = select(Post).where(
    Post.data['tags'].astext.contains('python')
)

# Length of JSON array
from sqlalchemy import func
stmt = select(Post).where(
    func.jsonb_array_length(Post.data['tags']) > 3
)

# Extract array element by index (0-based)
stmt = select(Post).where(
    Post.data['tags'][0].astext == 'python'
)
\`\`\`

### JSON Indexing for Performance

\`\`\`python
"""
Index Strategies for JSONB
"""

from sqlalchemy import Index
from sqlalchemy.dialects.postgresql import JSONB

class Product(Base):
    __tablename__ = 'products'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    metadata: Mapped[dict] = mapped_column(JSONB)
    
    __table_args__ = (
        # GIN index on entire JSONB column (supports @>, ?, ?|, ?&)
        Index('ix_product_metadata_gin', 'metadata', postgresql_using='gin'),
        
        # GIN index with jsonb_path_ops (faster, but only supports @>)
        Index('ix_product_metadata_path', 'metadata', 
              postgresql_using='gin', postgresql_ops={'metadata': 'jsonb_path_ops'}),
        
        # B-tree index on specific JSON field (for = queries)
        Index('ix_product_brand', (metadata['brand'].astext)),
        
        # Expression index on nested field
        Index('ix_product_cpu', (metadata['specs']['cpu'].astext)),
    )

# Index choice:
# - GIN default: Use for mixed queries (@>, ?, etc)
# - GIN jsonb_path_ops: Use for containment (@>) only, 30% smaller, faster
# - B-tree expression: Use for specific field equality/range queries
\`\`\`

---

## Array Operations (PostgreSQL)

### Array Column Type

\`\`\`python
"""
PostgreSQL Array Type
"""

from sqlalchemy.dialects.postgresql import ARRAY

class Post(Base):
    __tablename__ = 'posts'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    tags: Mapped[list[str]] = mapped_column(ARRAY(String))
    view_counts: Mapped[list[int]] = mapped_column(ARRAY(Integer))
    
    # Multi-dimensional array
    matrix: Mapped[list[list[int]]] = mapped_column(ARRAY(Integer, dimensions=2))

# Create with arrays
post = Post(
    tags=['python', 'sqlalchemy', 'tutorial'],
    view_counts=[100, 200, 150]
)
\`\`\`

### Array Containment

\`\`\`python
"""
Array Containment and Overlap
"""

# Array contains all elements (uses @> operator)
stmt = select(Post).where(
    Post.tags.contains(['python'])
)
# SQL: SELECT * FROM posts WHERE tags @> ARRAY['python']

# Array contains multiple elements
stmt = select(Post).where(
    Post.tags.contains(['python', 'sql'])
)
# Finds posts with BOTH 'python' AND 'sql'

# Array is contained by (inverse)
stmt = select(Post).where(
    Post.tags.contained_by(['python', 'sql', 'database'])
)
# Finds posts where ALL tags are in the provided list

# Array overlap (shares at least one element)
stmt = select(Post).where(
    Post.tags.overlap(['python', 'java', 'go'])
)
# SQL: SELECT * FROM posts WHERE tags && ARRAY['python', 'java', 'go']
# Finds posts with python OR java OR go
\`\`\`

### Array Element Queries

\`\`\`python
"""
Querying Array Elements
"""

from sqlalchemy import any_, all_

# ANY: At least one element matches
stmt = select(Post).where(
    any_(Post.tags) == 'python'
)
# SQL: SELECT * FROM posts WHERE 'python' = ANY(tags)

# ANY with comparison
stmt = select(Post).where(
    any_(Post.view_counts) > 1000
)
# At least one view count > 1000

# ALL: All elements match
stmt = select(Post).where(
    all_(Post.view_counts) > 100
)
# SQL: SELECT * FROM posts WHERE 100 < ALL(view_counts)
# All view counts > 100

# Array length
from sqlalchemy import func
stmt = select(Post).where(
    func.array_length(Post.tags, 1) > 5
)
# Posts with more than 5 tags

# Array index access (1-based in PostgreSQL!)
stmt = select(Post).where(
    Post.tags[1] == 'python'
)
# SQL: SELECT * FROM posts WHERE tags[1] = 'python'
# First element (PostgreSQL arrays are 1-indexed)
\`\`\`

### Array Aggregations

\`\`\`python
"""
Array Aggregate Functions
"""

from sqlalchemy import func

# Concatenate arrays
stmt = select(
    Post.user_id,
    func.array_agg(Post.tags).label('all_tags')
).group_by(Post.user_id)
# Aggregates all tags for each user

# Remove duplicates
stmt = select(
    func.array_agg (func.distinct (any_(Post.tags))).label('unique_tags')
)

# Unnest array (expand to rows)
stmt = select(
    Post.id,
    func.unnest(Post.tags).label('tag')
)
# One row per tag

# Count array elements across all rows
stmt = select(
    func.count (func.unnest(Post.tags))
)
\`\`\`

### Array Indexing

\`\`\`python
"""
Indexing Array Columns
"""

from sqlalchemy import Index

class Post(Base):
    __tablename__ = 'posts'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    tags: Mapped[list[str]] = mapped_column(ARRAY(String))
    
    __table_args__ = (
        # GIN index for array containment (@>, &&)
        Index('ix_post_tags_gin', 'tags', postgresql_using='gin'),
        
        # GiST index (alternative, supports different operators)
        Index('ix_post_tags_gist', 'tags', postgresql_using='gist'),
    )

# GIN vs GiST for arrays:
# GIN: Faster lookups, slower updates, larger index
# GiST: Faster updates, slower lookups, smaller index
# Default: Use GIN for read-heavy, GiST for write-heavy
\`\`\`

---

## Full-Text Search

### Basic Full-Text Search

\`\`\`python
"""
PostgreSQL Full-Text Search with tsvector
"""

from sqlalchemy import Text, Index, func
from sqlalchemy.dialects.postgresql import TSVECTOR

class Article(Base):
    __tablename__ = 'articles'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(Text)
    
    # tsvector column for full-text search
    search_vector: Mapped[str] = mapped_column(TSVECTOR)
    
    __table_args__ = (
        # GIN index on tsvector (essential for performance)
        Index('ix_article_search', 'search_vector', postgresql_using='gin'),
    )

# Create trigger to auto-update search_vector
\`\`\`

Migration:
\`\`\`sql
-- Create trigger function
CREATE FUNCTION articles_search_trigger() RETURNS trigger AS $$
BEGIN
    NEW.search_vector :=
        setweight (to_tsvector('english', coalesce(NEW.title, '')), 'A') ||
        setweight (to_tsvector('english', coalesce(NEW.content, '')), 'B');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger
CREATE TRIGGER tsvectorupdate
BEFORE INSERT OR UPDATE ON articles
FOR EACH ROW
EXECUTE FUNCTION articles_search_trigger();
\`\`\`

### Full-Text Queries

\`\`\`python
"""
Full-Text Search Queries
"""

from sqlalchemy import func

# Simple search (match any word)
stmt = select(Article).where(
    Article.search_vector.match('python database')
)
# SQL: SELECT * FROM articles WHERE search_vector @@ to_tsquery('python | database')

# Phrase search
stmt = select(Article).where(
    Article.search_vector.match('python & database')
)
# Requires BOTH words

# NOT operator
stmt = select(Article).where(
    Article.search_vector.match('python & !java')
)
# Python but not Java

# OR operator (default)
stmt = select(Article).where(
    Article.search_vector.match('python | java | go')
)

# Phrase proximity
stmt = select(Article).where(
    Article.search_vector.match('python <-> tutorial')
)
# Words adjacent: "python tutorial"

stmt = select(Article).where(
    Article.search_vector.match('python <2> tutorial')
)
# Within 2 words: "python and tutorial"
\`\`\`

### Ranking and Relevance

\`\`\`python
"""
Rank Search Results by Relevance
"""

from sqlalchemy import func

# Basic ranking
stmt = select(
    Article,
    func.ts_rank(Article.search_vector, func.to_tsquery('python')).label('rank')
).where(
    Article.search_vector.match('python')
).order_by('rank DESC')

# Weighted ranking (title more important than content)
stmt = select(
    Article,
    func.ts_rank_cd(
        Article.search_vector,
        func.to_tsquery('python'),
        32  # Normalization flag
    ).label('rank')
).where(
    Article.search_vector.match('python')
).order_by('rank DESC')

# Custom weight calculation
stmt = select(
    Article,
    (
        func.ts_rank(Article.search_vector, func.to_tsquery('python')) * 10 +
        func.length(Article.title) * 0.1
    ).label('custom_rank')
).where(
    Article.search_vector.match('python')
).order_by('custom_rank DESC')

# Headline snippet (show matching context)
stmt = select(
    Article.title,
    func.ts_headline(
        'english',
        Article.content,
        func.to_tsquery('python'),
        'MaxWords=50, MinWords=20'
    ).label('snippet')
).where(
    Article.search_vector.match('python')
)
\`\`\`

### Multi-Language Search

\`\`\`python
"""
Multi-Language Full-Text Search
"""

class Article(Base):
    __tablename__ = 'articles'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(String(10))  # 'en', 'es', 'fr'
    search_vector: Mapped[str] = mapped_column(TSVECTOR)

# Update trigger with dynamic language
\`\`\`

\`\`\`sql
CREATE FUNCTION articles_search_trigger() RETURNS trigger AS $$
BEGIN
    NEW.search_vector := to_tsvector(NEW.language::regconfig, coalesce(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
\`\`\`

---

## Pattern Matching and Similarity

### LIKE and ILIKE

\`\`\`python
"""
Pattern Matching
"""

# LIKE (case-sensitive)
stmt = select(User).where(User.email.like('%@gmail.com'))

# ILIKE (case-insensitive, PostgreSQL)
stmt = select(User).where(User.email.ilike('%@GMAIL.COM'))

# Multiple patterns (OR)
from sqlalchemy import or_
stmt = select(User).where(
    or_(
        User.email.like('%@gmail.com'),
        User.email.like('%@yahoo.com')
    )
)

# NOT LIKE
stmt = select(User).where(User.email.notlike('%@spam.com'))

# Wildcards:
# % - any number of characters
# _ - single character
stmt = select(User).where(User.phone.like('555-____'))  # 555-1234
\`\`\`

### Regular Expressions

\`\`\`python
"""
PostgreSQL Regular Expression Matching
"""

from sqlalchemy import func

# Regex match (case-sensitive)
stmt = select(User).where(
    User.email.op('~')(r'^[a-z]+@example\.com$')
)

# Regex match (case-insensitive)
stmt = select(User).where(
    User.email.op('~*')(r'^[A-Z]+@EXAMPLE\.COM$')
)

# Regex NOT match
stmt = select(User).where(
    User.email.op('!~')(r'.*spam.*')
)

# Extract with regex
stmt = select(
    User.email,
    func.regexp_match(User.email, r'(.+)@(.+)').label('parts')
)
# Returns array: ['user', 'domain.com']

# Replace with regex
stmt = select(
    func.regexp_replace(User.phone, r'\D', '', 'g').label('digits_only')
)
# Remove all non-digits: (555) 123-4567 → 5551234567
\`\`\`

### Similarity Search (pg_trgm)

\`\`\`python
"""
Fuzzy String Matching with Trigrams
"""

# Enable extension first
# CREATE EXTENSION IF NOT EXISTS pg_trgm;

from sqlalchemy import Index

class User(Base):
    __tablename__ = 'users'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    email: Mapped[str] = mapped_column(String(255))
    
    __table_args__ = (
        # GIN index for similarity search
        Index('ix_user_email_trgm', 'email', postgresql_using='gin',
              postgresql_ops={'email': 'gin_trgm_ops'}),
    )

# Similarity operator
stmt = select(User).where(
    func.similarity(User.email, 'test@example.com') > 0.3
).order_by(
    func.similarity(User.email, 'test@example.com').desc()
)

# Alternative: % operator (PostgreSQL)
stmt = select(User).where(
    User.email.op('%')('test@example.com')
)

# Word similarity (PostgreSQL 9.6+)
stmt = select(User).where(
    func.word_similarity('test', User.email) > 0.5
)
\`\`\`

---

## Custom Operators and Functions

### Custom SQL Functions

\`\`\`python
"""
Define Custom SQL Functions
"""

from sqlalchemy.sql import func as sqlfunc
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import FunctionElement

# Custom function class
class levenshtein(FunctionElement):
    name = 'levenshtein'
    type = Integer()

@compiles (levenshtein)
def compile_levenshtein (element, compiler, **kwargs):
    arg1, arg2 = list (element.clauses)
    return "levenshtein(%s, %s)" % (
        compiler.process (arg1),
        compiler.process (arg2)
    )

# Usage
stmt = select(User).where(
    levenshtein(User.username, 'john') < 3
).order_by (levenshtein(User.username, 'john'))
# Find usernames within edit distance 3 of 'john'
\`\`\`

### Custom Binary Operators

\`\`\`python
"""
Define Custom Operators
"""

from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import BinaryExpression

# Similarity operator (%%)
class similar_to(BinaryExpression):
    pass

@compiles (similar_to)
def compile_similar (element, compiler, **kwargs):
    return "%s %% %s" % (
        compiler.process (element.left),
        compiler.process (element.right)
    )

# Usage
stmt = select(User).where(
    similar_to(User.email, 'test@example.com')
)
\`\`\`

---

## Hybrid Properties

### Basic Hybrid Properties

\`\`\`python
"""
Computed Columns Queryable at SQL Level
"""

from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method

class User(Base):
    __tablename__ = 'users'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    first_name: Mapped[str] = mapped_column(String(50))
    last_name: Mapped[str] = mapped_column(String(50))
    email: Mapped[str] = mapped_column(String(255))
    
    @hybrid_property
    def full_name (self):
        """Python-level: property access"""
        return f"{self.first_name} {self.last_name}"
    
    @full_name.expression
    def full_name (cls):
        """SQL-level: used in queries"""
        return func.concat (cls.first_name, ' ', cls.last_name)
    
    @full_name.setter
    def full_name (self, value):
        """Setter for Python-level"""
        self.first_name, self.last_name = value.split(' ', 1)

# Python usage
user = session.get(User, 1)
print(user.full_name)  # "John Doe"
user.full_name = "Jane Smith"  # Sets first_name and last_name

# SQL usage
stmt = select(User).where(User.full_name == 'John Doe')
# SQL: SELECT * FROM users WHERE concat (first_name, ' ', last_name) = 'John Doe'

stmt = select(User).order_by(User.full_name)
# SQL: SELECT * FROM users ORDER BY concat (first_name, ' ', last_name)
\`\`\`

### Hybrid Methods

\`\`\`python
"""
Hybrid Methods with Parameters
"""

from datetime import datetime, timedelta

class Post(Base):
    __tablename__ = 'posts'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    
    @hybrid_method
    def created_within (self, days):
        """Python-level: instance method"""
        return datetime.utcnow() - self.created_at < timedelta (days=days)
    
    @created_within.expression
    def created_within (cls, days):
        """SQL-level: class method"""
        return cls.created_at > func.now() - timedelta (days=days)

# Python usage
post = session.get(Post, 1)
if post.created_within(7):
    print("Recent post")

# SQL usage
stmt = select(Post).where(Post.created_within(7))
# SQL: SELECT * FROM posts WHERE created_at > (NOW() - INTERVAL '7 days')
\`\`\`

### Complex Hybrid Properties

\`\`\`python
"""
Hybrid Properties with Complex Logic
"""

class Order(Base):
    __tablename__ = 'orders'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    subtotal: Mapped[float] = mapped_column(Numeric(10, 2))
    tax: Mapped[float] = mapped_column(Numeric(10, 2))
    shipping: Mapped[float] = mapped_column(Numeric(10, 2))
    discount_percent: Mapped[float] = mapped_column(Numeric(5, 2), default=0)
    
    @hybrid_property
    def total (self):
        """Computed total with discount"""
        discount_amount = self.subtotal * (self.discount_percent / 100)
        return self.subtotal - discount_amount + self.tax + self.shipping
    
    @total.expression
    def total (cls):
        """SQL expression for total"""
        discount_amount = cls.subtotal * (cls.discount_percent / 100)
        return cls.subtotal - discount_amount + cls.tax + cls.shipping
    
    @hybrid_property
    def is_high_value (self):
        """Boolean: order total > $1000"""
        return self.total > 1000
    
    @is_high_value.expression
    def is_high_value (cls):
        return cls.total > 1000

# Query high-value orders
stmt = select(Order).where(Order.is_high_value)

# Order by total
stmt = select(Order).order_by(Order.total.desc())

# Aggregate on hybrid property
stmt = select (func.avg(Order.total))
\`\`\`

---

## Type Casting and Coercion

\`\`\`python
"""
Type Casting in Queries
"""

from sqlalchemy import cast, type_coerce, String, Integer, Date

# CAST: Explicit type conversion at SQL level
stmt = select(
    cast(User.created_at, Date).label('date')
).group_by (cast(User.created_at, Date))
# SQL: SELECT CAST(created_at AS DATE) FROM users GROUP BY CAST(created_at AS DATE)

# type_coerce: Type hint to SQLAlchemy (no SQL CAST)
stmt = select(
    type_coerce(User.metadata['age'], Integer)
).where(
    type_coerce(User.metadata['age'], Integer) > 18
)
# Treats JSON field as Integer for comparison

# Cast for string operations on non-string columns
stmt = select(User).where(
    cast(User.id, String).like('123%')
)
# Find IDs starting with '123'

# Cast for date operations
stmt = select(
    cast (func.extract('year', User.created_at), Integer).label('year'),
    func.count(User.id)
).group_by('year')
\`\`\`

---

## Database-Specific Features

### PostgreSQL-Specific

\`\`\`python
"""
PostgreSQL Advanced Features
"""

from sqlalchemy.dialects.postgresql import (
    ARRAY, JSONB, HSTORE, UUID, INET, CIDR, MACADDR, ENUM as PG_ENUM
)

class Connection(Base):
    __tablename__ = 'connections'
    
    id: Mapped[int] = mapped_column (primary_key=True)
    
    # Network types
    ip_address: Mapped[str] = mapped_column(INET)  # IPv4/IPv6
    ip_range: Mapped[str] = mapped_column(CIDR)    # Network range
    mac_address: Mapped[str] = mapped_column(MACADDR)
    
    # Key-value store
    properties: Mapped[dict] = mapped_column(HSTORE)

# Network queries
stmt = select(Connection).where(
    Connection.ip_address.op('<<')(func.cast('192.168.1.0/24', CIDR))
)
# IPs within subnet

# HSTORE queries
stmt = select(Connection).where(
    Connection.properties['status'] == 'active'
)
\`\`\`

---

## Summary

### Key Takeaways

✅ **JSON**: Use JSONB for querying, GIN indexes essential  
✅ **Arrays**: PostgreSQL ARRAY type with containment/overlap operators  
✅ **Full-text**: tsvector + GIN index + triggers for auto-update  
✅ **Pattern matching**: LIKE for simple, regex for complex, similarity for fuzzy  
✅ **Hybrid properties**: Computed columns usable in Python and SQL  
✅ **Custom functions**: Extend SQLAlchemy for database-specific features  
✅ **Indexes**: GIN for JSON/arrays/text, B-tree for equality

### Performance Tips

✅ Always index JSONB columns with GIN  
✅ Use expression indexes for frequently queried JSON fields  
✅ Full-text search requires tsvector + GIN index  
✅ Similarity search (pg_trgm) needs GIN trigram index  
✅ Test query performance with EXPLAIN ANALYZE

### Best Practices

✅ Prefer JSONB over JSON in PostgreSQL  
✅ Use jsonb_path_ops for containment-only queries (30% smaller index)  
✅ Create triggers to auto-update tsvector columns  
✅ Hybrid properties: Keep SQL expression equivalent to Python logic  
✅ Custom functions: Document behavior and test thoroughly

### Next Steps

In the next section, we'll explore **Relationship Loading Strategies**: solving the N+1 query problem, eager loading patterns, and performance optimization for relationships.
`,
};
