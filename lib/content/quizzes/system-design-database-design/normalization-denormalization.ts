/**
 * Quiz questions for Database Normalization & Denormalization section
 */

export const normalizationdenormalizationQuiz = [
  {
    id: 'norm-disc-1',
    question:
      'Design a database schema for a blogging platform that supports posts, comments, tags, and user profiles. First, create a fully normalized (3NF) schema. Then, identify specific denormalization strategies you would apply for performance, explaining the trade-offs.',
    sampleAnswer: `Complete schema design with normalization and denormalization strategies:

**Step 1: Fully Normalized Schema (3NF)**

\`\`\`sql
-- Users
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    bio TEXT,
    avatar_url TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Posts
CREATE TABLE posts (
    post_id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    published_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);

-- Comments
CREATE TABLE comments (
    comment_id SERIAL PRIMARY KEY,
    post_id INT NOT NULL,
    user_id INT NOT NULL,
    parent_comment_id INT,  -- for nested comments
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (post_id) REFERENCES posts (post_id),
    FOREIGN KEY (user_id) REFERENCES users (user_id),
    FOREIGN KEY (parent_comment_id) REFERENCES comments (comment_id)
);

-- Tags
CREATE TABLE tags (
    tag_id SERIAL PRIMARY KEY,
    tag_name VARCHAR(50) UNIQUE NOT NULL
);

-- Post-Tag relationship (many-to-many)
CREATE TABLE post_tags (
    post_id INT,
    tag_id INT,
    PRIMARY KEY (post_id, tag_id),
    FOREIGN KEY (post_id) REFERENCES posts (post_id),
    FOREIGN KEY (tag_id) REFERENCES tags (tag_id)
);

-- Likes
CREATE TABLE post_likes (
    like_id SERIAL PRIMARY KEY,
    post_id INT NOT NULL,
    user_id INT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (post_id, user_id),
    FOREIGN KEY (post_id) REFERENCES posts (post_id),
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);
\`\`\`

**Benefits of Normalized Design:**
- Single source of truth for user data
- Easy to update username (one UPDATE query)
- Referential integrity enforced
- No data duplication

**Performance Problems:**
- Displaying posts requires 3-4 JOINs:
  - posts → users (get author name/avatar)
  - posts → post_likes (get like count)
  - posts → comments (get comment count)
  - posts → post_tags → tags (get tag names)
- Complex query, slow (100-500ms for homepage)

**Step 2: Denormalization Strategies**

**Strategy 1: Duplicate Frequently Accessed Columns**

\`\`\`sql
-- Add author info to posts table
ALTER TABLE posts ADD COLUMN author_username VARCHAR(50);
ALTER TABLE posts ADD COLUMN author_avatar_url TEXT;

-- Trigger to keep in sync
CREATE OR REPLACE FUNCTION update_post_author()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE posts 
    SET author_username = NEW.username,
        author_avatar_url = NEW.avatar_url
    WHERE user_id = NEW.user_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER user_update_trigger
AFTER UPDATE ON users
FOR EACH ROW
WHEN (OLD.username IS DISTINCT FROM NEW.username 
      OR OLD.avatar_url IS DISTINCT FROM NEW.avatar_url)
EXECUTE FUNCTION update_post_author();

-- Similarly for comments
ALTER TABLE comments ADD COLUMN commenter_username VARCHAR(50);
ALTER TABLE comments ADD COLUMN commenter_avatar_url TEXT;
\`\`\`

*Benefits:*
- Display posts without JOIN to users table
- 50-100ms query time improvement

*Trade-offs:*
- If user changes username, must update all their posts/comments
- Trigger adds complexity
- Extra storage (~100 bytes per post/comment)

*Acceptable because:*
- Usernames change rarely
- Reading posts is 1000x more common than updating usernames
- Can rebuild from normalized users table if inconsistency occurs

**Strategy 2: Pre-compute Aggregations**

\`\`\`sql
-- Add denormalized counts to posts
ALTER TABLE posts ADD COLUMN like_count INT DEFAULT 0;
ALTER TABLE posts ADD COLUMN comment_count INT DEFAULT 0;

-- Triggers to maintain counts
CREATE OR REPLACE FUNCTION update_like_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE posts SET like_count = like_count + 1 WHERE post_id = NEW.post_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE posts SET like_count = like_count - 1 WHERE post_id = OLD.post_id;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER like_count_trigger
AFTER INSERT OR DELETE ON post_likes
FOR EACH ROW EXECUTE FUNCTION update_like_count();

-- Similar trigger for comment_count
\`\`\`

*Benefits:*
- No need to COUNT(*) on every post display
- 10-50ms query improvement

*Trade-offs:*
- Writes are slower (trigger overhead)
- Potential inconsistency (can drift due to bugs/failed transactions)

*Mitigation:*
- Background job to verify counts:
\`\`\`sql
-- Find posts with incorrect counts
SELECT p.post_id, p.like_count, COUNT(pl.like_id) as actual_count
FROM posts p
LEFT JOIN post_likes pl ON p.post_id = pl.post_id
GROUP BY p.post_id, p.like_count
HAVING p.like_count != COUNT(pl.like_id);
\`\`\`

**Strategy 3: Array Column for Tags**

\`\`\`sql
-- PostgreSQL array column
ALTER TABLE posts ADD COLUMN tag_names TEXT[];

-- Update tags on post creation/edit
UPDATE posts 
SET tag_names = ARRAY(
    SELECT t.tag_name 
    FROM post_tags pt 
    JOIN tags t ON pt.tag_id = t.tag_id 
    WHERE pt.post_id = posts.post_id
);

-- Query posts by tag (GIN index)
CREATE INDEX idx_posts_tags ON posts USING GIN(tag_names);
SELECT * FROM posts WHERE tag_names @> ARRAY['postgresql',];
\`\`\`

*Benefits:*
- No JOIN to post_tags and tags tables
- Fast tag filtering with GIN index

*Trade-offs:*
- Redundant storage (tags in both post_tags and tag_names)
- Must update array when tags change

**Strategy 4: Materialized View for Homepage**

\`\`\`sql
CREATE MATERIALIZED VIEW homepage_posts AS
SELECT 
    p.post_id,
    p.title,
    p.content,
    p.published_at,
    u.username as author_username,
    u.avatar_url as author_avatar_url,
    COUNT(DISTINCT pl.like_id) as like_count,
    COUNT(DISTINCT c.comment_id) as comment_count,
    ARRAY_AGG(DISTINCT t.tag_name) as tags
FROM posts p
JOIN users u ON p.user_id = u.user_id
LEFT JOIN post_likes pl ON p.post_id = pl.post_id
LEFT JOIN comments c ON p.post_id = c.post_id
LEFT JOIN post_tags pt ON p.post_id = pt.post_id
LEFT JOIN tags t ON pt.tag_id = t.tag_id
WHERE p.published_at > NOW() - INTERVAL '30 days'  -- Recent posts only
GROUP BY p.post_id, p.title, p.content, p.published_at, u.username, u.avatar_url
ORDER BY p.published_at DESC;

CREATE INDEX idx_homepage_posts_published ON homepage_posts (published_at DESC);

-- Refresh every 5 minutes
REFRESH MATERIALIZED VIEW CONCURRENTLY homepage_posts;
\`\`\`

*Benefits:*
- Ultra-fast homepage queries (<5ms)
- No complex JOINs at query time
- All data pre-computed

*Trade-offs:*
- Data can be up to 5 minutes stale
- Refresh operation overhead
- Extra storage for materialized view

**Step 3: Final Hybrid Architecture**

*Write Path (Normalized):*
- All writes go to normalized tables (users, posts, comments, post_likes, tags, post_tags)
- Source of truth
- ACID guarantees

*Read Path (Denormalized):*
- Homepage: Query materialized view
- Individual post: Query posts table with denormalized columns (author info, counts)
- Comments: Query comments table with denormalized commenter info
- Search: Replicate to Elasticsearch with all denormalized fields

**Performance Comparison:**

| Query | Normalized | Denormalized | Improvement |
|-------|-----------|--------------|-------------|
| Homepage | 500ms | 5ms | 100x faster |
| Single Post | 100ms | 10ms | 10x faster |
| Comment List | 50ms | 5ms | 10x faster |

**Storage Impact:**

| Component | Size |
|-----------|------|
| Normalized tables | 10GB |
| Denormalized columns | +1GB (10% overhead) |
| Materialized view | +500MB |
| Total | 11.5GB (15% overhead) |

**Acceptable trade-off for 10-100x query performance improvement.**`,
    keyPoints: [
      'Start with normalized schema for data integrity',
      'Denormalize selectively based on measured query patterns',
      'Common strategies: duplicate count columns, embed relationships, materialized views',
      'Trade-offs: 15% storage increase, write complexity, 10-100x read speedup',
      'Monitor and maintain consistency with triggers or application logic',
    ],
  },
  {
    id: 'norm-disc-2',
    question:
      'Explain the CQRS (Command Query Responsibility Segregation) pattern. How does it relate to normalization/denormalization? Design a CQRS architecture for an e-commerce order system, detailing the write model, read models, and synchronization strategy.',
    sampleAnswer: `Comprehensive CQRS architecture for e-commerce:

**CQRS Fundamentals:**

CQRS separates the write model (commands) from the read model (queries):

- **Write Model:** Optimized for data integrity, validation, business logic
  - Typically normalized (3NF)
  - Handles commands: CreateOrder, UpdateInventory, CancelOrder
  - Source of truth
  
- **Read Model:** Optimized for query performance
  - Typically denormalized
  - Handles queries: GetOrderDetails, SearchProducts, UserOrderHistory
  - Eventually consistent with write model

**Connection to Normalization:**
- Write model = Normalized (enforce consistency, avoid anomalies)
- Read model = Denormalized (optimize for queries, reduce JOINs)
- Events bridge the gap (eventual consistency)

**E-Commerce CQRS Architecture:**

**Write Model (Normalized - PostgreSQL)**

\`\`\`sql
-- Customers
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE,
    name VARCHAR(255),
    shipping_address JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Products
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    sku VARCHAR(100) UNIQUE,
    name VARCHAR(255),
    description TEXT,
    price DECIMAL(10,2),
    category_id INT
);

-- Inventory (separate for concurrency control)
CREATE TABLE inventory (
    product_id INT PRIMARY KEY,
    quantity INT NOT NULL CHECK (quantity >= 0),
    reserved INT DEFAULT 0,
    FOREIGN KEY (product_id) REFERENCES products (product_id)
);

-- Orders
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INT NOT NULL,
    order_status VARCHAR(50) NOT NULL,  -- pending, confirmed, shipped, delivered
    total_amount DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);

-- Order Items
CREATE TABLE order_items (
    order_item_id SERIAL PRIMARY KEY,
    order_id INT NOT NULL,
    product_id INT NOT NULL,
    quantity INT NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders (order_id),
    FOREIGN KEY (product_id) REFERENCES products (product_id)
);

-- Payments
CREATE TABLE payments (
    payment_id SERIAL PRIMARY KEY,
    order_id INT NOT NULL,
    amount DECIMAL(10,2),
    payment_method VARCHAR(50),
    status VARCHAR(50),  -- pending, completed, failed
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (order_id) REFERENCES orders (order_id)
);

-- Event Store (for CQRS)
CREATE TABLE domain_events (
    event_id BIGSERIAL PRIMARY KEY,
    aggregate_id INT NOT NULL,     -- order_id, customer_id, etc.
    aggregate_type VARCHAR(50),    -- 'Order', 'Product', etc.
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    processed BOOLEAN DEFAULT FALSE
);
CREATE INDEX idx_events_unprocessed ON domain_events (processed, created_at) WHERE NOT processed;
\`\`\`

**Write Model Business Logic:**

\`\`\`python
# Command: CreateOrder
def create_order (customer_id, items):
    # Start transaction
    with transaction():
        # 1. Validate customer exists
        customer = customers.get (customer_id)
        
        # 2. Validate product availability
        for item in items:
            inventory = inventory_table.get (item.product_id)
            if inventory.quantity < item.quantity:
                raise InsufficientInventoryError()
        
        # 3. Reserve inventory
        for item in items:
            inventory_table.update(
                product_id=item.product_id,
                reserved=reserved + item.quantity
            )
        
        # 4. Create order
        order = orders.create(
            customer_id=customer_id,
            status='pending',
            total_amount=calculate_total (items)
        )
        
        # 5. Create order items
        for item in items:
            order_items.create(
                order_id=order.order_id,
                product_id=item.product_id,
                quantity=item.quantity,
                unit_price=get_current_price (item.product_id)
            )
        
        # 6. Emit event
        domain_events.create(
            aggregate_id=order.order_id,
            aggregate_type='Order',
            event_type='OrderCreated',
            event_data={
                'order_id': order.order_id,
                'customer_id': customer_id,
                'items': items,
                'total_amount': order.total_amount
            }
        )
        
        return order.order_id
\`\`\`

**Read Model 1: Order Summary (Denormalized - PostgreSQL)**

\`\`\`sql
-- Fully denormalized table for fast order lookups
CREATE TABLE order_summary (
    order_id INT PRIMARY KEY,
    
    -- Customer info (denormalized)
    customer_id INT,
    customer_name VARCHAR(255),
    customer_email VARCHAR(255),
    shipping_address JSONB,
    
    -- Order info
    order_status VARCHAR(50),
    total_amount DECIMAL(10,2),
    item_count INT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    
    -- Items (denormalized as JSON array)
    items JSONB,  -- [{ product_id, name, quantity, price }]
    
    -- Payment info (denormalized)
    payment_status VARCHAR(50),
    payment_method VARCHAR(50)
);

-- Indexes for common queries
CREATE INDEX idx_order_summary_customer ON order_summary (customer_id, created_at DESC);
CREATE INDEX idx_order_summary_status ON order_summary (order_status, created_at DESC);
CREATE INDEX idx_order_summary_created ON order_summary (created_at DESC);
\`\`\`

**Read Model 2: User Order History (Redis Cache)**

\`\`\`
Key: "user:orders:{customer_id}"
Value: Sorted Set (by timestamp)
[
  {
    score: 1704067200,  // timestamp
    value: {
      "order_id": 12345,
      "total": 99.99,
      "status": "delivered",
      "item_count": 3,
      "created_at": "2024-01-01T00:00:00Z"
    }
  },
  ...
]

TTL: 1 hour
\`\`\`

**Read Model 3: Product Catalog (Elasticsearch)**

\`\`\`json
{
  "product_id": 123,
  "sku": "LAPTOP-001",
  "name": "Gaming Laptop",
  "description": "High-performance laptop...",
  "price": 1299.99,
  "category": "Electronics > Computers",
  
  // Denormalized fields
  "inventory_available": 45,
  "avg_rating": 4.7,
  "review_count": 342,
  "total_sales": 1523,
  "trending_score": 0.85
}
\`\`\`

**Synchronization Strategy: Event Processor**

\`\`\`python
# Background worker: Process domain events
def event_processor():
    while True:
        # Poll for unprocessed events
        events = domain_events.query (processed=False, limit=100)
        
        for event in events:
            try:
                # Route to appropriate handler
                if event.event_type == 'OrderCreated':
                    handle_order_created (event)
                elif event.event_type == 'OrderShipped':
                    handle_order_shipped (event)
                elif event.event_type == 'PaymentCompleted':
                    handle_payment_completed (event)
                
                # Mark as processed
                domain_events.update (event_id=event.event_id, processed=True)
            except Exception as e:
                # Log error, implement retry logic
                log_error (event, e)
        
        sleep(1)  # Poll every second

def handle_order_created (event):
    data = event.event_data
    
    # 1. Update PostgreSQL read model
    order_summary.insert({
        'order_id': data['order_id',],
        'customer_id': data['customer_id',],
        'customer_name': get_customer_name (data['customer_id',]),
        'customer_email': get_customer_email (data['customer_id',]),
        'order_status': 'pending',
        'total_amount': data['total_amount',],
        'item_count': len (data['items',]),
        'items': serialize_items (data['items',]),
        'created_at': event.created_at
    })
    
    # 2. Update Redis cache
    redis.zadd(
        f"user:orders:{data['customer_id',]}",
        {
            json.dumps({
                'order_id': data['order_id',],
                'total': data['total_amount',],
                'status': 'pending',
                'item_count': len (data['items',])
            }): event.created_at.timestamp()
        }
    )
    
    # 3. Update product sales count (Elasticsearch)
    for item in data['items',]:
        elasticsearch.update(
            index='products',
            id=item['product_id',],
            script={
                'source': 'ctx._source.total_sales += params.quantity',
                'params': {'quantity': item['quantity',]}
            }
        )
\`\`\`

**Query Patterns:**

\`\`\`python
# Fast queries from read models

# Get order details (PostgreSQL read model)
def get_order_details (order_id):
    # Single query, no JOINs, <5ms
    return order_summary.get (order_id)

# Get user's orders (Redis cache)
def get_user_orders (customer_id, limit=10):
    # Check cache first
    cached = redis.zrevrange (f"user:orders:{customer_id}", 0, limit-1)
    if cached:
        return cached
    
    # Cache miss: query database and populate cache
    orders = order_summary.query (customer_id=customer_id, limit=limit)
    for order in orders:
        redis.zadd (f"user:orders:{customer_id}", {json.dumps (order): order.created_at})
    
    return orders

# Search products (Elasticsearch)
def search_products (query, filters):
    return elasticsearch.search(
        index='products',
        query={
            'bool': {
                'must': [{'match': {'name': query}}],
                'filter': [
                    {'range': {'price': {'gte': filters.min_price, 'lte': filters.max_price}}},
                    {'term': {'category': filters.category}}
                ]
            }
        },
        sort=[{'trending_score': 'desc'}]
    )
\`\`\`

**Benefits of CQRS Architecture:**1. **Scalability:**
   - Scale write and read databases independently
   - Read replicas for order_summary
   - Redis for hot data, Elasticsearch for search

2. **Performance:**
   - Writes: 50ms (normalized, transactional)
   - Reads: 5ms (denormalized, no JOINs)

3. **Flexibility:**
   - Multiple read models optimized for different use cases
   - Add new read models without touching write model

4. **Resilience:**
   - Event store provides audit trail
   - Rebuild read models from events if corrupted

**Trade-offs:**1. **Complexity:** More components, more code
2. **Eventual Consistency:** Read models lag behind writes (typically 100-1000ms)
3. **Debugging:** Harder to trace bugs across models
4. **Storage:** Redundant data in multiple read models

**When to Use CQRS:**

✅ High scale (10k+ requests/sec)
✅ Complex domain logic
✅ Different read/write patterns
✅ Need for multiple representations of data

❌ Simple CRUD apps
❌ Low traffic
❌ Strict consistency requirements

This architecture balances data integrity (normalized write model) with query performance (denormalized read models) using event-driven synchronization.`,
    keyPoints: [
      'CQRS separates write model (normalized) from read models (denormalized)',
      'Write model is source of truth, read models are projections',
      'Event-driven synchronization keeps read models eventually consistent',
      'Enables independent scaling and optimization for reads vs writes',
      'Best for complex domains with different read/write patterns',
    ],
  },
  {
    id: 'norm-disc-3',
    question:
      'A data warehouse team wants to migrate from a normalized OLTP database to a denormalized star schema for analytics. Explain the differences between OLTP and OLAP database design, describe the star schema pattern, and walk through the migration strategy including ETL processes.',
    sampleAnswer: `Complete guide to OLTP vs OLAP and star schema migration:

**OLTP vs OLAP Database Design**

**OLTP (Online Transaction Processing):**

*Characteristics:*
- **Workload:** Many small read/write transactions
- **Queries:** Simple, predictable (get order by ID, update user)
- **Users:** Thousands to millions of concurrent users
- **Data:** Current operational data (last few months)
- **Schema:** Highly normalized (3NF) to ensure data integrity
- **Performance Goal:** Low latency (ms), high throughput

*Example: E-commerce Order System*
\`\`\`sql
-- Normalized (3NF)
customers (customer_id, name, email)
orders (order_id, customer_id, order_date, total)
order_items (order_item_id, order_id, product_id, quantity, price)
products (product_id, name, category_id, price)
categories (category_id, name, parent_category_id)
\`\`\`

*Typical Query:*
\`\`\`sql
-- Get specific order details
SELECT * FROM orders WHERE order_id = 12345;
\`\`\`

**OLAP (Online Analytical Processing):**

*Characteristics:*
- **Workload:** Complex analytical queries, aggregations
- **Queries:** Complex, unpredictable (sales by region by month, top products)
- **Users:** Tens to hundreds of analysts
- **Data:** Historical data (years)
- **Schema:** Denormalized (star/snowflake) for query performance
- **Performance Goal:** Handle complex aggregations on billions of rows

*Example: Sales Analytics*
\`\`\`sql
-- Denormalized star schema
fact_sales (sale_id, date_key, customer_key, product_key, quantity, amount)
dim_date (date_key, date, month, quarter, year)
dim_customer (customer_key, name, city, state, country)
dim_product (product_key, name, category, subcategory, brand)
\`\`\`

*Typical Query:*
\`\`\`sql
-- Aggregate sales by category and quarter
SELECT 
    p.category,
    d.quarter,
    d.year,
    SUM(f.amount) as total_sales,
    COUNT(DISTINCT f.customer_key) as unique_customers
FROM fact_sales f
JOIN dim_date d ON f.date_key = d.date_key
JOIN dim_product p ON f.product_key = p.product_key
WHERE d.year >= 2022
GROUP BY p.category, d.quarter, d.year
ORDER BY d.year, d.quarter, total_sales DESC;
\`\`\`

**Star Schema Pattern**

**Structure:**
- **Fact Table (Center):** Contains measurable events (sales, clicks, orders)
  - Foreign keys to dimension tables
  - Numeric measures (quantity, amount, count)
  - Granularity defined (one row per order line item, per day, etc.)

- **Dimension Tables (Points):** Contain descriptive attributes
  - Denormalized (all attributes in one table)
  - Slowly changing dimensions (SCD)

**Example: E-commerce Star Schema**

\`\`\`sql
-- Fact table (large: millions to billions of rows)
CREATE TABLE fact_orders (
    order_key BIGINT PRIMARY KEY,    -- Surrogate key
    
    -- Foreign keys to dimensions
    date_key INT NOT NULL,
    customer_key INT NOT NULL,
    product_key INT NOT NULL,
    store_key INT NOT NULL,
    
    -- Measures (what we want to analyze)
    quantity INT NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    total_amount DECIMAL(10,2) NOT NULL,
    discount_amount DECIMAL(10,2) NOT NULL,
    shipping_cost DECIMAL(10,2) NOT NULL,
    
    -- Degenerate dimension (stays in fact table)
    order_number VARCHAR(50),
    
    FOREIGN KEY (date_key) REFERENCES dim_date (date_key),
    FOREIGN KEY (customer_key) REFERENCES dim_customer (customer_key),
    FOREIGN KEY (product_key) REFERENCES dim_product (product_key),
    FOREIGN KEY (store_key) REFERENCES dim_store (store_key)
);

CREATE INDEX idx_fact_orders_date ON fact_orders (date_key);
CREATE INDEX idx_fact_orders_customer ON fact_orders (customer_key);
CREATE INDEX idx_fact_orders_product ON fact_orders (product_key);

-- Dimension: Date (small: 3650 rows for 10 years)
CREATE TABLE dim_date (
    date_key INT PRIMARY KEY,
    full_date DATE NOT NULL,
    day_of_week VARCHAR(10),
    day_of_month INT,
    day_of_year INT,
    week_of_year INT,
    month INT,
    month_name VARCHAR(10),
    quarter INT,
    year INT,
    is_weekend BOOLEAN,
    is_holiday BOOLEAN,
    holiday_name VARCHAR(50),
    fiscal_year INT,
    fiscal_quarter INT
);

-- Dimension: Customer (medium: thousands to millions)
CREATE TABLE dim_customer (
    customer_key INT PRIMARY KEY,  -- Surrogate key
    customer_id INT,               -- Natural key from OLTP
    customer_name VARCHAR(255),
    email VARCHAR(255),
    
    -- Denormalized geography
    address VARCHAR(500),
    city VARCHAR(100),
    state VARCHAR(50),
    postal_code VARCHAR(20),
    country VARCHAR(50),
    region VARCHAR(50),
    
    -- Demographics
    customer_segment VARCHAR(50),  -- VIP, Regular, New
    customer_since DATE,
    lifetime_value DECIMAL(12,2),
    
    -- SCD Type 2 (track history)
    effective_date DATE NOT NULL,
    expiration_date DATE,
    is_current BOOLEAN DEFAULT TRUE
);

-- Dimension: Product (medium: thousands)
CREATE TABLE dim_product (
    product_key INT PRIMARY KEY,  -- Surrogate key
    product_id INT,               -- Natural key from OLTP
    product_name VARCHAR(255),
    sku VARCHAR(100),
    
    -- Denormalized hierarchy
    category VARCHAR(100),
    subcategory VARCHAR(100),
    brand VARCHAR(100),
    manufacturer VARCHAR(255),
    
    -- Attributes
    product_color VARCHAR(50),
    product_size VARCHAR(50),
    product_weight DECIMAL(8,2),
    unit_cost DECIMAL(10,2),
    unit_price DECIMAL(10,2),
    
    -- SCD Type 2
    effective_date DATE NOT NULL,
    expiration_date DATE,
    is_current BOOLEAN DEFAULT TRUE
);

-- Dimension: Store (small: hundreds)
CREATE TABLE dim_store (
    store_key INT PRIMARY KEY,
    store_id INT,
    store_name VARCHAR(255),
    store_type VARCHAR(50),  -- Online, Retail, Outlet
    
    -- Denormalized location
    address VARCHAR(500),
    city VARCHAR(100),
    state VARCHAR(50),
    postal_code VARCHAR(20),
    country VARCHAR(50),
    region VARCHAR(50),
    
    -- Attributes
    opening_date DATE,
    square_footage INT,
    manager_name VARCHAR(255)
);
\`\`\`

**Why Denormalize Dimensions?**

Normalized (Bad for OLAP):
\`\`\`sql
products (product_id, name, category_id)
categories (category_id, name, parent_category_id)
-- Query requires JOIN: slow for billions of fact rows
\`\`\`

Denormalized (Good for OLAP):
\`\`\`sql
dim_product (product_key, name, category, subcategory, brand)
-- Single JOIN to fact table: much faster
\`\`\`

**Migration Strategy: OLTP → Star Schema**

**Phase 1: Design Star Schema**1. **Identify Facts:** What business processes to analyze?
   - Orders, shipments, returns, payments

2. **Define Grain:** What does one fact row represent?
   - One order line item (product × order)

3. **Choose Measures:** What to analyze?
   - quantity, amount, discount, shipping_cost

4. **Identify Dimensions:** How to slice/dice?
   - date, customer, product, store, promotion

**Phase 2: Build ETL Pipeline**

**ETL = Extract, Transform, Load**

\`\`\`python
# ETL Job (runs daily at 2 AM)

def daily_etl():
    # 1. EXTRACT: Get new/updated data from OLTP
    new_orders = extract_orders (since=yesterday)
    new_customers = extract_customers (since=yesterday)
    new_products = extract_products (since=yesterday)
    
    # 2. TRANSFORM: Clean, enrich, conform
    transformed_data = transform (new_orders, new_customers, new_products)
    
    # 3. LOAD: Insert into data warehouse
    load (transformed_data)

def extract_orders (since):
    # Query OLTP database (read replica to avoid impacting production)
    return oltp_db.query("""
        SELECT 
            o.order_id,
            o.customer_id,
            o.order_date,
            oi.product_id,
            oi.quantity,
            oi.unit_price,
            oi.total_amount
        FROM orders o
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.order_date >= %s
    """, since)

def transform (orders, customers, products):
    result = []
    
    for order in orders:
        # Lookup dimension keys (or create if new)
        date_key = lookup_or_create_date (order.order_date)
        customer_key = lookup_or_create_customer (order.customer_id, customers)
        product_key = lookup_or_create_product (order.product_id, products)
        store_key = get_store_key (order.store_id)
        
        # Build fact row
        fact_row = {
            'date_key': date_key,
            'customer_key': customer_key,
            'product_key': product_key,
            'store_key': store_key,
            'quantity': order.quantity,
            'unit_price': order.unit_price,
            'total_amount': order.total_amount,
            'discount_amount': calculate_discount (order),
            'shipping_cost': calculate_shipping (order),
            'order_number': order.order_number
        }
        
        result.append (fact_row)
    
    return result

def lookup_or_create_customer (customer_id, customers_data):
    # Check if customer already exists
    existing = dw_db.query(
        "SELECT customer_key FROM dim_customer WHERE customer_id = %s AND is_current = TRUE",
        customer_id
    )
    
    if existing:
        customer_key = existing[0].customer_key
        
        # Check if attributes changed (SCD Type 2)
        customer_data = customers_data[customer_id]
        if customer_changed (customer_key, customer_data):
            # Expire old record
            dw_db.execute("""
                UPDATE dim_customer 
                SET is_current = FALSE, expiration_date = CURRENT_DATE
                WHERE customer_key = %s
            """, customer_key)
            
            # Insert new record
            customer_key = dw_db.insert_returning_key("""
                INSERT INTO dim_customer 
                (customer_id, customer_name, email, city, state, country, ..., effective_date, is_current)
                VALUES (%s, %s, %s, %s, %s, %s, ..., CURRENT_DATE, TRUE)
            """, customer_data)
        
        return customer_key
    else:
        # New customer: insert
        return dw_db.insert_returning_key("""
            INSERT INTO dim_customer 
            (customer_id, customer_name, email, city, state, country, ..., effective_date, is_current)
            VALUES (%s, %s, %s, %s, %s, %s, ..., CURRENT_DATE, TRUE)
        """, customers_data[customer_id])

def load (transformed_data):
    # Bulk insert into fact table
    dw_db.bulk_insert('fact_orders', transformed_data)
    
    # Update indexes
    dw_db.analyze('fact_orders')
\`\`\`

**Phase 3: Slowly Changing Dimensions (SCD)**

**Type 1: Overwrite (no history)**
\`\`\`sql
-- Product price changes: just update
UPDATE dim_product SET unit_price = 29.99 WHERE product_key = 123;
\`\`\`

**Type 2: Add New Row (track history)**
\`\`\`sql
-- Customer moves to new city: keep history
-- 1. Expire old record
UPDATE dim_customer 
SET is_current = FALSE, expiration_date = '2024-01-15'
WHERE customer_key = 456;

-- 2. Insert new record
INSERT INTO dim_customer (customer_id, name, city, state, effective_date, is_current)
VALUES (123, 'Alice', 'Boston', 'MA', '2024-01-15', TRUE);

-- Queries automatically use current record
SELECT * FROM dim_customer WHERE customer_id = 123 AND is_current = TRUE;
\`\`\`

**Type 3: Add New Column (track current + original)**
\`\`\`sql
-- Track both original and current values
ALTER TABLE dim_customer ADD COLUMN original_segment VARCHAR(50);
ALTER TABLE dim_customer ADD COLUMN current_segment VARCHAR(50);
\`\`\`

**Phase 4: Optimization**1. **Partitioning:**
\`\`\`sql
-- Partition fact table by date
CREATE TABLE fact_orders_2024_01 PARTITION OF fact_orders
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
-- Queries only scan relevant partitions
\`\`\`

2. **Columnar Storage:**
\`\`\`sql
-- Use columnar format for analytical queries (Parquet, ORC)
-- Much faster for aggregations: "SELECT SUM(total_amount)"
\`\`\`

3. **Aggregation Tables:**
\`\`\`sql
-- Pre-aggregate for common queries
CREATE TABLE fact_orders_daily AS
SELECT 
    date_key,
    product_key,
    SUM(quantity) as total_quantity,
    SUM(total_amount) as total_sales,
    COUNT(*) as order_count
FROM fact_orders
GROUP BY date_key, product_key;
-- Monthly dashboards query this (1000x fewer rows)
\`\`\`

**Performance Comparison:**

| Query | OLTP (Normalized) | OLAP (Star) | Improvement |
|-------|-------------------|-------------|-------------|
| Sales by category by month | 30s (5 JOINs) | 200ms (2 JOINs) | 150x |
| Top 10 products | 10s | 50ms | 200x |
| Customer lifetime value | 60s | 300ms | 200x |

**Summary:**

- **OLTP:** Normalized, optimized for writes, current data
- **OLAP:** Denormalized star schema, optimized for complex analytical queries, historical data
- **Migration:** ETL pipeline extracts from OLTP, transforms to star schema, loads into data warehouse
- **SCD:** Handle changing dimensions (Type 1/2/3)
- **Optimization:** Partitioning, columnar storage, aggregation tables

Star schema provides 100-200x query performance improvement for analytics workloads.`,
    keyPoints: [
      'OLTP: normalized for data integrity and transactional writes',
      'OLAP: denormalized star schema for analytical query performance',
      'Star schema: fact table (metrics) + dimension tables (context)',
      'ETL process: extract, transform (denormalize + SCD), load',
      'Optimizations: partitioning by date, columnar storage, aggregation tables',
    ],
  },
];
