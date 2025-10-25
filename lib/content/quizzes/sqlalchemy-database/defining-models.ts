import { MultipleChoiceQuestion } from '@/lib/types';

export const definingModelsQuiz = [
  {
    id: 'sql-models-q-1',
    question:
      'Design a schema for an e-commerce platform with products, categories (hierarchical), orders, and order items. Address: (1) how to model the hierarchical category structure, (2) the relationship between products and categories, (3) the order-product many-to-many relationship with extra data (quantity, price at purchase), (4) foreign key CASCADE strategies, (5) indexes needed. Include complete model definitions.',
    sampleAnswer:
      'E-commerce schema design: (1) Hierarchical categories: Use self-referential relationship with parent_id. class Category: id, name, parent_id = Column(ForeignKey("categories.id")), parent = relationship("Category", remote_side=[id], back_populates="children"), children = relationship("Category", back_populates="parent"). This creates unlimited depth tree structure. Query: SELECT * FROM categories WHERE parent_id IS NULL (roots). (2) Product-category: Many-to-many with junction table. product_categories = Table with product_id, category_id. class Product: categories = relationship("Category", secondary=product_categories). Allows products in multiple categories. (3) Order items: Use association object pattern. class OrderItem: order_id, product_id (composite primary key), quantity, price_at_purchase (store price when ordered, not current price!), subtotal (computed: quantity × price_at_purchase). class Order: items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan"). Delete order items when order deleted. (4) CASCADE strategies: Order → OrderItem: CASCADE (delete items with order). Product → OrderItem: RESTRICT (prevent product deletion if ordered). User → Order: SET NULL or RESTRICT (keep order history). Category → Product: SET NULL (product survives category deletion). (5) Indexes: Index foreign keys always: idx_order_items_order_id, idx_order_items_product_id. Index frequently queried: idx_products_category, idx_orders_user_id, idx_orders_created_at. Partial index: idx_pending_orders WHERE status = "pending". Composite: idx_order_items_order_product (order_id, product_id) for JOIN optimization.',
    keyPoints: [
      'Self-referential: parent_id with remote_side=[id] for hierarchical categories',
      'Many-to-many: Use association object (OrderItem) for extra data (quantity, price)',
      'CASCADE: Delete children (order items) when parent (order) deleted',
      'RESTRICT: Prevent product deletion if in orders (preserve order history)',
      'Indexes: All foreign keys + composite indexes for common JOINs',
    ],
  },
  {
    id: 'sql-models-q-2',
    question:
      'Explain the difference between single table inheritance and joined table inheritance in SQLAlchemy. Provide: (1) complete code examples for both, (2) when to use each, (3) query performance implications, (4) trade-offs (normalization vs performance), (5) real-world scenarios for each pattern.',
    sampleAnswer:
      'Inheritance patterns comparison: (1) Single table inheritance: All subclasses in one table with discriminator column. class Employee: __tablename__ = "employees", type = Column(String), __mapper_args__ = {"polymorphic_on": type, "polymorphic_identity": "employee"}. class Engineer(Employee): programming_language = Column(String), __mapper_args__ = {"polymorphic_identity": "engineer"}. All columns in employees table: id, type, name, programming_language (NULL for non-engineers), department (NULL for non-managers). Joined table inheritance: Separate table per subclass. class Engineer(Employee): __tablename__ = "engineers", id = Column(ForeignKey("employees.id"), primary_key=True), programming_language = Column(String). Two tables: employees (id, type, name) and engineers (id, programming_language). Requires JOIN to fetch engineer. (2) When to use: Single table: Subclasses have few differences (1-3 unique columns), query performance critical, limited hierarchy depth. Example: User types (regular, premium, admin) with small differences. Joined table: Subclasses have many differences (5+ unique columns), normalization important, avoid NULL columns. Example: Payment methods (credit card has card_number, PayPal has paypal_email). (3) Performance: Single table: Faster queries (no JOIN), but wastes space (many NULL columns), slower scans (more data per row). Joined table: Slower queries (JOIN required), but efficient storage (no NULLs), faster scans. Benchmark: Single table 2-3x faster for single row lookup, joined table better for full table scans. (4) Trade-offs: Single table: Pros: Fast, simple. Cons: Denormalized, NULL columns, unclear schema. Joined table: Pros: Normalized, clear schema. Cons: Slower, complex queries. (5) Real-world: Single table: Content types (blog post, video, image) with similar metadata. Notification types (email, SMS, push) with small config differences. Joined table: Vehicle types (car has mileage, boat has engine_hours, plane has flight_hours). Payment instruments (complex, unique data per type).',
    keyPoints: [
      'Single table: All in one table with discriminator, fast but wastes space with NULLs',
      'Joined table: Separate tables with JOINs, normalized but slower',
      'Use single table: Few differences, performance critical, shallow hierarchy',
      'Use joined table: Many differences, normalization important, complex attributes',
      'Single table 2-3x faster for lookups, joined table better for scans',
    ],
  },
  {
    id: 'sql-models-q-3',
    question:
      'You have a many-to-many relationship between posts and tags. Initially you use a simple association table, but now you need to track who added the tag and when. Explain: (1) how to migrate from association table to association object, (2) complete model definitions before and after, (3) data migration strategy, (4) how existing queries change, (5) performance implications.',
    sampleAnswer:
      'Migrating to association object: (1) Migration steps: Add association class, create new table with extra columns, copy data, update relationships, drop old table. (2) Before (association table): post_tags = Table("post_tags", metadata, Column("post_id", ForeignKey("posts.id"), primary_key=True), Column("tag_id", ForeignKey("tags.id"), primary_key=True)). class Post: tags = relationship("Tag", secondary=post_tags, back_populates="posts"). class Tag: posts = relationship("Post", secondary=post_tags, back_populates="tags"). (3) After (association object): class PostTag(Base): __tablename__ = "post_tags", post_id = Column(ForeignKey("posts.id"), primary_key=True), tag_id = Column(ForeignKey("tags.id"), primary_key=True), added_by_user_id = Column(ForeignKey("users.id")), added_at = Column(DateTime, default=datetime.utcnow), post = relationship("Post", back_populates="tag_associations"), tag = relationship("Tag", back_populates="post_associations"). class Post: tag_associations = relationship("PostTag", back_populates="post", cascade="all, delete-orphan"), @property def tags (self): return [assoc.tag for assoc in self.tag_associations]. (4) Data migration: Alembic migration: def upgrade(): op.add_column("post_tags", sa.Column("added_by_user_id", Integer, ForeignKey("users.id"))), op.add_column("post_tags", sa.Column("added_at", DateTime, default=datetime.utcnow)). Set added_by_user_id = post.user_id (reasonable default), added_at = post.created_at. (5) Query changes: Before: post.tags returns List[Tag]. After: post.tag_associations returns List[PostTag], post.tags property returns List[Tag]. Accessing extra data: for assoc in post.tag_associations: print(assoc.tag.name, assoc.added_by_user_id, assoc.added_at). Adding tag: Before: post.tags.append (tag). After: assoc = PostTag (post=post, tag=tag, added_by_user_id=current_user_id); session.add (assoc). (6) Performance: Association object adds slight overhead (object creation), but minimal. Querying extra data requires fetching PostTag instead of just Tag (more memory). Loading strategy matters: use selectinload to avoid N+1: session.query(Post).options (selectinload(Post.tag_associations).selectinload(PostTag.tag)). Overall impact: Negligible for most apps, worth it for audit trail.',
    keyPoints: [
      'Association object: Create class with primary key (post_id, tag_id) + extra columns',
      'Data migration: Add columns, populate with reasonable defaults (post.user_id, created_at)',
      'Query changes: post.tags becomes post.tag_associations, add convenience @property',
      'Adding relationship: Create PostTag instance instead of appending to list',
      'Performance: Minimal overhead, use selectinload to avoid N+1 queries',
    ],
  },
];
