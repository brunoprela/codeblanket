/**
 * Airbnb Architecture Section
 */

export const airbnbarchitectureSection = {
  id: 'airbnb-architecture',
  title: 'Airbnb Architecture',
  content: `Airbnb is a global online marketplace connecting travelers with hosts offering accommodations and experiences. With 7+ million listings in 100,000+ cities and 150+ million users, Airbnb's architecture must handle complex challenges including geospatial search, dynamic pricing, payments, trust & safety, and global scale. This section explores the technical systems powering Airbnb.

## Overview

Airbnb's scale and challenges:
- **150+ million users** worldwide
- **7+ million listings** in 220+ countries/regions
- **100,000+ cities** with listings
- **$73+ billion** in gross booking value (2022)
- **1 billion+ searches** per year
- **Complex transactions**: Multi-day reservations, payments, cancellations

### Key Architectural Challenges

1. **Search**: Geospatial queries with complex filters (dates, price, amenities, guests)
2. **Availability**: Prevent double bookings across distributed system
3. **Pricing**: Dynamic pricing based on demand, supply, events
4. **Payments**: Handle complex payment flows (deposits, payouts, refunds, multi-currency)
5. **Trust & Safety**: Verify users, detect fraud, moderate content
6. **Scale**: Millions of concurrent users, billions of searches

---

## Evolution of Airbnb's Architecture

### Phase 1: Monolithic Rails App (2008-2012)

Early Airbnb was a simple Ruby on Rails monolith.

\`\`\`
Browser → Rails App → MySQL Database → S3 (photos)
\`\`\`

**Simple Architecture**:
- Single Rails application
- Single MySQL database
- Photos stored in S3
- Manual deployment

**Scaling Challenges**:
- Database bottleneck (single instance)
- Slow deployments (entire app redeployed)
- Tight coupling (changes risky)

---

### Phase 2: Service-Oriented Architecture (2013-2017)

Airbnb decomposed monolith into services.

**Services**:
- **Search Service**: Listing search with filters
- **Booking Service**: Reservation management
- **Payment Service**: Payments, payouts, refunds
- **Messaging Service**: Host-guest communication
- **User Service**: User profiles, authentication

**Communication**:
- REST APIs between services
- Message queues (RabbitMQ) for async operations

**Benefits**:
- Independent scaling (search scales separately from booking)
- Team ownership (each team owns a service)
- Faster deployments (deploy single service, not entire app)

---

### Phase 3: Microservices and Platform (2018-present)

Airbnb evolved to hundreds of microservices.

**Key Improvements**:
- **Service mesh**: Istio for service-to-service communication
- **API gateway**: Centralized routing, authentication
- **Event-driven architecture**: Kafka for event streaming
- **Infrastructure as code**: Terraform, Kubernetes

---

## Core Components

### 1. Search Infrastructure

Airbnb search is one of the most complex parts of the platform.

**Search Requirements**:
- **Geospatial**: Find listings near location (city, landmark, coordinates)
- **Date availability**: Only show available listings for travel dates
- **Filters**: Price range, number of guests, bedrooms, amenities, property type
- **Ranking**: Show most relevant listings first
- **Performance**: Sub-second response time
- **Scale**: Billions of searches per year

**Architecture**:

**Search Pipeline**:

\`\`\`
User Query → API Gateway → Search Service
                                ↓
                           Query Parser (extract filters)
                                ↓
                           Elasticsearch (candidate generation)
                                ↓
                           Ranking Service (ML model)
                                ↓
                           Results (listings)
\`\`\`

---

### Search Index (Elasticsearch)

Airbnb uses **Elasticsearch** for the search index.

**Index Structure**:

\`\`\`json
{
  "listing_id": "12345",
  "title": "Cozy apartment in downtown SF",
  "description": "Beautiful 2BR apartment with city views...",
  "location": {
    "lat": 37.7749,
    "lon": -122.4194,
    "city": "San Francisco",
    "neighborhood": "SOMA",
    "country": "US"
  },
  "property_type": "apartment",
  "room_type": "entire_place",
  "accommodates": 4,
  "bedrooms": 2,
  "bathrooms": 1,
  "amenities": ["wifi", "kitchen", "washer", "dryer", "parking"],
  "price_per_night": 150,
  "minimum_nights": 2,
  "available_dates": ["2024-10-25", "2024-10-26", "2024-10-27", ...],
  "rating": 4.8,
  "review_count": 127,
  "host_id": "98765",
  "host_response_rate": 0.95,
  "instant_book": true,
  "cancellation_policy": "flexible",
  "photos": ["photo1.jpg", "photo2.jpg"],
  "booking_score": 0.85  // ML-predicted booking probability
}
\`\`\`

**Indexing Process**:

1. **New listing created** → Publish event to Kafka
2. **Indexer service** consumes event → Fetch listing details
3. **Transform data** → Add computed fields (booking_score, etc.)
4. **Index in Elasticsearch** → Store in appropriate shard
5. **Listing searchable** within seconds

**Index Updates**:
- Price change → Update index
- Booking made → Update available_dates
- Review added → Update rating, review_count
- Near real-time updates (within 1-5 seconds)

---

### Geospatial Queries

Finding listings near a location is core to search.

**Elasticsearch Geospatial Query**:

\`\`\`json
{
  "query": {
    "bool": {
      "must": [
        {
          "geo_distance": {
            "distance": "10km",
            "location": {
              "lat": 37.7749,
              "lon": -122.4194
            }
          }
        },
        {
          "range": {
            "price_per_night": {
              "gte": 50,
              "lte": 200
            }
          }
        },
        {
          "term": {
            "accommodates": {"gte": 2}
          }
        },
        {
          "terms": {
            "amenities": ["wifi", "kitchen"]
          }
        }
      ],
      "filter": [
        {
          "terms": {
            "available_dates": ["2024-10-25", "2024-10-26", "2024-10-27"]
          }
        }
      ]
    }
  },
  "size": 300,
  "sort": [
    {"booking_score": "desc"}
  ]
}
\`\`\`

**How it works**:
1. **Geospatial filter**: Find listings within 10km radius
2. **Price filter**: $50-$200 per night
3. **Capacity filter**: Accommodates >= 2 guests
4. **Amenity filter**: Must have wifi AND kitchen
5. **Availability filter**: All dates (Oct 25-27) available
6. **Ranking**: Sort by booking_score (ML model)
7. **Return top 300** candidates (pagination on client)

**Performance**:
- P50 latency: <100ms
- P99 latency: <500ms
- Elasticsearch cluster: 100s of nodes, sharded by geography

---

### Availability Checking

Airbnb must ensure dates are truly available (no double bookings).

**Problem**: Elasticsearch index may be stale (updated every 1-5 seconds).

**Solution: Two-Phase Availability Check**:

**Phase 1: Elasticsearch (Fast, Approximate)**
- Search returns listings marked as available
- Good enough for search results (10-20 listings shown)

**Phase 2: Database (Slow, Accurate)**
- When user clicks listing → Check real-time availability in database
- Query availability calendar table (PostgreSQL)
- Ensures no race condition

\`\`\`sql
SELECT date, status FROM availability_calendar
WHERE listing_id = 12345
  AND date BETWEEN '2024-10-25' AND '2024-10-27'
\`\`\`

If all dates available → Show listing as bookable
If any date unavailable → Show "Dates not available"

---

### Search Ranking (Machine Learning)

Airbnb uses ML to rank search results by predicted booking probability.

**Goal**: Show listings user most likely to book (optimize for bookings, not clicks)

**Features** (100s of features):

**Listing Features**:
- Price per night
- Number of bedrooms, bathrooms
- Amenities count
- Rating, review count
- Photos count, quality score
- Instant book enabled
- Cancellation policy
- Host response rate, response time

**User Features**:
- Search history
- Past bookings (property types, price ranges)
- Demographics (age, location)
- Device type (mobile, desktop)

**Context Features**:
- Distance from search location to listing
- Days until check-in (lead time)
- Length of stay (nights)
- Time of day, day of week
- Local events (conferences, concerts)

**Model**:
- **Gradient Boosted Decision Trees** (XGBoost)
- Trained on billions of search sessions
- Label: Did user book this listing? (1/0)
- Predict: P(booking | listing, user, context)

**Training Pipeline**:
1. Collect search sessions (user saw listings, booked one or none)
2. Extract features for each (listing, user, context) tuple
3. Train XGBoost model (distributed training on Spark)
4. Evaluate on held-out test set (AUC, precision, recall)
5. A/B test in production (measure booking rate)
6. Deploy model if better than baseline

**Serving**:
- Model deployed as microservice (gRPC)
- Inference: <10ms per listing
- Search service calls ranking service with candidates
- Returns ranked listings

**Continuous Improvement**:
- Retrain model weekly with new data
- Online learning for personalization (user-specific features)

---

### 2. Dynamic Pricing (Smart Pricing)

Airbnb's Smart Pricing suggests optimal prices to hosts.

**Goals**:
- **Maximize bookings**: Price too high → No bookings
- **Maximize revenue**: Price too low → Lost revenue
- **Optimize occupancy**: Balance between high price and high occupancy

**Pricing Factors**:

**Demand Factors**:
- **Seasonality**: Summer vs winter, weekends vs weekdays
- **Local events**: Concerts, conferences, sports games (spike demand)
- **Holidays**: Thanksgiving, New Year's Eve, etc.
- **Lead time**: Last-minute bookings often pricier

**Supply Factors**:
- **Competing listings**: Prices of similar listings nearby
- **Market saturation**: Too many listings → Lower prices

**Listing Quality Factors**:
- **Reviews**: Higher-rated listings command premium
- **Amenities**: Pool, hot tub, parking add value
- **Photos**: High-quality photos increase booking probability
- **Location**: Downtown vs suburbs

**Historical Data**:
- Past booking rates at different price points
- How quickly similar listings book

**ML Model**:

**Approach**: Predict optimal price that maximizes expected revenue.

\`\`\`
expected_revenue(price) = price × P(booked | price)

where P(booked | price) = ML model prediction
\`\`\`

**Training**:
- Features: All pricing factors above
- Label: Was listing booked at this price? (1/0)
- Model: Gradient boosted trees or neural network
- Optimize: Find price that maximizes expected revenue

**Example**:
\`\`\`
Listing in San Francisco during Dreamforce (major conference):
- Base price: $200/night
- Event demand multiplier: 2.5x
- Competing listings: Average $450/night
- Smart pricing suggests: $480/night
- Host accepts or overrides
\`\`\`

**Price Range Constraints**:
- Host sets minimum and maximum price
- Smart pricing adjusts within range
- Host can override anytime

**Real-Time Adjustments**:
- Prices updated daily based on latest data
- If listing not booked 2 weeks before → Reduce price
- If booking rate high → Increase price

---

### 3. Booking and Payment Flow

Airbnb's booking flow handles complex payment logic.

**Booking Flow**:

**Step 1: Guest Initiates Booking**
\`\`\`
1. Guest selects listing, dates, guests
2. Client calculates total:
   - Nightly rate × nights
   - Cleaning fee (one-time)
   - Service fee (Airbnb's commission, ~14% of subtotal)
   - Occupancy taxes (varies by location)
3. Guest clicks "Request to Book" or "Reserve"
\`\`\`

**Step 2: Payment Capture**
\`\`\`
1. Guest enters payment method (credit card, PayPal, Apple Pay)
2. Airbnb tokenizes payment (via Stripe/Braintree)
3. Airbnb captures payment (funds held in escrow)
4. Payment not transferred to host yet (held until check-in)
\`\`\`

**Step 3: Host Acceptance** (for non-instant-book)
\`\`\`
1. Host has 24 hours to accept or decline
2. If accepted:
   - Booking confirmed
   - Calendar blocked (dates unavailable)
   - Confirmation email sent
3. If declined:
   - Booking canceled
   - Funds returned to guest (full refund)
   - Guest suggested alternative listings
4. If no response in 24 hours:
   - Auto-decline (protect guests)
\`\`\`

**Step 4: Funds Release**
\`\`\`
1. Day of check-in arrives
2. Airbnb releases funds to host (minus Airbnb's service fee)
3. Payout typically within 24 hours
4. Host receives ~85% of nightly rate (15% to Airbnb)
\`\`\`

**Cancellation Handling**:

**Guest Cancels**:
- Depends on cancellation policy (flexible, moderate, strict)
- Flexible: Full refund up to 24 hours before check-in
- Strict: 50% refund up to 7 days before, 0% after
- Refund processed automatically

**Host Cancels**:
- Heavily penalized (damages ranking, potential suspension)
- Guest receives full refund + Airbnb credit
- Host charged cancellation fee

---

### Payment Architecture

**Payment Service**:

**Data Model** (PostgreSQL):
\`\`\`sql
Table: bookings
- booking_id (primary key)
- listing_id
- guest_id
- host_id
- check_in_date
- check_out_date
- num_guests
- nightly_rate
- cleaning_fee
- service_fee
- total_amount
- status (pending, confirmed, completed, canceled)
- payment_intent_id (Stripe)
- created_at

Table: payments
- payment_id (primary key)
- booking_id
- amount
- currency
- payment_method (card, paypal, etc.)
- status (captured, held, released, refunded)
- stripe_charge_id
- created_at

Table: payouts
- payout_id (primary key)
- booking_id
- host_id
- amount
- currency
- status (pending, completed, failed)
- payout_method (bank_transfer, paypal)
- scheduled_date (check-in date)
- completed_date
\`\`\`

**Payment Flow** (Detailed):

\`\`\`
1. Create booking record (status: pending)
2. Create payment intent with Stripe:
   - Amount: total_amount
   - Currency: USD (or guest's currency)
   - Metadata: booking_id, guest_id, listing_id
3. Stripe returns payment_intent_id
4. Guest confirms payment in Stripe's UI (3D Secure if required)
5. Stripe calls webhook: payment_intent.succeeded
6. Airbnb updates booking status to confirmed
7. Airbnb updates payment status to captured
8. Block calendar dates (availability_calendar)
9. Create payout record (scheduled for check-in date)
10. On check-in date:
    - Background job processes payouts
    - Create Stripe transfer to host's bank account
    - Update payout status to completed
\`\`\`

**Multi-Currency Support**:
- Guest pays in their currency (EUR)
- Host receives in their currency (USD)
- Airbnb handles conversion (uses Stripe's rates + markup)
- Currency locked at booking time (protect from rate changes)

---

### 4. Availability Calendar and Double-Booking Prevention

Preventing double bookings is critical for Airbnb's reputation.

**Challenge**: Two guests attempt booking same dates simultaneously.

**Solution: Distributed Locking with Optimistic Concurrency Control**

**Approach 1: Pessimistic Locking (Used for Critical Path)**

\`\`\`python
def create_booking(listing_id, check_in, check_out, guest_id):
    # Acquire distributed lock on listing
    lock = redis.lock(f"listing:{listing_id}:booking", timeout=10)
    
    if not lock.acquire(blocking=True, timeout=5):
        raise BookingError("Could not acquire lock, try again")
    
    try:
        # Start database transaction
        with db.transaction():
            # Check availability
            dates = get_dates_between(check_in, check_out)
            unavailable = db.query(
                "SELECT date FROM availability_calendar "
                "WHERE listing_id = ? AND date IN ? AND status != 'available'",
                listing_id, dates
            )
            
            if unavailable:
                raise BookingError("Dates no longer available")
            
            # Mark dates as booked
            for date in dates:
                db.execute(
                    "UPDATE availability_calendar "
                    "SET status = 'booked', booking_id = ? "
                    "WHERE listing_id = ? AND date = ?",
                    booking_id, listing_id, date
                )
            
            # Create booking record
            booking_id = db.insert("bookings", {...})
            
            # Commit transaction
        
        return booking_id
    
    finally:
        # Release lock
        lock.release()
\`\`\`

**Key Properties**:
- **Distributed lock** (Redis) prevents concurrent bookings
- **Database transaction** ensures atomicity (all dates booked or none)
- **Lock timeout** (10 seconds) prevents deadlocks
- **Acquire timeout** (5 seconds) fails fast if lock unavailable

**Approach 2: Optimistic Locking (Used for Non-Critical)**

\`\`\`sql
Table: availability_calendar
- listing_id
- date
- status (available, booked, blocked)
- booking_id (if booked)
- version (for optimistic locking)

Update with version check:
UPDATE availability_calendar
SET status = 'booked', booking_id = ?, version = version + 1
WHERE listing_id = ? AND date = ? AND version = ? AND status = 'available'

If rows affected = 0 → Conflict detected (someone else booked) → Retry
\`\`\`

**Calendar Storage**:

\`\`\`sql
Table: availability_calendar
- listing_id (indexed)
- date (indexed)
- status (available, booked, blocked)
- booking_id (nullable, FK to bookings)

Composite index: (listing_id, date)

Query example:
SELECT date, status FROM availability_calendar
WHERE listing_id = 12345
  AND date BETWEEN '2024-10-25' AND '2024-12-31'
ORDER BY date
\`\`\`

**Syncing with Elasticsearch**:
- After calendar update → Publish event to Kafka
- Indexer service consumes event → Update Elasticsearch
- Available_dates field updated in index
- Near real-time sync (1-5 second delay acceptable)

---

### 5. Reviews and Ratings

Reviews are central to trust on Airbnb.

**Two-Way Reviews** (Unique Feature):
- Guest reviews host (after checkout)
- Host reviews guest (after checkout)
- Both reviews hidden until both submitted OR 14 days pass
- Prevents retaliation (host can't see guest's review before writing theirs)

**Review Process**:

\`\`\`
Day 0: Check-out date
Day 1: Airbnb sends review reminders (email, push)
Day 1-14: Review window open
  - Guest submits review: rating (1-5 stars), text
  - Host submits review: rating, text
Day 14: Window closes
  - If both submitted: Reveal simultaneously
  - If only one submitted: Publish only that review
  - If neither submitted: No reviews
\`\`\`

**Data Model**:

\`\`\`sql
Table: reviews
- review_id (primary key)
- booking_id (FK)
- reviewer_id (guest_id or host_id)
- reviewee_id (host_id or guest_id)
- reviewer_type (guest, host)
- overall_rating (1-5)
- category_ratings (JSON):
  - cleanliness (1-5)
  - accuracy (1-5)  // listing matched description
  - communication (1-5)
  - location (1-5)
  - check_in (1-5)
  - value (1-5)
- review_text (max 1000 chars)
- is_visible (boolean)  // false until both submitted
- submitted_at
- published_at

Table: review_responses
- response_id
- review_id (FK)
- responder_id (reviewee responding to review)
- response_text
- created_at
\`\`\`

**Aggregate Ratings**:

\`\`\`python
def calculate_listing_rating(listing_id):
    reviews = db.query(
        "SELECT overall_rating, category_ratings FROM reviews "
        "WHERE reviewee_id = ? AND reviewer_type = 'guest' AND is_visible = true",
        listing_id
    )
    
    overall_avg = mean([r.overall_rating for r in reviews])
    cleanliness_avg = mean([r.category_ratings['cleanliness'] for r in reviews])
    # ... other categories
    
    db.update("listings", listing_id, {
        "average_rating": overall_avg,
        "review_count": len(reviews),
        "category_ratings": {
            "cleanliness": cleanliness_avg,
            # ...
        }
    })
    
    # Update Elasticsearch index
    elasticsearch.update(listing_id, {"rating": overall_avg, "review_count": len(reviews)})
\`\`\`

**Review Moderation**:
- Automated: ML models detect spam, fake reviews, policy violations
- Manual: Flagged reviews reviewed by trust & safety team
- Violations: Remove review, penalize reviewer

---

### 6. Trust and Safety

Trust & safety is critical for peer-to-peer platform.

**Verification Systems**:

**1. Identity Verification**:
- Government-issued ID (passport, driver's license)
- Selfie verification (match face to ID)
- Third-party services (Jumio, Onfido)

**2. Email and Phone Verification**:
- Verify email (click link)
- Verify phone (SMS code)
- Required before booking

**3. Social Connections**:
- Link Facebook, LinkedIn (optional)
- Shows mutual friends (increases trust)

**4. Background Checks** (Host-specific):
- Criminal records check (some regions)
- Sex offender registry check
- Opt-in for hosts (increases trust)

**Fraud Detection**:

**Rule-Based System**:
- Flag suspicious patterns:
  - New account books expensive listing immediately
  - Multiple accounts from same IP
  - Payment method from different country than IP
  - Listing photos match other listings (stolen)

**ML-Based System**:
- Train models on historical fraud cases
- Features:
  - Account age, verification status
  - Booking patterns (frequency, price range)
  - Device fingerprint, IP reputation
  - Communication patterns (messages to host)
- Predict fraud probability (0-100%)
- High-risk bookings held for manual review

**Content Moderation**:

**Listing Photos**:
- ML classifiers detect inappropriate content
- Flag: Weapons, drugs, nudity, offensive symbols
- Manual review for borderline cases

**Messages**:
- Scan for policy violations
- Flag: Off-platform payments, discrimination, harassment
- Keyword matching + NLP sentiment analysis

**Actions**:
- Warning (first violation)
- Temporary suspension (repeat violation)
- Permanent ban (severe violation)

---

### 7. Messaging System

Host-guest communication is essential.

**Architecture**:

**Real-Time Messaging**:
- WebSocket for instant delivery
- Message storage in PostgreSQL (sharded by conversation_id)
- Push notifications for offline users

**Data Model**:

\`\`\`sql
Table: conversations
- conversation_id (primary key)
- booking_id (nullable, FK)
- participants (array of user_ids)
- created_at
- last_message_at

Table: messages
- message_id (primary key)
- conversation_id (FK, indexed)
- sender_id (FK to users)
- text (max 5000 chars)
- attachments (JSON array)  // photos, pdfs
- created_at
- read_by (array of user_ids)

Index: (conversation_id, created_at DESC)
\`\`\`

**Message Flow**:

\`\`\`
1. Guest sends message to host
2. Client → WebSocket → Messaging Service
3. Store message in database (conversation_id)
4. If host online:
   - Push message via WebSocket
   - Display in real-time
5. If host offline:
   - Store as unread
   - Send push notification (mobile/web)
   - Send email (if enabled)
6. Host reads message → Update read_by array
\`\`\`

**Message Safety**:
- Scan for off-platform transactions ("pay me via Venmo")
- Detect discrimination ("no [protected class]")
- Flag suspicious messages for manual review
- Auto-block messages with banned keywords

---

### 8. Experiences (Activities)

Airbnb Experiences: Book activities hosted by locals (cooking classes, tours, workshops).

**Architecture** (Similar to Listings):

**Search**:
- Elasticsearch index (experiences)
- Filters: Date, time, category, location, price

**Booking**:
- Group bookings (multiple guests per experience)
- Capacity limits (max 10 guests per class)
- Real-time availability (track spots remaining)

**Payments**:
- Pay upfront (experience must occur for payout)
- Cancellation policies (flexible, moderate, strict)

**Data Model**:

\`\`\`sql
Table: experiences
- experience_id
- host_id
- title
- description
- category (food, art, sports, etc.)
- duration (hours)
- location
- capacity (max guests)
- price_per_person
- available_times (JSON)  // recurring schedule

Table: experience_bookings
- booking_id
- experience_id
- guest_id
- date
- time
- num_guests
- total_amount
- status
\`\`\`

---

## Technology Stack

### Backend

- **Ruby on Rails**: Original monolith, still used for some services
- **Java**: Core services (search, booking, payments)
- **Go**: High-performance services (real-time messaging)
- **Python**: Data science, ML models
- **Node.js**: Some frontend services

### Data Storage

- **MySQL**: Primary database (sharded by listing_id, user_id)
- **PostgreSQL**: Some services (newer)
- **Redis**: Caching, session storage, rate limiting, distributed locks
- **Elasticsearch**: Search index (listings, experiences, users)
- **S3**: Photos, documents

### Data Processing

- **Apache Kafka**: Event streaming (bookings, reviews, searches)
- **Apache Spark**: Batch processing (ML training, analytics)
- **Apache Airflow**: Workflow orchestration (data pipelines)
- **Presto**: SQL query engine (data warehouse queries)

### Machine Learning

- **TensorFlow**: Deep learning models (image quality, fraud detection)
- **XGBoost**: Gradient boosted trees (search ranking, pricing)
- **Scikit-learn**: Classical ML (feature engineering, prototyping)
- **MLflow**: ML experiment tracking, model registry

### Infrastructure

- **AWS**: Primary cloud provider (EC2, S3, RDS, etc.)
- **Kubernetes**: Container orchestration
- **Istio**: Service mesh (traffic management, observability)
- **Terraform**: Infrastructure as code
- **Spinnaker**: Deployment platform

### Monitoring & Observability

- **Datadog**: Metrics, logs, traces
- **PagerDuty**: On-call, alerting
- **Sentry**: Error tracking
- **New Relic**: APM (application performance monitoring)

---

## Key Lessons from Airbnb Architecture

### 1. Geospatial Search is Complex

Combining geospatial queries with date availability, filters, and ranking requires sophisticated indexing (Elasticsearch) and ML models.

### 2. Preventing Double Bookings Requires Distributed Locking

Distributed locking (Redis) + database transactions ensure atomicity and prevent race conditions.

### 3. Two-Way Reviews Build Trust

Airbnb's innovation: Reviews revealed simultaneously prevents retaliation, encourages honesty.

### 4. ML Drives User Experience

Search ranking, dynamic pricing, fraud detection all powered by ML models trained on billions of interactions.

### 5. Trust & Safety is Non-Negotiable

Verification, fraud detection, content moderation essential for peer-to-peer marketplace.

### 6. Payments are Complex

Multi-currency, escrow, payouts, refunds, cancellation policies require robust payment architecture.

---

## Interview Tips

**Q: How would you design Airbnb's search system?**

A: Use Elasticsearch for search index with geospatial capabilities. Index listing data: location (lat/lon), price, amenities, available_dates. Search query: (1) Geospatial filter (geo_distance within radius). (2) Date filter (all dates in available_dates). (3) Capacity filter (accommodates >= guests). (4) Price range, amenity filters. (5) Return top 300 candidates. Rank candidates using ML model: predict booking probability based on listing features (price, rating, photos), user features (search history, demographics), context (distance, lead time). Train XGBoost model on billions of search sessions. Serve predictions via microservice (<10ms latency). Handle staleness: Elasticsearch updated every 1-5 seconds (acceptable for search results), verify availability in real-time on listing page (query PostgreSQL).

**Q: How would you prevent double bookings in Airbnb?**

A: Use distributed locking with database transactions. When guest attempts booking: (1) Acquire distributed lock on listing (Redis: SETNX listing:{id}:booking with 10-second timeout). (2) Start database transaction. (3) Query availability_calendar: SELECT date, status WHERE listing_id=? AND date IN (check_in, ..., check_out). (4) If any date unavailable, rollback transaction, release lock, return error. (5) If all dates available, UPDATE availability_calendar SET status='booked', booking_id=? for each date. (6) INSERT booking record. (7) Commit transaction. (8) Release lock. Lock prevents concurrent requests, transaction ensures atomicity (all dates booked or none). Alternative: Optimistic locking with version numbers (detect conflicts, retry). Sync availability to Elasticsearch via Kafka (near real-time updates for search results).

**Q: How does Airbnb implement dynamic pricing?**

A: Use ML model to predict optimal price maximizing expected revenue. Features: Demand (seasonality, local events, lead time), supply (competing listings' prices), listing quality (rating, amenities, photos), historical data (past booking rates at different prices). Model predicts P(booked | price) for various prices. Calculate expected_revenue(price) = price × P(booked | price). Find price maximizing expected revenue. Train gradient boosted trees (XGBoost) on historical bookings. Retrain weekly with new data. Serve as pricing recommendation to hosts (Smart Pricing). Host sets min/max price bounds, can override. Update prices daily based on latest demand/supply. Handle events: Scrape event calendars (Eventbrite, Ticketmaster), increase prices for event dates. Monitor booking velocity: If listing not booked 2 weeks before → Reduce price.

---

## Summary

Airbnb's architecture demonstrates building a global marketplace at scale:

**Key Takeaways**:

1. **Elasticsearch + ML ranking**: Geospatial search with complex filters, ML model predicts booking probability
2. **Distributed locking**: Prevents double bookings using Redis locks + PostgreSQL transactions
3. **Dynamic pricing**: ML model optimizes price based on demand, supply, listing quality
4. **Two-way reviews**: Reviews revealed simultaneously, prevents retaliation
5. **Trust & safety**: Verification (ID, email, phone), fraud detection (ML), content moderation
6. **Payment complexity**: Multi-currency, escrow, payouts, cancellation policies handled via Stripe/Braintree
7. **Event-driven architecture**: Kafka for event streaming (bookings, reviews, searches)
8. **Microservices**: Hundreds of services for independent scaling and team autonomy

Airbnb's success comes from solving complex marketplace challenges: geospatial search, availability management, dynamic pricing, trust & safety, and seamless payments across 220 countries.
`,
};
