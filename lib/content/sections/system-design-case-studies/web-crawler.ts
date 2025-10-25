/**
 * Design Web Crawler Section
 */

export const webCrawlerSection = {
  id: 'web-crawler',
  title: 'Design Web Crawler',
  content: `A web crawler (or spider) is a system that systematically browses the web to index content for search engines. Google's crawler processes billions of web pages daily. The core challenges are: handling massive scale (billions of URLs), politeness (respecting robots.txt, rate limits), duplicate detection, handling dynamic content, and maintaining freshness of indexed data.

## Problem Statement

Design a web crawler that:
- **Discovers URLs**: Extract links from crawled pages
- **Fetches Content**: Download HTML, parse structure
- **Respects Politeness**: Follow robots.txt, rate limiting per domain
- **Avoids Duplicates**: Don't crawl same URL twice
- **Handles Failures**: Retry failed requests, handle timeouts
- **Prioritizes URLs**: Important pages crawled first (PageRank-like)
- **Supports Scale**: Billions of URLs, petabytes of data
- **Maintains Freshness**: Re-crawl pages periodically

**Scale**: 10 billion web pages, 1000 pages/second crawl rate, 30-day full crawl cycle

---

## Step 1: Requirements Gathering

### Functional Requirements

1. **URL Discovery**: Extract all links from crawled pages
2. **Content Fetching**: Download HTML, JavaScript, CSS
3. **Robots.txt Compliance**: Check before crawling each domain
4. **Deduplication**: Use URL normalization and content hashing
5. **Politeness**: Rate limit per domain (1 request/second)
6. **Priority Queue**: High-value pages crawled first
7. **Re-crawling**: Fresh content (news sites daily, static sites monthly)

### Non-Functional Requirements

1. **Scalability**: 1000+ pages/second, billions of URLs
2. **Fault Tolerance**: Handle server errors, timeouts gracefully
3. **Extensibility**: Support new content types (PDF, images)
4. **Politeness**: Don't overload target servers
5. **Efficiency**: Minimize bandwidth, storage

---

## Step 2: Capacity Estimation

**Crawl Target**: 10 billion web pages

**Crawl Rate**: 1000 pages/second

**Crawl Cycle**: 10B pages / 1000 pages/sec = 10M seconds = 115 days (let's target 30 days)

**Revised Rate**: 10B / (30 days × 86,400 sec/day) = 3,858 pages/second

**Bandwidth**:
- Average page size: 500 KB (HTML + embedded resources)
- Bandwidth: 3,858 pages/sec × 500 KB = 1.9 GB/sec = 164 TB/day

**Storage**:
- 10 billion pages × 500 KB = 5 PB (raw HTML)
- Compressed (3:1): ~1.7 PB
- Metadata (URL, timestamp, hash): 10B × 100 bytes = 1 TB

**URL Frontier** (queue of URLs to crawl):
- Discovered URLs: 100 billion (10 URLs per page)
- Storage: 100B × 200 bytes (URL + metadata) = 20 TB

---

## Step 3: High-Level Architecture

\`\`\`
                     ┌─────────────────┐
                     │   Seed URLs     │
                     │  (Entry points) │
                     └────────┬────────┘
                              │
                     ┌────────▼────────┐
                     │ URL Frontier    │
                     │ (Priority Queue)│
                     └────────┬────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│Crawler Worker│      │Crawler Worker│      │Crawler Worker│
│  (Fetcher)   │      │  (Fetcher)   │      │  (Fetcher)   │
└──────┬───────┘      └──────┬───────┘      └──────┬───────┘
       │                     │                     │
       └─────────────────────┼─────────────────────┘
                             │
                     ┌───────▼────────┐
                     │  HTML Parser   │
                     │ (Extract Links)│
                     └───────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐     ┌──────────────┐    ┌──────────────┐
│URL Seen?     │     │  Content     │    │   Link       │
│(Bloom Filter)│     │  Storage     │    │  Extractor   │
└──────────────┘     │     (S3)     │    └──────┬───────┘
                     └──────────────┘           │
                                                 │
                                        ┌────────▼────────┐
                                        │New URLs → Frontier│
                                        └─────────────────┘
\`\`\`

---

## Step 4: URL Frontier (Priority Queue)

**Purpose**: Manage which URLs to crawl next, with prioritization and politeness.

### Naive Approach: Single Queue (Wrong)

\`\`\`
queue = ["url1", "url2", "url3", ...]
while queue:
    url = queue.pop()
    html = fetch(url)
    links = extract_links(html)
    queue.extend(links)
\`\`\`

**Problems**:
- ❌ No prioritization (all URLs equal)
- ❌ No politeness (might crawl same domain 1000x/sec)
- ❌ No duplicate detection (same URL crawled repeatedly)

### Better Approach: Multi-Queue with Politeness

**Structure**:

\`\`\`
Priority Queue (Front):
  - High Priority: News sites, popular domains
  - Medium Priority: General web pages
  - Low Priority: Rarely updated pages

Back Queues (Per-Domain Politeness):
  - Queue for example.com (rate: 1 req/sec)
  - Queue for wikipedia.org (rate: 5 req/sec)
  - Queue for news.com (rate: 2 req/sec)

Flow:
  1. Front queue: Prioritize URL by importance score
  2. Route to back queue based on domain
  3. Back queue enforces rate limit (1 req/sec per domain)
  4. Worker fetches URL from back queue
\`\`\`

### Implementation

**Front Queue** (Priority-based):

\`\`\`python
class URLFrontier:
    def __init__(self):
        self.high_priority = PriorityQueue()    # PageRank > 0.8
        self.medium_priority = PriorityQueue()  # PageRank 0.3-0.8
        self.low_priority = PriorityQueue()     # PageRank < 0.3
    
    def add_url(self, url, priority_score):
        if priority_score > 0.8:
            self.high_priority.put((priority_score, url))
        elif priority_score > 0.3:
            self.medium_priority.put((priority_score, url))
        else:
            self.low_priority.put((priority_score, url))
    
    def get_next_url(self):
        # 70% from high, 20% from medium, 10% from low
        rand = random.random()
        if rand < 0.7 and not self.high_priority.empty():
            return self.high_priority.get()[1]
        elif rand < 0.9 and not self.medium_priority.empty():
            return self.medium_priority.get()[1]
        else:
            return self.low_priority.get()[1]
\`\`\`

**Back Queue** (Politeness-based):

\`\`\`python
class PolitenessQueue:
    def __init__(self):
        self.queues = {}  # domain → deque of URLs
        self.last_access = {}  # domain → timestamp
        self.min_delay = 1.0  # 1 second between requests to same domain
    
    def add_url(self, url):
        domain = extract_domain(url)
        if domain not in self.queues:
            self.queues[domain] = deque()
        self.queues[domain].append(url)
    
    def get_next_url(self):
        now = time.time()
        for domain, queue in self.queues.items():
            if not queue:
                continue
            
            # Check if enough time elapsed since last request
            last = self.last_access.get(domain, 0)
            if now - last >= self.min_delay:
                url = queue.popleft()
                self.last_access[domain] = now
                return url
        
        return None  # No URLs ready (all domains rate-limited)
\`\`\`

---

## Step 5: Duplicate Detection

**Problem**: Same URL might be discovered from multiple pages. Don't crawl duplicates.

### Challenge: Scale

- 100 billion discovered URLs
- Need to check "have we seen this URL?" in O(1) time
- Naive set: 100B × 200 bytes = 20 TB memory (too expensive)

### Solution 1: URL Normalization

\`\`\`python
def normalize_url(url):
    # Convert to lowercase
    url = url.lower()
    
    # Remove default port
    url = url.replace(":80/", "/").replace(":443/", "/")
    
    # Remove trailing slash
    url = url.rstrip("/")
    
    # Sort query parameters (consistent ordering)
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    sorted_query = urlencode(sorted(query.items()))
    url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{sorted_query}"
    
    # Remove fragments
    url = url.split("#")[0]
    
    return url

# Example:
# "HTTP://Example.com:80/Path?b=2&a=1#section"
# → "http://example.com/path?a=1&b=2"
\`\`\`

### Solution 2: Bloom Filter (Memory-Efficient)

\`\`\`python
class BloomFilter:
    def __init__(self, size=100_000_000_000, num_hashes=7):
        self.size = size
        self.bit_array = bitarray(size)  # 100B bits = 12.5 GB
        self.num_hashes = num_hashes
    
    def add(self, url):
        for i in range(self.num_hashes):
            index = hash_function(url, i) % self.size
            self.bit_array[index] = 1
    
    def contains(self, url):
        for i in range(self.num_hashes):
            index = hash_function(url, i) % self.size
            if self.bit_array[index] == 0:
                return False  # Definitely not seen
        return True  # Probably seen (false positive rate ~1%)

bloom = BloomFilter()

def should_crawl(url):
    url = normalize_url(url)
    if bloom.contains(url):
        return False  # Already crawled (or false positive)
    bloom.add(url)
    return True
\`\`\`

**Bloom Filter Benefits**:
- ✅ Memory: 12.5 GB (vs 20 TB for full set)
- ✅ O(1) lookups
- ✅ No false negatives (never miss a URL)
- ❌ 1% false positive rate (acceptable trade-off)

### Solution 3: Content Hash (Catch Near-Duplicates)

\`\`\`python
def content_hash(html):
    # Remove dynamic content (timestamps, ads, comments)
    cleaned = remove_boilerplate(html)
    return hashlib.md5(cleaned.encode()).hexdigest()

# Store in database
crawled_hashes = set()

def is_duplicate_content(html):
    h = content_hash(html)
    if h in crawled_hashes:
        return True  # Near-duplicate
    crawled_hashes.add(h)
    return False
\`\`\`

---

## Step 6: Robots.txt Handling

**Purpose**: Websites specify crawl rules in \`/robots.txt\`.

**Example** (\`example.com/robots.txt\`):

\`\`\`
User-agent: *
Disallow: /admin/
Disallow: /private/
Crawl-delay: 2

User-agent: Googlebot
Disallow: /temp/
Crawl-delay: 1

Sitemap: https://example.com/sitemap.xml
\`\`\`

**Implementation**:

\`\`\`python
class RobotsCache:
    def __init__(self):
        self.cache = {}  # domain → RobotsRules
        self.ttl = 86400  # 24 hours
    
    def get_rules(self, domain):
        if domain in self.cache:
            rules, timestamp = self.cache[domain]
            if time.time() - timestamp < self.ttl:
                return rules
        
        # Fetch robots.txt
        robots_url = f"https://{domain}/robots.txt"
        response = requests.get(robots_url, timeout=5)
        rules = parse_robots_txt(response.text)
        self.cache[domain] = (rules, time.time())
        return rules
    
    def can_crawl(self, url):
        domain = extract_domain(url)
        rules = self.get_rules(domain)
        path = extract_path(url)
        return rules.is_allowed(path, user_agent="MyBot")

robots_cache = RobotsCache()

def fetch_url(url):
    if not robots_cache.can_crawl(url):
        return None  # Respect robots.txt
    
    return requests.get(url)
\`\`\`

**Crawl-delay**: Wait specified seconds between requests.

\`\`\`python
crawl_delay = rules.get_crawl_delay("MyBot")  # Returns 2
# Wait 2 seconds before next request to this domain
\`\`\`

---

## Step 7: Crawler Worker (Fetcher)

\`\`\`python
class CrawlerWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.session = requests.Session()  # Connection pooling
        self.user_agent = "MyBot/1.0"
    
    def run(self):
        while True:
            # Get next URL from frontier
            url = url_frontier.get_next_url()
            if not url:
                time.sleep(1)  # No URLs ready (politeness delays)
                continue
            
            # Check robots.txt
            if not robots_cache.can_crawl(url):
                continue
            
            # Check if already crawled (Bloom filter)
            if not should_crawl(url):
                continue
            
            # Fetch HTML
            try:
                response = self.session.get(
                    url,
                    timeout=10,
                    headers={"User-Agent": self.user_agent}
                )
                
                if response.status_code != 200:
                    # Log failure, might retry later
                    continue
                
                html = response.text
                
                # Check content duplication
                if is_duplicate_content(html):
                    continue
                
                # Store HTML in S3
                store_content(url, html)
                
                # Extract links
                links = extract_links(html, base_url=url)
                
                # Add links to frontier
                for link in links:
                    priority = calculate_priority(link)
                    url_frontier.add_url(link, priority)
                
            except Exception as e:
                # Handle timeouts, connection errors
                log_error(url, e)
\`\`\`

---

## Step 8: Link Extraction

\`\`\`python
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def extract_links(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    
    # Find all <a> tags
    for tag in soup.find_all('a', href=True):
        href = tag['href']
        
        # Convert relative URL to absolute
        absolute_url = urljoin(base_url, href)
        
        # Filter out unwanted URLs
        if not is_valid_url(absolute_url):
            continue
        
        links.append(absolute_url)
    
    return links

def is_valid_url(url):
    # Filter out non-HTTP protocols
    if not url.startswith(("http://", "https://")):
        return False
    
    # Filter out media files (we only want HTML)
    if url.endswith((".jpg", ".png", ".pdf", ".mp4")):
        return False
    
    # Filter out query strings with session IDs (likely duplicates)
    if "sessionid=" in url.lower() or "jsessionid=" in url.lower():
        return False
    
    return True
\`\`\`

---

## Step 9: URL Prioritization

**Goal**: Crawl important pages first (news, popular sites).

**Factors**:

1. **PageRank**: Measure importance based on incoming links
2. **Update Frequency**: News sites re-crawled daily, static sites monthly
3. **Traffic**: Popular sites prioritized
4. **User Signals**: Bookmarks, search clicks

**Simple Scoring**:

\`\`\`python
def calculate_priority(url):
    score = 0.5  # Base score
    
    # Domain reputation (pre-computed)
    domain = extract_domain(url)
    if domain in high_value_domains:  # ["wikipedia.org", "nytimes.com", ...]
        score += 0.3
    
    # Freshness (pages not crawled recently)
    last_crawl = get_last_crawl_time(url)
    if last_crawl:
        days_since = (time.now() - last_crawl) / 86400
        score += min(days_since / 30, 0.2)  # Max +0.2 for old pages
    
    # URL depth (shorter paths often more important)
    depth = url.count("/") - 2  # Subtract protocol://
    score -= depth * 0.05
    
    return max(0, min(1, score))  # Clamp to [0, 1]
\`\`\`

---

## Step 10: Handling Dynamic Content (JavaScript)

**Problem**: Modern sites use JavaScript to render content (React, Vue).

**Challenge**: Traditional crawlers fetch HTML, but content loaded via AJAX isn't visible.

**Solution 1: Headless Browser** (Expensive)

\`\`\`python
from selenium import webdriver

def fetch_dynamic_content(url):
    driver = webdriver.Chrome(options=headless_options)
    driver.get(url)
    time.sleep(3)  # Wait for JavaScript to execute
    html = driver.page_source
    driver.quit()
    return html
\`\`\`

**Pros**:
- ✅ Renders JavaScript, gets full content

**Cons**:
- ❌ Slow (3-10 seconds per page)
- ❌ Resource-intensive (100x more CPU than plain HTTP)
- ❌ Not scalable for billions of pages

**Solution 2: API Detection** (Smart)

\`\`\`python
# Detect AJAX API calls by analyzing JavaScript
# Example: Site loads data from https://api.example.com/posts
# Directly call API instead of rendering page

def detect_apis(html):
    # Parse JavaScript, find fetch() or XMLHttpRequest calls
    apis = regex_find_api_calls(html)
    return apis

apis = detect_apis(html)
for api_url in apis:
    data = requests.get(api_url).json()
    # Extract structured data directly
\`\`\`

**Hybrid Approach**:
- Crawl static HTML normally (99% of pages)
- Use headless browser for high-value dynamic pages (1%)

---

## Step 11: Distributed Crawling

**Scale**: Single machine can't handle 3,858 pages/second.

**Architecture**: Distribute across 1000 worker machines.

\`\`\`
┌─────────────────┐
│Coordinator (Master)│
│  - Manages URL Frontier
│  - Assigns URLs to workers
│  - Monitors progress
└────────┬────────┘
         │
         ├──────────────────┬──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│Worker Node 1   │  │Worker Node 2   │  │Worker Node 1000│
│ - Fetch URLs   │  │ - Fetch URLs   │  │ - Fetch URLs   │
│ - Extract links│  │ - Extract links│  │ - Extract links│
│ - Store content│  │ - Store content│  │ - Store content│
└────────────────┘  └────────────────┘  └────────────────┘
\`\`\`

**URL Partitioning** (Consistent Hashing):

\`\`\`python
def assign_url_to_worker(url, num_workers=1000):
    domain = extract_domain(url)
    worker_id = hash(domain) % num_workers
    return worker_id

# All URLs from same domain go to same worker
# Ensures politeness (one worker controls rate per domain)
\`\`\`

**Benefits**:
- ✅ Same domain always handled by same worker (politeness)
- ✅ Fault tolerance: If worker fails, reassign its domains
- ✅ Load balancing: Distribute domains evenly

---

## Step 12: Fault Tolerance

**Scenarios**:

**1. Worker Failure**:
\`\`\`
- Coordinator detects heartbeat timeout
- Reassign URLs from failed worker to healthy workers
- Failed worker's progress lost (acceptable—URLs re-crawled)
\`\`\`

**2. Network Timeout**:
\`\`\`
- Set timeout: 10 seconds per request
- If timeout: Mark URL as "failed", retry later (exponential backoff)
- Max retries: 3, then skip URL
\`\`\`

**3. Rate Limiting (429 HTTP)**:
\`\`\`
- Server returns 429 Too Many Requests
- Back off: Double delay (1s → 2s → 4s → 8s)
- Respect Retry-After header if provided
\`\`\`

**4. Corrupt HTML**:
\`\`\`
- Parser crashes on malformed HTML
- Catch exception, log URL, skip page
- Use robust parser (BeautifulSoup with lxml)
\`\`\`

---

## Step 13: Data Storage

**HTML Storage** (S3):

\`\`\`
s3://webcrawl/2025/01/15/example.com/page1.html.gz
\`\`\`

**Structure**:
\`\`\`
- Bucket: webcrawl
- Key: YYYY/MM/DD/domain/hash.html.gz
- Compress HTML (gzip) before storing
- Append-only (immutable for audit)
\`\`\`

**Metadata Database** (PostgreSQL):

\`\`\`sql
CREATE TABLE crawled_urls (
    url VARCHAR(2048) PRIMARY KEY,
    domain VARCHAR(255),
    first_crawled_at TIMESTAMP,
    last_crawled_at TIMESTAMP,
    crawl_count INT,
    status_code INT,
    content_hash VARCHAR(32),
    s3_key VARCHAR(512),
    INDEX idx_domain (domain),
    INDEX idx_last_crawled (last_crawled_at)
);
\`\`\`

---

## Step 14: Re-crawling Strategy

**Problem**: Pages change over time. Need to re-crawl for freshness.

**Frequencies**:
- News sites: Every 1 hour
- Blogs: Every 1 day
- Static sites: Every 30 days

**Priority Re-crawl**:

\`\`\`sql
-- Find pages that need re-crawling
SELECT url FROM crawled_urls
WHERE last_crawled_at < NOW() - INTERVAL '1 DAY'
AND domain IN ('nytimes.com', 'cnn.com')  -- High-priority domains
ORDER BY last_crawled_at ASC
LIMIT 10000;
\`\`\`

**Change Detection**:

\`\`\`python
# On re-crawl, check if content changed
new_hash = content_hash(new_html)
old_hash = db.get_content_hash(url)

if new_hash == old_hash:
    # No change, update timestamp only
    db.update_last_crawled(url)
else:
    # Content changed, store new version
    store_content(url, new_html)
    db.update(url, new_hash)
\`\`\`

---

## Step 15: Monitoring & Observability

**Key Metrics**:

1. **Crawl Rate**: Pages/second (target: 3,858)
2. **Queue Size**: URLs in frontier (should not grow unbounded)
3. **Duplicate Rate**: % of URLs filtered by Bloom filter
4. **HTTP Status Distribution**: 200 (success) vs 404/500 (errors)
5. **Average Fetch Time**: Latency per page
6. **Robots.txt Cache Hit Rate**: Avoid re-fetching robots.txt

**Alerts**:
- Crawl rate drops below 3000 pages/sec → Scale workers
- Queue size > 1 billion → Pause discovery, focus on crawling
- Error rate > 10% → Investigate network/server issues

---

## Trade-offs

**Breadth-First vs Depth-First**:
- **BFS**: Discover many domains quickly (preferred for web crawler)
- **DFS**: Deep into single domain (not useful for general crawling)

**Politeness vs Speed**:
- Respecting 1 req/sec per domain slows crawling
- But necessary to avoid IP bans, legal issues

**Bloom Filter False Positives**:
- 1% false positive = skip 1% of unique URLs
- Acceptable trade-off for 99% memory savings

**Headless Browser**:
- Renders JavaScript, but 100x slower
- Reserve for high-value pages only

---

## Interview Tips

**Clarify**:
1. Scale: Millions or billions of pages?
2. Content types: HTML only, or PDFs/images?
3. Real-time: Crawl continuously or batch jobs?
4. Politeness: Respect robots.txt? Rate limits?

**Emphasize**:
1. **URL Frontier**: Priority queue with politeness
2. **Bloom Filter**: Memory-efficient duplicate detection
3. **Robots.txt**: Always check before crawling
4. **Distributed System**: Consistent hashing for domain partitioning
5. **Fault Tolerance**: Handle timeouts, retries, failures

**Common Mistakes**:
- No duplicate detection (crawl same URL repeatedly)
- No politeness (overload target servers, get banned)
- Single-threaded (too slow for scale)
- Ignoring robots.txt (legal/ethical issues)

**Follow-up Questions**:
- "How to handle infinite loops? (Limit crawl depth to 10 levels)"
- "What if site uses CAPTCHAs? (Skip, or use CAPTCHA-solving service)"
- "How to detect spam pages? (ML classifier, blacklist domains)"
- "How to crawl dark web? (Tor network, different seed URLs)"

---

## Summary

**Core Components**:
1. **URL Frontier**: Multi-queue with priority and politeness
2. **Crawler Workers**: Distributed fetchers (1000 machines)
3. **Bloom Filter**: Duplicate URL detection (12.5 GB for 100B URLs)
4. **Robots.txt Cache**: Respect crawl rules, rate limits
5. **Content Storage**: S3 for HTML, PostgreSQL for metadata
6. **Link Extractor**: Parse HTML, discover new URLs
7. **Re-crawl Scheduler**: Keep content fresh (news sites daily)

**Key Decisions**:
- ✅ Bloom filter for duplicate detection (99% memory savings)
- ✅ Consistent hashing for domain partitioning (ensures politeness)
- ✅ Priority queue for important pages first (PageRank-like)
- ✅ BFS traversal for broad coverage
- ✅ Exponential backoff for failed requests
- ✅ Robots.txt compliance (legal, ethical)
- ✅ Headless browser for 1% of pages (dynamic content)

**Capacity**:
- 10 billion pages crawled in 30-day cycle
- 3,858 pages/second crawl rate
- 1000 distributed worker machines
- 164 TB/day bandwidth consumption
- 1.7 PB storage (compressed HTML)
- 20 TB URL frontier storage

A well-designed web crawler balances **speed** (billions of pages), **politeness** (respect server limits), and **efficiency** (minimize duplicates, storage costs) to systematically index the web for search engines.`,
};
