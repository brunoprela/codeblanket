/**
 * Design Web Crawler Quiz
 */

export const webCrawlerQuiz = [
    {
        question: "Your web crawler needs to track 100 billion discovered URLs to prevent duplicate crawling. A naive HashSet would require 20 TB of memory. Explain how to use a Bloom filter to solve this problem efficiently, including the trade-offs involved, false positive rate implications, and why false negatives are impossible in Bloom filters.",
        sampleAnswer: "I would implement a Bloom filter with a bit array of 100 billion bits (12.5 GB) and 7 hash functions. When discovering a URL, normalize it first (lowercase, remove trailing slashes, sort query parameters), then hash it with each of the 7 functions to generate 7 bit positions. Set all 7 bits to 1. To check if a URL was already seen, hash it again and verify all 7 corresponding bits are 1. If any bit is 0, the URL definitely wasn't seen before (no false negatives). If all bits are 1, the URL was probably seen, but there's a ~1% false positive rate—meaning we might skip crawling ~1% of genuinely new URLs because their hash positions collide with previously seen URLs. This trade-off is acceptable because: (1) we save 99.94% memory (12.5 GB vs 20 TB), (2) missing 1% of URLs doesn't significantly impact web coverage, and (3) false negatives (crawling duplicates) would be far worse than false positives (missing some URLs). The false positive rate can be calculated as (1 - e^(-kn/m))^k where k=7 hash functions, n=100B insertions, m=100B bits, yielding ~1%. To reduce false positive rate to 0.1%, we'd need to increase the bit array to 125 billion bits (15.6 GB) or use more hash functions, both reasonable trade-offs for better accuracy.",
        keyPoints: [
            "Bloom filter uses 12.5 GB (100B bits) vs 20 TB for naive set, achieving 99.94% memory savings",
            "Multiple hash functions (7) map each URL to multiple bit positions, providing probabilistic membership testing",
            "False positives (~1%) are acceptable: we skip some valid URLs but save massive memory",
            "False negatives are impossible: if any bit is 0, URL definitely wasn't seen before",
            "URL normalization (lowercase, sorted params) is critical before hashing to detect semantic duplicates"
        ]
    },
    {
        question: "Design a URL frontier system that handles both prioritization (important pages crawled first) and politeness (max 1 request/second per domain). Explain the front-queue/back-queue architecture, how you route URLs between queues, and how you prevent one slow domain from blocking the entire crawler.",
        sampleAnswer: "I would implement a two-tier queue system. The front tier has three priority queues: high-priority (PageRank > 0.8, news sites), medium-priority (0.3-0.8, general web), and low-priority (<0.3, rarely updated pages). When a URL is discovered, calculate its priority score based on domain reputation, freshness (days since last crawl), and URL depth, then enqueue it in the appropriate front queue. Use weighted random selection to dequeue: 70% from high, 20% from medium, 10% from low—this ensures important pages are crawled first while still processing the long tail. The back tier implements per-domain politeness: maintain a separate queue for each domain (e.g., queue_example_com, queue_wikipedia_org) with a last_access timestamp. When dequeuing a URL from the front tier, route it to the corresponding domain's back queue. A worker can only dequeue from a domain's back queue if current_time - last_access >= min_delay (1 second). If all back queues are rate-limited, the worker sleeps 100ms and tries again. This architecture prevents blocking because: (1) workers skip rate-limited domains and check others, (2) high-priority domains get more back queues (partitioned by subdomain), and (3) we can dynamically adjust min_delay per domain based on robots.txt crawl-delay directives. For distributed crawling, use consistent hashing on domain names to assign domains to workers, ensuring all URLs for a domain go to the same worker (maintaining politeness).",
        keyPoints: [
            "Front queues (high/medium/low priority) implement importance-based crawling with weighted random selection",
            "Back queues (per-domain) enforce rate limiting using last_access timestamps and min_delay checks",
            "Two-tier architecture decouples prioritization from politeness, preventing slow domains from blocking important URLs",
            "Consistent hashing assigns domains to workers, ensuring single worker controls rate per domain",
            "Dynamic delays respect robots.txt crawl-delay, adjusting per-domain rate limits automatically"
        ]
    },
    {
        question: "Your crawler discovers that 30% of modern websites use JavaScript frameworks (React, Vue) to load content dynamically, making the content invisible to traditional HTTP-based crawling. Compare the trade-offs of using headless browsers (Selenium, Puppeteer) versus API detection for handling dynamic content, and propose a hybrid strategy that balances coverage and cost.",
        sampleAnswer: "Headless browsers render JavaScript and execute AJAX calls, providing complete content visibility, but they're 100x slower (3-10 seconds per page vs 30-100ms for HTTP) and consume 100x more resources (Chrome process uses ~200 MB RAM vs 2 MB for HTTP client). At 3,858 pages/second target rate, using headless browsers for all pages would require 11,574 concurrent browser instances, costing ~$500K/month in compute. API detection is faster and cheaper: analyze the page's JavaScript to find fetch() or XMLHttpRequest calls (e.g., 'https://api.example.com/posts'), then directly call those APIs to get structured JSON data. This works well for modern SPAs (single-page applications) but requires JavaScript parsing and may miss poorly documented APIs. My hybrid strategy: (1) Use traditional HTTP crawling for 99% of pages (fast, cheap). (2) Detect client-side rendering by checking for telltale signs: minimal HTML (<5KB), large JavaScript bundles, or known framework signatures (React, Vue). (3) For detected dynamic sites, first attempt API extraction by parsing JavaScript for API endpoints. (4) Only use headless browsers for high-value pages where API detection fails (news sites, e-commerce, social media)—roughly 1% of all pages. (5) Cache rendered content for 24 hours to avoid re-rendering unchanged pages. This approach provides 99.9% coverage while keeping headless browser costs to ~$5K/month, a 99% cost reduction compared to rendering everything.",
        keyPoints: [
            "Headless browsers provide complete JavaScript rendering but are 100x slower and more resource-intensive",
            "API detection extracts structured data directly from backend calls, avoiding rendering overhead",
            "Hybrid strategy reserves headless browsers for high-value pages only (1%), reducing costs by 99%",
            "Detecting client-side rendering (minimal HTML, large JS bundles) helps identify pages needing special handling",
            "Caching rendered content (24-hour TTL) prevents redundant re-rendering of unchanged dynamic pages"
        ]
    }
];

