/**
 * Design Web Crawler Multiple Choice Questions
 */

export const webCrawlerMultipleChoice = [
  {

    id: 'web-crawler-q1',

    
    question:
      "You're implementing URL normalization for your crawler's duplicate detection. Which of the following URLs should be considered duplicates after proper normalization: (1) 'HTTP://Example.com:80/Path?b=2&a=1#section', (2) 'http://example.com/path?a=1&b=2', (3) 'http://example.com:443/path?a=1&b=2'?",
    options: [
      'Only (1) and (2) are duplicates; (3) uses HTTPS port and is different',
      'All three are duplicates after normalization (lowercase, port removal, query sort, fragment removal)',
      'None are duplicates because they have different original forms',
      '(1) and (2) are duplicates; (3) is different because 443 is HTTPS default, not HTTP',
    ],
    correctAnswer: 3,
    explanation:
      "The correct answer is that (1) and (2) are duplicates, but (3) is different due to port/protocol mismatch. Proper URL normalization includes: lowercase ('HTTP' → 'http'), remove default ports (HTTP port 80, HTTPS port 443), sort query parameters ('b=2&a=1' → 'a=1&b=2'), remove fragments ('#section'), and remove trailing slashes. Applying this to (1): 'HTTP://Example.com:80/Path?b=2&a=1#section' becomes 'http://example.com/path?a=1&b=2' (lowercase domain, remove :80 default port, sort query params, remove fragment, keep path case). URL (2) is already in normalized form: 'http://example.com/path?a=1&b=2'. These match, so they're duplicates. However, (3) 'http://example.com:443/path?a=1&b=2' specifies port 443 (HTTPS default) on an HTTP scheme, which is unusual and semantically different—port 443 is not the default for HTTP (port 80 is), so it should NOT be removed. This URL is non-standard (likely misconfigured) and represents a different resource. In practice, normalizers may convert 'http://domain:443' to 'https://domain' if following HTTP → HTTPS upgrade rules, but strict normalization treats different scheme/port combinations as distinct URLs to avoid false positive duplicates.",
  },
  {

    id: 'web-crawler-q2',

    
    question:
      "Your crawler's back-queue system maintains per-domain rate limiting (1 request/second). Worker 1 last accessed 'example.com' at timestamp 1000. Worker 2 has a URL for 'example.com' and checks at timestamp 1000.5. What should happen in a properly designed distributed crawler?",
    options: [
      'Worker 2 proceeds immediately since 0.5 seconds have passed since last access',
      'Worker 2 waits until timestamp 1001 (1 full second) before accessing example.com',
      "This scenario shouldn't occur—consistent hashing ensures only Worker 1 handles example.com",
      'Worker 2 checks a distributed lock in Redis and proceeds if the lock is available',
    ],
    correctAnswer: 2,
    explanation:
      "The correct answer is that this scenario shouldn't occur with proper architecture. In a well-designed distributed crawler, consistent hashing assigns all URLs from a domain to the same worker: worker_id = hash('example.com') % num_workers. This ensures that only one worker (e.g., Worker 1) handles all example.com URLs, making politeness enforcement simple—Worker 1 maintains a local last_access timestamp for example.com and ensures 1 second elapses between requests. If Worker 2 also had example.com URLs, we'd face a distributed coordination problem: both workers would need to synchronize access via a distributed lock (Redis) or shared state, adding latency and complexity. Consistent hashing eliminates this by partitioning domains deterministically—Worker 1 owns example.com, Worker 2 owns other domains (e.g., wikipedia.org). This approach has multiple benefits: (1) no distributed locking overhead, (2) local rate limiting is sufficient, (3) fault tolerance is simpler (if Worker 1 fails, reassign its domains to Worker 3), and (4) cache locality (Worker 1 caches example.com's robots.txt). Option A is wrong because 0.5 seconds violates the 1 req/sec limit. Option B would work but requires distributed coordination. Option D (Redis lock) adds unnecessary latency. The architectural principle: use consistent hashing to partition work and avoid distributed coordination when possible.",
  },
  {

    id: 'web-crawler-q3',

    
    question:
      "A crawler's Bloom filter has 100 billion bits and uses 7 hash functions to track 100 billion URLs. After running for 2 weeks, it has processed 50 billion URLs. A new URL hashes to positions [1234, 5678, 9012, 3456, 7890, 2345, 6789], and all 7 bits are set to 1. What can you conclusively determine?",
    options: [
      'This URL was definitely crawled before',
      "This URL was probably crawled before, but there's a small chance it's a false positive",
      'This URL was not crawled before; the bits were set by other URLs (hash collision)',
      'Cannot determine anything without checking the database',
    ],
    correctAnswer: 1,
    explanation:
      "The correct answer is that the URL was probably crawled before, but there's a chance of a false positive. Bloom filters work by setting k bits (7 in this case) for each inserted item. To check membership, we hash the item and verify all k bits are set. If all bits are 1, the item is 'probably present'—but there's a chance that those 7 specific bit positions were set by different URLs (hash collisions). The false positive rate is calculated as (1 - e^(-kn/m))^k where k=7, n=50B insertions, m=100B bits: (1 - e^(-7*50B/100B))^7 = (1 - e^(-3.5))^7 ≈ (0.97)^7 ≈ 0.81... wait, let me recalculate: (1 - e^(-7*50/100))^7 = (1 - e^(-3.5))^7 = (1 - 0.0302)^7 = (0.9698)^7 = 0.805, so ~80% of queries will show 'present' even for non-inserted items when the filter is this full. This is quite high! In practice, with n=50B URLs in m=100B bits, the filter is 50% full (many collisions), so false positive rate is significant. However, the key insight is: if ALL 7 bits are 1, we cannot be certain—it's probabilistic. Only if ANY bit is 0 can we be 100% certain the URL was NOT crawled (no false negatives in Bloom filters). Option A is wrong because we can't be certain. Option C contradicts the Bloom filter principle. Option D is inefficient—the whole point of Bloom filters is to avoid database checks for most URLs. The correct strategy: treat 'probably seen' as 'skip,' accepting 1-5% false positive rate as a trade-off for massive memory savings.",
  },
  {

    id: 'web-crawler-q4',

    
    question:
      "Your crawler encounters robots.txt with 'Crawl-delay: 5' for your user agent. You have 1000 URLs queued for this domain. Without changing the crawl-delay, what's the minimum time required to crawl all 1000 URLs?",
    options: [
      '5 seconds (crawl-delay applies to the entire batch, not per-URL)',
      '1000 seconds (1 URL every 1 second, interpreting crawl-delay as minimum interval)',
      '5000 seconds (1 URL every 5 seconds as specified by crawl-delay)',
      'Depends on parallelization—multiple workers can crawl simultaneously if using different IPs',
    ],
    correctAnswer: 2,
    explanation:
      "The correct answer is 5000 seconds. The robots.txt 'Crawl-delay' directive specifies the number of seconds to wait between successive requests to the same domain. 'Crawl-delay: 5' means: after fetching a URL from example.com, wait 5 seconds before fetching the next URL from example.com. For 1000 URLs, the sequence is: fetch URL1 at t=0, wait 5 seconds, fetch URL2 at t=5, wait 5 seconds, fetch URL3 at t=10, ..., fetch URL1000 at t=4995. Total time: 999 intervals × 5 seconds = 4995 seconds ≈ 5000 seconds. Option A misunderstands the directive—crawl-delay is per-request, not per-batch. Option B confuses crawl-delay with rate limiting (requests/second); crawl-delay is an absolute wait time between requests. Option D raises an important point about parallelization: technically, if you use multiple workers with different IP addresses, you could crawl faster. However, this violates the spirit of politeness—robots.txt is the website's request for respectful crawling, and using multiple IPs to circumvent crawl-delay is considered aggressive behavior that could lead to IP bans or legal issues. Ethical crawlers respect the crawl-delay directive across all their infrastructure. Some crawlers may slightly parallelize (e.g., 2 concurrent connections with 5-second delay, reducing total time to ~2500 seconds), but dramatically increasing concurrency defeats the purpose of the directive. Best practice: single-threaded crawling per domain with strict adherence to crawl-delay.",
  },
  {

    id: 'web-crawler-q5',

    
    question:
      "Your distributed crawler uses consistent hashing to assign domains to workers: worker_id = hash(domain) % 1000. Worker 237 fails. How should the system handle URLs from Worker 237's assigned domains (e.g., wikipedia.org, reddit.com)?",
    options: [
      'Recompute consistent hash with 999 workers, redistributing ALL domains across remaining workers',
      'Use a virtual nodes approach: each domain hashes to multiple workers, failover to backup workers',
      "Temporarily pause crawling Worker 237's domains until it recovers",
      "Reassign Worker 237's domains to a replacement worker (Worker 1001) maintaining the same hash assignments",
    ],
    correctAnswer: 1,
    explanation:
      "The correct answer is to use virtual nodes (vnodes) with consistent hashing for graceful failover. Standard modulo hashing (hash(domain) % 1000) is problematic when workers fail because removing Worker 237 would require recalculating with 999 workers, causing massive redistribution: nearly every domain would rehash to a different worker, invalidating cached state (robots.txt, politeness timers, Bloom filters). A better approach uses consistent hashing with virtual nodes: each worker is assigned multiple positions on a hash ring (e.g., Worker 237 gets positions hash('Worker237-vnode1'), hash('Worker237-vnode2'), ..., hash('Worker237-vnode100')). Domains are placed on the ring via hash(domain), and each domain is assigned to the next 2-3 workers clockwise (primary, backup1, backup2). When Worker 237 fails, only domains primarily assigned to it need reassignment—they failover to their backup workers already on the ring. This minimizes disruption: ~0.1% of domains (Worker 237's share) move, rather than 100% with modulo hashing. Option A (recompute % 999) causes mass redistribution. Option C (pause) reduces crawl throughput unnecessarily. Option D (replacement worker) works but doesn't handle scaling (adding workers) gracefully—vnodes handle both failures and scaling uniformly. Implementation: use a library like HashRing or implement your own with virtual nodes (100-200 vnodes per worker), ensuring smooth failover, load balancing, and minimal disruption when the worker pool changes.",
  },
];
