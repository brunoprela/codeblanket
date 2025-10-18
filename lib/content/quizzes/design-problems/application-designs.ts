/**
 * Quiz questions for Application Designs section
 */

export const applicationdesignsQuiz = [
  {
    id: 'q1',
    question:
      'In Design Twitter, why do we use a heap to merge timelines instead of sorting all tweets?',
    sampleAnswer:
      'We use heap because we only need top K (usually 10) tweets, not all tweets sorted. If a user follows 1000 people with 1000 tweets each = 1M tweets, sorting all would be O(1M log 1M) = ~20M operations. With heap: take last 10 tweets from each user (10K tweets), build max heap O(10K), extract top 10 = O(10K log 10K) = ~130K operations, 150x faster! Heap is perfect for "top K from multiple sorted lists" pattern. We only sort the small result set, not everything. This is why Twitter/Facebook feeds load quickly - they don\'t sort your entire timeline, just enough to show the top.',
    keyPoints: [
      'Only need top K, not all sorted',
      'Sorting all: O(N log N) where N = all tweets',
      'Heap: O(N log K) where K = feed size',
      'Huge savings when N >> K',
      'Top K from sorted lists = heap pattern',
    ],
  },
  {
    id: 'q2',
    question:
      'Why is a Trie the best data structure for autocomplete? What alternatives did you consider?',
    sampleAnswer:
      'Trie is optimal for autocomplete because: (1) Prefix search is O(p) where p = prefix length, not dependent on number of queries. (2) All queries sharing prefix stored once (memory efficient). (3) Easy to collect all matches with DFS. Alternatives considered: (1) Array with linear search - O(N) per search, too slow. (2) Sorted array with binary search - O(log N) to find start, but still need to collect all with prefix. (3) HashMap - great for exact match O(1), but cannot efficiently find "all strings starting with X". (4) Suffix tree - overkill, used for substring search not prefix. Trie specializes in prefix operations, making it perfect for autocomplete, dictionary, and spell-check applications.',
    keyPoints: [
      'Prefix search O(p), independent of total queries',
      'Shared prefixes save memory',
      'Natural DFS to collect all matches',
      'Alternatives worse: array O(N), HashMap no prefix',
      'Trie = specialized tool for prefix problems',
    ],
  },
  {
    id: 'q3',
    question:
      'In Browser History, why does visiting a new page clear forward history?',
    sampleAnswer:
      "Visiting new page clears forward history to match expected browser behavior: when you're at page 3, go back to page 2, then visit new page 4, page 3 is no longer accessible via forward button - it's in an alternate timeline. This creates a tree structure where each visit creates a new branch. If we kept forward history, you could visit google.com, go back, visit facebook.com, then forward to google.com, which is confusing - forward should continue from where you were, not jump to alternate timeline. Clearing forward stack maintains linear history property. The alternative (keeping tree of all pages) is complex and not what users expect. Real browsers work this way.",
    keyPoints: [
      'Matches expected browser behavior',
      'Visit creates new branch, old forward is alternate timeline',
      'Maintains linear history (no confusion)',
      'Alternative: tree of pages (complex)',
      'Real browsers clear forward on visit',
    ],
  },
];
