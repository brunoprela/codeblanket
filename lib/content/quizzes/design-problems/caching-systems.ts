/**
 * Quiz questions for Caching Systems section
 */

export const cachingsystemsQuiz = [
  {
    id: 'q1',
    question:
      'Why does LRU Cache require a doubly linked list instead of a singly linked list? Explain in detail.',
    sampleAnswer:
      'LRU Cache requires doubly linked list because we need to remove arbitrary nodes from the middle in O(1) time. When we access a key via get(), we find the node using HashMap in O(1), then must move that node to the front. Moving requires: (1) Remove node from current position, (2) Insert at front. With doubly linked list, removal is O(1): node.prev.next = node.next; node.next.prev = node.prev - we have direct access to both neighbors. With singly linked list, we only have node.next, so to remove a node we need to find its previous node, which requires O(N) traversal from head. Since get() must be O(1), we cannot afford O(N) removal. The extra prev pointer in doubly linked list is essential.',
    keyPoints: [
      'Need to remove arbitrary nodes in O(1)',
      'get() finds node via HashMap, must move to front',
      'Doubly linked: node.prev gives O(1) removal',
      'Singly linked: need O(N) to find previous',
      'Extra prev pointer essential for O(1)',
    ],
  },
  {
    id: 'q2',
    question:
      'When would you choose LFU over LRU, and vice versa? Give concrete examples.',
    sampleAnswer:
      'Choose LRU for most general-purpose caching: web browsers (recently visited pages likely revisited), database query caches, API caches. LRU handles temporal locality well - if something was used recently, it will likely be used again soon. Choose LFU when access patterns are heavily skewed with "hot" items: video streaming (popular videos get 80% of views), CDNs (viral content), autocomplete (common words). LFU prevents cache pollution - one sequential scan won\'t evict all your hot items. For example, if scanning through IDs 1-1000 once but repeatedly accessing a few hot items, LRU would evict the hot items (temporarily not accessed), while LFU keeps them (high frequency). However, LFU is more complex to implement and can struggle with changing patterns.',
    keyPoints: [
      'LRU: general purpose, temporal locality',
      'LRU examples: browsers, query cache',
      'LFU: skewed access patterns, hot items',
      'LFU examples: video streaming, CDN',
      'LFU resistant to cache pollution',
    ],
  },
  {
    id: 'q3',
    question:
      'Explain why the "dummy head and tail" technique simplifies LRU Cache implementation.',
    sampleAnswer:
      'Dummy head and tail eliminate all edge case checks when adding/removing nodes. Without dummies, adding to an empty list requires "if head is None: head = tail = new_node", and removing last node requires "if head == tail: head = tail = None", etc. With dummies, the list is NEVER empty (always has head and tail), so adding after head is always "new_node.next = head.next; head.next.prev = new_node; head.next = new_node" with no conditionals. Removing is always "prev.next = next; next.prev = prev". The actual data nodes are between head and tail. This eliminates null checks, simplifies code, and prevents edge case bugs. It\'s a standard technique for doubly linked lists in production code.',
    keyPoints: [
      'Eliminates null checks and edge cases',
      'List never empty (always has dummies)',
      'Add/remove operations have no conditionals',
      'Data nodes always between head and tail',
      'Standard technique in production code',
    ],
  },
];
