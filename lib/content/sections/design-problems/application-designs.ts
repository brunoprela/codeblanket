/**
 * Application Designs Section
 */

export const applicationdesignsSection = {
  id: 'application-designs',
  title: 'Application Designs',
  content: `Application design problems ask you to implement real-world features or components. These test your ability to combine data structures, handle state, and design clean APIs.

---

## Design Browser History

**Problem**: Implement browser back/forward functionality.

**Operations**:
- \`visit(url)\`: Visit a URL, clearing forward history
- \`back(steps)\`: Go back \`steps\` pages
- \`forward(steps)\`: Go forward \`steps\` pages

### Approach 1: Two Stacks

**Idea**: Back stack for history, forward stack for forward navigation.

\`\`\`python
class BrowserHistory:
    def __init__(self, homepage):
        self.back_stack = [homepage]
        self.forward_stack = []
        self.current = homepage
    
    def visit(self, url):
        # Clear forward history
        self.forward_stack = []
        # Push current to back stack
        self.back_stack.append(self.current)
        self.current = url
    
    def back(self, steps):
        # Move from current/back to forward
        while steps > 0 and len(self.back_stack) > 1:
            self.forward_stack.append(self.current)
            self.current = self.back_stack.pop()
            steps -= 1
        return self.current
    
    def forward(self, steps):
        # Move from forward to current/back
        while steps > 0 and self.forward_stack:
            self.back_stack.append(self.current)
            self.current = self.forward_stack.pop()
            steps -= 1
        return self.current
\`\`\`

**Time**: O(1) per step  
**Space**: O(N) where N is number of visited pages

### Approach 2: Array with Pointer

**Idea**: Store all pages in array, track current position.

\`\`\`python
class BrowserHistory:
    def __init__(self, homepage):
        self.history = [homepage]
        self.current_idx = 0
    
    def visit(self, url):
        # Remove everything after current
        self.history = self.history[:self.current_idx + 1]
        self.history.append(url)
        self.current_idx += 1
    
    def back(self, steps):
        self.current_idx = max(0, self.current_idx - steps)
        return self.history[self.current_idx]
    
    def forward(self, steps):
        self.current_idx = min(len(self.history) - 1, 
                               self.current_idx + steps)
        return self.history[self.current_idx]
\`\`\`

**Simpler**, same complexity.

---

## Design Twitter (Simplified)

**Problem**: Implement core Twitter features.

**Operations**:
- \`postTweet(userId, tweetId)\`: User posts a tweet
- \`getNewsFeed(userId)\`: Get 10 most recent tweets from user + followees
- \`follow(followerId, followeeId)\`
- \`unfollow(followerId, followeeId)\`

### Solution: HashMap + Heap

\`\`\`python
from collections import defaultdict
import heapq

class Twitter:
    def __init__(self):
        self.tweets = defaultdict(list)  # userId -> [(timestamp, tweetId)]
        self.following = defaultdict(set)  # userId -> set of followees
        self.timestamp = 0
    
    def postTweet(self, userId, tweetId):
        self.tweets[userId].append((self.timestamp, tweetId))
        self.timestamp += 1
    
    def getNewsFeed(self, userId):
        # Get tweets from user + followees
        max_heap = []
        
        # Add own tweets (last 10)
        for tweet in self.tweets[userId][-10:]:
            heapq.heappush(max_heap, (-tweet[0], tweet[1]))
        
        # Add followees' tweets
        for followeeId in self.following[userId]:
            for tweet in self.tweets[followeeId][-10:]:
                heapq.heappush(max_heap, (-tweet[0], tweet[1]))
        
        # Extract top 10
        result = []
        while max_heap and len(result) < 10:
            result.append(heapq.heappop(max_heap)[1])
        
        return result
    
    def follow(self, followerId, followeeId):
        if followerId != followeeId:  # Can't follow self
            self.following[followerId].add(followeeId)
    
    def unfollow(self, followerId, followeeId):
        self.following[followerId].discard(followeeId)
\`\`\`

**Time Complexity**:
- postTweet: O(1)
- getNewsFeed: O(N log K) where N = total tweets to consider, K = feed size
- follow/unfollow: O(1)

**Space**: O(users * tweets)

**Key Insight**: Use max heap to merge sorted timelines efficiently. Each user's tweets are in timestamp order, so we can merge K sorted lists.

---

## Design Search Autocomplete

**Problem**: Implement search autocomplete with top K suggestions.

**Operations**:
- \`input(c)\`: User types character \`c\`, return top suggestions
- \`recordSearch(query)\`: Record completed search (update frequencies)

### Solution: Trie + Frequency Tracking

\`\`\`python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.frequency = 0  # How many times this search was completed
        self.query = ""

class AutocompleteSystem:
    def __init__(self, sentences, times):
        self.root = TrieNode()
        self.current_input = ""
        
        # Build trie with initial data
        for sentence, freq in zip(sentences, times):
            self.add_to_trie(sentence, freq)
    
    def add_to_trie(self, sentence, frequency):
        node = self.root
        for char in sentence:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.query = sentence
        node.frequency += frequency
    
    def search_with_prefix(self, prefix):
        node = self.root
        # Navigate to prefix
        for char in prefix:
            if char not in node.children:
                return []  # No matches
            node = node.children[char]
        
        # Find all completions from this node
        results = []
        self.dfs_collect(node, results)
        
        # Sort by frequency (desc), then lexicographically
        results.sort(key=lambda x: (-x[1], x[0]))
        return [query for query, freq in results[:3]]  # Top 3
    
    def dfs_collect(self, node, results):
        if node.is_end:
            results.append((node.query, node.frequency))
        for child in node.children.values():
            self.dfs_collect(child, results)
    
    def input(self, c):
        if c == '#':
            # End of query, record it
            self.add_to_trie(self.current_input, 1)
            self.current_input = ""
            return []
        else:
            self.current_input += c
            return self.search_with_prefix(self.current_input)
\`\`\`

**Time Complexity**:
- input: O(p + m log m) where p = prefix length, m = matching queries
- Optimized: Store top K at each node (precompute)

**Space**: O(total characters in all queries)

**Optimization**: Store top K results at each Trie node:
\`\`\`python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.top_k = []  # Precomputed top K queries from this prefix
\`\`\`

Then input() is O(p) - just navigate to prefix and return cached top_k.

---

## Pattern Recognition

### When to Use Multiple Data Structures:
- Twitter: HashMap for users + Heap for timeline merge
- Autocomplete: Trie for prefix search + Sorting for ranking
- Browser: Stack for navigation + might add HashMap for bookmarks

### State Management:
- **Mutable state**: Tweets, follows, history
- **Derived state**: News feed, autocomplete suggestions
- **Transient state**: Current input, navigation position

### API Design Principles:
1. **Clear names**: \`postTweet\` not \`add\`, \`getNewsFeed\` not \`get\`
2. **Consistent return types**: Always list, always id
3. **Edge case handling**: Can't follow self, empty history
4. **Efficient operations**: What's called frequently? (getNewsFeed - optimize!)

---

## Interview Tips

1. **Clarify scale**: "How many users? Tweets per user? Follows per user?"

2. **Start with data structures**: "I'll use HashMap for users, list for tweets..."

3. **Think about queries**: "GetNewsFeed is called frequently, so I'll optimize that"

4. **Handle edge cases**: "Can user follow themselves? What if no tweets?"

5. **Discuss improvements**: "In production, we'd cache news feeds, use database..."

**Common Mistakes**:
- Forgetting to clear forward history on visit() in Browser History
- Not handling "can't follow self" in Twitter
- Inefficient autocomplete (linear search instead of Trie)
- Not sorting by frequency in autocomplete`,
};
