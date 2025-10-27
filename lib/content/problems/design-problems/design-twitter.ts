/**
 * Design Twitter
 * Problem ID: design-twitter
 * Order: 8
 */

import { Problem } from '../../../types';

export const design_twitterProblem: Problem = {
  id: 'design-twitter',
  title: 'Design Twitter',
  difficulty: 'Medium',
  topic: 'Design Problems',
  description: `Design a simplified version of Twitter where users can post tweets, follow/unfollow another user, and is able to see the \`10\` most recent tweets in the user's news feed.

Implement the \`Twitter\` class:

- \`Twitter()\` Initializes your twitter object.
- \`void postTweet(int userId, int tweetId)\` Composes a new tweet with ID \`tweetId\` by the user \`userId\`. Each call to this function will be made with a unique \`tweetId\`.
- \`List<Integer> getNewsFeed(int userId)\` Retrieves the \`10\` most recent tweet IDs in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user themself. Tweets must be **ordered from most recent to least recent**.
- \`void follow(int followerId, int followeeId)\` The user with ID \`followerId\` started following the user with ID \`followeeId\`.
- \`void unfollow(int followerId, int followeeId)\` The user with ID \`followerId\` started unfollowing the user with ID \`followeeId\`.`,
  hints: [
    'Store tweets per user in chronological order',
    'Use timestamp to order tweets across users',
    'Max heap to merge K sorted lists (user timelines)',
    'Use set for following (O(1) add/remove)',
  ],
  approach: `## Intuition

Twitter features:
1. **Post tweet**: Add to user's timeline
2. **Get news feed**: Merge timelines from user + followees
3. **Follow/Unfollow**: Manage following relationship

**Challenge**: Efficiently get top 10 recent tweets from multiple users.

---

## Approach: HashMap + Heap for Merging

**Data Structures:**1. \`tweets\`: userId -> list of (timestamp, tweetId)
2. \`following\`: userId -> set of followeeIds
3. \`timestamp\`: Global counter for ordering

### getNewsFeed - Merge K Sorted Lists:

Each user's tweets are in timestamp order (sorted). Need to merge multiple sorted lists efficiently.

**Naive**: Collect all tweets, sort → O(N log N)  
**Better**: Max heap → O(K log K) where K = total recent tweets to consider

\`\`\`python
# Get last 10 tweets from user + each followee
# Use max heap to keep top 10 by timestamp
\`\`\`

**Optimization**: Only look at last 10 tweets per user (not all history).

---

## Example:

\`\`\`
postTweet(1, 5):  # User 1 tweets 5
  tweets = {1: [(0, 5)]}

follow(1, 2):     # User 1 follows user 2
  following = {1: {2}}

postTweet(2, 6):  # User 2 tweets 6
  tweets = {1: [(0, 5)], 2: [(1, 6)]}

getNewsFeed(1):   # Get feed for user 1
  # Collect from user 1 and user 2 (followee)
  # Tweets: [(0, 5), (1, 6)]
  # Sort by timestamp desc: [6, 5]
  returns [6, 5]
\`\`\`

---

## Time Complexity:
- postTweet: O(1)
- getNewsFeed: O(F * 10 * log K) where F = followees, K = feed size
- follow/unfollow: O(1)

## Space Complexity: O(U * T) where U = users, T = tweets per user`,
  testCases: [
    {
      input: [
        ['Twitter'],
        ['postTweet', 1, 5],
        ['getNewsFeed', 1],
        ['follow', 1, 2],
        ['postTweet', 2, 6],
        ['getNewsFeed', 1],
        ['unfollow', 1, 2],
        ['getNewsFeed', 1],
      ],
      expected: [null, null, [5], null, null, [6, 5], null, [5]],
    },
  ],
  solution: `from collections import defaultdict
import heapq

class Twitter:
    def __init__(self):
        self.tweets = defaultdict(list)  # userId -> [(timestamp, tweetId)]
        self.following = defaultdict(set)  # userId -> set of followeeIds
        self.timestamp = 0  # Global timestamp for ordering
    
    def postTweet(self, userId: int, tweetId: int) -> None:
        """Post a tweet - O(1)"""
        self.tweets[userId].append((self.timestamp, tweetId))
        self.timestamp += 1
    
    def getNewsFeed(self, userId: int) -> list[int]:
        """Get 10 most recent tweets from user + followees"""
        max_heap = []
        
        # Add own tweets (last 10)
        for timestamp, tweetId in self.tweets[userId][-10:]:
            heapq.heappush(max_heap, (-timestamp, tweetId))
        
        # Add followees' tweets (last 10 each)
        for followeeId in self.following[userId]:
            for timestamp, tweetId in self.tweets[followeeId][-10:]:
                heapq.heappush(max_heap, (-timestamp, tweetId))
        
        # Extract top 10
        result = []
        while max_heap and len(result) < 10:
            _, tweetId = heapq.heappop(max_heap)
            result.append(tweetId)
        
        return result
    
    def follow(self, followerId: int, followeeId: int) -> None:
        """Follow a user - O(1)"""
        if followerId != followeeId:  # Can't follow yourself
            self.following[followerId].add(followeeId)
    
    def unfollow(self, followerId: int, followeeId: int) -> None:
        """Unfollow a user - O(1)"""
        self.following[followerId].discard(followeeId)

# Example usage:
# twitter = Twitter()
# twitter.postTweet(1, 5)
# twitter.getNewsFeed(1)  # returns [5]
# twitter.follow(1, 2)
# twitter.postTweet(2, 6)
# twitter.getNewsFeed(1)  # returns [6, 5]
# twitter.unfollow(1, 2)
# twitter.getNewsFeed(1)  # returns [5]`,
  timeComplexity:
    'postTweet O(1), getNewsFeed O(F * 10 * log K), follow/unfollow O(1)',
  spaceComplexity: 'O(U * T) where U=users, T=average tweets per user',
  patterns: ['HashMap', 'Heap', 'Design', 'Merge K Sorted Lists'],
  companies: ['Twitter', 'Amazon', 'Microsoft', 'Facebook'],
};
