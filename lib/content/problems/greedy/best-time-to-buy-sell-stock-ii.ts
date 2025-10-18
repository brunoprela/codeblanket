/**
 * Best Time to Buy and Sell Stock II
 * Problem ID: best-time-to-buy-sell-stock-ii
 * Order: 2
 */

import { Problem } from '../../../types';

export const best_time_to_buy_sell_stock_iiProblem: Problem = {
  id: 'best-time-to-buy-sell-stock-ii',
  title: 'Best Time to Buy and Sell Stock II',
  difficulty: 'Easy',
  description: `You are given an integer array \`prices\` where \`prices[i]\` is the price of a given stock on the \`ith\` day.

On each day, you may decide to buy and/or sell the stock. You can only hold **at most one** share of the stock at any time. However, you can buy it then immediately sell it on the **same day**.

Find and return the **maximum profit** you can achieve.


**Greedy Approach:**
The key insight is that we can capture every upward price movement. Whenever price[i+1] > price[i], we can "buy" at i and "sell" at i+1 to capture that profit. Sum all positive differences.

**Key Insight:**
Any profit can be decomposed into a sum of single-day profits. Instead of finding optimal buy/sell pairs, just collect profit from every price increase.`,
  examples: [
    {
      input: 'prices = [7,1,5,3,6,4]',
      output: '7',
      explanation:
        'Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 4. Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 3. Total = 7.',
    },
    {
      input: 'prices = [1,2,3,4,5]',
      output: '4',
      explanation:
        'Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 4. Or equivalently, capture each daily increase: (2-1)+(3-2)+(4-3)+(5-4) = 4.',
    },
    {
      input: 'prices = [7,6,4,3,1]',
      output: '0',
      explanation: 'There are no profitable transactions, so max profit = 0.',
    },
  ],
  constraints: ['1 <= prices.length <= 10^5', '0 <= prices[i] <= 10^4'],
  hints: [
    'Think about capturing every upward price movement',
    'Whenever prices[i+1] > prices[i], you can make a profit',
    'Sum all positive differences between consecutive days',
    'You do not need to track actual buy / sell pairs',
    'Single pass O(n) solution',
  ],
  starterCode: `from typing import List

def max_profit(prices: List[int]) -> int:
    """
    Find maximum profit from multiple buy-sell transactions.
    
    Args:
        prices: Array of stock prices
        
    Returns:
        Maximum profit possible
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[7, 1, 5, 3, 6, 4]],
      expected: 7,
    },
    {
      input: [[1, 2, 3, 4, 5]],
      expected: 4,
    },
    {
      input: [[7, 6, 4, 3, 1]],
      expected: 0,
    },
  ],
  solution: `from typing import List


def max_profit(prices: List[int]) -> int:
    """
    Greedy: sum all positive price differences.
    Time: O(n), Space: O(1)
    """
    profit = 0
    
    for i in range(1, len(prices)):
        # If price increased, capture the profit
        if prices[i] > prices[i - 1]:
            profit += prices[i] - prices[i - 1]
    
    return profit


# Alternative: More explicit buy/sell tracking
def max_profit_explicit(prices: List[int]) -> int:
    """
    Track valleys (buy) and peaks (sell).
    Time: O(n), Space: O(1)
    """
    if len(prices) <= 1:
        return 0
    
    profit = 0
    i = 0
    
    while i < len(prices) - 1:
        # Find valley (local minimum)
        while i < len(prices) - 1 and prices[i] >= prices[i + 1]:
            i += 1
        valley = prices[i]
        
        # Find peak (local maximum)
        while i < len(prices) - 1 and prices[i] <= prices[i + 1]:
            i += 1
        peak = prices[i]
        
        profit += peak - valley
    
    return profit


# Alternative: One-liner
def max_profit_oneliner(prices: List[int]) -> int:
    """
    Pythonic one-liner solution.
    """
    return sum(max(prices[i] - prices[i-1], 0) for i in range(1, len(prices)))`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',

  leetcodeUrl:
    'https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/',
  youtubeUrl: 'https://www.youtube.com/watch?v=3SJ3pUkPQMc',
  order: 2,
  topic: 'Greedy',
};
