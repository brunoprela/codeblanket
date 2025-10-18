/**
 * Best Time to Buy and Sell Stock
 * Problem ID: best-time-to-buy-sell-stock
 * Order: 1
 */

import { Problem } from '../../../types';

export const best_time_to_buy_sell_stockProblem: Problem = {
  id: 'best-time-to-buy-sell-stock',
  title: 'Best Time to Buy and Sell Stock',
  difficulty: 'Easy',
  description: `You are given an array \`prices\` where \`prices[i]\` is the price of a given stock on the \`i\`th day.

You want to maximize your profit by choosing a **single day** to buy one stock and choosing a **different day in the future** to sell that stock.

Return **the maximum profit** you can achieve from this transaction. If you cannot achieve any profit, return \`0\`.


**Approach:**
This is a sliding window variant. Track the minimum price seen so far (buy price) and calculate profit at each day. The "window" conceptually represents the buying and selling days, expanding to find the maximum profit.`,
  examples: [
    {
      input: 'prices = [7,1,5,3,6,4]',
      output: '5',
      explanation:
        'Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.',
    },
    {
      input: 'prices = [7,6,4,3,1]',
      output: '0',
      explanation:
        'In this case, no transactions are done and the max profit = 0.',
    },
  ],
  constraints: ['1 <= prices.length <= 10^5', '0 <= prices[i] <= 10^4'],
  hints: [
    'Track the minimum price encountered so far',
    'For each price, calculate the profit if we sell at that price',
    'Keep track of the maximum profit seen',
    'You only need one pass through the array - O(N) time',
  ],
  starterCode: `from typing import List

def max_profit(prices: List[int]) -> int:
    """
    Find the maximum profit from buying and selling stock once.
    
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
      expected: 5,
    },
    {
      input: [[7, 6, 4, 3, 1]],
      expected: 0,
    },
    {
      input: [[2, 4, 1]],
      expected: 2,
    },
    {
      input: [[3, 2, 6, 5, 0, 3]],
      expected: 4,
    },
  ],
  solution: `from typing import List

def max_profit(prices: List[int]) -> int:
    """
    Sliding window approach: track min price and max profit.
    Time: O(N), Space: O(1)
    """
    if not prices:
        return 0
    
    min_price = float('inf')
    max_profit = 0
    
    for price in prices:
        # Update minimum price (best day to buy)
        min_price = min(min_price, price)
        
        # Calculate profit if we sell today
        profit = price - min_price
        
        # Update maximum profit
        max_profit = max(max_profit, profit)
    
    return max_profit


# Alternative: More explicit window tracking
def max_profit_window(prices: List[int]) -> int:
    left = 0  # Buy day
    max_profit = 0
    
    for right in range(1, len(prices)):  # Sell day
        # If price decreased, move buy day forward
        if prices[right] < prices[left]:
            left = right
        else:
            # Calculate profit for this window
            profit = prices[right] - prices[left]
            max_profit = max(max_profit, profit)
    
    return max_profit`,
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(1)',

  leetcodeUrl: 'https://leetcode.com/problems/best-time-to-buy-and-sell-stock/',
  youtubeUrl: 'https://www.youtube.com/watch?v=1pkOgXD63yU',
  order: 1,
  topic: 'Sliding Window',
};
