/**
 * Accounts Merge
 * Problem ID: accounts-merge
 * Order: 5
 */

import { Problem } from '../../../types';

export const accounts_mergeProblem: Problem = {
  id: 'accounts-merge',
  title: 'Accounts Merge',
  difficulty: 'Easy',
  topic: 'Advanced Graphs',
  description: `Given a list of \`accounts\` where each element \`accounts[i]\` is a list of strings, where the first element \`accounts[i][0]\` is a name, and the rest of the elements are **emails** representing emails of the account.

Now, we would like to merge these accounts. Two accounts definitely belong to the same person if there is some common email to both accounts. Note that even if two accounts have the same name, they may belong to different people as people could have the same name. A person can have any number of accounts initially, but all of their accounts definitely have the same name.

After merging the accounts, return the accounts in the following format: the first element of each account is the name, and the rest of the elements are emails **in sorted order**. The accounts themselves can be returned in **any order**.`,
  examples: [
    {
      input:
        'accounts = [["John","johnsmith@mail.com","john_newyork@mail.com"],["John","johnsmith@mail.com","john00@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]',
      output:
        '[["John","john00@mail.com","john_newyork@mail.com","johnsmith@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]',
    },
  ],
  constraints: [
    '1 <= accounts.length <= 1000',
    '2 <= accounts[i].length <= 10',
    '1 <= accounts[i][j].length <= 30',
    'accounts[i][0] consists of English letters',
    'accounts[i][j] (for j > 0) is a valid email',
  ],
  hints: [
    'Use Union-Find on emails',
    'Build mapping from email to name',
    'Group emails by root parent',
  ],
  starterCode: `from typing import List

def accounts_merge(accounts: List[List[str]]) -> List[List[str]]:
    """
    Merge accounts with common emails.
    
    Args:
        accounts: List of [name, email1, email2, ...]
        
    Returns:
        Merged accounts
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [
        [
          ['John', 'johnsmith@mail.com', 'john_newyork@mail.com'],
          ['John', 'johnsmith@mail.com', 'john00@mail.com'],
          ['Mary', 'mary@mail.com'],
          ['John', 'johnnybravo@mail.com'],
        ],
      ],
      expected: [
        [
          'John',
          'john00@mail.com',
          'john_newyork@mail.com',
          'johnsmith@mail.com',
        ],
        ['Mary', 'mary@mail.com'],
        ['John', 'johnnybravo@mail.com'],
      ],
    },
  ],
  timeComplexity: 'O(n * k * α(n)) where k is avg emails per account',
  spaceComplexity: 'O(n * k)',
  leetcodeUrl: 'https://leetcode.com/problems/accounts-merge/',
  youtubeUrl: 'https://www.youtube.com/watch?v=wU6udHRIkcc',
};
