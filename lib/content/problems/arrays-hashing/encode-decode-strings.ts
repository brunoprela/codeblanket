/**
 * Encode and Decode Strings
 * Problem ID: encode-decode-strings
 * Order: 27
 */

import { Problem } from '../../../types';

export const encode_decode_stringsProblem: Problem = {
  id: 'encode-decode-strings',
  title: 'Encode and Decode Strings',
  difficulty: 'Medium',
  topic: 'Arrays & Hashing',
  description: `Design an algorithm to encode a list of strings to a single string. The encoded string is then decoded back to the original list of strings.

Please implement \`encode\` and \`decode\`.`,
  examples: [
    {
      input: 'strs = ["Hello","World"]',
      output: '["Hello","World"]',
      explanation: 'One possible encode method is: "5#Hello5#World".',
    },
    {
      input: 'strs = [""]',
      output: '[""]',
    },
  ],
  constraints: [
    '0 <= strs.length < 200',
    '0 <= strs[i].length < 200',
    'strs[i] contains only UTF-8 characters',
  ],
  hints: [
    'Use a delimiter that cannot appear in the strings',
    'Prefix each string with its length',
    'Use format: length#string',
  ],
  starterCode: `from typing import List

def encode(strs: List[str]) -> str:
    """
    Encode list of strings to a single string.
    
    Args:
        strs: List of strings
        
    Returns:
        Encoded string
    """
    # Write your code here
    pass

def decode(s: str) -> List[str]:
    """
    Decode single string back to list of strings.
    
    Args:
        s: Encoded string
        
    Returns:
        List of original strings
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [['Hello', 'World']],
      expected: ['Hello', 'World'],
    },
    {
      input: [['abc', 'def', 'ghi']],
      expected: ['abc', 'def', 'ghi'],
    },
  ],
  timeComplexity: 'O(n)',
  spaceComplexity: 'O(n)',
  leetcodeUrl: 'https://leetcode.com/problems/encode-and-decode-strings/',
  youtubeUrl: 'https://www.youtube.com/watch?v=B1k_sxOSgv8',
};
