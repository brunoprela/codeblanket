/**
 * Quiz questions for XOR Properties and Applications section
 */

export const xorpropertiesQuiz = [
  {
    id: 'q1',
    question:
      'Explain XOR properties and why XOR is special. What makes it useful for problem-solving?',
    sampleAnswer:
      'XOR properties: 1) Commutative: a ^ b = b ^ a. 2) Associative: (a ^ b) ^ c = a ^ (b ^ c). 3) Identity: a ^ 0 = a. 4) Self-inverse: a ^ a = 0. 5) Reversible: if a ^ b = c, then c ^ b = a. These make XOR unique - it is its own inverse. Applications: 1) Find unique element (all others appear twice): XOR all elements, duplicates cancel out. 2) Swap variables: a ^= b, b ^= a, a ^= b (no temp). 3) Detect different bits: a ^ b gives 1 where bits differ. 4) Simple encryption: msg ^ key encrypts, cipher ^ key decrypts. For example, [4,1,2,1,2] → 4^1^2^1^2 = 4 (1s and 2s cancel). The self-inverse property (a ^ a = 0) is key to many algorithms.',
    keyPoints: [
      'Properties: commutative, associative, self-inverse',
      'a ^ a = 0, a ^ 0 = a (key properties)',
      'Find unique: XOR all, duplicates cancel',
      'Swap without temp: a ^= b, b ^= a, a ^= b',
      'Encryption: XOR is reversible cipher',
    ],
  },
  {
    id: 'q2',
    question:
      'Walk me through the "find single number" problem where all others appear twice. Why does XOR solve it elegantly?',
    sampleAnswer:
      'Problem: array where every element appears twice except one. Find the single one. XOR solution: XOR all elements, result is the single number. Why it works: XOR is commutative and associative, so order does not matter. Pair elements: a^b^a^c^b = (a^a)^(b^b)^c = 0^0^c = c. Each duplicate pair XORs to 0, only single element remains. Example: [4,1,2,1,2] → 4^1^2^1^2 = 4^(1^1)^(2^2) = 4^0^0 = 4. Time O(n), space O(1) - optimal. Alternative approaches: hash set O(n) space, sort O(n log n) time. XOR is elegant because: single pass, constant space, leverages mathematical property. Extension: if all appear three times except one, need different technique (bit counting modulo 3).',
    keyPoints: [
      'XOR all elements, duplicates cancel (a^a=0)',
      'Order does not matter (commutative, associative)',
      'O(n) time, O(1) space - optimal',
      'vs Hash: O(n) space, vs Sort: O(n log n) time',
      'Extension: three times needs different approach',
    ],
  },
  {
    id: 'q3',
    question:
      'Describe how to find two unique numbers when all others appear twice using XOR. What is the key insight?',
    sampleAnswer:
      'Problem: array where every element appears twice except two unique numbers a and b. XOR all elements gives a^b (duplicates cancel). But how to separate a and b? Key insight: find any bit where a and b differ (rightmost set bit in a^b). Use this bit to partition array into two groups: bit set and bit clear. Each group has one unique number and matching pairs. XOR each group separately to find a and b. Example: [1,2,1,3,2,5], xor_all=3^5=6 (110). Rightmost set bit: 6 & -6 = 2 (010). Group 1 (bit 1 set): 2,3,2 → XOR=3. Group 2 (bit 1 clear): 1,1,5 → XOR=5. Result: 3 and 5. Time O(n), space O(1). Brilliant use of XOR partitioning.',
    keyPoints: [
      'XOR all → a^b (duplicates cancel)',
      'Find bit where a and b differ',
      'Partition by that bit: two groups',
      'XOR each group to find unique in each',
      'O(n) time, O(1) space, elegant solution',
    ],
  },
];
