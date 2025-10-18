/**
 * Delete Node in a Linked List
 * Problem ID: delete-node-in-linked-list
 * Order: 5
 */

import { Problem } from '../../../types';

export const delete_node_in_linked_listProblem: Problem = {
  id: 'delete-node-in-linked-list',
  title: 'Delete Node in a Linked List',
  difficulty: 'Easy',
  topic: 'Linked List',
  order: 5,
  description: `There is a singly-linked list \`head\` and we want to delete a node \`node\` in it.

You are given the node to be deleted \`node\`. You will **not be given access** to the first node of \`head\`.

All the values of the linked list are **unique**, and it is guaranteed that the given node \`node\` is **not the last node** in the linked list.

Delete the given node. Note that by deleting the node, we do not mean removing it from memory. We mean:
- The value of the given node should not exist in the linked list.
- The number of nodes in the linked list should decrease by one.
- All the values before \`node\` should be in the same order.
- All the values after \`node\` should be in the same order.`,
  examples: [
    {
      input: 'head = [4,5,1,9], node = 5',
      output: '[4,1,9]',
      explanation:
        'You are given the second node with value 5. After deleting it, the linked list becomes 4 -> 1 -> 9.',
    },
    {
      input: 'head = [4,5,1,9], node = 1',
      output: '[4,5,9]',
    },
  ],
  constraints: [
    'The number of the nodes in the given list is in the range [2, 1000]',
    '-1000 <= Node.val <= 1000',
    'The value of each node in the list is unique',
    'The node to be deleted is in the list and is not a tail node',
  ],
  hints: [
    'Copy the value from the next node to current node',
    'Then delete the next node instead',
    'This effectively "deletes" the current node',
  ],
  starterCode: `class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def delete_node(node: ListNode) -> None:
    """
    Delete given node (not given head!).
    
    Args:
        node: The node to delete
    """
    # Write your code here
    pass
`,
  testCases: [
    {
      input: [[4, 5, 1, 9], 5],
      expected: [4, 1, 9],
    },
    {
      input: [[4, 5, 1, 9], 1],
      expected: [4, 5, 9],
    },
  ],
  solution: `class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def delete_node(node: ListNode) -> None:
    """
    Copy next node's value and skip next node.
    Time: O(1), Space: O(1)
    """
    # Copy value from next node
    node.val = node.next.val
    
    # Skip the next node
    node.next = node.next.next
`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  leetcodeUrl: 'https://leetcode.com/problems/delete-node-in-a-linked-list/',
  youtubeUrl: 'https://www.youtube.com/watch?v=f1r_jFWRbH8',
};
