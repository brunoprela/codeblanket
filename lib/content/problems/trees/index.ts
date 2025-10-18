/**
 * trees Problems
 * 10 problems total
 */

import { invert_binary_treeProblem } from './invert-binary-tree';
import { validate_bstProblem } from './validate-bst';
import { binary_tree_max_path_sumProblem } from './binary-tree-max-path-sum';
import { same_treeProblem } from './same-tree';
import { symmetric_treeProblem } from './symmetric-tree';
import { diameter_binary_treeProblem } from './diameter-binary-tree';
import { lowest_common_ancestor_bstProblem } from './lowest-common-ancestor-bst';
import { construct_tree_preorder_inorderProblem } from './construct-tree-preorder-inorder';
import { lowest_common_ancestor_binary_treeProblem } from './lowest-common-ancestor-binary-tree';
import { serialize_deserialize_binary_treeProblem } from './serialize-deserialize-binary-tree';

export const treesProblems = [
  invert_binary_treeProblem, // 1. Invert Binary Tree
  validate_bstProblem, // 2. Validate Binary Search Tree
  binary_tree_max_path_sumProblem, // 3. Binary Tree Maximum Path Sum
  same_treeProblem, // 4. Same Tree
  symmetric_treeProblem, // 5. Symmetric Tree
  diameter_binary_treeProblem, // 6. Diameter of Binary Tree
  lowest_common_ancestor_bstProblem, // 7. Lowest Common Ancestor of a Binary Search Tree
  construct_tree_preorder_inorderProblem, // 8. Construct Binary Tree from Preorder and Inorder Traversal
  lowest_common_ancestor_binary_treeProblem, // 9. Lowest Common Ancestor of a Binary Tree
  serialize_deserialize_binary_treeProblem, // 10. Serialize and Deserialize Binary Tree
];
