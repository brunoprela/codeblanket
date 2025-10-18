/**
 * Memory-Efficient File Reader
 * Problem ID: generator-file-reader
 * Order: 5
 */

import { Problem } from '../../../types';

export const generator_file_readerProblem: Problem = {
  id: 'generator-file-reader',
  title: 'Memory-Efficient File Reader',
  difficulty: 'Medium',
  description: `Create a generator that reads a large file line by line and filters lines containing a keyword.

The generator should:
- Read file lazily (one line at a time)
- Filter lines containing the keyword
- Strip whitespace from each line
- Work with files of any size without loading into memory

**Use Case:** Processing huge log files efficiently.`,
  examples: [
    {
      input: 'File with "ERROR" keyword',
      output: 'Yields only lines containing "ERROR"',
    },
  ],
  constraints: [
    'Must use generator (yield)',
    'Cannot load entire file into memory',
    'Case-sensitive search',
  ],
  hints: [
    'Use with open() for proper file handling',
    'Check if keyword in line',
    'Yield matching lines',
  ],
  starterCode: `def read_matching_lines(filepath, keyword):
    """
    Generator that yields lines containing keyword.
    
    Args:
        filepath: Path to file
        keyword: Keyword to search for
        
    Yields:
        Lines containing the keyword
    """
    # Your code here
    pass


# Test with simulated file lines (for testing without actual file)
def test_read_matching(keyword, lines):
    """Test helper that simulates file reading"""
    # Mock file by iterating over lines
    def mock_generator():
        for line in lines:
            if keyword in line:
                yield line.strip()
    return list(mock_generator())

# Test
result = test_read_matching('ERROR', ['INFO: Starting', 'ERROR: Failed', 'INFO: Done'])
`,
  testCases: [
    {
      input: [],
      expected: ['ERROR: Failed'],
    },
  ],
  solution: `def read_matching_lines(filepath, keyword):
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if keyword in line:
                yield line


# Test with simulated file lines (for testing without actual file)
def test_read_matching(keyword, lines):
    """Test helper that simulates file reading"""
    def mock_generator():
        for line in lines:
            if keyword in line:
                yield line.strip()
    return list(mock_generator())

result = test_read_matching('ERROR', ['INFO: Starting', 'ERROR: Failed', 'INFO: Done'])`,
  timeComplexity: 'O(n) where n is number of lines',
  spaceComplexity: 'O(1)',
  order: 5,
  topic: 'Python Advanced',
};
