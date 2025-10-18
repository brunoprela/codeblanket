/**
 * advanced-protocol-duck-typing
 * Order: 46
 */

import { Problem } from '../../../types';

export const protocol_duck_typingProblem: Problem = {
  id: 'advanced-protocol-duck-typing',
  title: 'Protocol and Duck Typing',
  difficulty: 'Medium',
  description: `Use Protocol for structural subtyping (duck typing with type hints).

Protocol features (Python 3.8+):
- Define interface without inheritance
- Structural typing (duck typing)
- Type checker friendly
- No runtime enforcement

**Example:** Anything with .read() is file-like

This tests:
- Protocol definition
- Structural typing
- Type annotations`,
  examples: [
    {
      input: 'File-like protocol',
      output: 'Any object with read() method',
    },
  ],
  constraints: ['Use typing.Protocol', 'Define method signatures'],
  hints: [
    'Inherit from Protocol',
    'Define method signatures',
    'No implementation needed',
  ],
  starterCode: `from typing import Protocol

class Readable(Protocol):
    """Protocol for readable objects"""
    def read(self) -> str:
        ...


class FileWrapper:
    """File-like object"""
    def __init__(self, content):
        self.content = content
    
    def read(self) -> str:
        return self.content


class StringWrapper:
    """Another file-like object"""
    def __init__(self, text):
        self.text = text
    
    def read(self) -> str:
        return self.text


def process_readable(obj: Readable) -> int:
    """Process any readable object"""
    content = obj.read()
    return len(content)


def test_protocol():
    """Test protocol"""
    f = FileWrapper("hello world")
    s = StringWrapper("test")
    
    result1 = process_readable(f)
    result2 = process_readable(s)
    
    return result1 + result2
`,
  testCases: [
    {
      input: [],
      expected: 15,
      functionName: 'test_protocol',
    },
  ],
  solution: `from typing import Protocol

class Readable(Protocol):
    def read(self) -> str:
        ...


class FileWrapper:
    def __init__(self, content):
        self.content = content
    
    def read(self) -> str:
        return self.content


class StringWrapper:
    def __init__(self, text):
        self.text = text
    
    def read(self) -> str:
        return self.text


def process_readable(obj: Readable) -> int:
    content = obj.read()
    return len(content)


def test_protocol():
    f = FileWrapper("hello world")
    s = StringWrapper("test")
    
    result1 = process_readable(f)
    result2 = process_readable(s)
    
    return result1 + result2`,
  timeComplexity: 'O(1)',
  spaceComplexity: 'O(1)',
  order: 46,
  topic: 'Python Advanced',
};
