import { MultipleChoiceQuestion } from '@/lib/types';

export const mockingWithUnittestMockMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mum-mc-1',
    question: 'What is the key difference between Mock() and MagicMock()?',
    options: [
      'MagicMock is faster than Mock for performance-critical tests',
      'MagicMock supports Python magic methods like __len__ and __iter__, Mock does not',
      'Mock is for unit tests, MagicMock is for integration tests',
      'MagicMock automatically mocks all imported modules, Mock does not',
    ],
    correctAnswer: 1,
    explanation:
      'MagicMock supports magic methods (__len__, __iter__, __str__, __enter__, etc.): mock = Mock(); len(mock) → unexpected behavior. magic = MagicMock(); magic.__len__.return_value = 5; len(magic) → 5. Use MagicMock when mocking objects that use magic methods (containers, context managers, iterables). Mock is lighter/faster for simple cases. Not related to test type (unit vs integration) or module mocking. Both work the same way except for magic method support.',
  },
  {
    id: 'mum-mc-2',
    question:
      'When using @patch decorator, where should you patch an imported object?',
    options: [
      'Where the object is defined (original module)',
      'Where the object is used (importing module)',
      'In the test file itself',
      'Patching location does not matter',
    ],
    correctAnswer: 1,
    explanation:
      'Patch where object is USED, not where defined: module_a.py: from module_b import Service; def func(): service = Service(). Test: @patch("module_a.Service") not @patch("module_b.Service"). Why? Python imports create reference in module_a namespace. Patching module_b.Service doesn\'t affect module_a.Service. Rule: patch("where_its_imported.Object"). Common mistake causes "mock not working" issues. Always patch at import site.',
  },
  {
    id: 'mum-mc-3',
    question: 'What does side_effect do in a mock?',
    options: [
      'Creates side effects in the production code being tested',
      'Allows mock to return different values on successive calls or raise exceptions',
      'Automatically verifies that the mock was called correctly',
      'Patches additional methods beyond the primary mock',
    ],
    correctAnswer: 1,
    explanation:
      'side_effect allows dynamic mock behavior: Different values: mock.method.side_effect = [1,2,3] → first call returns 1, second 2, third 3. Exceptions: side_effect = Exception("error") → raises exception when called. Custom function: side_effect = lambda x: x*2 → calls function with arguments. Use when: Need different results per call, simulating errors, complex logic based on args. Not for: Verifying calls (use assert_called), patching additional methods, or affecting production code.',
  },
  {
    id: 'mum-mc-4',
    question: 'What is the purpose of spec parameter in Mock?',
    options: [
      'Specifies the return value specification for the mock',
      'Restricts mock to only have attributes/methods of the specified class (type safety)',
      'Specifies which tests should use this mock',
      'Defines the mock specification file location',
    ],
    correctAnswer: 1,
    explanation:
      'spec provides type safety by restricting mock to specified class interface: mock = Mock(spec=PaymentGateway) → mock.charge() works (exists in PaymentGateway), mock.typo_method() raises AttributeError (not in spec). Benefits: Catches typos in tests, ensures mock matches real object interface, prevents accessing non-existent methods. spec_set even stricter (prevents adding new attributes). Not for: return values (use return_value), test selection, or file locations. Use spec to make mocks match real objects.',
  },
  {
    id: 'mum-mc-5',
    question:
      'Why might a test fail with "AssertionError: Expected call not found"?',
    options: [
      'The mock was never created',
      'The mock was called with different arguments than asserted',
      'The mock object was garbage collected',
      'The assert statement has incorrect syntax',
    ],
    correctAnswer: 1,
    explanation:
      'assert_called_with checks exact arguments: mock.method.assert_called_with(1, 2, key="value") fails if called as mock.method(1, 2, key="different") or mock.method(2, 1, key="value") or mock.method(1, 2). Common causes: Argument order wrong, using assert_called_once_with when multiple calls, typo in argument. Debug: print(mock.method.call_args_list) shows all calls. Fix: Match exact args or use assert_called() for any args. Not syntax error (would raise SyntaxError not AssertionError).',
  },
];
