/**
 * Quiz questions for The Algorithm Step-by-Step section
 */

export const algorithmQuiz = [
  {
    id: 'q1',
    question:
      'Say you are at the middle element and it is smaller than your target. Walk me through your next move - what do you update and why?',
    hint: 'Where could the target possibly be if the middle is too small?',
    sampleAnswer:
      'If the middle element is smaller than my target, I know the target has to be somewhere to the right because the array is sorted. So I update left to be mid + 1, which means I am now only looking at the right half of the array. The key is using mid + 1, not just mid, because I have already checked mid and know it is not my target. So I want to exclude it and look at everything to the right of it. This is how I eliminate half the search space in one comparison.',
    keyPoints: [
      'Target must be in the right half',
      'Update left = mid + 1 to search right',
      'Use mid + 1 (not mid) to exclude the checked element',
      'Eliminates left half of search space',
    ],
  },
  {
    id: 'q2',
    question:
      'You might see two different ways to calculate the middle: (left + right) // 2 versus left + (right - left) // 2. Talk about which one you would use and why.',
    sampleAnswer:
      'I would use left + (right - left) // 2, especially if writing code in languages like Java or C++. The reason is integer overflow. When you add two very large integers together with (left + right), you can actually overflow and get a negative number or wrap around, which would break your algorithm completely. By doing left + (right - left) // 2 instead, you keep the numbers smaller during the calculation and avoid that overflow. In Python it does not matter as much because Python handles big integers automatically, but it is a good habit to use the safer formula.',
    keyPoints: [
      'left + (right - left) // 2 prevents integer overflow',
      '(left + right) can overflow in Java/C++ with large numbers',
      'Both give the same mathematical result',
      'Safer formula is a best practice',
    ],
  },
  {
    id: 'q3',
    question:
      'Should your loop condition be "while left < right" or "while left <= right"? Explain your choice.',
    hint: 'What happens when left and right are pointing to the same element?',
    sampleAnswer:
      'It should be "while left <= right" with the equal sign. Here is why: when left equals right, there is still one element left that needs to be checked - they are both pointing to the same index. If I used just "left < right", the loop would stop before checking that final element, and I might miss my target if it happens to be that last one. The equal sign ensures I check every single element in my search space before giving up and returning that the target was not found.',
    keyPoints: [
      'Use "while left <= right" with equal sign',
      'When left == right, one element still needs checking',
      'Without <=, you skip the final element',
      'Ensures complete search space coverage',
    ],
  },
];
