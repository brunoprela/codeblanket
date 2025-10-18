/**
 * Multiple choice questions for System Design Basics section
 */

import { MultipleChoiceQuestion } from '../../../types';

export const systemdesignbasicsMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'mc1',
    question:
      'What is the main advantage of using Base62 encoding in URL shortener?',
    options: [
      'It is more secure',
      'It generates shorter codes than other encodings',
      'It is faster than all alternatives',
      'It uses less memory',
    ],
    correctAnswer: 1,
    explanation:
      'Base62 uses 62 characters (0-9, a-z, A-Z) instead of Base10 (10 chars) or Base16 (16 chars), so it generates much shorter codes. Example: 1,000,000 in Base10 = "1000000" (7 chars), Base62 = "4c92" (4 chars). More compact = shorter URLs.',
  },
  {
    id: 'mc2',
    question:
      'In Parking Lot design, what design pattern is used for Car, Truck, Motorcycle?',
    options: ['Singleton', 'Factory', 'Inheritance (polymorphism)', 'Observer'],
    correctAnswer: 2,
    explanation:
      'Car, Truck, Motorcycle all inherit from Vehicle abstract class. This is polymorphism - each vehicle type implements can_fit_in() differently. ParkingLot code works with Vehicle interface without knowing specific type.',
  },
  {
    id: 'mc3',
    question:
      'What is the time complexity of shortening a URL with counter-based approach?',
    options: ['O(1)', 'O(log N)', 'O(N)', 'O(N log N)'],
    correctAnswer: 0,
    explanation:
      'Shortening is O(1): increment counter (O(1)), encode to Base62 (O(log counter) = O(1) for practical sizes), insert to HashMap (O(1)). No collision checking needed since counter is unique.',
  },
  {
    id: 'mc4',
    question:
      'According to CAP theorem, which two properties does a typical SQL database prioritize?',
    options: [
      'Consistency and Availability',
      'Consistency and Partition Tolerance',
      'Availability and Partition Tolerance',
      'All three',
    ],
    correctAnswer: 1,
    explanation:
      'Traditional SQL databases (PostgreSQL, MySQL) prioritize Consistency and Partition Tolerance (CP). During network partition, they may sacrifice Availability to maintain consistency. NoSQL databases often choose AP (Available and Partition Tolerant) with eventual consistency.',
  },
  {
    id: 'mc5',
    question:
      'Why do we need both url_to_short and short_to_url HashMaps in URL shortener?',
    options: [
      'One HashMap is not enough memory',
      'To support both directions: shorten() needs url_to_short, expand() needs short_to_url',
      'To make it faster',
      'We only need one',
    ],
    correctAnswer: 1,
    explanation:
      'We need both for O(1) operations in both directions. shorten(long) checks url_to_short to avoid duplicates and returns existing short code. expand(short) looks up short_to_url to get original. Single HashMap would only support one direction efficiently.',
  },
];
