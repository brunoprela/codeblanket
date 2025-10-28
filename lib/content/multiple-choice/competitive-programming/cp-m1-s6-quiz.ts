export default [
  {
    id: 'cp-m1-s6-q1',
    section: 'C++ Basics Review',
    question:
      'What is the correct way to declare and initialize a vector of integers with size n and all elements set to 0?',
    options: [
      'vector<int> v(n);',
      'vector<int> v[n];',
      'vector<int> v = new vector(n);',
      'int v[n]; vector<int> vec(v, v+n);',
    ],
    correctAnswer: 0,
    explanation:
      "`vector<int> v(n);` creates a vector with n elements, all initialized to 0 (default value for int). This is the standard and most efficient way. `vector<int> v[n];` creates an array of n empty vectors (not what we want). `vector<int> v = new vector(n);` is incorrect syntax (new returns a pointer, and vector isn't allocated this way in modern C++). The fourth option works but is unnecessarily complex. Other common initializations: `vector<int> v(n, value);` (all elements = value), `vector<int> v = {1, 2, 3};` (specific values), `vector<int> v;` then `v.resize(n);` (dynamic sizing). Time complexity: O(n) for initialization.",
    difficulty: 'beginner',
  },
  {
    id: 'cp-m1-s6-q2',
    section: 'C++ Basics Review',
    question:
      'Which loop construct is most appropriate for iterating through a vector when you need both the element and its index?',
    options: [
      'for(auto x : v) { /* use x */ }',
      'for(size_t i = 0; i < v.size(); i++) { /* use v[i] and i */ }',
      'while loop with manual indexing',
      'for_each with lambda function',
    ],
    correctAnswer: 1,
    explanation:
      "When you need both element and index, traditional indexed loop is clearest: `for(size_t i = 0; i < v.size(); i++)` or `for(int i = 0; i < (int)v.size(); i++)` (cast to avoid signed/unsigned warnings). Range-based for (`for(auto x : v)`) is great when you only need the element value. While loops work but are more verbose. For_each can work but is less readable. Common pattern: `for(int i = 0; i < n; i++) { /* use v[i] and i */ }`. The cast to int prevents warnings: `v.size()` returns unsigned, comparing with signed int causes warnings with -Wall. Alternative modern approach: `for(auto [i, x] : enumerate(v))` but C++ doesn't have built-in enumerate (Python does).",
    difficulty: 'beginner',
  },
  {
    id: 'cp-m1-s6-q3',
    section: 'C++ Basics Review',
    question:
      'What is the time complexity of inserting an element at the end of a std::vector?',
    options: ['Always O(1)', 'Amortized O(1)', 'O(n)', 'O(log n)'],
    correctAnswer: 1,
    explanation:
      'push_back() on a vector is amortized O(1). Most insertions are O(1), but occasionally the vector needs to reallocate (double capacity), which is O(n). Over many insertions, this averages to O(1) per operation. Example: inserting 1000 elements requires ~10 reallocations (capacities: 1, 2, 4, 8, ..., 1024), but total work is still O(n) over n insertions = O(1) amortized. For guaranteed O(1) without any O(n) spikes, reserve capacity first: `v.reserve(n);` then n push_backs are all true O(1). Other operations: v[i] access is O(1), insert at beginning is O(n), sort is O(n log n). Understanding amortized complexity is key for algorithm analysis.',
    difficulty: 'intermediate',
  },
  {
    id: 'cp-m1-s6-q4',
    section: 'C++ Basics Review',
    question:
      'How do you correctly compare two strings lexicographically in C++?',
    options: [
      'if(strcmp(s1, s2) == 0)',
      'if(s1 == s2)',
      'if(s1.equals(s2))',
      'if(s1.compare(s2) == 0)',
    ],
    correctAnswer: 1,
    explanation:
      'For C++ strings (std::string), use `s1 == s2` for equality. This is overloaded to work correctly with strings. `strcmp()` is for C-style strings (char*), not std::string. Java has `.equals()` but C++ doesn\'t. `s1.compare(s2)` works (returns 0 for equal) but is more verbose than ==. For lexicographic ordering: `s1 < s2`, `s1 > s2` etc. work as expected with strings. Example: `"apple" < "banana"` is true. For case-insensitive comparison, you need custom comparison or transform to lowercase first. Common pitfall: using strcmp with std::string (won\'t compile). Remember: == for equality, <, >, <=, >= for ordering, all work naturally with std::string.',
    difficulty: 'beginner',
  },
  {
    id: 'cp-m1-s6-q5',
    section: 'C++ Basics Review',
    question: 'What is the difference between map and unordered_map in C++?',
    options: [
      'map is faster than unordered_map for all operations',
      'map keeps elements sorted by key, unordered_map uses hashing',
      "unordered_map doesn't allow duplicate keys",
      'map is for integers only, unordered_map is for all types',
    ],
    correctAnswer: 1,
    explanation:
      'map uses a balanced binary tree (typically red-black tree), keeping elements sorted by key with O(log n) operations. unordered_map uses a hash table with O(1) average operations but no ordering. Choose based on needs: Need sorted keys or ordered traversal? Use map. Need fast lookup/insert? Use unordered_map. Both disallow duplicate keys (use multimap/unordered_multimap for duplicates). Both work with any comparable/hashable types. Example: `map<int, int> m; m[key] = value;` - O(log n), sorted. `unordered_map<int, int> um; um[key] = value;` - O(1) average, unsorted. For CP: unordered_map is usually faster unless you need sorted iteration. Caveat: unordered_map worst case is O(n) with bad hash collisions (rare but possible with adversarial inputs).',
    difficulty: 'intermediate',
  },
] as const;
