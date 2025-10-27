export const lowLatencyProgrammingMC = [
  {
    id: 'low-latency-programming-mc-1',
    question:
      'Your trading system needs to process market data in <10μs. Which programming language is BEST?',
    options: ['Python', 'Java', 'C++', 'JavaScript'],
    correctAnswer: 2,
    explanation:
      'Answer: C++.\n\n' +
      'Latency comparison for market data processing:\n' +
      '- **C++**: 1-10μs (compiled, manual memory management)\n' +
      '- **Java**: 10-100μs (JVM overhead, garbage collection pauses)\n' +
      '- **Python**: 100-1000μs (interpreted, slow)\n' +
      '- **JavaScript**: 50-500μs (V8 JIT, but still GC pauses)\n\n' +
      'Why C++ for HFT:\n' +
      '1. No GC pauses (manual memory management)\n' +
      '2. Zero-cost abstractions\n' +
      '3. Direct hardware access (RDTSC, memory barriers)\n' +
      '4. Industry standard (all major HFT firms use C++)\n\n' +
      'Real-world: Citadel, Jane Street, Hudson River Trading all use C++ for latency-critical code.',
  },
  {
    id: 'low-latency-programming-mc-2',
    question:
      'What is a "lock-free" data structure and why is it important for low-latency trading?',
    options: [
      "A data structure that doesn't require locks (mutexes), avoiding thread contention",
      'A data structure stored on disk instead of memory',
      'A data structure that is read-only',
      'A data structure encrypted for security',
    ],
    correctAnswer: 0,
    explanation:
      "Answer: A data structure that doesn't require locks (mutexes), avoiding thread contention.\n\n" +
      'Lock-free data structures use atomic operations instead of mutexes:\n' +
      '```cpp\n' +
      '// With locks (slow, 100+μs)\n' +
      'std::mutex mtx;\n' +
      'void push(Item item) {\n' +
      '    std::lock_guard<std::mutex> lock(mtx);  // Blocks other threads\n' +
      '    queue.push(item);\n' +
      '}\n\n' +
      '// Lock-free (fast, <1μs)\n' +
      'void push(Item item) {\n' +
      '    size_t current = write_idx.load(std::memory_order_relaxed);\n' +
      '    size_t next = current + 1;\n' +
      '    write_idx.store(next, std::memory_order_release);  // Atomic\n' +
      '}\n' +
      '```\n\n' +
      'Why important: Mutexes cause 100+μs delays when contended. Lock-free structures achieve <1μs latency even under high contention.',
  },
  {
    id: 'low-latency-programming-mc-3',
    question: 'What is "CPU pinning" and why is it used in HFT systems?',
    options: [
      'Running trading software on dedicated CPU cores to avoid context switches',
      'Overclocking CPUs for higher performance',
      'Using CPU hardware encryption',
      'Monitoring CPU temperature',
    ],
    correctAnswer: 0,
    explanation:
      'Answer: Running trading software on dedicated CPU cores to avoid context switches.\n\n' +
      'CPU pinning binds threads to specific CPU cores:\n' +
      '```python\n' +
      'import os\n' +
      '# Pin to core 0\n' +
      'os.sched_setaffinity(0, {0})\n' +
      '```\n\n' +
      'Benefits:\n' +
      '1. No context switches (save 10-50μs per switch)\n' +
      '2. Data stays in L1/L2 cache (faster access)\n' +
      '3. Predictable latency (no OS scheduler interference)\n\n' +
      'Typical setup:\n' +
      '- Core 0: Market data processing\n' +
      '- Core 1: Order generation\n' +
      '- Core 2: Risk checks\n' +
      '- Cores 3-7: Non-critical tasks\n\n' +
      'Real-world: All HFT firms pin critical threads to dedicated cores.',
  },
  {
    id: 'low-latency-programming-mc-4',
    question:
      'Your system allocates memory every time it processes an order. This causes 10μs latency. What is the solution?',
    options: [
      'Use a faster malloc implementation',
      'Pre-allocate a memory pool and reuse objects',
      'Increase RAM',
      'Use garbage collection',
    ],
    correctAnswer: 1,
    explanation:
      'Answer: Pre-allocate a memory pool and reuse objects.\n\n' +
      'Problem: malloc/new takes 1-10μs (heap allocation, locks, fragmentation).\n\n' +
      'Solution - Object Pool:\n' +
      '```cpp\n' +
      'class OrderPool {\n' +
      '    std::array<Order, 10000> orders;  // Pre-allocated\n' +
      '    std::atomic<size_t> next_idx{0};\n' +
      '    \n' +
      'public:\n' +
      '    Order* get_order() {\n' +
      '        size_t idx = next_idx.fetch_add(1) % 10000;\n' +
      '        return &orders[idx];  // No allocation, <100ns\n' +
      '    }\n' +
      '};\n' +
      '```\n\n' +
      'Result: 10μs → <0.1μs (100x faster).\n\n' +
      'HFT rule: Never allocate memory in hot path. Pre-allocate everything at startup.',
  },
  {
    id: 'low-latency-programming-mc-5',
    question:
      'What latency do HFT firms target for order-to-exchange transmission?',
    options: [
      '1 second',
      '1 millisecond (1,000μs)',
      '100 microseconds (100μs)',
      '10 microseconds (10μs)',
    ],
    correctAnswer: 3,
    explanation:
      'Answer: 10 microseconds (10μs).\n\n' +
      'HFT latency breakdown:\n' +
      '- Market data processing: 1-5μs\n' +
      '- Strategy decision: 1-3μs\n' +
      '- Order generation: 1-2μs\n' +
      '- Network to exchange: 5-10μs\n' +
      '- **Total: 10-20μs**\n\n' +
      'Top HFT firms (Citadel, Jump, Virtu) achieve sub-10μs latency.\n\n' +
      'Latency tiers:\n' +
      '- **HFT**: 10-100μs (compete on nanoseconds)\n' +
      '- **Market makers**: 100μs-1ms (still very fast)\n' +
      '- **Prop traders**: 1-10ms (acceptable)\n' +
      '- **Retail traders**: 100-1000ms (not critical)\n\n' +
      'Every microsecond counts: A 1μs advantage can be worth millions in profit per year.',
  },
];
