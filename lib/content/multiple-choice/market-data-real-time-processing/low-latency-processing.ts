import { MultipleChoiceQuestion } from '@/lib/types';

export const lowLatencyProcessingMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'low-latency-processing-mc-1',
    question: 'Python function takes 50μs. After Cython optimization, it takes 10μs. What is the speedup?',
    options: ['5× faster (50/10 = 5)', '40μs faster', '10× faster', '2.5× faster'],
    correctAnswer: 0,
    explanation: 'Speedup calculation: Original / Optimized = 50μs / 10μs = 5× speedup. Time saved: 40μs per call. At 10K calls/sec: 40μs × 10K = 400ms saved/sec = 40% CPU reduction. Cython achieves 5-10× speedup typically through: (1) Type annotations (no Python type checks), (2) C compilation (no interpreter overhead), (3) nogil (true parallelism). For 10× speedup, need aggressive optimization: C structs, pointer arithmetic, SIMD instructions. 50μs → 10μs reasonable for typical Cython conversion.',
  },
  {
    id: 'low-latency-processing-mc-2',
    question: 'O(1) dict lookup takes 50ns. O(n) list scan of 1000 items takes how long?',
    options: ['50,000ns = 50μs (1000 × 50ns)', '50ns (same as dict)', '500ns', '5,000ns'],
    correctAnswer: 0,
    explanation: 'List scan complexity: O(n) = 1000 items × 50ns/comparison = 50,000ns = 50μs. Dict lookup: O(1) = 50ns regardless of size. Difference: 1000× slower for lists. At 10K lookups/sec: List = 500ms CPU (50%), Dict = 0.5ms CPU (0.05%). For order books: NEVER use lists (O(n) scan unacceptable). Use dict (O(1) lookup) or SortedDict (O(log n) = 10 comparisons = 500ns). Even O(log n) is 100× faster than O(n) for 1000 items.',
  },
  {
    id: 'low-latency-processing-mc-3',
    question: 'Pre-allocated NumPy array (16 bytes/element) vs Python list (200 bytes/element). Memory savings for 10K elements?',
    options: ['92% savings (16 vs 200 bytes, 12.5× less)', '50% savings', '10× savings', 'No savings'],
    correctAnswer: 0,
    explanation: 'Memory calculation: NumPy: 10K × 16 bytes = 160 KB. Python list: 10K × 200 bytes = 2 MB. Savings: (2000 - 160) / 2000 = 92% reduction. Why Python uses more: Object overhead (40 bytes), reference counting (8 bytes), type info (8 bytes), pointer (8 bytes) = 64+ bytes + actual data. NumPy: Contiguous C array, no per-element overhead, just raw data. Benefits: (1) 12.5× less memory, (2) Cache-friendly (contiguous), (3) SIMD vectorization possible. For HFT, memory bandwidth is bottleneck - NumPy arrays critical.',
  },
  {
    id: 'low-latency-processing-mc-4',
    question: 'Python GC pause causes 10ms latency spike every 5 seconds. How to prevent?',
    options: ['Disable GC during trading, use pre-allocated memory', 'Use faster CPU', 'Increase memory', 'Restart Python'],
    correctAnswer: 0,
    explanation: 'GC pause prevention: (1) Disable GC: gc.disable() during market hours, manually collect during off-hours. (2) Pre-allocate: Use fixed-size arrays (no malloc = no GC). (3) Cython nogil: Release GIL, no Python objects in hot path. Python GC: Generational, stop-the-world (all threads pause). At 10ms pause every 5s = 0.2% of time, but unacceptable latency spike. Alternative: Use languages without GC (C++, Rust) or predictable GC (Java Shenandoah/ZGC < 1ms pauses). For production HFT: Disable Python GC entirely, use memory pools.',
  },
  {
    id: 'low-latency-processing-mc-5',
    question: 'CPU affinity pins trading thread to core 2. What is the benefit?',
    options: ['Eliminates context switching (10μs → 0μs)', 'Faster CPU speed', 'More memory', 'Better network'],
    correctAnswer: 0,
    explanation: 'CPU affinity benefit: Pins process to specific core, eliminating context switches (10-50μs each). Without affinity: OS schedules on any core, moves process between cores, evicts L1/L2 cache (100ns → 10μs cache miss penalty). With affinity: Process stays on core 2, L1/L2 cache hot (100ns access), no migration overhead. Setup: isolcpus=2,3 in kernel (reserve cores), taskset -c 2 python trade.py (pin to core 2). For HFT: Dedicate cores 2-3 for trading (isolated), cores 0-1 for OS. Typical gain: 10-20% latency reduction from cache hits + no context switches.',
  },
];
