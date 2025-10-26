export const lowLatencyProgrammingQuiz = [
    {
        id: 'low-latency-programming-q-1',
        question:
            'Your trading system currently processes market data in 50μs. Your competitor does it in 10μs, giving them a 40μs advantage. What are 3 optimization techniques you would implement to close this gap?',
        sampleAnswer:
            'Optimization Techniques:\n\n' +
            '1. **Lock-Free Data Structures** (save 20μs):\n' +
            '   - Replace mutexes with atomic operations\n' +
            '   - Use lock-free ring buffers for message passing\n' +
            '   - Eliminate contention between threads\n\n' +
            '2. **Zero-Copy Memory** (save 10μs):\n' +
            '   - Memory-mapped files for market data\n' +
            '   - Shared memory between processes\n' +
            '   - Avoid memcpy, pass pointers instead\n\n' +
            '3. **CPU Pinning** (save 10μs):\n' +
            '   - Pin critical threads to specific CPU cores\n' +
            '   - Avoid context switches\n' +
            '   - Keep data in L1 cache\n\n' +
            'Result: 50μs → 10μs (competitive parity)',
        keyPoints: [
            'Lock-free data structures: Replace mutexes with atomics, use lock-free ring buffers, eliminate thread contention (save 20μs)',
            'Zero-copy memory: Memory-mapped files, shared memory, pass pointers not data (save 10μs)',
            'CPU pinning: Pin threads to cores, avoid context switches, keep data in L1 cache (save 10μs)',
            'Total optimization: 50μs → 10μs, achieving competitive parity with 40μs improvement',
            'Additional techniques: Pre-allocate memory, avoid system calls, compile to C++, use FPGA for critical path',
        ],
    },
    {
        id: 'low-latency-programming-q-2',
        question:
            'Explain the difference between latency and throughput in trading systems. Why might optimizing for one hurt the other?',
        sampleAnswer:
            'Latency vs Throughput:\n\n' +
            '**Latency**: Time for single operation\n' +
            '- Example: Process one order in 10μs\n' +
            '- Optimizations: Lock-free, zero-copy, CPU pinning\n\n' +
            '**Throughput**: Operations per second\n' +
            '- Example: Process 100,000 orders/sec\n' +
            '- Optimizations: Batching, pipelining, parallelism\n\n' +
            'Trade-off: Batching increases throughput but adds latency',
        keyPoints: [
            'Latency: Time for single operation (10μs per order), optimized via lock-free, zero-copy, CPU pinning',
            'Throughput: Operations per second (100K orders/sec), optimized via batching, pipelining, parallelism',
            'Trade-off: Batching increases throughput (process 100 orders at once) but adds latency (wait for batch to fill)',
            'HFT priority: Latency over throughput (better to process 1 order in 10μs than 100 orders in 1000μs)',
            'Market maker priority: Throughput over latency (handle bursts of orders during volatility)',
        ],
    },
    {
        id: 'low-latency-programming-q-3',
        question:
            'Why do HFT firms use FPGAs (Field-Programmable Gate Arrays) for critical trading operations? What are the trade-offs?',
        sampleAnswer:
            'FPGA for Trading:\n\n' +
            '**Why FPGAs:**\n' +
            '1. **Ultra-low latency**: <1μs (vs 10μs CPU)\n' +
            '2. **Deterministic**: No OS interrupts\n' +
            '3. **Parallel processing**: Process multiple orders simultaneously\n\n' +
            '**Trade-offs:**\n' +
            '- Cost: $100K+ per FPGA\n' +
            '- Complexity: Requires hardware engineers (Verilog/VHDL)\n' +
            '- Flexibility: Hard to change logic (vs software)\n\n' +
            'Use case: Market data parsing, order matching, risk checks',
        keyPoints: [
            'Ultra-low latency: <1μs processing (vs 10μs CPU), giving microsecond advantage in competitive markets',
            'Deterministic: No OS interrupts/context switches, consistent performance critical for risk management',
            'Parallel processing: Process multiple orders simultaneously, higher throughput during market spikes',
            'Trade-offs: High cost ($100K+ per FPGA), requires hardware engineers (Verilog/VHDL), difficult to modify logic',
            'Use cases: Market data parsing, order matching, risk checks, typically for market makers and HFT firms',
        ],
    },
];

