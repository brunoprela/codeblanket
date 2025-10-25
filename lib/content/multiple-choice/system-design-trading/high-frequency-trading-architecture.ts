import { MultipleChoiceQuestion } from '@/lib/types';

export const highFrequencyTradingArchitectureMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'hfta-mc-1',
      question:
        'What is the primary advantage of FPGA-based trading over software-based trading?',
      options: [
        'FPGAs are cheaper to develop and maintain',
        'FPGAs provide 100-900ns latency vs 2-10μs for software',
        'FPGAs can implement more complex strategies',
        'FPGAs are easier to debug and modify',
      ],
      correctAnswer: 1,
      explanation:
        'FPGAs provide ultra-low latency (100-900ns) vs software (2-10μs) through parallel hardware processing. No OS overhead, no CPU scheduling, deterministic performance. FPGAs are actually MORE expensive ($1-2M dev vs $200-500K software), LESS complex (simple strategies only), and HARDER to debug (waveform analysis vs gdb). Advantage is purely latency for latency-critical strategies where microseconds matter.',
    },
    {
      id: 'hfta-mc-2',
      question:
        'Why is kernel bypass (DPDK) faster than standard Linux networking?',
      options: [
        'DPDK uses faster network cards',
        'DPDK bypasses the kernel TCP/IP stack and polls network cards directly',
        'DPDK compresses network packets',
        'DPDK uses multiple threads for packet processing',
      ],
      correctAnswer: 1,
      explanation:
        'DPDK bypasses kernel: polls network card directly from userspace, no system calls (~1μs), no kernel TCP/IP processing (~10-50μs), zero-copy (no memory copies). Standard path: App → syscall → kernel → driver → NIC = 20-100μs. DPDK path: App → NIC (direct memory access) = 2-5μs. Same NICs, but avoiding kernel reduces latency 10-20×. Threading helps but not the primary benefit.',
    },
    {
      id: 'hfta-mc-3',
      question:
        'What is the purpose of object pools in low-latency trading systems?',
      options: [
        'To save memory by sharing objects between threads',
        'To eliminate dynamic memory allocation and garbage collection pauses',
        'To implement thread-safe data structures',
        'To improve CPU cache performance',
      ],
      correctAnswer: 1,
      explanation:
        'Object pools eliminate dynamic allocation: pre-allocate all objects at startup, acquire/release from pool during trading (~10-20ns), never call malloc/new (100-500ns + fragmentation). Prevents garbage collection pauses (10-100ms in Python/Java) that kill HFT strategies. Not primarily for sharing (separate pools per thread better), thread-safety (pools can be lock-free), or cache (though alignment helps). Main benefit: zero allocation, zero GC, deterministic latency.',
    },
    {
      id: 'hfta-mc-4',
      question:
        'Why does co-location in the same datacenter as the exchange provide a latency advantage?',
      options: [
        'Datacenters have faster computers',
        'It reduces network distance from 1000s of km to 10-100 meters',
        'Exchanges give co-located servers priority',
        'Co-location is required by regulations',
      ],
      correctAnswer: 1,
      explanation:
        "Co-location reduces physical distance: speed of light in fiber = 200,000 km/s. Distance matters: Chicago-NY (1,200 km) = 6ms one-way. Same rack (10-100m) = 0.05-0.5μs. Reduces latency from 10ms+ to 10-50μs total. Computers same speed, exchanges don't give priority (first-come-first-served in matching engine), co-location not required (just competitive advantage). Physics dominates: shorter distance = lower latency.",
    },
    {
      id: 'hfta-mc-5',
      question:
        'What is the minimum theoretical latency from Chicago to New York for trading signals?',
      options: [
        '100 microseconds',
        '1 millisecond',
        '4 milliseconds (speed of light in fiber)',
        '100 milliseconds',
      ],
      correctAnswer: 2,
      explanation:
        'Speed of light limit: Chicago to NY = 1,200 km. Light in fiber = 200,000 km/s (2/3 speed in vacuum due to refractive index). Time = 1,200 / 200,000 = 0.006 seconds = 6ms one-way. Round trip = 12ms. With microwave (straight line, faster): ~4ms one-way. Physical limit cannot be beaten—no software/hardware optimization can exceed speed of light. Actual latency higher due to processing (10-100μs added).',
    },
  ];
