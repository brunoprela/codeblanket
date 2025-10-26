import { MultipleChoiceQuestion } from '@/lib/types';

export const timestampManagementMultipleChoice: MultipleChoiceQuestion[] = [
  {
    id: 'timestamp-management-mc-1',
    question: 'Your system clock drifts +2ms per hour. With hourly NTP sync, what is maximum clock error?',
    options: ['±2ms (drift between syncs)', '±10ms (NTP accuracy)', '±12ms (drift + NTP)', '0ms (always synced)'],
    correctAnswer: 2,
    explanation: 'Clock error combines drift and sync accuracy. Drift: +2ms/hour means clock gains 2ms between hourly syncs. NTP accuracy: ±10ms (query time, network jitter). Total error: drift + NTP = 2ms + 10ms = 12ms. Worst case: Just before sync, clock is +2ms ahead. After sync, NTP correction has ±10ms uncertainty. Total: 12ms. To reduce error: (1) Sync more frequently (every 10 min → 0.33ms drift), (2) Use better time source (PTP → 0.001ms accuracy), (3) Measure your specific NTP accuracy (may be better than 10ms on LAN).',
  },
  {
    id: 'timestamp-management-mc-2',
    question: 'NTP server responds with offset=-5ms (your clock is 5ms behind). What action to take?',
    options: ['Adjust clock +5ms immediately', 'Ignore (5ms acceptable)', 'Alert if > 50ms, otherwise adjust', 'Reboot system'],
    correctAnswer: 0,
    explanation: 'NTP offset correction: Offset = -5ms means your system clock is 5ms slow. Correction: Adjust +5ms to match NTP server. NTP daemon typically uses "slew" mode for small offsets (<128ms): gradually speeds up clock over minutes rather than instant jump. For offsets > 128ms, uses "step" mode: instant adjustment (can break running programs). 5ms offset: Apply immediately via slew. This prevents issues with monotonic time guarantees. Monitor: If offset consistently > 50ms, investigate NTP server connectivity or hardware clock issues. Never ignore offsets - they accumulate.',
  },
  {
    id: 'timestamp-management-mc-3',
    question: 'Exchange timestamp: 10:30:00.123456. Your receive timestamp: 10:30:00.125890. What is network latency?',
    options: ['2.434ms (receive - exchange)', 'Unable to determine (clock skew unknown)', '2ms (approximate)', '0.125890 seconds'],
    correctAnswer: 1,
    explanation: 'Latency measurement ambiguity: Cannot determine latency without synchronized clocks! If exchange clock = your clock (both UTC synced), then latency = 125890μs - 123456μs = 2434μs = 2.434ms. But if exchange clock is 5ms fast, actual latency = 2.434ms + 5ms = 7.434ms. To measure accurately: (1) Sync both clocks to atomic reference (NTP/PTP), (2) Measure clock offset vs reference, (3) Subtract offset: latency = (receive - exchange) - offset. Example: exchange +2ms fast, receive time 10:30:00.125890, exchange time 10:30:00.123456, offset = 2ms, latency = (125890 - 123456 - 2000) = 434μs actual latency.',
  },
  {
    id: 'timestamp-management-mc-4',
    question: 'SEC CAT requires microsecond precision. Python datetime.now() provides microsecond precision. Are you compliant?',
    options: ['Yes (datetime has microsecond precision)', 'No (need nanosecond precision)', 'Partial (also need accuracy requirement)', 'No (datetime is millisecond only)'],
    correctAnswer: 2,
    explanation: 'CAT requirements: Two separate requirements: (1) PRECISION = granularity of timestamp (microsecond = 1μs). Python datetime has microsecond precision ✓. (2) ACCURACY = how close to atomic clock (must be within 100ms). This requires NTP/PTP sync, not just datetime precision. Example: datetime.now() can represent 10:30:00.123456 (microsecond precision ✓), but if system clock is wrong by 200ms, you're non-compliant ✗. Solution: (1) Use datetime for precision ✓, (2) Sync clock with NTP to ensure accuracy ✓. Verify: Compare datetime.now() to time.time() * 1000000 (microseconds since epoch) - should match.',
  },
  {
    id: 'timestamp-management-mc-5',
    question: 'PTP grandmaster clock has GPS sync. What accuracy can PTP slaves achieve in same datacenter?',
    options: ['<1μs (sub-microsecond)', '1-10ms (like NTP)', '100μs typical', '<100ns (sub-100 nanoseconds)'],
    correctAnswer: 0,
    explanation: 'PTP datacenter accuracy: With hardware support (PTP NICs + switches), slaves achieve <1μs accuracy typically, <100ns in ideal conditions. GPS grandmaster: ±10ns accuracy (atomic reference). PTP switch: Hardware timestamps packets (no software delay). PTP NIC: Hardware timestamps on transmit/receive (no OS latency). Total: GPS (10ns) + network jitter (100-500ns) + NIC (50ns) = <1μs typical. Without hardware: PTP accuracy degrades to 10-100μs (software timestamps have OS latency). Investment: $1K NIC + $10K grandmaster + $5K switches = <1μs accuracy. HFT use case: Strategy latency 10μs, timestamp must be <1μs (10% budget), PTP essential.',
  },
];
