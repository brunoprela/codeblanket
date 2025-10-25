import { MultipleChoiceQuestion } from '@/lib/types';

export const moduleProjectMarketDataDashboardMultipleChoice: MultipleChoiceQuestion[] =
    [
        {
            id: 'fm-1-14-mc-1',
            question:
                'A market data dashboard ingests 10 symbols at 100 ticks/second each (1,000 ticks/sec total). Using TimescaleDB, what is the recommended compression and retention strategy?',
            options: [
                'Keep all raw ticks forever, no compression',
                'Compress after 7 days, aggregate to 1-min bars after 30 days, delete after 1 year',
                'Delete ticks after 24 hours',
                'Store only daily OHLC bars',
            ],
            correctAnswer: 1,
            explanation:
                'Best practice: Raw ticks for 7 days (recent trading needs full detail), compress after 7 days (saves 95% space), downsample to 1-min bars after 30 days (historical analysis), delete raw ticks after 1 year (keep aggregates). This balances storage cost ($), query performance, and data usefulness. 1,000 ticks/sec = 86M ticks/day = 2.5TB/year uncompressed → 125GB compressed.',
        },
        {
            id: 'fm-1-14-mc-2',
            question:
                'Your dashboard websocket connection drops due to network issues. What is the best reconnection strategy?',
            options: [
                'Immediately reconnect (might overwhelm)',
                'Exponential backoff: 1s, 2s, 4s, 8s, max 60s',
                'Wait 5 minutes then reconnect',
                "Don't reconnect, wait for manual restart",
            ],
            correctAnswer: 1,
            explanation:
                'Exponential backoff prevents reconnection storms that overwhelm server. Start with 1s delay, double each attempt (1s, 2s, 4s, 8s, 16s, 32s, max 60s). If network issue affects 1000 clients, they reconnect at staggered times instead of simultaneously. Include jitter: randomize ±20% to further spread reconnections. Industry standard for resilient systems.',
        },
        {
            id: 'fm-1-14-mc-3',
            question:
                'A dashboard shows real-time P&L for a $100K portfolio. Using TimescaleDB continuous aggregates, how often should you recalculate portfolio value?',
            options: [
                'Every tick (sub-second)',
                'Every second',
                'Every minute',
                'Every hour',
            ],
            correctAnswer: 1,
            explanation:
                'Every second is optimal balance: Frequent enough for "real-time" feel, but allows batching multiple ticks into single calculation (efficient). Sub-second is overkill (humans can\'t perceive, wastes CPU). Every minute is too slow for active trading.Implementation: Continuous aggregate with 1 - second time bucket, frontend polls every second.For 10 symbols, this is ~10 queries / sec(trivial load).',
        },
        {
            id: 'fm-1-14-mc-4',
            question:
                'Your primary market data feed (Polygon.io) goes down. What is the best failover strategy?',
            options: [
                'Show error, wait for recovery',
                'Automatically failover to backup feed (Alpaca/IEX)',
                'Use cached data and freeze display',
                'Shut down entire system',
            ],
            correctAnswer: 1,
            explanation:
                'Best practice: Automatic failover to backup feed. Architecture: Primary feed (Polygon.io) + backup (Alpaca) + fallback (IEX free delayed). Monitor primary health (heartbeat), if no data for 10 seconds → switch to backup. Alert operators but keep system running. Cost: 2× data feeds, but critical for production. Cached data is stale (dangerous for trading). Manual intervention is too slow.',
        },
        {
            id: 'fm-1-14-mc-5',
            question:
                'A dashboard tracks 20 symbols with 1-second P&L updates. To optimize database queries, you should:',
            options: [
                'Query all 20 symbols individually (20 queries)',
                'Use single query with IN clause for all symbols',
                'Cache prices in Redis, update every second',
                'Pre-compute P&L with continuous aggregates',
            ],
            correctAnswer: 2,
            explanation:
                'Best: Cache latest prices in Redis, update from websocket stream, calculate P&L in-memory. Redis handles millions of reads/writes per second. Alternative: Database continuous aggregates work but slower. Querying DB 20×/second is inefficient. Architecture: Websocket → Redis (latest prices) → Dashboard polls Redis every second → Calculate P&L client-side. Database for historical only. Redis latency: <1ms vs DB: 10-50ms.',
        },
    ];
