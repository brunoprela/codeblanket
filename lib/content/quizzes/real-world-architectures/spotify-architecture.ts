/**
 * Quiz questions for Spotify Architecture section
 */

export const spotifyarchitectureQuiz = [
    {
        id: 'q1',
        question: 'Explain Spotify\'s audio streaming and caching strategy. How does it achieve smooth playback while minimizing bandwidth and storage?',
        sampleAnswer: 'Spotify uses adaptive streaming with local caching. Streaming: (1) Songs encoded in multiple bitrates (96, 160, 320 kbps). (2) Client selects bitrate based on network (cellular=160kbps, WiFi=320kbps). (3) Song split into 15-second chunks. (4) Prefetch next 30-60 seconds while playing current chunk. (5) Stream from CDN (Spotify uses Fastly + own CDN). Caching: (1) Recently played songs cached locally (5GB limit configurable). (2) Offline mode: download songs in highest quality. (3) Preload likely next songs based on playlist order. (4) P2P sharing: devices on same network share cached chunks (reduces bandwidth). Result: Instant playback from cache, smooth transition between songs, <5% rebuffering. Bandwidth optimization: Don\'t stream full song if user skips 10 seconds in, adaptive bitrate saves bandwidth on poor networks.',
        keyPoints: [
            'Adaptive streaming: Multiple bitrates (96-320 kbps), select based on network',
            '15-second chunks, prefetch next 30-60 seconds',
            'Local cache (5GB) for instant playback, P2P sharing on local network',
            'Optimizations: Preload playlist songs, adaptive bitrate, skip detection',
        ],
    },
    {
        id: 'q2',
        question: 'How does Spotify\'s recommendation system (Discover Weekly, Radio) work? What data and algorithms power personalized playlists?',
        sampleAnswer: 'Spotify recommendations use three main approaches: (1) Collaborative filtering - users with similar listening history likely enjoy similar songs. Matrix factorization: users × songs matrix, find latent factors. (2) Natural Language Processing - analyze blog posts, articles, song descriptions to understand genre, mood, themes. Extract keywords, build semantic understanding. (3) Audio analysis - extract features from raw audio (tempo, key, loudness, danceability). ML model classifies songs by characteristics. Combine all three: For Discover Weekly, find songs that (a) similar users enjoyed, (b) match semantic profile of user\'s favorites, (c) have audio features user likes. Generate weekly on Sunday night (batch job processes 200M users). Infrastructure: Spark for batch processing, Cassandra for storing user profiles, TensorFlow for ML models. Result: 40% of Discover Weekly songs saved by users.',
        keyPoints: [
            'Three approaches: collaborative filtering, NLP on text, audio analysis',
            'Matrix factorization for user-song latent factors',
            'Audio features: tempo, key, loudness, danceability',
            'Batch generation: Spark jobs process 200M users weekly',
        ],
    },
    {
        id: 'q3',
        question: 'Describe Spotify\'s microservices architecture and how it evolved from a monolith. What benefits and challenges came with the migration?',
        sampleAnswer: 'Spotify started with Python monolith (2008-2013). As team grew to 1000+ engineers, monolith became bottleneck: slow deployments, scaling issues, team dependencies. Migration to microservices (2013-2016): (1) Identify bounded contexts (User Service, Playlist Service, Recommendation Service, Payment Service). (2) Extract services one at a time. (3) Build infrastructure: service registry, API gateway, circuit breakers. (4) Event-driven: Kafka for inter-service communication. (5) Each team owns services end-to-end. Benefits: (1) Independent deployments - teams deploy 1000s of times daily. (2) Independent scaling - Recommendation Service scales differently than User Service. (3) Technology flexibility - mix of Java, Python, Go. (4) Team autonomy - small teams (6-12 people) own services. Challenges: (1) Debugging distributed systems. (2) Service discovery. (3) Data consistency across services. (4) Testing complexity. (5) Operational overhead (monitoring 100s of services).',
        keyPoints: [
            'Evolved from monolith to 100+ microservices (2013-2016)',
            'Benefits: Independent deployment/scaling, team autonomy, tech flexibility',
            'Infrastructure: Kafka for events, service registry, API gateway',
            'Challenges: Distributed debugging, service discovery, data consistency',
        ],
    },
];

