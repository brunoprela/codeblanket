export const edgeComputingForLlms = {
  title: 'Edge Computing for LLMs',
  content: `

# Edge Computing for LLM Applications

## Introduction

Edge computing brings your LLM application closer to users by deploying at the "edge" of the network—on CDN nodes worldwide, serverless functions at regional data centers, or even on user devices. For LLM applications, edge deployment offers dramatic benefits:

- **50-90% Latency Reduction**: Serve from nearby locations (200ms → 20ms)
- **Better User Experience**: Sub-100ms responses feel instant
- **Lower Bandwidth Costs**: Reduce data transfer between regions
- **Improved Reliability**: Continue operating if main region fails
- **Regulatory Compliance**: Keep EU data in EU, etc.
- **Scalability**: Leverage CDN infrastructure globally

**Key Insight**: For LLM apps, the bottleneck is often the network round-trip time, not the LLM API call itself. Edge computing eliminates this bottleneck.

---

## Edge vs Traditional Deployment

### Traditional Centralized Deployment

\`\`\`
User in Tokyo → (150ms) → US-East Server → (2000ms) → OpenAI API → (2000ms) → Server → (150ms) → User
Total: ~4300ms
\`\`\`

### Edge Deployment

\`\`\`
User in Tokyo → (10ms) → Tokyo Edge Node → (50ms) → OpenAI API (Tokyo) → (50ms) → Edge → (10ms) → User
Total: ~120ms (36x faster!)
\`\`\`

### When to Use Edge Computing

✅ **Good For**:
- Cached responses (highest benefit)
- Simple API routing/proxying
- Authentication and rate limiting
- Static content delivery
- Geolocation-based routing

❌ **Not Ideal For**:
- Complex server-side logic
- Large file processing
- Long-running computations
- Stateful sessions (without external state)

---

## Cloudflare Workers for LLM Apps

Cloudflare Workers run on 300+ edge locations worldwide.

### Basic LLM Proxy

\`\`\`typescript
// worker.ts - Deploy to Cloudflare Workers
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    // Handle CORS
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type',
        },
      });
    }

    const { prompt, user_id } = await request.json();

    // Check rate limit at the edge (using Cloudflare KV)
    const rateLimitKey = \`ratelimit:\${user_id}\`;
    const requests = await env.KV.get(rateLimitKey);
    
    if (requests && parseInt(requests) > 100) {
      return new Response('Rate limit exceeded', { status: 429 });
    }

    // Increment rate limit counter
    await env.KV.put(
      rateLimitKey, 
      (parseInt(requests || '0') + 1).toString(), 
      { expirationTtl: 3600 }
    );

    // Proxy to OpenAI
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': \`Bearer \${env.OPENAI_API_KEY}\`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-3.5-turbo',
        messages: [{ role: 'user', content: prompt }],
      }),
    });

    const data = await response.json();

    return new Response(JSON.stringify(data), {
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
      },
    });
  },
};

// Deploy with: npx wrangler deploy
\`\`\`

### Edge Caching for LLM Responses

\`\`\`typescript
// Advanced edge caching
import { createHash } from 'crypto';

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const { prompt, model } = await request.json();

    // Generate cache key from prompt
    const cacheKey = \`llm:\${createHash('sha256').update(prompt + model).digest('hex')}\`;

    // Check cache first
    const cached = await env.KV.get(cacheKey);
    if (cached) {
      console.log('✅ Edge cache HIT');
      return new Response(cached, {
        headers: {
          'Content-Type': 'application/json',
          'X-Cache': 'HIT',
          'Access-Control-Allow-Origin': '*',
        },
      });
    }

    console.log('❌ Edge cache MISS');

    // Call LLM API
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': \`Bearer \${env.OPENAI_API_KEY}\`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        messages: [{ role: 'user', content: prompt }],
      }),
    });

    const data = await response.text();

    // Cache response for 1 hour
    await env.KV.put(cacheKey, data, {
      expirationTtl: 3600,
    });

    return new Response(data, {
      headers: {
        'Content-Type': 'application/json',
        'X-Cache': 'MISS',
        'Access-Control-Allow-Origin': '*',
      },
    });
  },
};
\`\`\`

**Performance Impact**:
- Cache HIT: ~10ms (edge only)
- Cache MISS: ~2000ms (edge → LLM API)
- **99x faster for cached responses!**

---

## Vercel Edge Functions

Vercel Edge Functions run on Vercel's global edge network.

### Edge API Route

\`\`\`typescript
// app/api/chat/route.ts
import { NextRequest, NextResponse } from 'next/server';

// Enable edge runtime
export const runtime = 'edge';

export async function POST(req: NextRequest) {
  const { messages, user_id } = await req.json();

  // Get user's location from request
  const country = req.geo?.country || 'US';
  const city = req.geo?.city || 'Unknown';
  
  console.log(\`Request from \${city}, \${country}\`);

  // Check rate limit (using Vercel KV)
  const { kv } = await import('@vercel/kv');
  const rateLimitKey = \`ratelimit:\${user_id}\`;
  const requests = await kv.get<number>(rateLimitKey);

  if (requests && requests > 100) {
    return NextResponse.json(
      { error: 'Rate limit exceeded' },
      { status: 429 }
    );
  }

  await kv.incr(rateLimitKey);
  await kv.expire(rateLimitKey, 3600);

  // Call OpenAI
  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': \`Bearer \${process.env.OPENAI_API_KEY}\`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'gpt-3.5-turbo',
      messages,
    }),
  });

  const data = await response.json();

  // Add metadata
  return NextResponse.json({
    ...data,
    metadata: {
      edge_location: city,
      country,
      cached: false,
    },
  });
}
\`\`\`

### Edge Middleware for Auth

\`\`\`typescript
// middleware.ts
import { NextRequest, NextResponse } from 'next/server';

export const config = {
  matcher: '/api/chat/:path*',
};

export async function middleware(req: NextRequest) {
  // Verify auth token at the edge
  const token = req.headers.get('authorization')?.replace('Bearer ', ');

  if (!token) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  // Validate token (using edge-compatible JWT library)
  try {
    // Simple validation (in production, verify JWT signature)
    const isValid = token.length > 20;

    if (!isValid) {
      return NextResponse.json({ error: 'Invalid token' }, { status: 401 });
    }

    // Token valid, continue to API route
    return NextResponse.next();
  } catch (error) {
    return NextResponse.json({ error: 'Invalid token' }, { status: 401 });
  }
}
\`\`\`

---

## AWS Lambda@Edge

Lambda@Edge runs on AWS CloudFront edge locations.

### Edge Function for Caching

\`\`\`javascript
// lambda-edge-cache.js
const crypto = require('crypto');
const AWS = require('aws-sdk');
const dynamodb = new AWS.DynamoDB.DocumentClient();

exports.handler = async (event) => {
  const request = event.Records[0].cf.request;
  
  // Parse request body
  const body = Buffer.from(request.body.data, 'base64').toString();
  const { prompt, model } = JSON.parse(body);

  // Generate cache key
  const cacheKey = crypto
    .createHash('sha256')
    .update(prompt + model)
    .digest('hex');

  // Check DynamoDB for cached response
  try {
    const result = await dynamodb.get({
      TableName: 'llm-cache',
      Key: { cacheKey },
    }).promise();

    if (result.Item && result.Item.ttl > Date.now() / 1000) {
      console.log('✅ Cache HIT');
      
      // Return cached response
      return {
        status: '200',
        statusDescription: 'OK',
        headers: {
          'content-type': [{ key: 'Content-Type', value: 'application/json' }],
          'x-cache': [{ key: 'X-Cache', value: 'HIT' }],
        },
        body: result.Item.response,
      };
    }
  } catch (error) {
    console.error('Cache lookup error:', error);
  }

  console.log('❌ Cache MISS - forwarding to origin');
  
  // Add cache key to request for origin to use
  request.headers['x-cache-key'] = [{ key: 'X-Cache-Key', value: cacheKey }];
  
  return request;
};
\`\`\`

---

## Semantic Caching at the Edge

Cache similar queries, not just exact matches.

\`\`\`typescript
// edge-semantic-cache.ts
import { createHash } from 'crypto';

interface CachedQuery {
  prompt: string;
  embedding: number[];
  response: string;
  timestamp: number;
}

async function getEmbedding(text: string): Promise<number[]> {
  // Use a fast, edge-compatible embedding model
  // Or call OpenAI embeddings API
  const response = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      'Authorization': \`Bearer \${OPENAI_API_KEY}\`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'text-embedding-3-small',
      input: text,
    }),
  });

  const data = await response.json();
  return data.data[0].embedding;
}

function cosineSimilarity(a: number[], b: number[]): number {
  const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const { prompt } = await request.json();

    // Get query embedding
    const queryEmbedding = await getEmbedding(prompt);

    // Check recent cached queries (stored in KV with metadata)
    const recentKeys = JSON.parse(await env.KV.get('recent_queries') || '[]');
    
    for (const key of recentKeys.slice(0, 20)) {
      const cached = await env.KV.get<CachedQuery>(key, 'json');
      
      if (cached) {
        const similarity = cosineSimilarity(queryEmbedding, cached.embedding);
        
        if (similarity > 0.95) {
          console.log(\`✅ Semantic cache HIT (similarity: \${similarity.toFixed(2)})\`);
          
          return new Response(JSON.stringify({
            response: cached.response,
            cached: true,
            similarity,
          }), {
            headers: { 'Content-Type': 'application/json' },
          });
        }
      }
    }

    console.log('❌ Semantic cache MISS');

    // Call LLM API
    const llmResponse = await callLLM(prompt);

    // Cache with embedding
    const cacheKey = \`semantic:\${createHash('sha256').update(prompt).digest('hex')}\`;
    await env.KV.put(cacheKey, JSON.stringify({
      prompt,
      embedding: queryEmbedding,
      response: llmResponse,
      timestamp: Date.now(),
    }), { expirationTtl: 3600 });

    // Update recent queries list
    recentKeys.unshift(cacheKey);
    await env.KV.put('recent_queries', JSON.stringify(recentKeys.slice(0, 100)));

    return new Response(JSON.stringify({
      response: llmResponse,
      cached: false,
    }), {
      headers: { 'Content-Type': 'application/json' },
    });
  },
};
\`\`\`

---

## Geographic Routing

Route users to nearest LLM API endpoint.

\`\`\`typescript
// geo-routing.ts
const OPENAI_ENDPOINTS = {
  'US': 'https://api.openai.com',
  'EU': 'https://api.openai.com', // OpenAI doesn't have regional endpoints
  'ASIA': 'https://api.openai.com',
};

// For providers with regional endpoints (e.g., Azure OpenAI)
const AZURE_OPENAI_ENDPOINTS = {
  'US': 'https://your-resource-us.openai.azure.com',
  'EU': 'https://your-resource-eu.openai.azure.com',
  'ASIA': 'https://your-resource-asia.openai.azure.com',
};

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    // Get user's location
    const country = request.headers.get('cf-ipcountry') || 'US';
    
    // Determine region
    let region = 'US';
    if (['GB', 'FR', 'DE', 'IT', 'ES', 'NL'].includes(country)) {
      region = 'EU';
    } else if (['JP', 'CN', 'KR', 'SG', 'IN'].includes(country)) {
      region = 'ASIA';
    }

    console.log(\`User from \${country} → routing to \${region}\`);

    // Route to nearest endpoint
    const endpoint = AZURE_OPENAI_ENDPOINTS[region];

    const { prompt } = await request.json();

    const response = await fetch(\`\${endpoint}/openai/deployments/gpt-35-turbo/chat/completions?api-version=2023-05-15\`, {
      method: 'POST',
      headers: {
        'api-key': env.AZURE_OPENAI_KEY,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messages: [{ role: 'user', content: prompt }],
      }),
    });

    return new Response(await response.text(), {
      headers: {
        'Content-Type': 'application/json',
        'X-Region': region,
      },
    });
  },
};
\`\`\`

---

## Edge Computing Limitations

### What You CAN'T Do at the Edge

1. **CPU Time Limits**:
   - Cloudflare Workers: 50ms (free) or 30 seconds (paid)
   - Vercel Edge: 25 seconds
   - Lambda@Edge: 5 seconds

2. **Memory Limits**:
   - Cloudflare Workers: 128 MB
   - Vercel Edge: 4 GB
   - Lambda@Edge: 128-512 MB

3. **No Persistent Connections**:
   - Can't maintain WebSocket connections
   - Can't keep database connections open
   - Each request is isolated

4. **Limited Libraries**:
   - Must be edge-compatible (no Node.js APIs)
   - Smaller runtime environment
   - Some npm packages won't work

### Workarounds

\`\`\`typescript
// BAD: This won't work at the edge
import fs from 'fs'; // ❌ Node.js API not available
import database from 'pg'; // ❌ Can't maintain connections

// GOOD: Edge-compatible alternatives
import { kv } from '@vercel/kv'; // ✅ Edge-compatible storage
import { fetch } from '@vercel/edge'; // ✅ Built-in fetch

// For long-running tasks, offload to background
export default {
  async fetch(request: Request) {
    // Handle quick response at edge
    const quickResponse = await generateQuickResponse();

    // Trigger background job for expensive work
    await fetch('https://your-api.com/background-job', {
      method: 'POST',
      body: JSON.stringify({ task: 'expensive-processing' }),
    });

    return new Response(quickResponse);
  },
};
\`\`\`

---

## Monitoring Edge Performance

\`\`\`typescript
// edge-monitoring.ts
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const startTime = Date.now();
    const { prompt } = await request.json();

    try {
      // Check cache
      const cached = await env.KV.get(\`cache:\${prompt}\`);
      
      if (cached) {
        const latency = Date.now() - startTime;
        
        // Log metrics
        await logMetric({
          type: 'cache_hit',
          latency,
          location: request.headers.get('cf-ray'),
          timestamp: Date.now(),
        });

        return new Response(cached, {
          headers: {
            'X-Cache': 'HIT',
            'X-Latency': latency.toString(),
          },
        });
      }

      // Call LLM
      const response = await callLLM(prompt);
      const latency = Date.now() - startTime;

      // Log metrics
      await logMetric({
        type: 'cache_miss',
        latency,
        location: request.headers.get('cf-ray'),
        timestamp: Date.now(),
      });

      // Cache response
      await env.KV.put(\`cache:\${prompt}\`, response, {
        expirationTtl: 3600,
      });

      return new Response(response, {
        headers: {
          'X-Cache': 'MISS',
          'X-Latency': latency.toString(),
        },
      });

    } catch (error) {
      const latency = Date.now() - startTime;
      
      // Log error
      await logMetric({
        type: 'error',
        error: error.message,
        latency,
        timestamp: Date.now(),
      });

      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
      });
    }
  },
};

async function logMetric(metric: any) {
  // Send to analytics service
  await fetch('https://analytics.example.com/metrics', {
    method: 'POST',
    body: JSON.stringify(metric),
  });
}
\`\`\`

---

## Best Practices

### 1. Cache Aggressively at the Edge
- 90%+ cache hit rate possible for common queries
- Use semantic caching for similar queries
- Set appropriate TTLs (1 hour for stable content)

### 2. Keep Edge Functions Lightweight
- < 1MB code size
- Minimal dependencies
- Fast execution (< 50ms)

### 3. Handle Failures Gracefully
- Fallback to origin on edge failure
- Return cached response if API fails
- Provide degraded service vs no service

### 4. Monitor Edge Performance
- Track latency by location
- Monitor cache hit rates
- Alert on edge errors

### 5. Use Edge for What It's Good At
- Authentication
- Rate limiting
- Caching
- Routing
- Simple transformations

---

## Summary

Edge computing for LLM applications provides:

- **50-90% latency reduction** by serving from nearby locations
- **Massive cost savings** through aggressive edge caching
- **Better reliability** with distributed infrastructure
- **Improved UX** with sub-100ms responses
- **Global scale** leveraging CDN infrastructure

**Perfect for**: Caching, routing, auth, rate limiting  
**Not ideal for**: Complex logic, long computations, stateful sessions

With proper edge deployment, achieve <100ms response times globally while reducing costs by 60-80%.

`,
  exercises: [
    {
      prompt:
        'Deploy an LLM proxy to Cloudflare Workers with edge caching. Measure latency improvement from your location vs a centralized server.',
      solution: `Use Cloudflare Workers template above, deploy with wrangler. Test from multiple locations using tools like Pingdom or AWS CloudWatch Synthetics. Expected: 50-90% latency reduction.`,
    },
    {
      prompt:
        'Implement semantic caching at the edge that caches similar queries with 95%+ similarity. Measure cache hit rate improvement vs exact matching.',
      solution: `Use semantic caching implementation above. Test with paraphrased queries. Expected: 30-50% higher cache hit rate than exact matching.`,
    },
    {
      prompt:
        'Build geographic routing that sends EU users to EU endpoints and US users to US endpoints. Measure latency differences.',
      solution: `Use geo-routing template, deploy to edge. Test with VPN from different regions. Expected: 30-70% latency reduction vs single region.`,
    },
  ],
};
