/**
 * Webhook Design Section
 */

export const webhookdesignSection = {
  id: 'webhook-design',
  title: 'Webhook Design',
  content: `Webhooks allow your API to push real-time notifications to clients. Well-designed webhooks are reliable, secure, and easy to integrate.

## What are Webhooks?

**Webhooks** are HTTP callbacks: your API makes POST requests to client-specified URLs when events occur.

**Polling vs Webhooks**:

\`\`\`
Polling (inefficient):
Client: Are there new orders? → Server: No
Client: Are there new orders? → Server: No
Client: Are there new orders? → Server: Yes! Order #123

Webhooks (efficient):
Order created → Server: POST https://client.com/webhooks → Client: Received!
\`\`\`

## Webhook Design Patterns

### **Event Types**

Define clear event naming:

\`\`\`javascript
// Good: Hierarchical, clear
const WEBHOOK_EVENTS = {
  'order.created': 'Order was created',
  'order.updated': 'Order was updated',
  'order.cancelled': 'Order was cancelled',
  'order.fulfilled': 'Order was fulfilled',
  
  'payment.succeeded': 'Payment succeeded',
  'payment.failed': 'Payment failed',
  'payment.refunded': 'Payment refunded'
};
\`\`\`

### **Payload Structure**

Consistent payload format:

\`\`\`json
{
  "id": "evt_1234567890",
  "type": "order.created",
  "created_at": "2024-01-01T00:00:00Z",
  "data": {
    "object": "order",
    "id": "ord_abc123",
    "customer_id": "cus_xyz789",
    "total": 99.99,
    "status": "pending",
    "items": [
      {
        "product_id": "prod_123",
        "quantity": 2,
        "price": 49.99
      }
    ]
  }
}
\`\`\`

## Security

### **1. Webhook Signatures**

Verify requests come from your API:

\`\`\`javascript
const crypto = require('crypto');

// Server: Sign webhook payload
function signWebhook (payload, secret) {
  const signature = crypto
    .createHmac('sha256', secret)
    .update(JSON.stringify (payload))
    .digest('hex');
  
  return signature;
}

// Send webhook
async function sendWebhook (url, payload, secret) {
  const signature = signWebhook (payload, secret);
  
  await fetch (url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Webhook-Signature': signature,
      'X-Webhook-ID': payload.id
    },
    body: JSON.stringify (payload)
  });
}

// Client: Verify signature
function verifyWebhook (payload, signature, secret) {
  const expectedSignature = signWebhook (payload, secret);
  return crypto.timingSafeEqual(
    Buffer.from (signature),
    Buffer.from (expectedSignature)
  );
}

// Usage
app.post('/webhooks', (req, res) => {
  const signature = req.headers['x-webhook-signature'];
  const secret = process.env.WEBHOOK_SECRET;
  
  if (!verifyWebhook (req.body, signature, secret)) {
    return res.status(401).json({ error: 'Invalid signature' });
  }
  
  // Process webhook
  processWebhook (req.body);
  res.status(200).send('OK');
});
\`\`\`

### **2. Timestamp Validation**

Prevent replay attacks:

\`\`\`javascript
function verifyWebhookTimestamp (timestamp, maxAge = 300) {
  const now = Math.floor(Date.now() / 1000);
  const age = now - timestamp;
  
  if (age > maxAge) {
    throw new Error('Webhook too old');
  }
}
\`\`\`

## Reliability

### **1. Retry Strategy**

Retry failed webhooks with exponential backoff:

\`\`\`javascript
async function sendWebhookWithRetry (url, payload, maxRetries = 3) {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const response = await fetch (url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify (payload),
        timeout: 10000  // 10s timeout
      });
      
      if (response.ok) {
        return { success: true, attempt };
      }
      
      // Retry on 5xx errors
      if (response.status >= 500) {
        throw new Error(\`Server error: \${response.status}\`);
      }
      
      // Don't retry on 4xx errors (client error)
      return { success: false, status: response.status };
      
    } catch (error) {
      if (attempt === maxRetries - 1) {
        return { success: false, error: error.message };
      }
      
      // Exponential backoff: 1s, 2s, 4s
      const delay = Math.pow(2, attempt) * 1000;
      await sleep (delay);
    }
  }
}
\`\`\`

### **2. Dead Letter Queue**

Store failed webhooks for manual retry:

\`\`\`javascript
async function processWebhookQueue() {
  while (true) {
    const webhook = await webhookQueue.pop();
    
    if (!webhook) {
      await sleep(1000);
      continue;
    }
    
    const result = await sendWebhookWithRetry (webhook.url, webhook.payload);
    
    if (!result.success) {
      // Move to dead letter queue
      await deadLetterQueue.push({
        ...webhook,
        failedAt: Date.now(),
        error: result.error
      });
      
      // Alert team
      await alertFailedWebhook (webhook);
    }
  }
}
\`\`\`

### **3. Idempotency**

Clients should handle duplicate deliveries:

\`\`\`javascript
// Client: Track processed webhook IDs
const processedWebhooks = new Set();

app.post('/webhooks', (req, res) => {
  const webhookId = req.body.id;
  
  // Check if already processed
  if (processedWebhooks.has (webhookId)) {
    return res.status(200).send('Already processed');
  }
  
  // Process webhook
  processWebhook (req.body);
  
  // Mark as processed
  processedWebhooks.add (webhookId);
  
  res.status(200).send('OK');
});
\`\`\`

## Best Practices

1. **Return 200 quickly**: Acknowledge receipt, process async
2. **Retry with backoff**: Don't overwhelm clients
3. **Sign payloads**: Verify webhook authenticity
4. **Timestamp validation**: Prevent replay attacks
5. **Idempotency**: Clients handle duplicates
6. **Timeout**: Don't wait forever (10-30s)
7. **Dead letter queue**: Store failed webhooks
8. **Monitor**: Track delivery success rates
9. **Documentation**: Clear event types and payloads
10. **Test endpoints**: Provide webhook testing tools`,
};
