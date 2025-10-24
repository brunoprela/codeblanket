export const multiModalFundamentalsQuiz = [
  {
    id: 'mmas-fundamentals-q-1',
    question:
      'Design a multi-modal AI system for a real estate platform that needs to process property listings. The listings contain images, text descriptions, virtual tour videos, and neighborhood audio descriptions. How would you architect the system to handle all these modalities efficiently? Consider data ingestion, processing pipelines, storage, and retrieval for property search.',
    hint: 'Think about separate processing pipelines, unified embeddings, and hybrid search.',
    sampleAnswer: `A comprehensive multi-modal real estate system would include:

**Architecture:**
1. **Ingestion Layer**: Separate queues for each modality (images, videos, audio, text)
2. **Processing Pipelines**: 
   - Images: Extract features, generate descriptions, detect key features (bedrooms, kitchen, etc.)
   - Videos: Extract frames, transcribe narration, detect room transitions
   - Audio: Transcribe neighborhood descriptions, extract ambient sound features
   - Text: Parse structured data, extract key features, sentiment analysis
3. **Multi-Modal Fusion**: CLIP embeddings for unified text-image space, combined with audio transcriptions
4. **Storage**: Vector database (Pinecone/Weaviate) for embeddings, PostgreSQL for structured data, S3 for media
5. **Search**: Hybrid search combining vector similarity and traditional filters (price, location, size)

**Key Design Decisions:**
- Use CLIP for unified image-text embeddings to enable "show me similar properties" queries
- Cache popular searches and property analyses
- Process asynchronously - users don't wait for full processing
- Generate comprehensive metadata at ingestion for faster retrieval
- Implement quality checks to ensure all modalities are present and valid

**Production Considerations:**
- Batch process similar properties together
- Use lower quality models for initial processing, upgrade for popular listings
- Monitor costs per listing (target: <$0.10 per property)
- Implement CDN for media delivery
- Regular reindexing as property status changes`,
    keyPoints: [
      'Separate ingestion and processing pipelines for each modality',
      'Unified multi-modal embedding space for semantic search',
      'Hybrid search combining keyword and semantic retrieval',
      'Parallel processing and caching for efficiency',
      'Scalable architecture for ingestion, processing, and storage',
    ],
  },
  {
    id: 'mmas-fundamentals-q-2',
    question:
      'Imagine you are building a multi-modal content moderation system for a social media platform. Users can upload images, videos, and text posts. Describe how you would design a system to detect harmful content (e.g., hate speech, violence, explicit material) that might be present in a single modality or across multiple modalities (e.g., a benign image with harmful text overlay, or a video with harmful audio).',
    hint: 'Consider individual modality analysis, cross-modal consistency, and confidence scoring.',
    sampleAnswer: `Cross-modal content moderation presents unique challenges:

**Key Challenges:**
1. **Context Dependency**: Content may be benign in one modality but harmful when combined
2. **Evasion Techniques**: Users deliberately split harmful content across modalities
3. **Cultural Nuances**: Interpretation varies by context, culture, region
4. **Scale**: Must process millions of items daily with low latency
5. **False Positives**: Over-moderation damages user experience
6. **Evolving Threats**: New forms of harmful content emerge constantly

**System Design:**

**1. Multi-Stage Processing:**
- **Level 1 - Independent**: Process each modality separately for obvious violations
- **Level 2 - Combined**: Analyze modalities together for contextual violations
- **Level 3 - Human Review**: Queue borderline cases for human moderators

**2. Technical Implementation:**
\`\`\`
[Content Submission]
       ↓
[Separate Modality Analysis]
  - Image: Violence, nudity, hate symbols
  - Text: Hate speech, threats, spam
  - Audio: Transcribe + analyze speech
  - Video: Frame analysis + audio
       ↓
[Cross-Modal Analysis]
  - Combined context understanding
  - Relationship between modalities
  - Coded/hidden meaning detection
       ↓
[Risk Scoring]
  - Weighted combination of signals
  - Confidence thresholds
  - Human review triggering
       ↓
[Action Decision]
  - Auto-approve (high confidence safe)
  - Auto-remove (high confidence harmful)
  - Human review (medium confidence)
\`\`\`

**3. Key Features:**
- **Pattern Detection**: Learn patterns of harmful cross-modal combinations
- **User History**: Factor in user's past violations
- **Community Reports**: Weight human reports heavily
- **Regional Adaptation**: Different thresholds by region/culture
- **Appeal Process**: Allow users to contest decisions

**4. Production Patterns:**
- Process fast modalities first (text) for early filtering
- Batch similar content for efficiency
- Cache results for duplicate/similar content
- Monitor false positive/negative rates
- Regular model retraining with human feedback
- A/B test threshold changes carefully

**5. Metrics to Track:**
- Precision and recall per modality
- Cross-modal detection rate
- Processing latency (target: <2s for 90% of content)
- Human review queue size
- False positive rate (target: <0.1%)
- User appeal success rate

The system must balance safety, speed, and user experience while being adaptable to new threat types.`,
    keyPoints: [
      'Multi-stage processing with independent and combined analysis',
      'Risk scoring with confidence thresholds for human review',
      'Pattern detection for harmful cross-modal combinations',
      'Balance between safety, speed, and user experience',
    ],
  },
  {
    id: 'mmas-fundamentals-q-3',
    question:
      'Compare and contrast early fusion vs. late fusion architectures for multi-modal AI systems. For a customer service chatbot that needs to handle text queries, product images uploaded by customers, and voice messages, which approach would you choose and why? Include cost, latency, and accuracy considerations.',
    hint: 'Consider modularity, cost per query type, latency requirements, and flexibility.',
    sampleAnswer: `**Early Fusion vs. Late Fusion Analysis:**

**Early Fusion (Combining at Input):**
\`\`\`
[Image] → [Encoder] \\
[Text]  → [Encoder] → [Combined Features] → [Single Model] → [Output]
[Audio] → [Encoder] /
\`\`\`

**Advantages:**
- Deep integration of modalities
- Model learns cross-modal relationships
- Can capture subtle interactions between modalities
- Single unified representation

**Disadvantages:**
- More complex to train
- Requires aligned multi-modal training data
- Expensive inference (process all modalities always)
- Less flexible (hard to add/remove modalities)
- Higher latency (must process everything together)

**Late Fusion (Combining at Output):**
\`\`\`
[Image] → [Image Model] → [Image Output] \\
[Text]  → [Text Model]  → [Text Output]  → [Combine] → [Final Output]
[Audio] → [Audio Model] → [Audio Output] /
\`\`\`

**Advantages:**
- Modular (can upgrade individual models)
- Process modalities independently (parallel)
- Can use pre-trained models
- Flexible (optional modalities)
- Easier to debug and optimize

**Disadvantages:**
- May miss cross-modal interactions
- Multiple model calls (higher cost)
- Integration point requires careful design

**Recommendation for Customer Service Chatbot:**

**Choose Late Fusion with Smart Routing:**

**Reasoning:**
1. **Cost Efficiency**: Most queries are text-only. Don't process image/audio unnecessarily
   - Text-only: ~$0.001 per query
   - Text+Image: ~$0.015 per query
   - Text+Image+Audio: ~$0.025 per query

2. **Latency Optimization**: 
   - Text response: 1-2s
   - +Image analysis: +2-3s
   - +Audio transcription: +3-5s
   - Process modalities in parallel when multiple present

3. **Flexibility**: Customer service needs evolve
   - Easy to add video support later
   - Can upgrade text model without affecting image processing
   - A/B test different models per modality

**Implementation Design:**

\`\`\`python
class CustomerServiceSystem:
    def handle_query(self, text, image=None, audio=None):
        # Route based on modalities present
        results = {}
        
        # Always process text (fast)
        results['text_analysis',] = self.analyze_text(text)
        
        # Process other modalities in parallel if present
        if image and audio:
            with ThreadPoolExecutor() as executor:
                img_future = executor.submit(self.analyze_image, image)
                audio_future = executor.submit(self.transcribe_audio, audio)
                
                results['image_analysis',] = img_future.result()
                results['audio_transcript',] = audio_future.result()
        
        elif image:
            results['image_analysis',] = self.analyze_image(image)
        
        elif audio:
            results['audio_transcript',] = self.transcribe_audio(audio)
        
        # Combine insights
        return self.generate_response(results)
\`\`\`

**Hybrid Approach for Complex Cases:**
- Use late fusion for initial processing (fast, cheap)
- If confidence is low or query is ambiguous, use early fusion (GPT-4V with all modalities) for deeper analysis
- This gives best of both worlds: fast/cheap for common cases, accurate for complex ones

**Metrics to Optimize:**
- 90% of queries should be text-only (fastest path)
- Target latency: <2s for text, <5s with image, <8s with audio
- Cost per query: <$0.01 for 80% of queries
- Accuracy: >95% for single-modality, >90% for multi-modal
- User satisfaction: >4.5/5 stars

**Production Patterns:**
- Cache common product images
- Batch similar audio transcriptions
- Smart routing based on query type
- Progressive enhancement (text first, add modalities as needed)
- Monitor which modalities actually help accuracy

For customer service, late fusion with smart routing provides the best balance of cost, latency, accuracy, and flexibility.`,
    keyPoints: [
      'Late fusion offers modularity and cost efficiency',
      'Smart routing processes only necessary modalities',
      'Parallel processing of modalities when multiple present',
      'Hybrid approach: late fusion for common cases, early fusion for complex ones',
    ],
  },
];
