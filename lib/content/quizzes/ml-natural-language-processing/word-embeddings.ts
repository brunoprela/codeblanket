import { QuizQuestion } from '../../../types';

export const wordEmbeddingsQuiz: QuizQuestion[] = [
  {
    id: 'word-embeddings-dq-1',
    question:
      'Explain the fundamental difference between Word2Vec Skip-gram and CBOW architectures. When would you choose one over the other, and how does this choice affect training speed and quality?',
    sampleAnswer: `Skip-gram and CBOW are inverse architectures with different prediction tasks and trade-offs:

**Skip-gram: Target → Context**

Given a single target word, predict multiple surrounding context words.

Example: "I love machine learning"
- Input: "machine" (target)
- Outputs: "I", "love", "learning" (context)
- Creates multiple training examples per word

**CBOW: Context → Target**

Given multiple context words, predict the single target word.

Example: "I love machine learning"
- Inputs: "I", "love", "learning" (context)
- Output: "machine" (target)
- Averages context embeddings to predict target

**Key Differences:**

1. **Training Examples:**
   - Skip-gram: One target → many contexts (more training signals)
   - CBOW: Many contexts → one target (fewer training signals)

2. **Speed:**
   - Skip-gram: Slower (more training examples generated)
   - CBOW: Faster (one example per window)

3. **Rare Words:**
   - Skip-gram: Better for rare words (gets more training examples)
   - CBOW: Worse for rare words (averaged away in context)

4. **Quality:**
   - Skip-gram: Generally better embeddings, especially for rare words
   - CBOW: Good for frequent words, less computation

**When to Choose:**

Use **Skip-gram** when:
- Quality matters more than speed
- Dealing with rare words
- Have sufficient computational resources
- Need best possible embeddings

Use **CBOW** when:
- Speed is critical
- Working with frequent words only
- Limited computational resources
- Good-enough embeddings suffice

**Practical Impact:**

Research shows:
- Skip-gram: Better on word similarity tasks
- Skip-gram: Better for small corpora
- CBOW: 2-5x faster training
- CBOW: Good enough for many applications

The original Word2Vec paper (Mikolov et al., 2013) found Skip-gram superior for quality, which is why it's more commonly used despite being slower.`,
    keyPoints: [
      'Skip-gram predicts context from target; CBOW predicts target from context',
      'Skip-gram generates more training examples, better for rare words',
      'CBOW is faster (2-5x) but Skip-gram produces better embeddings',
      'Skip-gram recommended for quality; CBOW for speed',
      'Trade-off between computational cost and embedding quality',
    ],
  },
  {
    id: 'word-embeddings-dq-2',
    question:
      'FastText can generate embeddings for out-of-vocabulary (OOV) words using character n-grams, while Word2Vec cannot. Explain how this works and why it matters for production NLP systems, especially for handling typos, rare words, and morphological variations.',
    sampleAnswer: `FastText's character n-gram approach fundamentally changes how words are represented, enabling robust OOV handling:

**How FastText Works:**

Instead of representing each word as atomic, FastText represents words as bags of character n-grams.

Example: "learning" with n=3 to 6:
- 3-grams: <le, lea, ear, arn, rni, nin, ing, ng>
- 4-grams: <lea, lear, earn, arni, rnin, ning, ing>
- ...
- Special boundary markers < and > help distinguish prefixes/suffixes

**Word Embedding:**
\`\`\`
embedding("learning") = sum(
    embedding("<le") +
    embedding("lea") +
    embedding("ear") +
    ... +
    embedding("ing>")
)
\`\`\`

**OOV Word Handling:**

When encountering "learnings" (unseen):
\`\`\`
embedding("learnings") = sum(
    embedding("<le") +
    embedding("lea") +
    embedding("ear") +
    ... +
    embedding("ngs>")  # New suffix
)
\`\`\`

Many n-grams overlap with "learning", so the embedding is similar!

**Why This Matters in Production:**

**1. Typos and Misspellings:**

Word2Vec:
- "machine" → has embedding
- "machien" (typo) → KeyError, replaced with <UNK>
- Lost semantic information

FastText:
- "machine" and "machien" share n-grams: "mach", "ach", "chi", "ien"
- Generated embedding is similar to correct spelling
- Model remains robust

**2. Morphological Variations:**

Word2Vec:
- "learn", "learning", "learned", "learner" are completely separate
- If "learner" is rare, poor embedding quality

FastText:
- All share root n-grams: "lea", "ear", "arn"
- Automatically captures morphological relationship
- Similar embeddings without explicit rules

**3. Rare and Domain-Specific Terms:**

Production scenario: Medical NLP
- Training: General corpus
- Deployment: Medical texts with specialized terms

Word2Vec:
- "cardiopulmonary" (rare/unseen) → <UNK>
- Lost critical medical information

FastText:
- Breaks into: "card", "cardio", "iopul", "pulmo", "onary"
- Some n-grams seen in: "cardiac", "pulmonary"
- Generates reasonable embedding from components

**4. Multilingual Applications:**

Languages with rich morphology (German, Finnish, Turkish):
- Compound words: German "Donaudampfschifffahrt"
- Agglutination: Many morphemes per word

Word2Vec:
- Explosion of vocabulary
- Many rare words

FastText:
- Captures morphemes via n-grams
- Better generalization across word forms

**5. Social Media and User-Generated Content:**

Real production challenge:
- Users write: "sooooo goooood", "amazingggg"
- Hashtags: "#MachineLearning"
- Creative spelling

Word2Vec:
- Each variation is OOV
- Cannot understand meaning

FastText:
- Shared n-grams with standard spellings
- Generates reasonable embeddings
- More robust to noise

**Production Implementation:**

\`\`\`python
from gensim.models import FastText, Word2Vec

# Word2Vec fails on OOV
w2v = Word2Vec(...)
try:
    vec = w2v.wv['unseen_word']
except KeyError:
    vec = w2v.wv['<UNK>']  # Loses all semantic info

# FastText handles gracefully
ft = FastText(...)
vec = ft.wv['unseen_word']  # Always works!
similar = ft.wv.most_similar('unseen_word')  # Can find similar words
\`\`\`

**Performance Impact:**

Studies show:
- FastText handles 20-40% of words in test data that are OOV
- Reduces embedding failures in production
- Maintains accuracy on OOV words close to in-vocabulary performance
- Particularly valuable when fine-tuning is expensive

**Trade-offs:**

Advantages:
- Robust OOV handling
- Morphological awareness
- Better for rare words
- More production-ready

Disadvantages:
- Slower training (more n-grams)
- Larger model size (stores n-gram embeddings)
- More memory during inference

**Recommendation:**

Use FastText for production when:
- Expecting OOV words (social media, user input)
- Morphologically rich languages
- Cannot retrain frequently
- Typos and variations are common

Use Word2Vec when:
- Controlled vocabulary
- Speed is critical
- Memory constrained
- OOV rate is very low

**Modern Context:**

Transformer models (BERT) use subword tokenization (BPE, WordPiece) which provides similar OOV robustness at the architecture level. However, FastText remains valuable for:
- Lightweight applications
- Interpretable embeddings
- When transformers are too heavy`,
    keyPoints: [
      'FastText represents words as sum of character n-gram embeddings',
      'OOV words generated by summing their component n-grams',
      'Handles typos, morphology, rare words automatically',
      'Critical for production: 20-40% of words may be OOV',
      'Trade-off: slower training and larger models vs robustness',
      'Particularly valuable for social media, user input, morphologically rich languages',
    ],
  },
  {
    id: 'word-embeddings-dq-3',
    question:
      'The famous word embedding analogy "king - man + woman ≈ queen" demonstrates vector arithmetic. Explain what this reveals about learned semantic relationships and discuss limitations of static word embeddings using polysemous words like "bank" as an example.',
    sampleAnswer: `Vector arithmetic in embeddings reveals that semantic relationships are encoded as geometric relationships in vector space:

**Understanding the Analogy:**

\`\`\`
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
\`\`\`

**What This Means:**

1. **Gender as a Vector:**
   - vector("king") - vector("man") ≈ "royalty + masculine"
   - This difference captures the "masculine gender" direction
   - Adding vector("woman") adds "feminine gender"
   - Result points near "queen" (royalty + feminine)

2. **Learned Relationships:**
   - Model learned that kings:men :: queens:women
   - Not programmed—discovered from text patterns
   - "king" appears with male pronouns, "queen" with female
   - Co-occurrence patterns encode relationships geometrically

3. **Geometric Interpretation:**
   - Relationships become directions in vector space
   - Gender is one direction
   - Plural/singular is another direction
   - Verb tense is another direction

**More Examples:**

\`\`\`
Paris - France + Italy ≈ Rome (country capitals)
walking - walk + swim ≈ swimming (verb forms)
bigger - big + small ≈ smaller (comparatives)
\`\`\`

**What Embeddings Capture:**

The model learns:
- Semantic similarity: "king" near "queen", "monarch"
- Syntactic patterns: verb tenses, plurals
- Relational patterns: capital-country, gender, size
- Analogical reasoning: relationships as vectors

**Critical Limitation: Polysemy**

**The "bank" Problem:**

"bank" has multiple meanings:
1. Financial institution: "I deposited money at the bank"
2. River edge: "We sat on the river bank"

**Static Embeddings Limitation:**

Word2Vec/GloVe/FastText assign ONE vector to "bank":
\`\`\`
vector("bank") = average of all contexts
\`\`\`

This single vector must represent:
- Financial concepts (money, deposit, account)
- Geographical concepts (river, shore, edge)

Result: Compromised embedding that doesn't perfectly represent either meaning.

**Real Example:**

\`\`\`python
model.wv.most_similar('bank')
# Returns mix: ['financial', 'river', 'institution', 'shore', ...]
# Confusion from conflated meanings
\`\`\`

**More Polysemy Examples:**

- **"apple"**: fruit vs company (Apple Inc.)
- **"python"**: snake vs programming language
- **"rock"**: stone vs music genre
- **"bow"**: weapon vs bending gesture vs front of ship

**Production Impact:**

Document: "Apple released new iPhone"
- Static embedding for "apple" includes fruit context
- Irrelevant fruit semantics pollute representation
- Reduces precision in similarity/classification

Document: "I deposited check at bank"
- "bank" embedding includes river semantics
- Model might find spurious similarity with water-related docs

**Attempted Solutions (Pre-Transformer):**

1. **Multiple embeddings per word:**
   - Train separate embeddings for each sense
   - Requires sense disambiguation (complex)

2. **Context-aware averaging:**
   - Weight embedding by surrounding words
   - Still fundamentally static

3. **Multi-prototype embeddings:**
   - Cluster contexts, learn multiple vectors
   - Adds complexity, doesn't fully solve

**Why Static Embeddings Fail Here:**

Fundamental assumption:
\`\`\`
One word = One meaning = One vector
\`\`\`

Reality:
\`\`\`
One word = Multiple meanings = Needs multiple vectors
\`\`\`

**Frequency Matters:**

If "bank" appears 90% in financial contexts, 10% in river contexts:
- Embedding biased toward financial meaning
- River usage poorly represented
- Rare senses get lost

**The Solution: Contextualized Embeddings**

Modern approaches (ELMo, BERT, GPT):
\`\`\`
vector("bank", sentence_1) ≠ vector("bank", sentence_2)
\`\`\`

Each occurrence gets unique embedding based on context:
- "deposited at bank" → financial-context vector
- "river bank" → geographical-context vector

**Comparison:**

Static (Word2Vec):
\`\`\`python
vec_1 = w2v['bank']  # In "financial bank"
vec_2 = w2v['bank']  # In "river bank"
# vec_1 == vec_2 (same vector always!)
\`\`\`

Contextualized (BERT):
\`\`\`python
vec_1 = bert("I went to the bank", word_idx=4)
vec_2 = bert("We fished from the bank", word_idx=4)
# vec_1 ≠ vec_2 (different based on context!)
\`\`\`

**When Static Embeddings Still Work:**

Despite limitations, static embeddings remain useful when:
- Most words have dominant single meaning in domain
- Task doesn't require fine-grained disambiguation
- Speed/memory constraints prohibit transformers
- Interpretability matters (transformers are black boxes)
- Working with short phrases where context is minimal

**Key Insights:**

1. Vector arithmetic reveals learned semantic structure
2. Relationships encoded as geometric directions
3. Static embeddings average all contexts → polysemy problem
4. Contextualized embeddings solve this with dynamic representations
5. Trade-off: simplicity/speed (static) vs accuracy/context (dynamic)

**Historical Importance:**

Word2Vec/GloVe were breakthrough innovations that:
- Proved semantic relationships could be learned
- Enabled transfer learning in NLP
- Laid foundation for modern transformers
- Remain valuable baselines and lightweight alternatives

The analogy examples like "king - man + woman ≈ queen" demonstrated that machines could learn conceptual relationships from data alone—a fundamental insight that transformed NLP.`,
    keyPoints: [
      'Vector arithmetic captures learned semantic relationships as geometric directions',
      'king - man + woman ≈ queen shows gender encoded as vector direction',
      'Relationships (gender, plurality, tense) become navigable in vector space',
      'Critical limitation: one vector per word cannot handle polysemy (bank, apple)',
      'Static embeddings average all contexts, losing meaning disambiguation',
      'Contextualized embeddings (BERT) solve this with dynamic, context-dependent vectors',
      'Static embeddings remain useful baselines despite polysemy limitation',
    ],
  },
];
