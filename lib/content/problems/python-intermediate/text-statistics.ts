/**
 * Advanced Text Statistics
 * Problem ID: intermediate-text-statistics
 * Order: 19
 */

import { Problem } from '../../../types';

export const intermediate_text_statisticsProblem: Problem = {
  id: 'intermediate-text-statistics',
  title: 'Advanced Text Statistics',
  difficulty: 'Medium',
  description: `Calculate comprehensive statistics for text documents.

**Statistics to Calculate:**
- Total words, characters, lines
- Unique words count
- Average word/sentence length
- Most common words
- Reading level (Flesch-Kincaid)
- Lexical diversity (unique words / total words)

**Flesch-Kincaid Reading Ease:**
\`206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)\``,
  examples: [
    {
      input: 'analyze_text("The quick brown fox...")',
      output: 'Comprehensive text statistics',
    },
  ],
  constraints: [
    'Calculate all statistics',
    'Handle edge cases',
    'Estimate syllable count',
  ],
  hints: [
    'Use collections.Counter for word frequency',
    'Split on sentence punctuation',
    'Estimate syllables by counting vowel groups',
  ],
  starterCode: `import re
from collections import Counter
import string

class TextAnalyzer:
    """
    Comprehensive text analysis tool.
    
    Examples:
        >>> analyzer = TextAnalyzer(text)
        >>> stats = analyzer.get_statistics()
        >>> print(stats['word_count'])
    """
    
    def __init__(self, text):
        """
        Initialize analyzer with text.
        
        Args:
            text: Text to analyze
        """
        self.text = text
        self.sentences = self._extract_sentences()
        self.words = self._extract_words()
    
    def _extract_sentences(self):
        """Extract sentences from text."""
        pass
    
    def _extract_words(self):
        """Extract and clean words from text."""
        pass
    
    def _count_syllables(self, word):
        """
        Estimate syllable count in word.
        
        Args:
            word: Word to analyze
            
        Returns:
            Estimated syllable count
        """
        pass
    
    def get_word_count(self):
        """Return total word count."""
        pass
    
    def get_character_count(self, include_spaces=True):
        """Return character count."""
        pass
    
    def get_unique_word_count(self):
        """Return count of unique words."""
        pass
    
    def get_average_word_length(self):
        """Return average word length."""
        pass
    
    def get_average_sentence_length(self):
        """Return average words per sentence."""
        pass
    
    def get_most_common_words(self, n=10):
        """
        Get most common words.
        
        Args:
            n: Number of words to return
            
        Returns:
            List of (word, count) tuples
        """
        pass
    
    def get_lexical_diversity(self):
        """
        Calculate lexical diversity ratio.
        
        Returns:
            Ratio of unique words to total words (0-1)
        """
        pass
    
    def get_reading_level(self):
        """
        Calculate Flesch-Kincaid reading ease score.
        
        Returns:
            Reading ease score (0-100, higher is easier)
        """
        pass
    
    def get_statistics(self):
        """
        Get all statistics as dictionary.
        
        Returns:
            Dictionary of statistics
        """
        pass


# Test
sample_text = """
The quick brown fox jumps over the lazy dog. This is a simple sentence.
Natural language processing is fascinating. It involves analyzing text data.
Python makes text analysis easy and efficient. We can calculate many statistics.
Reading level indicates text complexity. Short sentences are easier to read.
"""

analyzer = TextAnalyzer(sample_text)
stats = analyzer.get_statistics()

print("Text Statistics:")
print("=" * 50)
for key, value in stats.items():
    if isinstance(value, float):
        print(f"{key}: {value:.2f}")
    elif isinstance(value, list):
        print(f"{key}:")
        for item in value[:5]:  # Show first 5
            print(f"  {item}")
    else:
        print(f"{key}: {value}")
`,
  testCases: [
    {
      input: ['The quick brown fox jumps over the lazy dog.'],
      expected: 9,
    },
  ],
  solution: `import re
from collections import Counter
import string

class TextAnalyzer:
    def __init__(self, text):
        self.text = text
        self.sentences = self._extract_sentences()
        self.words = self._extract_words()
    
    def _extract_sentences(self):
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+', self.text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_words(self):
        # Remove punctuation and split
        text_clean = self.text.translate(str.maketrans(', ', string.punctuation))
        words = text_clean.lower().split()
        return [w for w in words if w]
    
    def _count_syllables(self, word):
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent 'e'
        if word.endswith('e'):
            syllable_count -= 1
        
        # Ensure at least one syllable
        return max(1, syllable_count)
    
    def get_word_count(self):
        return len(self.words)
    
    def get_character_count(self, include_spaces=True):
        if include_spaces:
            return len(self.text)
        return len(self.text.replace(' ', '))
    
    def get_unique_word_count(self):
        return len(set(self.words))
    
    def get_average_word_length(self):
        if not self.words:
            return 0
        return sum(len(word) for word in self.words) / len(self.words)
    
    def get_average_sentence_length(self):
        if not self.sentences:
            return 0
        return len(self.words) / len(self.sentences)
    
    def get_most_common_words(self, n=10):
        counter = Counter(self.words)
        return counter.most_common(n)
    
    def get_lexical_diversity(self):
        if not self.words:
            return 0
        return len(set(self.words)) / len(self.words)
    
    def get_reading_level(self):
        if not self.sentences or not self.words:
            return 0
        
        total_syllables = sum(self._count_syllables(word) for word in self.words)
        words_per_sentence = len(self.words) / len(self.sentences)
        syllables_per_word = total_syllables / len(self.words)
        
        # Flesch-Kincaid Reading Ease
        score = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
        return max(0, min(100, score))  # Clamp to 0-100
    
    def get_statistics(self):
        return {
            'total_characters': self.get_character_count(),
            'total_words': self.get_word_count(),
            'total_sentences': len(self.sentences),
            'unique_words': self.get_unique_word_count(),
            'average_word_length': self.get_average_word_length(),
            'average_sentence_length': self.get_average_sentence_length(),
            'lexical_diversity': self.get_lexical_diversity(),
            'reading_ease': self.get_reading_level(),
            'most_common_words': self.get_most_common_words(10)
        }`,
  timeComplexity: 'O(n) where n is text length',
  spaceComplexity: 'O(w) where w is number of words',
  order: 19,
  topic: 'Python Intermediate',
};
