"""Vocabulary management class"""

import pickle
from collections import Counter
from typing import Dict, List, Optional

from nltk.tokenize import word_tokenize


class Vocabulary:
    """Vocabulary management class"""

    def __init__(
        self,
        min_freq: int = 3,
        max_size: Optional[int] = None,
        special_tokens: Optional[List[str]] = None,
    ):
        """
        Args:
            min_freq: Minimum frequency
            max_size: Maximum vocabulary size
            special_tokens: Special tokens
        """
        self.min_freq = min_freq
        self.max_size = max_size

        # Special tokens
        self.special_tokens = special_tokens or ["<PAD>", "<UNK>", "<START>", "<END>"]

        # Vocabulary dictionaries
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}

        # Vocabulary size
        self.vocab_size = 0

        # Initialize
        self._build_vocab()

    def _build_vocab(self):
        """Build vocabulary"""
        # Add special tokens
        for i, token in enumerate(self.special_tokens):
            self.word2idx[token] = i
            self.idx2word[i] = token

        self.vocab_size = len(self.special_tokens)

    def build_from_texts(self, texts: List[str]):
        """Build vocabulary from texts"""
        # Count word occurrences
        word_counts: Counter[str] = Counter()

        for text in texts:
            tokens = word_tokenize(text.lower())
            word_counts.update(tokens)

        # Filter by minimum frequency
        filtered_words = [
            word for word, count in word_counts.items() if count >= self.min_freq
        ]

        # Limit by maximum size
        if self.max_size:
            filtered_words = filtered_words[: self.max_size - self.vocab_size]

        # Add to vocabulary
        for word in filtered_words:
            if word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

    def encode(self, text: str) -> List[int]:
        """Convert text to indices"""
        tokens = word_tokenize(text.lower())
        indices = []

        for token in tokens:
            if token in self.word2idx:
                indices.append(self.word2idx[token])
            else:
                indices.append(self.word2idx["<UNK>"])

        return indices

    def decode(self, indices: List[int]) -> str:
        """Convert indices to text"""
        tokens = []
        for idx in indices:
            if idx in self.idx2word:
                tokens.append(self.idx2word[idx])
            else:
                tokens.append("<UNK>")

        return " ".join(tokens)

    def save(self, filepath: str):
        """Save vocabulary"""
        vocab_data = {
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "vocab_size": self.vocab_size,
            "min_freq": self.min_freq,
            "max_size": self.max_size,
            "special_tokens": self.special_tokens,
        }

        with open(filepath, "wb") as f:
            pickle.dump(vocab_data, f)

    @classmethod
    def load(cls, filepath: str) -> "Vocabulary":
        """Load vocabulary"""
        with open(filepath, "rb") as f:
            vocab_data = pickle.load(f)

        vocab = cls(
            min_freq=vocab_data["min_freq"],
            max_size=vocab_data["max_size"],
            special_tokens=vocab_data["special_tokens"],
        )

        vocab.word2idx = vocab_data["word2idx"]
        vocab.idx2word = vocab_data["idx2word"]
        vocab.vocab_size = vocab_data["vocab_size"]

        return vocab
