# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=unnecessary-pass
# pylint: disable=unused-argument

import itertools
from abc import ABC, abstractmethod
from collections import Counter
from typing import Iterable, Iterator, List, Tuple, Dict
from .dictionary import InMemoryDictionary
from .normalizer import Normalizer
from .tokenizer import Tokenizer
from .corpus import Corpus
from .posting import Posting
from .postinglist import CompressedInMemoryPostingList, InMemoryPostingList, PostingList


class InvertedIndex(ABC):
    """
    Abstract base class for a simple inverted index.
    """

    def __getitem__(self, term: str) -> Iterator[Posting]:
        return self.get_postings_iterator(term)

    def __contains__(self, term: str) -> bool:
        return self.get_document_frequency(term) > 0

    @abstractmethod
    def get_terms(self, buffer: str) -> Iterator[str]:
        """
        Processes the given text buffer and returns an iterator that yields normalized
        terms as they are indexed. Useful when both query strings and documents need to
        be identically processed. The terms produced from the given text buffer may or
        may not have been encountered index construction.
        """
        pass

    @abstractmethod
    def get_indexed_terms(self) -> Iterator[str]:
        """
        Returns an iterator over all unique terms that have been encountered during the
        construction of the inverted index, i.e., our term vocabulary or all terms for which
        there exists a posting list. The vocabulary is not listed in any particular order.
        This method might be useful if a client needs to build up some additional data
        structure over the term vocabulary, that is external to the inverted index.
        """
        pass

    @abstractmethod
    def get_postings_iterator(self, term: str) -> Iterator[Posting]:
        """
        Returns an iterator that can be used to iterate over the term's associated
        posting list. For out-of-vocabulary terms we associate empty posting lists.
        """
        pass

    @abstractmethod
    def get_document_frequency(self, term: str) -> int:
        """
        Returns the number of documents in the indexed corpus that contain the given term.
        """
        pass

    def get_collection_frequency(self, term: str) -> int:
        """
        Returns the number of times the given term occurs in the indexed corpus, across
        all documents.
        """
        return sum(p.term_frequency for p in self.get_postings_iterator(term))


class InMemoryInvertedIndex(InvertedIndex):
    """
    A simple in-memory implementation of an inverted index, suitable for small corpora.

    In a serious application we'd have configuration to allow for field-specific NLP,
    scale beyond current memory constraints, have a positional index, and so on.

    If index compression is enabled, only the posting lists are compressed. Dictionary
    compression is currently not supported.
    """

    def __init__(self, corpus: Corpus, fields: Iterable[str], normalizer: Normalizer, tokenizer: Tokenizer, compressed: bool = False):
        self._corpus = corpus
        self._normalizer = normalizer
        self._tokenizer = tokenizer
        self._posting_lists: List[PostingList] = []
        self._dictionary = InMemoryDictionary()
        self._build_index(fields, compressed)

    def __repr__(self):
        return str({term: self._posting_lists[term_id] for term, term_id in self._dictionary})

    def _build_index(self, fields: Iterable[str], compressed: bool) -> None:
        """
        Kicks off the indexing process. Basically implements a flavor of SPIMI indexing as described in
        https://nlp.stanford.edu/IR-book/html/htmledition/single-pass-in-memory-indexing-1.html but with
        the vastly simplifying assumption that everything fits in memory so we just have a single block
        and thus no need to merge per-block results.

        Note that we currently don't keep track of which field each term occurs in. If we were to allow
        fielded searches (e.g., "find documents that contain 'foo' in the 'title' field") then we would
        have to keep track of that, either as a synthetic term in the dictionary (e.g., 'foo.title') or
        as extra data in the posting. See https://nlp.stanford.edu/IR-book/html/htmledition/parametric-and-zone-indexes-1.html
        for further details.

        Also note that we are building a non-positional index, for simplicity. With a positional index
        we could offer clients the ability to do, e.g., phrase searches and proximity-based filtering and
        ranking. See https://nlp.stanford.edu/IR-book/html/htmledition/positional-indexes-1.html for
        further details.
        """
        raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

    def _add_to_dictionary(self, term: str) -> int:
        """
        Adds the given term to the dictionary, if it's not already present. If it's already present,
        the dictionary stays unchanged. Returns the term identifier assigned to the term.
        """
        return self._dictionary.add_if_absent(term)

    def _append_to_posting_list(self, term_id: int, document_id: int, term_frequency: int, compressed: bool) -> None:
        """
        Appends a new posting to the right posting list. The posting lists
        must be kept sorted so that we can efficiently traverse and
        merge them when querying the inverted index.
        """
        raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

    def _finalize_index(self):
        """
        Invoked at the very end after all documents have been processed. Provides
        implementations that need it with the chance to tie up any loose ends,
        if needed.
        """
        raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

    def get_terms(self, buffer: str) -> Iterator[str]:
        # In a serious large-scale application there could be field-specific tokenizers.
        # We choose to keep it simple here.
        tokens = self._tokenizer.strings(self._normalizer.canonicalize(buffer))
        return (self._normalizer.normalize(t) for t in tokens)

    def get_indexed_terms(self) -> Iterator[str]:
        # Assume that everything fits in memory. This would not be the case in a serious
        # large-scale application, even with compression.
        return (s for s, _ in self._dictionary)

    def get_postings_iterator(self, term: str) -> Iterator[Posting]:
        raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

    def get_document_frequency(self, term: str) -> int:
        raise NotImplementedError("You need to implement this as part of the obligatory assignment.")
