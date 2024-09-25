# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

import math
import sys
from bisect import bisect_left, insort
from itertools import takewhile
from typing import Any, Dict, Iterator, Iterable, Tuple, List
from collections import Counter
from .corpus import Corpus
from .normalizer import Normalizer
from .tokenizer import Tokenizer


class SuffixArray:
    """
    A simple suffix array implementation. Allows us to conduct efficient substring searches.
    The prefix of a suffix is an infix!

    In a serious application we'd make use of least common prefixes (LCPs), pay more attention
    to memory usage, and add more lookup/evaluation features.
    """

    def __init__(self, corpus: Corpus, fields: Iterable[str], normalizer: Normalizer, tokenizer: Tokenizer):
        self.__corpus = corpus
        self.__normalizer = normalizer
        self.__tokenizer = tokenizer
        self.__haystack: List[Tuple[int, str]] = []  # The (<document identifier>, <searchable content>) pairs.
        self.__suffixes: List[Tuple[int, int]] = []  # The sorted (<haystack index>, <start offset>) pairs.
        self.__build_suffix_array(fields)  # Construct the haystack and the suffix array itself.

    def __build_suffix_array(self, fields: Iterable[str]) -> None:
        """
        Builds a simple suffix array from the set of named fields in the document collection.
        The suffix array allows us to search across all named fields in one go.
        """

        for doc in self.__corpus :
            text= ''

            for field in fields :
                text += doc.get_field(field, '') + ' \0 '
            
            doc_id = doc.get_document_id()

            text = self.__normalizer.canonicalize(text)

            spans = list(self.__tokenizer.spans(text))

            for i, (start, _) in enumerate(spans) :
                self.__haystack.append((doc_id, self.__normalize(text[start:])))
                self.__suffixes.append((i, start)) # kanskje token index i stedet

        self.__suffixes.sort(key = lambda x : self.__haystack[x[0]][1])

    def __normalize(self, buffer: str) -> str: #for query
        """
        Produces a normalized version of the given string. Both queries and documents need to be
        identically processed for lookups to succeed.
        """
   
        return self.__normalizer.normalize(buffer)

    def __binary_search(self, needle: str) -> int:
        """
        Does a binary search for a given normalized query (the needle) in the suffix array (the haystack).
        Returns the position in the suffix array where the normalized query is either found, or, if not found,
        should have been inserted.

        Kind of silly to roll our own binary search instead of using the bisect module, but seems needed
        prior to Python 3.10 due to how we represent the suffixes via (index, offset) tuples. Version 3.10
        added support for specifying a key.
        """

        high = len(self.__haystack)-1 
        low = 0

        while (low <= high) :
            s_idx = (low + high) //2

            h_idx, _ = self.__suffixes[s_idx]
            _, substring = self.__haystack[h_idx]

            if (substring== needle) :
                return s_idx
            elif (substring < needle) :
                low = s_idx +1
            elif (substring > needle) :
                high = s_idx -1
        return low

    def evaluate(self, query: str, options: dict) -> Iterator[Dict[str, Any]]:
        """
        Evaluates the given query, doing a "phrase prefix search".  E.g., for a supplied query phrase like
        "to the be", we return documents that contain phrases like "to the bearnaise", "to the best",
        "to the behemoth", and so on. I.e., we require that the query phrase starts on a token boundary in the
        document, but it doesn't necessarily have to end on one.

        The matching documents are ranked according to how many times the query substring occurs in the document,
        and only the "best" matches are yielded back to the client. Ties are resolved arbitrarily.

        The client can supply a dictionary of options that controls this query evaluation process: The maximum
        number of documents to return to the client is controlled via the "hit_count" (int) option.

        The results yielded back to the client are dictionaries having the keys "score" (int) and
        "document" (Document).
        """
        hit_count = options["hit_count"]
    
        s_idx = self.__binary_search(self.__normalize(query))

        h_idx, _ = self.__suffixes[s_idx]
        de = self.__haystack[h_idx:h_idx+hit_count]

        # print(f'query {query} should be located at index {index} in doc {doc_id}')
        
        
        print(de)


        # content[index:] 

        for subs in self.__haystack[h_idx:h_idx+hit_count] :
            counter = Counter(subs.split())
            

            yield {"score": 0, "document" : "doc"}
