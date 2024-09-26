# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

import math
import sys
from bisect import bisect_left
from itertools import takewhile
from typing import Any, Dict, Iterator, Iterable, Tuple, List
from collections import Counter, defaultdict

from in3120.sieve import Sieve
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
            text = self.__normalize(text)
            spans = list(self.__tokenizer.spans(text))

            self.__haystack.append((doc_id, text))
            for start, _ in spans :
                self.__suffixes.append((doc_id, start)) 

        self.__suffixes.sort(key = self.__get_substring)

    def __get_substring(self, x) :
        i, offset = x 
        _, string = self.__haystack[i]
        return string[offset:]

    def __normalize(self, buffer: str) -> str: #for query
        """
        Produces a normalized version of the given string. Both queries and documents need to be
        identically processed for lookups to succeed.
        """

        tokens = self.__tokenizer.strings(self.__normalizer.canonicalize(buffer))
        return " ".join([self.__normalizer.normalize(t) for t in tokens])

    def __binary_search(self, needle: str) -> int:
        """
        Does a binary search for a given normalized query (the needle) in the suffix array (the haystack).
        Returns the position in the suffix array where the normalized query is either found, or, if not found,
        should have been inserted.

        Kind of silly to roll our own binary search instead of using the bisect module, but seems needed
        prior to Python 3.10 due to how we represent the suffixes via (index, offset) tuples. Version 3.10
        added support for specifying a key.
        """
        return bisect_left(self.__suffixes, needle, key=self.__get_substring)

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
        hit_count = options['hit_count']
        debug = options['debug']

        matches = []
        query = self.__normalize(query)
        index = self.__binary_search(query)-1
        counter = defaultdict(int)

        if not query : # if is an empty string
            return
        
        subs = self.__get_substring(self.__suffixes[index])

        while subs.startswith(query) : #moves backwards in case there are matches 
            index -=1
            subs = self.__get_substring(self.__suffixes[index])

        if index+1 >= len(self.__suffixes) : # if last index and no match
            return
        
        index+=1
        subs = self.__get_substring(self.__suffixes[index])

        while subs.startswith(query): # finds and counts all documents that match with the query

            doc_id, _ = self.__suffixes[index]
            matches.append(doc_id)
            counter[doc_id] +=1

            index +=1
            if index < len(self.__suffixes) :
                subs = self.__get_substring(self.__suffixes[index])
            else :
                subs = "" 

        if not matches : # if no matches
            return

        sieve = Sieve(len(matches))

        for doc_id in matches :
            sieve.sift(counter[doc_id], doc_id)
        
        for winner in list(sieve.winners())[:hit_count] :
            score = winner[0]
            document = self.__corpus.get_document(winner[1])
            yield {"score": score, "document": document}
