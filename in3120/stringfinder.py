# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=too-few-public-methods

from typing import Iterator, Dict, Any, List, Tuple
from .normalizer import Normalizer
from .tokenizer import Tokenizer
from .trie import Trie


class StringFinder:
    """
    Given a trie encoding a dictionary of strings, efficiently finds the subset of strings in the dictionary
    that are also present in a given text buffer. I.e., in a sense computes the "intersection" or "overlap"
    between the dictionary and the text buffer.

    Uses a trie-walk algorithm similar to the Aho-Corasick algorithm with some simplifications and some minor
    NLP extensions. The running time of this algorithm is virtually independent of the size of the dictionary,
    and linear in the length of the buffer we are searching in.

    The tokenizer we use when scanning the input buffer is assumed to be the same as the one that was used
    when adding strings to the trie.
    """

    def __init__(self, trie: Trie, normalizer: Normalizer, tokenizer: Tokenizer):
        self.__trie = trie
        self.__normalizer = normalizer  # The same as was used for trie building.
        self.__tokenizer = tokenizer  # The same as was used for trie building.

    def scan(self, buffer: str) -> Iterator[Dict[str, Any]]:
        """
        Scans the given buffer and finds all dictionary entries in the trie that are also present in the
        buffer. We only consider matches that begin and end on token boundaries.

        The matches, if any, are yielded back to the client as dictionaries having the keys "match" (str),
        "surface" (str), "meta" (Optional[Any]), and "span" (Tuple[int, int]). Note that "match" refers to
        the matching dictionary entry, "surface" refers to the content of the input buffer that triggered the
        match (the surface form), and "span" refers to the exact location in the input buffer where the surface
        form is found. Depending on the normalizer that is used, "match" and "surface" may or may not differ.

        A space-normalized version of the surface form is emitted as "surface", for convenience. Clients
        that require an exact surface form that is not space-normalized can easily reconstruct the desired
        string using the emitted "span" value.

        In a serious application we'd add more lookup/evaluation features, e.g., support for prefix matching,
        support for leftmost-longest matching (instead of reporting all matches), and more.
        """
        normalized_buffer = self.__normalize(buffer, True)
        terms = list(self.__tokenizer.strings(normalized_buffer)) + ["zzzzz"]
        raw_spans = list(self.__tokenizer.spans(buffer)) + [(0,0)]

        states = []
        
        for i, term in enumerate(terms):
            a, b = raw_spans[i]


            for matching_string, matching_span in states.copy() :
                match = self.__trie.consume(matching_string) 
                c, d = matching_span

                
                new_string =f'{matching_string} {term}' # should be "" when testing test_with_unigram_tokenizer_for_finding_arbitrary_substrings

                new_match = self.__trie.consume(new_string) 

                if match.is_final() :
                    states.remove((matching_string, (c,d)))
                    yield {"surface" : self.__normalize(buffer[c:d], False), "span" : (c,d), "match" : matching_string, "meta" : match.get_meta()}
                if new_match :
                    states.append((new_string, (c, b)))
                    if (matching_string, (c,d)) in states :
                        states.remove((matching_string, (c,d)))
 
            match = self.__trie.consume(term)

            if match : 
                states.append((term, (a,b)))
            

    def __normalize(self, buffer: str, normalize: bool) -> str :
        buffer = self.__normalizer.canonicalize(buffer)
        normalized_spans = self.__tokenizer.spans(buffer)
        if normalize :
            return self.__tokenizer.join([(self.__normalizer.normalize(buffer[a:b]), (a,b)) for a,b in normalized_spans])
       
        return self.__tokenizer.join([(buffer[a:b], (a,b)) for a,b in normalized_spans])
