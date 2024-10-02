# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-locals

from ast import List
from collections import Counter
from typing import Iterator, Dict, Any

from in3120.posting import Posting
from .sieve import Sieve
from .ranker import Ranker
from .corpus import Corpus
from .invertedindex import InvertedIndex


class SimpleSearchEngine:
    """
    Realizes a simple query evaluator that efficiently performs N-of-M matching over an inverted index.
    I.e., if the query contains M unique query terms, each document in the result set should contain at
    least N of these m terms. For example, 2-of-3 matching over the query 'orange apple banana' would be
    logically equivalent to the following predicate:

       (orange AND apple) OR (orange AND banana) OR (apple AND banana)
       
    Note that N-of-M matching can be viewed as a type of "soft AND" evaluation, where the degree of match
    can be smoothly controlled to mimic either an OR evaluation (1-of-M), or an AND evaluation (M-of-M),
    or something in between.

    The evaluator uses the client-supplied ratio T = N/M as a parameter as specified by the client on a
    per query basis. For example, for the query 'john paul george ringo' we have M = 4 and a specified
    threshold of T = 0.7 would imply that at least 3 of the 4 query terms have to be present in a matching
    document.
    """

    def __init__(self, corpus: Corpus, inverted_index: InvertedIndex):
        self.__corpus = corpus
        self.__inverted_index = inverted_index

    def evaluate(self, query: str, options: Dict[str, Any], ranker: Ranker) -> Iterator[Dict[str, Any]]:
        """
        Evaluates the given query, doing N-out-of-M ranked retrieval. I.e., for a supplied query having M
        unique terms, a document is considered to be a match if it contains at least N <= M of those terms.

        The matching documents, if any, are ranked by the supplied ranker, and only the "best" matches are yielded
        back to the client as dictionaries having the keys "score" (float) and "document" (Document).

        The client can supply a dictionary of options that controls the query evaluation process: The value of
        N is inferred from the query via the "match_threshold" (float) option, and the maximum number of documents
        to return to the client is controlled via the "hit_count" (int) option.
        """
        # raise NotImplementedError("You need to implement this as part of the obligatory assignment.")

        query_terms = list(self.__inverted_index.get_terms(query))
        query_counts = Counter(query_terms)
        match_threshold = options["match_threshold"]
        hit_count = options["hit_count"]

        M = len(query_terms)
        N = max(1, min(M, int(match_threshold * M)))
        
        window = []
        its = [self.__inverted_index[term], term for term in query_counts.keys()]
        for it, term in its :
            window.append(next(its, False), its, term)


        sieve = Sieve(hit_count)
        i=0
        while all(window) :
            pivot, _, _ = sorted(window, key= lambda x : x[0].document_id)[i]
            copy = window.copy()

            # increments posting if lower than pivot
            for posting, it, term in window :
                if posting and posting.document_id < pivot.document_id :
                    next_posting = next(it, False)
                    window.remove((posting, it, term))
                    window.append((next_posting, it, term))

            # checks for match in window
            counter = Counter([posting.document_id for (posting, _, _) in window])
            for posting, it, term in window.copy() :
                if posting and counter[posting.document_id] >= N :
                    #compute score and add to sieve:
                    ranker.reset(posting.document_id)
                    ranker.update(term, query_counts[term], posting)
                    score = ranker.evaluate()
                    sieve.sift(score, posting.document_id)
                    # increments posting
                    next_posting = next(it, False)
                    window.remove((posting, it, term))
                    window.append((next_posting, it, term))
                    

            i+=1
            
            # checks if window has been modified, if not: pivot is next smallest posting
            if copy != window :
                i=0
    
        for score, doc_id in  sieve.winners() :
            document = self.__corpus.get_document(doc_id)
            yield {"score": score, "document": document}



        
