# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-locals

from collections import Counter
from typing import Iterator, Dict, Any
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
        match_threshold = options["match_threshold"]
        hit_count = options["hit_count"]

        M = len(query_terms)
        N = max(1, min(M, int(match_threshold * M)))

        counter = Counter()
        query_counts = Counter(query_terms)

        for term in query_terms :

            doc_ids = [posting.document_id for posting in self.__inverted_index[term]]
            counter.update(doc_ids) #counts all doc_id occurences across the posting lists for the terms in the query

        top_doc_ids = [doc_id for doc_id, count in counter.items() if count >= N] # extracts doc_id that appear more than N times
        sieve = Sieve(len(top_doc_ids))

        for term in query_terms :
            for posting in self.__inverted_index[term] :

                if posting.document_id in top_doc_ids :

                    ranker.reset(posting.document_id)
                    ranker.update(term, query_counts[term], posting)
                    score = ranker.evaluate()
                    sieve.sift(score, posting.document_id)

        for winner in list(sieve.winners())[:hit_count] :
            score = winner[0]
            document = self.__corpus.get_document(winner[1])
            yield {"score": score, "document": document}

"""
for each term in query (of length M):
    count occurences of docs for every term

prune documents to only get documents that appears in N out of the M terms of the query in the inverted index
rank them and sieve them
yield back hit_count best documents {score: float, document: Document}
"""