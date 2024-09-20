# pylint: disable=missing-module-docstring

from typing import Iterator
from .posting import Posting


class PostingsMerger:
    """
    Utility class for merging posting lists.

    It is currently left unspecified what to do with the term frequency field
    in the returned postings when document identifiers overlap. Different
    approaches are possible, e.g., an arbitrary one of the two postings could
    be returned, or the posting having the smallest/largest term frequency, or
    a new one that produces an averaged value, or something else.
    """

    @staticmethod
    def intersection(
        iter1: Iterator[Posting], iter2: Iterator[Posting]
    ) -> Iterator[Posting]:
        """
        A generator that yields a simple AND(A, B) of two posting
        lists A and B, given iterators over these.

        The posting lists are assumed sorted in increasing order according
        to the document identifiers.
        """

        it1 = next(iter1, None)
        it2 = next(iter2, None)

        while it1 and it2:

            if it1.document_id > it2.document_id:
                it2 = next(iter2, None)
            elif it1.document_id < it2.document_id:
                it1 = next(iter1, None)
            else:
                yield it1
                it1 = next(iter1, None)
                it2 = next(iter2, None)

    @staticmethod
    def union(iter1: Iterator[Posting], iter2: Iterator[Posting]) -> Iterator[Posting]:
        """
        A generator that yields a simple OR(A, B) of two posting
        lists A and B, given iterators over these.

        The posting lists are assumed sorted in increasing order according
        to the document identifiers.
        """
        it1 = next(iter1, None)
        it2 = next(iter2, None)

        while it1 or it2:

            if it1 and not it2:
                yield it1
                it1 = next(iter1, None)
            elif it2 and not it1:
                yield it2
                it2 = next(iter2, None)
            elif it1 and it2:

                if it1.document_id > it2.document_id:
                    yield it2
                    it2 = next(iter2, None)
                elif it1.document_id < it2.document_id:
                    yield it1
                    it1 = next(iter1, None)
                else:
                    yield it1
                    it1 = next(iter1, None)
                    it2 = next(iter2, None)

    @staticmethod
    def difference(
        iter1: Iterator[Posting], iter2: Iterator[Posting]
    ) -> Iterator[Posting]:
        """
        A generator that yields a simple ANDNOT(A, B) of two posting
        lists A and B, given iterators over these.

        The posting lists are assumed sorted in increasing order according
        to the document identifiers.
        """

        it1 = next(iter1, None)
        it2 = next(iter2, None)

        while it1 or it2:

            if it1 and not it2:
                yield it1
                it1 = next(iter1, None)
            elif it2 and not it1:
                it2 = None
            elif it1 and it2:

                if it1.document_id > it2.document_id:
                    it2 = next(iter2, None)
                elif it1.document_id < it2.document_id:
                    yield it1
                    it1 = next(iter1, None)
                else:
                    it1 = next(iter1, None)
                    it2 = next(iter2, None)
