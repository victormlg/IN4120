# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

from __future__ import annotations
from typing import Iterable, Iterator, Dict, Tuple, Optional
from math import sqrt
from .sieve import Sieve

class SparseDocumentVector:
    """
    A simple representation of a sparse document vector. The vector space has one dimension
    per vocabulary term, and our representation only lists the dimensions that have non-zero
    values.

    Being able to place text buffers, be they documents or queries, in a vector space and
    thinking of them as point clouds (or, equivalently, as vectors from the origin) enables us
    to numerically assess how similar they are according to some suitable metric. Cosine
    similarity (the inner product of the vectors normalized by their lengths) is a very
    common metric.
    """

    def __init__(self, values: Dict[str, float]):
        # An alternative, effective representation would be as a
        # [(term identifier, weight)] list kept sorted by integer
        # term identifiers. Computing dot products would then be done
        # pretty much in the same way we do posting list AND-scans.
        self._values = values

        # We cache the length. It might get used over and over, e.g., for cosine
        # computations. A value of None triggers lazy computation.
        self._length : Optional[float] = None

    def __iter__(self):
        return iter(self._values.items())

    def __getitem__(self, term: str) -> float:
        return self._values.get(term, 0.0)

    def __setitem__(self, term: str, weight: float) -> None:
        self._values[term] = weight
        self._length = None

    def __contains__(self, term: str) -> bool:
        return term in self._values

    def __len__(self) -> int:
        """
        Enables use of the built-in len/1 function to count the number of non-zero
        dimensions in the vector. It is not for computing the vector's norm.
        """
        return len([v for v in self._values.values() if v != 0])

    def get_length(self) -> float:
        """
        Returns the length (L^2 norm, also called the Euclidian norm) of the vector.
        """
        self._length = sqrt(sum([v**2 for v in self._values.values()]))
        return self._length

    def normalize(self) -> None:
        """
        Divides all weights by the length of the vector, thus rescaling it to
        have unit length.
        """
        if not self._length :
            self.get_length()

        self._values = {k:(v/self._length) for k,v in self._values.items()}
        self._length = self.get_length()

    def top(self, count: int) -> Iterable[Tuple[str, float]]:
        """
        Returns the top weighted terms, i.e., the "most important" terms and their weights.
        """
        assert count >= 0 

        value_list = [(k,v) for k,v in self._values.items()]
        return sorted(value_list, key=lambda x : x[1], reverse=True)[:count]

    def truncate(self, count: int) -> None:
        """
        Truncates the vector so that it contains no more than the given number of terms,
        by removing the lowest-weighted terms.
        """
        top = self.top(count)
        self._values = {k:v for k,v in self._values.items() if (k,v) in top}
        self._length = self.get_length()

    def scale(self, factor: float) -> None:
        """
        Multiplies every vector component by the given factor.
        """
        self._values = {k:v*factor for k,v in self._values.items()}
        self._length = self.get_length()

    def dot(self, other: SparseDocumentVector) -> float:
        """
        Returns the dot product (inner product, scalar product) between this vector
        and the other vector.
        """
        return sum([v*other[k] for k,v in self._values.items()])

    def cosine(self, other: SparseDocumentVector) -> float:
        """
        Returns the cosine of the angle between this vector and the other vector.
        See also https://en.wikipedia.org/wiki/Cosine_similarity.
        """
        if self._length is None :
            self.get_length()
        if other._length is None :
            other.get_length()
            
        length1, length2 = self._length, other._length

        if length1 == 0 or length2 == 0 :
            return 0
        
        return self.dot(other)/(length1*length2)

    @staticmethod
    def centroid(vectors: Iterator[SparseDocumentVector]) -> SparseDocumentVector:
        """
        Computes the centroid of all the vectors, i.e., the average vector.
        """
        keys = set()
        [[keys.add(k) for k in vec._values] for vec in vectors]

        centroid = {k:sum([vec._values.get(k, 0) for vec in vectors])/len(vectors) for k in keys}
        return SparseDocumentVector(centroid)
