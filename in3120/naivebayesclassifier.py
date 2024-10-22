# pylint: disable=missing-module-docstring
# pylint: disable=line-too-long

import math
from collections import Counter
from typing import Any, Dict, Iterable, Iterator
from .dictionary import InMemoryDictionary
from .normalizer import Normalizer
from .tokenizer import Tokenizer
from .corpus import Corpus


class NaiveBayesClassifier:
    """
    Defines a multinomial naive Bayes text classifier. For a detailed primer, see
    https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html.
    """

    def __init__(self, training_set: Dict[str, Corpus], fields: Iterable[str],
                 normalizer: Normalizer, tokenizer: Tokenizer):
        """
        Trains the classifier from the named fields in the documents in the
        given training set.
        """
        # Used for breaking the text up into discrete classification features.
        self.__normalizer = normalizer
        self.__tokenizer = tokenizer

        # The vocabulary we've seen during training.
        self.__vocabulary = InMemoryDictionary()

        # Maps a category c to the logarithm of its prior probability,
        # i.e., c maps to log(Pr(c)).
        self.__priors: Dict[str, float] = {}

        # Maps a category c and a term t to the logarithm of its conditional probability,
        # i.e., (c, t) maps to log(Pr(t | c)).
        self.__conditionals: Dict[str, Dict[str, float]] = {}

        # Maps a category c to the denominator used when doing Laplace smoothing.
        self.__denominators: Dict[str, int] = {}

        # Train the classifier, i.e., estimate all probabilities.
        self.__compute_priors(training_set)
        self.__compute_vocabulary(training_set, fields)
        self.__compute_posteriors(training_set, fields)

    def __compute_priors(self, training_set) -> None:
        """
        Estimates all prior probabilities (or, rather, log-probabilities) needed for
        the naive Bayes classifier.
        """
        # number of documents of class c / total number of documents
        N = sum(len(corpus) for corpus in training_set.values())

        for c, corpus in training_set.items() :
            self.__priors[c] = math.log(len(corpus)/N)

    def __compute_vocabulary(self, training_set, fields) -> None:
        """
        Builds up the overall vocabulary as seen in the training set.
        """
        for corpus in training_set.values() : 
            for field in fields : 
                for doc in corpus :
                    for term in self.__get_terms(doc.get_field(field, "")) :
                        self.__vocabulary.add_if_absent(term)
                    

    def __compute_posteriors(self, training_set, fields) -> None:
        """
        Estimates all conditional probabilities (or, rather, log-probabilities) needed for
        the naive Bayes classifier.
        """
        # Number of occurences of word in docs of class c / total number of words in documents of class c
        for c, corpus in training_set.items():
            N = 0
            counter = Counter()
            for field in fields :
                tokens = [token for doc in corpus for token in self.__get_terms(doc.get_field(field, None))]
                counter += Counter(tokens)
                N+=len(tokens)
            self.__denominators[c] = N+len(self.__vocabulary)

            if c not in self.__conditionals :
                self.__conditionals[c] = {}

            for term, _ in self.__vocabulary :
                self.__conditionals[c][term] = math.log((counter[term]+1)/self.__denominators[c])

    def __get_terms(self, buffer) -> Iterator[str]:
        """
        Processes the given text buffer and returns the sequence of normalized
        terms as they appear. Both the documents in the training set and the buffers
        we classify need to be identically processed.
        """
        tokens = self.__tokenizer.strings(self.__normalizer.canonicalize(buffer))
        return (self.__normalizer.normalize(t) for t in tokens)

    def get_prior(self, category: str) -> float:
        """
        Given a category c, returns the category's prior log-probability log(Pr(c)).

        This is an internal detail having public visibility to facilitate testing.
        """
        return self.__priors[category]

    def get_posterior(self, category: str, term: str) -> float:
        """
        Given a category c and a term t, returns the posterior log-probability log(Pr(t | c)).

        This is an internal detail having public visibility to facilitate testing.
        """
        return self.__conditionals[category][term]

    def classify(self, buffer: str) -> Iterator[Dict[str, Any]]:
        """
        Classifies the given buffer according to the multinomial naive Bayes rule. The computed (score, category) pairs
        are emitted back to the client via the supplied callback sorted according to the scores. The reported scores
        are log-probabilities, to minimize numerical underflow issues. Logarithms are base e.

        The results yielded back to the client are dictionaries having the keys "score" (float) and
        "category" (str).
        """
        terms = self.__get_terms(buffer)
        posteriors = {c:sum(self.__conditionals[c].get(t, math.log(1/self.__denominators[c])) for t in terms) for c in self.__conditionals.keys()}

        predicted_scores = [(c, posteriors[c]+self.__priors[c]) for c in self.__conditionals.keys()]
        predicted_scores.sort(key=lambda x : x[1], reverse=True)

        for c, score in predicted_scores :
            yield {"score" : score, "category" : c}

        
