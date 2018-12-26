"""
The inverted index class. This file only contain a simple in-memory inverted index.

Authors: Hamed Zamani (zamani@cs.umass.edu)
"""

import pickle as pkl


class InvertedIndex(object):
    def __init__(self, n_latent_terms):
        self.n_latent_terms = n_latent_terms

    def add(self, doc_ids, doc_repr):
        raise Exception('Please implement your own inverted index! It can be in memory inverted index or any kind of '
                        'database in the hard disk. This method adds a batch of documents to the index.')


class InMemoryInvertedIndex(InvertedIndex):
    def __init__(self, n_latent_terms):
        super(InMemoryInvertedIndex, self).__init__(n_latent_terms)
        self.index = dict()

    def add(self, doc_ids, doc_repr):
        for i in range(len(doc_ids)):
            for j in range(len(doc_repr[i])):
                if doc_repr[i][j] > 0.:
                    if j not in self.index:
                        self.index[j] = []
                    self.index[j].append((doc_ids[i], doc_repr[i][j]))

    def store(self, index_path):
        pkl.dump(self.index, open(index_path, 'wb'))

    def load(self, index_path):
        self.index = pkl.load(open(index_path, 'rb'))
