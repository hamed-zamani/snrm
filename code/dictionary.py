"""
A dictionary class that contains vocabulary terms and their IDs.

Authors: Hamed Zamani (zamani@cs.umass.edu)
"""

class Dictionary(object):
    """
    This class contains a list of vocabulary terms and their mappings to term ids. The ids are zero indexed. The index
    zero is corresponding to 'UNKNOWN' terms.

    Attributes:
        id_to_term (list): a list of string containing vocabulary terms.
        term_to_id (dict): a dict of terms (str) to their ids (int).
    """

    def __init__(self):
        self.id_to_term = ['UNKNOWN']
        self.term_to_id = {'UNKNOWN': 0}

    def load_from_galago_dump(self, file_name, min_freq):
        """
        loading vocabulary terms from the output of Galago's 'dump_term_stats' function. For more information, visit
        https://sourceforge.net/p/lemur/wiki/Galago%20Functions/.

        Args:
            file_name: The file address of the Galago's dump output.
            min_freq: Minimum term frequency for each valid term. Other terms are assumed to be 'UNKNOWN'.
        """
        id = 1
        with open(file_name) as f:
            for line in f:
                term, freq, doc_freq = line.split('\t')
                freq = int(freq)
                if freq >= min_freq:
                    self.id_to_term.append(term)
                    self.term_to_id[term] = id
                    id += 1
        print(str(id) + ' terms have been loaded to the dictionary with the minimum frequency of ' + str(min_freq))

    def size(self):
        return len(self.id_to_term)

    def get_emb_list(self, str, delimiter=''):
        words = str.strip().split(delimiter)
        return [(self.term_to_id[w] if w in self.term_to_id else self.term_to_id['UNKNOWN']) for w in words]


# just for test
if __name__ == '__main__':
    dictionary = Dictionary()
    dictionary.load_from_galago_dump('data/robust04-stats.txt', 20)