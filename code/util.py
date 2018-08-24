"""
Some useful methods.

Authors: Hamed Zamani (zamani@cs.umass.edu)
"""

import math


def load_word_embeddings(file_name, dim, normalize=True):
    term_ids = {}
    we_matrix = []  # a term_num * dim matrix for word embeddings
    term_ids['NULL'] = 0
    term_by_id = ['NULL']
    we_matrix.append([0] * dim)
    term_num = 1
    with open(file_name) as FileObj:
        for line in FileObj:
            line = line.split()
            term_ids[line[0].strip()] = term_num
            term_by_id.append(line[0].strip())
            norm = 1
            if normalize is True:
                norm = math.sqrt(sum(float(i) * float(i) for i in line[1: dim + 1]))
            we_matrix.append([float(i) / norm for i in line[1: dim + 1]])
            term_num += 1
    return term_ids, term_by_id, we_matrix


def write_trec_format(result_file, qid, docs, scores, model='DocRep'):
    sorted_index = [j[0] for j in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
    rank = 1
    for i in sorted_index:
        result_file.write(qid + ' Q0 ' + docs[i] + ' ' + str(rank) + ' ' + str(scores[i]) + ' ' + model + '\n')
        rank += 1


def read_trec_format(file_name, judgments=None):
    result = dict()
    with open(file_name) as file:
        for line in file:
            q, _, d, rank, score, _ = line.split()
            if q not in result:
                result[q] = {'did': [], 'label': []}
            result[q]['did'].append(d)

            if judgments is not None:
                if q in judgments:
                    if d in judgments[q]:
                        result[q]['label'].append(judgments[q][d])
                    else:
                        result[q]['label'].append(0)
                else:
                    result[q]['label'].append(0)
                # else:
                #     print('query ' + q + ' is not in judgments!')
    return result


def read_trec_judgment_file(file_name):
    result = dict()
    with open(file_name) as file:
        for line in file:
            q, _, d, label = line.split()
            if q not in result:
                result[q] = dict()
            result[q][d] = int(label)
    return result


def read_query_file(file_name, dictionary):
    result = dict()
    with open(file_name) as file:
        for line in file:
            qid, q_text = line.split()
            if qid in result:
                raise Exception('duplicate query id')
            result[qid] = dictionary.get_emb_list(q_text, delimiter="_")
    return result
