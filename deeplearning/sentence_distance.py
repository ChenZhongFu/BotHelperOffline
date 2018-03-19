# -*- coding: utf-8 -*-
import sys
import numpy as np
import pyemd
from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    double, uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
    ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray
import cPickle as pickle

# import logging
# logger = logging.getLogger(__name__)

reload(sys)
sys.setdefaultencoding('utf-8')

def two_sentence_dis(sentence1, sentence2):
    embedding = pickle.load(open('/home/wxr/blazer/BotHelperOffline/deeplearning/model/word2vec/embedding.pkl'))
    word_index = pickle.load(open('/home/wxr/blazer/BotHelperOffline/deeplearning/model/word2vec/word_index.pkl'))

    len_sentence1 = len(sentence1)
    len_sentence2 = len(sentence2)

    # Remove out-of-vocabulary words.
    sentence1 = [word_index.get(token) for token in sentence1 if word_index.has_key(token)]
    sentence2 = [word_index.get(token) for token in sentence2 if word_index.has_key(token)]

    diff1 = len_sentence1 - len(sentence1)
    diff2 = len_sentence2 - len(sentence2)
    if diff1 > 0 or diff2 > 0:
        print ('Removed %d and %d OOV words from document 1 and 2 (respectively).',
               diff1, diff2)

    if len(sentence1) == 0 or len(sentence2) == 0:
        print ('At least one of the documents had no words that were'
               'in the vocabulary. Aborting (returning inf).')
        return float('inf')

    dictionary_temp = list(set(sentence1 + sentence2))
    dictionary = dict(enumerate(dictionary_temp))
    vocab_len = len(dictionary)

    sen_set1 = set(sentence1)
    sen_set2 = set(sentence2)

    distance_matrix = np.zeros((vocab_len, vocab_len), dtype=double)
    for i, t1 in dictionary.items():
        for j, t2 in dictionary.items():
            if not t1 in sen_set1 or not t2 in sen_set2:
                continue
            # 计算距离
            distance_matrix[i, j] = sqrt(np_sum((embedding[t1] - embedding[t2]) ** 2))

    if np_sum(distance_matrix) == 0.0:
        # `emd` gets stuck if the distance matrix contains only zeros.
        print ('The distance matrix is all zeros. Aborting (returning inf).')
        return float('inf')

    def doc2bow(document, dictionary):
        freq_dic = dict()
        for i in document:
            if freq_dic.has_key(i):
                freq_dic[i] = freq_dic[i] + 1
            else:
                freq_dic[i] = 1

        return_freq = dict()
        for i in range(len(document)):
            if return_freq.has_key(i):
                for key in range(len(dictionary)):
                    if dictionary[key] == document[i]:
                        return_freq[key] = freq_dic[document[i]]
            else:
                for key in range(len(dictionary)):
                    if dictionary[key] == document[i]:
                        return_freq[key] = freq_dic[document[i]]
        return return_freq

    def nbow(document):
        d = zeros(vocab_len, dtype=double)
        nbow = doc2bow(document, dictionary)  # Word frequencies.
        doc_len = len(document)
        for (idx, freq) in nbow.items():
        #for idx, freq in nbow:
            d[idx] = float(freq) / float(doc_len)  # Normalized word frequencies.
        return d

    # Compute nBOW representation of documents.
    d1 = nbow(sentence1)
    d2 = nbow(sentence2)

    # Compute WMD.
    #print pyemd.emd(d1, d2, distance_matrix)
    return pyemd.emd(d1, d2, distance_matrix)
