import json
import numpy as np

import params
import utils

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

captions = utils.readCaptions(params.TRAIN_CAPTIONS_PATH)
stemmed_dict = utils.stemming(captions)

#flatten corpus
corpus =[]
for x in stemmed_dict.keys():
    for y in stemmed_dict.get(x):
        corpus.append(y)

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))
transformer = TfidfTransformer(smooth_idf=False)


counts = bigram_vectorizer.fit_transform(corpus)
transformer.fit(counts)

utils.savePickle(object=bigram_vectorizer,PATH=params.BIGRAM_MODEL)
utils.savePickle(object=transformer, PATH=params.TF_IDF_MODEL)