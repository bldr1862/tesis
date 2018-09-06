import json
import numpy as np
from collections import defaultdict

import params
import utils

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from tqdm import tqdm

DATA_PATH = params.VAL_CAPTIONS_PATH
OUTPUT_PATH = params.TFIDF_DESCRIPTORS_VAL_PATH

captions = utils.readCaptions(DATA_PATH)
stemmed_dict = utils.stemming(captions)

bigram_vectorizer = utils.importPickle(params.BIGRAM_MODEL)
transformer = utils.importPickle(params.TF_IDF_MODEL)

for x in tqdm(stemmed_dict.keys()):
    bigram = bigram_vectorizer.transform(stemmed_dict.get(x))
    tf_idf = transformer.transform(bigram)
    name = OUTPUT_PATH+str(x)+'.npy'
    np.save(name, tf_idf)