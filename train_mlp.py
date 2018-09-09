import utils
import params
import random
import os
import numpy as np
import time

from scipy.sparse import vstack
from sklearn.neural_network import MLPRegressor

TRAIN_SIZE = 0.8
BATCH_SIZE = 100
print('Reading Files...')
text = utils.getFiles(params.TFIDF_DESCRIPTORS_TRAIN_PATH)
images = utils.getFiles(params.CNN_PCA_DESCRIPTORS_TRAIN_PATH)

random.seed(42)
random.shuffle(text)

mlp = MLPRegressor(hidden_layer_sizes=(150,130),
                    activation='relu',
                    solver='sgd',
                    batch_size = 1000,
                    learning_rate = 'adaptive'
                    )

print('Training...')
i = 0
for text_descriptors, img_descriptors in utils.getBatch(text, images, BATCH_SIZE):

    """    tmp = list(zip(text_descriptors, img_descriptors))
        random.seed(42)
        random.shuffle(tmp)
        text_descriptors, img_descriptors = zip(*tmp)"""

    t0 = time.time()
    mlp.partial_fit(vstack(text_descriptors), img_descriptors)
    print('    Partial fit {} took: {} min, Score {}'.format(i,round((time.time()-t0)/60,2),mlp.loss_))
    i = i+1
    if i == 2:
        break

print('Saving model...')
utils.savePickle(mlp, params.CNN_MLP)
print('Done!')