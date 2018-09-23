import utils
import params
import random
import os
import numpy as np
import time

from scipy.sparse import vstack

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.objectives import MSE, MAE
from keras.callbacks import EarlyStopping

TRAIN_SIZE = 0.8
BATCH_SIZE = 100000
EPOCHS = 1
print('Reading Files...')
text = utils.getFiles(params.TFIDF_DESCRIPTORS_TRAIN_PATH)
images = utils.getFiles(params.CNN_PCA_DESCRIPTORS_TRAIN_PATH)
INPUT_SHAPE = 369537
OUTPUT_SHAPE = 128

model = Sequential()
model.add(Dense(input_dim=INPUT_SHAPE, output_dim=150))
model.add(Activation("relu"))
model.add(Dropout(0.50))
model.add(Dense(output_dim=130))
model.add(Activation("relu"))
model.add(Dropout(0.50))
model.add(Dense(output_dim=128))
model.compile("nadam", "mae")

print('Training...')
i = 0
for epoch in range(EPOCHS):
    random.seed(42)
    random.shuffle(text)
    print('    EPOCH:', epoch)
    for text_descriptors, img_descriptors in utils.getBatch(text, images, BATCH_SIZE):

        print(vstack(text_descriptors).shape)
        """    tmp = list(zip(text_descriptors, img_descriptors))
            random.seed(42)
            random.shuffle(tmp)
            text_descriptors, img_descriptors = zip(*tmp)"""
        t0 = time.time()
        mlp.partial_fit(vstack(text_descriptors), img_descriptors)
        print('        Partial fit {} took: {} min, Score {}'.format(i,round((time.time()-t0)/60,2),mlp.loss_))

        """    
            mlp.fit(vstack(text_descriptors), img_descriptors)
            """

        """    i = i+1
            if i == 2:
                break"""

print('Saving model...')
utils.savePickle(mlp, params.CNN_MLP)
print('Done!')