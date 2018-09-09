import sys
import utils
import params
import numpy as np
import pickle
import os

from sklearn.neighbors import BallTree
from scipy.spatial.distance import cosine

ficheros = utils.getFiles(params.CNN_PCA_DESCRIPTORS_VAL_PATH)
arrays = utils.loadNumpyArrays(ficheros)

tree = BallTree(arrays,leaf_size=40, metric=cosine)

with open(params.CNN_VAL_INDEX,'wb') as handle:
    pickle.dump(tree, handle, 2) 

print('INDEX  SAVED')