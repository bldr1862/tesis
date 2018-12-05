import sys
import utils
import params
import numpy as np
import pickle
import os

from sklearn.neighbors import BallTree
#from scipy.spatial.distance import cosine

ficheros = utils.getFiles(params.VLAD_PCA_DESCRIPTORS_VAL_PATH)
arrays = utils.loadNumpyArrays(ficheros)
n_arrays = []
for array in arrays:
    n_arrays.append(array/np.linalg.norm(array))
del arrays

order = dict(zip(range(len(ficheros)), ficheros))

tree = BallTree(n_arrays,leaf_size=40, metric='euclidean')

with open(params.VLAD_VAL_INDEX,'wb') as handle:
    pickle.dump(tree, handle, 2) 

with open(params.VLAD_VAL_ORDER,'wb') as handle:
    pickle.dump(order, handle, 2) 

print('INDEX AND ORDER SAVED')