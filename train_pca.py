import numpy as np
import params
import utils
import pickle
from sklearn.decomposition import PCA

input_path = params.VLAD_DESCRIPTORS_TRAIN_PATH
#output_path = params.CNN_PCA_DESCRIPTORS_TRAIN_PATH

ficheros = utils.getFiles(input_path)
print('Loading Files...')
input_arrays = utils.loadNumpyArrays(ficheros)

print('Training model...')
pca = PCA(n_components = params.DIM)
pca.fit(input_arrays)

print('Saving Model...')

with open(params.PCA_MODEL_VLAD,'wb') as handle:
    pickle.dump(pca,handle,protocol=2)

print('Done!')

#for array, name in zip(input_arrays,ficheros):
#    array_pca = pca.transform(array)
#    np.load(name, array_pca)