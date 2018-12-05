import params
import utils
import os
import numpy as np

input_path = params.VLAD_DESCRIPTORS_VAL_PATH
output_path = params.VLAD_PCA_DESCRIPTORS_VAL_PATH

ficheros = utils.getFiles(input_path)
print('Loading Files...')
input_arrays = utils.loadNumpyArrays(ficheros)

print('Loading PCA MODEL...')
pca = utils.importPickle(params.PCA_MODEL_VLAD)

print('Saving Descriptors...')
for array, name in zip(input_arrays, ficheros):
    array_pca = pca.transform(array)
    name_ = output_path + 'PCA_' + os.path.basename(name)
    np.save(name_, array_pca)

print('Done!')