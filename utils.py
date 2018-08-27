import os
import numpy as np
from tqdm import tqdm

def getFiles(PATH):
    _files = []
    for root, dirs, files in os.walk(PATH):
        for fichero in files:
                _files.append(os.path.join(PATH,fichero))
    return _files

def loadNumpyArrays(ficheros):
    arrays = []
    for fichero in ficheros:
        arrays.append(np.load(fichero).reshape(-1))

    return np.asarray(arrays)


def DeleteFiles(FOLDER_PATH):
    for root, dirs, files in os.walk(FOLDER_PATH):
        for fichero in files:
            os.unlink(os.path.join(FOLDER_PATH,fichero))

def getDescriptors(ficheros, output_folder ,cnn_network = 'vgg16'):
    descriptors_dict = {}
    if cnn_network == 'vgg16':
        from keras.applications import vgg16
        from keras.models import Model
        from keras.applications.vgg16 import preprocess_input
        from keras.preprocessing import image

        model = vgg16.VGG16(weights='imagenet', include_top=True)
        model = Model(inputs=model.input, outputs=model.get_layer('fc1').output)

    DeleteFiles(output_folder)

    for fichero in tqdm(ficheros):
        img = image.load_img(fichero, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        descriptor = model.predict(x)
        id = os.path.basename(fichero).split('.')[0]+'.npy'
        name = os.path.join(output_folder+'\\'+id)
        np.save(name,descriptor)
    return 'SUCEED'

def getDescriptor(fichero, cnn_network = 'vgg16'):
    if cnn_network == 'vgg16':
        from keras.applications import vgg16
        from keras.models import Model
        from keras.applications.vgg16 import preprocess_input
        from keras.preprocessing import image

        model = vgg16.VGG16(weights='imagenet', include_top=True)
        model = Model(inputs=model.input, outputs=model.get_layer('fc1').output)

    img = image.load_img(fichero, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    descriptor = model.predict(x)
    return descriptor

