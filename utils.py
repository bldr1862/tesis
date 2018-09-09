import os
import numpy as np
import pickle
import json
import params
from tqdm import tqdm
from collections import defaultdict

from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def loadArray(path):
    return np.load(path)

def buildImgName(basename, folder):
    l = len(basename)
    z = '0'*(16-l)
    return os.path.join(folder,'PCA_COCO_train2014_'+z+basename)

def getBatch(ficheros_text, ficheros_img, batch_size):

    for text_batch in chunks(ficheros_text,batch_size):
        text_descriptors = []
        img_descriptors = []
        
        for file in text_batch:
            file_name = os.path.basename(file)
            descriptor_text = loadArray(file).tolist()
            for element in descriptor_text:
                text_descriptors.append(element[0])
                img_descriptors.append(loadArray(buildImgName(file_name, params.CNN_PCA_DESCRIPTORS_TRAIN_PATH))[0])
            
        yield np.asarray(text_descriptors), np.asarray(img_descriptors)

def stemming(captions):
    stemmed_dict = defaultdict(list)
    stop_words = set (stopwords.words( 'english' ))
    ps = SnowballStemmer("english", ignore_stopwords=False)
    unigram_vectorizer = CountVectorizer()
    analyze = unigram_vectorizer.build_analyzer()

    for caption in captions:
        name = caption.get('image_id')
        des = caption.get('caption').lower()

        #remove stopwords
        des_stemmed = []
        for word in analyze(des):
            if word not in stop_words:
                des_stemmed.append(ps.stem(word))
        des_stemmed = ' '.join(des_stemmed)
        
        stemmed_dict[name].append(des_stemmed)
    return stemmed_dict

def readCaptions(PATH):
    with open(PATH,'rb') as handle:
        return json.load(handle)['annotations']

def savePickle(object, PATH):
    with open(PATH,'wb') as handle:
        pickle.dump(object,handle,2)

def importPickle(PATH):
    with open(PATH,'rb') as handle:
        object = pickle.load(handle)
    return object

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

def getCnnDescriptors(ficheros, output_folder ,cnn_network = 'vgg16'):
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

