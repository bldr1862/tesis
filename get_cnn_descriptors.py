import utils
import params

train_ficheros = utils.getFiles(params.TRAIN_IMAGES_PATH)
val_ficheros = utils.getFiles(params.VAL_IMAGES_PATH)
test_ficheros = utils.getFiles(params.TEST_IMAGES_PATH)

utils.getCnnDescriptors(train_ficheros, 
                        params.CNN_DESCRIPTORS_TRAIN_PATH,
                        'vgg16')

utils.getCnnDescriptors(val_ficheros, 
                        params.CNN_DESCRIPTORS_VAL_PATH,
                        'vgg16')

utils.getCnnDescriptors(test_ficheros, 
                        params.CNN_DESCRIPTORS_TEST_PATH,
                        'vgg16')