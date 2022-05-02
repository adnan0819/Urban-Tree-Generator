import os
import os.path
from model import *
from data import *
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import History
import tensorflow as tf
import matplotlib.pyplot as plt 
from tensorflow.keras import backend as K
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger
import segmentation_models as sm
from keras.callbacks import ReduceLROnPlateau
from keras.utils.generic_utils import get_custom_objects



batch_size = 16
steps_per_epoch = 4397 
epochs = 300
save_result_folder = './data/results/'
csvfilename = './history.csv'

'''
#model_name= IF PRETRAINED MODEL IS USED, PUT PATH OF PRETRAINED MODEL HERE 
'''

plt_save_name = './plot_train.png'
val_plt_name = './plot_val.png'
img_num = 31278
filenum = 7040

rotation_range = 0.2
width_shift_range = 0.05
height_shift_range = 0.05
horizontal_flip = True
fill_mode = 'nearest'


data_gen_args = dict(
                    featurewise_center=True,
                    featurewise_std_normalization=True,
                    rotation_range=rotation_range,
                    width_shift_range=width_shift_range,
                    height_shift_range=height_shift_range,
                    horizontal_flip=horizontal_flip,
                    fill_mode=fill_mode,
                    cval=0)


def show_train_history(train_history, train, loss, plt_save_name=plt_save_name):
    plt.plot(train_history.history['acc'])
    plt.plot(train_history.history['loss'])
    plt.title('Train hist')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['acc','loss'], loc='upper left')
    plt.savefig(plt_save_name)


# training and validation generators
myGene = trainGenerator()
valGen = valGenerator()

get_custom_objects().update({'f1-score': sm.metrics.FScore(threshold=0.5)})
get_custom_objects().update({'iou_score': sm.metrics.IOUScore(threshold=0.5)})
get_custom_objects().update({'dice_loss_plus_1focal_loss': sm.losses.DiceLoss(class_weights=np.array([1, 1, 1]),per_image=True)  + (1 * sm.losses.CategoricalFocalLoss())})


model = unet()

'''
*******
******* UNCOMMENT THE NEXT LINE IF USING PRE-TRAINED MODEL WEIGHTS
*******
model.load_weights(model_name)
'''

csv_logger = CSVLogger('/logs.csv', append=True)
csv_logger2 = CSVLogger('./logs_backup.csv', append=True)

model_checkpoint = ModelCheckpoint(model_nameEqual, monitor='loss',verbose=1, save_best_only=False)

rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)



training = model.fit_generator(myGene, validation_data=valGen, validation_steps=620,steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[csv_logger , csv_logger2 , model_checkpoint, rlrop])


##### infer
model = load_model(model_name)
testGene = testGenerator(test_img_path)
results = model.predict_generator(testGene, img_num, verbose=1)
#####


if not os.path.exists(save_result_folder):
    os.makedirs(save_result_folder)

saveResult( save_result_folder, results)


if (os.path.isfile(csvfilename)!=True):
    csv_create(csvfilename, filenum, batch_size, steps_per_epoch, epochs, learning_rate, learning_decay_rate, rotation_range)
else:
    csv_append(csvfilename, filenum, batch_size, steps_per_epoch, epochs, learning_rate, learning_decay_rate, rotation_range)


K.clear_session()
    