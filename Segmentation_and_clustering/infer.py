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

#arg = command_arguments()


batch_size = 16
steps_per_epoch = 4397 # the 10 is just for testing - actually make it 4397
epochs = 300
save_result_folder = './data/results/'
csvfilename = './history.csv'

'''
model_name= PUT PATH OF PRETRAINED MODEL HERE IF PRETRAINED MODEL IS USED
'''

data_gen_args = dict(
                    featurewise_center=True,
                    featurewise_std_normalization=True,
                    rotation_range=rotation_range,
                    width_shift_range=width_shift_range,
                    height_shift_range=height_shift_range,
                    #shear_range=0.05,
                    #zoom_range=zoom_range,
                    horizontal_flip=horizontal_flip,
                    fill_mode=fill_mode,
                    cval=0)


#draw the training process of every epoch
def show_train_history(train_history, train, loss, plt_save_name=plt_save_name):
    plt.plot(train_history.history['acc'])
    plt.plot(train_history.history['loss'])
    plt.title('Train hist')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['acc','loss'], loc='upper left')
    plt.savefig(plt_save_name)


get_custom_objects().update({'f1-score': sm.metrics.FScore(threshold=0.5)})
get_custom_objects().update({'iou_score': sm.metrics.IOUScore(threshold=0.5)})
get_custom_objects().update({'dice_loss_plus_1focal_loss': sm.losses.DiceLoss(class_weights=np.array([1, 1, 1]),per_image=True)  + (1 * sm.losses.CategoricalFocalLoss())})

model.load_weights(model_name)



##### inference
model = load_model(model_name)
testGene = testGenerator(test_img_path)
#testGene_for_eval = testGenerator_for_evaluation(test_img_path)
results = model.predict_generator(testGene, img_num, verbose=1)
#loss, acc = model.evaluate_generator(testGene_for_eval, steps=img_num, verbose=1)
#print("test loss:",loss,"  test accuracy:", acc)
#####


##### draw inference results
if not os.path.exists(save_result_folder):
    os.makedirs(save_result_folder)

saveResult( save_result_folder, results)
#####




K.clear_session()
    