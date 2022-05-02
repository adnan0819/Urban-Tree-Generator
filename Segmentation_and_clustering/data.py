from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import sys
import skimage
np.set_printoptions(threshold=sys.maxsize, precision=5, suppress=True)

# Setting up labels and classes
grass = [128,0,0]
tree = [0,255,0]
Unlabelled = [0,0,0]

trainStepCount=4397 #because 4397 x 16 (batch size) = total dataset training size
valStepCount= 673 

traingencount=0
traingencount=0
valgencount=0
valgencount=0


COLOR_DICT = np.array([ grass, tree, Unlabeled])
class_name = [ 'grass', 'tree', 'Unlabeled']  # You must define by yourself

color = 'grayscale'

num_classes = 3 # include grass, tree and None.
num_of_test_img = 1772

test_img_size = 256 * 256

img_size = (256,256)
###############################################################
import glob

image_path='./data/train/'
image_prefix='npz/' 
image_name_arr = glob.glob(os.path.join(image_path,"%s*.npz"%image_prefix))
image_name_arr.sort()

mask_path='./data/train/'
mask_prefix='mask/'
mask_name_arr = glob.glob(os.path.join(mask_path,"%s*.png"%mask_prefix))
mask_name_arr.sort()

val_path='./data/val/'
val_image_prefix='npz/'
val_image_name_arr = glob.glob(os.path.join(val_path,"%s*.npz"%val_image_prefix))
val_image_name_arr.sort()

val_path='./data/val/'
val_mask_prefix='mask/'
val_mask_name_arr = glob.glob(os.path.join(val_path,"%s*.png"%val_mask_prefix))
val_mask_name_arr.sort()


def traingen(name_arr, batch_size):
    batch=np.zeros((1,256,256,48))
    count = 0
    for index,item in enumerate(name_arr):
        
       

        a1=np.load(item,allow_pickle=True)
        a1=a1['arr_0']
        a1=np.array([a1])
        
        if count==0:
            batch=a1.copy()
        else:
            batch=np.concatenate((batch,a1))

        if count==batch_size:
            
            count=0
           
            yield batch
            batch=np.zeros((1,256,256,48))
            continue
        count+=1
  

def maskgen(name_arr, batch_size):
    batch=np.zeros((1,256,256,3))
    count = 0
    for index,item in enumerate(name_arr):
        
    
        a2=skimage.io.imread(item)
        a2 = a2[..., np.newaxis]
        a2=np.array([a2])
        if count==0:
            batch=a2.copy()
        else:
            batch=np.concatenate((batch,a2))

        if count==batch_size:
            count=0
            yield batch
            batch=np.zeros((1,256,256,3))
            continue
        count+=1
    
    

def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        #TODO Normalize

        #img = img / 255.
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        mask[(mask!=0.)&(mask!=255.)&(mask!=128.)] = 0.
        new_mask = np.zeros(mask.shape + (num_class,))
        new_mask[mask == 128.,   0] = 1
        new_mask[mask == 255.,   1] = 1
        new_mask[mask == 0.,   2] = 1
        mask = new_mask
       
    return (img,mask)

def trainGenerator():
    '''
    training data generator
    '''
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.npz"%image_prefix))
    image_name_arr.sort()
    mask_name_arr = glob.glob(os.path.join(mask_path,"%s*.png"%mask_prefix))
    mask_name_arr.sort()

    image_datagen = traingen(image_name_arr, 16)
    mask_datagen = maskgen(mask_name_arr, 16)
    
    train_generator = zip(image_datagen, mask_datagen)
    while True:
      
      traingencount=0
      
      image_name_arr = glob.glob(os.path.join(image_path,"%s*.npz"%image_prefix))
      image_name_arr.sort()
      mask_name_arr = glob.glob(os.path.join(mask_path,"%s*.png"%mask_prefix))
      mask_name_arr.sort()

      image_datagen = traingen(image_name_arr,16)
      mask_datagen = maskgen(mask_name_arr,16)
      
      train_generator = zip(image_datagen, mask_datagen)
      for (img,mask) in train_generator:
          
          if traingencount==trainStepCount:
            
            break
          
          img,mask = adjustData(img,mask,True,3)
          
          yield (img,mask)
          traingencount+=1

def valGenerator():
    '''
    Validation data generator
    '''
    val_image_name_arr = glob.glob(os.path.join(val_path,"%s*.npz"%val_image_prefix))
    val_image_name_arr.sort()
    val_mask_name_arr = glob.glob(os.path.join(val_path,"%s*.png"%val_mask_prefix))
    val_mask_name_arr.sort()

    val_image_datagen = traingen(val_image_name_arr,16)
    val_mask_datagen = maskgen(val_mask_name_arr,16)
    
    train_generator = zip(val_image_datagen, val_mask_datagen)
    
    while True:
      
      valgencount=0
      
      val_image_name_arr = glob.glob(os.path.join(val_path,"%s*.npz"%val_image_prefix))
      val_image_name_arr.sort()
      val_mask_name_arr = glob.glob(os.path.join(val_path,"%s*.png"%val_mask_prefix))
      val_mask_name_arr.sort()

      val_image_datagen = traingen(val_image_name_arr,16)
      val_mask_datagen = maskgen(val_mask_name_arr,16)
      

      val_generator = zip(val_image_datagen, val_mask_datagen)
      for (img,mask) in val_generator:
          if valgencount==valStepCount:
            
            break
          
          img,mask = adjustData(img,mask,True,3)
         
          yield (img,mask)
          valgencount+=1

def testGenerator(test_path,num_image = num_of_test_img, target_size = img_size, flag_multi_class=True, as_gray=True):
    for i in range(num_image):
        i = i + 1
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        #img = img / 255.
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


### Test data generator (unused)
def testGenerator_for_evaluation(test_path, mask_path, num_image=num_of_test_img, num_class=num_classes ,target_size=(256,256), flag_multi_class = True, as_gray = True):
    for i in range(num_image):
        i = i + 1
        # read images
        img = io.imread(os.path.join(test_path,"%d.png"%i), as_gray = as_gray)
        img = trans.resize(img, target_size)
        img = np.reshape(img,img.shape+(1,)) if (flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        # read masks
        mask = io.imread(os.path.join(mask_path,"%d.png"%i), as_gray = as_gray)
        mask = trans.resize(mask, target_size)
        mask = np.expand_dims(mask,0)
        mask = np.expand_dims(mask,-1)
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        
        ## Set invalid pixels to zeroo ##
        mask[(mask!=0.)&(mask!=255.)&(mask!=128.)] = 0.
        new_mask = np.zeros(mask.shape + (num_class,))
        new_mask[(mask == 128.),   0] = 1 #FIX THIS IN COLAB
        new_mask[(mask == 255.),   1] = 1 #FIX THIS IN COLAB
        new_mask[(mask ==   0.),   2] = 1
        mask = new_mask
        yield (img,mask)



def labelVisualize(num_class,  color_dict, img):
    img_out = np.zeros(img[:,:,0].shape + (3,))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            index_of_class = np.argmax(img[i,j])
            img_out[i,j] = color_dict[index_of_class]
    return img_out

def saveResult(save_path,npyfile,flag_multi_class = True,num_class = num_classes ):
    count = 1
    for i,item in enumerate(npyfile):
        if flag_multi_class:
            img = labelVisualize(num_class,COLOR_DICT,item)
            img = img.astype(np.uint8)
            io.imsave(os.path.join(save_path,"%d.png"%count),img)
        else:
            img=item[:,:,0]
            print(np.max(img),np.min(img))
            img[img>0.5]=1
            img[img<=0.5]=0
            print(np.max(img),np.min(img))
            img = img * 255.
            io.imsave(os.path.join(save_path,"%d.png"%count),img)
        count += 1

