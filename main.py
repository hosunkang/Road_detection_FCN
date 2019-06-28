import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import tensorflow as tf
import keras, sys, time, warnings
import pandas as pd 
import argparse
from keras.backend.tensorflow_backend import set_session
from sklearn.utils import shuffle
from keras import optimizers
from model import fcn8
from keras.models import load_model
from datetime import datetime

############ Check environment
warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.visible_device_list = "0" 
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))   
print("python {}".format(sys.version))
print("keras version {}".format(keras.__version__)); del keras
print("tensorflow version {}".format(tf.__version__))

############ Train Setup
dir_data = "data_road/training"
dir_seg = dir_data + "/gt_image/"
dir_img = dir_data + "/image/"
n_classes = 3
epoch_size = 300
input_height , input_width = 224 , 672
output_height , output_width = 224 , 672

############ Definitions
def getImageArr( path , width , height ):
    img = cv2.imread(path, 1)
    img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
    return img

def getOrigin( path , width , height ):
    img = cv2.imread(path, 1)
    img = np.float32(cv2.resize(img, ( width , height )))
    return img


def getSegmentationArr( path , nClasses ,  width , height  ):

    seg_labels = np.zeros((  height , width  , nClasses ))
    img = cv2.imread(path, 1)
    img = cv2.resize(img, ( width , height ))
    img_normal = cv2.normalize(img, None, 0,2, cv2.NORM_MINMAX)
    img_normal = img_normal[:, : , 0]

    for c in range(nClasses):
        seg_labels[: , : , c ] = (img_normal == c ).astype(int)
    ##seg_labels = np.reshape(seg_labels, ( width*height,nClasses  ))
    return seg_labels

def IoU(Yi,y_predi):
    ## mean Intersection over Union
    ## Mean IoU = TP/(FN + TP + FP)

    IoUs = []
    Nclass = int(np.max(Yi)) + 1
    for c in range(Nclass):
        TP = np.sum( (Yi == c)&(y_predi==c) )
        FP = np.sum( (Yi != c)&(y_predi==c) )
        FN = np.sum( (Yi == c)&(y_predi != c)) 
        IoU = TP/float(TP + FP + FN)
        print("class {:02.0f}: #TP={:6.0f}, #FP={:6.0f}, #FN={:5.0f}, IoU={:4.3f}".format(c,TP,FP,FN,IoU))
        IoUs.append(IoU)
    mIoU = np.mean(IoUs)
    print("_________________")
    print("Mean IoU: {:4.3f}".format(mIoU))

def train(model, n_classes):
    images = os.listdir(dir_img)
    images.sort()
    segmentations  = os.listdir(dir_seg)
    segmentations.sort()
        
    X = []
    Y = []
    for im , seg in zip(images,segmentations) :
        X.append( getImageArr(dir_img + im , input_width , input_height )  )
        Y.append( getSegmentationArr( dir_seg + seg , n_classes , output_width , output_height )  )

    X, Y = np.array(X) , np.array(Y)
    Full_dataset = 'Full  dataset : {}, {}'.format(X.shape, Y.shape)
    print(Full_dataset)

    train_rate = 0.85
    index_train = np.random.choice(X.shape[0],int(X.shape[0]*train_rate),replace=False)
    index_valid = list(set(range(X.shape[0])) - set(index_train))
                                
    X, Y = shuffle(X,Y)
    X_train, y_train = X[index_train],Y[index_train]
    X_valid, y_valid = X[index_valid],Y[index_valid]
    Train_dataset = 'Train dataset : {}, {}'.format(X_train.shape, y_train.shape)
    print(Train_dataset)
    Valid_dataset = 'Valid dataset : {}, {}'.format(X_valid.shape, y_valid.shape)
    print(Valid_dataset)

    sgd = optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(X_train,y_train, validation_data=(X_valid,y_valid), batch_size=32,epochs=epoch_size,verbose=2)
    model.save('road_detection.h5')
    print("=======Saved model to 'Road_detection_FCN' directory=======")

def predict():
    dir_test_image = "data_road/testing/image_2/"
    test_images = os.listdir(dir_test_image)
    test_images.sort()
    
    TEST_X = []
    origin_x = []
    for im in test_images:
        TEST_X.append( getImageArr(dir_test_image + im , input_width , input_height )  )
        origin_x.append( getOrigin(dir_test_image + im , input_width , input_height )  )
    x_test = np.array(TEST_X)
    Test_dataset = 'Test dataset : {}'.format(x_test.shape)
    print(Test_dataset)

    model = load_model("road_detection.h5")
    y_test = model.predict(x_test)
    y_test_arg = np.argmax(y_test, axis=3)
    print(y_test_arg.shape)

    run_dir = 'run/'
    now = datetime.now()
    run_folder_name = 'time_{}.{:02d}.{:02d}_{:02d}.{:02d}/'.format(now.year,
                                                                    now.month,
                                                                    now.day,
                                                                    now.hour,
                                                                    now.minute)
    run_dir = run_dir + run_folder_name
    os.mkdir(run_dir)

    for image ,seg, name in zip(origin_x, y_test_arg, test_images):
        seg_img = give_color_to_seg_img(image, seg)
        cv2.imwrite(run_dir+name, seg_img)
    
    print("=======Saved all Segmentation Images=======")

def give_color_to_seg_img(image, seg):
    seg = 255 * seg # Now scale by 255
    seg = seg.astype(np.uint8)
    seg = cv2.cvtColor(seg,cv2.COLOR_GRAY2BGR)
    blank_image = np.zeros((input_height,input_width,3), np.uint8)
    blank_image[:,:] = (0,255,0)
    seg = cv2.bitwise_and(seg, blank_image)
    seg_img = seg + image

    return(seg_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Semi-Supervised Learning Road detection')
    parser.add_argument('command', metavar="<command>", help="'train' or 'expect'")
    args = parser.parse_args()

    if args.command == "train":
        # Load model
        model = fcn8.FCN8(nClasses = n_classes, input_height = input_height, input_width  = input_width)
        model.summary()
        train(model, n_classes)
    elif args.command == "predict":
        predict()