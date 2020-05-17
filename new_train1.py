# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 02:38:07 2020

@author: Anirudh
"""

from gr_pr import TrafficSignNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint,History
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from skimage import transform
from skimage import exposure
from skimage import io
import numpy as np
import keras
import random
import cv2
import os
from PIL import Image
import tensorflow as tf
'''
config=tf.compat.v1.ConfigProto(device_count={'GPU':1 , 'CPU':4})
sess=tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
'''

def load_split(basePath, csvPath):
# initialize the list of data and labels
    data = []
    labels = []

# load the contents of the CSV file, remove the first line (since
# it contains the CSV header), and shuffle the rows (otherwise
# all examples of a particular class will be in sequential order)
    rows = open(csvPath).read().strip().split("\n")[0:]
    random.shuffle(rows)

# loop over the rows of the CSV file
    for (i, row) in enumerate(rows):
# check to see if we should show a status update
        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {} total images".format(i))

        (imagepath,label) = row.strip().split(",")[:]
        imagePath=os.path.sep.join([basePath, imagepath])
        #print(imagepath)
# derive the full path to the image file and load it
        

        image = cv2.imread(imagePath)
        image = cv2.resize(image,(64, 64),fx=0.5,fy=0.5)
#image = exposure.equalize_adapthist(image, clip_limit=0.1)

# update the list of data and labels, respectively
        data.append(image)
        labels.append(int(label))

# convert the data and labels to NumPy arrays
    data = np.array(data)
    labels = np.array(labels)

# return a tuple of the data and labels
    return (data, labels)

# construct the argument parser and parse the arguments
# initialize the number of epochs to train for, base learning rate,
# and batch size
NUM_EPOCHS = 100
INIT_LR = 1e-3
BS = 32

# load the label names
labelNames = open("signnames.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]

# derive the path to the training and testing CSV files
trainPath ="Train2.csv"
testPath ="test2.csv"

# load the training and testing data
print("[INFO] loading training and testing data...")
(trainX, trainY) = load_split(r"C:\Users\Anirudh\Desktop\daksh_extra-final",trainPath)
(testX, testY) = load_split(r"C:\Users\Anirudh\Desktop\daksh_extra-final",testPath)

# scale data to the range of [0, 1]

trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# one-hot encode the training and testing labels

numLabels = 8#len(np.unique(trainY))

trainY = to_categorical(trainY)
testY = to_categorical(testY)


# account for skew in the labeled data
'''
classTotals = trainY.sum(axis=0)
classWeight = classTotals.max() / classTotals
'''
# construct the image generator for data augmentation


aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.15,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=False,
	vertical_flip=True,
	fill_mode="nearest")

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / (NUM_EPOCHS * 0.5))
#opt=SGD(lr=INIT_LR, decay=INIT_LR / (NUM_EPOCHS * 0.5),momentum=0.9, nesterov=True)
model = TrafficSignNet.build(width=64, height=64, depth=3,classes=numLabels)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])



#trainX=np.expand_dims(trainX,axis=3)
#trainY=np.expand_dims(trainY,axis=2)

#testX=np.expand_dims(testX,axis=3)
#trainX=np.expand_dims(trainX,axis=3)



model_checkpoint = ModelCheckpoint('Daksh1.h5', monitor='loss',verbose=1, save_best_only=True)
history=model.fit(aug.flow(trainX,trainY,batch_size=BS),epochs=NUM_EPOCHS,callbacks=[model_checkpoint],verbose=1)
#callbacks=[model_checkpoint]
# evaluate the network

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=labelNames))

# save the network to disk
#model.save(r"C:\Users\Indrajithu\Downloads\Grievance_model\Grievance_model\model.json")
from keras.models import model_from_json
from keras.models import save_model,load_model
model_json = model.to_json()

with open("model.json", "w") as json_file:
    json_file.write(model_json)

# save weights to HDF5
model.save_weights("Daksh.h5")
print("Model saved")



