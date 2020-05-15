# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:52:10 2020
@author: Anirudh
"""
from gr_pr import TrafficSignNet
# Modify 'test1.jpg' and 'test2.jpg' to the images you want to predict on
from keras.models import load_model
import cv2
from keras.preprocessing import image
import numpy as np
#from pyimagesearch.gr_pr import TrafficSignNet

model = TrafficSignNet.build(width=64, height=64, depth= 3,classes=7)

# dimensions of our images
img_width, img_height = 64,64

# load the model we saved
model.load_weights('Daksh.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# predicting images
img = cv2.imread(r"C:\Users\Anirudh\Desktop\daksh_extra - final\patch1.jpg")
img=cv2.resize(img,(64,64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)


#images = np.vstack([x])
classes = model.predict_classes(x)
c=''
if(classes[0]==1):
    c='PATCHY ROAD'
elif(classes[0]==2):
    c='OVERFLOWING GARBAGE BIN'
elif(classes[0]==3):
    c='OPEN MANHOLE'

elif(classes[0]==4):
    c='RECONSTRUNCTED MANHOLE'
    
elif(classes[0]==5):
    c='RECONSTRUCTED ROAD'
    
elif(classes[0]==6):
    c='CLEANED GARBAGE BIN'

else:
    c='OTHERS'
    

#dd=model.predict(x)
print(c)

