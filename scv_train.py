# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 23:35:17 2020

@author: Indrajithu
"""

import cv2
import numpy
import glob
import csv

ll=[]
class_id=0
for img in glob.glob(r"E:\Daksh\Train\overflowing sewers\*.*"):
    l=[]
    l.append(img)
    l.append(class_id)
    ll.append(l)

class_id=1
for img in glob.glob(r"E:\Daksh\Train\patchy roads\*.*"):
    l=[]
    l.append(img)
    l.append(class_id)
    ll.append(l)
    
class_id=2
for img in glob.glob(r"E:\Daksh\Train\overflowing garbage\*.*"):
    l=[]
    l.append(img)
    l.append(class_id)
    ll.append(l)    
    
class_id=3 
for img in glob.glob(r"E:\Daksh\Train\open manholes\*.*"):
    l=[]
    l.append(img)
    l.append(class_id)
    ll.append(l)   
    
def writeCsvFile(fname, data, *args, **kwargs):
    """
    @param fname: string, name of file to write
    @param data: list of list of items

    Write data to file
    """
    mycsv = csv.writer(open(fname, 'w'), *args, **kwargs)
    for row in data:
        mycsv.writerow(row)


writeCsvFile(r'E:\Daksh\Train.csv', ll)    