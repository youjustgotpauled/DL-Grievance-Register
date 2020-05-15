# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 22:45:28 2020

@author: Anirudh
"""

import cv2
import numpy
import glob
import csv

ll=[]
count=0
class_id=1
for img in glob.glob(r"C:\Users\Anirudh\Desktop\daksh_extra - final\Train\patchy roads\*.*") :
    
    l=[]
    l.append(img)
    l.append(class_id)
    ll.append(l)

class_id=2
for img in glob.glob(r"C:\Users\Anirudh\Desktop\daksh_extra - final\Train\overflowing garbage\*.*") :
    
    l=[]
    l.append(img)
    l.append(class_id)
    ll.append(l)
    
class_id=3
for img in glob.glob(r"C:\Users\Anirudh\Desktop\daksh_extra - final\Train\open manholes\*.*") :
    
    l=[]
    l.append(img)
    l.append(class_id)
    ll.append(l)

class_id=4
for img in glob.glob(r"C:\Users\Anirudh\Desktop\daksh_extra - final\Train\proper_manholes\*.*") :
    
    l=[]
    l.append(img)
    l.append(class_id)
    ll.append(l)
    
class_id=5
for img in glob.glob(r"C:\Users\Anirudh\Desktop\daksh_extra - final\Train\Proper roads\*.*") :
    
    l=[]
    l.append(img)
    l.append(class_id)
    ll.append(l)
    
class_id=6
for img in glob.glob(r"C:\Users\Anirudh\Desktop\daksh_extra - final\Train\proper_dustbin\*.*") :
    
    l=[]
    l.append(img)
    l.append(class_id)
    ll.append(l)
    
class_id=7
for img in glob.glob(r"C:\Users\Anirudh\Desktop\daksh_extra - final\Train\Others\*.*") :
    
    l=[]
    l.append(img)
    l.append(class_id)
    ll.append(l)
    
   # imcrop=n[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]


print(count)
def writeCsvFile(fname, data, *args, **kwargs):
    """
    @param fname: string, name of file to write
    @param data: list of list of items

    Write data to file
    """
    mycsv = csv.writer(open(fname, 'w'), *args, **kwargs)
    for row in data:
        mycsv.writerow(row)


writeCsvFile(r'C:\Users\Anirudh\Desktop\daksh_extra - final\Train2.csv', ll)