# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 06:43:19 2020

@author: Anirudh
"""



# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 23:12:53 2020

@author: Anirudh
"""


import glob
import csv
import os




list=[]        


for img in glob.glob(r"C:\Users\Anirudh\Desktop\daksh_extra - final\Test\*.*") :
    
    l=[]
    l.append(img)
    list.append(l)
    
def writeCsvFile(fname, data, *args, **kwargs):
    """
    @param fname: string, name of file to write
    @param data: list of list of items

    Write data to file
    """
    mycsv = csv.writer(open(fname, 'w'), *args, **kwargs)
    for row in data:
        mycsv.writerow(row)
        
writeCsvFile(r'C:\Users\Anirudh\Desktop\daksh_extra\test2.csv', list)