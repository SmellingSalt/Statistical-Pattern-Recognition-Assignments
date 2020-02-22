#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 19:44:00 2020

@author: sa1
"""
import os
from glob import glob
from pandas import read_csv
import numpy as np
path_to_data=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'Data'))
csv_data=glob(os.path.join(path_to_data, "*.csv"))
iter=0
data=[]
for files in csv_data:
    temp = read_csv(csv_data[iter])
    temp=temp.values
    data.append(temp)
    iter=iter+10
