#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:15:27 2020

@author: sa1
"""
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def hyper_plane(p1, p2):
    """https://stackoverflow.com/questions/36470343/how-to-draw-a-line-with-matplotlib"""
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l

import numpy as np
x = np.linspace(0,10)
y = x**2

p1 = [1,20]
p2 = [6,70]

plt.plot(x, y)
newline(p1,p2)
plt.show()