#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 12:10:17 2019

@author: vladislav
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def leeimagen(filename, flag_color):
    img = cv.imread(filename, flag_color)
    
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    plt.imshow(img)
    plt.axis("off")
    plt.show()

def pintaI(filename):
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    normalized = np.zeros(img.shape)
    
    normalized = cv.normalize(img, normalized, 0, 255, cv.NORM_MINMAX)
    plt.imshow(normalized)
    plt.axis('off')
    plt.show()

def pintaMI(vim, titles=None):
    
    columns = len(vim)
    fig, axarr = plt.subplots(1, columns)

    
    for img, i in zip(vim, range(columns)):
        axarr[i].imshow(img)
        if titles != None:
            axarr[i].set_title(titles[i])
        axarr[i].axis('off')
    
    plt.show()

def modifica_color(img, lista_coord, color):
    for coord in lista_coord:
        y, x = coord


def pintaMITitulo(vim, titles):
    pintaMI(vim, titles)
        

leeimagen('img/orapple.jpg', True)
pintaI('img/messi.jpg')

image1 = cv.imread('img/orapple.jpg')
image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)

image2 = cv.imread('img/messi.jpg')
image2 = cv.cvtColor(image2, cv.COLOR_BGR2RGB)

image3 = cv.imread('img/messi-chikito.jpg')
image3 = cv.cvtColor(image3, cv.COLOR_BGR2RGB)

pintaMI([image1, image2, image3], ["orapple", "messi", "messi chiquito"])


