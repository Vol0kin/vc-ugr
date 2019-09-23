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
    """
    Funcion que lee una imagen que se le pasa como argumento, la pasa a RGB
    y la muestra por pantalla.
    
    Args:
        filename: Nombre de la imagen
        :flag_color: Booleano o valor entero 0 o 1 que indica si se esta usando
                     color o no
    """

    # Cargar imagen y pasarla de BGR a RGB
    img = cv.imread(filename, flag_color)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # Dibujar imagen desactivando los ejes
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def pintaI(filename):
    """
    Funcion que lee una imagen y la normaliza, haciendo que los valores RGB
    esten en el rango [0, 255]

    Args:
        filename: Nombre de la imagen
    """

    # Cargar imagen y pasarla de BGR a RGB
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # Crear una matriz que sera la imagen normalizada inicializada a 0
    normalized = np.zeros(img.shape)
    
    # Normalizar la imagen
    normalized = cv.normalize(img, normalized, 0, 255, cv.NORM_MINMAX)

    # Dibujar la imagen desactivando los ejes
    plt.imshow(normalized)
    plt.axis('off')
    plt.show()


def pintaMI(vim, titles=None):
    """
    Funcion que pinta multiples imagenes en una misma ventana. Adicionalmente
    se pueden especificar que titulos tendran las imagenes.

    Args:
        vim: Lista con las imagenes cargas
        titles: Titulos que tendran las imagenes (default None)
    """

    # Obtener el numero de imagenes
    columns = len(vim)

    # Crear columns sublots (un subplot por cada imagen)
    # El formato sera 1 fula con columns columnas
    _, axarr = plt.subplots(1, columns)

    # Pintar cada imagen
    for i in range(columns):
        # Obtener siguiente imagen
        axarr[i].imshow(vim[i])
        
        # Determinar si hay que poner o no titulo
        if titles != None:
            axarr[i].set_title(titles[i])
        
        axarr[i].axis('off')
    
    # Mostar imagenes
    plt.show()


def modifica_color(img, points, color):
    """
    Funcion que modifica los colores de una imagen segun las posiciones dadas
    al color especificado.

    Args:
        img: Imagen cargada
        points: Lista de tuplas (x, y) que contiene las posiciones a modificar
        color: Lista que contiene el color en formato RGB
    Return:
        Devuelve una nueva imagen con los colores modificados
    """

    # Copiar la imagen original
    new_img = np.copy(img)

    # Crear un array de numpy con el color
    color = np.array(color)
    
    # Para cada coordenada, establecer el nuevo color
    for coord in points:
        # Las coordenadas estan invertidas (filas son y, columnas son x)
        y, x = coord
        new_img[x, y] = color
    
    return new_img


def pintaMITitulo(vim, titles):
    """
    Funcion que permite dibujar multiples imagenes en una misma ventana con un
    titulo para cada iamgen.

    Args:
        vim: Lista que contiene las imagenes cargadas
        titles: Titulo asociado a cada imagen
    """
    pintaMI(vim, titles)


###############################################################################
# Prueba de la funcion leeimagen
leeimagen('img/orapple.jpg', True)

###############################################################################
# Prueba de la funcion pintaI
pintaI('img/messi.jpg')

###############################################################################
# Prueba de la funcion pintaMI

# Leer imagenes y pasarlas de BGR a RGB
image1 = cv.imread('img/orapple.jpg')
image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)

image2 = cv.imread('img/messi.jpg')
image2 = cv.cvtColor(image2, cv.COLOR_BGR2RGB)

image3 = cv.imread('img/messi-chikito.jpg')
image3 = cv.cvtColor(image3, cv.COLOR_BGR2RGB)

# Crear lista con imagenes
image_list = [image1, image2, image3]

pintaMI(image_list)

###############################################################################
# Prueba de la funcion modifica_color

# Leer imagen y pasarla de BGR a RGB
image = cv.imread('img/orapple.jpg')
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

# Estableser las posiciones de la imagen a modificar
positions = [(x, y) for x in range(100, 200) for y in range(100)]

# Establecer el nuevo color en formato RGB
color = [255, 0, 0]

# Obtener nueva imagen con colores modificados
new_img = modifica_color(image, positions, color)

# Mostrar imagen modificada
plt.imshow(new_img)
plt.axis('off')
plt.show()

###############################################################################
# Prueba de la funcion pintaMITitulo
pintaMITitulo(image_list, ["orapple", "messi", "messi chiquito"])
