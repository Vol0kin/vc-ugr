#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 12:10:17 2019

@author: Vladislav
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


def pintaI(img):
    """
    Funcion que normaliza una imagen que se le pasas como parametro. Normaliza
    cada uno de los canales.

    Args:
        img: Imagen a normalizar
    Return:
        Devuelve una imagen normalizada
    """

    # Copiar la imagen a normalizar y convertirla a float64
    normalized = np.copy(img)
    normalized = normalized.astype(np.float64)

    # Segun si la imagen es monobanda (2 dimensiones) o tribanda (3 dimensiones)
    # obtener los valores maximos y minimos de GRIS o RGB para cada pixel
    if normalized.ndim == 2:
        min_val = np.min(normalized)
        max_val = np.max(normalized)
    else:
        min_val = np.min(normalized, axis=(0, 1))
        max_val = np.max(normalized, axis=(0, 1))
    
    # Normalizar la imagen al rango [0, 1] y multiplicarla por 255
    normalized = (normalized - min_val) / (max_val - min_val) * 255

    # Redondear los valores de la imagen y convertir la imagen a uint8
    normalized = np.round(normalized).astype(np.uint8)

    return normalized

def pintaMI(vim, titles=None):
    """
    Funcion que pinta multiples imagenes en una misma ventana. Adicionalmente
    se pueden especificar que titulos tendran las imagenes.

    Args:
        vim: Lista con las imagenes cargas
        titles: Titulos que tendran las imagenes (default None)
    """

    # Obtener el numero de imagenes
    n_cols = len(vim)

    # Crear n_cols sublots (un subplot por cada imagen)
    # El formato sera 1 fila con n_cols columnas
    _, axarr = plt.subplots(1, n_cols)

    # Pintar cada imagen
    for i in range(n_cols):
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
leeimagen('imagenes/orapple.jpg', True)

###############################################################################
# Prueba de la funcion pintaI
image = cv.imread('imagenes/orapple.jpg')
normalized = pintaI(image)

# Pasar de BGR a RGB
normalized = cv.cvtColor(normalized, cv.COLOR_BGR2RGB)

# Dibujar imagen desactivando los ejes
plt.imshow(normalized)
plt.axis("off")
plt.show()

###############################################################################
# Prueba de la funcion pintaMI

# Leer imagenes y pasarlas de BGR a RGB
image1 = cv.imread('imagenes/orapple.jpg')
image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)

image2 = cv.imread('imagenes/messi.jpg')
image2 = cv.cvtColor(image2, cv.COLOR_BGR2RGB)

image3 = cv.imread('imagenes/orapple.jpg', 0)
image3 = cv.cvtColor(image3, cv.COLOR_BGR2RGB)

# Crear lista con imagenes
image_list = [image1, image2, image3]

# Para poder pintar las imagenes bien, he cambiado los canales de BGR a RGB,
# como se puede ver en las lineas anteriores
pintaMI(image_list)

###############################################################################
# Prueba de la funcion modifica_color

# Leer imagen y pasarla de BGR a RGB
image = cv.imread('imagenes/orapple.jpg')
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
pintaMITitulo(image_list, ["Orapple", "Messi", "Orapple B/N"])
