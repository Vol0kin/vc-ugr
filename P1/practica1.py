#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 12:10:17 2019

@author: Vladislav
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Bonus 1
# - Correlación 1D
# - Flip horizontal y vertical al kernel
# - Pasar la correalcion por filas y por columnas

def read_image(file_name, depth):
    img = cv.imread(file_name, depth)

    img = transform_img_float64(img)

    return img


def transform_img_float64(img):
    """
    Funcion que pasa una imagen a float64.

    Args:
        img: Imagen a transformar
    Return:
        Devuelve una imagen en float64
    """

    # Copiar la imagen y convertirla a float64
    transformed = np.copy(img)
    transformed = transformed.astype(np.float64)

    return transformed


def transform_img_uint8(img):
    """
    Funcion que transforma una en float64 a una imagen en uint8 donde cada pixel
    esta en el rango [0, 255]

    Args:
        img: Imagen a transformar
    Return:
        Devuelve la imagen en el rango [0, 255]
    """
    # Copiar la imagen
    trans = np.copy(img)

    # Segun si la imagen es monobanda (2 dimensiones) o tribanda (3 dimensiones)
    # obtener los valores maximos y minimos de GRIS o RGB para cada pixel
    if trans.ndim == 2:
        min_val = np.min(trans)
        max_val = np.max(trans)
    else:
        min_val = np.min(trans, axis=(0, 1))
        max_val = np.max(trans, axis=(0, 1))

    # Normalizar la imagen al rango [0, 1]
    norm = (trans - min_val) / (max_val - min_val)

    # Multiplicar cada pixel por 255
    norm = norm * 255

    # Redondear los valores y convertirlos a uint8
    trans_uint8 = np.round(norm).astype(np.uint8)

    return trans_uint8


def visualize_image(img, title=None):
    """
    Funcion que visualiza una imagen por pantalla.

    Args:
        img: Imagen a visualizar
        title: Titulo de la imagen (por defecto None)
    """
    # Pasar la imagen a uint8
    vis = transform_img_uint8(img)

    # Pasar de una imagen BGR a RGB
    vis = cv.cvtColor(vis, cv.COLOR_BGR2RGB)

    # Visualizar la imagen
    plt.imshow(vis)
    plt.axis('off')

    if title is not None:
        plt.title(title)

    plt.show()


def visualize_mult_img(images, titles=None):
    """
    Funcion que pinta multiples imagenes en una misma ventana. Adicionalmente
    se pueden especificar que titulos tendran las imagenes.

    Args:
        images: Lista con las imagenes
        titles: Titulos que tendran las imagenes (default None)
    """

    # Obtener el numero de imagenes
    n_img = len(images)

    # Crear n_cols sublots (un subplot por cada imagen)
    # El formato sera 1 fila con n_cols columnas
    _, axarr = plt.subplots(1, n_img)

    # Pintar cada imagen
    for i in range(n_img):
        # Convertir la imagen a uint8
        img = transform_img_uint8(images[i])

        # Pasarla de BGR a RGB
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Obtener siguiente imagen
        axarr[i].imshow(img)

        # Determinar si hay que poner o no titulo
        if titles != None:
            axarr[i].set_title(titles[i])

        axarr[i].axis('off')
        print(img.shape)

    # Mostar imagenes
    plt.show()


def gaussian_kernel(img, ksize, sigma_x, sigma_y, border):
    """
    Funcion que aplica un kernel Gaussiano sobre una imagen.

    Args:
        img: Imagen sobre la que aplicar el kernel
        ksize: Tamaño del kernel
        sigma_x: Sigma sobre el eje X
        sigma_y: Sigma sobre el eje y
        border: Tipo de borde
    Return:
        Devuelve una imagen sobre la que se ha aplicado un kernel Gaussiano
    """

    # Aplicar kernel Gaussiano
    gauss = cv.GaussianBlur(img, ksize, sigmaX=sigma_x,
                            sigmaY=sigma_y, borderType=border)

    return gauss


def derivative_kernel(img, dx, dy, ksize, border):
    """
    Funcion que aplica un kernel de derivadas a una imagen.

    Args:
        img: Imagen sobre la que aplicar el kernel.
        dx: Numero de derivadas que aplicar sobre el eje X.
        dy: Numero de derivadas que aplicar sobre el eje Y.
        ksize: Tamaño del kernel
        border: Tipo de borde
    Return:
        Devuelve una imagen sobre la que se ha aplicado el filtro de derivadas.
    """
    # Obtener los kernels que aplicar a cada eje (es descomponible porque es
    # el kernel de Sobel)
    kx, ky = cv.getDerivKernels(dx, dy, ksize, normalize=True)

    # Aplicar los kernels sobre la imagen
    der = cv.sepFilter2D(img, cv.CV_64F, kx, ky, borderType=border)

    return der


def log_kernel(img, ksize, sigma_x, sigma_y, border):
    """
    Funcion que aplica un kernel LoG (Laplacian of Gaussian) sobre una imagen.

    Args:
        img: Imagen sobre la que aplicar el kernel
        ksize: Tamaño del kernel Gaussiano y Laplaciano
        sigma_x: Valor de sigma en el eje X de la Gaussiana
        sigma_y: Valor de sigma en el eje Y de la Gaussiana
        border: Tipo de borde
    Return:
        Devuelve una imagen sobre la que se ha aplicado un filtro LoG
    """

    # Aplicar filtro Gaussiano
    gauss = gaussian_kernel(img, (ksize, ksize), sigma_x, sigma_y, border)

    # Aplicar filtro Laplaciano
    laplace = cv.Laplacian(gauss, cv.CV_64F, ksize=ksize, borderType=border)

    return laplace


def gaussian_pyramid(img, ksize, sigma_x, sigma_y, border, N=4):
    """
    Funcion que devuelve una piramide Gaussiana de tamaño N

    Args:
        img: Imagen de la que extraer la piramide
        ksize: Tamaño del kernel
        sigma_x: Valor de sigma en el eje X
        sigma_y: Valor de sigma en el eje Y
        border: Tipo de borde a utilizar
        N: Numero de imagenes que componen la piramide (default 4)

    """

    gaussian_pyr = [img]

    for i in range(1, N):
        # Obtener el elemento anterior de la piramide Gaussiana
        prev_img = np.copy(gaussian_pyr[i - 1])

        # Aplicar Gaussian Blur
        gauss = gaussian_kernel(prev_img, ksize, sigma_x, sigma_y, border)

        # Reducir el tamaño de la imagen a la mitad
        down_sampled_gauss = gauss[1::2, 1::2]
        print(down_sampled_gauss.shape)

        # Añadir imagen a la piramide
        gaussian_pyr.append(down_sampled_gauss)


    return gaussian_pyr


def laplacian_pyramid(img, ksize, sigma_x, sigma_y, border, N=4):

    # Obtener piramide Gaussiana de un nivel mas
    gaussian_pyr = gaussian_pyramid(img, ksize, sigma_x, sigma_y, border, N+1)

    # Crear la lista que contendra la piramide Laplaciana
    # Se inserta el ultimo elemento de la piramide Gaussiana primero
    laplacian_pyr = [gaussian_pyr[-1]]

    # Recorrer en orden inverso la piramide Gaussiana y generar la Laplaciana
    for i in reversed(range(1, N+1)):
        # Obtener la imagen actual y la anterior
        current_img = gaussian_pyr[i]
        previous_img = gaussian_pyr[i - 1]

        # Hacer upsampling de la imagen actual
        upsampled_img = np.repeat(current_img, 2, axis=0)
        upsampled_img = np.repeat(upsampled_img, 2, axis=1)

        # Si falta una fila, copiar la ultima
        if upsampled_img.shape[0] < previous_img.shape[0]:
            upsampled_img = np.vstack([upsampled_img, upsampled_img[-1]])

        # Si falta una fila, copiar la ultima
        if upsampled_img.shape[1] < previous_img.shape[1]:
            upsampled_img = np.hstack([upsampled_img, upsampled_img[:, -1].reshape(-1, 1)])

        # Pasar un Gaussian Blur a la imagen escalada
        upsampled_gauss = gaussian_kernel(upsampled_img, ksize, sigma_x, sigma_y, border)

        # Restar la imagen orignal de la escalada para obtener detalles
        diff_img = previous_img - upsampled_gauss

        # Guardar en la piramide
        laplacian_pyr.insert(0, diff_img)


    return laplacian_pyr


def non_max_supression(img):
    dim1 = img.shape[0]
    dim2 = img.shape[1]

    supressed_img = np.zeros_like(img)

    for i in range(dim1):
        for j in range(dim2):
            region = img[max(i-1, 0):i+2, max(j-1, 0):j+2]
            current_val = img[i, j]
            max_val = np.max(region)

            if max_val == current_val:
                supressed_img[i, j] = current_val

    return supressed_img


def laplacian_scale_space():
    return None


###############################################################################
###############################################################################
# Ejercicio 1

#######################################
# Apartado A
# Aplicacion de filtros Gaussianos

# Cargar una imagen y visualizarla
img = read_image('imagenes/cat.bmp', 0)
visualize_image(img, 'Original image')

# Aplicar Gaussian Blur de tamaño (5, 5) con sigma = 1 y BORDER_REPLICATE
gauss = gaussian_kernel(img, (5,5), 1, 1, cv.BORDER_REPLICATE)
visualize_image(gauss, r'$5 \times 5$ Gaussian Blur with $\sigma = 1$ and BORDER_REPLICATE')

# Aplicar Gaussian Blur de tamaño (5, 5) con sigma = 3 y BORDER_REPLICATE
gauss = gaussian_kernel(img, (5,5), 3, 3, cv.BORDER_REPLICATE)
visualize_image(gauss, r'$5 \times 5$ Gaussian Blur with $\sigma = 3$ and BORDER_REPLICATE')

# Aplicar Gaussian Blur de tamaño (11, 11) con sigma = 4 y BORDER_REPLICATE
gauss = gaussian_kernel(img, (11,11), 4, 4, cv.BORDER_REPLICATE)
visualize_image(gauss, r'$11 \times 11$ Gaussian Blur with $\sigma = 4$ and BORDER_REPLICATE')

# Aplicar Gaussian Blur de tamaño (11, 11) con sigma = 4 y BORDER_REFLECT
gauss = gaussian_kernel(img, (101,101), 15, 15, cv.BORDER_REFLECT)
visualize_image(gauss, r'$11 \times 11$ Gaussian Blur with $\sigma = 4$ and BORDER_REFLECT')

# Aplicar Gaussian Blur de tamaño (11, 11) con sigma = 4 y BORDER_CONSTANT
gauss = gaussian_kernel(img, (101,101), 15, 15, cv.BORDER_CONSTANT)
visualize_image(gauss, r'$11 \times 11$ Gaussian Blur with $\sigma = 4$ and BORDER_CONSTANT')

# Aplicar Gaussian Blur de tamaño (11, 11) con sigma = 4 y BORDER_DEFAULT
gauss = gaussian_kernel(img, (101,101), 15, 15, cv.BORDER_REPLICATE)
visualize_image(gauss, r'$11 \times 11$ Gaussian Blur with $\sigma = 4$ and BORDER_DEFAULT')

# Aplicar Gaussian Blur de tamaño (11, 11) con sigmax = 5, sigmay = 2 y BORDER_REPLICATE
gauss = gaussian_kernel(img, (11,11), 5, 2, cv.BORDER_REPLICATE)
visualize_image(gauss, r'$11 \times 11$ Gaussian Blur with $\sigma_x = 5$, $\sigma_y = 2$ and BORDER_REPLICATE')

# Aplicar Gaussian Blur de tamaño (11, 11) con sigmax = 2, sigmay = 5 y BORDER_REPLICATE
gauss = gaussian_kernel(img, (11,11), 2, 5, cv.BORDER_REPLICATE)
visualize_image(gauss, r'$11 \times 11$ Gaussian Blur with $\sigma_x = 2$, $\sigma_y = 5$ and BORDER_REPLICATE')

####################################
# Aplicacion de filtros de derivadas

# Aplicar filtro de primera derivada en el eje X con tamaño 5 y BORDER_DEFAULT
der = derivative_kernel(img, 1, 0, 5, cv.BORDER_DEFAULT)
visualize_image(der, r'$5 \times 5$ First Derivative Kernel in X-axis and BORDER_DEFAULT')

# Aplicar filtro de primera derivada en el eje Y con tamaño 5 y BORDER_DEFAULT
der = derivative_kernel(img, 0, 1, 5, cv.BORDER_DEFAULT)
visualize_image(der, r'$5 \times 5$ First Derivative Kernel in Y-axis and BORDER_DEFAULT')

# Aplicar filtro de segunda derivada en el eje X con tamaño 5 y BORDER_DEFAULT
der = derivative_kernel(img, 2, 0, 5, cv.BORDER_DEFAULT)
visualize_image(der, r'$5 \times 5$ Second Derivative Kernel in X-axis and BORDER_DEFAULT')

# Aplicar filtro de segunda derivada en el eje Y con tamaño 5 y BORDER_DEFAULT
der = derivative_kernel(img, 0, 2, 5, cv.BORDER_DEFAULT)
visualize_image(der, r'$5 \times 5$ Second Derivative Kernel in Y-axis and BORDER_DEFAULT')

# Aplicar filtro de primera derivada en ambos ejes con tamaño 5 y BORDER_DEFAULT
der = derivative_kernel(img, 1, 1, 5, cv.BORDER_DEFAULT)
visualize_image(der, r'$5 \times 5$ First Derivative Kernel in both axis and BORDER_DEFAULT')

# Aplicar filtro de segunda derivada en ambos ejes con tamaño 5 y BORDER_DEFAULT
der = derivative_kernel(img, 2, 2, 5, cv.BORDER_DEFAULT)
visualize_image(der, r'$5 \times 5$ Second Derivative Kernel in both axis and BORDER_DEFAULT')

# Aplicar filtro de primera derivada en ambos ejes con tamaño 7 y BORDER_DEFAULT
der = derivative_kernel(img, 1, 1, 7, cv.BORDER_DEFAULT)
visualize_image(der, r'$7 \times 7$ First Derivative Kernel in both axis and BORDER_DEFAULT')

# Aplicar filtro de primera derivada en ambos ejes con tamaño 11 y BORDER_DEFAULT
der = derivative_kernel(img, 1, 1, 11, cv.BORDER_DEFAULT)
visualize_image(der, r'$11 \times 11$ First Derivative Kernel in both axis and BORDER_DEFAULT')

# Aplicar filtro de primera derivada en ambos ejes con tamaño 15 y BORDER_DEFAULT
der = derivative_kernel(img, 1, 1, 15, cv.BORDER_DEFAULT)
visualize_image(der, r'$15 \times 15$ First Derivative Kernel in both axis and BORDER_DEFAULT')

# Aplicar filtro de primera derivada en ambos ejes con tamaño 5 y BORDER_REPLICATE
der = derivative_kernel(img, 1, 1, 5, cv.BORDER_REPLICATE)
visualize_image(der, r'$5 \times 5$ First Derivative Kernel in both axis and BORDER_REPLICATE')

# Aplicar filtro de primera derivada en ambos ejes con tamaño 5 y BORDER_REFLECT
der = derivative_kernel(img, 1, 1, 5, cv.BORDER_REFLECT)
visualize_image(der, r'$5 \times 5$ First Derivative Kernel in both axis and BORDER_REFLECT')

#######################################
# Apartado B

laplace = log_kernel(img, 5, 7, 7, cv.BORDER_DEFAULT)
visualize_image(laplace)


###############################################################################
###############################################################################
# Ejercicio 2

pyr = gaussian_pyramid(img, (5,5), 3, 3, cv.BORDER_REFLECT)
visualize_mult_img(pyr)

pyr = laplacian_pyramid(img, (5,5), 3, 3, cv.BORDER_REFLECT)
visualize_mult_img(pyr)
