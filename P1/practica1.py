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

def normalize_image(img):
    """
    Funcion que normaliza una imagen que se le pasa como parametro. Normaliza
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
    
    # Normalizar la imagen al rango [0, 1]
    normalized = (normalized - min_val) / (max_val - min_val)

    return normalized

def transform_img_uint8(norm):
	"""
	Funcion que transforma una cuyos pixeles estan en el rango [0, 1] al rango
	[0, 255].

	Args:
		norm: Imagen normalizada
	Return:
		Devuelve la imagen en el rango [0, 255]
	"""
	# Multiplicar cada pixel por 255
	transformed = norm * 255

	# Redondear los valores y convertirlos a uint8
	transformed = np.round(transformed).astype(np.uint8)

	return transformed


def visualize_image(img):
	"""
	Funcion que visualiza una imagen por pantalla.

	Args:
		img: Imagen a visualizar
	"""
	# Pasar de una imagen BGR a RGB
	vis = cv.cvtColor(img, cv.COLOR_BGR2RGB)

	# Visualizar la imagen
	plt.imshow(vis)
	plt.axis('off')
	plt.show()


def gaussian_kernel_norm(img, ksize, sigma_x, sigma_y, border):
	"""
	Funcion que aplica un kernel Gaussiano sobre una imagen. Devuelve la imagen
	con los pixels en el rango [0, 1].

	Args:
		img: Imagen sobre la que aplicar el kernel
		ksize: Tamaño del kernel
		sigma_x: Sigma sobre el eje X
		sigma_y: Sigma sobre el eje y
		border: Tipo de borde
	Return:
		Devuelve una imagen sobre la que se ha aplicado un kernel Gaussiano
	"""
	# Normalizar la imagen
	img_norm = normalize_image(img)

	# Aplicar kernel Gaussiano
	gauss = cv.GaussianBlur(img_norm, ksize, sigmaX=sigma_x,
							sigmaY=sigma_y, borderType=border)
	
	return gauss


def gaussian_kernel(img, ksize, sigma_x, sigma_y, border):
	"""
	Funcion que aplica un kernel Gaussiano sobre una imagen. Devuelve la imagen
	con los pixels en el rango [0, 255]. Hace uso de gaussian_kernel_norm.

	Args:
		img: Imagen sobre la que aplicar el kernel
		ksize: Tamaño del kernel
		sigma_x: Sigma sobre el eje X
		sigma_y: Sigma sobre el eje y
		border: Tipo de borde
	Return:
		Devuelve una imagen sobre la que se ha aplicado un kernel Gaussiano
	"""
	# Aplicar el kernel Gaussiano
	gauss = gaussian_kernel_norm(img, ksize, sigma_x, sigma_y, border)
	
	# Pasar los valores de los pixels a uint8
	gauss = transform_img_uint8(gauss)

	return gauss


def derivative_kernel(img, dx, dy, ksize):
	"""
	Funcion que aplica un kernel de derivadas a una imagen.

	Args:
		img: Imagen sobre la que aplicar el kernel.
		dx: Numero de derivadas que aplicar sobre el eje X.
		dy: Numero de derivadas que aplicar sobre el eje Y.
		ksize: Tamaño del kernel
	Return:
		Devuelve una imagen sobre la que se ha aplicado el filtro de derivadas.
	"""
	# Normalizar la imagen
	img_norm = normalize_image(img)

	# Obtener los kernels que aplicar a cada eje (es descomponible porque es
	# el kernel de Sobel)
	kx, ky = cv.getDerivKernels(dx, dy, ksize, normalize=True)

	# Aplicar los kernels sobre la imagen
	der = cv.sepFilter2D(img_norm, cv.CV_64F, kx, ky)

	# Pasar los pixeles de la imagen a uint8
	der = transform_img_uint8(der)

	return der






img = cv.imread('imagenes/cat.bmp')
visualize_image(img)

gauss = gaussian_kernel(img, (11,11), 15, 15, cv.BORDER_DEFAULT)
visualize_image(gauss)

der = derivative_kernel(img, 1, 0, 5)
visualize_image(der)



