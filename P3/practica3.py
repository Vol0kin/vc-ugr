# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2

###############################################################################
# 							Funciones auxiliares                              #
###############################################################################

def read_image(file_name, depth):
    """
    Funcion que carga una imagen con la profundidad especificada

    Args:
        file_name: Nombre de la imagen a cargar
        depth: Profundiad (1 color, 0 ByN)
    Return:
        Devuelve una imagen en float64
    """
    # Cargar imagen
    img = cv2.imread(file_name, depth)

    # Transformar imagen a float64
    img = transform_img_float32(img)

    return img


def transform_img_float32(img):
    """
    Funcion que pasa una imagen a float32.

    Args:
        img: Imagen a transformar
    Return:
        Devuelve una imagen en float64
    """

    # Copiar la imagen y convertirla a float32
    transformed = np.copy(img)
    transformed = transformed.astype(np.float32)

    return transformed


def transform_img_uint8(img):
    """
    Funcion que transforma una en float32 a una imagen en uint8 donde cada pixel
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
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    # Visualizar la imagen
    plt.imshow(vis)
    plt.axis('off')

    if title is not None:
        plt.title(title)

    plt.show()


def non_max_supression(img, block_size):
    """
    Funcion que realiza la supresion de no maximos dada una imagen de entrada

    Args:
        img: Imagen sobre la que realizar la supresion de no maximos
        block_size: Tamaño del bloque
    Return:
        Devuelve una nueva imagen sobre la que se han suprimido los maximos
    """
    # Crear imagen inicial para la supresion de maximos (inicializada a 0)
    supressed_img = np.zeros_like(img)

    region_range = block_size // 2

    # Para cada pixel, aplicar una caja 3x3 para determinar el maximo local
    for i,j in np.ndindex(img.shape):
        # Obtener la region 3x3 (se consideran los bordes para que la caja
        # tenga el tamaño adecuado, sin salirse)
        region = img[max(i-region_range, 0):i+region_range+1, max(j-region_range, 0):j+region_range+1]

        # Obtener el valor actual y el maximo de la region
        current_val = img[i, j]
        max_val = np.max(region)

        # Si el valor actual es igual al maximo, copiarlo en la imagen
        # de supresion de no maximos
        if max_val == current_val:
            supressed_img[i, j] = current_val

    return supressed_img

###############################################################################
#                   Apartado 1: Deteccion de puntos de Harris                 #
###############################################################################

def compute_points_of_interest(img, block_size, k_size):

    # Obtener valores singulares y vectores asociados
    sv_vectors = cv2.cornerEigenValsAndVecs(img, block_size, k_size)

    print(sv_vectors)

    # Quedarse solo con los valores singulares
    # Los valores singulares son los dos primeros valores de la matriz
    sv = sv_vectors[:, :, :2]

    print(sv)

    # Calcular valor de cada píxel como \frac{lamb1 * lamb2}{lamb1 + lamb2}
    points_interest = np.prod(sv, axis=2) / np.sum(sv, axis=2)

    return points_interest


def threshold_points_of_interest(points, threshold=0.7):
    points[points < threshold] = 0.0


def harris_corner_detection(img, block_size, k_size, num_octaves):

    points_interest = compute_points_of_interest(img, block_size, k_size)

    # Aplicar umbralizacion
    threshold_points_of_interest(points_interest)

    # Aplicar supresion de no maximos
    points_interest = non_max_supression(points_interest, block_size)

    # Obtener indices
    points_idx = np.where(points_interest > 0.0)

    for y, x in zip(*points_idx):
        scale = block_size

    return points_interest

###############################################################################
###############################################################################

# Cargar imagen de Yosemite
yosemite = read_image('imagenes/Yosemite1.jpg', 0)
print(yosemite.shape)

pi = harris_corner_detection(yosemite, 5, 3, 1)

visualize_image(pi)