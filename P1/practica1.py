#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 12:10:17 2019

@author: Vladislav
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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
    img = cv.imread(file_name, depth)

    # Transformar imagen a float64
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


def visualize_mult_images(images, titles=None):
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

    # Mostar imagenes
    plt.show()


def apply_kernel(img, kx, ky, border):
    """
    Funcion que aplica un kernel separable sobre una imagen, realizando una
    convolucion

    Args:
        img: Imagen sobre la que aplicar el filtro
        kx: Kernel en el eje X
        ky: Kernel en el eje Y
        border: Tipo de borde
    Return:
        Devuelve una imagen filtrada
    """
    # Hacer el flip a los kernels para aplicarlos como una convolucion
    kx_flip = np.flip(kx)
    ky_flip = np.flip(ky)

    # Realizar la convolucion
    conv_x = cv.filter2D(img, cv.CV_64F, kx_flip.T, borderType=border)
    conv = cv.filter2D(conv_x, cv.CV_64F, ky_flip, borderType=border)

    return conv


def gaussian_kernel(img, ksize_x, ksize_y, sigma_x, sigma_y, border):
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
    # Obtener un kernel para cada eje
    kernel_x = cv.getGaussianKernel(ksize_x, sigma_x)
    kernel_y = cv.getGaussianKernel(ksize_y, sigma_y)

    # Aplicar kernel Gaussiano
    gauss = apply_kernel(img, kernel_x, kernel_y, border)

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
    der = apply_kernel(img, kx, ky, border)

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
    gauss = gaussian_kernel(img, ksize, ksize, sigma_x, sigma_y, border)
    
    # Obtener los filtros de derivada segunda en cada eje, aplicados sobre
    # el filtro Gaussiano
    dx2 = derivative_kernel(gauss, 2, 0, ksize, border)
    dy2 = derivative_kernel(gauss, 0, 2, ksize, border)
    
    # Combinar los filtros de derivadas y obtener Laplaciana de Gaussiana
    laplace = dx2 + dy2

    return laplace


def create_img_pyramid(pyramid):
    """
    Funcion que crea una imagen a partir de una lista de imagenes que forman una
    piramide

    Args:
        pyramid: Lista que contiene las imagenes
    Return:
        Devuelve una imagen compuesta por todas las imagenes de la piramide
    """
    # Crear una copia de la primera imagen (la mas grande y a la que se añadiran
    # el resto de imagenes)
    pyr_img = np.copy(pyramid[0])

    # Para cada imagen de la lista, concatenarla a la primera imagen
    for i in range(1, len(pyramid)):
        # Obtener imagen actual
        insert_img = np.copy(pyramid[i])

        # Determinar cuantas filas de blancos se tienen que insertar respecto a
        # la imagen original para que tengan el mismo tamaño
        n_new_rows = pyramid[0].shape[0] - insert_img.shape[0]

        # Crear una imagen de blancos que se concatenara con la imagen actual para
        # que tenga las mismas dimensiones que la actual
        if insert_img.ndim == 2:
            # Se escoge el maximo entre el maximo valor de la imagen mas grande y la mas pequeña
            # Se hace asi debido a que, esta parte de la funcion se utiliza tanto para la piramide
            # Gaussiana como la Laplaciana, con lo cual, hay que ver cual es el maximo valor que se
            # puede encontrar en cualquiera de las dos piramides
            white_rows = np.full((n_new_rows, insert_img.shape[1]), max(np.max(pyramid[-1]), np.max(pyramid[0])))
        else:
            # Si es una imagen a color, se escoge el maximo para cada uno de los canales
            # para que el fondo sea blanco
            white_rows = np.full((n_new_rows, insert_img.shape[1], 3), np.max(pyramid[0], axis=(0, 1)))

        # Insertar los blancos verticalmente, por encima de la imagen actual
        insert_img = np.vstack([white_rows, insert_img])

        # Insertar la nueva imagen en la piramide, concatenandola por la derecha
        pyr_img = np.hstack([pyr_img, insert_img])

    return pyr_img


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
    # Inicializar la piramide con la primera imagen
    gaussian_pyr = [img]

    for i in range(1, N):
        # Obtener el elemento anterior de la piramide Gaussiana
        prev_img = np.copy(gaussian_pyr[i - 1])

        # Aplicar Gaussian Blur
        gauss = gaussian_kernel(prev_img, ksize, ksize, sigma_x, sigma_y, border)

        # Reducir el tamaño de la imagen a la mitad
        down_sampled_gauss = gauss[1::2, 1::2]

        # Añadir imagen a la piramide
        gaussian_pyr.append(down_sampled_gauss)


    return gaussian_pyr


def laplacian_pyramid(img, ksize, sigma_x, sigma_y, border, N=4):
    """
    Funcion que crea una piramide Laplaciana de una imagen

    Args:
        img: Imagen de la que crar la piramide
        ksize: Tamaño del kernel
        sigma_x: Valor de sigma en el eje X
        sigma_y: Valor de sigma en el eje Y
        border: Tipo de borde
        N: Numero de componentes de la piramide. El nivel Gaussiano (ultimo)
           no esta incluido(default 4)
    Return:
        Devuelve una lista que contiene las imagenes que forman la piramide
    """

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

        # Pasar un Gaussian Blur a la imagen escalada para intentar suavizar el efecto de las
        # filas y las columnas repetidas
        upsampled_gauss = gaussian_kernel(upsampled_img, ksize, ksize, sigma_x, sigma_y, border)

        # Restar la imagen orignal de la escalada para obtener detalles
        diff_img = previous_img - upsampled_gauss

        # Guardar en la piramide
        laplacian_pyr.insert(0, diff_img)


    return laplacian_pyr


def non_max_supression(img):
    """
    Funcion que realiza la supresion de no maximos dada una imagen de entrada

    Args:
        img: Imagen sobre la que realizar la supresion de no maximos
    Return:
        Devuelve una nueva imagen sobre la que se han suprimido los maximos
    """
    # Crear imagen inicial para la supresion de maximos (inicializada a 0)
    supressed_img = np.zeros_like(img)

    # Para cada pixel, aplicar una caja 3x3 para determinar el maximo local
    for i,j in np.ndindex(img.shape):
        # Obtener la region 3x3 (se consideran los bordes para que la caja
        # tenga el tamaño adecuado, sin salirse)
        region = img[max(i-1, 0):i+2, max(j-1, 0):j+2]

        # Obtener el valor actual y el maximo de la region
        current_val = img[i, j]
        max_val = np.max(region)

        # Si el valor actual es igual al maximo, copiarlo en la imagen de supresion
        # de no maximos
        if max_val == current_val:
            supressed_img[i, j] = current_val

    return supressed_img


def laplacian_scale_space(img, ksize, border, N, sigma=1.0, sigma_inc=1.2):
    """
    Funcion que construye el espacio de escalas Laplaciano de una imagen dada

    Args:
        img: Imagen de la que extraer el espacio de escalas
        ksize: Tamaño del kernel
        border: Tipo de borde
        N: Numero de escalas
        sigma: Valor inicial del kernel (default 1)
        sigma_inc: Multiplicador que incrementa el sigma (default 1.2)
    Return:
        Devuelve una lista con N imagenes, formando el espacio de escalas y los
        valores de sigma utilizados
    """
    # Crear listas que contendran las imagenes y los sigma
    scale_space = []
    sigma_list = []

    # Crear las N escalas
    for _ in range(N):
        # Aplicar Laplacian of Gaussian
        level_img = log_kernel(img, ksize, sigma, sigma, border)

        # Normalizar multiplicando por sigma^2
        level_img *= sigma ** 2

        # Elevar al cuadrado la imagen resultante
        level_img = level_img ** 2

        # Suprimir no maximos
        supressed_level = non_max_supression(level_img)

        # Guardar imagen y sigma
        scale_space.append(supressed_level)
        sigma_list.append(sigma)

        # Incrementar sigma
        sigma *= sigma_inc

    return scale_space, sigma_list


def visualize_laplacian_scale_space(img, sigma, title=None):
    """
    Funcion que permite visualizar una imagen generada en el espacio de escalas
    Laplaciano con circulos en las zonas destacadas

    Args:
        img: Imagen que mostrar
        sigma: Valor de sigma que se ha utilizado para generar la iamgen
        title: Titulo de la imagen (default None)
    """
    # Pasar la imagen a uint8
    vis = transform_img_uint8(img)

    # Obtener los indices de las filas y columnas donde los pixels tienen un valor
    # por encima de la media (es decir, que hayan sido destacados)
    # Las filas y columnas estan invertidas
    idx_col, idx_row = np.where(vis > 128)

    # Pasar de una imagen BGR a RGB
    vis = cv.cvtColor(vis, cv.COLOR_BGR2RGB)

    # Pintar un circulo verde por cada punto
    for point in zip(idx_row, idx_col):
        cv.circle(vis, point, int(15 * sigma), (0, 255, 0))

    # Visualizar la imagen
    plt.imshow(vis)
    plt.axis('off')

    if title is not None:
        plt.title(title)

    plt.show()


def hybrid_image_generator(img_low, img_high, ksize, sigma_low, sigma_high, border):
    """
    Funcion que permite crear una imagen hibrida combinando dos imagenes

    Args:
        img_low: Imagen que sera utilizada para las bajas frecuencias
        img_high: Imagen que sera utilizada para las altas frecuencias
        ksize: Tamaño del kernel (es una tupla)
        sigma_low: Sigma para la imagen de bajas frecuencias
        sigma_high: Sigma para la imgane de altas frecuencias
        border: Tipo de borde
    Return:
        Devuelve una lista que contiene la imagen de bajas frecuencias, la de altas
        y la hibrida
    """
    # Crear la imagen de bajas frecuencias aplicando filtro Gaussiano
    low_freq_img = gaussian_kernel(img_low, ksize, ksize, sigma_low, sigma_low, border)

    # Crear imagen de altas frecuencias aplicando filtro Gaussiano y restando a la original
    gauss_high_freq = gaussian_kernel(img_high, ksize, ksize, sigma_high, sigma_high, border)
    high_freq_img = img_high - gauss_high_freq

    # Crear la imagen hibrida combinando las dos
    hybrid = low_freq_img + high_freq_img

    return [low_freq_img, high_freq_img, hybrid]


def correlation_1D(img, kernel):
    """
    Funcion que aplica la correlacion 1D sobre una imagen de entrada utilizando
    un kernel dado

    Args:
        img: Imagen sobre la que aplicar la correlacion
        kernel: Kernel que se aplicara
    Return:
        Devuelve una imagen sobre la que se ha aplicado correlacion 1D por filas
    """
    # Obtener el numero de elementos que se deben replicar al principio y al final
    # de la imagen
    n_replica = kernel.shape[0] // 2

    # Replicar elemento inicial y final de la imagen n_replica veces
    replica_img = np.hstack([np.tile(img[:, 0].reshape(-1, 1), n_replica), img])
    replica_img = np.hstack([replica_img, np.tile(img[:, -1].reshape(-1, 1), n_replica)])

    # Crear matriz de salida con el mismo tamaño que img
    correlation_mat = np.empty_like(img)

    # Aplicar la correlacion sobre cada elemento (i, j) de la imagen
    for i in range(img.shape[0]):
        for j in range(n_replica, img.shape[1] + n_replica):
            # Obtener sub imagen del mismo tamaño que el kernel
            sub_img = replica_img[i, j - n_replica:j + n_replica + 1]

            # Realizar producto escalar de la sub imagen y el kernel
            corr_value = np.dot(sub_img, kernel)

            # Actualizar valor de la matriz de correlacion
            correlation_mat[i, j - n_replica] = corr_value

    return correlation_mat


def convolution(img, kernel_x, kernel_y):
    """
    Funcion que aplica la convolucion sobre una imagen dados un kernel para
    el eje X y el eje Y
    
    Args:
        img: Imagen sobre la que aplicar la convolucion
        kernel_x: Kernel en el eje X
        kernel_y: Kernel en el eje Y
    Return:
        Devuelve la convolucion de la imagen con los kernels de entrada
    """
    # Hacer que los kernels sean vectores columna
    kernel_x = kernel_x.reshape(-1, 1)
    kernel_y = kernel_y.reshape(-1, 1)
    
    # Realizar un flip sobre los kernels
    kernel_x_flip = np.flip(kernel_x)
    kernel_y_flip = np.flip(kernel_y)

    # Realizar la convolucion (primero sobre el eje X y luego sobre el eje Y)
    convolution = correlation_1D(img, kernel_x_flip)
    convolution = correlation_1D(convolution.T, kernel_y_flip)

    # Trasponer la convolucion (porque se han cambiado filas por columnas anteriormente)
    convolution = convolution.T

    return convolution


###############################################################################
###############################################################################
# Ejercicio 1

#######################################
# Apartado A
# Aplicacion de filtros Gaussianos

# Cargar las imagenes en blanco y negro
cat = read_image('imagenes/cat.bmp', 0)
dog = read_image('imagenes/dog.bmp', 0)

bird = read_image('imagenes/bird.bmp', 0)
plane = read_image('imagenes/plane.bmp', 0)

einstein = read_image('imagenes/einstein.bmp', 0)
marilyn = read_image('imagenes/marilyn.bmp', 0)

bicycle = read_image('imagenes/bicycle.bmp', 0)
motorcycle = read_image('imagenes/motorcycle.bmp', 0)

fish = read_image('imagenes/fish.bmp', 0)
submarine = read_image('imagenes/submarine.bmp', 0)

# Visualizar imagen del gato
visualize_image(cat, 'Original image')

# Aplicar Gaussian Blur de tamaño (5, 5) con sigma = 1 y BORDER_REPLICATE
gauss = gaussian_kernel(cat, 5,5, 1, 1, cv.BORDER_REPLICATE)
visualize_image(gauss, r'$5 \times 5$ Gaussian Blur with $\sigma = 1$ and BORDER_REPLICATE')

# Aplicar Gaussian Blur de tamaño (5, 5) con sigma = 3 y BORDER_REPLICATE
gauss = gaussian_kernel(cat, 5,5, 3, 3, cv.BORDER_REPLICATE)
visualize_image(gauss, r'$5 \times 5$ Gaussian Blur with $\sigma = 3$ and BORDER_REPLICATE')

# Aplicar Gaussian Blur de tamaño (11, 11) con sigma = 3 y BORDER_REPLICATE
gauss = gaussian_kernel(cat, 11,11, 3, 3, cv.BORDER_REPLICATE)
visualize_image(gauss, r'$11 \times 11$ Gaussian Blur with $\sigma = 3$ and BORDER_REPLICATE')

# Aplicar Gaussian Blur de tamaño (101, 101) con sigma = 15 y BORDER_REFLECT
gauss = gaussian_kernel(cat, 101,101, 15, 15, cv.BORDER_REFLECT)
visualize_image(gauss, r'$101 \times 101$ Gaussian Blur with $\sigma = 15$ and BORDER_REFLECT')

# Aplicar Gaussian Blur de tamaño (101, 101) con sigma = 15 y BORDER_CONSTANT
gauss = gaussian_kernel(cat, 101,101, 15, 15, cv.BORDER_CONSTANT)
visualize_image(gauss, r'$101 \times 101$ Gaussian Blur with $\sigma = 15$ and BORDER_CONSTANT')

# Aplicar Gaussian Blur de tamaño (11, 11) con sigma = 4 y BORDER_DEFAULT
gauss = gaussian_kernel(cat, 101,101, 15, 15, cv.BORDER_REPLICATE)
visualize_image(gauss, r'$101 \times 101$ Gaussian Blur with $\sigma = 15$ and BORDER_DEFAULT')

# Aplicar Gaussian Blur de tamaño (11, 11) con sigmax = 5, sigmay = 2 y BORDER_REPLICATE
gauss = gaussian_kernel(cat, 11,11, 5, 2, cv.BORDER_REPLICATE)
visualize_image(gauss, r'$11 \times 11$ Gaussian Blur with $\sigma_x = 5$, $\sigma_y = 2$ and BORDER_REPLICATE')

# Aplicar Gaussian Blur de tamaño (11, 11) con sigmax = 2, sigmay = 5 y BORDER_REPLICATE
gauss = gaussian_kernel(cat, 11,11, 2, 5, cv.BORDER_REPLICATE)
visualize_image(gauss, r'$11 \times 11$ Gaussian Blur with $\sigma_x = 2$, $\sigma_y = 5$ and BORDER_REPLICATE')

####################################
# Aplicacion de filtros de derivadas

# Aplicar filtro de primera derivada en el eje X con tamaño 5 y BORDER_REPLICATE
der = derivative_kernel(cat, 1, 0, 5, cv.BORDER_REPLICATE)
visualize_image(der, r'$5 \times 5$ First Derivative Kernel in X-axis and BORDER_REPLICATE')

# Aplicar filtro de primera derivada en el eje Y con tamaño 5 y BORDER_REPLICATE
der = derivative_kernel(cat, 0, 1, 5, cv.BORDER_REPLICATE)
visualize_image(der, r'$5 \times 5$ First Derivative Kernel in Y-axis and BORDER_REPLICATE')

# Aplicar filtro de segunda derivada en el eje X con tamaño 5 y BORDER_REPLICATE
der = derivative_kernel(cat, 2, 0, 5, cv.BORDER_REPLICATE)
visualize_image(der, r'$5 \times 5$ Second Derivative Kernel in X-axis and BORDER_REPLICATE')

# Aplicar filtro de segunda derivada en el eje Y con tamaño 5 y BORDER_REPLICATE
der = derivative_kernel(cat, 0, 2, 5, cv.BORDER_REPLICATE)
visualize_image(der, r'$5 \times 5$ Second Derivative Kernel in Y-axis and BORDER_REPLICATE')

# Aplicar filtro de primera derivada en ambos ejes con tamaño 5 y BORDER_REPLICATE
der = derivative_kernel(cat, 1, 1, 5, cv.BORDER_REPLICATE)
visualize_image(der, r'$5 \times 5$ First Derivative Kernel in both axis and BORDER_REPLICATE')

# Aplicar filtro de segunda derivada en ambos ejes con tamaño 5 y BORDER_REPLICATE
der = derivative_kernel(cat, 2, 2, 5, cv.BORDER_REPLICATE)
visualize_image(der, r'$5 \times 5$ Second Derivative Kernel in both axis and BORDER_REPLICATE')

# Aplicar filtro de primera derivada en eje X con tamaño 7 y BORDER_REPLICATE
der = derivative_kernel(cat, 1, 0, 7, cv.BORDER_REPLICATE)
visualize_image(der, r'$7 \times 7$ First Derivative Kernel in X-axis and BORDER_REPLICATE')

# Aplicar filtro de primera derivada en eje X con tamaño 11 y BORDER_REPLICATE
der = derivative_kernel(cat, 1, 0, 11, cv.BORDER_REPLICATE)
visualize_image(der, r'$11 \times 11$ First Derivative Kernel in X-axis and BORDER_REPLICATE')

# Aplicar filtro de primera derivada en eje X con tamaño 15 y BORDER_REPLICATE
der = derivative_kernel(cat, 1, 0, 15, cv.BORDER_REPLICATE)
visualize_image(der, r'$15 \times 15$ First Derivative Kernel in X-axis and BORDER_REPLICATE')

# Aplicar filtro de primera derivada en eje X con tamaño 31 y BORDER_REFLECT
der = derivative_kernel(cat, 1, 0, 31, cv.BORDER_REFLECT)
visualize_image(der, r'$31 \times 31$ First Derivative Kernel in X-axis and BORDER_REFLECT')

# Aplicar filtro de primera derivada en eje X con tamaño 31 y BORDER_CONSTANT
der = derivative_kernel(cat, 1, 0, 31, cv.BORDER_CONSTANT)
visualize_image(der, r'$31 \times 31$ First Derivative Kernel in X-axis and BORDER_CONSTANT')

# Aplicar filtro de primera derivada en eje X con tamaño 31 y BORDER_DEFAULT
der = derivative_kernel(cat, 1, 0, 31, cv.BORDER_DEFAULT)
visualize_image(der, r'$31 \times 31$ First Derivative Kernel in X-axis and BORDER_DEFAULT')

#######################################
# Apartado B

# Laplaciana de Gaussiana 5x5 con sigma=1 en cada eje
laplace = log_kernel(cat, 5, 1, 1, cv.BORDER_REPLICATE)
visualize_image(laplace, r'$5 \times 5$ Laplacian of Gaussian with $\sigma = 1$ and BORDER_REPLICATE')

# Laplaciana de Gaussiana 5x5 con sigma=3 en cada eje
laplace = log_kernel(cat, 5, 3, 3, cv.BORDER_REPLICATE)
visualize_image(laplace, r'$5 \times 5$ Laplacian of Gaussian with $\sigma = 3$ and BORDER_REPLICATE')

# Laplaciana de Gaussiana 31x31 con sigma=1 en cada eje
laplace = log_kernel(cat, 31, 1, 1, cv.BORDER_REPLICATE)
visualize_image(laplace, r'$31 \times 31$ Laplacian of Gaussian with $\sigma = 3$ and BORDER_REPLICATE')

# Laplaciana de Gaussiana 5x5 con sigma=1 en cada eje
laplace = log_kernel(cat, 31, 3, 3, cv.BORDER_CONSTANT)
visualize_image(laplace, r'$31 \times 31$ Laplacian of Gaussian with $\sigma = 3$ and BORDER_CONSTANT')

###############################################################################
###############################################################################
# Ejercicio 2

#######################################
# Apartado A

# Obtener una piramide Gaussiana utilizando un kernel 5x5 con sigma de 3 en cada eje
# y BORDER_REPLICATE
pyr = gaussian_pyramid(cat, 5, 3, 3, cv.BORDER_REPLICATE)

# Componer la piramide en una unica imagen
pyr_img = create_img_pyramid(pyr)

# Visualizar la piramide
visualize_image(pyr_img, r'Gaussian Pyramid using a $5 \times 5$ kernel with $\sigma=3$ and BORDER_REPLICATE')

# Obtener una piramide Gaussiana utilizando un kernel 5x5 con sigma de 3 en cada eje
# y BORDER_CONSTANT
pyr = gaussian_pyramid(cat, 5, 3, 3, cv.BORDER_CONSTANT)

# Componer la piramide en una unica imagen
pyr_img = create_img_pyramid(pyr)

# Visualizar la piramide
visualize_image(pyr_img, r'Gaussian Pyramid using a $5 \times 5$ kernel with $\sigma=3$ and BORDER_CONSTANT')

#######################################
# Apartado B

# Obtener una piramide Laplaciana utilizando un kernel 5x5 con sigma 3 en cada eje
# y BORDER_REPLICATE
pyr = laplacian_pyramid(cat, 5, 3, 3, cv.BORDER_REPLICATE)

# Componer la piramide en una unica imagen
pyr_img = create_img_pyramid(pyr)

# Visualizar la piramide
visualize_image(pyr_img, r'Laplacian Pyramid using a $5 \times 5$ kernel with $\sigma=3$ and BORDER_REPLICATE')

# Obtener una piramide Laplaciana utilizando un kernel 5x5 con sigma 3 en cada eje
# y BORDER_CONSTANT
pyr = laplacian_pyramid(cat, 5, 3, 3, cv.BORDER_CONSTANT)

# Componer la piramide en una unica imagen
pyr_img = create_img_pyramid(pyr)

# Visualizar la piramide
visualize_image(pyr_img, r'Laplacian Pyramid using a $5 \times 5$ kernel with $\sigma=3$ and BORDER_CONSTANT')


#######################################
# Apartado C

# Obtener la escala Laplaciana junto con los sigma utilizados en cada nivel
# Aplicar un kernel de tamaño 5 con borde replicado, creando una escala de 5 niveles
scale, sigma = laplacian_scale_space(cat, 5, cv.BORDER_REPLICATE, 5)

# Para cada elemento del conjunto, visualizar el resultado y las regiones con circulos
# verdes, los cuales tienen una escala de 15*sigma
for img, sigma  in zip(scale, sigma):
    visualize_image(img, r'Non-max supressed image using $\sigma={}$'.format(sigma))
    visualize_laplacian_scale_space(img, sigma, r'Relevant features using $\sigma={}$'.format(sigma))


###############################################################################
###############################################################################
# Ejercicio 3

# Composicion gato-perro
hybrid = hybrid_image_generator(cat, dog, 31, 15, 5, cv.BORDER_REFLECT)
visualize_mult_images(hybrid)

pyr = gaussian_pyramid(hybrid[-1], 5, 3, 3, cv.BORDER_REFLECT)
pyr_img = create_img_pyramid(pyr)
visualize_image(pyr_img)

# Composicion Einstein-Marilyn
hybrid = hybrid_image_generator(einstein, marilyn, 9, 5, 3, cv.BORDER_REFLECT)
visualize_mult_images(hybrid)

pyr = gaussian_pyramid(hybrid[-1], 5, 3, 3, cv.BORDER_REFLECT)
pyr_img = create_img_pyramid(pyr)
visualize_image(pyr_img)

# Composicion bicicleta-moto
hybrid = hybrid_image_generator(bicycle, motorcycle, 27, 13, 5, cv.BORDER_REFLECT)
visualize_mult_images(hybrid)

pyr = gaussian_pyramid(hybrid[-1], 5, 3, 3, cv.BORDER_REFLECT)
pyr_img = create_img_pyramid(pyr)
visualize_image(pyr_img)

# Composicion ave-avion
hybrid = hybrid_image_generator(bird, plane, 17, 9, 3, cv.BORDER_REFLECT)
visualize_mult_images(hybrid)

pyr = gaussian_pyramid(hybrid[-1], 5, 3, 3, cv.BORDER_REFLECT)
pyr_img = create_img_pyramid(pyr)
visualize_image(pyr_img)

# Composicion pez-submarino
hybrid = hybrid_image_generator(fish, submarine, 23, 11, 5, cv.BORDER_REFLECT)
visualize_mult_images(hybrid)

pyr = gaussian_pyramid(hybrid[-1], 5, 3, 3, cv.BORDER_REFLECT)
pyr_img = create_img_pyramid(pyr)
visualize_image(pyr_img)

###############################################################################
###############################################################################
# BONUS

#######################################
# BONUS 1

# Hacer la convolucion con el kernel gaussiano
kx = cv.getGaussianKernel(5, 3)
ky = cv.getGaussianKernel(5, 3)
gauss = convolution(cat, kx, ky)
visualize_image(gauss, r'$5 \times 5$ Gaussian Blur with $\sigma = 3$ using convolution')

# Hacer la convolucion con el kernel de la primera derivada en el eje X
kx, ky = cv.getDerivKernels(1, 0, 5, normalize=True)
der = convolution(cat, kx, ky)
visualize_image(der, r'$5 \times 5$ First Derivative Kernel in X-axis using convolution')

#######################################
# BONUS 2

# Cargar imagenes a color
cat_color = read_image('imagenes/cat.bmp', 1)
dog_color = read_image('imagenes/dog.bmp', 1)

bird_color = read_image('imagenes/bird.bmp', 1)
plane_color = read_image('imagenes/plane.bmp', 1)

einstein_color = read_image('imagenes/einstein.bmp', 1)
marilyn_color = read_image('imagenes/marilyn.bmp', 1)

bicycle_color = read_image('imagenes/bicycle.bmp', 1)
motorcycle_color = read_image('imagenes/motorcycle.bmp', 1)

fish_color = read_image('imagenes/fish.bmp', 1)
submarine_color = read_image('imagenes/submarine.bmp', 1)

# Composicion gato-perro
hybrid = hybrid_image_generator(cat_color, dog_color, 31, 17, 15, cv.BORDER_REFLECT)
visualize_mult_images(hybrid)

pyr = gaussian_pyramid(hybrid[-1], 5, 3, 3, cv.BORDER_REFLECT)
pyr_img = create_img_pyramid(pyr)
visualize_image(pyr_img)

# Composicion Einstein-Marilyn
hybrid = hybrid_image_generator(einstein_color, marilyn_color, 9, 5, 3, cv.BORDER_REFLECT)
visualize_mult_images(hybrid)

pyr = gaussian_pyramid(hybrid[-1], 5, 3, 3, cv.BORDER_REFLECT)
pyr_img = create_img_pyramid(pyr)
visualize_image(pyr_img)

# Composicion bicicleta-moto
hybrid = hybrid_image_generator(bicycle_color, motorcycle_color, 27, 13, 5, cv.BORDER_REFLECT)
visualize_mult_images(hybrid)

pyr = gaussian_pyramid(hybrid[-1], 5, 3, 3, cv.BORDER_REFLECT)
pyr_img = create_img_pyramid(pyr)
visualize_image(pyr_img)

# Composicion ave-avion
hybrid = hybrid_image_generator(bird_color, plane_color, 17, 9, 3, cv.BORDER_REFLECT)
visualize_mult_images(hybrid)

pyr = gaussian_pyramid(hybrid[-1], 5, 3, 3, cv.BORDER_REFLECT)
pyr_img = create_img_pyramid(pyr)
visualize_image(pyr_img)

# Composicion pez-submarino
hybrid = hybrid_image_generator(fish_color, submarine_color, 23, 11, 5, cv.BORDER_REFLECT)
visualize_mult_images(hybrid)

pyr = gaussian_pyramid(hybrid[-1], 5, 3, 3, cv.BORDER_REFLECT)
pyr_img = create_img_pyramid(pyr)
visualize_image(pyr_img)

#######################################
# BONUS 3

# Leer imagenes ByN
mrbean = read_image('imagenes/mrbean.bmp', 0)
zapatero = read_image('imagenes/zapatero.bmp', 0)

# Leer imagenes a color
mrbean_color = read_image('imagenes/mrbean.bmp', 1)
zapatero_color = read_image('imagenes/zapatero.bmp', 1)

# Composición ByN
hybrid = hybrid_image_generator(mrbean, zapatero, 23, 19, 5, cv.BORDER_REFLECT)
visualize_mult_images(hybrid)

pyr = gaussian_pyramid(hybrid[-1], 5, 3, 3, cv.BORDER_REFLECT)
pyr_img = create_img_pyramid(pyr)
visualize_image(pyr_img)

# Composición color
hybrid = hybrid_image_generator(mrbean_color, zapatero_color, 23, 19, 5, cv.BORDER_REFLECT)
visualize_mult_images(hybrid)

pyr = gaussian_pyramid(hybrid[-1], 5, 3, 3, cv.BORDER_REFLECT)
pyr_img = create_img_pyramid(pyr)
visualize_image(pyr_img)
