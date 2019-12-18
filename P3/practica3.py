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


def apply_kernel(img, kx, ky):
    """
    Funcion que aplica un kernel separable sobre una imagen, realizando una
    convolucion

    Args:
        img: Imagen sobre la que aplicar el filtro
        kx: Kernel en el eje X
        ky: Kernel en el eje Y
    Return:
        Devuelve una imagen filtrada
    """
    # Hacer el flip a los kernels para aplicarlos como una convolucion
    kx_flip = np.flip(kx)
    ky_flip = np.flip(ky)

    # Realizar la convolucion
    conv_x = cv2.filter2D(img, -1, kx_flip.T)
    conv = cv2.filter2D(conv_x, -1, ky_flip)

    return conv


def gaussian_kernel(img, ksize, sigma):
    """
    Funcion que aplica un kernel Gaussiano sobre una imagen.

    Args:
        img: Imagen sobre la que aplicar el kernel
        ksize: Tamaño del kernel
        sigma: Valor de sigma
    Return:
        Devuelve una imagen sobre la que se ha aplicado un kernel Gaussiano
    """
    # Obtener un kernel para cada eje
    kernel_x = cv2.getGaussianKernel(ksize, sigma)
    kernel_y = cv2.getGaussianKernel(ksize, sigma)

    # Aplicar kernel Gaussiano
    gauss = apply_kernel(img, kernel_x, kernel_y)

    return gauss


def derivative_kernel(img, ksize, dx, dy):
    """
    Funcion que aplica un kernel de derivadas a una imagen.

    Args:
        img: Imagen sobre la que aplicar el kernel.
        ksize: Tamaño del kernel
        dx: Numero de derivadas que aplicar sobre el eje X.
        dy: Numero de derivadas que aplicar sobre el eje Y.
    Return:
        Devuelve una imagen sobre la que se ha aplicado el filtro de derivadas.
    """
    # Obtener los kernels que aplicar a cada eje (es descomponible porque es
    # el kernel de Sobel)
    kx, ky = cv2.getDerivKernels(dx, dy, ksize, normalize=True)

    # Aplicar los kernels sobre la imagen
    der = apply_kernel(img, kx, ky)

    return der


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

def compute_gaussian_pyramid(img, ksize, n_octaves):

    # Crear lista que contendra la piramide Gaussiana
    gauss_pyr = [img]

    # Obtener piramide
    for i in range(1, n_octaves):
        gauss_pyr.append(cv2.pyrDown(gauss_pyr[i-1]))
    
    return gauss_pyr
    

def compute_derivative_pyramids(img, ksize_der, n_octaves, sigma=4.5):
    smooth = gaussian_kernel(img, int(3*sigma) * 2 + 1, sigma)

    dx = derivative_kernel(smooth, ksize_der, 1, 0)
    dy = derivative_kernel(smooth, ksize_der, 0, 1)

    dx_pyr = [dx]
    dy_pyr = [dy]

    for i in range(1, n_octaves):
        dx_pyr.append(cv2.pyrDown(dx_pyr[i-1]))
        dy_pyr.append(cv2.pyrDown(dy_pyr[i-1]))
    
    return dx_pyr, dy_pyr


def compute_points_of_interest(img, block_size, ksize):

    # Obtener valores singulares y vectores asociados
    sv_vectors = cv2.cornerEigenValsAndVecs(img, block_size, ksize)

    # Quedarse solo con los valores singulares
    # Los valores singulares son los dos primeros valores de la matriz
    sv = sv_vectors[:, :, :2]

    # Calcular valor de cada píxel como \frac{lamb1 * lamb2}{lamb1 + lamb2}
    prod_vals = np.prod(sv, axis=2)
    sum_vals = np.sum(sv, axis=2)
    points_interest = np.divide(prod_vals, sum_vals, out=np.zeros_like(img), where=sum_vals!=0.0)

    return points_interest


def threshold_points_of_interest(points, threshold):
    points[points < threshold] = 0.0


def compute_orientation(dx_grad, dy_grad):
    # Obtener vectores u y sus normas
    u = np.concatenate([dx_grad.reshape(-1,1), dy_grad.reshape(-1,1)], axis=1)
    u_norm = np.linalg.norm(u, axis=1)

    # Calcular vectores [cos \theta, sen \theta]
    vec_cos_sen = u / u_norm.reshape(-1, 1)
    cos_vals = vec_cos_sen[:, 0]
    sen_vals = vec_cos_sen[:, 1]

    # Calcular calcular sen/cos (arreglando posibles errores como 0/0 y x/0)
    # Se arreglan los errores poniendolos a 0.0
    orientations = np.divide(sen_vals, cos_vals, out=np.zeros_like(sen_vals), where=cos_vals!=0.0)

    # Obtener \theta usando arcotangente
    orientations_rad = np.arctan(orientations)

    # Obtener angulos y arreglarlos (sumar 180º en caso de que cos < 0
    # y pasarlos al rango [0, 360], eliminando negativos)
    orientations_degrees = np.degrees(orientations_rad)
    orientations_degrees[vec_cos_sen[:, 0] < 0.0] += 180.0
    orientations_degrees[orientations_degrees < 0.0] += 360.0
    
    return orientations_degrees


def harris_corner_detection(img, block_size, window_size, ksize, ksize_der, n_octaves):

    # Obtener piramide gaussiana de la imagen
    img_pyr = compute_gaussian_pyramid(img, ksize, n_octaves)

    # Obtener piramides de las derivadas
    dx_pyr, dy_pyr = compute_derivative_pyramids(img, ksize_der, n_octaves)

    keypoints = []

    for i in range(n_octaves):
        # Obtener puntos de interes de la escala
        points_interest = compute_points_of_interest(img_pyr[i], block_size, ksize)

        # Aplicar umbralizacion
        threshold_points_of_interest(points_interest, 10.0)

        # Aplicar supresion de no maximos
        points_interest = non_max_supression(points_interest, window_size)

        # Obtener valores mayores que 0.0 (aquellos que no han sido eliminados)
        points_idx = np.where(points_interest > 0.0)

        # Calcular escala del KeyPoint
        # Hace falta incrementar el valor de i en 1 porque se empieza en 0
        scale = (i+1) * block_size

        dx_grad = dx_pyr[i][points_idx]
        dy_grad = dy_pyr[i][points_idx]

        orientations = compute_orientation(dx_grad, dy_grad)

        scale_keypoints = []

        for y, x, o in zip(*points_idx, orientations):
            scale_keypoints.append(cv2.KeyPoint(x*2**i, y*2**i, scale, o))
        
        keypoints.append(scale_keypoints)

    return keypoints


def draw_keypoints(img, keypoints):
    keypoints_list = [k for sublist in keypoints for k in sublist]

    vis = transform_img_uint8(img)
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    out = np.empty_like(vis)

    out = cv2.drawKeypoints(vis, keypoints_list, out, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Visualizar la imagen
    plt.imshow(out)
    plt.axis('off')

    plt.show()



###############################################################################
###############################################################################

# Cargar imagen de Yosemite
yosemite = read_image('imagenes/Yosemite1.jpg', 0)
yosemite_color = read_image('imagenes/Yosemite1.jpg', 1)



pi = harris_corner_detection(yosemite, block_size=5, window_size=3, ksize=3, ksize_der=3, n_octaves=5)
draw_keypoints(yosemite_color, pi)

#visualize_image(pi)