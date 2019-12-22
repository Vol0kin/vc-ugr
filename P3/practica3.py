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


def transform_img_uint8_RGB(img):
    """
    Funcion que transforma una imagen float32 a uint8 y a RGB

    Args:
        img: Imagen a transformar
    Return:
        Devuelve la imagen en el rango [0, 255] y RGB
    """
    # Transformar a uint8
    img_uint8 = transform_img_uint8(img)

    # Pasar a RGB
    img_uint8_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)

    return img_uint8_rgb


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
    # Visualizar la imagen
    plt.imshow(img)
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

def compute_gaussian_pyramid(img, n_octaves):
    """
    Funcion que permite calcular una piramide Gaussiana de n_octaves escalas

    Args:
        img: Imagen de la que extraer la piramide
        n_octaves: Numero de octavas que tiene que tener la piramide
    Return:
        Devuleve la piramide Gaussiana de la imagen de entrada
    """
    # Crear lista que contendra la piramide Gaussiana
    # Inicialmente contiene la imagen de entrada (el nivel mas bajo)
    gauss_pyr = [img]

    # Obtener piramide
    for i in range(1, n_octaves):
        gauss_pyr.append(cv2.pyrDown(gauss_pyr[i-1]))
    
    return gauss_pyr
    

def compute_derivative_pyramids(img, ksize_der, n_octaves, sigma=4.5):
    """
    Funcion que calcula las piramides Gaussianas de las derivadas en los
    ejes X e Y dada una imagen de entrada. La imagen de entrada ese alisada
    inicialmente con un filtro Gaussiano de sigma 4.5

    Args:
        img: Imagen de la que extraer las piramides de las derivadas
        ksize_der: Tamaño del kernel de la derivada
        n_octaves: Numero de imagenes que compondran las piramides
        sigma: Sigma del alisamiento Gaussiano (default: 4.5)
    Return:
        Devuelve dos listas, una para la piramide de las derivadas en el eje
        X y otra para la piramide de las derivadas en el eje Y
    """
    # Aplicar alisamiento Gaussiano
    smooth = gaussian_kernel(img, int(3*sigma) * 2 + 1, sigma)

    # Calcular derivadas
    dx = derivative_kernel(smooth, ksize_der, 1, 0)
    dy = derivative_kernel(smooth, ksize_der, 0, 1)

    # Añadir derivadas a las correspondiendtes listas
    dx_pyr = [dx]
    dy_pyr = [dy]

    # Crear piramide
    for i in range(1, n_octaves):
        dx_pyr.append(cv2.pyrDown(dx_pyr[i-1]))
        dy_pyr.append(cv2.pyrDown(dy_pyr[i-1]))
    
    return dx_pyr, dy_pyr


def compute_points_of_interest(img, block_size, ksize):
    """
    Funcion que calucla los puntos de interes dada una imagen de entrada.
    Dichos puntos de interes son calculados mediante el operador de Harris.
    
    Args:
        img: Imagen de la que sacar los puntos de interes
        block_size: Tamaño del bloque que se va a tener en cuenta a la hora de
                    calcular los valores singulares.
        ksize: Tamaño del operador de Sobel
    Return:
        Devuelve una imagen del mismo tamaño que la entrada que contiene los
        puntos de interes calculados con el operador de Harris
    """
    # Obtener valores singulares y vectores asociados
    sv_vectors = cv2.cornerEigenValsAndVecs(img, block_size, ksize)

    # Quedarse solo con los valores singulares
    # Los valores singulares son los dos primeros valores de la matriz
    sv = sv_vectors[:, :, :2]

    # Calcular valor de cada píxel como \frac{lamb1 * lamb2}{lamb1 + lamb2}
    # Ahí donde el denominador sea 0, se pone un 0, para evitar que se calcule
    # un infinito
    prod_vals = np.prod(sv, axis=2)
    sum_vals = np.sum(sv, axis=2)
    points_interest = np.divide(prod_vals, 
        sum_vals,
        out=np.zeros_like(img),
        where=sum_vals!=0.0
    )

    return points_interest


def threshold_points_of_interest(points, threshold):
    """
    Funcion que aplica un umbral sobre una imagen, poniendo los pixels por
    debajo del umbral a 0

    Args:
        points: Puntos/Imagen sobre la que aplicar la umbralizacion
        threshold: Valor umbral
    Return:
        Devuelve una imagen en la que los valores por debajo del umbral han
        sido puestos a 0
    """
    points[points < threshold] = 0.0


def compute_orientation(dx_grad, dy_grad):
    """
    Funcion que calcula la orientacion del gradiente de una serie de puntos

    Args:
        dx_grad: Derivadas en el eje X
        dy_grad: Derivadas en el eje Y
    Return:
        Devuelve un array en el que estan las orientaciones de todos los
        pares de gradientes de dx_grad y dy_grad. Las orientaciones estan
        en grados, y se encuentran en el rango [0, 360)
    """
    # Obtener vectores u y sus normas
    u = np.concatenate([dx_grad.reshape(-1,1), dy_grad.reshape(-1,1)], axis=1)
    u_norm = np.linalg.norm(u, axis=1)

    # Calcular vectores [cos \theta, sen \theta]
    vec_cos_sen = u / u_norm.reshape(-1, 1)
    cos_vals = vec_cos_sen[:, 0]
    sen_vals = vec_cos_sen[:, 1]

    # Calcular sen/cos arreglando posibles errores como 0/0 y x/0
    # Se arreglan los errores poniendolos a 0.0
    orientations = np.divide(sen_vals,
        cos_vals,
        out=np.zeros_like(sen_vals),
        where=cos_vals!=0.0
    )

    # Obtener \theta usando arcotangente (resultado en radianes
    # entre [-pi/2, pi/2])
    orientations_rad = np.arctan(orientations)

    # Obtener angulos y arreglarlos (sumar 180º en caso de que cos < 0
    # y pasarlos al rango [0, 360], eliminando negativos)
    orientations_degrees = np.degrees(orientations_rad)
    orientations_degrees[cos_vals < 0.0] += 180.0
    orientations_degrees[orientations_degrees < 0.0] += 360.0
    
    return orientations_degrees


def harris_corner_detection(img, block_size, window_size, ksize_der,
                            n_octaves, threshold=10.0):
    """
    Funcion que detecta los puntos de Harris de una imagen a distintas
    escaslas.

    Args:
        img: Imagen de la que se quieren extraer los puntos de Harris
        block_size: Tamaño del bloque que se va a tener en cuenta a la hora de
                    calcular los valores singulares.
        window_size: Tamaño de la ventana al realizar la supresion de no
                     maximos
        ksize_der: Tamaño del operador de Sobel (utilizado en el calculo
                   de los valores singulares)
        n_octaves: Numero de octavas/escalas de la imagen de la que sacar
                   puntos
        threshold: Umbral utilizado para eliminar todos los valores inferiores
                   a este.
    Return:
        Devuelve dos listas: una que contiene los keypoints extraidos y otra
        que contiene los keypoints corregidos
    """
    # Obtener piramide gaussiana de la imagen
    img_pyr = compute_gaussian_pyramid(img, n_octaves)

    # Obtener piramides de las derivadas
    dx_pyr, dy_pyr = compute_derivative_pyramids(img, ksize_der, n_octaves)

    # Lista de keypoints y keypoints corregidos
    keypoints = []
    corrected_keypoints = []

    for i in range(n_octaves):
        # Obtener puntos de interes de la escala
        points_interest = compute_points_of_interest(img_pyr[i],
            block_size,
            ksize_der
        )

        # Aplicar umbralizacion
        threshold_points_of_interest(points_interest, threshold)

        # Aplicar supresion de no maximos
        points_interest = non_max_supression(points_interest, window_size)

        # Obtener valores mayores que 0.0 (aquellos que no han sido eliminados)
        points_idx = np.where(points_interest > 0.0)

        # Calcular escala del KeyPoint
        # Hace falta incrementar el valor de i en 1 porque se empieza en 0
        scale = (i+1) * block_size

        # Obtener las derivadas correspondientes a los puntos no eliminados
        dx_grad = dx_pyr[i][points_idx]
        dy_grad = dy_pyr[i][points_idx]

        # Calcular orientaciones de los puntos no eliminados
        orientations = compute_orientation(dx_grad, dy_grad)

        # Lista que contiene los keypoints de la octava/escala
        # Se corrigen las coordenadas segun la escala
        keypoints_octave = [cv2.KeyPoint(x*2**i, y*2**i, scale, o)
                            for y, x, o in zip(*points_idx, orientations)]

        # Unir las coordenadas de forma que sean n vectores [x,y] formando una  
        # matriz
        points_x = points_idx[0].reshape(-1,1)
        points_y = points_idx[1].reshape(-1,1)
        points = np.concatenate([points_x, points_y], axis=1)

        # Establecer criterio de parada
        # Se parará o bien a las 15 iteraciones o cuando epsilon sea menor a 0.01
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.01)

        # Corregir keypoints
        points = cv2.cornerSubPix(img_pyr[i],
            points.astype(np.float32),
            (3,3),
            (-1,-1),
            criteria
        )

        # Redondear, cambiar x por y y viceversa (OpenCV carga las imagenes
        # invirtiendo los ejes) y transformar coordenada a la de la imagen original
        points = np.round(points)
        points = np.flip(points, axis=1)
        points *= 2**i

        # Guardar keypoints y keypoints corregidos
        keypoints.append(keypoints_octave)
        corrected_keypoints.append(points)

    return keypoints, corrected_keypoints


def compute_number_keypoints(keypoints_list):
    """
    Funcion que calcula el numero de keypoints detectados en una imagen para
    todas las escalas y lo muestra por pantalla

    Args:
        keypoints_list: Lista con los keypoints
    """
    num_kp = 0

    for kp in keypoints_list:
        num_kp += len(kp)
    
    print(f"Number of keypoints found across all scales: {num_kp}")


def draw_all_keypoints(img, keypoint_list):
    """
    Funcion que dibuja todos los keypoints detectados

    Args:
        img: Imagen sobre la que pintar los keypoints
        keypoint_list: Lista con los keypoints
    """
    # Juntar los keypoints de todas las escalas
    keypoints= [k for keypoints_octave in keypoint_list for k in keypoints_octave]

    # Transformar imagen a uint8 y RGB
    vis = transform_img_uint8_RGB(img)

    # Crear imagen de salida vacía del mismo tamaño que la original
    out = np.empty_like(vis)

    # Dibujar keypoints
    out = cv2.drawKeypoints(vis,
        keypoints,
        out,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Visualizar la imagen
    visualize_image(out)

def draw_keypoints_octave(img, keypoint_list):
    """
    Funcion que dibuja los keypoints por cada escala

    Args:
        img: Imagen sobre la que pintar los keypoints
        keypoint_list: Lista con los keypoints
    """
    # Transformar imagen a uint8 y RGB
    vis = transform_img_uint8_RGB(img)

    for keypoints in keypoint_list:
        # Crear imagen de salida vacía del mismo tamaño que la original
        out = np.empty_like(vis)

        # Dibujar keypoints de la octava
        out = cv2.drawKeypoints(vis,
            keypoints,
            out,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        # Visualizar la imagen
        visualize_image(out)
    

def compare_keypoints_orig_corrected(img, keypoints, corrected):
    # Transformar datos de entrada a matrices de numpy
    keypoints_coord = np.array([list(k.pt) for keys in keypoints for k in keys])
    corrected_coord = np.array([c for correct_octave in corrected for c in correct_octave])

    # Encontrar los índices de los keypoints que difieren
    same_x = keypoints_coord[:, 0] == corrected_coord[:, 0]
    same_y = keypoints_coord[:, 1] == corrected_coord[:, 1]
    same_values = same_x * same_y
    idx_different_vals = np.where(same_values == False)

    # Escoger keypoints que mostrar (escoger sin repetición)
    random_keypoints = np.random.choice(idx_different_vals[0], 3, replace=False)

    # Normalizar imagen y pasarla a RGB
    vis = transform_img_uint8_RGB(img)    

    for idx in random_keypoints:
        # Obtener píxeles donde el keypoint original y el corregido no coinciden
        x_kp, y_kp  = keypoints_coord[idx].astype(np.int)
        x_c_kp, y_c_kp = corrected_coord[idx].astype(np.int)

        # Obtener centro del circulo del keypoint corregido
        corr_center_x = x_c_kp - x_kp + 5
        corr_center_y = y_c_kp - y_kp + 5

        # Obtener la region 11 \times 11
        region = np.copy(vis[x_kp-5:x_kp+6, y_kp-5:y_kp+6])

        # Dibujar círculos alrededor del keypoint y el corregido
        region = cv2.circle(region, (5,5), 2, (0,255,0))
        region = cv2.circle(region, (corr_center_x, corr_center_y), 2, (255, 0, 0))

        # Hacer un resize de la region
        region = cv2.resize(region, None, fx=5, fy=5)

        # Mostrar imagen
        visualize_image(region)


###############################################################################
#                       Apartado 2: Descriptor AKAZE                          #
###############################################################################

def compute_akaze_keypoints_descriptors(img, threshold=0.1):
    # Crear objeto AKAZE para extraer las caracteristicas
    akaze = cv2.AKAZE_create(threshold=threshold)

    # Extraer keypoints y descriptores
    keypoints, descriptors = akaze.detectAndCompute(img, None)

    return keypoints, descriptors


def brute_force_crosscheck_matcher(desc1, desc2):

    # Crear matcher (va a utilizar el método Brute Force + Cross Check)
    bf = cv2.BFMatcher_create(crossCheck=True)

    # Obtener los matches
    matches = bf.match(desc1, desc2)

    return matches


def nn2_matcher(desc1, desc2):

    # Crear matcher
    knn = cv2.BFMatcher_create()

    # Obtener matches
    matches = knn.knnMatch(desc1, desc2, k=2)

    return matches


def lowe_average_2nn_matcher(desc1, desc2):

    # Obtener matches
    matches = nn2_matcher(desc1, desc2)

    # Quedarse el mejor match según el criterio de Lowe
    # El mejor match es m1 sii m1 < 0.8 * m2
    lowe_matches = [m1 for m1, m2 in matches if m1.distance < 0.8* m2.distance]

    return lowe_matches


def draw_matches(img1, img2, keypoints1, keypoints2, matches):
    # Transformar imagen a uint8 y RGB
    vis1 = transform_img_uint8_RGB(img1)
    vis2 = transform_img_uint8_RGB(img2)

    rand_matches = np.random.choice(matches, 100, replace=False)

    out_img = np.concatenate([vis1, vis2], axis=1)

    out_img = cv2.drawMatches(vis1,
        keypoints1,
        vis2,
        keypoints2,
        rand_matches,
        out_img,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Mostrar imagen
    visualize_image(out_img)


###############################################################################
#                      Apartado 3: Mosaicos para 2 imagenes                   #
###############################################################################

def generate_canvas(width, height):
    # Generar matriz de 0's
    canvas = np.zeros((height, width), dtype=np.uint8)

    return canvas


def draw_panorama_2_images(img1, img2, canv_width, canv_height):

    # Obtener keypoints y descriptores utilizando Lowe
    kp_img1, desc_img1 = compute_akaze_keypoints_descriptors(img1)
    kp_img2, desc_img2 = compute_akaze_keypoints_descriptors(img2)

    # Obtener matches utilizando Lowe
    matches = lowe_average_2nn_matcher(desc_img1, desc_img2)

    # Obtener coordenadas de los keypoints de los matches
    kp_match1 = np.array([kp_img1[m.queryIdx].pt for m in matches],
        dtype=np.float32
    )

    kp_match2 = np.array([kp_img2[m.trainIdx].pt for m in matches],
        dtype=np.float32
    )

    # Obtener homografia usando RANSAC con threshold de 5
    # Se recomienda que el threshold este en el rango [1,10]
    homo, _ = cv2.findHomography(kp_match2, kp_match1, cv2.RANSAC, 5)


    # Crear canvas en negro donde se pintara el mosaico
    canvas = generate_canvas(canv_width, canv_height)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # Crear homografia al canvas
    homo_canvas = np.array([[1, 0, 500],
                            [0, 1, 300],
                            [0, 0, 1]], 
                            dtype=np.float64
    )

    # Transformar imagen a uint8 y RGB
    vis1 = transform_img_uint8_RGB(img1)
    vis2 = transform_img_uint8_RGB(img2)

    # Crear mosaico juntando imagenes
    cv2.warpPerspective(vis1,
        homo_canvas,
        (canv_width, canv_height),
        dst=canvas,
        borderMode = cv2.BORDER_TRANSPARENT
    )
    
    # En esta parte se componen las transformaciones
    cv2.warpPerspective(vis2,
        np.dot(homo_canvas, homo),
        (canv_width, canv_height),
        dst=canvas,
        borderMode = cv2.BORDER_TRANSPARENT
    )

    # Mostrar canvas
    visualize_image(canvas)


###############################################################################
#                      Apartado 4: Mosaicos para N imagenes                   #
###############################################################################

def draw_panorama_N_images(image_list, canv_width, canv_height, canv_homo_x, canv_homo_y):
    # Obtener lista con keypoints y descriptores para las imagenes
    #kp_desc_list = [compute_akaze_keypoints_descriptors(img) for img in image_list]

    # Determinar la imagen central
    center_idx = len(image_list) // 2

    # Crear canvas en negro donde se pintara el mosaico
    canvas = generate_canvas(canv_width, canv_height)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # Crear homografia al canvas
    homo_canvas = np.array([[1, 0, canv_homo_x],
                            [0, 1, canv_homo_y],
                            [0, 0, 1]], 
                            dtype=np.float64
    )

    # Transformar  central imagen a uint8 y RGB
    center = transform_img_uint8_RGB(image_list[center_idx])

    # Poner imagen central en el mosaico
    cv2.warpPerspective(center,
        homo_canvas,
        (canv_width, canv_height),
        dst=canvas,
        borderMode = cv2.BORDER_TRANSPARENT
    )

    homo_product = np.copy(homo_canvas)

    # Componer parte derecha del mosaico
    for i in range(center_idx, len(image_list)-1):
        # Obtener imagenes fuente y destino
        dst_img = image_list[i]
        src_img = image_list[i+1]
 
        # Obtener keypoints y descriptores utilizando Lowe
        kp_dst, desc_dst = compute_akaze_keypoints_descriptors(dst_img)
        kp_src, desc_src = compute_akaze_keypoints_descriptors(src_img)

        # Obtener matches utilizando Lowe
        matches = lowe_average_2nn_matcher(desc_dst, desc_src)

        # Obtener coordenadas de los keypoints de los matches
        kp_match_dst = np.array([kp_dst[m.queryIdx].pt for m in matches],
            dtype=np.float32
        )
        
        kp_match_src = np.array([kp_src[m.trainIdx].pt for m in matches],
            dtype=np.float32
        )


        # Obtener homografia usando RANSAC con threshold de 5
        # Se recomienda que el threshold este en el rango [1,10]
        homo, _ = cv2.findHomography(kp_match_src, kp_match_dst, cv2.RANSAC, 5)
        homo_product = np.dot(homo_product, homo)

        vis = transform_img_uint8_RGB(src_img)

        cv2.warpPerspective(vis,
            homo_product,
            (canv_width, canv_height),
            dst=canvas,
            borderMode=cv2.BORDER_TRANSPARENT
        )
    
    homo_product = np.copy(homo_canvas)
    
    # Componer parte izquierda del mosaico
    for i in reversed(range(1, center_idx+1)):
        # Obtener imagenes fuente y destino
        dst_img = image_list[i]
        src_img = image_list[i-1]
 
        # Obtener keypoints y descriptores utilizando Lowe
        kp_dst, desc_dst = compute_akaze_keypoints_descriptors(dst_img)
        kp_src, desc_src = compute_akaze_keypoints_descriptors(src_img)

        # Obtener matches utilizando Lowe
        matches = lowe_average_2nn_matcher(desc_dst, desc_src)

        # Obtener coordenadas de los keypoints de los matches
        kp_match_dst = np.array([kp_dst[m.queryIdx].pt for m in matches],
            dtype=np.float32
        )
        
        kp_match_src = np.array([kp_src[m.trainIdx].pt for m in matches],
            dtype=np.float32
        )


        # Obtener homografia usando RANSAC con threshold de 5
        # Se recomienda que el threshold este en el rango [1,10]
        homo, _ = cv2.findHomography(kp_match_src, kp_match_dst, cv2.RANSAC, 5)

        homo_product = np.dot(homo_product, homo)

        vis = transform_img_uint8_RGB(src_img)

        cv2.warpPerspective(vis,
            homo_product,
            (canv_width, canv_height),
            dst=canvas,
            borderMode=cv2.BORDER_TRANSPARENT
        )

    # Mostrar canvas
    visualize_image(canvas)
    

###############################################################################
###############################################################################
# Inicializar semilla aleatoria
np.random.seed(1)

# Cargar imagen de Yosemite
yosemite = read_image('imagenes/Yosemite1.jpg', 0)
yosemite_color = read_image('imagenes/Yosemite1.jpg', 1)

# Caso base
keypoints_list1, corrected_keypoints1 = harris_corner_detection(
    yosemite,
    block_size=5,
    window_size=3,
    ksize_der=3, 
    n_octaves=5,
    threshold=10.0
)

compute_number_keypoints(keypoints_list1)
draw_all_keypoints(yosemite_color, keypoints_list1)

# Ventana = 5
keypoints_list2, corrected_keypoints2 = harris_corner_detection(
    yosemite,
    block_size=5,
    window_size=5,
    ksize_der=3, 
    n_octaves=5,
    threshold=10.0
)

compute_number_keypoints(keypoints_list2)
draw_all_keypoints(yosemite_color, keypoints_list2)

# Tamaño kernel derivada = 5
keypoints_list3, corrected_keypoints3 = harris_corner_detection(
    yosemite,
    block_size=5,
    window_size=3,
    ksize_der=5, 
    n_octaves=5,
    threshold=10.0
)

compute_number_keypoints(keypoints_list3)
draw_all_keypoints(yosemite_color, keypoints_list3)

# Numero bloques = 3
keypoints_list4, corrected_keypoints4 = harris_corner_detection(
    yosemite,
    block_size=3,
    window_size=3,
    ksize_der=3, 
    n_octaves=5,
    threshold=10.0
)

compute_number_keypoints(keypoints_list4)
draw_all_keypoints(yosemite_color, keypoints_list4)

# Umbral = 60
keypoints_list5, corrected_keypoints5 = harris_corner_detection(
    yosemite,
    block_size=5,
    window_size=3,
    ksize_der=3, 
    n_octaves=5,
    threshold=60.0
)

compute_number_keypoints(keypoints_list5)
draw_all_keypoints(yosemite_color, keypoints_list5)

# Umbral = 90
keypoints_list6, corrected_keypoints6 = harris_corner_detection(
    yosemite,
    block_size=5,
    window_size=3,
    ksize_der=3, 
    n_octaves=5,
    threshold=90.0
)

compute_number_keypoints(keypoints_list6)
draw_all_keypoints(yosemite_color, keypoints_list6)

# Tamaño ventana = 5 y Umbral = 90
keypoints_list7, corrected_keypoints7 = harris_corner_detection(
    yosemite,
    block_size=5,
    window_size=5,
    ksize_der=3, 
    n_octaves=5,
    threshold=90.0
)

compute_number_keypoints(keypoints_list7)
draw_all_keypoints(yosemite_color, keypoints_list7)

# Mostrar keypoints para cada octava para el ultimo caso
draw_keypoints_octave(yosemite_color, keypoints_list7)

#compare_keypoints_orig_corrected(yosemite_color, keypoints_list, corrected_keypoints)

# Apartado 2
yosemite1 = read_image('imagenes/Yosemite1.jpg', 0)
yosemite2 = read_image('imagenes/Yosemite2.jpg', 0)

yosemite1_c = read_image('imagenes/Yosemite1.jpg', 1)
yosemite2_c = read_image('imagenes/Yosemite2.jpg', 1)

# Extraer descriptores
kp_yosemite1, desc_yosemite1 = compute_akaze_keypoints_descriptors(yosemite1)
kp_yosemite2, desc_yosemite2 = compute_akaze_keypoints_descriptors(yosemite2)

matches_bf_xcheck = brute_force_crosscheck_matcher(desc_yosemite1, desc_yosemite2)
matches_lowe = lowe_average_2nn_matcher(desc_yosemite1, desc_yosemite2)

#draw_matches(yosemite1_c, yosemite2_c, kp_yosemite1, kp_yosemite2, matches_bf_xcheck)
#draw_matches(yosemite1_c, yosemite2_c, kp_yosemite1, kp_yosemite2, matches_lowe)

# Apartado 3


#draw_panorama_2_images(yosemite1_c, yosemite2_c, 1920, 1080)

board1 = read_image('imagenes/yosemite6.jpg', 0)
board2 = read_image('imagenes/yosemite7.jpg', 0)

#draw_panorama_2_images(board1, board2, 1920, 1080)


# Apartado 4
yosemite_names1 = [f"imagenes/yosemite{n}.jpg" for n in range(1,5)]
yosemite_names2 = [f"imagenes/yosemite{n}.jpg" for n in range(5, 8)]
etsiit_names = [f"imagenes/mosaico00{num}.jpg" for num in range(2,10)]
etsiit_names += [f"imagenes/mosaico0{num}.jpg" for num in range(10, 12)]

yosemite_images1 = [read_image(img, 1) for img in yosemite_names1]
yosemite_images2 = [read_image(img, 1) for img in yosemite_names2]
etsiit_images = [read_image(etsiit, 1) for etsiit in etsiit_names]

draw_panorama_N_images(yosemite_images1, 2048, 750, 800, 128)
draw_panorama_N_images(yosemite_images2, 3000, 1700, 2000, 500)
draw_panorama_N_images(etsiit_images, 950, 500, 350, 100)
