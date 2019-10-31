# -*- coding: utf-8 -*-

#########################################################################
############ CARGAR LAS LIBRERÍAS NECESARIAS ############################
#########################################################################

# A completar: esquema disponible en las diapositivas

import numpy as np
import keras
import matplotlib.pyplot as plt
import keras.utils as np_utils

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import Adam
from keras.optimizers import SGD

from keras.datasets import cifar100

from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K


#########################################################################
######## FUNCIÓN PARA CARGAR Y MODIFICAR EL CONJUNTO DE DATOS ###########
#########################################################################

# A completar: función disponible en las diapositivas

def cargarImagenes():
    # Cargamos Cifar100. Cata imagen tiene tamaño
    # (32, 32, 3). Nos vamos a quedar con las
    # imagenes de 25 de las clases.

    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Convertir imagenes al rango [0, 1]
    x_train /= 255
    x_test /= 255

    # Escoger las 25 clases en el conjunto de entrenamiento
    train_idx = np.isin(y_train, np.arange(25))
    train_idx = np.reshape(train_idx, -1)
    x_train = x_train[train_idx]
    y_train = y_train[train_idx]

    # Escoger las 25 clases en el conjunto de test
    test_idx = np.isin(y_test, np.arange(25))
    test_idx = np.reshape(test_idx, -1)
    x_test = x_test[test_idx]
    y_test = y_test[test_idx]

    # Transformamos los vectores de clases en matrices.
    # Cada componente se convierte en un vector de ceros
    # con un uno en la componente corresponiente a la
    # clase a la que pertenece la imagen. Este paso es
    # necesario para la clasificacion multiclase en keras.

    y_train = np_utils.to_categorical(y_train, 25)
    y_test = np_utils.to_categorical(y_test, 25)

    return x_train, y_train, x_test, y_test


#########################################################################
######## FUNCIÓN PARA OBTENER EL ACCURACY DEL CONJUNTO DE TEST ##########
#########################################################################

# A completar: función disponible en las diapositivas

def calcularAccuracy(labels, preds):
    labels = np.argmax(labels, axis=1)
    preds = np.argmax(preds, axis=1)

    accuracy = sum(labels == preds) / len(labels)

    return accuracy

#########################################################################
## FUNCIÓN PARA PINTAR LA PÉRDIDA Y EL ACCURACY EN TRAIN Y VALIDACIÓN ###
#########################################################################

def mostrarEvolucion(hist):

    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training loss', 'Validation loss'])
    plt.show()

    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Training accuracy', 'Validation accuracy'])
    plt.show()

# A completar: función disponible en las diapositivas

#########################################################################
################## DEFINICIÓN DEL MODELO BASENET ########################
#########################################################################

# Cargar los datos de entrenamiento y test
x_train, y_train, x_test, y_test = cargarImagenes()

# Establecer parametros
img_rows = 32
img_cols = 32
epochs = 10
batch_size = 32

# Creacion del modelo
model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', padding='valid', input_shape=(img_rows, img_cols, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', padding='valid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=25, activation='softmax'))

#########################################################################
######### DEFINICIÓN DEL OPTIMIZADOR Y COMPILACIÓN DEL MODELO ###########
#########################################################################

# Compilar el modelo
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# Una vez tenemos el modelo base, y antes de entrenar, vamos a guardar los
# pesos aleatorios con los que empieza la red, para poder reestablecerlos
# después y comparar resultados entre no usar mejoras y sí usarlas.
weights = model.get_weights()

#########################################################################
###################### ENTRENAMIENTO DEL MODELO #########################
#########################################################################

# Crear un generador de datos para las particiones de entrenamiento y validacion
datagen = ImageDataGenerator(validation_split=0.1)

# Entrenar el modelo
history = model.fit(x_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=1)


# Imprimir resumen del modelo
print(model.summary())

# Mostrar graficas
mostrarEvolucion(history)

#########################################################################
################ PREDICCIÓN SOBRE EL CONJUNTO DE TEST ###################
#########################################################################

# A completar

#########################################################################
########################## MEJORA DEL MODELO ############################
#########################################################################

# A completar. Tanto la normalización de los datos como el data
# augmentation debe hacerse con la clase ImageDataGenerator.
# Se recomienda ir entrenando con cada paso para comprobar
# en qué grado mejora cada uno de ellos.
