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
from keras.layers import Conv2D, MaxPooling2D, Activation

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

    key = 'acc' if 'acc' in hist.history.keys() else 'accuracy'
    acc = hist.history[key]
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
input_shape = (32, 32, 3)
epochs = 30
batch_size = 32

# Creacion del modelo
model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5), padding='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, kernel_size=(5, 5), padding='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=50))
model.add(Activation('relu'))
model.add(Dense(units=25))
model.add(Activation('softmax'))

#########################################################################
######### DEFINICIÓN DEL OPTIMIZADOR Y COMPILACIÓN DEL MODELO ###########
#########################################################################

# Establecer optimizador a utilizar
optimizer = Adam()

# Compilar el modelo
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])

# Una vez tenemos el modelo base, y antes de entrenar, vamos a guardar los
# pesos aleatorios con los que empieza la red, para poder reestablecerlos
# después y comparar resultados entre no usar mejoras y sí usarlas.
weights = model.get_weights()

# Imprimir resumen del modelo
print(model.summary())

#########################################################################
###################### ENTRENAMIENTO DEL MODELO #########################
#########################################################################

# Entrenar el modelo
history = model.fit(x_train, y_train, validation_split=0.1, epochs=epochs,
                    batch_size=batch_size, verbose=1)

# Mostrar graficas
#mostrarEvolucion(history)

#########################################################################
################ PREDICCIÓN SOBRE EL CONJUNTO DE TEST ###################
#########################################################################

# Predecir los datos
prediction = model.predict(x_test, batch_size=batch_size, verbose=1)

# Obtener accuracy de test y mostrarla
accuracy = calcularAccuracy(y_test, prediction)
print("Test accuracy: {}".format(accuracy))

#########################################################################
########################## MEJORA DEL MODELO ############################
#########################################################################

# A completar. Tanto la normalización de los datos como el data
# augmentation debe hacerse con la clase ImageDataGenerator.
# Se recomienda ir entrenando con cada paso para comprobar
# en qué grado mejora cada uno de ellos.

# 1. Normalizacion de los datos

# Crear instancias de ImageDataGenerator, una para train y al otra para test
# Datagen de train tiene tambien split entre tain y validación
datagen_train = ImageDataGenerator(featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   validation_split=0.1)
datagen_test = ImageDataGenerator(featurewise_center=True,
                                  featurewise_std_normalization=True)

# Entrenar generadores
datagen_train.fit(x_train)
datagen_test.fit(x_train)

# Crear flow de entrenamiento y validacion
train_iter = datagen_train.flow(x_train,
                                y_train,
                                batch_size=batch_size,
                                subset="training")

validation_iter = datagen_train.flow(x_train,
                                     y_train,
                                     batch_size=batch_size,
                                     subset="validation")

# Restaurar los pesos del modelo antes de continuar
model.set_weights(weights)

# Entrenar el modelo
history = model.fit_generator(train_iter,
                              steps_per_epoch=len(train_iter),
                              epochs=epochs,
                              validation_data=validation_iter,
                              validation_steps=len(validation_iter))

# Mostrar graficas
mostrarEvolucion(history)

# Predecir los datos
prediction = model.predict_generator(datagen_test.flow(x_test, batch_size=1, shuffle=False),
                                     steps=len(x_test),
                                     verbose=1)


# Obtener accuracy de test y mostrarla
accuracy = calcularAccuracy(y_test, prediction)
print("Test accuracy: {}".format(accuracy))

# 2. Aumento de los datos
