# -*- coding: utf-8 -*-

#########################################################################
############ CARGAR LAS LIBRERÍAS NECESARIAS ############################
#########################################################################

import numpy as np
import keras
import matplotlib.pyplot as plt
import keras.utils as np_utils

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization

from keras.optimizers import Adam
from keras.optimizers import SGD

from keras.datasets import cifar100

from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K


#########################################################################
######## FUNCIÓN PARA CARGAR Y MODIFICAR EL CONJUNTO DE DATOS ###########
#########################################################################

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

    key_acc = 'acc' if 'acc' in hist.history.keys() else 'accuracy'
    key_val_acc = 'val_acc' if 'val_acc' in hist.history.keys() else 'val_accuracy'

    acc = hist.history[key_acc]
    val_acc = hist.history[key_val_acc]
    
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Training accuracy', 'Validation accuracy'])
    plt.show()


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
optimizer = SGD()

# Compilar el modelo
model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=optimizer,
    metrics=['accuracy']
)

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
print('Training base model')
history = model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)

# Mostrar graficas
mostrarEvolucion(history)

#########################################################################
################ PREDICCIÓN SOBRE EL CONJUNTO DE TEST ###################
#########################################################################

# Predecir los datos
prediction = model.predict(
    x_test,
    batch_size=batch_size,
    verbose=1
)

# Obtener accuracy de test y mostrarla
accuracy = calcularAccuracy(y_test, prediction)
print('Test accuracy: {}'.format(accuracy))

#########################################################################
########################## MEJORA DEL MODELO ############################
#########################################################################

# A completar. Tanto la normalización de los datos como el data
# augmentation debe hacerse con la clase ImageDataGenerator.
# Se recomienda ir entrenando con cada paso para comprobar
# en qué grado mejora cada uno de ellos.

# 1. Normalizacion de los datos

# Crear instancias de ImageDataGenerator, una para train
# Datagen de train tiene tambien split entre tain y validación
datagen_train = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    validation_split=0.1
)


# Entrenar generador
datagen_train.fit(x_train)

# Crear flow de entrenamiento y validacion
train_iter = datagen_train.flow(
    x_train,
    y_train,
    batch_size=batch_size,
    subset='training'
)

validation_iter = datagen_train.flow(
    x_train,
    y_train,
    batch_size=batch_size,
    subset='validation'
)

# Restaurar los pesos del modelo antes de continuar
model.set_weights(weights)

# Entrenar el modelo
print('Training normalization')
history = model.fit_generator(
    train_iter,
    steps_per_epoch=len(x_train)*0.9/batch_size,
    epochs=epochs,
    validation_data=validation_iter,
    validation_steps=len(x_train)*0.1/batch_size
)

# Mostrar graficas
mostrarEvolucion(history)

###############################################################################
# 2. Aumento de los datos

# Aumentar el numero de epocas para ver mejor la evolucion
epochs = 50

# Datagen con flip horizontal
datagen_train_flip = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    validation_split=0.1,
    horizontal_flip=True
)

# Entrenar generador con datos de train
datagen_train_flip.fit(x_train)

# Crear flow de entrenamiento y validacion
train_iter_flip = datagen_train_flip.flow(
    x_train,
    y_train,
    batch_size=batch_size,
    subset='training'
)

validation_iter_flip = datagen_train_flip.flow(
    x_train,
    y_train,
    batch_size=batch_size,
    subset='validation'
)

# Restaurar los pesos del modelo antes de continuar
model.set_weights(weights)

# Entrenar el modelo
print('Training with data augmentation: horizontal_flip=True')
history = model.fit_generator(
    train_iter_flip,
    steps_per_epoch=len(x_train)*0.9/batch_size,
    epochs=epochs,
    validation_data=validation_iter_flip,
    validation_steps=len(x_train)*0.1/batch_size
)

# Mostrar graficas
mostrarEvolucion(history)

# Datagen con zoom de 0.2
datagen_train_zoom = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    validation_split=0.1,
    zoom_range=0.2
)

# Entrenar generador con datos de train
datagen_train_zoom.fit(x_train)

# Crear flow de entrenamiento y validacion
train_iter_zoom = datagen_train_zoom.flow(
    x_train,
    y_train,
    batch_size=batch_size,
    subset='training'
)

validation_iter_zoom = datagen_train_zoom.flow(
    x_train,
    y_train,
    batch_size=batch_size,
    subset='validation'
)

# Restaurar los pesos del modelo antes de continuar
model.set_weights(weights)

# Entrenar el modelo
print('Training with data augmentation: zoom_range=0.2')
history = model.fit_generator(
    train_iter_zoom,
    steps_per_epoch=len(x_train)*0.9/batch_size,
    epochs=epochs,
    validation_data=validation_iter_zoom,
    validation_steps=len(x_train)*0.1/batch_size
)

# Mostrar graficas
#mostrarEvolucion(history)

# Datagen con rotacion de 25
datagen_train_rot = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    validation_split=0.1,
    rotation_range=25
)

# Entrenar generador con datos de train
datagen_train_rot.fit(x_train)

# Crear flow de entrenamiento y validacion
train_iter_rot = datagen_train_rot.flow(
    x_train,
    y_train,
    batch_size=batch_size,
    subset='training'
)

validation_iter_rot = datagen_train_rot.flow(
    x_train,
    y_train,
    batch_size=batch_size,
    subset='validation'
)

# Restaurar los pesos del modelo antes de continuar
model.set_weights(weights)

# Entrenar el modelo
print('Training with data augmentation: rotation_range=25')
history = model.fit_generator(
    train_iter_rot,
    steps_per_epoch=len(x_train)*0.9/batch_size,
    epochs=epochs,
    validation_data=validation_iter_rot,
    validation_steps=len(x_train)*0.1/batch_size
)

# Mostrar graficas
#mostrarEvolucion(history)

# Datagen con flip horizontal y zoom de 0.2
datagen_train_fz = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    validation_split=0.1,
    horizontal_flip=True,
    zoom_range=0.2
)

# Entrenar generador con datos de train
datagen_train_fz.fit(x_train)

# Crear flow de entrenamiento y validacion
train_iter_fz = datagen_train_fz.flow(
    x_train,
    y_train,
    batch_size=batch_size,
    subset='training'
)

validation_iter_fz = datagen_train_fz.flow(
    x_train,
    y_train,
    batch_size=batch_size,
    subset='validation'
)

# Restaurar los pesos del modelo antes de continuar
model.set_weights(weights)

# Entrenar el modelo
print('Training with data augmentation: horizontal_flip=True, zoom_range=0.2')
history = model.fit_generator(
    train_iter_fz,
    steps_per_epoch=len(x_train)*0.9/batch_size,
    epochs=epochs,
    validation_data=validation_iter_fz,
    validation_steps=len(x_train)*0.1/batch_size
)

# Mostrar graficas
mostrarEvolucion(history)

###############################################################################
# 3. Red más profunda

epochs = 35

# Definicion del nuevo modelo
model_v2 = Sequential()
model_v2.add(Conv2D(8, kernel_size=(5, 5), padding='valid', input_shape=input_shape))
model_v2.add(Activation('relu'))
model_v2.add(Conv2D(16, kernel_size=(5, 5), padding='valid'))
model_v2.add(Activation('relu'))
model_v2.add(MaxPooling2D(pool_size=(2, 2)))

model_v2.add(Conv2D(32, kernel_size=(3, 3), padding='valid'))
model_v2.add(Activation('relu'))
model_v2.add(Conv2D(64, kernel_size=(3, 3), padding='valid'))
model_v2.add(Activation('relu'))
model_v2.add(MaxPooling2D(pool_size=(2, 2)))

model_v2.add(Flatten())
model_v2.add(Dense(units=128))
model_v2.add(Activation('relu'))
model_v2.add(Dense(units=50))
model_v2.add(Activation('relu'))
model_v2.add(Dense(units=25))
model_v2.add(Activation('softmax'))

# Compilar el modelo
model_v2.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=optimizer,
    metrics=['accuracy']
)

weights_v2 = model_v2.get_weights()

print(model_v2.summary())


# Entrenar el modelo
print('Training first "enhanced" model with no Dropout')
history = model_v2.fit_generator(
    train_iter,
    steps_per_epoch=len(x_train)*0.9/batch_size,
    epochs=epochs,
    validation_data=validation_iter,
    validation_steps=len(x_train)*0.1/batch_size
)

# Mostrar graficas
mostrarEvolucion(history)

# Definicion del nuevo modelo
model_v3 = Sequential()
model_v3.add(Conv2D(16, kernel_size=(3, 3), padding='valid', input_shape=input_shape))
model_v3.add(Activation('relu'))
model_v3.add(Conv2D(32, kernel_size=(3, 3), padding='valid'))
model_v3.add(Activation('relu'))
model_v3.add(MaxPooling2D(pool_size=(2, 2)))

model_v3.add(Conv2D(64, kernel_size=(3, 3), padding='valid'))
model_v3.add(Activation('relu'))
model_v3.add(Conv2D(64, kernel_size=(3, 3), padding='valid'))
model_v3.add(Activation('relu'))
model_v3.add(MaxPooling2D(pool_size=(2, 2)))

model_v3.add(Flatten())
model_v3.add(Dense(units=128))
model_v3.add(Activation('relu'))
model_v3.add(Dense(units=50))
model_v3.add(Activation('relu'))
model_v3.add(Dense(units=25))
model_v3.add(Activation('softmax'))

# Compilar el modelo
model_v3.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=optimizer,
    metrics=['accuracy']
)

weights_v3 = model_v3.get_weights()

print(model_v3.summary())

# Entrenar el modelo
print('Training second "enhanced" model with no Dropout')
history = model_v3.fit_generator(
    train_iter,
    steps_per_epoch=len(x_train)*0.9/batch_size,
    epochs=epochs,
    validation_data=validation_iter,
    validation_steps=len(x_train)*0.1/batch_size
)

# Mostrar graficas
mostrarEvolucion(history)

#######################################
# Uso de Dropout

# Definicion del nuevo modelo
model_v2 = Sequential()
model_v2.add(Conv2D(8, kernel_size=(5, 5), padding='valid', input_shape=input_shape))
model_v2.add(Activation('relu'))
model_v2.add(Conv2D(16, kernel_size=(5, 5), padding='valid'))
model_v2.add(Activation('relu'))
model_v2.add(MaxPooling2D(pool_size=(2, 2)))
model_v2.add(Dropout(0.2))

model_v2.add(Conv2D(32, kernel_size=(3, 3), padding='valid'))
model_v2.add(Activation('relu'))
model_v2.add(Conv2D(64, kernel_size=(3, 3), padding='valid'))
model_v2.add(Activation('relu'))
model_v2.add(MaxPooling2D(pool_size=(2, 2)))
model_v2.add(Dropout(0.5))

model_v2.add(Flatten())
model_v2.add(Dense(units=128))
model_v2.add(Activation('relu'))
model_v2.add(Dense(units=50))
model_v2.add(Activation('relu'))
model_v2.add(Dense(units=25))
model_v2.add(Activation('softmax'))

# Compilar el modelo
model_v2.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=optimizer,
    metrics=['accuracy']
)

weights_v2 = model_v2.get_weights()

print(model_v2.summary())


# Entrenar el modelo
print('Training first "enhanced" model with Dropout')
history = model_v2.fit_generator(
    train_iter,
    steps_per_epoch=len(x_train)*0.9/batch_size,
    epochs=epochs,
    validation_data=validation_iter,
    validation_steps=len(x_train)*0.1/batch_size
)

# Mostrar graficas
mostrarEvolucion(history)

# Definicion del nuevo modelo
model_v3 = Sequential()
model_v3.add(Conv2D(16, kernel_size=(3, 3), padding='valid', input_shape=input_shape))
model_v3.add(Activation('relu'))
model_v3.add(Conv2D(32, kernel_size=(3, 3), padding='valid'))
model_v3.add(Activation('relu'))
model_v3.add(MaxPooling2D(pool_size=(2, 2)))
model_v3.add(Dropout(0.2))

model_v3.add(Conv2D(64, kernel_size=(3, 3), padding='valid'))
model_v3.add(Activation('relu'))
model_v3.add(Conv2D(64, kernel_size=(3, 3), padding='valid'))
model_v3.add(Activation('relu'))
model_v3.add(MaxPooling2D(pool_size=(2, 2)))
model_v3.add(Dropout(0.5))

model_v3.add(Flatten())
model_v3.add(Dense(units=128))
model_v3.add(Activation('relu'))
model_v3.add(Dense(units=50))
model_v3.add(Activation('relu'))
model_v3.add(Dense(units=25))
model_v3.add(Activation('softmax'))

# Compilar el modelo
model_v3.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=optimizer,
    metrics=['accuracy']
)

weights_v3 = model_v3.get_weights()

print(model_v3.summary())

# Entrenar el modelo
print('Training second "enhanced" model with Dropout')
history = model_v3.fit_generator(
    train_iter,
    steps_per_epoch=len(x_train)*0.9/batch_size,
    epochs=epochs,
    validation_data=validation_iter,
    validation_steps=len(x_train)*0.1/batch_size
)

# Mostrar graficas
mostrarEvolucion(history)

# Establecer epocas
epochs = 50

# Restaurar pesos
model_v2.set_weights(weights_v2)

# Entrenar el modelo
print('Training first "enhanced" model with Dropout')
history = model_v2.fit_generator(
    train_iter,
    steps_per_epoch=len(x_train)*0.9/batch_size,
    epochs=epochs,
    validation_data=validation_iter,
    validation_steps=len(x_train)*0.1/batch_size
)

# Mostrar graficas
mostrarEvolucion(history)

# Restaurar pesos
model_v3.set_weights(weights_v3)

# Entrenar el modelo
print('Training first "enhanced" model with Dropout')
history = model_v3.fit_generator(
    train_iter,
    steps_per_epoch=len(x_train)*0.9/batch_size,
    epochs=epochs,
    validation_data=validation_iter,
    validation_steps=len(x_train)*0.1/batch_size
)

# Mostrar graficas
mostrarEvolucion(history)


# Restaurar pesos
model_v2.set_weights(weights_v2)

# Entrenar el modelo
print('Training first "enhanced" model with Dropout')
history = model_v2.fit_generator(
    train_iter_flip,
    steps_per_epoch=len(x_train)*0.9/batch_size,
    epochs=epochs,
    validation_data=validation_iter_flip,
    validation_steps=len(x_train)*0.1/batch_size
)

# Mostrar graficas
mostrarEvolucion(history)

# Restaurar pesos
model_v3.set_weights(weights_v3)

# Entrenar el modelo
print('Training first "enhanced" model with Dropout')
history = model_v3.fit_generator(
    train_iter_flip,
    steps_per_epoch=len(x_train)*0.9/batch_size,
    epochs=epochs,
    validation_data=validation_iter_flip,
    validation_steps=len(x_train)*0.1/batch_size
)

# Mostrar graficas
mostrarEvolucion(history)

###############################################################################
# 4. Capas de normalizacion

# Definicion del nuevo modelo
model_batch_pre = Sequential()
model_batch_pre.add(Conv2D(16, kernel_size=(3, 3), padding='valid', input_shape=input_shape))
model_batch_pre.add(BatchNormalization())
model_batch_pre.add(Activation('relu'))
model_batch_pre.add(Conv2D(32, kernel_size=(3, 3), padding='valid'))
model_batch_pre.add(BatchNormalization())
model_batch_pre.add(Activation('relu'))
model_batch_pre.add(MaxPooling2D(pool_size=(2, 2)))
model_batch_pre.add(Dropout(0.2))

model_batch_pre.add(Conv2D(64, kernel_size=(3, 3), padding='valid'))
model_batch_pre.add(BatchNormalization())
model_batch_pre.add(Activation('relu'))
model_batch_pre.add(Conv2D(64, kernel_size=(3, 3), padding='valid'))
model_batch_pre.add(BatchNormalization())
model_batch_pre.add(Activation('relu'))
model_batch_pre.add(MaxPooling2D(pool_size=(2, 2)))
model_batch_pre.add(Dropout(0.5))

model_batch_pre.add(Flatten())
model_batch_pre.add(Dense(units=128))
model_batch_pre.add(BatchNormalization())
model_batch_pre.add(Activation('relu'))
model_batch_pre.add(Dense(units=50))
model_batch_pre.add(BatchNormalization())
model_batch_pre.add(Activation('relu'))
model_batch_pre.add(Dense(units=25))
model_batch_pre.add(Activation('softmax'))

# Compilar el modelo
model_batch_pre.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=optimizer,
    metrics=['accuracy']
)

# Guardar los pesos
weights_batch_pre = model_batch_pre.get_weights()

print(model_batch_pre.summary())

# Entrenar el modelo
print('Training first batch normalization model')
history = model_batch_pre.fit_generator(
    train_iter_flip,
    steps_per_epoch=len(x_train)*0.9/batch_size,
    epochs=epochs,
    validation_data=validation_iter_flip,
    validation_steps=len(x_train)*0.1/batch_size
)

# Mostrar graficas
mostrarEvolucion(history)

# Definicion del nuevo modelo
model_batch_post = Sequential()
model_batch_post.add(Conv2D(16, kernel_size=(3, 3), padding='valid', input_shape=input_shape))
model_batch_post.add(Activation('relu'))
model_batch_post.add(BatchNormalization())
model_batch_post.add(Conv2D(32, kernel_size=(3, 3), padding='valid'))
model_batch_post.add(Activation('relu'))
model_batch_post.add(BatchNormalization())
model_batch_post.add(MaxPooling2D(pool_size=(2, 2)))
model_batch_post.add(Dropout(0.2))

model_batch_post.add(Conv2D(64, kernel_size=(3, 3), padding='valid'))
model_batch_post.add(Activation('relu'))
model_batch_post.add(BatchNormalization())
model_batch_post.add(Conv2D(64, kernel_size=(3, 3), padding='valid'))
model_batch_post.add(Activation('relu'))
model_batch_post.add(BatchNormalization())
model_batch_post.add(MaxPooling2D(pool_size=(2, 2)))
model_batch_post.add(Dropout(0.5))

model_batch_post.add(Flatten())
model_batch_post.add(Dense(units=128))
model_batch_post.add(Activation('relu'))
model_batch_post.add(BatchNormalization())
model_batch_post.add(Dense(units=50))
model_batch_post.add(Activation('relu'))
model_batch_post.add(BatchNormalization())
model_batch_post.add(Dense(units=25))
model_batch_post.add(Activation('softmax'))

# Compilar el modelo
model_batch_post.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=optimizer,
    metrics=['accuracy']
)

# Guardar los pesos
weights_batch_post = model_batch_post.get_weights()

print(model_batch_post.summary())

# Entrenar el modelo
print('Training second batch normalization model')
history = model_batch_post.fit_generator(
    train_iter_flip,
    steps_per_epoch=len(x_train)*0.9/batch_size,
    epochs=epochs,
    validation_data=validation_iter_flip,
    validation_steps=len(x_train)*0.1/batch_size
)

# Mostrar graficas
mostrarEvolucion(history)

###############################################################################
# Ajuste final

epochs = 60

# Crear generador de test
datagen_test = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True
)

# Entrenar generador
datagen_test.fit(x_train)

# Reestablecer pesos
model_batch_pre.set_weights(weights_batch_pre)

# Entrenar modelo utilizando aumento de datos
print('Training first batch normalization model with data augmentation')
history = model_batch_pre.fit_generator(
    train_iter_flip,
    steps_per_epoch=len(x_train)*0.9/batch_size,
    epochs=epochs,
    validation_data=validation_iter_flip,
    validation_steps=len(x_train)*0.1/batch_size
)

# Mostrar graficas
mostrarEvolucion(history)

# Predecir los datos
prediction = model_batch_pre.predict_generator(
    datagen_test.flow(x_test, batch_size=1, shuffle=False),
    steps=len(x_test),
    verbose=1
)


# Obtener accuracy de test y mostrarla
accuracy = calcularAccuracy(y_test, prediction)
print('Test accuracy: {}'.format(accuracy))