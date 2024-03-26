#Importar la biblioteca TensorFlow
import tensorflow as tf

#Definir un conjunto de datos de ejemplo
x = [1, 2, 3, 4]
y = [2, 4, 6, 8]

#Definir un modelo de aprendizaje automático
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

#Compilar el modelo
modelo.compile(optimizer=tf.keras.optimizers.Adam(1), loss='mean_error')

#Entrenar el modelo

modelo.fit(x, y, epochs=500)

#Predecir el valor para una nueva muestra
print(modelo.predict([5]))