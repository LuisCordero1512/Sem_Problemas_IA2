import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, tamaño_entrada, lr=0.1):
        self.pesos = np.random.rand(tamaño_entrada)
        self.bias = np.random.rand()
        self.lr = lr

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivada(self, x):
        return x * (1 - x)

    def predecir(self, entradas):
        return self.sigmoid(np.dot(entradas, self.pesos) + self.bias)

    def entrenar(self, entradas, etiquetas, max_epocas):
        errores = []
        for epoca in range(max_epocas):
            error_total = 0
            for fila_entrada, etiqueta in zip(entradas, etiquetas):
                prediccion = self.predecir(fila_entrada)
                error = etiqueta - prediccion
                self.pesos += self.lr * error * fila_entrada
                self.bias += self.lr * error
                error_total += abs(error)
            errores.append(error_total)
            if error_total == 0:
                print(f"Convergió en la época {epoca}")
                break
        else:
            print("Entrenamiento detenido sin convergencia.")
        return errores

# Cargar datos de entrenamiento y prueba desde archivos CSV
datos_entrenamiento = pd.read_csv('XOR_trn.csv', header=None)
datos_prueba = pd.read_csv('XOR_tst.csv', header=None)

X_entrenamiento = datos_entrenamiento.iloc[:, :2].values
y_entrenamiento = datos_entrenamiento.iloc[:, 2].values
X_prueba = datos_prueba.iloc[:, :2].values
y_prueba = datos_prueba.iloc[:, 2].values

# Agregar una columna de unos a las entradas para representar el término de sesgo
X_entrenamiento = np.c_[X_entrenamiento, np.ones(X_entrenamiento.shape[0])]
X_prueba = np.c_[X_prueba, np.ones(X_prueba.shape[0])]

# Crear y entrenar el perceptrón
perceptron = Perceptron(tamaño_entrada=X_entrenamiento.shape[1])
max_epocas = 1000
tasa_aprendizaje = 0.1
errores = perceptron.entrenar(X_entrenamiento, y_entrenamiento, max_epocas)

# Hacer predicciones en los datos de prueba
predicciones = []
for fila_entrada in X_prueba:
    prediccion = perceptron.predecir(fila_entrada)
    predicciones.append(prediccion)

# Comparar predicciones con las etiquetas verdaderas
predicciones_correctas = (np.array(predicciones) >= 0.5).astype(int)
exactitud = np.mean(predicciones_correctas == y_prueba)
print(f"Exactitud en los datos de prueba: {exactitud}")

# Graficar los datos de entrenamiento y la línea de separación
plt.figure(figsize=(8, 6))
plt.scatter(X_entrenamiento[:, 0], X_entrenamiento[:, 1], c=y_entrenamiento, cmap='viridis', marker='o')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')

# Generar puntos para la línea de separación
valores_x = np.linspace(-2, 2, 100)
valores_y = -(perceptron.pesos[0] * valores_x + perceptron.bias) / perceptron.pesos[1]
plt.plot(valores_x, valores_y, color='red')

plt.title('Problema XOR')
plt.legend()
plt.grid(True)
plt.show()

