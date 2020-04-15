import pandas as pd
import numpy as np
from numpy import loadtxt
import tensorflow as tf
import matplotlib.pyplot as plot

def plotagem(coluna):
    eixoY = []
    aux = []
    print(coluna)
    coluna.sort()
    eixoX = ['Sim','Não']
    labelX = 'Histórico de radioterapia'
    for item in coluna:
        if not aux.__contains__(item):
            aux.append(item)
            eixoY.append(coluna.count(item))
    if eixoY.__len__() != aux.__len__():
        print('diferentes')
    x = np.arange(eixoY.__len__())
    plot.bar(x,eixoY)
    plot.xticks(x, eixoX)
    plot.title('Câncer de Mama')
    plot.ylabel('Quantidade')
    plot.xlabel(labelX)
    plot.show() 

# Carrega todos os valores do arquivo para a variável.
numpydados = loadtxt('breast-cancer.csv', delimiter=',')
# [:, w] -> seleciona todas as linhas e coluna w
# [x, w:y] -> seleciona a linha x e as colunas de w até y-1
X = numpydados[:,1:] # Valores de entrada, [linha, coluna]
Y = numpydados[:,0] # Valores de saida, [linha, coluna]
# plot.bar(X, np.arange(1,13))
plotagem(list(X[:, 8]))


# Definição do modelo da rede neural, primeira camada com 12 neurônios, segunda com 4 e 1 de saída.
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(12, input_dim=9, activation=tf.nn.sigmoid),
#     tf.keras.layers.Dense(4, activation=tf.nn.sigmoid),
#     tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
# ])
# # Seleção do metódo de treinamento do modelo
# model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.12),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# # Treinamento 
# model.fit(X, Y, epochs=200)   
# # Resultados
# _, accuracy = model.evaluate(X, Y)
# # print('Accuracy: %.2f' % (accuracy*100))

# predictions = model.predict_classes(X)
# summarize the first 5 cases
# for i in range(len(X)):
# 	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], Y[i]))
