import numpy as np
from numpy import loadtxt
import tensorflow as tf
import matplotlib.pyplot as plot

def plotagemDados(coluna, eixoX, labelX):
    eixoY = []
    aux = []
    coluna.sort()
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

def plotarResultados(resultado, titulo, ylabel, xlabel):
    plot.plot(listAcc)
    plot.title(titulo)
    plot.ylabel(ylabel)
    plot.xlabel(xlabel)
    plot.show()
# Carrega todos os valores do arquivo para a variável.
numpydados = loadtxt('breast-cancer.csv', delimiter=',')
# [:, w] -> seleciona todas as linhas e coluna w
# [x, w:y] -> seleciona a linha x e as colunas de w até y-1
X = numpydados[:266,1:] # Valores de entrada, [linha, coluna]
Z = numpydados[266:,1:]
Y = numpydados[:266,0] # Valores de saida, [linha, coluna]
W = numpydados[266:,0]
# plot.bar(X, np.arange(1,13))
# plotagem(list(X[:, 8]), ['Sim','Não'], 'Histórico de radioterapia')


# Definição do modelo da rede neural, primeira camada com 12 neurônios, segunda com 4 e 1 de saída.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(12, input_dim=9, activation=tf.nn.sigmoid, kernel_initializer='random_uniform'),
    tf.keras.layers.Dense(8, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(4, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
# Seleção do metódo de treinamento do modelo
model.compile(optimizer=tf.keras.optimizers.Adamax(lr=0.11),
              loss='mean_squared_logarithmic_error',
              metrics=['accuracy'])

# Treinamento 
history = model.fit(X, Y, epochs=197)   
listAcc = list(history.history['acc'])
avgAcc = np.average(listAcc)
listLoss = list(history.history['loss'])
avgLoss = np.average(listLoss)
print('Treinamento:\nAcerto médio: %.2f ' % (avgAcc*100))
print('Erro médio: %.2f ' % (avgLoss))
# Resultados
print('Com base de treinamento:')
loss, accuracy = model.evaluate(X, Y)
print('Acerto: %.2f' % (accuracy*100))
print('Erro: %.2f' % (loss))
print('Com bas de predição:')
loss, accuracy = model.evaluate(Z, W)
print('Acerto: %.2f' % (accuracy*100))
print('Erro: %.2f' % (loss))
predictions = model.predict_classes(Z)
# summarize the first 5 cases
for i in range(len(Z)):
	print('%s => %d (expected %d)' % (Z[i].tolist(), predictions[i], W[i]))

plotarResultados(listAcc,'Model accuracy', 'Accuracy', 'Epoch')
plotarResultados(listLoss,'Model loss', 'Loss', 'Epoch')