import numpy as np
from numpy import loadtxt
import tensorflow as tf
import matplotlib.pyplot as plot

def loadData():
    # Carrega todos os valores do arquivo para a variável.
    numpydados = loadtxt('BC_alterado.csv', delimiter=',')
    # [:, saidaTeste] -> seleciona todas as linhas e coluna saidaTeste
    # [entradaTreino, saidaTeste:saidaTreino] -> seleciona a linha entradaTreino e as colunas de saidaTeste até saidaTreino-1
    # Valores de entrada, [linha, coluna]
    entradaTreino = numpydados[:85,1:]
    entradaEva = numpydados[85:170,1:]
    entradaTeste = numpydados[170:255,1:]
    # Valores de saida, [linha, coluna]
    saidaTreino = numpydados[:85,0]
    saidaEva = numpydados[85:170,0]
    saidaTeste = numpydados[170:255,0]
    # print(saidaTreino)
    # print(saidaEva)
    # print(saidaTeste)
    return entradaTreino, entradaTeste, entradaEva, saidaTreino, saidaTeste, saidaEva

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
    entradaTreino = np.arange(eixoY.__len__())
    plot.bar(entradaTreino,eixoY)
    plot.xticks(entradaTreino, eixoX)
    plot.title('Câncer de Mama')
    plot.ylabel('Quantidade')
    plot.xlabel(labelX)
    plot.show() 

def plotarResultados(resultado, titulo, ylabel, xlabel):
    plot.plot(resultado)
    plot.title(titulo)
    plot.ylabel(ylabel)
    plot.xlabel(xlabel)
    plot.show()

def MLP(entradaTreino, entradaTeste, entradaEva, saidaTreino, saidaTeste, saidaEva):

    # plot.bar(entradaTreino, np.arange(1,13))
    # plotagem(list(entradaTreino[:, 8]), ['Sim','Não'], 'Histórico de radioterapia')


    # Definição do modelo da rede neural, primeira camada com 12 neurônios, segunda com 4 e 1 de saída.
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(12, input_dim=9, activation=tf.nn.softmax),
        tf.keras.layers.Dense(8, activation=tf.nn.softmax),
        tf.keras.layers.Dense(4, activation=tf.nn.softmax),
        tf.keras.layers.Dense(1, activation=tf.nn.relu)
    ])
    # Seleção do metódo de treinamento do modelo
    model.compile(optimizer=tf.keras.optimizers.Adamax(lr=0.12),
                loss='mean_squared_logarithmic_error',
                metrics=['accuracy'])

    # Treinamento 
    history = model.fit(entradaTreino, saidaTreino, epochs=200, batch_size=16)   
    listAcc = list(history.history['acc'])
    avgAcc = np.average(listAcc)
    listLoss = list(history.history['loss'])
    avgLoss = np.average(listLoss)
    plotarResultados(listAcc,'Model accuracy', 'Accuracy', 'Epoch')
    plotarResultados(listLoss,'Model loss', 'Loss', 'Epoch')
    print('Treinamento:\nAcerto médio: %.2f ' % (avgAcc*100))
    print('Erro médio: %.2f ' % (avgLoss))
    history = model.fit(entradaTreino, saidaTreino, epochs=125, validation_data=(entradaEva, saidaEva), batch_size=64)   
    listAcc = list(history.history['acc'])
    avgAcc = np.average(listAcc)
    listLoss = list(history.history['loss'])
    avgLoss = np.average(listLoss)
    plotarResultados(listAcc,'Model accuracy', 'Accuracy', 'Epoch')
    plotarResultados(listLoss,'Model loss', 'Loss', 'Epoch')
    print('Treinamento:\nAcerto médio: %.2f ' % (avgAcc*100))
    print('Erro médio: %.2f ' % (avgLoss))
    # Resultados
    print('Com base de treinamento:')
    loss, accuracy = model.evaluate(entradaEva, saidaEva)
    print('Acerto: %.2f' % (accuracy*100))
    print('Erro: %.2f' % (loss))
    print('Com base de predição:')
    predictions = model.predict_classes(entradaTeste)
    # summarize the first 5 cases
    for i in range(len(entradaTeste)):
        print('%s => %d (expected %d)' % (entradaTeste[i].tolist(), predictions[i], saidaTeste[i]))
    txAcc = 0
    nAcc = 0
    for i in range(len(entradaTeste)):
        if predictions[i] == saidaTeste[i]:
            nAcc += 1
    txAcc = nAcc/len(entradaTeste)*100
    print('Acerto na predição: %d ' % (txAcc))

def Analise():
    import pandas

    name_coluns = ['classification','age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']
    dataset = pandas.read_csv('BC_alterado.csv', header=None, sep=',')
    dataset.columns = name_coluns
    df = dataset.loc[:,'age':'irradiat']
    print(df)
    print(df.describe())
    df.boxplot()
    plot.show()

print('Inicio...')
entradaTreino, entradaTeste, entradaEva, saidaTreino, saidaTeste, saidaEva = loadData()
MLP(entradaTreino, entradaTeste, entradaEva, saidaTreino, saidaTeste, saidaEva)
