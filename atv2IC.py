import numpy as np
import math
import random


points = [[30.0,0],
          [25.0,5.0],
          [20.0,3.0],
          [15.0,8.0],
          [0.0,10.0],
          [6.0,7.0]]

valores = [[1],
        [1],
        [1],
        [0],
        [0],
        [0]]

class NomeDaClasse:
    def __init__(self):
        self.bias = random.randint(0,37)
        self.pesos=[]
        self.sig=[]
        self.lr = 0,5

    def predict(self,point,peso):
        activation = 0
        for j in range(len(point) - 1):              
            # peso[j]=random.randint(0,15)
            activation += (point[j]*peso[j])
        activation+= self.bias
        return  1/(1+math.exp(- activation ))

    def function(self):
        for i in range(len(points) - 1):
            print(i)
            err = 9999
            while(err!=0):
                self.sig[i] = self.predict(points[i],self.pesos[i])
                err, self.pesos[i] = self.fit(valores[i],self.sig[i],points[i],self.pesos[i])


    def fit(self, valor,valorpred,point,peso):
        e = valor - valorpred
        err = self.lr*e*point
        peso += err
        return err, peso


precep = NomeDaClasse()
print(precep.pesos)
precep.function()
print(precep.pesos)