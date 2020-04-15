# -*- coding: utf-8 -*-
"""Copy of JogoBuscaA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-KI0Vn4IrZc2-2I4qDxtTHzqpvqbjeR5

# O Quebra-cabeças das Quinze Pastilhas
"""
import libaryIC

"""## Ex:

espander.getFilhos() sempre retorna uma lista com 4 posições sendo que a ordem é | 0=Esquerda | 1=baixo | 2=Direita | 3=Cima | 
quando não é um filho valido, este tem valor None no vetor
"""

def testaJogadas(inicial, final, lista):
    if lista is None:
        print("Não existe inicial - final")
        return
    tmp = libaryIC.Node(inicial)
    print("##INICIO##")
    helper.printNode(inicial)
    final = [[]]
    for i in lista:
        tmp = tmp.getFilhos(i)
        print("| "+i+" |")
        helper.printNode(tmp)
        final = tmp
        tmp = libaryIC.Node(tmp)
        
    if helper.comparar(final, final):
        print("CHEGOU!!")
    else:
        print("NAO CHEGOU!!")

def buscaProfundidade(no, noFinal, visitados, caminho):
    helper = libaryIC.Helper()
    visitados.append(no)
    if no.nNode is None:
        no.nNode = 0
    helper.printNode(no)

    if helper.comparar(no.info, noFinal.info):
        print('Possui caminho')
        return caminho

    listValueFilhos = no.getFilhos()
    index = 0 
    while index < len(listValueFilhos):
        if listValueFilhos[index] is not None:
            noFilho = libaryIC.Node(listValueFilhos[index])
            noFilho.nNode = no.nNode + 1
            if not existe(visitados, noFilho):
                caminho.append(noFilho)
                testeCaminho = buscaProfundidade(noFilho, noFinal, visitados, caminho)
                if testeCaminho is None:
                    caminho = []
        index += 1
    return None

def buscaLargura(infoNo, infoNoFinal):
    helper = libaryIC.Helper()
    noAtual = libaryIC.Node(infoNo)
    listValueFilhos = noAtual.getFilhos()
    index = 0 
    
    while index < len(listValueFilhos):
        if listValueFilhos[index] is not None:
            if helper.comparar(listValueFilhos[index], infoNoFinal):
                noAtual.direcao = helper.direcao[index]
                caminho.append(noAtual)
                return caminho
        
def existe(visitados, noFilho):
    helper = libaryIC.Helper()
    for noVisitado in visitados:
        if helper.comparar(noFilho.info, noVisitado.info):
            return True
    return False

testefim = [
    [1, 2, 3],
    [6, 0, 5],
    [7, 4, 8]
]

teste = [
    [1,2,3],
    [4,0,5],
    [6,7,8]
]

helper = libaryIC.Helper()
caminho = []
visitados = []
caminho = buscaProfundidade(libaryIC.Node(teste), libaryIC.Node(testefim), visitados, caminho)


# Ex
# Dado um nó inicial, um final e uma lista com as direções:
# - É exibido o caminho na lista e comparado se chegou ou não no fim, a aprtir do ínicio
# inicial1 = [
#     [11,12,13],
#     [14,15,16],
#     [17,18,0]
# ]

# inicial2 = [
#     [11,12,13],
#     [14,0,15],
#     [16,17,18]
# ]
# helper.printNode(inicial1)

# espander = Node(inicial1)
# print("------------------------")
# var12 = helper.printNodeFilhos(espander.getFilhos())

# final1 = [
#     [11,12,13],
#     [14,0,15],
#     [16,17,18]
# ]


# final2 = [
#     [11, 12, 13],
#     [16, 0, 15],
#     [17, 14, 18]
# ]
# resp = buscaProfundidade(inicial1, final1, [])
# print(resp)
# testaJogadas(inicial1, final1, resp)

# resp = buscaProfundidade(inicial2, final2, [])
# print(resp)
# testaJogadas(inicial2, final2, resp)

# def buscaLargura(inicial, final):
#     pass:
    
#     def __init__(self):
#         self.direcao = ["Esquerda", "Baixo", "Direita", "Cima"]

#     def printNode(self, node):
#         for n in node:
#             print(n)
            
#     def printNodeFilhos(self, listnode):
#         strconcat = "| "
#         for dire in self.direcao:
#             strconcat += dire+" | "
  


# resp = buscaLargura(inicial1, final1)
# print(resp)
# testaJogadas(inicial1, final1, resp)

# resp = buscaLargura(inicial2, final2)
# print(resp)
# testaJogadas(inicial2, final2, resp)