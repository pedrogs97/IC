import copy

class Helper:
    
    def __init__(self):
        self.direcao = ["Esquerda", "Baixo", "Direita", "Cima"]

    def printNode(self, node):
        print('\tNode : '+node.nNode.__str__())
        for item in node.info:
            print('\t'+item.__str__())
        print('\n')
            
    def printNodeFilhos(self, listnode):
        strconcat = "| "
        for dire in self.direcao:
            strconcat += dire+" | "
        print(strconcat)

        for i in range(3):
            strconcat = "|"
            for esc in listnode:
                if esc is not None:
                    strconcat += str(esc[i]) + "|"
                else:
                    strconcat += "[  ,  ,  ] |"

            print(strconcat)
    
    def comparar(self, a, b):
        return len([x for x, y in zip(a,b) if x != y]) == 0

class Node:
    def __init__(self, value):
        self.info = value
        self.nNode = None
        self.direcao = None
        
    def getFilhos(self):
        tmp = []
        tmp.append(self.goEsquerda())
        tmp.append(self.goBaixo())
        tmp.append(self.goDireita())
        tmp.append(self.goCima())
        return tmp
    
    def goBaixo(self):
        newnode = copy.deepcopy(self.info)
        flag = False
        for i in range(1, len(newnode)):
            for j in range(len(newnode[0])):
                if newnode[i-1][j] == 0:
                    newnode[i-1][j] = newnode[i][j]
                    newnode[i][j] = 0
                    flag = True
                    break
            if flag:
                break
        if not flag:
            return None
        return newnode
    
    def goCima(self):
        newnode = copy.deepcopy(self.info)
        flag = False
        for i in range(0, len(newnode)-1):
            for j in range(len(newnode[0])):
                if newnode[i+1][j] == 0:
                    newnode[i+1][j] = newnode[i][j]
                    newnode[i][j] = 0
                    flag = True
                    break
            if flag:
                break
        if not flag:
            return None
        return newnode
    
    def goDireita(self):
        newnode = copy.deepcopy(self.info)
        flag = False
        for i in range(len(newnode)):
            for j in range(1,len(newnode[0])):
                if newnode[i][j-1] == 0:
                    newnode[i][j-1] = newnode[i][j]
                    newnode[i][j] = 0
                    flag = True
                    break
            if flag:
                break
        if not flag:
            return None
        return newnode
    
    def goEsquerda(self):
        newnode = copy.deepcopy(self.info)
        flag = False
        for i in range(len(newnode)):
            for j in range(0,len(newnode[0])-1):
                if newnode[i][j+1] == 0:
                    newnode[i][j+1] = newnode[i][j]
                    newnode[i][j] = 0
                    flag = True
                    break
            if flag:
                break
        if not flag:
            return None
        return newnode

