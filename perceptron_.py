"""
O perceptron esta sendo utilizado como um modelo supervisionado 
No caso cada feature são as entradas e a saida da porta recebe uma base de treinamento que são as targets 
que no exemplo abaixo é a variavel saidasClassificadas, e a partir de então treinamos nosso modelo
que implica em achar o peso  
"""
import numpy as np 
#referencias de modelos lineares
referencia_porta_OR =[0,1,1,1,1,1,1,1] 
referencia_porta_AND =[0,0,0,0,0,0,0,1] 

#testando com o a porta XOR que não é linear
referencia_porta_XOR = [0,1,1,0,1,0,0,1] # impossivel classufucar esse dataset dado que é um conjunto que não pode ser separado de formar linear


entradas = np.array([[0,0,0],\
                    [0,0,1],\
                    [0,1,0],\
                    [0,1,1],\
                    [1,0,0],\
                    [1,0,1],\
                    [1,1,0],\
                    [1,1,1]])


saidasClassificadas = np.array(referencia_porta_OR) # essa saida corresponde a porta OR
pesos = np.array([.0 for x in range(len(entradas[0]))]) # quantidade de entradas
taxaDeApredizagem = 0.1


def funcAtivacao(produtoEscalar)->int:
    if produtoEscalar>=1:
        return 1 
    return 0 


def soma(registro):
    s = registro.dot(pesos)
    return funcAtivacao(s)

def treinar():
    erroTotal = 1
    cont=0
    while erroTotal!=0:
        erroTotal=0
        for i in range(len(saidasClassificadas)):
            saidaCalculada = soma(np.asarray(entradas[i]))
            #erro = abs(saidasClassificadas[i]-saidaCalculada) # retorna sempre valor absoluto do erro eliminando valores negativos, so pegando a maginitude
            erro = saidasClassificadas[i]-saidaCalculada
            erroTotal +=erro
            for j in range(len(pesos)):
                pesos[j] = pesos[j]+(taxaDeApredizagem*entradas[i][j]*erro)
                print('Peso atualizado --> ',str(pesos[j]))
        print('Total de erros: '+str(erroTotal))
        cont+=1
    return cont
    
cont = treinar()
print('qtd de iterações -> ', cont)
print('Vetor peso -> ', pesos)