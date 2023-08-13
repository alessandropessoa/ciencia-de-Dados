# -*- coding: utf-8 -*-
"""
	Implementação simples de uma mlp com um neuronio na camanda oculta e um neuronio de saida,
	8 -> 8 ->1
      entrada      pesos    neuronios camada oculta
	n0          w0           ncoculta0                   w8     
	n1          w1	         ncoculta1                   w9		
    	n2          w2           ncoculta2                   w10
	n3          w3           ncoculta3                   w11	         ncsaida
	n4          w4           ncoculta4                   w12
	n5          w5           ncoculta5                   w13
	n6          w6           ncoculta6                   w14
	n7          w7           ncoculta7                   w15
"""


import numpy as np

def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

def sigmoidDerivada(sig):
    return sig * (1 - sig)

#a = sigmoid(0.5)
#b = sigmoidDerivada(a)

#a = sigmoid(-1.5)
#b = np.exp(0)

entradas = np.array([[0,0,0],
                     [0,0,1],
                     [0,1,0],
                     [0,1,1],
                     [1,0,0],                     
                     [1,0,1],
                     [1,1,0],
                     [1,1,1]])
                     
print("dimensão da entrada -> ",entradas.shape)

saidas = np.array([[0],[1],[1],[0],[1],[0],[0],[1]]) # like impar

#pesos0 = np.array([[-0.424, -0.740, -0.961],
#                   [0.358, -0.577, -0.469]])
    
#pesos1 = np.array([[-0.017], [-0.893], [0.148]])

pesos0 = 2*np.random.random((3,8)) - 1 # 3 entradas e 
print(pesos0)
pesos1 = 2*np.random.random((8,1)) - 1

epocas = 900
taxaAprendizagem = 0.5
momento = 1

#treunando a rede
for j in range(epocas):
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    print('produto escalar camada de entrada \n',somaSinapse0)
    camadaOculta = sigmoid(somaSinapse0)

    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)
    
    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
    print("Erro medio: " + str(mediaAbsoluta))
    
    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida
    
    pesos1Transposta = pesos1.T
    deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)
    
    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)
    
    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)
    
print('pesos da camada oculta ',pesos0)
print()
print('pesos da camada saida ',pesos1)
print()
print("camada de saida calculada -> \n",camadaSaida)
print()
print("camada de saida esperada -> \n",saidas)
print()
print("camada de saida calculada -> \n",(np.array(camadaSaida)>.5))

