import math as math
import random as random


TAXA_MUTACAO = 0.1 # chance de mutação

def funcaoAvaliacao(individuo):
    x = int(individuo[:3], 2) # converte os 3 primeiros bits para decimal
    y = int(individuo[3:], 2) # converte os 3 ultimos bits para decimal

    return math.sqrt(x**3 + 2*y**4) # avalia o indivíduo com a fórmula do trabalho

def gerarPopulacaoInicial(tamanhoPopulacao):
    populacao = []
    for _ in range(tamanhoPopulacao):
        xy = format(random.randint(51, 63), '06b') # gera um numero aleatorio entre 10 e 63 e converte para binario de 6 bits
        populacao.append(xy)

    return populacao

# função de cuzamento de um ponto gerando dois filhos
def crossoverUmPonto(pai1, pai2):
    pontoCorte = random.randint(1, len(pai1) - 1) # sorteia um ponto de corte exceto no início e no final
    filho = pai1[:pontoCorte] + pai2[pontoCorte:] # filho1 recebe a primeira parte do pai1 e a segunda parte do pai2

    return filho

def mutacao(individuo):
    l_individuo = list(individuo) # transforma o indivíduo temporariamente em lista
    for i in range(len(l_individuo)): # percorre o cromossomo testando a mutação bit a bit
        if random.random() < TAXA_MUTACAO: # chance de mutação

            l_individuo[i] = '1' if l_individuo[i] == '0' else '0' # troca o bit

    return ''.join(l_individuo) # transforma o indivíduo em string novamente

def selecao(populacao, probabilidades):
    selecionados = random.choices(populacao, weights=probabilidades, k=len(populacao)//2) 

    return selecionados