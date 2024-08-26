import utils as utils
import random as random
import matplotlib.pyplot as plt

def algoritmoGenetico(qtGeracoes, tamanhoPopulacao=20):
    populacao = utils.gerarPopulacaoInicial(tamanhoPopulacao)
    melhores = [] # lista dos melhores de cada geração
    melhores_scores = [] # lista das avaliações dos melhores de cada geração

    for geracao in range(qtGeracoes):
        avaliacoes = [utils.funcaoAvaliacao(individuo) or 0.001 for individuo in populacao] # avalia a população

        avaliacoesInv = [1 / avaliacao for avaliacao in avaliacoes] # inverte os valores para que o menor valor seja o melhor

        probabilidades = [(1 / avaliacao) / sum(avaliacoesInv) for avaliacao in avaliacoes] # calcula a probabilidade de cada individuo ser selecionado
        selecionados = utils.selecao(populacao, probabilidades) # seleciona os individuos para cruzamento 
        filhos = []

        while len(filhos) < tamanhoPopulacao:
            pai1, pai2 = random.sample(selecionados, 2) # seleciona dois pais aleatórios

            filho = utils.crossoverUmPonto(pai1, pai2) # cruza os pais
            filho = utils.mutacao(filho) # testa mutações no filho      

            filhos.append(filho)

        populacao = filhos
        melhorIndividuo = min(populacao, key=utils.funcaoAvaliacao) # seleciona o melhor individuo da geração (menor valor)
        melhores.append(melhorIndividuo)
        melhores_scores.append(utils.funcaoAvaliacao(melhorIndividuo))

        print(f'Geração {geracao + 1} - Melhor indivíduo: {melhorIndividuo} ({int(melhorIndividuo, 2)}) - Avaliação: {utils.funcaoAvaliacao(melhorIndividuo)}')

        if (utils.funcaoAvaliacao(melhorIndividuo) == 0): 
            break

    # converte os melhores individuos para decimal
    melhores = [int (melhor, 2) for melhor in melhores] 

    # plot dos gráficos
    plt.plot(range(geracao + 1), melhores_scores) 
    plt.plot(range(geracao + 1), melhores)

    # legenda do gráfico
    plt.legend(['Avaliação', 'Melhor indivíduo'])

    plt.show()
    return melhores