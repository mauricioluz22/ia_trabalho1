import sys, time
from functools import reduce

import numpy as np

# rng = np.random.default_rng(seed=time.clock_gettime_ns(time.CLOCK_MONOTONIC))
rng = np.random.default_rng(seed=42)
# [-1, 1]? [-5, 5]?
# 0 = b
LIM_A = -1 # o
LIM_B = 1 # x

# critérios de parada:
#   90% das linhas com mesma aptidão - critério de convergência
#   Número de iterações
#   Encontro do valor ótimo

# crossover aritmético. pagina 27 no pdf JAI-ag (no moodle)
# que nem interpolacao linear :O
# nao extrapola o intervalo [p1, p2] em relacao ao blx_alpha
# gera dois filhos
# pode levar a perda de diversidade em relacao a alguns operadores, como o BLX-a
def crossover_arith(p1: list[float], p2: list[float]) -> tuple[list[float], list[float]]:
    beta = rng.random()
    b = 1 - beta
    return (
        [float(x) for x in (beta * np.array(p1) + b * np.array(p2))],
        [float(x) for x in (b * np.array(p1) + beta * np.array(p2))],
    )

# blx-alpha ("crossover mistura"). pagina 24 no pdf JAI-ag (no moodle)
# gera so um filhinho :(
# necessario verificar se filho eh factivel (i.e. se os cromossomos nao ultrapassam um determinado intervalo [a,b]; no nosso caso, [-1,1])
# diferentes valores de alpha afetam a diversidade do novo cromossomo
# @single_beta determina se cada gene usará um valor beta aleatório diferente (afeta diversidade)
def crossover_blx_a(p1: list[float], p2: list[float], alpha: float = 0.25, single_beta: bool = True) -> list[float]:
    if not single_beta:
        res: list[float] = []
        for p, q in zip(p1, p2):
            beta = rng.uniform(-alpha, 1 + alpha)
            res.append(p + beta * (q - p))
        return res
    
    beta = rng.uniform(-alpha, 1 + alpha)
    return [float(x) for x in (np.array(p1) + beta * (np.array(p2) - np.array(p1)))]

class Genetic:
    def __init__(self, population_size: int, mlp_hidden_layer: int = 18, iterations: int = 300, mutation_rate: float = 0.001, verbose: bool = False):
        self.matrix = []
        self.max_it = iterations
        self.pop_size = population_size
        self.tasks = []
        self.mutation = mutation_rate
        self.verbose = verbose
        self.iteration = 0
        self.mlp_hidden_layer = mlp_hidden_layer

        # row_len = len(self.tasks)
        row_len = self.mlp_hidden_layer * 9 * 2 + self.mlp_hidden_layer + 9
        matrix = []
        for _ in range(population_size):
            row = self.__gen_row(row_len)
            # row.append(self.__heuristic(row))
            matrix.append(row)
        self.__heuristic(matrix)
        self.matrix = matrix
        
        if self.verbose:
            print('populacao inicial:')
            [print(row) for row in self.matrix]
            print()

    def run(self) -> list[float]:
        res = self.__apt()
        self.iteration = 0
        while res == None:
            if self.iteration >= self.max_it or self.__has_converged():
                res = self.__closest_best()
            else:
                self.__update_matrix()

                res = self.__apt()
                self.iteration += 1
            
        return res[:-1] if res != None else []

    # TODO - alterar tambem ao atualizar calculo da heuristica
    def __has_converged(self) -> bool:
        h_to_count = {}
        for row in self.matrix:
            if row[-1] not in h_to_count:
                h_to_count[row[-1]] = 0
            h_to_count[row[-1]] += 1

        max = float("-inf")
        for (_, count) in h_to_count.items():
            if max < count:
                max = count
        return max / self.pop_size >= 0.9

    def __update_matrix(self):
        def __valid_kid(kid: list[float]) -> bool:
            return all(e >= LIM_A and e <= LIM_B for e in kid)
        # selecao por elitismo
        elit = self.__closest_best()
        intermediate = []
        # coluna da heuristica removida nesse passo ate gerarmos toda a nova populacao
        intermediate.append(elit[:-1])
        while len(intermediate) < self.pop_size:
            # selecao por torneio
            candidates = rng.choice(range(0, self.pop_size), 4)
            father_candidate1 = candidates[0]
            father_candidate2 = candidates[1]
            mother_candidate1 = candidates[2]
            mother_candidate2 = candidates[3]

            # TODO - alterar comparacao para coincidir com a maneira que estamos calculando a heuristica
            father = self.matrix[father_candidate1][:-1] if self.matrix[father_candidate1][-1] < self.matrix[father_candidate2][-1] else self.matrix[father_candidate2][:-1]
            mother = self.matrix[mother_candidate1][:-1] if self.matrix[mother_candidate1][-1] < self.matrix[mother_candidate2][-1] else self.matrix[mother_candidate2][:-1]

            # reproducao usando BLX-alpha (gera um filho)
            # kid: list[float] = []
            # while not __valid_kid(kid):
            #     kid = crossover_blx_a(father, mother)

            # reproducao usando crossover aritmetico (gera dois filhos)
            kid1, kid2 = crossover_arith(father, mother)
            if rng.random() < self.mutation:
                # mutacao limite (pagina 29 do pdf). Util quando usamos crossover aritmetico conforme o documento
                chromossome = rng.choice([kid1, kid2])
                # descomentar linha abaixo e comentar a de cima ao usar BLX-a
                # chromossome = kid
                gene = rng.choice(range(0, len(intermediate[chromossome])))
                if rng.random() < 0.5:
                    chromossome[gene] = LIM_A
                else:
                    chromossome[gene] = LIM_B

            # resultado do BLX-a, caso usado:
            # intermediate.append(kid)

            # resultado do crossover aritmetico, caso usado
            # primeira comparacao eh redundante. a segunda eh necessaria. mantida por estetica
            if len(intermediate) < self.pop_size: intermediate.append(kid1)
            if len(intermediate) < self.pop_size: intermediate.append(kid2)

        self.__heuristic(intermediate)
        if self.verbose:
            print(f"populacao intermediaria {self.iteration + 1}")
            [print(x) for x in intermediate]
            print()
        self.matrix = intermediate

    def __gen_row(self, lim: int) -> list[float]:
        return [float(x) for x in rng.uniform(LIM_A,LIM_B,size=lim)]

    # TODO - alterar calculo da heuristica (cada individuo deve ser testado na rede neural e verificado o resultado em jogo)
    def __heuristic(self, intermediate: list[list[float]]) -> None:
        ai = None
        board = ['b'] * 9
        # minimax funciona normalmente com o tabuleiro real
        mapping_o = { 'b': 0, 'o':1, 'x':-1 } # sugestão do claude: peça em jogo é sempre positiva, enquanto oponente é negativo
        mapping_x = { 'b': 0, 'o':-1, 'x':1 } # sugestão do claude: peça em jogo é sempre positiva, enquanto oponente é negativo
        for row in intermediate:
            score = 0
            # TODO - separar direitinho os pesos e biases
            # TODO - adicinoar verificacao de dificuldade (dita se todas as jogadas contra a ia sao feitas pelo minimax ou nao)
            row.append(score)

    # TODO - atualizar para funcionar de acordo com o valor de heuristica que considerarmos como efetivamente melhor (valor mais alto?)
    def __closest_best(self) -> list[float]:
        best = (float('+inf'), [])
        for individual in self.matrix:
            if individual[-1] < best[0]:
                best = (individual[-1], individual)
        return best[1]

    # TODO - adaptar à heurística
    def __best(self) -> list[float]:
        best = []
        for individual in self.matrix:
            best = individual if individual[-1] == 0 else best
        return best

    def __apt(self) -> list[float] | None:
        best = self.__best()
        return None if len(best) == 0 else best

if __name__ == "__main__":
    g = None
    if len(sys.argv) >= 2:
        g = Genetic(int(sys.argv[1]), 2, verbose=True)
    else:
        g = Genetic(100, 18, verbose=False)
    # print(g.tasks)
    # [print(x) for x in g.matrix]
    # print(g.run())
    print(g.iteration)
    print(crossover_arith([float(x) for x in rng.uniform(LIM_A,LIM_B,size=3)], [float(x) for x in rng.uniform(LIM_A,LIM_B,size=3)]))
    print(crossover_blx_a([float(x) for x in rng.uniform(LIM_A,LIM_B,size=3)], [float(x) for x in rng.uniform(LIM_A,LIM_B,size=3)], single_beta=False))