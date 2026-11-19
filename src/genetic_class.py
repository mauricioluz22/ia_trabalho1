import sys, time
from functools import reduce
from minimax import ttt_wins, pick_move, pick_move_list

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
def crossover_arith(p1, p2) -> tuple:
    beta = rng.random()
    b = 1 - beta
    return (
        (beta * np.array(p1) + b * np.array(p2)),
        (b * np.array(p1) + beta * np.array(p2)),
    )

# blx-alpha ("crossover mistura"). pagina 24 no pdf JAI-ag (no moodle)
# gera so um filhinho :(
# necessario verificar se filho eh factivel (i.e. se os cromossomos nao ultrapassam um determinado intervalo [a,b]; no nosso caso, [-1,1])
# diferentes valores de alpha afetam a diversidade do novo cromossomo
# @single_beta determina se cada gene usará um valor beta aleatório diferente (afeta diversidade)
def crossover_blx_a(p1, p2, alpha: float = 0.25, single_beta: bool = True):
    if not single_beta:
        res: list[float] = []
        for p, q in zip(p1, p2):
            beta = rng.uniform(-alpha, 1 + alpha)
            res.append(p + beta * (q - p))
        return res
    
    beta = rng.uniform(-alpha, 1 + alpha)
    return np.array(p1) + beta * (np.array(p2) - np.array(p1))

# TODO - alterar todas as listas para usarem NDArray do numpy
class Genetic:
    def __init__(self, population_size: int, mlp_hidden_layer: int = 18, iterations: int = 300, mutation_rate: float = 0.001, hardmode: bool = True, verbose: bool = False):
        self.mlp_hidden_layer = mlp_hidden_layer
        # + 1 ao final para a heuristica
        self.row_len = (self.mlp_hidden_layer * 9 + self.mlp_hidden_layer) + (self.mlp_hidden_layer * 9 + 9) + 1
        self.max_it = iterations
        self.pop_size = population_size
        self.mutation = mutation_rate
        self.verbose = verbose
        self.iteration = 0
        self.hardmode = hardmode

        self.matrix = rng.uniform(LIM_A,LIM_B,size=(self.pop_size, self.row_len))
        self.__heuristic(self.matrix)
        
        if self.verbose:
            print('populacao inicial:')
            [print(row) for row in self.matrix]
            print()

    def run(self) -> list[float]:
        res = []
        self.iteration = 0
        while True:
            if self.iteration >= self.max_it or self.__has_converged():
                res = self.__closest_best()
                break
            else:
                self.__update_matrix()
                self.iteration += 1
            
        return [float(x) for x in res[:-1]]

    def __has_converged(self) -> bool:
        _, counts = np.unique(self.matrix[:, -1], return_counts=True)
        return counts.max() / self.pop_size >= 0.9

    def __update_matrix(self):
        # necessario se usarmos BLX-a
        # def __valid_kid(kid: list[float]) -> bool:
        #     return all(e >= LIM_A and e <= LIM_B for e in kid)

        # selecao por elitismo
        elit = self.__closest_best()
        # intermediate = np.array([])
        intermediate = np.zeros((self.pop_size, self.row_len))
        # coluna da heuristica removida nesse passo ate gerarmos toda a nova populacao
        # intermediate.append(elit[:-1])
        intermediate[0] = elit
        idx = 1
        while idx < self.pop_size:
            # selecao por torneio
            candidates = rng.choice(range(0, self.pop_size), 4)
            father_candidate1 = candidates[0]
            father_candidate2 = candidates[1]
            mother_candidate1 = candidates[2]
            mother_candidate2 = candidates[3]

            father = self.matrix[father_candidate1] if self.matrix[father_candidate1][-1] > self.matrix[father_candidate2][-1] else self.matrix[father_candidate2]
            mother = self.matrix[mother_candidate1] if self.matrix[mother_candidate1][-1] > self.matrix[mother_candidate2][-1] else self.matrix[mother_candidate2]

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
                gene = rng.choice(range(0, len(chromossome) - 1))
                chromossome[gene] = LIM_A if rng.random() < 0.5 else LIM_B

            # resultado do BLX-a, caso usado:
            # intermediate.append(kid)

            # resultado do crossover aritmetico, caso usado
            # primeira comparacao eh redundante. a segunda eh necessaria. mantida por estetica
            if idx < self.pop_size:
                intermediate[idx] = kid1
                idx += 1
            if idx < self.pop_size:
                intermediate[idx] = kid2
                idx += 1
        self.__heuristic(intermediate)
        if self.verbose:
            print(f"populacao intermediaria {self.iteration + 1}")
            [print(x) for x in intermediate]
            print()
        self.matrix = intermediate

    # Cálculo da heurística:
    #  
    # Vitória: +10 pontos
    # Empate: +4 pontos
    # Derrota: -2 pontos
    # Jogada inválida: -10 pontos
    # Jogada válida: +0.1
    #
    # Justificativa para as pontuações: sugeridas pelo Claude. Devem ser capazes de prevenir jogadas ilegais (a rede aprende que
    # elas não compensam) e incentivar jogadas legais, principalemnte se estas levarem a empate ou a vitória. A seleção do indivíduo,
    # portanto, se dá por aquele cujo fitness é o mais alto
    def __heuristic(self, intermediate) -> None:
        ai = None
        # minimax funciona normalmente com o tabuleiro real
        mapping_o = { 'b': 0, 'o':1, 'x':-1 } # sugestão do claude: peça em jogo é sempre positiva, enquanto oponente é negativo
        mapping_x = { 'b': 0, 'o':-1, 'x':1 } # sugestão do claude: peça em jogo é sempre positiva, enquanto oponente é negativo
        for row in intermediate:
            # TODO - separar direitinho os pesos e biases
            me, other = rng.choice([('x','o'),('o','x')])
            mapping = mapping_x if me == 'x' else mapping_o
            minimax_plays = rng.random() < 0.5 # controla se minimax joga primeiro ou não. definido aleatoriamente
            
            loops = 1 if self.hardmode else 5
            subscores = []
            while loops > 0:
                me_wins = False
                they_win = False
                tie = False
                game_finished = False
                board = [['b','b','b'], ['b','b','b'], ['b','b','b']]
                loops -= 1
                subscores.append(0)

                while not game_finished:
                    if minimax_plays:
                        # begin = time.time_ns()
                        (i, j) = pick_move(board, other, hardmode=self.hardmode)
                        # print(f"end={not minimax_plays}={(time.time_ns() - begin) / 1e6}")
                        if j < 0 or i < 0:
                            break # jogo terminou. checagem potencialmente redundante
                        board[i][j] = other
                        minimax_plays = False
                    else:
                        # TODO - Fazer IA realizar as jogadas
                        (i, j) = (0,0)
                        # verifica se a jogada é inválida
                        if board[i][j] != 'b':
                            subscores[-1] -= 10
                            break # jogo termina imediatamente em caso de jogada inválida
                        subscores[-1] += 0.1
                        board[i][j] = me
                        minimax_plays = True
                    me_wins = ttt_wins(board, me)
                    they_win = ttt_wins(board, other)
                    tie = not me_wins and not they_win and all(x != 'b' for x in [e for row in board for e in row])
        
                    game_finished = me_wins or they_win or tie
                if me_wins:
                    subscores[-1] += 10
                elif they_win:
                    subscores[-1] -= 2
                elif tie:
                    subscores[-1] += 4
            # np.append(row, float(np.average(subscores)))
            row[-1] = np.average(subscores)

    def __closest_best(self):
        best = (float('-inf'), [])
        for individual in self.matrix:
            if individual[-1] > best[0]:
                best = (individual[-1], individual)
        return best[1]

    # difícil dizer quem é efetivamente APTO
    def __apt(self):
        # best = None
        # for individual in self.matrix:
        #     best = individual if individual[-1] == 0 else best
        # return best
        return None

if __name__ == "__main__":
    g = None
    if len(sys.argv) >= 2:
        g = Genetic(int(sys.argv[1]), 2, verbose=False)
    else:
        g = Genetic(50, 18, iterations=20, hardmode=True, verbose=False)
    print(g.run())
    print(g.iteration)
    # print(crossover_arith([float(x) for x in rng.uniform(LIM_A,LIM_B,size=3)], [float(x) for x in rng.uniform(LIM_A,LIM_B,size=3)]))
    # print(crossover_blx_a([float(x) for x in rng.uniform(LIM_A,LIM_B,size=3)], [float(x) for x in rng.uniform(LIM_A,LIM_B,size=3)], single_beta=False))