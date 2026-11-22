import numpy as np

# MLP de 2 camadas (9 -> hidden -> 9), apenas propagacao (sem backpropagation).
# Layout do cromossomo (sem a coluna final de aptidao), mesmo tamanho calculado em Genetic.row_len:
#   W1 (hidden*9) | b1 (hidden) | W2 (hidden*9) | b2 (9)
def mlp_forward(weights, inputs, hidden: int) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    x = np.asarray(inputs, dtype=float)
    a = hidden * 9
    W1 = w[0:a].reshape(hidden, 9)
    b1 = w[a:a + hidden]
    c = a + hidden
    W2 = w[c:c + hidden * 9].reshape(9, hidden)
    b2 = w[c + hidden * 9:c + hidden * 9 + 9]
    h = np.tanh(W1 @ x + b1)
    return np.tanh(W2 @ h + b2)  # 9 saidas: uma por celula

# Retorna o indice (0..8) da celula escolhida pela rede (argmax das saidas).
# Nao filtra celulas ocupadas: escolher celula ocupada deve ser penalizado pela aptidao.
def mlp_pick_move(weights, board_flat_mapped: list, hidden: int) -> int:
    return int(np.argmax(mlp_forward(weights, board_flat_mapped, hidden)))

# Joga como `me` ('x' ou 'o') num tabuleiro de 9 posicoes ('x','o','b'). Minha peca = +1, oponente = -1.
def net_move(weights, board: list, me: str, hidden: int) -> int:
    mapped = [0 if c == 'b' else (1 if c == me else -1) for c in board]
    return mlp_pick_move(weights, mapped, hidden)