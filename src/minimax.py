import numpy as np
import time

rng = np.random.default_rng(seed=42)

# TODO - dificuldade pode ser recebida por parametro nestas funcoes.
# calculo das chances de jogada ocorreria na propria funcao portanto
def pick_move_list(board: list, player, hardmode: bool) -> int:
    i, j = pick_move([
        board[0:3],
        board[3:6],
        board[6:9]
    ], player, hardmode)
    if i < 0 or j < 0:
        return -1
    return i * 3 + j

def pick_move(board: list[list], player, hardmode: bool) -> tuple[int, int]:
    options = [(i, j) for i, row in enumerate(board) for j, item in enumerate(row) if item == 'b']
    if len(options) == 0 or ttt_wins(board, 'x') or ttt_wins(board, 'o'):
        return (-1, -1)
    if not hardmode and rng.random() <= 0.5:
        move = rng.choice(options)
        return move
    
    score, move = float("-inf"), (-1, -1)
    alpha = float("-inf")
    # provavelmente a parte mais lenta do código inteiro :P
    empty_slots = len(options) - 1
    table: dict[tuple, tuple[int, str]] = {}
    for i, j in options:
        if board[i][j] == 'b':
            board[i][j] = player
            tmp_score = __minimax(
                board, 
                to_move='x' if player == 'o' else 'o',
                maxxing_player=player,
                depth=1,
                empty_slots=empty_slots,
                alpha=alpha,
                beta=float("+inf"),
                table=table,
                last_move=(i, j)
            )
            board[i][j] = 'b'
            if tmp_score > score:
                score = tmp_score
                move = (i, j)
            alpha = max(alpha,score)
    return move

def __minimax(
    board: list[list],
    to_move: str,
    maxxing_player: str,
    depth: int,
    empty_slots: int,
    alpha: float,
    beta: float,
    table: dict[tuple, tuple[int, str]],
    last_move: tuple[int, int]
) -> int:
    # tambem conta como jogador anterior
    next_player = 'x' if to_move == 'o' else 'o'
    if ttt_wins(board, next_player):
        return 10 - depth if next_player == maxxing_player else depth - 10
    elif empty_slots == 0:
        return 0

    key = (
        tuple(cell for board_row in board for cell in board_row),
        to_move,
        maxxing_player,
        depth,
        last_move,
    )
    alpha_original = alpha
    beta_original = beta

    cached = table.get(key)
    if cached is not None:
        cached_score, bound = cached
        if bound == 'exact':
            return cached_score
        if bound == 'lower':
            alpha = max(alpha, cached_score)
        else:  # upper bound
            beta = min(beta, cached_score)

        if alpha >= beta:
            return cached_score
    
    maxxing = to_move == maxxing_player
    comparator = lambda x, prev: min(x, prev)
    if maxxing:
        comparator = lambda x, prev: max(x, prev)
    score = float("-inf") if maxxing else float("+inf")

    for i in range(0,3):
        for j in range(0,3):
            if board[i][j] != 'b':
                continue
            board[i][j] = to_move
            tmpscore =  __minimax(
                board,
                next_player,
                maxxing_player,
                depth + 1,
                empty_slots - 1,
                alpha, beta,
                table, (i, j)
            )
            board[i][j] = 'b'

            score = comparator(score, tmpscore)

            if maxxing:
                alpha = comparator(alpha, score)
            else:
                beta = comparator(beta, score)
            if beta <= alpha:
                break
        if beta <= alpha:
            break
        
    result = int(score)

    # indica qual score retornar ao encontrar uma tabela igual
    if result <= alpha_original:
        bound = 'upper'
    elif result >= beta_original:
        bound = 'lower'
    else:
        bound = 'exact'

    table[key] = (result, bound)
    return result

def ttt_wins(board: list[list], player: str):
    for i in range(0, 3):
        if board[i][0] == player and board[i][1] == player and board[i][2] == player:
            return True
        if board[0][i] == player and board[1][i] == player and board[2][i] == player:
            return True

    if board[0][0] == player and board[1][1] == player and board[2][2] == player:
        return True
    elif board[2][0] == player and board[1][1] == player and board[0][2] == player:
        return True
    return False

if __name__ == "__main__":
    board = [
        ['b','b','b'],
        ['b','b','b'],
        ['b','b','b']
    ]
    # # boardlist = ['o','x','o','b','b','x','x','b','x']
    boardlist = [x for row in board for x in row]
    
    player = 'o'
    begin = time.time_ns()
    print(pick_move(board, player, hardmode=True))
    print(f"end={(time.time_ns() - begin) / 1e6}")
    print(pick_move_list(boardlist, player, hardmode=True))

    print(ttt_wins(board, 'o'))
    print(ttt_wins(board, 'x'))
    # print([(i, j) for i, row in enumerate(board) for j, item in enumerate(row) if item == 'b'])