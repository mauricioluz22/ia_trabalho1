import numpy as np

rng = np.random.default_rng(seed=42)

# TODO - dificuldade pode ser recebida por parametro nestas funcoes.
# calculo das chances de jogada ocorreria na propria funcao portanto
def pick_move_list(board: list, player, hardmode: bool) -> int:
    i, j = pick_move([
        board[0:3],
        board[3:6],
        board[6:9]
    ], player, hardmode)
    return i * 3 + j

def pick_move(board: list[list], player, hardmode: bool) -> tuple[int, int]:
    if not hardmode and rng.random() <= 0.5:
        options = [(i, j) for i, row in enumerate(board) for j, item in enumerate(row) if item == 'b']
        move = rng.choice(options)
        return move
    
    score, move = -2**20, (-1, -1)
    for i in range(0,3):
        for j in range(0,3):
            if board[i][j] == 'b':
                board[i][j] = player
                tmp_score = __minimax(board, False, player, 1)
                board[i][j] = 'b'
                if tmp_score > score:
                    score = tmp_score
                    move = (i, j)
    return move

def __minimax(board: list[list], maxxing: bool, player, depth: int) -> int:
    opp = 'x' if player == 'o' else 'o'
    if __wins(board, player):
        return 10
    if __wins(board, opp):
        return -10
    elif __board_full(board):
        return 0
    
    # player = 'o' if maxxing else 'x'
    comparator = lambda x, prev: x if x < prev else prev
    if maxxing:
        comparator = lambda x, prev: x if x > prev else prev

    # scores = []
    score = -2**20 if maxxing else 2**20
    for i in range(0,3):
        for j in range(0,3):
            if board[i][j] == 'b':
                board[i][j] = player
                # scores.append(__minimax(board, not maxxing, depth + 1))
                tmpscore =  __minimax(board, not maxxing, opp, depth + 1)
                board[i][j] = 'b'

                score = comparator(score, tmpscore)

    return score
    # return max(scores) if maxxing else min(scores)

def __board_full(board: list[list]):
    for row in board:
        for p in row:
            if p == 'b':
                return False
    return True

def __wins(board: list[list], player: str):
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
        ['o','b','o'],
        ['b','b','b'],
        ['x','b','x']
    ]
    # boardlist = ['o','x','o','b','b','x','x','b','x']
    boardlist = [x for row in board for x in row]
    
    player = 'o'
    print(pick_move(board, player, hardmode=True))
    print(pick_move_list(boardlist, player, hardmode=True))
    # print([(i, j) for i, row in enumerate(board) for j, item in enumerate(row) if item == 'b'])