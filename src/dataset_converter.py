import csv

def line_conv(line: str):
    clean = line.rstrip().split(',')
    for i in range(len(clean)):
        clean[i] = clean[i].replace('x', '1,0,0') if clean[i] == 'x' else clean[i]
        clean[i] = clean[i].replace('o', '0,1,0') if clean[i] == 'o' else clean[i]
        clean[i] = clean[i].replace('b', '0,0,1') if clean[i] == 'b' else clean[i]
        clean[i] = clean[i].replace('tem_jogo', '0') if clean[i] == 'tem_jogo' else clean[i]
        clean[i] = clean[i].replace('x_venceu', '1') if clean[i] == 'x_venceu' else clean[i]
        clean[i] = clean[i].replace('o_venceu', '2') if clean[i] == 'o_venceu' else clean[i]
        clean[i] = clean[i].replace('empate', '3') if clean[i] == 'empate' else clean[i]
    return ','.join(clean) + '\n'

def line_conv_alternative(line: str):
    clean = line.rstrip().split(',')
    clean[9] = clean[9].replace('tem_jogo', '0') if clean[9] == 'tem_jogo' else clean[9]
    clean[9] = clean[9].replace('x_venceu', '1') if clean[9] == 'x_venceu' else clean[9]
    clean[9] = clean[9].replace('o_venceu', '2') if clean[9] == 'o_venceu' else clean[9]
    clean[9] = clean[9].replace('empate', '3') if clean[9] == 'empate' else clean[9]
    ls = []
    ls.append(clean[:9].count('x')) # numero de X
    ls.append(clean[:9].count('o')) # numero de O
    ls.append(ls[0] + ls[1]) # numero de posicoes ocupadas 
    ls.append(count_ch_in_lines(clean, 'x')) # numero de linhas com exatamente 2 X
    ls.append(count_ch_in_lines(clean, 'o')) # numero de linhas com exatamente 2 O
    ls.append(clean[:9].count('b')) # numero de posicoes nao ocupadas
    # jogador X joga quando o numero de posicoes ocupadas eh par
    # caso contrario, jogador eh O
    # 1 - X, 0 - O
    ls.append(1 if ls[2] % 2 == 0 else 0)
    ls.append(clean[9])
    return ','.join([str(x) for x in ls]) + '\n'

def count_ch_in_lines(line, ch):
    count = 0
    count += 1 if line[0:3].count(ch) == 2 else 0
    count += 1 if line[3:6].count(ch) == 2 else 0
    count += 1 if line[6:9].count(ch) == 2 else 0
    return count

with open("tic-tac-toe.data.4classes", 'r') as tictacfile:
    with open("tic-tac-toe.data.4classes.converted", 'w') as tictacconv:
        skip = True
        for line in tictacfile:
            if skip:
                skip = False
                continue
            tictacconv.write(line_conv(line))

    tictacfile.seek(0) # rewind

    with open("tic-tac-toe.data.4classes.converted.alternative", 'w') as tictacconv:
        skip = True
        for line in tictacfile:
            if skip:
                skip = False
                continue
            tictacconv.write(line_conv_alternative(line))
