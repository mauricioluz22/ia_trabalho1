# Codigo de conversao gerado pelo Claude

"""
Gera todos os tabuleiros alcançáveis do jogo do galo (X joga primeiro)
e etiqueta-os em 4 classes: tem_jogo, x_venceu, o_venceu, empate.
"""
import pandas as pd

LINHAS = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
          (0, 3, 6), (1, 4, 7), (2, 5, 8),
          (0, 4, 8), (2, 4, 6)]

COLUNAS = ["top-left", "top-middle", "top-right",
           "middle-left", "middle-middle", "middle-right",
           "bottom-left", "bottom-middle", "bottom-right"]


def vencedor(t):
    for a, b, c in LINHAS:
        if t[a] != "b" and t[a] == t[b] == t[c]:
            return t[a]
    return None


def classe(t):
    v = vencedor(t)
    if v == "x":
        return "x_venceu"
    if v == "o":
        return "o_venceu"
    if "b" not in t:
        return "empate"
    return "tem_jogo"


def gerar_estados():
    """Percorre a árvore de jogo; não expande tabuleiros já terminados."""
    inicio = tuple("b" * 9)
    vistos = {inicio}
    fila = [inicio]
    while fila:
        t = fila.pop()
        if classe(t) != "tem_jogo":
            continue
        jogador = "x" if t.count("x") == t.count("o") else "o"
        for i in range(9):
            if t[i] == "b":
                novo = t[:i] + (jogador,) + t[i + 1:]
                if novo not in vistos:
                    vistos.add(novo)
                    fila.append(novo)
    return vistos


if __name__ == "__main__":
    estados = gerar_estados()
    df = pd.DataFrame([list(t) for t in estados], columns=COLUNAS)
    df["class"] = [classe(t) for t in estados]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv("galo_4_classes.csv", index=False)
    print(len(df), "tabuleiros")
    print(df["class"].value_counts())
