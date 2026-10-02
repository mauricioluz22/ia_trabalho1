# Codigo de conversao gerado pelo Claude

"""
Adaptação mínima do dataset Tic-Tac-Toe Endgame (UCI) para 4 classes.

1. Mantém os 958 exemplos originais e só reetiqueta a classe
   (positive -> x_venceu; negative -> o_venceu ou empate).
2. Acrescenta uma amostra pequena de tabuleiros legais ainda em jogo.

Uso:  python galo_minimo.py tic-tac-toe.data [n_tem_jogo]
Requer galo_4_classes.py na mesma pasta.
"""
import sys
import pandas as pd
from galo_4_classes import gerar_estados, classe, COLUNAS


def main(caminho, n_jogo=400, seed=42):
    orig = pd.read_csv(caminho, header=None, names=COLUNAS + ["class_original"])
    orig["class"] = orig[COLUNAS].apply(lambda r: classe(tuple(r)), axis=1)

    # verificação: x_venceu tem de coincidir exatamente com "positive"
    assert ((orig["class"] == "x_venceu") == (orig["class_original"] == "positive")).all()
    assert (orig["class"] != "tem_jogo").all()

    em_jogo = [t for t in gerar_estados() if classe(t) == "tem_jogo"]
    extra = pd.DataFrame(em_jogo, columns=COLUNAS).sample(n=n_jogo, random_state=seed)
    extra["class"] = "tem_jogo"

    final = pd.concat([orig.drop(columns="class_original"), extra], ignore_index=True)
    final = final.sample(frac=1, random_state=seed).reset_index(drop=True)
    final.to_csv("galo_minimo.csv", index=False)
    print(final["class"].value_counts())


if __name__ == "__main__":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    main(sys.argv[1], n)