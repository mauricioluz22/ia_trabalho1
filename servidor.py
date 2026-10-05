# Servidor do front end do jogo da velha.
# Uso (a partir da pasta do projeto):  python servidor.py   ->  http://localhost:5000
# Usa as funcoes de src/classifiers.py como estao (inicializar_algoritmos e classificar_estado).
import os
import sys
import threading
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # os datasets sao lidos com caminho relativo ("src/...")
sys.path.insert(0, "src")                             # classifiers.py fica em src/

from flask import Flask, jsonify, request, send_file
import classifiers

app = Flask(__name__)

# classifiers.py guarda um unico conjunto de modelos: ao trocar de dataset, retreina com inicializar_algoritmos
dataset_atual = "alternativo"
classifiers.inicializar_algoritmos(dataset_atual)
trava = threading.Lock()


def classificar_com(ds, nome, estado):
    global dataset_atual
    with trava:
        if ds != dataset_atual:
            classifiers.inicializar_algoritmos(ds)
            dataset_atual = ds
        return int(classifiers.classificar_estado(estado, nome, ds)[0])


# Conversao do tabuleiro ('x', 'o', 'b' x 9) para o formato de cada dataset.
# Replica dataset_converter.py (line_conv e line_conv_alternative); ele nao pode ser
# importado porque reescreve os arquivos de dados ao ser executado.
def converter_original(t):
    mapa = {"x": [1, 0, 0], "o": [0, 1, 0], "b": [0, 0, 1]}
    return [n for c in t for n in mapa[c]]


def converter_alternativo(t):
    def linhas_com_2(ch):  # igual a count_ch_in_lines: so olha as 3 linhas horizontais
        return sum(1 for i in (0, 3, 6) if t[i:i + 3].count(ch) == 2)

    qx, qo = t.count("x"), t.count("o")
    ocupadas = qx + qo
    # jogador: 1 - X, 0 - O (X joga quando o numero de posicoes ocupadas eh par)
    return [qx, qo, ocupadas, linhas_com_2("x"), linhas_com_2("o"), t.count("b"),
            1 if ocupadas % 2 == 0 else 0]


CONVERSORES = {"original": converter_original, "alternativo": converter_alternativo}


@app.get("/")
def index():
    return send_file("index.html")


@app.get("/opcoes")
def opcoes():
    return jsonify({"datasets": list(CONVERSORES), "classificadores": list(classifiers.classificadores)})


@app.post("/classificar")
def classificar():
    d = request.get_json()
    t, nome, ds = d["tabuleiro"], d["classificador"], d["dataset"]
    if ds not in CONVERSORES or nome not in classifiers.classificadores \
            or len(t) != 9 or any(c not in "xob" for c in t):
        return jsonify({"erro": "entrada invalida"}), 400
    # 0 = tem jogo, 1 = x venceu, 2 = o venceu, 3 = empate
    return jsonify({"classe": classificar_com(ds, nome, CONVERSORES[ds](t))})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)