# Front end do T2: Minimax x usuario, treino da rede (MLP + AG) contra o Minimax, usuario x rede.
# Uso (a partir da pasta do projeto):  python servidor.py   ->  http://localhost:5000
import os
import sys
import threading
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "src")

import numpy as np
from flask import Flask, jsonify, request, send_file
from minimax import pick_move_list, ttt_wins
from genetic_class import Genetic
from mlp import net_move, mlp_forward

app = Flask(__name__)
trava = threading.Lock()
estado = {"g": None, "hidden": 0, "max_it": 0, "pesos": None}  # AG em andamento / pesos da melhor rede


def venceu(b, p):
    return ttt_wins([b[0:3], b[3:6], b[6:9]], p)


def partida(pesos, hidden, hard, rede_comeca):
    """Rede x Minimax. X sempre comeca. Retorna (lances, resultado do ponto de vista da rede)."""
    b = ['b'] * 9
    rede, mm = ('x', 'o') if rede_comeca else ('o', 'x')
    vez, lances = 'x', []
    while True:
        if vez == rede:
            m = net_move(pesos, b, rede, hidden)
            if b[m] != 'b':  # jogada invalida: partida encerrada (mesma regra da aptidao)
                lances.append({"casa": m, "peca": vez, "invalida": True})
                return lances, "invalida"
        else:
            m = int(pick_move_list(b, mm, hard))
        b[m] = vez
        lances.append({"casa": m, "peca": vez, "invalida": False})
        if venceu(b, vez):
            return lances, "vitoria" if vez == rede else "derrota"
        if 'b' not in b:
            return lances, "empate"
        vez = 'o' if vez == 'x' else 'x'


def resumo_ag():
    g = estado["g"]
    h = g.history[-1]
    fit = sorted(g.fitness_list(), reverse=True)
    return {"geracao": h["geracao"], "modo": h["modo"], "historico": g.history, "populacao": fit,
            "max_geracoes": estado["max_it"], "hard_from": g.hard_from}


@app.get("/")
def index():
    return send_file("index.html")


@app.post("/api/minimax")
def api_minimax():
    d = request.get_json()
    t = d["tabuleiro"]
    if len(t) != 9 or any(c not in "xob" for c in t) or d["jogador"] not in "xo" or "b" not in t:
        return jsonify({"erro": "entrada invalida"}), 400
    return jsonify({"casa": int(pick_move_list(list(t), d["jogador"], d["modo"] == "dificil"))})


@app.post("/api/treino/iniciar")
def treino_iniciar():
    d = request.get_json()
    pop, hidden, its = int(d["populacao"]), int(d["ocultos"]), int(d["geracoes"])
    with trava:
        g = Genetic(pop, hidden, iterations=its, mutation_rate=float(d["mutacao"]),
                    hard_from=int(d["hard_from"]))
        estado.update(g=g, hidden=hidden, max_it=its, pesos=g.best_weights())
        return jsonify(resumo_ag())


@app.post("/api/treino/passo")
def treino_passo():
    with trava:
        g = estado["g"]
        if g is None:
            return jsonify({"erro": "treino nao iniciado"}), 400
        fim, _ = g.run_single_loop()
        estado["pesos"] = g.best_weights()
        r = resumo_ag()
        r["fim"] = fim
        if fim:
            r["motivo"] = "numero de geracoes" if g.iteration >= g.max_it else "convergencia"
        # partida de demonstracao: melhor rede x minimax (modo atual), alternando quem comeca
        lances, res = partida(estado["pesos"], estado["hidden"], g.hardmode, r["geracao"] % 2 == 0)
        r["demo"] = {"lances": lances, "resultado": res, "rede_comeca": r["geracao"] % 2 == 0}
        return jsonify(r)


@app.post("/api/rede/jogar")
def rede_jogar():
    d = request.get_json()
    t, me = d["tabuleiro"], d["jogador"]
    if estado["pesos"] is None:
        return jsonify({"erro": "rede nao treinada"}), 400
    mapped = [0 if c == 'b' else (1 if c == me else -1) for c in t]
    saida = mlp_forward(estado["pesos"], mapped, estado["hidden"])
    casa = int(np.argmax(saida))
    invalida = t[casa] != 'b'
    if invalida:  # no jogo contra o usuario usamos a melhor casa livre, mas sinalizamos o erro
        livres = [i for i in range(9) if t[i] == 'b']
        casa = max(livres, key=lambda i: saida[i])
    return jsonify({"casa": casa, "invalida": invalida})


@app.post("/api/rede/acuracia")
def rede_acuracia():
    """Rede x Minimax em N partidas por modo (inicio alternado). Acuracia = partidas sem derrota e sem jogada invalida."""
    if estado["pesos"] is None:
        return jsonify({"erro": "rede nao treinada"}), 400
    n = int(request.get_json().get("partidas", 20))
    out = {}
    with trava:
        for modo in ("medio", "dificil"):
            hard = modo == "dificil"
            cont = {"vitoria": 0, "empate": 0, "derrota": 0, "invalida": 0}
            total = 2 if hard else n  # modo dificil e deterministico: so 2 partidas distintas (quem comeca)
            for k in range(total):
                _, res = partida(estado["pesos"], estado["hidden"], hard, k % 2 == 0)
                cont[res] += 1
            cont["total"] = total
            cont["acuracia"] = (cont["vitoria"] + cont["empate"]) / total
            out[modo] = cont
    return jsonify(out)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)