# T2 IA PUCRS – Aprendizagem por Reforço: RN + AG + Minimax

Uma rede neural (MLP) aprende a jogar o jogo da velha. Os pesos são evoluídos por um algoritmo genético (sem backpropagation) e o Minimax é o adversário de treino.

## Como rodar

Dependências:
```
pip install flask numpy
```

Front end (a partir da pasta do projeto):
```
python servidor.py
```
Abrir http://localhost:5000

Só o AG, sem front end:
```
cd src
python genetic_class.py
```

## Estrutura

| Arquivo | Função |
|---|---|
| `servidor.py` | Servidor Flask: Minimax, treino passo a passo, rede jogando, acurácia |
| `index.html` | Front end com 3 abas |
| `src/minimax.py` | Minimax com poda alfa-beta. Modo Médio (50% das jogadas aleatórias) e Difícil (sempre minimax) |
| `src/mlp.py` | MLP 9 → ocultas → 9 (tanh), só propagação. A jogada é o argmax das saídas |
| `src/genetic_class.py` | AG: população, aptidão, elitismo, torneio, cruzamento aritmético (double), mutação, parada |

## Front end

1. **Jogar contra o Minimax**: escolha o modo (Médio/Difícil) e quem começa.
2. **Treinar a rede**: defina os parâmetros e clique em *Iniciar / reiniciar*. Use *1 geração (passo)* para acompanhar passo a passo ou *Treinar automático*. A tela mostra o gráfico melhor/média/pior, a população ordenada por aptidão e uma partida da melhor rede contra o Minimax.
3. **Jogar contra a rede**: usa a melhor rede da última geração treinada. O botão *Calcular acurácia* joga a rede contra o Minimax nos dois modos.

## Rede neural

- Entrada: 9 casas (peça da rede = +1, oponente = -1, vazia = 0)
- Camada oculta: N neurônios, tanh
- Saída: 9 neurônios, tanh. A jogada é a casa de maior saída
- Cromossomo: `W1 (N*9) | b1 (N) | W2 (N*9) | b2 (9)` + aptidão na última coluna

## Algoritmo genético

- População de tamanho N configurável, pesos iniciais em [-1, 1]
- Aptidão (média das partidas contra o Minimax, início alternado a cada partida):
  vitória +10, empate +4, derrota -2, jogada válida +0,1, jogada inválida -10 (encerra a partida)
- Currículo: modo Médio até a geração `hard_from`, depois Difícil (com 4 partidas extras no Médio, `mix_medium`)
- Seleção: elitismo (1 melhor) e torneio de 2
- Cruzamento: aritmético para valores reais (gera 2 filhos, padrão). BLX-α disponível com `crossover="blx"`
- Mutação: por gene, gaussiana (σ = 0,3), limitada a [-1, 1]
- Parada: número máximo de gerações ou convergência (90% da população a até 0,5 da melhor aptidão, só no modo Difícil)

## Acurácia

Partidas da rede contra o Minimax, com início alternado. Acurácia = partidas sem derrota e sem jogada inválida. No modo Difícil o Minimax é determinístico, então só há 2 partidas distintas (quem começa).

## Parâmetros usados nos testes

População 120, 9 neurônios ocultos, 120 gerações, Difícil a partir da geração 80, mutação 0,03.