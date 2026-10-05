# Trabalho 1 IA PUCRS

## O que falta fazer
- avaliar qual modelo é, de fato, melhor

## Conversão do Dataset
Foi sugerido por ferramentas de IA (Claude e Gemini) que o dataset fosse alterado das seguintes maneiras:
uso de one-hot encoder (basicamente, em vez de x, o e b representarem, por exemplo, 1, 2 e 0, cada um representa um
digido especifico dentro de um trio de digitos, isto é, teriamos, por exemplo: x = 1,0,0; o = 0,1,0; e b = 0,0,1)

Também foram gerados mais estados com base naqueles ja existentes, mantendo todos os que ja existiam no dataset original.
de 958 instancias, temos agora 1359, que inclui, como novas instancias, mais instancias de empate (embora so 16 sejam possiveis, aparentemente),
mais instancias de jogo em andamento (nao finalizado) e mais instancias de o jogando. desse jeito, o dataset ficou aproximadamente balanceado.

Os nomes dos arquivos sao prefixados pelo nome original (tic-tac-toe.data). O final indica qual versão ele corresponde (4classes - versao com 4 classes diferentes,
4classes.converted - versao que o algoritmo consegue ler e usar, e, por fim, 4classes.converted.alternative, que é a versão alternativa sugerida no enunciado)

## Formato alternativo do dataset

Implementado conforme o enunciado

## Uso do programa
Só rodar `python` com o arquivo que se deseja executar. Só tem que ser a partir do mesmo diretório (oops)

`dataset_converter.py` converte os datasets pro formato que os classificadores leem.

`classifiers.py` é o nosso testador de classificadores

Os outros arquivos .py sao scripts gerados pelo Claude para gerar as novas instancias do jogo (com base nas pré-existentes!!!)

### classifiers.py
nao ha classes, o que pode ficar meio bagunçado na hora de inicializar e testar os algoritmos...

as funcoes relevantes pra outros modulos são: `inicializar_algoritmos`, `testar_dataset` e `classificar_estado`

### classifiers_class.py

Alternativa pro `classifiers.py`. Inclui uma classe que tem a exata mesma funcionalidade que as funcoes soltas no outro script.
Diferenca principal e que, com classes, fica mais facil de usar datasets diferentes ao mesmo tempo.

A classe definida se chama `PyClassificadores`. Metodos relevantes pra outros modulos tem os mesmos nomes que as funcoes em `classifiers.py`:
`testar_dataset` (aqui, executa o teste com o dataset de TESTE) e `classificar_estado`, com a adicao de `validar_dataset` (aqui, executa o teste com o dataset de VALIDACAO). Os algoritmos sao inicializados durante a construcao da instancia, sendo, portanto,
equivalente a chamar `inicializar_algoritmos` em `classifiers.py`.