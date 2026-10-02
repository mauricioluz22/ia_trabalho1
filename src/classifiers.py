import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# so para verificar os resultados
from sklearn.dummy import DummyClassifier

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

# de acordo com a documentacao do scikit, 0 e 42 sao seeds bastante populares
__random_state = 42

def __divide_dataset(dataset):
    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]

    # random_state = 42 por questao de determinismo ao dividir o dataset em varias execucoes
    X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.2, random_state=__random_state)
    # 0.25 porque 0.8 * 0.25 = 20% do dataset original (que ja foi dividido em parte 80% (temp) e 20% (test))
    X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.25, random_state=__random_state)

    return (X_train, X_val, X_test, Y_train, Y_val, Y_test)

arquivo_original = "tic-tac-toe.data.4classes.converted"
arquivo_alternativo = "tic-tac-toe.data.4classes.converted.alternative"

dataset_original = pd.read_csv(arquivo_original)
dataset_alternativo = pd.read_csv(arquivo_alternativo)

# TODO -  importar mais dois algoritmos
# random_state pode ser definido como um valor fixo posteriormente para que resultados sejam replicaveis
# entre diferentes execucoes
classificadores = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    # regra da piramide geometrica = sqrt(27 * 4),
    # onde 27 = numero de entradas para o arquivo_original e 4 = classes (saídas: tem jogo, x venceu etc)
    # UPDATE: 50 neuronios na camada oculta se provou bom!! heuristica: p deve ser menor que o dobro da camada de entrada
    # UPDATE 2: deixa pra la kk
    "MLP": MLPClassifier(hidden_layer_sizes=11, max_iter=200, learning_rate_init=0.01, random_state=__random_state),
    # valores padrao da funcao, definidos explicitamente para poderem ser alterados depois
    # max_depth = 12 porque: teste empirico. tende a retornar melhores valores
    "DecisionTree": DecisionTreeClassifier(max_depth=12, min_samples_split=2, min_samples_leaf=1, random_state=__random_state, class_weight='balanced')

    # TODO - definir outro algoritmo aqui
    # TODO - definir outro algoritmo aqui tambem
}

classif_idx_to_name = { 0 : "KNN", 1 : "MLP", 2 : "DecisionTree" }

# sugestao da professora: comecar com 200 amostras de cada classe
part_tem_jogo_og_ds = dataset_original[dataset_original['classe'] == 0].sample(n=200, random_state=__random_state)
part_x_venceu_og_ds = dataset_original[dataset_original['classe'] == 1].sample(n=200, random_state=__random_state)
part_o_venceu_og_ds = dataset_original[dataset_original['classe'] == 2].sample(n=200, random_state=__random_state)
# empate tem somente 16 estados
part_empate_og_ds = dataset_original[dataset_original['classe'] == 3]

part_original = pd.concat([part_tem_jogo_og_ds, part_x_venceu_og_ds, part_o_venceu_og_ds, part_empate_og_ds])

# sugestao da professora: comecar com 200 amostras de cada classe
part_tem_jogo_alt_ds = dataset_alternativo[dataset_alternativo['classe'] == 0].sample(n=200, random_state=__random_state)
part_x_venceu_alt_ds = dataset_alternativo[dataset_alternativo['classe'] == 1].sample(n=200, random_state=__random_state)
part_o_venceu_alt_ds = dataset_alternativo[dataset_alternativo['classe'] == 2].sample(n=200, random_state=__random_state)
# empate tem somente 16 estados
part_empate_alt_ds = dataset_alternativo[dataset_alternativo['classe'] == 3]

part_alternativo = pd.concat([part_tem_jogo_alt_ds, part_x_venceu_alt_ds, part_o_venceu_alt_ds, part_empate_alt_ds])

# NOTE - Se particionarmos o dataset original em 200 amostras por classe, o dataset orignal apresenta uma 
# queda ALTA em todos os aspectos quando se usa DecisionTree. KNN e MLP apresentam queda de quase 10%

# train = dataset de treinamento
# val = dataset de validacao (usado para verificar os parametros dos algoritmos)
# test = dataset de teste (para verificar se esta, de fato, bom)
# X_train_original_ds, X_val_original_ds, X_test_original_ds, Y_train_original_ds, Y_val_original_ds, Y_test_original_ds = __divide_dataset(dataset_original)
X_train_original_ds, X_val_original_ds, X_test_original_ds, Y_train_original_ds, Y_val_original_ds, Y_test_original_ds = __divide_dataset(part_original)
# X_train_alternative_ds, X_val_alternative_ds, X_test_alternative_ds, Y_train_alternative_ds, Y_val_alternative_ds, Y_test_alternative_ds = __divide_dataset(dataset_alternativo)
X_train_alternative_ds, X_val_alternative_ds, X_test_alternative_ds, Y_train_alternative_ds, Y_val_alternative_ds, Y_test_alternative_ds = __divide_dataset(part_alternativo)

def inicializar_algoritmos(dataset_a_usar_str):
    # columns=foo.columns mapeia o estado do jogo as colunas do dataset
    if dataset_a_usar_str == "original":
        for name, model in classificadores.items():
            model.fit(X_train_original_ds, Y_train_original_ds)
    elif dataset_a_usar_str == "alternativo":
        for name, model in classificadores.items():
            model.fit(X_train_alternative_ds, Y_train_alternative_ds)

def __encontrar_melhor_modelo(X_train, Y_train, X_val, Y_val):
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, Y_train)
    print("Baseline accuracy:", baseline.score(X_val, Y_val))
    for name, model in classificadores.items():
        print(classification_report(Y_val, model.predict(X_val)))
        print(confusion_matrix(Y_val, model.predict(X_val)))
        print(name + " accuracy:", model.score(X_val, Y_val))

    scores = cross_val_score(baseline, X_val, Y_val, cv=4, scoring='f1_macro')
    print()
    print("Model\t\tMean\t\tStandard deviation")
    print("Baseline:", scores.mean(), scores.std())
    for name, model in classificadores.items():
        # model.fit(X_train, Y_train)
        scores = cross_val_score(model, X_val, Y_val, cv=4, scoring='f1_macro')
        print(name + ":", scores.mean(), scores.std())


def testar_dataset(dataset_str_id):
    dic = {
        # TODO - trocar _val_ por _test_ para TESTAR os algoritmos
        "original" : (X_train_original_ds, Y_train_original_ds, X_val_original_ds, Y_val_original_ds),
        "alternativo": (X_train_alternative_ds, Y_train_alternative_ds, X_val_alternative_ds, Y_val_alternative_ds)
    }
    print("Usando dataset:", dataset_str_id)
    # as funcoes sao chamadas aqui
    # operador * expande a tupla
    __encontrar_melhor_modelo(*dic[dataset_str_id])


# a depender do dataset, a conversao se dara de um jeito ou de outro
# TODO - a terminar esta implementacao
def __converter_tabuleiro(tabuleiro, dataset_str):
    return [int(x) for x in tabuleiro]

# Argumentos exemplo: matriz do jogo, "KNN", "original"
# Argumentos exemplo: matriz do jogo, "MLP", "alternativo"
# Argumentos exemplo: matriz do jogo, "DecisionTree", "original"
def classificar_estado(estado, classificador_nome, dataset_a_comparar):
    # TODO - tabuleiro precisara ser convertido

    # columns=foo.columns mapeia o estado do jogo as colunas do dataset
    if dataset_a_comparar == "original":
        print(classificadores[classificador_nome].predict(pd.DataFrame([__converter_tabuleiro(estado, dataset_a_comparar)], columns=X_train_original_ds.columns)))
    elif dataset_a_comparar == "alternativo":
        print(classificadores[classificador_nome].predict(pd.DataFrame([__converter_tabuleiro(estado, dataset_a_comparar)], columns=X_train_alternative_ds.columns)))

inicializar_algoritmos("alternativo")
testar_dataset("alternativo")
classificar_estado([5,2,7,1,1,2,0], "DecisionTree", "alternativo")
# classificar_estado(['1','0','0','0','1','0','1','0','0','1','0','0','1','0','0','0','1','0','0','1','0','1','0','0','0','1','0'], "DecisionTree", "original")

