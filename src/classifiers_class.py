import time

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# so para verificar os resultados
from sklearn.dummy import DummyClassifier

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

class PyClassificadores:
    __random_state = 42

    arquivo_original = "src/tic-tac-toe.data.4classes.converted"
    arquivo_alternativo = "src/tic-tac-toe.data.4classes.converted.alternative"
    
    dataset_original = pd.read_csv(arquivo_original)
    dataset_alternativo = pd.read_csv(arquivo_alternativo)

    __part_tem_jogo_og_ds = dataset_original[dataset_original['classe'] == 0].sample(n=200, random_state=__random_state)
    __part_x_venceu_og_ds = dataset_original[dataset_original['classe'] == 1].sample(n=200, random_state=__random_state)
    __part_o_venceu_og_ds = dataset_original[dataset_original['classe'] == 2].sample(n=200, random_state=__random_state)
    # empate tem somente 16 estados
    __part_empate_og_ds = dataset_original[dataset_original['classe'] == 3]
    
    part_original = pd.concat([__part_tem_jogo_og_ds, __part_x_venceu_og_ds, __part_o_venceu_og_ds, __part_empate_og_ds])
    
    # sugestao da professora: comecar com 200 amostras de cada classe
    __part_tem_jogo_alt_ds = dataset_alternativo[dataset_alternativo['classe'] == 0].sample(n=200, random_state=__random_state)
    __part_x_venceu_alt_ds = dataset_alternativo[dataset_alternativo['classe'] == 1].sample(n=200, random_state=__random_state)
    __part_o_venceu_alt_ds = dataset_alternativo[dataset_alternativo['classe'] == 2].sample(n=200, random_state=__random_state)
    # empate tem somente 16 estados
    __part_empate_alt_ds = dataset_alternativo[dataset_alternativo['classe'] == 3]
    
    part_alternativo = pd.concat([__part_tem_jogo_alt_ds, __part_x_venceu_alt_ds, __part_o_venceu_alt_ds, __part_empate_alt_ds])

    def __init__(self, dataset_usado: str):
        self.dataset = dataset_usado
        if dataset_usado == "original":
            self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test = self.__divide_dataset(self.part_original)
        elif dataset_usado == "alternativo":
            self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test = self.__divide_dataset(self.part_alternativo)
        self.classificadores = {
            # justificativa n_neighbors=5: testes empiricos. k maior que 5 tende a apresentar poucas melhoras (e eventualmente apresenta PIORAS!),
            # enquanto valores pequenos, menores que 5, tendem a apresentar resultados pouco bons ou muito ruins
            "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
            # regra da piramide geometrica = sqrt(27 * 4) = 11,
            # onde 27 = numero de entradas para o arquivo_original e 4 = classes (saídas: tem jogo, x venceu etc)
            # UPDATE: 50 neuronios na camada oculta se provou bom!! heuristica: p deve ser menor que o dobro da camada de entrada
            # UPDATE 2: deixa pra la kk
            # max_iter = 200 porque: valores mais altos nao levam o erro a cair mais. so fazem o algoritmo demorar mais
            # learning_rate_init = 0.01 porque: o valor tende a lever a resultados melhores E faz o algoritmo
            # convergir mais rapido. me impressiona um pouco que nn prejudique as predicoes
            "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=11, max_iter=200, learning_rate_init=0.01, random_state=self.__random_state)),
            # min_samples_split e min_samples_leaf usam os valores padrao da funcao, definidos explicitamente para poderem ser alterados depois
            # max_depth = 12 porque: teste empirico. tende a retornar melhores valores. valores maiores nao trazem melhora alguma,
            # e valores menores nao sao tao bons
            # class-weight = 'balanced' porque: reduz as consequencias do numero de instancias de classe 'empate'
            # serem tao pequenas em relacao as outras. previne problemas de divisao por zero, que, ate o momento, acontecem um bocado
            # nos outros algoritmos
            "DecisionTree": DecisionTreeClassifier(max_depth=12, min_samples_split=2, min_samples_leaf=1, random_state=self.__random_state, class_weight='balanced'),

            "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", random_state=self.__random_state)),

            "NaiveBayes": GaussianNB()
        }

        for name, model in self.classificadores.items():
            start = time.time_ns()
            model.fit(self.X_train, self.Y_train)
            delta_time = time.time_ns() - start
            print(f"Tempo de treino, {name}: {delta_time / 1e+6}")
        pass

    def __divide_dataset(self, pd_dataset):
        X = pd_dataset.iloc[:, :-1]
        Y = pd_dataset.iloc[:, -1]

        # random_state = 42 por questao de determinismo ao dividir o dataset em varias execucoes, adicionado estratificação para ser proporcional cada classe
        X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.2, random_state=self.__random_state, stratify=Y)
        # 0.25 porque 0.8 * 0.25 = 20% do dataset original (que ja foi dividido em parte 80% (temp) e 20% (test)), adicionado estratificação aqui tambem
        X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.25, random_state=self.__random_state, stratify=Y_temp)

        return (X_train, X_val, X_test, Y_train, Y_val, Y_test)

    def __converter_tabuleiro(self, tabuleiro):
        # TODO - implementar conversao corretamente
        return [int(x) for x in tabuleiro]

    def __benchmark_dataset(self, X_train, Y_train, X_out, Y_out):
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(X_train, Y_train)
        print("Baseline accuracy:", baseline.score(X_out, Y_out))
        for name, model in self.classificadores.items():
            print(classification_report(Y_out, model.predict(X_out)))
            print(confusion_matrix(Y_out, model.predict(X_out)))
            print(name + " accuracy:", model.score(X_out, Y_out))

        #cross validation no treino e nao na validação
        scores = cross_val_score(baseline, X_train, Y_train, cv=4, scoring='f1_macro')
        print()
        print("Model\t\tMean\t\tStandard deviation")
        print("Baseline:", scores.mean(), scores.std())
        for name, model in self.classificadores.items():
                # model.fit(X_train, Y_train)
                scores = cross_val_score(model, X_out, Y_out, cv=4, scoring='f1_macro')
                print(name + ":", scores.mean(), scores.std())

    def classificar_estado(self, estado, classificador_nome):
        # TODO - tabuleiro precisara ser convertido
        return self.classificadores[classificador_nome].predict(pd.DataFrame([self.__converter_tabuleiro(estado)], columns=self.X_train.columns))

    def testar_dataset(self):
        print("Executando benchmark de TESTE")
        self.__benchmark_dataset(self.X_train, self.Y_train, self.X_test, self.Y_test)

    def validar_dataset(self):
        print("Executando benchmark de VALIDAÇÃO")
        self.__benchmark_dataset(self.X_train, self.Y_train, self.X_val, self.Y_val)