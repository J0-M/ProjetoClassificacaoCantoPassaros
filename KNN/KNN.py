import os
import numpy as np
import pickle
import logging

from datetime import datetime
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, top_k_accuracy_score

CV_SPLITS = [5, 10]

versoes_validas = ["v1_media", "v2_media_std", "v3_media_std_freq", "v4_novas_features"]

print("Selecione a versão do dataset:")
for i, v in enumerate(versoes_validas, 1):
    print(f"{i} - {v}")

entrada = input("Digite o número da versão desejada: ").strip()

if not entrada.isdigit():
    logging.error("Entrada inválida! Digite um número.")
    exit()

idx = int(entrada) - 1

if idx < 0 or idx >= len(versoes_validas):
    logging.error("Versão inválida!")
    exit()

DATA_VERSION = versoes_validas[idx]

########## CONFIGURAÇÃO LOGGING #################

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

@dataclass
class DatasetConfig:
    nome: str
    path_matrizes: str
    path_modelos: str
    path_folds: str # somente leitura

    
DATASET_CONFIGS = {
    "segmentado": DatasetConfig(
        nome="Áudios Segmentados",
        path_matrizes=f"{DATA_VERSION}/matrizesProba_knn_treinoSegmentado",
        path_modelos=f"{DATA_VERSION}/modelos_knn_treinoSegmentado",
        path_folds=f"../folds/{DATA_VERSION}/segmentado"
    ),
    "completo": DatasetConfig(
        nome="Áudios Completos",
        path_matrizes=f"{DATA_VERSION}/matrizesProba_knn_treinoCompleto",
        path_modelos=f"{DATA_VERSION}/modelos_knn_treinoCompleto",
        path_folds=f"../folds/{DATA_VERSION}/completo"
    ),
}

#################################################


########## FUNÇÕES UTILITÁRIAS ##################

def salvar_objeto(obj, caminho):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, "wb") as f:
        pickle.dump(obj, f)


def carregar_objeto(caminho):
    with open(caminho, "rb") as f:
        return pickle.load(f)


def preparar_pastas(*pastas):
    for pasta in pastas:
        os.makedirs(pasta, exist_ok=True)

#################################################


def selecionar_melhor_k(ks, X_treino, X_val, y_treino, y_val, X_teste, y_teste):
    
    acuracias_val = []

    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_treino, y_treino)
        pred = knn.predict(X_val)
        acuracias_val.append(f1_score(y_val, pred, average="macro"))
        
    melhor_val = max(acuracias_val)
    melhor_k = ks[np.argmax(acuracias_val)]
    
    logging.info(f"Melhor k na validação: {melhor_k} (acc={melhor_val:.2f})")

    knn_final = KNeighborsClassifier(n_neighbors=melhor_k)
    knn_final.fit(np.vstack((X_treino, X_val)), [*y_treino, *y_val])
    
    return knn_final, melhor_k, melhor_val


def exibir_resultados_matriz(y_true, y_proba, classes, ka):
    y_pred = classes[np.argmax(y_proba, axis=1)]
    
    f1 = f1_score(y_true, y_pred, average="macro")
    
    topk = top_k_accuracy_score(
        y_true,
        y_proba,
        k=ka,
        labels=classes
    )

    logging.info(f"F1-score do KNN: {f1:.2f}")
    logging.info(f"Top-{ka} Accuracy: {topk:.2f}")
    
def treinar_knn_com_validacao_cruzada(ka, n_splits, config: DatasetConfig):
    
    path_matrizes = os.path.join(config.path_matrizes, f"{n_splits}fold")
    path_modelos = os.path.join(config.path_modelos, f"{n_splits}fold")
    
    preparar_pastas(path_matrizes, path_modelos)
    
    path_folds = os.path.join(config.path_folds, f"stratified_group_kfold_{n_splits}.pkl")
    
    if not os.path.exists(path_folds):
        raise FileNotFoundError("Folds ainda não foram gerados.")
    
    folds = carregar_objeto(path_folds)
    
    acuracias, topKScores = [], []
        
    for fold_dict in folds:

        foldId = fold_dict["fold"]
        
        X_treino = fold_dict["X_train"]
        y_treino = fold_dict["y_train"]

        X_teste = fold_dict["X_test"]
        y_teste = fold_dict["y_test"]
        
        logging.info(f"Amostras treino: {len(X_treino)}")
        logging.info(f"Amostras teste: {len(X_teste)}")
        logging.info(f"Espécies treino: {y_treino.nunique()}")
        logging.info(f"Espécies teste: {y_teste.nunique()}")

        logging.info(f"\n=== {n_splits}-FOLD | Fold {foldId + 1} ===")
        
        logging.info(f"Amostras treino: {len(X_treino)}")
        logging.info(f"Amostras teste: {len(X_teste)}")

        matriz_filename = os.path.join(path_matrizes, f"matriz_{foldId + 1}.pkl")
        modelo_filename = os.path.join(path_modelos, f"KNN_model_fold_{foldId + 1}.pkl")

        if os.path.exists(matriz_filename):
            logging.info("Carregando matriz salva...")

            matriz = carregar_objeto(matriz_filename)

            y_true = matriz["y_true"]
            y_proba = matriz["y_proba"]
            classes = matriz["classes"]
            
            print("Classes fora do modelo:", set(y_true) - set(classes))

            y_pred = classes[np.argmax(y_proba, axis=1)]

            f1 = f1_score(y_true, y_pred, average="macro")

            topk = top_k_accuracy_score(
                y_true,
                y_proba,
                k=ka,
                labels=classes
            )
            
            exibir_resultados_matriz(y_true, y_proba, classes, ka)

        else:

            if os.path.exists(modelo_filename):
                logging.info("Carregando modelo salvo...")
                knn = carregar_objeto(modelo_filename)

                ss = StandardScaler().fit(X_treino)
                X_teste_scaled = ss.transform(X_teste)

            else:
                logging.info("Treinando modelo...")

                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_treino,
                    y_treino,
                    test_size=0.2,
                    stratify=y_treino,
                    shuffle=True,
                    random_state=10
                )

                ss = StandardScaler().fit(X_tr)

                X_tr = ss.transform(X_tr)
                X_val = ss.transform(X_val)
                X_teste_scaled = ss.transform(X_teste)

                knn, _, _ = selecionar_melhor_k(
                    range(1, 30, 2),
                    X_tr,
                    X_val,
                    y_tr,
                    y_val,
                    X_teste_scaled,
                    y_teste
                )

                salvar_objeto(knn, modelo_filename)

            y_pred = knn.predict(X_teste_scaled)
            y_proba = knn.predict_proba(X_teste_scaled)

            f1 = f1_score(y_teste, y_pred, average="macro")

            topk = top_k_accuracy_score(
                y_teste,
                y_proba,
                k=ka,
                labels=knn.classes_
            )

            salvar_objeto({
                "fold": foldId,
                "y_true": y_teste.values,
                "y_proba": y_proba,
                "classes": knn.classes_
            }, matriz_filename)

            logging.info("Matriz salva.")

        logging.info(f"F1-score: {f1:.2f}")
        logging.info(f"Top-{ka} Accuracy: {topk:.2f}")

        acuracias.append(f1)
        topKScores.append(topk)

    return acuracias, topKScores

def main():
    ka = int(input("Hiperparâmetro K (Top-K): "))
    
    print("\n##########################\n")
    
    print(f"VERSÃO = {DATA_VERSION}")
    print(f"Top-K = {ka}")
    
    # Seleção do dataset
    print("Selecione o tipo de dataset:")
    print("1 - Segmentado")
    print("2 - Completo")
    
    opcoes = {"1": "segmentado", "2": "completo"}
    tipo = opcoes.get(input("Digite sua escolha: ").strip())
    
    if tipo is None:
        logging.error("Escolha inválida!")
        return

    config = DATASET_CONFIGS[tipo]
    
    resultados = {}
    
    for n_splits in CV_SPLITS:
        
        logging.info(f"EXPERIMENTO {n_splits}-FOLD")
        
        inicio_experimento = datetime.now()
        
        acuracias, topKAcuracias = treinar_knn_com_validacao_cruzada(ka, n_splits, config)
        
        final_experimento = datetime.now()
        
        resultados[n_splits] = {
            "f1": acuracias,
            "topk": topKAcuracias
        }
        
        logging.info(f"Tempo de Experimento - {n_splits} FOLD = {final_experimento - inicio_experimento}")
    
        #print(f"\n-- TESTE {config.nome.upper()} ({n_splits}-FOLD)--")
        #print("F1-Score Macro:")
        #print(f"min: {min(acuracias):.2f}, max: {max(acuracias):.2f}, avg ± std: {np.mean(acuracias):.2f} ± {np.std(acuracias):.2f}")
        #print(f"\nTop-{ka} Score:")
        #print(f"min: {min(topKAcuracias):.2f}, max: {max(topKAcuracias):.2f}, avg ± std: {np.mean(topKAcuracias):.2f} ± {np.std(topKAcuracias):.2f}")
    
    for n_splits, resultado in resultados.items():
        f1s = resultado["f1"]
        topks = resultado["topk"]
        
        print(f"\n{n_splits}-FOLD:")
        print(f"F1 Macro = {np.mean(f1s):.2f}±{np.std(f1s):.2f}")
        print(f"Top-{ka} = {np.mean(topks):.2f} ± {np.std(topks):.2f}")
        

if __name__ == '__main__':
    startTime = datetime.now()
    main()
    endTime = datetime.now()
    print("Tempo de execução = ", endTime - startTime)