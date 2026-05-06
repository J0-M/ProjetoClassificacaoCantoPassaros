import os
import numpy as npy
import pickle
import logging

from datetime import datetime
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, top_k_accuracy_score

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
    path_dataframe: str
    path_matrizes: str
    path_modelos: str
    path_folds: str # somente leitura

    
DATASET_CONFIGS = {
    "segmentado": DatasetConfig(
        nome="Áudios Segmentados",
        path_dataframe=f"../dataframes/{DATA_VERSION}/dataframeSegmentado.pkl",
        path_matrizes=f"{DATA_VERSION}/matrizesProba_knn_treinoSegmentado",
        path_modelos=f"{DATA_VERSION}/modelos_knn_treinoSegmentado",
        path_folds=f"../folds/{DATA_VERSION}/segmentado/stratified_group_kfold_10.pkl"
    ),
    "completo": DatasetConfig(
        nome="Áudios Completos",
        path_dataframe=f"../dataframes/{DATA_VERSION}/dataframeAudioCompleto.pkl",
        path_matrizes=f"{DATA_VERSION}/matrizesProba_knn_treinoCompleto",
        path_modelos=f"{DATA_VERSION}/modelos_knn_treinoCompleto",
        path_folds=f"../folds/{DATA_VERSION}/completo/stratified_group_kfold_10.pkl"
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
    melhor_k = ks[npy.argmax(acuracias_val)]
    
    logging.info(f"Melhor k na validação: {melhor_k} (acc={melhor_val:.2f})")

    knn_final = KNeighborsClassifier(n_neighbors=melhor_k)
    knn_final.fit(npy.vstack((X_treino, X_val)), [*y_treino, *y_val])
    
    return knn_final, melhor_k, melhor_val


def exibir_resultados_matriz(y_true, y_proba, classes, ka):
    y_pred = classes[npy.argmax(y_proba, axis=1)]
    
    f1 = f1_score(y_true, y_pred, average="macro")

    mask = npy.isin(y_true, classes)
    
    topk = top_k_accuracy_score(
        y_true[mask],
        y_proba[mask],
        k=ka,
        labels=classes
    )

    logging.info(f"F1-score do KNN: {f1:.2f}")
    logging.info(f"Top-{ka} Accuracy: {topk:.2f}")
    
def treinar_knn_com_validacao_cruzada(X, y, ka, config: DatasetConfig):

    preparar_pastas(config.path_matrizes, config.path_modelos)
    
    if not os.path.exists(config.path_folds):
        raise FileNotFoundError("Folds ainda não foram gerados.")
    
    folds = carregar_objeto(config.path_folds)
    
    acuracias, topKScores = [], []
        
    for fold_dict in folds:

        foldId = fold_dict["fold"]
        idx_treino = fold_dict["train_idx"]
        idx_teste = fold_dict["test_idx"]

        logging.info(f"\n=== Fold {foldId + 1} ===")

        X_treino = X.iloc[idx_treino]
        y_treino = y.iloc[idx_treino]

        X_teste = X.iloc[idx_teste]
        y_teste = y.iloc[idx_teste]

        matriz_filename = os.path.join(
            config.path_matrizes, f"matriz_{foldId + 1}.pkl"
        )

        modelo_filename = os.path.join(
            config.path_modelos, f"KNN_model_fold_{foldId + 1}.pkl"
        )

        if os.path.exists(matriz_filename):
            logging.info("Carregando matriz salva...")

            matriz = carregar_objeto(matriz_filename)

            y_true = matriz["y_true"]
            y_proba = matriz["y_proba"]
            classes = matriz["classes"]
            
            print("Classes fora do modelo:", set(y_true) - set(classes))

            y_pred = classes[npy.argmax(y_proba, axis=1)]

            f1 = f1_score(y_true, y_pred, average="macro")

            mask = npy.isin(y_true, classes)

            if mask.sum() == 0:
                topk = 0
            else:
                topk = top_k_accuracy_score(
                    y_true[mask],
                    y_proba[mask],
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
                
                print(y_treino.value_counts().min())
                
                counts = y_treino.value_counts()
                classes_validas = counts[counts >= 2].index

                logging.info(f"Espécies antes do filtro: {len(counts)}")
                logging.info(f"Amostras antes do filtro: {len(y_treino)}")

                mask = y_treino.isin(classes_validas)
                X_treino = X_treino[mask]
                y_treino = y_treino[mask]
                
                logging.info(f"Espécies depois do filtro: {y_treino.nunique()}")
                logging.info(f"Amostras depois do filtro: {len(y_treino)}")
                
                logging.info(f"Espécies removidas: {len(set(counts.index) - set(classes_validas))}")

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

            mask = y_teste.isin(knn.classes_)
            y_teste_filtrado = y_teste[mask]
            y_proba_filtrado = y_proba[mask.values]

            classes_presentes = npy.intersect1d(
                knn.classes_,
                npy.unique(y_teste_filtrado)
            )

            idxs = [npy.where(knn.classes_ == c)[0][0]
                    for c in classes_presentes]

            y_proba_filtrado = y_proba_filtrado[:, idxs]

            if len(y_teste_filtrado) == 0:
                topk = 0
            else:
                topk = top_k_accuracy_score(
                    y_teste_filtrado,
                    y_proba_filtrado,
                    k=ka,
                    labels=classes_presentes
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
    
    if not os.path.exists(config.path_dataframe):
        logging.error("Dataframe não encontrado!")
        return
    
    df = carregar_objeto(config.path_dataframe)
    
    df = df.dropna(subset=["roi_label"])
    df["roi_label"] = df["roi_label"].astype(str)
    
    logging.info("Dataframe carregado com sucesso!")

    X = df.drop(columns=["roi_label", "audioSource"])
    y = df["roi_label"]

    logging.info(f"Quantidade de amostras: {X.shape}")
    logging.info(f"Quantidade de espécies: {y.nunique()}")
    
    acuracias, topKAcuracias = treinar_knn_com_validacao_cruzada(X, y, ka, config)
    
    print(f"\n-- TESTE {config.nome.upper()} --")
    print("F1-Score Macro:")
    print(f"min: {min(acuracias):.2f}, max: {max(acuracias):.2f}, avg ± std: {npy.mean(acuracias):.2f} ± {npy.std(acuracias):.2f}")
    print(f"\nTop-{ka} Score:")
    print(f"min: {min(topKAcuracias):.2f}, max: {max(topKAcuracias):.2f}, avg ± std: {npy.mean(topKAcuracias):.2f} ± {npy.std(topKAcuracias):.2f}")

if __name__ == '__main__':
    startTime = datetime.now()
    main()
    endTime = datetime.now()
    print("Tempo de execução = ", endTime - startTime)