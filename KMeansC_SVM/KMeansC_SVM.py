import os
import numpy as np
import pandas as pd
import pickle
import itertools
import logging

from datetime import datetime
from dataclasses import dataclass
from joblib import Parallel, delayed

from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, top_k_accuracy_score, classification_report

CV_SPLITS = [2, 3, 5, 10]

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
    path_folds: str
    
DATASET_CONFIGS = {
    "segmentado": DatasetConfig(
        nome="Áudios Segmentados",
        path_dataframe=f"../dataframes/{DATA_VERSION}/dataframeSegmentado.pkl",
        path_matrizes=f"{DATA_VERSION}/matrizesProba_kmeansc_svm_treinoSegmentado",
        path_modelos=f"{DATA_VERSION}/modelos_kmeansc_svm_treinoSegmentado",
        path_folds=f"../folds/{DATA_VERSION}/segmentado"
    ),
}

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
        
def calcular_metricas(y_true, y_pred, y_proba, classes, k):
    f1 = f1_score(y_true, y_pred, average="macro")

    mask = np.isin(y_true, classes)
    if not np.any(mask):
        return f1, 0

    topk = top_k_accuracy_score(
        y_true[mask],
        y_proba[mask],
        k=k,
        labels=classes
    )

    return f1, topk

############################################
# SPLIT
############################################

def split_train_val(X, y):
    counts = y.value_counts()
    classes_validas = counts[counts >= 2].index

    logging.info(f"Espécies antes do filtro: {len(counts)}")
    logging.info(f"Amostras antes do filtro: {len(y)}")

    mask = y.isin(classes_validas)
    X = X[mask]
    y = y[mask]
                
    logging.info(f"Espécies depois do filtro: {y.nunique()}")
    logging.info(f"Amostras depois do filtro: {len(y)}")
                
    logging.info(f"Espécies removidas: {len(set(counts.index) - set(classes_validas))}")
    
    return train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=1)
    
################################

def treinar_kmeansc(X_treino_scaled, y_treino, k_max):
    
    centroides = []
    labels = []
    slices = {}
    start = 0
    
    classes = np.unique(y_treino)
    
    for classe in classes:
        X_classe = X_treino_scaled[y_treino == classe]
        n_amostras = len(X_classe)
        
        k = min(len(X_classe), k_max)
        
        if k == n_amostras:
            c = X_classe
        else:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            kmeans.fit(X_classe)
            c = kmeans.cluster_centers_

        end = start + len(c)

        centroides.append(c)
        labels.extend([classe] * len(c))
        slices[classe] = (start, end)

        start = end
    
    return {
        "centroides": np.vstack(centroides),
        "labels": np.array(labels),
        "slices": slices
    }
    
############################################

def selecionar_svm(Cs, gammas, X_tr, X_val, y_tr, y_val):

    def treino(C, g):
        svm = SVC(C=C, gamma=g, cache_size=1000)
        svm.fit(X_tr, y_tr)
        pred = svm.predict(X_val)
        return f1_score(y_val, pred, average="macro")

    comb = list(itertools.product(Cs, gammas))

    scores = Parallel(n_jobs=4)(
        delayed(treino)(c, g) for c, g in comb
    )

    best = np.argmax(scores)
    C, g = comb[best]

    logging.info(f"SVM: C={C}, gamma={g}, F1={scores[best]:.3f}")

    svm = SVC(C=C, gamma=g, probability=True, cache_size=1000)
    svm.fit(X_tr, y_tr)

    return svm

def do_cv_kmeansc_svm(X, y, ka, n_splits, config, k_values, Cs, gammas):
    
    path_matrizes = os.path.join(config.path_matrizes, f"{n_splits}fold")
    path_modelos = os.path.join(config.path_modelos, f"{n_splits}fold")
                
    preparar_pastas(path_matrizes, path_modelos)
    
    path_folds = os.path.join(config.path_folds, f"stratified_group_kfold_{n_splits}.pkl")
            
    if not os.path.exists(path_folds):
        raise FileNotFoundError("Folds ainda não foram gerados.")
            
    folds = carregar_objeto(path_folds)

    f1_scores = []
    topk_scores = []

    for fold_dict in folds:

        foldId = fold_dict["fold"]
        idx_treino = fold_dict["train_idx"]
        idx_teste = fold_dict["test_idx"]

        print(f"\n=== {n_splits}-FOLD | Fold {foldId + 1} ===")

        X_treino = X.iloc[idx_treino]
        y_treino = y.iloc[idx_treino]

        X_teste = X.iloc[idx_teste]
        y_teste = y.iloc[idx_teste]
        
        modelo_filename = os.path.join(
            config.path_modelos,
            f"kmeansc_svm_model_fold_{foldId + 1}.pkl"
        )

        matriz_filename = os.path.join(
            config.path_matrizes,
            f"matriz_{foldId + 1}.pkl"
        )

        if os.path.exists(matriz_filename):
            logging.info("Carregando matriz salva...")

            matriz = carregar_objeto(matriz_filename)

            y_true = matriz["y_true"]
            y_proba = matriz["y_proba"]
            classes = matriz["classes"]

            y_pred = classes[np.argmax(y_proba, axis=1)]

            f1, topk = calcular_metricas(y_true, y_pred, y_proba, classes, ka)
            
            f1_report = classification_report(y_teste, y_pred)
            print(f"\n=== Classification Report Fold {foldId + 1} ===")
            print(f1_report)
            
        else:
            
            if os.path.exists(modelo_filename):
                logging.info("Carregando modelo salvo...")
                modelo = carregar_objeto(modelo_filename)

                svm = modelo["svm"]
                scaler_global = modelo["scaler_global"]
            
            else:
                logging.info("Treinando modelo...")
            
                X_train, X_val, y_train, y_val = split_train_val(X_treino, y_treino)
                
                scaler_global = StandardScaler()
                X_train_scaled = scaler_global.fit_transform(X_train)
                X_val_scaled = scaler_global.transform(X_val)

                melhor_f1 = -1
                
                for k in k_values:
                    
                    logging.info(f"Treinando com k={k}")

                    modelo_kmeans = treinar_kmeansc(X_train_scaled, y_train, k)
                    
                    X_proto = modelo_kmeans["centroides"]
                    y_proto = modelo_kmeans["labels"]
                    
                    svm = selecionar_svm(Cs, gammas, X_proto, X_val_scaled, y_proto, y_val)
                    
                    pred = svm.predict(X_val_scaled)
                    
                    f1_val = f1_score(y_val, pred, average="macro")
                    
                    logging.info(f"k={k} -> F1={f1_val:.3f}")
                    
                    if f1_val > melhor_f1:

                        melhor_f1 = f1_val
                        melhor_kmeans = (modelo_kmeans)
                        melhor_svm = svm
                        melhor_k = k
                
                svm = melhor_svm
                
                salvar_objeto({
                    "kmeans": melhor_kmeans,
                    "scaler_global": scaler_global,
                    "svm": svm,
                    "k": melhor_k
                }, modelo_filename)

                logging.info("Modelo salvo.")
            
            ## Teste

            X_teste_scaled = (scaler_global.transform(X_teste))

            y_pred = svm.predict(X_teste_scaled)
            y_proba = svm.predict_proba(X_teste_scaled)

            f1, topk = calcular_metricas(y_teste, y_pred, y_proba, svm.classes_, ka)
            
            salvar_objeto({
                "fold": foldId,
                "y_true": y_teste,
                "y_proba": y_proba,
                "classes": svm.classes_
            }, matriz_filename)
            
            logging.info("Matriz salva.")

        logging.info(f"F1={f1:.3f} | Top-{ka}={topk:.3f}")

        f1_scores.append(f1)
        topk_scores.append(topk)

    return f1_scores, topk_scores


def main():
    
    ka = int(input("Hiperparâmetro K (Top-K): "))
    
    print("\n##########################\n")
    
    print(f"VERSÃO = {DATA_VERSION}")
    print(f"Top-K = {ka}")
    
    logging.info("Selecione o tipo de dataset:\n1 - Segmentado\n2 - Completo")
    
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
    logging.info("Dataframe carregado com sucesso!")
    
    df = df.dropna(subset=["roi_label"])
    df["roi_label"] = df["roi_label"].astype(str)

    X = df.drop(columns=["roi_label", "audioSource"])
    y = df["roi_label"]

    logging.info(f"Quantidade de amostras: {X.shape}, Quantidade de classes: {y.nunique()}")
    resultados = {}
    
    for n_splits in CV_SPLITS:
        logging.info(f"EXPERIMENTO {n_splits}-FOLD")
        
        inicio_experimento = datetime.now()

        acuracias, topkAcuracias = do_cv_kmeansc_svm(
            X, y, ka, n_splits,
            config=config,
            k_values=[5, 10, 20, 50, 100],
            Cs = [100, 1000],
            gammas = ['scale', 2e-2]
        )
        
        final_experimento = datetime.now()
        
        resultados[n_splits] = {
            "f1": acuracias,
            "topk": topkAcuracias
        }
        
        logging.info(f"Tempo de Experimento - {n_splits} FOLD = {final_experimento - inicio_experimento}")
        
        #print(f"\n-- TESTE {config.nome.upper()} --")
        #print("F1-Score Macro:")
        #print(f"min: {min(acuracias):.2f}, max: {max(acuracias):.2f}, avg ± std: {np.mean(acuracias):.2f} ± {np.std(acuracias):.2f}")
        #print(f"\nTop-{ka} Score:")
        #print(f"min: {min(topkAcuracias):.2f}, max: {max(topkAcuracias):.2f}, avg ± std: {np.mean(topkAcuracias):.2f} ± {np.std(topkAcuracias):.2f}")
            
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