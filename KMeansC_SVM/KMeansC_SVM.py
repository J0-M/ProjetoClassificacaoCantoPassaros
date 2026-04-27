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
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, top_k_accuracy_score, pairwise_distances

DATA_VERSION = "v4_novas_features"

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
        path_folds=f"../folds/{DATA_VERSION}/segmentado/stratified_group_kfold_10.pkl"
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

    if mask.sum() == 0:
        return f1, 0

    topk = top_k_accuracy_score(
        y_true[mask],
        y_proba[mask],
        k=k,
        labels=classes
    )

    return f1, topk
        
###############################################

############################################
# SPLIT
############################################

def split_train_val(X, y):
    counts = pd.Series(y).value_counts()

    if counts.min() >= 2:
        return train_test_split(X, y, stratify=y, test_size=0.2, random_state=1)
    else:
        logging.info("Sem estratificação")
        return train_test_split(X, y, test_size=0.2, random_state=1)
    
################################

def treinar_kmeansc(X_treino, y_treino, k_por_classe=2):
    
    scaler = StandardScaler()
    X_treino = scaler.fit_transform(X_treino)
    
    centroides = []
    labels = []
    
    classes = np.unique(y_treino)
    
    for classe in classes:
        X_classe = X_treino[y_treino == classe]
        n_amostras = len(X_classe)
        
        if n_amostras <= k_por_classe: # se k > amostras, usar numero de amostras como centroides
            
            #logging.info(
            #    f"Classe {classe}: {n_amostras} amostras <= k={k_por_classe} -> usando numero de amostras como centroides"
            #)
            
            centroides.append(X_classe)
            labels.extend([classe] * n_amostras)
        
        else:
            
            #logging.info(
            #    f"Classe {classe}: {n_amostras} amostras > k={k_por_classe} -> usando KMeans com k={k_por_classe}"
            #)

            kmeans = KMeans(
                n_clusters=k_por_classe,
                random_state=42,
                n_init="auto"
            )

            kmeans.fit(X_classe)

            centroides.append(kmeans.cluster_centers_)
            labels.extend([classe] * k_por_classe)
    
    #centroides = np.concatenate(centroides, axis=0)
    #labels = np.array(labels)
    
    return {
        "centroides": np.vstack(centroides),
        "labels": np.array(labels),
        "scaler": scaler
    }
    
############################################
# FEATURE EXTRACTION
############################################

def extrair_features(modelo_kmeans, X):

    X_scaled = modelo_kmeans["scaler"].transform(X)
    centroides = modelo_kmeans["centroides"]

    X_dist = pairwise_distances(X_scaled, centroides)

    scaler_dist = StandardScaler()
    X_dist = scaler_dist.fit_transform(X_dist)

    return X_dist, scaler_dist

def transformar(X, modelo_kmeans, scaler_dist):

    X_scaled = modelo_kmeans["scaler"].transform(X)
    X_dist = pairwise_distances(X_scaled, modelo_kmeans["centroides"])

    return scaler_dist.transform(X_dist)

############################################

def selecionar_svm(Cs, gammas, X_tr, X_val, y_tr, y_val):

    def treino(C, g):
        svm = SVC(C=C, gamma=g, probability=True)
        svm.fit(X_tr, y_tr)
        pred = svm.predict(X_val)
        return f1_score(y_val, pred, average="macro")

    comb = list(itertools.product(Cs, gammas))

    scores = Parallel(n_jobs=-1)(
        delayed(treino)(c, g) for c, g in comb
    )

    best = np.argmax(scores)
    C, g = comb[best]

    logging.info(f"SVM: C={C}, gamma={g}, F1={scores[best]:.3f}")

    svm = SVC(C=C, gamma=g, probability=True)
    svm.fit(np.vstack([X_tr, X_val]), np.concatenate([y_tr, y_val]))

    return svm


def treinar_kmeans_svm(X_treino, y_treino, X_val, y_val, k_por_classe, Cs, gammas):
    
    # 1. Treinar KMeansC
    modelo_kmeans = treinar_kmeansc(X_treino, y_treino, k_por_classe)

    centroides = modelo_kmeans["centroides"]
    scaler = modelo_kmeans["scaler"]

    # 2. Transformar dados em distâncias
    X_tr_scaled = scaler.transform(X_treino)
    X_val_scaled = scaler.transform(X_val)

    X_tr_dist = transformar(X_tr_scaled, centroides)
    X_val_dist = transformar(X_val_scaled, centroides)

    melhor_f1 = -1
    melhor_svm = None
    melhor_params = None

    # 3. Grid search simples SVM
    for C in Cs:
        for gamma in gammas:
            svm = SVC(C=C, gamma=gamma, probability=True)
            svm.fit(X_tr_dist, y_treino)

            pred = svm.predict(X_val_dist)
            f1 = f1_score(y_val, pred, average="macro")

            if f1 > melhor_f1:
                melhor_f1 = f1
                melhor_svm = svm
                melhor_params = (C, gamma)

    return {
        "kmeans": modelo_kmeans,
        "svm": melhor_svm,
        "params": melhor_params
    }
    
def prever_kmeans_svm(modelo, X_teste, y_teste, ka):

    modelo_kmeans = modelo["kmeans"]
    svm = modelo["svm"]

    centroides = modelo_kmeans["centroides"]
    scaler = modelo_kmeans["scaler"]

    # transformar
    X_scaled = scaler.transform(X_teste)
    X_dist = transformar(X_scaled, centroides)

    # prever
    y_pred = svm.predict(X_dist)
    y_proba = svm.predict_proba(X_dist)

    f1 = f1_score(y_teste, y_pred, average="macro")

    # Top-K
    mask = np.isin(y_teste, svm.classes_)
    y_teste_filtrado = y_teste[mask]
    y_proba_filtrado = y_proba[mask]

    if len(y_teste_filtrado) == 0:
        topk = 0
    else:
        topk = top_k_accuracy_score(
            y_teste_filtrado,
            y_proba_filtrado,
            k=ka,
            labels=svm.classes_
        )

    return f1, topk, y_pred, y_proba


def do_cv_kmeans_svm(X, y, ka, config, k_values, Cs, gammas):
    
    preparar_pastas(config.path_matrizes, config.path_modelos)

    if not os.path.exists(config.path_folds):
        raise FileNotFoundError("Folds ainda não foram gerados.")

    folds = carregar_objeto(config.path_folds)

    f1_scores = []
    topk_scores = []

    for fold_dict in folds:

        foldId = fold_dict["fold"]
        idx_treino = fold_dict["train_idx"]
        idx_teste = fold_dict["test_idx"]

        print(f"\n=== Fold {foldId + 1} ===")

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
            
        else:
            
            if os.path.exists(modelo_filename):
                logging.info("Carregando modelo salvo...")
                modelo = carregar_objeto(modelo_filename)

                modelo_kmeans = modelo["kmeans"]
                scaler_dist = modelo["scaler_dist"]
                svm = modelo["svm"]
            
            else:
                logging.info("Treinando modelo...")
            
                X_train, X_val, y_train, y_val = split_train_val(X_treino, y_treino)
                
                melhor_f1 = -1
                melhor_modelo = None
                
                for k in k_values:

                    km = treinar_kmeansc(X_train, y_train, k)

                    X_tr_dist, scaler_dist = extrair_features(km, X_train)
                    X_val_dist = transformar(X_val, km, scaler_dist)

                    svm = selecionar_svm(Cs, gammas, X_tr_dist, X_val_dist, y_train, y_val)

                    pred = svm.predict(X_val_dist)
                    f1_val = f1_score(y_val, pred, average="macro")

                    print(f"k={k} | F1={f1_val:.3f}")

                    if f1_val > melhor_f1:
                        melhor_f1 = f1_val
                        melhor_modelo = (km, scaler_dist, svm)
                
                modelo_kmeans, scaler_dist, svm = melhor_modelo
                
                salvar_objeto({
                    "kmeans": modelo_kmeans,
                    "scaler_dist": scaler_dist,
                    "svm": svm
                }, modelo_filename)

                logging.info("Modelo salvo.")
            
            ## Teste
            
            X_te_dist = transformar(X_teste, modelo_kmeans, scaler_dist)

            y_pred = svm.predict(X_te_dist)
            y_proba = svm.predict_proba(X_te_dist)

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
    
    ka = 5 # Hiperparâmetro do Top-K
    
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
    
    acuracias, topkAcuracias = do_cv_kmeans_svm(
        X, y,
        ka=ka,
        config=config,
        k_values=[5, 10, 20],
        Cs = [1, 10, 100],
        gammas = ['scale', 1e-2, 1e-3, 1e-4]
    )
    
    print(f"\n-- TESTE {config.nome.upper()} --")
    print("F1-Score Macro:")
    print(f"min: {min(acuracias):.2f}, max: {max(acuracias):.2f}, avg ± std: {np.mean(acuracias):.2f} ± {np.std(acuracias):.2f}")
    print(f"\nTop-{ka} Score:")
    print(f"min: {min(topkAcuracias):.2f}, max: {max(topkAcuracias):.2f}, avg ± std: {np.mean(topkAcuracias):.2f} ± {np.std(topkAcuracias):.2f}")

if __name__ == '__main__':
    startTime = datetime.now()
    main()
    endTime = datetime.now()
    print("Tempo de execução = ", endTime - startTime)