import os
import numpy as np
import pickle
import logging

from datetime import datetime
from dataclasses import dataclass

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, top_k_accuracy_score, pairwise_distances

DATA_VERSION = "v3_media_std_freq"

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
        path_matrizes=f"{DATA_VERSION}/matrizesProba_kmeansc_treinoSegmentado",
        path_modelos=f"{DATA_VERSION}/modelos_kmeansc_treinoSegmentado",
        path_folds=f"../folds/{DATA_VERSION}/segmentado/stratified_group_kfold_10.pkl"
    ),
    "completo": DatasetConfig(
        nome="Áudios Completos",
        path_dataframe=f"../dataframes/{DATA_VERSION}/dataframeAudioCompleto.pkl",
        path_matrizes=f"{DATA_VERSION}/matrizesProba_kmeansc_treinoCompleto",
        path_modelos=f"{DATA_VERSION}/modelos_kmeansc_treinoCompleto",
        path_folds=f"../folds/{DATA_VERSION}/completo/stratified_group_kfold_10.pkl"
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
        
        
def selecionar_melhor_kmeans(k_values, X_treino, X_val, y_treino, y_val, ka):

    melhor_f1 = -1
    melhor_k = None
    melhor_modelo = None

    for k in k_values:

        modelo = treinar_kmeansc(X_treino, y_treino, k)

        f1, topk, _, _ = prever_kmeansc(
            modelo,
            X_val,
            y_val,
            ka
        )

        logging.info(f"k_por_classe={k} | F1 Val={f1:.2f}")

        if f1 > melhor_f1:
            melhor_f1 = f1
            melhor_k = k
            melhor_modelo = modelo

    logging.info(f"Melhor k_por_classe: {melhor_k} | F1 Val: {melhor_f1:.2f}")

    return melhor_modelo, melhor_k, melhor_f1
      
def treinar_kmeansc(X_treino, y_treino, k_por_classe=2):
    
    scaler = StandardScaler()
    X_treino = scaler.fit_transform(X_treino)
    
    centroides = []
    labels_centroides = []
    
    classes = np.unique(y_treino)
    
    for classe in classes:
        X_classe = X_treino[y_treino == classe]
        n_amostras = len(X_classe)

        if n_amostras == 0:
            continue
        
        if n_amostras <= k_por_classe: # se k > amostras, usar numero de amostras como centroides
            
            #logging.info(
            #    f"Classe {classe}: {n_amostras} amostras <= k={k_por_classe} -> usando numero de amostras como centroides"
            #)
            
            centroides.append(X_classe)
            labels_centroides.extend([classe] * n_amostras)
        
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
            labels_centroides.extend([classe] * k_por_classe)
    
    centroides = np.concatenate(centroides, axis=0)
    labels_centroides = np.array(labels_centroides)
    
    return {
        "centroides": centroides,
        "labels": labels_centroides,
        "scaler": scaler
    }
    
def prever_kmeansc(modelo, X_teste, y_teste, ka):
    
    scaler = modelo["scaler"]
    centroides = modelo["centroides"]
    labels_centroides = modelo["labels"]
    
    X_teste = scaler.transform(X_teste)
    
    distancias = pairwise_distances(X_teste, centroides, metric="euclidean")
    
    idx_sorted = np.argsort(distancias, axis=1) #ordenar centroides por distancia (menor p maior)
    
    y_pred = labels_centroides[idx_sorted[:, 0]]
    
    ## conversão centroide para classe
    classes = np.unique(labels_centroides) 
    n_samples = X_teste.shape[0]

    y_scores_classes = np.full((n_samples, len(classes)), -np.inf)

    for i, classe in enumerate(classes):
        mask = labels_centroides == classe
        # pegar melhor centroide da classe
        y_scores_classes[:, i] = -np.min(distancias[:, mask], axis=1)
    
    ####################
    
    f1 = f1_score(y_teste, y_pred, average="macro")
    
    classes_treino = np.unique(labels_centroides)

    mask = np.isin(y_teste, classes_treino)

    y_teste_filtrado = y_teste[mask]
    y_scores_filtrado = y_scores_classes[mask]

    if len(y_teste_filtrado) == 0:
        topk = 0
    else:
        topk = top_k_accuracy_score(
            y_teste_filtrado,
            y_scores_filtrado,
            k=ka,
            labels=classes_treino
        )
    
    return f1, topk, y_pred, y_scores_classes

def do_cv_kmeansc(X, y, ka, config: DatasetConfig, k_values):
    
    preparar_pastas(config.path_matrizes, config.path_modelos)

    if not os.path.exists(config.path_folds):
        raise FileNotFoundError("Folds ainda não foram gerados.")

    folds = carregar_objeto(config.path_folds)

    acuracias = []
    topkScores = []
    
    for fold_dict in folds:

        foldId = fold_dict["fold"]
        idx_treino = fold_dict["train_idx"]
        idx_teste = fold_dict["test_idx"]

        logging.info(f"\n=== Fold {foldId + 1} ===")

        X_treino = X.iloc[idx_treino]
        y_treino = y.iloc[idx_treino]

        X_teste = X.iloc[idx_teste]
        y_teste = y.iloc[idx_teste]
        
        modelo_filename = os.path.join(
            config.path_modelos,
            f"kmeansc_model_fold_{foldId + 1}.pkl"
        )

        matriz_filename = os.path.join(
            config.path_matrizes,
            f"matriz_{foldId + 1}.pkl"
        )

        if os.path.exists(modelo_filename):
            
            logging.info("Carregando modelo salvo...")
            modelo = carregar_objeto(modelo_filename)
            
        else:
            logging.info(f"Treinando modelo {foldId + 1}...")
            
            counts = y_treino.value_counts()

            if counts.min() >= 2:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_treino,
                    y_treino,
                    stratify=y_treino,
                    test_size=0.2,
                    random_state=1
                )
            else:
                logging.info("Fold contém classe com apenas 1 amostra. Split sem estratificação.")

                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_treino,
                    y_treino,
                    test_size=0.2,
                    random_state=1
                )
            
            modelo, melhor_k, _ = selecionar_melhor_kmeans(
                k_values,
                X_tr.values,
                X_val.values,
                y_tr.values,
                y_val.values,
                ka
            )
            
            salvar_objeto(modelo, modelo_filename)
            logging.info("Modelo salvo.")
            
        f1, topk, y_pred, y_scores = prever_kmeansc(
            modelo,
            X_teste.values,
            y_teste.values,
            ka
        )
        
        salvar_objeto({
            "fold": foldId,
            "y_true": y_teste,
            "y_scores": y_scores,
            "classes": modelo["labels"]
        }, matriz_filename)
        
        logging.info(f"F1-score: {f1:.2f}")
        logging.info(f"Top-{ka} Accuracy: {topk:.2f}")

        acuracias.append(f1)
        topkScores.append(topk)
        
    return acuracias, topkScores

def main():

    ka = 5

    print(f"VERSÃO = {DATA_VERSION}")
    print(f"Top-K = {ka}")
    
    logging.info("Selecione o tipo de dataset:\n1 - Segmentado")

    opcoes = {"1": "segmentado", "2": "completo"}
    tipo = opcoes.get(input("Digite sua escolha (1-2): ").strip())

    if tipo is None:
        logging.error("Escolha inválida!")
        return

    config = DATASET_CONFIGS[tipo]

    df = carregar_objeto(config.path_dataframe)

    df = df.dropna(subset=["roi_label"])
    df["roi_label"] = df["roi_label"].astype(str)

    X = df.drop(columns=["roi_label", "audioSource"])
    y = df["roi_label"]

    acuracias, topkAcuracias = do_cv_kmeansc(
        X, y, ka, config, k_values=[1, 5, 10, 20, 50]
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