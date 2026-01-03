import os
import numpy as npy
import pickle
import logging

from datetime import datetime
from dataclasses import dataclass

from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, top_k_accuracy_score

DATA_VERSION = "v2_media_std"

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
        path_matrizes=f"{DATA_VERSION}/matrizesProba_knn_treinoSegmentado",
        path_modelos=f"{DATA_VERSION}/modelos_knn_treinoSegmentado",
        path_folds=f"{DATA_VERSION}/folds_audiosSegmentados_knn"
    ),
    "completo": DatasetConfig(
        nome="Áudios Completos",
        path_dataframe=f"../dataframes/{DATA_VERSION}/dataframeAudioCompleto.pkl",
        path_matrizes=f"{DATA_VERSION}/matrizesProba_knn_treinoCompleto",
        path_modelos=f"{DATA_VERSION}/modelos_knn_treinoCompleto",
        path_folds=f"{DATA_VERSION}/folds_audiosCompletos_knn"
    ),
    "passaro_unico": DatasetConfig(
        nome="Áudios Pássaro Único",
        path_dataframe=f"../dataframes/{DATA_VERSION}/dataframeAudiosPassaroUnico.pkl",
        path_matrizes=f"{DATA_VERSION}/matrizesProba_knn_treinoPassaroUnico",
        path_modelos=f"{DATA_VERSION}/modelos_knn_treinoPassaroUnico",
        path_folds=f"{DATA_VERSION}/folds_audiosPassaroUnico_knn"
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
    
    logging.info(f"Acurácia no teste: {f1_score(y_teste, knn_final.predict(X_teste), average='macro'):.2f}")
    
    return knn_final, melhor_k, melhor_val


def exibir_resultados(knn, X_test_scaled, y_test, ka):
    
    y_pred = knn.predict(X_test_scaled)
    y_proba = knn.predict_proba(X_test_scaled)

    f1 = f1_score(y_test, y_pred, average="macro")
    mask = y_test.isin(knn.classes_)
    y_test_filtrado = y_test[mask]
    y_proba_filtrado = y_proba[mask.values]
    
    classes_presentes = npy.intersect1d(knn.classes_, npy.unique(y_test_filtrado))
    idxs = [npy.where(knn.classes_ == c)[0][0] for c in classes_presentes]
    y_proba_filtrado = y_proba_filtrado[:, idxs]
    
    topk_acc = top_k_accuracy_score(y_test_filtrado, y_proba_filtrado, k=ka, labels=classes_presentes)

    logging.info(f"F1-score do KNN: {f1:.2f}")
    logging.info(f"Top-{ka} Accuracy: {topk_acc:.2f}")
    
def treinar_knn_com_validacao_cruzada(X, y, ka, groups, config: DatasetConfig):
    
    k_vias = 10
    skf = StratifiedGroupKFold(n_splits=k_vias, shuffle=True, random_state=10)
    
    preparar_pastas(config.path_matrizes, config.path_modelos, config.path_folds)
    
    counts = y.value_counts()
    classes_validas = counts[counts >= k_vias].index
    filtro = y.isin(classes_validas)
    
    X, y, groups = X[filtro], y[filtro], groups[filtro]
    
    acuracias, topKScores = [], []
        
    for foldId, (idx_treino, idx_teste) in enumerate(skf.split(X, y, groups)):
        
        logging.info(f"\n=== Fold {foldId + 1} ===")
        
        sources_train = set(groups.iloc[idx_treino])
        sources_test = set(groups.iloc[idx_teste])
        intersec = sources_train.intersection(sources_test)
        assert len(intersec) == 0, f"Vazamento detectado em fold {foldId + 1} nos audioSource: {intersec}"
        
        matriz_filename = os.path.join(config.path_matrizes, f"matriz_{foldId + 1}.pkl")
        modelo_filename = os.path.join(config.path_modelos, f"KNN_model_fold_{foldId + 1}.pkl")
        
        X_treino, y_treino = X.iloc[idx_treino], y.iloc[idx_treino]
        X_teste, y_teste = X.iloc[idx_teste], y.iloc[idx_teste]
        
        salvar_objeto(X_treino, os.path.join(config.path_folds, f"X_treino_fold_{foldId + 1}.pkl"))
        salvar_objeto(y_treino, os.path.join(config.path_folds, f"y_treino_fold_{foldId + 1}.pkl"))
        salvar_objeto(X_teste, os.path.join(config.path_folds, f"X_teste_fold_{foldId + 1}.pkl"))
        salvar_objeto(y_teste, os.path.join(config.path_folds, f"y_teste_fold_{foldId + 1}.pkl"))
        
        if os.path.exists(modelo_filename):
            logging.info(f"Carregando modelo salvo do fold {foldId + 1}...")
            knn = carregar_objeto(modelo_filename)
            ss = StandardScaler().fit(X_treino)
            X_teste = ss.transform(X_teste)
            
        else:
            logging.info(f"Treinando novo modelo para o fold {foldId + 1}...")
            
            counts_treino = y_treino.value_counts()
            classes_validas_treino = counts_treino[counts_treino >= 2].index
            filtro_treino = y_treino.isin(classes_validas_treino)

            X_treino, y_treino = X_treino[filtro_treino], y_treino[filtro_treino]
            
            X_treino, X_val, y_treino, y_val = train_test_split(
                X_treino, y_treino, test_size=0.2, stratify=y_treino, shuffle=True, random_state=10)

            ss = StandardScaler().fit(X_treino)
            X_treino, X_teste, X_val = ss.transform(X_treino), ss.transform(X_teste), ss.transform(X_val)

            knn, _, _ = selecionar_melhor_k(range(1,30,2), X_treino, X_val, y_treino, y_val, X_teste, y_teste)
            salvar_objeto(knn, modelo_filename)
        
        y_pred = knn.predict(X_teste)
        y_proba = knn.predict_proba(X_teste)
        
        matriz_info = {"fold": foldId, "y_true": y_teste.values, "y_proba": y_proba, "classes": knn.classes_}
        salvar_objeto(matriz_info, matriz_filename)

        f1 = f1_score(y_teste, y_pred, average="macro")
        acuracias.append(f1)
        
        desconhecidas = (~y_teste.isin(knn.classes_)).sum()
        logging.info(f"Amostras com classe 'não vista no treino': {desconhecidas}/{len(y_teste)}")
        
        mask = y_teste.isin(set(knn.classes_))
        y_teste_filtrado, y_proba_filtrado = y_teste[mask], y_proba[mask.values]
        
        classes_presentes = npy.intersect1d(knn.classes_, npy.unique(y_teste_filtrado))
        idxs = [npy.where(knn.classes_ == c)[0][0] for c in classes_presentes]
        y_proba_filtrado = y_proba_filtrado[:, idxs]
        
        topKScores.append(top_k_accuracy_score(y_teste_filtrado, y_proba_filtrado, k=ka, labels=classes_presentes))
        
        exibir_resultados(knn, X_teste, y_teste, ka)
    
    return acuracias, topKScores

def main():
    ka = 5  # Hiperparâmetro do Top-K
    
    print(f"VERSÃO = {DATA_VERSION}")
    print(f"Top-K = {ka}")
    
    # Seleção do dataset
    print("Selecione o tipo de dataset:")
    print("1 - Segmentado")
    print("2 - Completo")
    print("3 - Pássaro Único")
    
    opcoes = {"1": "segmentado", "2": "completo", "3": "passaro_unico"}
    tipo = opcoes.get(input("Digite sua escolha (1-3): ").strip())
    
    if tipo is None:
        logging.error("Escolha inválida!")
        return

    config = DATASET_CONFIGS[tipo]
    
    if not os.path.exists(config.path_dataframe):
        logging.error("Dataframe não encontrado!")
        return
    
    df = carregar_objeto(config.path_dataframe)
    logging.info("Dataframe carregado com sucesso!")

    X = df.drop(columns=["roi_label", "audioSource"])
    y = df["roi_label"]
    groups = df["audioSource"]

    logging.info(f"Quantidade de amostras: {X.shape}")
    logging.info(f"Quantidade de espécies: {y.nunique()}")
    
    acuracias, topKAcuracias = treinar_knn_com_validacao_cruzada(X, y, ka, groups, config)
    
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