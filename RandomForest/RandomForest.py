import os
import numpy as npy
import pickle
import logging

from datetime import datetime
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
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
        path_matrizes=f"{DATA_VERSION}/matrizesProba_rf_treinoSegmentado",
        path_modelos=f"{DATA_VERSION}/modelos_rf_treinoSegmentado",
        path_folds=f"{DATA_VERSION}/folds_audiosSegmentados_rf"
    ),
    "completo": DatasetConfig(
        nome="Áudios Completos",
        path_dataframe=f"../dataframes/{DATA_VERSION}/dataframeAudioCompleto.pkl",
        path_matrizes=f"{DATA_VERSION}/matrizesProba_rf_treinoCompleto",
        path_modelos=f"{DATA_VERSION}/modelos_rf_treinoCompleto",
        path_folds=f"{DATA_VERSION}/folds_audiosCompletos_rf"
    ),
    "passaro_unico": DatasetConfig(
        nome="Áudios Pássaro Único",
        path_dataframe=f"../dataframes/{DATA_VERSION}/dataframeAudiosPassaroUnico.pkl",
        path_matrizes=f"{DATA_VERSION}/matrizesProba_rf_treinoAudiosPassaroUnico",
        path_modelos=f"{DATA_VERSION}/modelos_rf_treinoAudiosPassaroUnicoSegmentado",
        path_folds=f"{DATA_VERSION}/folds_audiosPassaroUnicoSegmentado_rf"
    ),
}

#################################################

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

def selecionar_melhor_rf(X_treino, X_val, y_treino, y_val, n_jobs=4):
    """
    Faz busca em grade (GridSearchCV) para encontrar os melhores hiperparâmetros
    do RandomForestClassifier com base no F1-score macro.
    """
    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False]
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=n_jobs,
        scoring="f1_macro",
        verbose=1
    )

    grid_search.fit(X_treino, y_treino)

    melhor_rf = grid_search.best_estimator_
    melhor_params = grid_search.best_params_
    melhor_val = grid_search.best_score_

    logging.info(f"Melhor Random Forest - Params: {melhor_params}, F1 Val: {melhor_val:.2f}")

    # Treina novamente o melhor modelo com treino + validação
    melhor_rf.fit(
        npy.vstack((X_treino, X_val)),
        npy.hstack((y_treino, y_val))
    )

    return melhor_rf, melhor_params, melhor_val


def exibir_resultados(modelo, X_test_scaled, y_test, ka):
    y_pred = modelo.predict(X_test_scaled)
    y_proba = modelo.predict_proba(X_test_scaled)

    f1 = f1_score(y_test, y_pred, average="macro")
    mask = y_test.isin(modelo.classes_)
    y_test_filtrado = y_test[mask]
    y_proba_filtrado = y_proba[mask.values]
    
    classes_presentes = npy.intersect1d(modelo.classes_, npy.unique(y_test_filtrado))
    idxs = [npy.where(modelo.classes_ == c)[0][0] for c in classes_presentes]
    y_proba_filtrado = y_proba_filtrado[:, idxs]
    
    topk_acc = top_k_accuracy_score(y_test_filtrado, y_proba_filtrado, k=ka, labels=classes_presentes)

    logging.info(f"F1-score: {f1:.2f}")
    logging.info(f"Top-{ka} Accuracy: {topk_acc:.2f}")


def do_cv_rf(X, y, ka, cv_splits, groups, config: DatasetConfig):
    skf = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=1)

    preparar_pastas(config.path_matrizes, config.path_modelos, config.path_folds)
    acuracias, topkScores = [], []
    
    counts = y.value_counts()
    classes_validas = counts[counts >= cv_splits].index
    filtro = y.isin(classes_validas)
    X, y, groups = X[filtro], y[filtro], groups[filtro]
    
    for foldId, (treino_idx, teste_idx) in enumerate(skf.split(X, y, groups)):
        logging.info(f"\n=== Fold {foldId + 1} ===")
        
        matriz_filename = os.path.join(config.path_matrizes, f"matriz_{foldId + 1}.pkl")
        modelo_filename = os.path.join(config.path_modelos, f"rf_model_fold_{foldId + 1}.pkl")
        
        ss = StandardScaler()
        
        X_treino, y_treino = X.iloc[treino_idx], y.iloc[treino_idx]
        X_teste, y_teste = X.iloc[teste_idx], y.iloc[teste_idx]
        
        if config.nome in ["Áudios Segmentados", "Áudios Pássaro Único"]:
            classes_validas_fold = y_treino.value_counts()[lambda x: x >= cv_splits].index
            mask_treino = y_treino.isin(classes_validas_fold)
            mask_teste = y_teste.isin(classes_validas_fold)
            X_treino, y_treino = X_treino[mask_treino], y_treino[mask_treino]
            X_teste, y_teste = X_teste[mask_teste], y_teste[mask_teste]
        
        salvar_objeto(X_treino, os.path.join(config.path_folds, f"X_treino_fold_{foldId + 1}.pkl"))
        salvar_objeto(y_treino, os.path.join(config.path_folds, f"y_treino_fold_{foldId + 1}.pkl"))
        salvar_objeto(X_teste, os.path.join(config.path_folds, f"X_teste_fold_{foldId + 1}.pkl"))
        salvar_objeto(y_teste, os.path.join(config.path_folds, f"y_teste_fold_{foldId + 1}.pkl"))
        
        if os.path.exists(modelo_filename):
            logging.info(f"Carregando modelo do fold {foldId + 1}...")
            rf = carregar_objeto(modelo_filename)
            ss.fit(X_treino)
            X_teste = ss.transform(X_teste)
            y_pred = rf.predict(X_teste)
            y_proba = rf.predict_proba(X_teste)

            if not os.path.exists(matriz_filename):
                matriz_info = {"fold": foldId, "y_true": y_teste.values, "y_proba": y_proba, "classes": rf.classes_}
                salvar_objeto(matriz_info, matriz_filename)
                logging.info(f"Probabilidades salvas em {matriz_filename}")
        
        else:
            logging.info(f"Criando modelo do fold {foldId + 1}...")

            X_treino, X_val, y_treino, y_val = train_test_split(
                X_treino, y_treino, stratify=y_treino, test_size=0.2, random_state=1)

            ss.fit(X_treino)
            X_treino = ss.transform(X_treino)
            X_teste = ss.transform(X_teste)
            X_val = ss.transform(X_val)

            rf, _, _ = selecionar_melhor_rf(X_treino, X_val, y_treino, y_val)
            y_pred = rf.predict(X_teste)
            y_proba = rf.predict_proba(X_teste)
        
            salvar_objeto({"fold": foldId, "y_true": y_teste.values, "y_proba": y_proba, "classes": rf.classes_},
                          matriz_filename)
            logging.info(f"Probabilidades salvas em {matriz_filename}")
            salvar_objeto(rf, modelo_filename)
            logging.info(f"Modelo salvo em {modelo_filename}")
            
        f1 = f1_score(y_teste, y_pred, average="macro")
        desconhecidas = (~y_teste.isin(rf.classes_)).sum()
        logging.info(f"Amostras com classe 'não vista no treino': {desconhecidas}/{len(y_teste)}")
        
        classes_treino = set(rf.classes_)
        mask = y_teste.isin(classes_treino)
        y_teste_filtrado = y_teste[mask]
        y_proba_filtrado = y_proba[mask.values]
        
        topk = top_k_accuracy_score(y_teste_filtrado, y_proba_filtrado, k=ka, labels=rf.classes_)
        exibir_resultados(rf, X_teste, y_teste, ka)

        acuracias.append(f1)
        topkScores.append(topk)
    
    return acuracias, topkScores


def main():
    cv = 10
    ka = 3
    print(f"VERSÃO = {DATA_VERSION}")
    
    logging.info("Selecione o tipo de dataset:\n1 - Segmentado\n2 - Completo\n3 - Pássaro Único")
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

    logging.info(f"Quantidade de amostras: {X.shape}, Quantidade de classes: {y.nunique()}")
    
    acuracias, topkAcuracias = do_cv_rf(X, y, ka, cv, groups, config)
    
    print(f"\n-- TESTE {config.nome.upper()} --")
    print("F1-Score Macro:")
    print(f"min: {min(acuracias):.2f}, max: {max(acuracias):.2f}, avg ± std: {npy.mean(acuracias):.2f} ± {npy.std(acuracias):.2f}")
    print(f"\nTop-{ka} Score:")
    print(f"min: {min(topkAcuracias):.2f}, max: {max(topkAcuracias):.2f}, avg ± std: {npy.mean(topkAcuracias):.2f} ± {npy.std(topkAcuracias):.2f}")


if __name__ == '__main__':
    startTime = datetime.now()
    main()
    endTime = datetime.now()
    print("Tempo de execução = ", endTime - startTime)
