import os
import numpy as npy
import pickle
import itertools
import logging
import pandas as pd

from datetime import datetime
from dataclasses import dataclass
from joblib import Parallel, delayed

from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
        path_matrizes=f"{DATA_VERSION}/matrizesProba_xgb_treinoSegmentado",
        path_modelos=f"{DATA_VERSION}/modelos_xgb_treinoSegmentado",
        path_folds=f"{DATA_VERSION}/folds_audiosSegmentados_xgb"
    ),
    "completo": DatasetConfig(
        nome="Áudios Completos",
        path_dataframe=f"../dataframes/{DATA_VERSION}/dataframeAudioCompleto.pkl",
        path_matrizes=f"{DATA_VERSION}/matrizesProba_xgb_treinoCompleto",
        path_modelos=f"{DATA_VERSION}/modelos_xgb_treinoCompleto",
        path_folds=f"{DATA_VERSION}/folds_audiosCompletos_xgb"
    )
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

def selecionar_melhor_xgb(param_grid, X_train, X_val, y_train, y_val, num_classes, n_jobs=4):

    def treinar(params):
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            tree_method="hist", #aproximately Greedy Algorithm
            #n_jobs=n_jobs,
            n_jobs=1,
            eval_metric="mlogloss",
            n_estimators=300,
            early_stopping_rounds=20,
            **params
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        logging.info(f"Best iteration: {model.best_iteration}")
        
        pred = model.predict(X_val)
        return f1_score(y_val, pred, average="macro")

    combinacoes = list(itertools.product(*param_grid.values()))
    dicts_param = [dict(zip(param_grid.keys(), combo)) for combo in combinacoes]

    scores = Parallel(n_jobs=n_jobs)(
        delayed(treinar)(p) for p in dicts_param
    )

    best_idx = npy.argmax(scores)
    best_params = dicts_param[best_idx]
    best_score = scores[best_idx]

    logging.info(f"Melhor XGBoost: {best_params}, F1 Val: {best_score:.2f}")

    final_model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        tree_method="hist",
        #n_jobs=n_jobs,
        n_jobs=1,
        eval_metric="mlogloss",
        n_estimators=300,
        early_stopping_rounds=20,
        **best_params
    )

    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return final_model, best_params, best_score

#################################################

def exibir_resultados(model, X_test, y_test, ka):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    if len(y_test) != len(y_pred):
        logging.error(f"Tamanho inconsistente: y_test ({len(y_test)}) e y_pred ({len(y_pred)})")
        return

    f1 = f1_score(y_test, y_pred, average="macro")

    mask = npy.isin(y_test, model.classes_)
    y_test_filtrado = y_test[mask]
    y_proba_filtrado = y_proba[mask]

    idxs = [npy.where(model.classes_ == c)[0][0] for c in npy.unique(y_test_filtrado)]
    y_proba_filtrado = y_proba_filtrado[:, idxs]

    topk = top_k_accuracy_score(y_test_filtrado, y_proba_filtrado, k=ka, labels=npy.unique(y_test_filtrado))

    logging.info(f"F1-score XGB: {f1:.2f}")
    logging.info(f"Top-{ka} Accuracy: {topk:.2f}")

#################################################

def do_cv_xgb(X, y, ka, cv_splits, groups, config, param_grid):

    preparar_pastas(config.path_matrizes, config.path_modelos, config.path_folds)

    counts = pd.Series(y).value_counts()
    classes_validas = counts[counts >= cv_splits].index
    filtro = npy.isin(y, classes_validas)
    
    X, y, groups = X[filtro], y[filtro], groups[filtro]
    
    acuracias, topkScores = [], []

    skf = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=1)

    for foldId, (train_idx, test_idx) in enumerate(skf.split(X, y, groups)):

        logging.info(f"\n=== Fold {foldId + 1} ===")

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        
        mask_teste = y_test.isin(le.classes_)
        X_test = X_test[mask_teste]
        y_test = y_test[mask_teste]

        y_test_encoded = le.transform(y_test)

        num_classes = len(le.classes_)

        salvar_objeto(X_train, os.path.join(config.path_folds, f"X_treino_fold_{foldId + 1}.pkl"))
        salvar_objeto(y_train, os.path.join(config.path_folds, f"y_treino_fold_{foldId + 1}.pkl"))
        salvar_objeto(X_test, os.path.join(config.path_folds, f"X_teste_fold_{foldId + 1}.pkl"))
        salvar_objeto(y_test, os.path.join(config.path_folds, f"y_teste_fold_{foldId + 1}.pkl"))

        modelo_filename = os.path.join(config.path_modelos, f"xgb_model_fold_{foldId + 1}.pkl")
        matriz_filename = os.path.join(config.path_matrizes, f"matriz_{foldId + 1}.pkl")

        if os.path.exists(modelo_filename):
            logging.info(f"Carregando modelo existente do fold {foldId + 1}...")

            modelo = carregar_objeto(modelo_filename)

            ss = StandardScaler()
            ss.fit(X_train)
            X_test = ss.transform(X_test)

            y_pred = modelo.predict(X_test)
            y_proba = modelo.predict_proba(X_test)

            if not os.path.exists(matriz_filename):
                salvar_objeto(
                    {"fold": foldId,
                     "y_true": y_test_encoded,
                     "y_proba": y_proba,
                     "classes": modelo.classes_},
                    matriz_filename)
                logging.info(f"Matriz criada para fold {foldId + 1}.")

        else:

            logging.info(f"Treinando modelo do fold {foldId + 1}...")

            _, counts = npy.unique(y_train_encoded, return_counts=True)

            if counts.min() < 2:
                logging.warning(
                    f"Fold {foldId + 1}: classe com < 2 amostras no treino "
                    "→ split sem estratificação"
                )
                
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train_encoded,
                    test_size=0.2,
                    random_state=1
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train_encoded,
                    stratify=y_train_encoded,
                    test_size=0.2,
                    random_state=1
                )

            ss = StandardScaler()
            ss.fit(X_train)
            X_train = ss.transform(X_train)
            X_val = ss.transform(X_val)
            X_test = ss.transform(X_test)

            train_unique = npy.unique(y_train)
            if len(train_unique) < num_classes:
                logging.warning(f"Fold {foldId + 1}: Apenas {len(train_unique)} classes no treino de {num_classes} totais")

            modelo, _, _ = selecionar_melhor_xgb(
                param_grid, X_train, X_val, y_train, y_val, num_classes
            )

            y_pred = modelo.predict(X_test)
            y_proba = modelo.predict_proba(X_test)

            salvar_objeto(
                {"fold": foldId,
                 "y_true": y_test_encoded,
                 "y_proba": y_proba,
                 "classes": modelo.classes_},
                matriz_filename)

            salvar_objeto(modelo, modelo_filename)
            logging.info(f"Modelo salvo no fold {foldId + 1}.")

        f1 = f1_score(y_test_encoded, y_pred, average="macro")
        topk = top_k_accuracy_score(y_test_encoded, y_proba, k=ka, labels=modelo.classes_)

        exibir_resultados(modelo, X_test, y_test_encoded, ka)

        acuracias.append(f1)
        topkScores.append(topk)

    return acuracias, topkScores


#################################################

def main():

    cv = 10
    ka = 3

    print(f"VERSÃO = {DATA_VERSION}")
    print(f"Top-K = {ka}")

    logging.info("Selecione o tipo de dataset:\n1 - Segmentado\n2 - Completo")

    opcoes = {"1": "segmentado", "2": "completo"}
    tipo = opcoes.get(input("Digite sua escolha (1-2): ").strip())

    if tipo is None:
        logging.error("Escolha inválida!")
        return

    config = DATASET_CONFIGS[tipo]

    df = carregar_objeto(config.path_dataframe)

    X = df.drop(columns=["roi_label", "audioSource"])
    y = df["roi_label"]
    groups = df["audioSource"]

    param_grid = {
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0]
        #"n_estimators": [200, 400]
    }

    acuracias, topkAcuracias = do_cv_xgb(X, y, ka, cv, groups, config, param_grid)

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