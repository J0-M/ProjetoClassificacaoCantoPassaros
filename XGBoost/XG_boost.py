import os
import numpy as np
import pickle
import itertools
import logging

from datetime import datetime
from dataclasses import dataclass
from joblib import Parallel, delayed

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, top_k_accuracy_score

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
        path_matrizes=f"{DATA_VERSION}/matrizesProba_xgb_treinoSegmentado",
        path_modelos=f"{DATA_VERSION}/modelos_xgb_treinoSegmentado",
        path_folds=f"../folds/{DATA_VERSION}/segmentado"
    ),
    "completo": DatasetConfig(
        nome="Áudios Completos",
        path_dataframe=f"../dataframes/{DATA_VERSION}/dataframeAudioCompleto.pkl",
        path_matrizes=f"{DATA_VERSION}/matrizesProba_xgb_treinoCompleto",
        path_modelos=f"{DATA_VERSION}/modelos_xgb_treinoCompleto",
        path_folds=f"../folds/{DATA_VERSION}/completo"
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

    best_idx = np.argmax(scores)
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

def calcular_metricas(y_true, y_proba, classes, ka):
    y_pred = classes[np.argmax(y_proba, axis=1)]
    
    f1 = f1_score(y_true, y_pred, average="macro")

    mask = np.isin(y_true, classes)
    
    if mask.sum() == 0:
        return f1, 0
    
    topk = top_k_accuracy_score(
        y_true[mask],
        y_proba[mask],
        k=ka,
        labels=classes
    )

    #logging.info(f"F1-score do XGBoost: {f1:.2f}")
    #logging.info(f"Top-{ka} Accuracy: {topk:.2f}")
    
    return f1, topk

#################################################

def do_cv_xgb(X, y, ka, n_splits, config, param_grid):

    path_matrizes = os.path.join(config.path_matrizes, f"{n_splits}fold")
    path_modelos = os.path.join(config.path_modelos, f"{n_splits}fold")
            
    preparar_pastas(path_matrizes, path_modelos)

    path_folds = os.path.join(config.path_folds, f"stratified_group_kfold_{n_splits}.pkl")
        
    if not os.path.exists(path_folds):
        raise FileNotFoundError("Folds ainda não foram gerados.")
        
    folds = carregar_objeto(path_folds)

    acuracias, topkScores = [], []

    for fold_dict in folds:

        foldId = fold_dict["fold"]
        idx_treino = fold_dict["train_idx"]
        idx_teste = fold_dict["test_idx"]

        logging.info(f"\n=== {n_splits}-FOLD | Fold {foldId + 1} ===")

        X_train = X.iloc[idx_treino]
        y_train = y.iloc[idx_treino]

        X_test = X.iloc[idx_teste]
        y_test = y.iloc[idx_teste]

        modelo_filename = os.path.join(
            config.path_modelos, f"xgb_model_fold_{foldId + 1}.pkl"
        )

        matriz_filename = os.path.join(
            config.path_matrizes, f"matriz_{foldId + 1}.pkl"
        )

        if os.path.exists(matriz_filename):

            logging.info("Carregando matriz salva...")

            matriz = carregar_objeto(matriz_filename)

            y_true = matriz["y_true"]
            y_proba = matriz["y_proba"]
            classes = matriz["classes"]
            
            print("Classes fora do modelo:", set(y_true) - set(classes))

            y_pred = classes[np.argmax(y_proba, axis=1)]

            f1, topk = calcular_metricas(y_true, y_proba, classes, ka)
            
        else:

            if os.path.exists(modelo_filename):

                logging.info("Carregando modelo salvo...")

                obj = carregar_objeto(modelo_filename)
                modelo = obj["modelo"]
                le = obj["label_encoder"]
                ss = obj["scaler"]

            else:

                logging.info("Treinando modelo...")

                print(y_train.value_counts().min())
                
                counts = y_train.value_counts()
                classes_validas = counts[counts >= 2].index
                
                logging.info(f"Espécies antes do filtro: {len(counts)}")
                logging.info(f"Amostras antes do filtro: {len(y_train)}")

                mask = y_train.isin(classes_validas)
                X_train = X_train[mask]
                y_train = y_train[mask]
                
                logging.info(f"Espécies depois do filtro: {y_train.nunique()}")
                logging.info(f"Amostras depois do filtro: {len(y_train)}")
                
                logging.info(f"Espécies removidas: {len(set(counts.index) - set(classes_validas))}")

                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train,
                    y_train,
                    test_size=0.2,
                    stratify=y_train,
                    shuffle=True,
                    random_state=10
                )

                le = LabelEncoder()
                y_tr = le.fit_transform(y_tr)

                mask_val = y_val.isin(le.classes_)
                X_val = X_val[mask_val]
                y_val = y_val[mask_val]
                y_val = le.transform(y_val)

                num_classes = len(le.classes_)

                ss = StandardScaler()
                ss.fit(X_tr)

                X_tr = ss.transform(X_tr)
                X_val = ss.transform(X_val)

                modelo, _, _ = selecionar_melhor_xgb(
                    param_grid,
                    X_tr,
                    X_val,
                    y_tr,
                    y_val,
                    num_classes
                )

                salvar_objeto(
                    {
                        "modelo": modelo,
                        "label_encoder": le,
                        "scaler": ss,
                    },
                    modelo_filename,
                )

                logging.info("Modelo salvo.")

            logging.info("Calculando matriz...")

            # filtrar apenas classes vistas
            mask_test = y_test.isin(le.classes_)
            X_test_filtrado = X_test[mask_test]
            y_test_filtrado = y_test[mask_test]

            y_test_encoded = le.transform(y_test_filtrado)
            X_test_filtrado = ss.transform(X_test_filtrado)

            y_pred = modelo.predict(X_test_filtrado)
            y_proba = modelo.predict_proba(X_test_filtrado)

            classes = modelo.classes_

            f1 = f1_score(y_test_encoded, y_pred, average="macro")

            topk = top_k_accuracy_score(
                y_test_encoded,
                y_proba,
                k=ka,
                labels=classes
            )

            salvar_objeto(
                {
                    "fold": foldId,
                    "y_true": y_test_encoded,
                    "y_proba": y_proba,
                    "classes": classes,
                },
                matriz_filename,
            )

            logging.info("Matriz salva.")

        logging.info(f"F1-score: {f1:.2f}")
        logging.info(f"Top-{ka} Accuracy: {topk:.2f}")

        acuracias.append(f1)
        topkScores.append(topk)

    return acuracias, topkScores

#################################################

def main():

    ka = int(input("Hiperparâmetro K (Top-K): "))
    
    print("\n##########################\n")

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

    param_grid = {
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0]
        #"n_estimators": [200, 400]
    }

    logging.info(f"Quantidade de amostras: {X.shape}, Quantidade de classes: {y.nunique()}")  
    resultados = {}
        
    for n_splits in CV_SPLITS:
        logging.info(f"EXPERIMENTO {n_splits}-FOLD")

        acuracias, topkAcuracias = do_cv_xgb(X, y, ka, n_splits, config, param_grid)
        
        resultados[n_splits] = {
            "f1": acuracias,
            "topk": topkAcuracias
        }

        print(f"\n-- TESTE {config.nome.upper()} --")
        print("F1-Score Macro:")
        print(f"min: {min(acuracias):.2f}, max: {max(acuracias):.2f}, avg ± std: {np.mean(acuracias):.2f} ± {np.std(acuracias):.2f}")
        print(f"\nTop-{ka} Score:")
        print(f"min: {min(topkAcuracias):.2f}, max: {max(topkAcuracias):.2f}, avg ± std: {np.mean(topkAcuracias):.2f} ± {np.std(topkAcuracias):.2f}")
        
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