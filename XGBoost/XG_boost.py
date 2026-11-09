import os
import numpy as np
import pickle
import itertools
import logging

from datetime import datetime
from dataclasses import dataclass
from joblib import Parallel, delayed

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, top_k_accuracy_score

from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler

CV_SPLITS = [5, 10]

versoes_validas = ["v1_media", "v2_media_std", "v3_media_std_freq", "v4_novas_features", "v5_novo_filtro", "v6_perch"]

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

estrategias_validas = ["none", "ros", "smote"]

NOMES_ESTRATEGIAS = {
    "none":  "Sem balanceamento (baseline)",
    "ros":   "Random Oversampling",
    "smote": "SMOTE (Synthetic Minority Over-sampling)",
}

print("\nSelecione a estratégia de balanceamento:")
for i, e in enumerate(estrategias_validas, 1):
    print(f"{i} - {e:<6s} ({NOMES_ESTRATEGIAS[e]})")

entrada = input("Digite o número da estratégia desejada: ").strip()

if not entrada.isdigit():
    logging.error("Entrada inválida! Digite um número.")
    exit()

idx = int(entrada) - 1

if idx < 0 or idx >= len(estrategias_validas):
    logging.error("Estratégia inválida!")
    exit()
    
REBALANCE = estrategias_validas[idx]

########## CONFIGURAÇÃO LOGGING #################
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

@dataclass
class DatasetConfig:
    nome: str
    path_matrizes: str
    path_modelos: str
    path_folds: str

DATASET_CONFIGS = {
    "segmentado": DatasetConfig(
        nome="Áudios Segmentados",
        path_matrizes=f"{DATA_VERSION}/{REBALANCE}/matrizesProba_xgboost_treinoSegmentado",
        path_modelos=f"{DATA_VERSION}/{REBALANCE}/modelos_xgboost_treinoSegmentado",
        path_folds=f"../folds/{DATA_VERSION}/segmentado"
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

def balancear(X, y, strategy, random_state=10):
    if strategy == "none":
        return X, y

    if strategy == "ros":
        return RandomOverSampler(random_state=random_state).fit_resample(X, y)

    if strategy == "smote":
        min_count = min(Counter(y).values())
        k = min(5, min_count - 1)
        
        if k < 1:
            logging.warning(f"Classe minoritária tem {min_count} amostra(s); SMOTE inviável, usando ROS como fallback.")
            return RandomOverSampler(random_state=random_state).fit_resample(X, y)
        
        return SMOTE(random_state=random_state, k_neighbors=k).fit_resample(X, y)
    
    if strategy == "all":
        raise ValueError("'all' não é uma estratégia de balanceamento; "
                        "é uma opção de execução (use lista de estratégias).")

    raise ValueError(f"Estratégia de balanceamento desconhecida: {strategy}")

#################################################

def selecionar_melhor_xgboost(param_grid, X_train, X_val, y_train, y_val, num_classes, n_jobs=4):

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
        return f1_score(y_val, pred, average="macro", labels=np.unique(y_val)), int(model.best_iteration)

    combinacoes = list(itertools.product(*param_grid.values()))
    dicts_param = [dict(zip(param_grid.keys(), combo)) for combo in combinacoes]

    resultados = Parallel(n_jobs=n_jobs)(
        delayed(treinar)(p) for p in dicts_param
    )
    
    scores = [r[0] for r in resultados]
    iters  = [r[1] for r in resultados]

    best_idx = np.argmax(scores)
    best_params = dicts_param[best_idx]
    best_iter = iters[best_idx]
    best_score = scores[best_idx]

    logging.info(f"Melhor XGBoost: {best_params}, best_iter={best_iter}, F1 Val: {best_score:.2f}")

    return best_params, best_iter, best_score

#################################################

def calcular_metricas(y_true, y_proba, classes, ka):
    y_pred = classes[np.argmax(y_proba, axis=1)]
    
    f1 = f1_score(y_true, y_pred, average="macro", labels=np.unique(y_true))
    
    topk = top_k_accuracy_score(
        y_true,
        y_proba,
        k=ka,
        labels=classes
    )

    #logging.info(f"F1-score do XGBoost: {f1:.2f}")
    #logging.info(f"Top-{ka} Accuracy: {topk:.2f}")
    
    return f1, topk

#################################################

def do_cv_xgb(ka, n_splits, config, param_grid):

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
                        
        X_train = fold_dict["X_train"]
        y_train = fold_dict["y_train"]
        
        X_test = fold_dict["X_test"]
        y_test = fold_dict["y_test"]
        
        logging.info(f"Amostras treino: {len(X_train)}")
        logging.info(f"Amostras teste: {len(X_test)}")
        logging.info(f"Espécies treino: {y_train.nunique()}")
        logging.info(f"Espécies teste: {y_test.nunique()}")

        logging.info(f"\n=== {n_splits}-FOLD | Fold {foldId + 1} ===")

        modelo_filename = os.path.join(path_modelos, f"xgboost_model_fold_{foldId + 1}.pkl")
        matriz_filename = os.path.join(path_matrizes, f"matriz_{foldId + 1}.pkl")

        if os.path.exists(matriz_filename):

            logging.info("Carregando matriz salva...")

            matriz = carregar_objeto(matriz_filename)

            y_true = matriz["y_true"]
            y_proba = matriz["y_proba"]
            classes = matriz["classes"]
            
            print("Classes fora do modelo:", set(y_true) - set(classes))
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

                '''
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train,
                    y_train,
                    test_size=0.2,
                    stratify=y_train,
                    shuffle=True,
                    random_state=10
                )
                '''
                
                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=10)
                tr_idx, val_idx = next(gss.split(X_train, y_train, groups=fold_dict["groups_train"]))
                
                X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

                all_labels = sorted(set(y_train) | set(y_test))
                le = LabelEncoder().fit(all_labels)

                y_tr_enc   = le.transform(y_tr)
                y_val_enc  = le.transform(y_val)
                y_test_encoded = le.transform(y_test)

                ss = StandardScaler().fit(X_tr)
                X_tr_scaled  = ss.transform(X_tr)
                X_val_scaled = ss.transform(X_val)
                X_test_scaled = ss.transform(X_test)
                
                num_classes = len(le.classes_)
                
                best_params, best_iter, _ = selecionar_melhor_xgboost(
                    param_grid,
                    X_tr_scaled, X_val_scaled,
                    y_tr_enc, y_val_enc,
                    num_classes
                )
                
                X_full = np.vstack((X_tr_scaled, X_val_scaled))
                y_full = np.concatenate((y_tr_enc, y_val_enc))
                X_full_bal, y_full_bal = balancear(X_full, y_full, REBALANCE)
                
                modelo = XGBClassifier(
                    objective="multi:softprob",
                    num_class=num_classes,
                    tree_method="hist",
                    n_jobs=1,
                    eval_metric="mlogloss",
                    n_estimators=best_iter,
                    **best_params
                )
                modelo.fit(X_full_bal, y_full_bal)

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

            X_test_scaled = ss.transform(X_test)
            y_proba = modelo.predict_proba(X_test_scaled)
            classes = modelo.classes_
            
            y_test_encoded = le.transform(y_test)
            y_pred = classes[np.argmax(y_proba, axis=1)]

            f1, topk = calcular_metricas(y_test_encoded, y_proba, classes, ka)

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

    logging.info("Selecione o tipo de dataset:\n1 - Segmentado")

    opcoes = {"1": "segmentado"}
    tipo = opcoes.get(input("Digite sua escolha (1-2): ").strip())

    if tipo is None:
        logging.error("Escolha inválida!")
        return

    config = DATASET_CONFIGS[tipo]

    param_grid = {
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0]
        #"n_estimators": [200, 400]
    }
      
    resultados = {}
        
    for n_splits in CV_SPLITS:
        logging.info(f"EXPERIMENTO {n_splits}-FOLD")
        
        inicio_experimento = datetime.now()

        acuracias, topkAcuracias = do_cv_xgb(ka, n_splits, config, param_grid)
        
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