import os
import numpy as np
import pickle
import itertools
import logging

from datetime import datetime
from dataclasses import dataclass
from joblib import Parallel, delayed

from xgboost import XGBClassifier

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, top_k_accuracy_score, pairwise_distances, classification_report

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
        path_matrizes=f"{DATA_VERSION}/matrizesProba_kmeansd_xgb_treinoSegmentado",
        path_modelos=f"{DATA_VERSION}/modelos_kmeansd_xgb_treinoSegmentado",
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

def treinar_kmeansc(X, y, ka):
    
    centroides = []
    labels = []
    slices = {}
    start = 0
    
    classes = sorted(np.unique(y))
    
    for classe in classes:
        X_classe = X[y == classe]
        n_amostras = len(X_classe)
        
        k = min(n_amostras, ka)
        
        if k == n_amostras:
            ctrds = X_classe
        else:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
            kmeans.fit(X_classe)
            ctrds = kmeans.cluster_centers_

        end = start + len(ctrds)

        centroides.append(ctrds)
        labels.extend([classe] * len(ctrds))
        slices[classe] = (start, end)

        start = end
    
    return {
        "centroides": np.vstack(centroides),
        "labels": np.array(labels),
        "slices": slices
    }

############################################
# FEATURE EXTRACTION
############################################

def gerar_distancias(X, centroides):
    # return pairwise_distances(X_scaled, centroides, n_jobs=4)
    return pairwise_distances(X, centroides)

def extrair_menor_dist_por_classe(X, modelo_kmeans, k):
    features = []

    for classe in sorted(modelo_kmeans["slices"].keys()):
        start, end = modelo_kmeans["slices"][classe]

        dist_classe = X[:, start:end]
        
        k_eff = min(k, dist_classe.shape[1])
        
        dist_classe = dist_classe[:, :k_eff] 

        menor_dist = np.min(dist_classe, axis=1)
        features.append(menor_dist)

    return np.column_stack(features)

############################################

def selecionar_melhor_xgb(param_grid, X_train, X_val, y_train, y_val, num_classes, n_jobs=4):

    def treinar(params):
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            tree_method="hist", #aproximately Greedy Algorithm
            #n_jobs=n_jobs,
            n_jobs=1,
            eval_metric="mlogloss",
            n_estimators=200,
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


def do_cv_kmeansd_xgb(X, y, ka, config, k_values, param_grid):
    
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
            f"kmeansd_xgb_model_fold_{foldId + 1}.pkl"
        )

        matriz_filename = os.path.join(
            config.path_matrizes,
            f"matriz_{foldId + 1}.pkl"
        )

        if os.path.exists(matriz_filename):
            logging.info("Carregando matriz salva...")

            matriz = carregar_objeto(matriz_filename)
            
            le = matriz["label_encoder"]

            y_true = matriz["y_true"]
            y_proba = matriz["y_proba"]
            classes = matriz["classes"]

            y_pred_encoded = np.argmax(y_proba, axis=1)

            f1, topk = calcular_metricas(y_true, y_pred_encoded, y_proba, classes, ka)
            
            y_true = le.inverse_transform(y_true)
            y_pred = le.inverse_transform(y_pred_encoded)
            
            mask_test = y_teste.isin(le.classes_)
            y_test_filtrado = y_teste[mask_test]
            
            f1_report = classification_report(y_test_filtrado, y_pred)
            print(f"\n=== Classification Report Fold {foldId + 1} ===")
            print(f1_report)
            
        else:
            
            if os.path.exists(modelo_filename):
                logging.info("Carregando modelo salvo...")
                modelo = carregar_objeto(modelo_filename)

                modelo_kmeans = modelo["kmeans"]
                scaler_global = modelo["scaler_global"]
                scaler_dist = modelo["scaler_dist"]
                xgb = modelo["xgboost"]
                le = modelo["label_encoder"]
                k = modelo["k"]
            
            else:
                logging.info("Treinando modelo...")
            
                X_train, X_val, y_train, y_val = split_train_val(X_treino, y_treino)
                
                le = LabelEncoder()
                y_train_encoded = le.fit_transform(y_train)

                mask_val = y_val.isin(le.classes_)
                X_val = X_val[mask_val]
                y_val = y_val[mask_val]
                y_val_encoded = le.transform(y_val)

                num_classes = len(le.classes_)
                
                scaler_global = StandardScaler()
                
                X_train_scaled = scaler_global.fit_transform(X_train)
                X_val_scaled = scaler_global.transform(X_val)

                k_max = max(k_values)
                modelo_kmeans = treinar_kmeansc(X_train_scaled,y_train.values,k_max)
                
                X_tr_dist = gerar_distancias(X_train_scaled, modelo_kmeans["centroides"])
                X_val_dist = gerar_distancias(X_val_scaled, modelo_kmeans["centroides"])
                
                scaler_dist = StandardScaler()

                X_tr_dist = scaler_dist.fit_transform(X_tr_dist)
                X_val_dist = scaler_dist.transform(X_val_dist)              

                melhor_f1 = -1
                melhor_k = -1
                
                for k in k_values:
                    
                    X_tr_k = extrair_menor_dist_por_classe(X_tr_dist, modelo_kmeans, k)
                    X_val_k = extrair_menor_dist_por_classe(X_val_dist, modelo_kmeans, k) 

                    xgb, _, _ = selecionar_melhor_xgb(param_grid, X_tr_k, X_val_k, y_train_encoded, y_val_encoded, num_classes)
                    
                    y_pred_val = xgb.predict(X_val_k)
                    f1_val = f1_score(y_val_encoded, y_pred_val, average="macro")

                    if f1_val > melhor_f1:
                        melhor_f1 = f1_val
                        melhor_xgb = xgb
                        melhor_k = k
                
                xgb = melhor_xgb
                k = melhor_k
                
                salvar_objeto({
                    "kmeans": modelo_kmeans,
                    "scaler_global": scaler_global,
                    "scaler_dist": scaler_dist,
                    "xgboost": melhor_xgb,
                    "label_encoder": le,
                    "k": k,
                }, modelo_filename)

                logging.info("Modelo salvo.")
            
            ## Teste

            mask_test = y_teste.isin(le.classes_)

            X_test_filtrado = X_teste[mask_test]
            y_test_filtrado = y_teste[mask_test]

            y_test_encoded = le.transform(y_test_filtrado)
            
            X_test_scaled = scaler_global.transform(X_test_filtrado)

            X_test_dist = gerar_distancias(X_test_scaled, modelo_kmeans["centroides"])
            X_test_dist = scaler_dist.transform(X_test_dist)
            
            X_test_k = extrair_menor_dist_por_classe(X_test_dist, modelo_kmeans, k)

            y_pred_encoded = xgb.predict(X_test_k)
            y_pred = le.inverse_transform(y_pred_encoded)
            
            y_proba = xgb.predict_proba(X_test_k)

            f1, topk = calcular_metricas(y_test_encoded, y_pred_encoded, y_proba, xgb.classes_, ka)
            
            salvar_objeto({
                "fold": foldId,
                "y_true": y_test_encoded,
                "y_proba": y_proba,
                "label_encoder": le,
                "classes": xgb.classes_
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
    
    param_grid = {
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0]
    }

    acuracias, topkAcuracias = do_cv_kmeansd_xgb(
        X,
        y,
        ka=ka,
        config=config,
        k_values=[50, 100],
        param_grid=param_grid
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