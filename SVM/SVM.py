import os
import numpy as np
import pickle
import itertools
import logging

from datetime import datetime
from dataclasses import dataclass
from joblib import Parallel, delayed

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
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
        path_matrizes=f"{DATA_VERSION}/{REBALANCE}/matrizesProba_svm_treinoSegmentado",
        path_modelos=f"{DATA_VERSION}/{REBALANCE}/modelos_svm_treinoSegmentado",
        path_folds=f"../folds/{DATA_VERSION}/segmentado"
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


def selecionar_melhor_svm(Cs, gammas, X_treino : np.ndarray, X_val : np.ndarray, 
                          y_treino : np.ndarray, y_val : np.ndarray, n_jobs=4):
    
    def treinar_svm(C, gamma):
        svm = SVC(C=C, gamma=gamma)
        svm.fit(X_treino, y_treino)
        pred = svm.predict(X_val) 
        return f1_score(y_val, pred, average="macro", labels=np.unique(y_val))
    
    # Gera todas as combinações de parâmetros C e gamma
    combinacoes_parametros = list(itertools.product(Cs, gammas))
    
    # Treinar modelos com todas as combinações de C e gamma
    acuracias_val = Parallel(n_jobs=n_jobs)(delayed(treinar_svm)
                                       (c, g) for c, g in combinacoes_parametros)       
    
    # Encontrar a combinação que levou ao melhor resultado no conjunto de validação
    melhor_val = max(acuracias_val)
    melhor_comb = combinacoes_parametros[np.argmax(acuracias_val)]   
    melhor_c = melhor_comb[0]
    melhor_gamma = melhor_comb[1]
    
    logging.info(f"Melhor SVM - C: {melhor_c}, gamma: {melhor_gamma}, F1 Val: {melhor_val:.2f}")
    
    # Treinar uma SVM com todos os dados de treino e validação usando a melhor combinação de C e gamma
    #svm_final = SVC(C=melhor_c, gamma=melhor_gamma, probability=True)
    #svm_final.fit(np.vstack((X_treino, X_val)), [*y_treino, *y_val])

    return melhor_comb, melhor_val


def do_cv_svm(ka, n_splits, config: DatasetConfig, Cs, gammas):

    path_matrizes = os.path.join(config.path_matrizes, f"{n_splits}fold")
    path_modelos = os.path.join(config.path_modelos, f"{n_splits}fold")
        
    preparar_pastas(path_matrizes, path_modelos)
        
    path_folds = os.path.join(config.path_folds, f"stratified_group_kfold_{n_splits}.pkl")
    
    if not os.path.exists(path_folds):
        raise FileNotFoundError("Folds ainda não foram gerados.")
    
    folds = carregar_objeto(path_folds)
    
    f1_scores, topkScores = [], []
    
    for fold_dict in folds:
        
        foldId = fold_dict["fold"]
                
        X_treino = fold_dict["X_train"]
        y_treino = fold_dict["y_train"]
        
        X_teste = fold_dict["X_test"]
        y_teste = fold_dict["y_test"]
        
        logging.info(f"Amostras treino: {len(X_treino)}")
        logging.info(f"Amostras teste: {len(X_teste)}")
        logging.info(f"Espécies treino: {y_treino.nunique()}")
        logging.info(f"Espécies teste: {y_teste.nunique()}")
        
        logging.info(f"\n=== {n_splits}-FOLD | Fold {foldId + 1} ===")

        matriz_filename = os.path.join(path_matrizes, f"matriz_{foldId + 1}.pkl")
        modelo_filename = os.path.join(path_modelos, f"svm_model_fold_{foldId + 1}.pkl")

        if os.path.exists(matriz_filename):
            logging.info("Carregando matriz salva...")

            matriz = carregar_objeto(matriz_filename)

            y_eval = matriz["y_true"]
            y_proba = matriz["y_proba"]
            classes = matriz["classes"]
            
            print("Classes fora do modelo:", set(y_eval) - set(classes))

            y_pred = classes[np.argmax(y_proba, axis=1)]

        else:

            if os.path.exists(modelo_filename):
                logging.info("Carregando modelo salvo...")
                data = carregar_objeto(modelo_filename)

                svm = data["svm"]
                scaler = data["scaler"]

            else:
                logging.info("Treinando modelo...")

                '''
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_treino, 
                    y_treino,
                    test_size=0.2,
                    stratify=y_treino,
                    shuffle=True,
                    random_state=1
                )
                '''
                
                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=10)
                tr_idx, val_idx = next(gss.split(X_treino, y_treino, groups=fold_dict["groups_train"]))
                
                X_tr, X_val = X_treino.iloc[tr_idx], X_treino.iloc[val_idx]
                y_tr, y_val = y_treino.iloc[tr_idx], y_treino.iloc[val_idx]

                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_tr)
                X_val_scaled = scaler.transform(X_val)

                melhor_comb, _ = selecionar_melhor_svm(
                    Cs, gammas,
                    X_tr_scaled, X_val_scaled,
                    y_tr.values, y_val.values
                )
                
                melhor_c, melhor_gamma = melhor_comb
                
                X_full = np.vstack((X_tr_scaled, X_val_scaled))
                y_full = np.concatenate((y_tr.values, y_val.values))
                X_full_bal, y_full_bal = balancear(X_full, y_full, REBALANCE)

                svm = SVC(C=melhor_c, gamma=melhor_gamma, probability=True)
                svm.fit(X_full_bal, y_full_bal)

                salvar_objeto({
                    "svm": svm,
                    "scaler": scaler
                }, modelo_filename)

                logging.info("Modelo salvo.")

            X_test_scaled = scaler.transform(X_teste)

            y_eval = y_teste.values
            y_pred = svm.predict(X_test_scaled)
            y_proba = svm.predict_proba(X_test_scaled)
            
            classes = svm.classes_

            salvar_objeto({
                "fold": foldId,
                "y_true": y_eval,
                "y_proba": y_proba,
                "classes": classes
            }, matriz_filename)

            logging.info("Matriz salva.")

        f1 = f1_score(y_eval, y_pred, average="macro", labels=np.unique(y_eval))

        topk = top_k_accuracy_score(
            y_eval,
            y_proba,
            k=ka,
            labels=classes
        )

        logging.info(f"F1={f1:.3f} | Top-{ka}={topk:.3f}")

        f1_scores.append(f1)
        topkScores.append(topk)

    return f1_scores, topkScores

def main():
    ka = int(input("Hiperparâmetro K (Top-K): "))
    
    print("\n##########################\n")
    
    print(f"VERSÃO = {DATA_VERSION}")
    print(f"Top-K = {ka}")
    
    logging.info("Selecione o tipo de dataset:\n1 - Segmentado")
    
    opcoes = {"1": "segmentado"}
    tipo = opcoes.get(input("Digite sua escolha: ").strip())
    
    if tipo is None:
        logging.error("Escolha inválida!")
        return

    config = DATASET_CONFIGS[tipo]
    
    resultados = {}
    
    for n_splits in CV_SPLITS:
        logging.info(f"EXPERIMENTO {n_splits}-FOLD")
        
        inicio_experimento = datetime.now()
    
        acuracias, topkAcuracias = do_cv_svm(
            ka, n_splits, config, 
            # kernel = ['rbf', 'poly', 'sigmoid'],
            Cs=[1, 10, 100, 1000], 
            gammas=['scale', 'auto', 2e-2, 2e-3, 2e-4]
        )
        
        final_experimento = datetime.now()
            
        resultados[n_splits] = {
            "f1": acuracias,
            "topk": topkAcuracias
        }
        
        logging.info(f"Tempo de Experimento - {n_splits} FOLD = {final_experimento - inicio_experimento}")

        #print(f"\n-- TESTE {config.nome.upper()} ({n_splits}-FOLD)--")
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