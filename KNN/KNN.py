import os
import numpy as np
import pickle
import logging

from datetime import datetime
from dataclasses import dataclass

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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

#################################################

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
    path_folds: str # somente leitura

    
DATASET_CONFIGS = {
    "segmentado": DatasetConfig(
        nome="Áudios Segmentados",
        path_matrizes=f"{DATA_VERSION}/{REBALANCE}/matrizesProba_knn_treinoSegmentado",
        path_modelos=f"{DATA_VERSION}/{REBALANCE}/modelos_knn_treinoSegmentado",
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

def selecionar_melhor_k(ks, X_treino, X_val, y_treino, y_val):
    
    acuracias_val = []

    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_treino, y_treino)
        pred = knn.predict(X_val)
        acuracias_val.append(f1_score(y_val, pred, average="macro", labels=np.unique(y_val)))
        
    melhor_val = max(acuracias_val)
    melhor_k = ks[np.argmax(acuracias_val)]
    
    logging.info(f"Melhor k na validação: {melhor_k} (acc={melhor_val:.2f})")   
    
    return melhor_k, melhor_val


def exibir_resultados_matriz(y_true, y_proba, classes, ka):
    y_pred = classes[np.argmax(y_proba, axis=1)]
    
    f1 = f1_score(y_true, y_pred, average="macro", labels=np.unique(y_true))
    
    topk = top_k_accuracy_score(
        y_true,
        y_proba,
        k=ka,
        labels=classes
    )

    logging.info(f"F1-score do KNN: {f1:.2f}")
    logging.info(f"Top-{ka} Accuracy: {topk:.2f}")
    
def treinar_knn_com_validacao_cruzada(ka, n_splits, rebalance, config: DatasetConfig):
    
    path_matrizes = os.path.join(config.path_matrizes, f"{n_splits}fold")
    path_modelos = os.path.join(config.path_modelos, f"{n_splits}fold")
    
    preparar_pastas(path_matrizes, path_modelos)
    
    path_folds = os.path.join(config.path_folds, f"stratified_group_kfold_{n_splits}.pkl")
    
    if not os.path.exists(path_folds):
        raise FileNotFoundError("Folds ainda não foram gerados.")
    
    folds = carregar_objeto(path_folds)
    
    acuracias, topKScores = [], []
        
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
        modelo_filename = os.path.join(path_modelos, f"KNN_model_fold_{foldId + 1}.pkl")

        if os.path.exists(matriz_filename):
            logging.info("Carregando matriz salva...")

            matriz = carregar_objeto(matriz_filename)

            y_true = matriz["y_true"]
            y_proba = matriz["y_proba"]
            classes = matriz["classes"]
            
            print("Classes fora do modelo:", set(y_true) - set(classes))

            y_pred = classes[np.argmax(y_proba, axis=1)]

            f1 = f1_score(y_true, y_pred, average="macro", labels=np.unique(y_true))

            topk = top_k_accuracy_score(
                y_true,
                y_proba,
                k=ka,
                labels=classes
            )
            
            exibir_resultados_matriz(y_true, y_proba, classes, ka)

        else:

            if os.path.exists(modelo_filename):
                logging.info("Carregando modelo salvo...")
                ckpt = carregar_objeto(modelo_filename)
                knn_final = ckpt["model"]
                ss = ckpt["scaler"]
    
                X_teste_scaled = ss.transform(X_teste)

            else:
                logging.info("Treinando modelo...")

                '''
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_treino,
                    y_treino,
                    test_size=0.2,
                    stratify=y_treino,
                    shuffle=True,
                    random_state=10
                )
                '''
                
                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=10)
                tr_idx, val_idx = next(gss.split(X_treino, y_treino, groups=fold_dict["groups_train"]))
                
                X_tr, X_val = X_treino.iloc[tr_idx], X_treino.iloc[val_idx]
                y_tr, y_val = y_treino.iloc[tr_idx], y_treino.iloc[val_idx]

                ss = StandardScaler().fit(X_tr)

                X_tr_scaled = ss.transform(X_tr)
                X_val_scaled = ss.transform(X_val)
                X_teste_scaled = ss.transform(X_teste)
                
                melhor_k, _ = selecionar_melhor_k(range(1, 30, 2), X_tr_scaled, X_val_scaled, y_tr, y_val) # seleção de hiperparametros FORA do rebalanceamento
                
                X_full = np.vstack((X_tr_scaled, X_val_scaled))
                y_full = np.concatenate((y_tr.values, y_val.values)) # recombinar + balancear de novo para o modelo final
                X_full_bal, y_full_bal = balancear(X_full, y_full, REBALANCE)
                
                knn_final = KNeighborsClassifier(n_neighbors=melhor_k)
                knn_final.fit(X_full_bal, y_full_bal)

                salvar_objeto({"model": knn_final, "scaler": ss}, modelo_filename)

            y_pred = knn_final.predict(X_teste_scaled)
            y_proba = knn_final.predict_proba(X_teste_scaled)

            f1 = f1_score(y_teste, y_pred, average="macro", labels=np.unique(y_teste))

            topk = top_k_accuracy_score(
                y_teste,
                y_proba,
                k=ka,
                labels=knn_final.classes_
            )

            salvar_objeto({
                "fold": foldId,
                "y_true": y_teste.values,
                "y_proba": y_proba,
                "classes": knn_final.classes_
            }, matriz_filename)

            logging.info("Matriz salva.")

        logging.info(f"F1-score: {f1:.2f}")
        logging.info(f"Top-{ka} Accuracy: {topk:.2f}")

        acuracias.append(f1)
        topKScores.append(topk)

    return acuracias, topKScores

def main():
    ka = int(input("Hiperparâmetro K (Top-K): "))
    
    print("\n##########################\n")
    
    print(f"VERSÃO = {DATA_VERSION}")
    print(f"Top-K = {ka}")
    
    # Seleção do dataset
    print("Selecione o tipo de dataset:")
    print("1 - Segmentado")
    
    opcoes = {"1": "segmentado"}
    tipo = opcoes.get(input("Digite sua escolha: ").strip())
    
    if tipo is None:
        logging.error("Escolha inválida!")
        return

    config = DATASET_CONFIGS[tipo]
    estrategia = REBALANCE
    
    print("\n##########################\n")
    print(f"VERSÃO       = {DATA_VERSION}")
    print(f"Tipo         = {tipo}")
    print(f"Top-K        = {ka}")
    print(f"Estratégia(s) = {estrategia}")
    print("\n##########################\n")
    
    resultados = {}
    
    for n_splits in CV_SPLITS:
        
        logging.info(f"EXPERIMENTO {n_splits}-FOLD | estratégia = {estrategia}")
        
        inicio_experimento = datetime.now()
        
        acuracias, topKAcuracias = treinar_knn_com_validacao_cruzada(ka, n_splits, estrategia, config)
        
        final_experimento = datetime.now()
        
        resultados[n_splits] = {
            "f1": acuracias,
            "topk": topKAcuracias
        }
        
        logging.info(f"Tempo de Experimento - {n_splits} FOLD [{estrategia}] = {final_experimento - inicio_experimento}")
    
        #print(f"\n-- TESTE {config.nome.upper()} ({n_splits}-FOLD)--")
        #print("F1-Score Macro:")
        #print(f"min: {min(acuracias):.2f}, max: {max(acuracias):.2f}, avg ± std: {np.mean(acuracias):.2f} ± {np.std(acuracias):.2f}")
        #print(f"\nTop-{ka} Score:")
        #print(f"min: {min(topKAcuracias):.2f}, max: {max(topKAcuracias):.2f}, avg ± std: {np.mean(topKAcuracias):.2f} ± {np.std(topKAcuracias):.2f}")
    
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