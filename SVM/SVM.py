import os
import numpy as npy
import pickle
import itertools
import logging

from datetime import datetime
from dataclasses import dataclass
from joblib import Parallel, delayed

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
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
        path_matrizes=f"{DATA_VERSION}/matrizesProba_svm_treinoSegmentado",
        path_modelos=f"{DATA_VERSION}/modelos_svm_treinoSegmentado",
        path_folds=f"../folds/{DATA_VERSION}/segmentado/stratified_group_kfold_10.pkl"
    ),
    "completo": DatasetConfig(
        nome="Áudios Completos",
        path_dataframe=f"../dataframes/{DATA_VERSION}/dataframeAudioCompleto.pkl",
        path_matrizes=f"{DATA_VERSION}/matrizesProba_svm_treinoCompleto",
        path_modelos=f"{DATA_VERSION}/modelos_svm_treinoCompleto",
        path_folds=f"../folds/{DATA_VERSION}/completo/stratified_group_kfold_10.pkl"
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


def selecionar_melhor_svm(Cs, gammas, X_treino : npy.ndarray, X_val : npy.ndarray, 
                          y_treino : npy.ndarray, y_val : npy.ndarray, n_jobs=4):
    
    def treinar_svm(C, gamma, X_treino, X_val, y_treino, y_val):
        svm = SVC(C=C, gamma=gamma, probability=True)
        svm.fit(X_treino, y_treino)
        pred = svm.predict(X_val) 
        return f1_score(y_val, pred, average="macro")
    
    # Gera todas as combinações de parâmetros C e gamma
    combinacoes_parametros = list(itertools.product(Cs, gammas))
    
    # Treinar modelos com todas as combinações de C e gamma
    acuracias_val = Parallel(n_jobs=n_jobs)(delayed(treinar_svm)
                                       (c, g, X_treino, X_val, y_treino, y_val) for c, g in combinacoes_parametros)       
    
    # Encontrar a combinação que levou ao melhor resultado no conjunto de validação
    melhor_val = max(acuracias_val)
    melhor_comb = combinacoes_parametros[npy.argmax(acuracias_val)]   
    melhor_c = melhor_comb[0]
    melhor_gamma = melhor_comb[1]
    
    logging.info(f"Melhor SVM - C: {melhor_c}, gamma: {melhor_gamma}, F1 Val: {melhor_val:.2f}")
    
    # Treinar uma SVM com todos os dados de treino e validação usando a melhor combinação de C e gamma
    svm_final = SVC(C=melhor_c, gamma=melhor_gamma, probability=True)
    svm_final.fit(npy.vstack((X_treino, X_val)), [*y_treino, *y_val])

    return svm_final, melhor_comb, melhor_val


def exibir_resultados(svm, X_test_scaled, y_test, ka):
    y_pred = svm.predict(X_test_scaled)
    y_proba = svm.predict_proba(X_test_scaled)

    f1 = f1_score(y_test, y_pred, average="macro")
    mask = y_test.isin(svm.classes_)
    y_test_filtrado = y_test[mask]
    y_proba_filtrado = y_proba[mask.values]
    
    classes_presentes = npy.intersect1d(svm.classes_, npy.unique(y_test_filtrado))
    idxs = [npy.where(svm.classes_ == c)[0][0] for c in classes_presentes]
    y_proba_filtrado = y_proba_filtrado[:, idxs]
    
    topk_acc = top_k_accuracy_score(y_test_filtrado, y_proba_filtrado, k=ka, labels=classes_presentes)

    logging.info(f"F1-score do SVM: {f1:.2f}")
    logging.info(f"Top-{ka} Accuracy: {topk_acc:.2f}")


def do_cv_svm(X, y, ka, config: DatasetConfig, Cs=[1], gammas=['scale']):

    preparar_pastas(config.path_matrizes, config.path_modelos)
    
    if not os.path.exists(config.path_folds):
        raise FileNotFoundError("Folds ainda não foram gerados.")
    
    folds = carregar_objeto(config.path_folds)
    
    acuracias, topkScores = [], []
    
    for fold_dict in folds:
        
        foldId = fold_dict["fold"]
        idx_treino = fold_dict["train_idx"]
        idx_teste = fold_dict["test_idx"]

        logging.info(f"\n=== Fold {foldId + 1} ===")

        X_treino = X.iloc[idx_treino]
        y_treino = y.iloc[idx_treino]

        X_teste = X.iloc[idx_teste]
        y_teste = y.iloc[idx_teste]

        matriz_filename = os.path.join(
            config.path_matrizes, f"matriz_{foldId + 1}.pkl"
        )

        modelo_filename = os.path.join(
            config.path_modelos, f"svm_model_fold_{foldId + 1}.pkl"
        )
        
        if os.path.exists(modelo_filename):
            logging.info("Carregando modelo salvo...")
            svm = carregar_objeto(modelo_filename)

            ss = StandardScaler()
            ss.fit(X_treino)
            X_teste = ss.transform(X_teste)

            y_pred = svm.predict(X_teste)
            y_proba = svm.predict_proba(X_teste)
        
        else:
            logging.info(f"Criando modelo do fold {foldId + 1}...")

            counts = y_treino.value_counts()

            if counts.min() >= 2:
                X_treino, X_val, y_treino, y_val = train_test_split(
                    X_treino,
                    y_treino,
                    stratify=y_treino,
                    test_size=0.2,
                    random_state=1
                )
            else:
                print("[INFO] Fold contém classe com apenas 1 amostra. Split sem estratificação.")
                
                X_treino, X_val, y_treino, y_val = train_test_split(
                    X_treino,
                    y_treino,
                    test_size=0.2,
                    random_state=1
                )

            ss = StandardScaler()
            ss.fit(X_treino)
            X_treino = ss.transform(X_treino)
            X_teste = ss.transform(X_teste)
            X_val = ss.transform(X_val)

            svm, _, _ = selecionar_melhor_svm(Cs, gammas, X_treino, X_val, y_treino, y_val)
            
            y_pred = svm.predict(X_teste)
            y_proba = svm.predict_proba(X_teste)
        
            salvar_objeto({"fold": foldId, "y_true": y_teste.values, "y_proba": y_proba, "classes": svm.classes_},
                          matriz_filename)
            logging.info(f"Probabilidades salvas em {matriz_filename}")
            
            salvar_objeto(svm, modelo_filename)
            logging.info(f"Modelo salvo em {modelo_filename}")
            
        f1 = f1_score(y_teste, y_pred, average="macro")
        desconhecidas = (~y_teste.isin(svm.classes_)).sum()
        logging.info(f"Amostras com classe 'não vista no treino': {desconhecidas}/{len(y_teste)}")
        
        classes_treino = set(svm.classes_)
        mask = y_teste.isin(classes_treino)
        y_teste_filtrado = y_teste[mask]
        y_proba_filtrado = y_proba[mask.values]
        
        topk = top_k_accuracy_score(y_teste_filtrado, y_proba_filtrado, k=ka, labels=svm.classes_)
        exibir_resultados(svm, X_teste, y_teste, ka)

        acuracias.append(f1)
        topkScores.append(topk)
    
    return acuracias, topkScores

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
    
    acuracias, topkAcuracias = do_cv_svm(
        X, y, ka, config, 
        Cs=[1, 10, 100, 1000], 
        gammas=['scale', 'auto', 2e-2, 2e-3, 2e-4]
    )
    
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